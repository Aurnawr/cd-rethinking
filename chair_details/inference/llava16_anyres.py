"""
Runtime AnyRes (LLaVA-1.6 / LLaVA-NeXT) support for this LLaVA-1.5-era repo,
WITHOUT modifying any existing files.

It monkeypatches at import/apply time:
  * LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal -> anyres version
    (spatial_unpad + image_newline + image_sizes), falling back to the original
    behaviour for non-anyres / single-patch inputs.
  * LlavaLlamaForCausalLM.forward -> accepts `image_sizes` and forwards it.
  * the three prepare_inputs_for_generation* helpers -> carry `image_sizes`.

It also provides AnyRes image preprocessing (process_images_anyres) and a helper
to load the `image_newline` weight that the 1.6 checkpoint ships but the
1.5-era model class does not register.

Usage (in an inference script):
    import llava16_anyres as anyres
    anyres.apply_anyres_patches()                 # patch classes (once)
    ... load model via load_pretrained_model ...
    anyres.load_image_newline(model, model_path)  # attach image_newline weight
    image_tensor = anyres.process_images_anyres([pil_img], image_processor, model.config)[0]
    # generate(images=[image_tensor.half().cuda()], image_sizes=[pil_img.size], ...)
"""
import ast
import json
import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


# --------------------------------------------------------------------------- #
# AnyRes image-processing utilities (ported from official LLaVA-1.6 mm_utils)  #
# --------------------------------------------------------------------------- #
def select_best_resolution(original_size, possible_resolutions):
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")
    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution
        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)
    return best_fit


def resize_and_pad_image(image, target_resolution):
    from PIL import Image
    original_width, original_height = image.size
    target_width, target_height = target_resolution
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)
    resized_image = image.resize((new_width, new_height))
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    return new_image


def divide_to_patches(image, patch_size):
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patches.append(image.crop(box))
    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)
    patches = divide_to_patches(image_padded, processor.crop_size["height"])
    shortest_edge = processor.size.get("shortest_edge", processor.crop_size["height"])
    image_original_resize = image.resize((shortest_edge, shortest_edge))
    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(p, return_tensors="pt")["pixel_values"][0] for p in image_patches]
    return torch.stack(image_patches, dim=0)


def expand2square(pil_img, background_color):
    from PIL import Image
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images_anyres(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def unpad_image(tensor, original_size):
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height
    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]
    return unpadded_tensor


# --------------------------------------------------------------------------- #
# Patched multimodal embedding (anyres-aware), preserving the repo merge loop  #
# --------------------------------------------------------------------------- #
def _num_patches_per_side(vision_tower):
    return int(vision_tower.num_patches ** 0.5)


def prepare_inputs_labels_for_multimodal_anyres(
    self, input_ids, attention_mask, past_key_values, labels, images, image_sizes=None
):
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones(
                (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
        return input_ids, attention_mask, past_key_values, None, labels

    mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
    image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")

    # ---- build per-image feature sequences -------------------------------- #
    if type(images) is list or images.ndim == 5:
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.encode_images(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)

        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type.startswith("spatial"):
            height = width = _num_patches_per_side(vision_tower)
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    assert height * width == base_image_feature.shape[0]
                    if image_aspect_ratio == "anyres":
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                            image_sizes[image_idx],
                            self.config.image_grid_pinpoints,
                            vision_tower.config.image_size,
                        )
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        image_feature = image_feature.view(2, 2, height, width, -1)
                    if "unpad" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat(
                            (
                                image_feature,
                                self.get_model().image_newline[:, None, None]
                                .expand(*image_feature.shape[:-1], 1)
                                .to(image_feature.device),
                            ),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat(
                            (image_feature, self.get_model().image_newline[None].to(image_feature.device)), dim=0
                        )
                new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {mm_patch_merge_type}")
    else:
        image_features = self.encode_images(images)

    # ---- repo merge loop (verbatim from llava_arch.py) -------------------- #
    new_input_embeds = []
    new_labels = [] if labels is not None else None
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
            half_len = cur_input_ids.shape[0] // 2
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
            cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue
        image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        cur_new_input_embeds = []
        if labels is not None:
            cur_labels = labels[batch_idx]
            cur_new_labels = []
            assert cur_labels.shape == cur_input_ids.shape
        while image_token_indices.numel() > 0:
            cur_image_features = image_features[cur_image_idx]
            image_token_start = image_token_indices[0]
            if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 2]))
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_new_labels.append(cur_labels[image_token_start:image_token_start + 1])
                    cur_labels = cur_labels[image_token_start + 2:]
            else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[image_token_start + 1:]
            cur_image_idx += 1
            if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
                cur_input_ids = cur_input_ids[image_token_start + 2:]
            else:
                cur_input_ids = cur_input_ids[image_token_start + 1:]
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        if cur_input_ids.numel() > 0:
            if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
            else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
            if labels is not None:
                cur_new_labels.append(cur_labels)
        cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        new_input_embeds.append(cur_new_input_embeds)
        if labels is not None:
            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            new_labels.append(cur_new_labels)

    if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
        max_len = max(x.shape[0] for x in new_input_embeds)
        new_input_embeds_align = []
        for cur_new_embed in new_input_embeds:
            cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
            new_input_embeds_align.append(cur_new_embed)
        new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
        if labels is not None:
            new_labels_align = []
            _new_labels = new_labels
            for cur_new_label in new_labels:
                cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                new_labels_align.append(cur_new_label)
            new_labels = torch.stack(new_labels_align, dim=0)
        if attention_mask is not None:
            new_attention_mask = []
            for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                new_attention_mask.append(cur_new_attention_mask)
            attention_mask = torch.stack(new_attention_mask, dim=0)
            assert attention_mask.shape == new_labels.shape
    else:
        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        if labels is not None:
            new_labels = torch.stack(new_labels, dim=0)
        if attention_mask is not None:
            new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
            assert attention_mask.shape == new_input_embeds.shape[:2]

    return None, attention_mask, past_key_values, new_input_embeds, new_labels


# --------------------------------------------------------------------------- #
# Patched forward / prepare_inputs_for_generation to thread image_sizes        #
# --------------------------------------------------------------------------- #
def forward_anyres(
    self,
    input_ids=None,
    attention_mask=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    images=None,
    return_dict=None,
    image_sizes=None,
    images_cd=None,
    input_ids_cd=None,
    cd_beta=None,
    cd_alpha=None,
    use_sid=None,
    use_olm=None,
    use_apc=None,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
        input_ids, attention_mask, past_key_values, labels, images, image_sizes
    )

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        use_sid=use_sid,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def _pig(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
    if past_key_values:
        input_ids = input_ids[:, -1:]
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}
    model_inputs.update({
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "attention_mask": attention_mask,
        "images": kwargs.get("images", None),
        "image_sizes": kwargs.get("image_sizes", None),
        "use_sid": None,
    })
    return model_inputs


def _pig_vcd(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
    if past_key_values:
        input_ids = input_ids[:, -1:]
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}
    model_inputs.update({
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "attention_mask": attention_mask,
        "images": kwargs.get("images_cd", None),
        "image_sizes": kwargs.get("image_sizes", None),
    })
    return model_inputs


def _pig_sid(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
    if past_key_values:
        input_ids = input_ids[:, -1:]
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}
    model_inputs.update({
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "attention_mask": attention_mask,
        "images": kwargs.get("images", None),
        "image_sizes": kwargs.get("image_sizes", None),
        "use_sid": True,
    })
    return model_inputs


_PATCHED = False


def apply_anyres_patches():
    """Monkeypatch the LLaVA classes for anyres support (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_anyres
    LlavaLlamaForCausalLM.forward = forward_anyres
    LlavaLlamaForCausalLM.prepare_inputs_for_generation = _pig
    LlavaLlamaForCausalLM.prepare_inputs_for_generation_vcd = _pig_vcd
    LlavaLlamaForCausalLM.prepare_inputs_for_generation_sid = _pig_sid
    _PATCHED = True


def load_image_newline(model, model_path):
    """Load the `model.image_newline` weight (present in 1.6 checkpoints but not
    registered by the 1.5-era model class) and attach it to model.model."""
    from safetensors.torch import load_file
    key = "model.image_newline"
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    tensor = None
    if os.path.isfile(idx_path):
        idx = json.load(open(idx_path))
        shard = idx["weight_map"].get(key)
        if shard is not None:
            sd = load_file(os.path.join(model_path, shard))
            tensor = sd[key]
    if tensor is None:
        # single-file fallback
        for fn in ("model.safetensors",):
            p = os.path.join(model_path, fn)
            if os.path.isfile(p):
                sd = load_file(p)
                if key in sd:
                    tensor = sd[key]
                    break
    if tensor is None:
        raise RuntimeError(f"Could not find {key} in checkpoint at {model_path}")
    dev = model.get_model().embed_tokens.weight.device
    dtype = model.get_model().embed_tokens.weight.dtype
    model.get_model().image_newline = nn.Parameter(tensor.to(device=dev, dtype=dtype))
    return model
