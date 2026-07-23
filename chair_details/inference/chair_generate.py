"""Generate free-form COCO captions with LLaVA-v1.5 for CHAIR evaluation.

Mirrors the model-loading / prompting style of ``inference/pope_infer_base.py``
but emits *free-form* descriptions (no "single word or phrase" suffix) for a
sample of COCO val2017 images, written in the format expected by
``eval/chair_eval.py``:

    {"image_id": <int coco id>, "caption": "<generated description>"}

CHAIR (Rohrbach et al., 2018) is traditionally computed over a random sample of
500 COCO validation images with greedy decoding, which are the defaults here.

Example
-------
    python inference/chair_generate.py \\
        --model-path models/llava-v1.5-7b \\
        --image-folder data/coco/val2017 \\
        --annotation-file data/coco/annotations/instances_val2017.json \\
        --answers-file outputs/chair/llava-7b-greedy/captions.jsonl \\
        --num-samples 500 \\
        --temperature 0 \\
        --conv-mode vicuna_v1
"""

import argparse
import json
import os
import random

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# transformers compatibility shims (must run BEFORE importing ``llava``).
#
# Newer transformers ship a native "llava" AutoConfig entry that collides with
# this repo's custom LlavaConfig registration, and dropped helpers that the
# vendored llava stack still imports.  These shims mirror the ones in
# ``inference/throne_cd_evaluatee.py`` so the custom LLaVA-v1.5 stack imports
# cleanly on the installed transformers build.
# ---------------------------------------------------------------------------
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if "llava" in CONFIG_MAPPING._mapping:
        del CONFIG_MAPPING._mapping["llava"]
except (ImportError, AttributeError, KeyError):
    pass

try:
    import torch as _torch
    import transformers.models.bloom.modeling_bloom as _bloom_mod
    import transformers.models.opt.modeling_opt as _opt_mod

    if not hasattr(_bloom_mod, "_expand_mask"):
        def _bloom_expand_mask(mask, tgt_length):
            batch_size, src_length = mask.shape
            tgt_length = tgt_length if tgt_length is not None else src_length
            expanded_mask = ~(mask[:, None, None, :].to(_torch.bool))
            return expanded_mask.expand(batch_size, 1, tgt_length, src_length)
        _bloom_mod._expand_mask = _bloom_expand_mask

    if not hasattr(_bloom_mod, "_make_causal_mask"):
        def _bloom_make_causal_mask(input_ids_shape, device, past_key_values_length):
            batch_size, target_length = input_ids_shape
            mask = _torch.empty(
                (target_length, target_length + past_key_values_length),
                dtype=_torch.bool, device=device,
            )
            seq_ids = _torch.arange(target_length, device=device)
            mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]
            if past_key_values_length > 0:
                mask[:, :past_key_values_length] = False
            return mask[None, None, :, :].expand(
                batch_size, 1, target_length, target_length + past_key_values_length
            )
        _bloom_mod._make_causal_mask = _bloom_make_causal_mask

    if not hasattr(_opt_mod, "_expand_mask"):
        def _opt_expand_mask(mask, dtype, tgt_len=None):
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len
            expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            inverted = 1.0 - expanded
            return inverted.masked_fill(inverted.to(_torch.bool), _torch.finfo(dtype).min)
        _opt_mod._expand_mask = _opt_expand_mask

    if not hasattr(_opt_mod, "_make_causal_mask"):
        def _opt_make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
            bsz, tgt_len = input_ids_shape
            mask = _torch.full((tgt_len, tgt_len), _torch.finfo(dtype).min, device=device)
            cond = _torch.arange(mask.size(-1), device=device)
            mask.masked_fill_(cond < (cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(dtype)
            if past_key_values_length > 0:
                mask = _torch.cat(
                    [_torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
                    dim=-1,
                )
            return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
        _opt_mod._make_causal_mask = _opt_make_causal_mask
except Exception:
    pass

try:
    import transformers as _tf
    _orig_autoset = _tf.PreTrainedModel._autoset_attn_implementation.__func__

    @classmethod
    def _patched_autoset(cls, config, **kwargs):
        try:
            return _orig_autoset(cls, config, **kwargs)
        except ValueError:
            config._attn_implementation = "eager"
            return config

    _tf.PreTrainedModel._autoset_attn_implementation = _patched_autoset
except Exception:
    pass

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

# Contrastive-decoding monkeypatches from this repo's ``inference/cd_utils``.
# chair_generate.py lives in ``inference/``, so that dir is on sys.path[0] when
# the script is run directly (same as pope_infer_cd.py).
from cd_utils.vcd_utils import (
    add_diffusion_noise,
    evolve_vcd_greedy_search,
    evolve_vcd_sampling,
)
from cd_utils.sid_utils import evolve_sid_greedy_search, evolve_sid_sampling


CD_METHODS = ("none", "vcd", "sid")


def install_cd_patches(cd_method):
    """Monkeypatch ``GenerationMixin`` for the selected CD method.

    Patches both the greedy and sampling search loops so the method works at any
    temperature (mirrors inference/pope_infer_cd.py).
    """
    if cd_method == "vcd":
        evolve_vcd_greedy_search()
        evolve_vcd_sampling()
    elif cd_method == "sid":
        evolve_sid_greedy_search()
        evolve_sid_sampling()
    elif cd_method == "none":
        pass
    else:
        raise ValueError(f"--cd-method must be one of {CD_METHODS}, got {cd_method!r}.")


def select_image_ids(annotation_file, num_samples, seed):
    """Return a deterministic sample of COCO image ids and their file names."""
    coco = json.load(open(annotation_file))
    images = coco["images"]
    images = sorted(images, key=lambda im: im["id"])
    if num_samples and num_samples < len(images):
        rng = random.Random(seed)
        images = rng.sample(images, num_samples)
        images = sorted(images, key=lambda im: im["id"])
    return [(im["id"], im["file_name"]) for im in images]


def eval_model(args):
    disable_torch_init()
    cd_method = (args.cd_method or "none").lower()
    install_cd_patches(cd_method)

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    samples = select_image_ids(args.annotation_file, args.num_samples, args.seed)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Resume support: skip image ids already written.
    done_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(int(json.loads(line)["image_id"]))
                    except (json.JSONDecodeError, KeyError):
                        pass

    qs = args.question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    ans_file = open(answers_file, "a")
    for image_id, file_name in tqdm(samples):
        if image_id in done_ids:
            continue

        image = Image.open(os.path.join(args.image_folder, file_name)).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # Per-method contrastive-decoding kwargs (mirrors inference/pope_infer_cd.py).
        images_cd = None
        use_sid = None
        if cd_method == "vcd":
            images_cd = add_diffusion_noise(image_tensor, args.noise_step).unsqueeze(0).half().cuda()
        elif cd_method == "sid":
            use_sid = True

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                # contrastive-decoding parameters
                images_cd=images_cd,
                use_sid=use_sid,
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()

        ans_file.write(json.dumps({"image_id": image_id, "caption": outputs}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="data/coco/val2017")
    parser.add_argument("--annotation-file", type=str,
                        default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--answers-file", type=str,
                        default="outputs/chair/llava-7b-greedy/captions.jsonl")
    parser.add_argument("--question", type=str, default="Describe this image in detail.")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    # contrastive-decoding options
    parser.add_argument("--cd-method", type=str, default="none", choices=list(CD_METHODS),
                        help="Decoding method: none (baseline), vcd, or sid.")
    parser.add_argument("--noise-step", type=int, default=500,
                        help="Diffusion noise step for the VCD negative image.")
    parser.add_argument("--cd-alpha", type=float, default=1.0)
    parser.add_argument("--cd-beta", type=float, default=0.1)
    args = parser.parse_args()

    eval_model(args)
