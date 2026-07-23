"""THRONE ``Evaluatee`` that drives this repo's contrastive-decoding (CD) methods.

THRONE (``third_party/THRONE``) evaluates *free-form* image captions for object
hallucination. Its generation step (``throne_generate.py``) loads a model through
the ``Evaluatee`` interface and calls, in order:

    model = get_evaluated_model(name, args)
    prompt = model.format_prompt("Describe this image in detail.")   # BEFORE load()
    model.load()
    input_ids = model.tokenize_prompt(prompt)                        # once
    for (coco_ids, pil_images) in dataloader:
        texts = model.generate_batch(input_ids, pil_images)          # per batch

This class implements that interface, but instead of THRONE's stock LLaVA-v1.6
loader it uses **this repository's** custom LLaVA-v1.5 stack
(``llava.model.builder.load_pretrained_model``) so the CD hooks work:

  * VCD / ICD / SID / APC monkeypatch ``transformers`` ``GenerationMixin``
    (``cd_utils`` / ``spurious_utils``), and
  * the patched search loops read ``images_cd`` / ``input_ids_cd`` / ``use_sid`` /
    ``use_apc`` out of the generate kwargs and call the model's
    ``prepare_inputs_for_generation_vcd`` / ``_sid`` helpers.

Generation is done **one image at a time** (batch size 1), exactly mirroring the
per-image ``model.generate(...)`` calls in ``inference/mme_infer_*.py`` so the CD
math is identical to the MME/POPE runs. THRONE's dataloader ``batch_size`` only
controls how many images are handed to :meth:`generate_batch` at once; each is
still decoded independently.

The prompt is the THRONE default ("Describe this image in detail.") with **no**
"single word or phrase" suffix — this is free-form captioning, not yes/no QA.

Excluded methods: **OLM** (greedy patch that hard-codes the yes/no token ids
3869/1939) and **PBA** (an "Answer yes whenever possible" prompt suffix) are
yes/no-QA tricks with no meaningful free-form behavior, so they are not offered
here. See ``README_THRONE.md``.
"""

import os

import torch

# transformers >= 4.36 ships a native "llava" AutoConfig entry that conflicts
# with this repo's custom LlavaConfig registration in llava_llama.py.  Remove
# the built-in entry before the llava package is imported so the repo's own
# registration (AutoConfig.register("llava", LlavaConfig)) succeeds.
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if "llava" in CONFIG_MAPPING._mapping:
        del CONFIG_MAPPING._mapping["llava"]
except (ImportError, AttributeError, KeyError):
    pass

# transformers >= 4.35 removed _expand_mask and _make_causal_mask from
# bloom.modeling_bloom and opt.modeling_opt, but
# llava/model/language_model/mpt/hf_prefixlm_converter.py still imports them.
# Restore all four so the llava package imports cleanly.
try:
    import torch as _torch
    import transformers.models.bloom.modeling_bloom as _bloom_mod
    import transformers.models.opt.modeling_opt as _opt_mod

    if not hasattr(_bloom_mod, "_expand_mask"):
        def _bloom_expand_mask(mask: _torch.Tensor, tgt_length: int) -> _torch.BoolTensor:
            batch_size, src_length = mask.shape
            tgt_length = tgt_length if tgt_length is not None else src_length
            expanded_mask = ~(mask[:, None, None, :].to(_torch.bool))
            return expanded_mask.expand(batch_size, 1, tgt_length, src_length)
        _bloom_mod._expand_mask = _bloom_expand_mask

    if not hasattr(_bloom_mod, "_make_causal_mask"):
        def _bloom_make_causal_mask(
            input_ids_shape: _torch.Size, device: _torch.device, past_key_values_length: int
        ) -> _torch.BoolTensor:
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
        from typing import Optional as _Optional
        def _opt_expand_mask(
            mask: _torch.Tensor, dtype: _torch.dtype, tgt_len: _Optional[int] = None
        ):
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len
            expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            inverted = 1.0 - expanded
            return inverted.masked_fill(inverted.to(_torch.bool), _torch.finfo(dtype).min)
        _opt_mod._expand_mask = _opt_expand_mask

    if not hasattr(_opt_mod, "_make_causal_mask"):
        def _opt_make_causal_mask(
            input_ids_shape: _torch.Size,
            dtype: _torch.dtype,
            device: _torch.device,
            past_key_values_length: int = 0,
        ):
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

except (ImportError, Exception):
    pass

# transformers 4.36+ defaults to SDPA attention but LlavaLlamaModel doesn't
# declare support, causing ValueError on model init.  Patch the method that
# sets attention implementation to fall back to "eager" on failure.
try:
    import transformers as _tf
    _orig_autoset = _tf.PreTrainedModel._autoset_attn_implementation.__func__  # type: ignore[attr-defined]

    @classmethod  # type: ignore[misc]
    def _patched_autoset(cls, config, **kwargs):
        try:
            return _orig_autoset(cls, config, **kwargs)
        except ValueError:
            config._attn_implementation = "eager"
            return config

    _tf.PreTrainedModel._autoset_attn_implementation = _patched_autoset
except Exception:
    pass

# CD / spurious monkeypatch installers from this repo's ``inference/`` package.
# (Requires ``inference/`` on PYTHONPATH, which the throne_*.sh scripts set.)
from cd_utils.vcd_utils import (
    add_diffusion_noise,
    evolve_vcd_greedy_search,
    evolve_vcd_sampling,
)
from cd_utils.icd_utils import (
    get_random_icd_prompt,
    evolve_icd_greedy_search,
    evolve_icd_sampling,
)
from cd_utils.sid_utils import evolve_sid_greedy_search, evolve_sid_sampling
from spurious_utils.apc_utils import evolve_apc_sampling


# Methods whose decoding logic is meaningful for free-form generation.
CD_METHODS = ("none", "vcd", "icd", "sid", "apc")


def install_cd_patches(cd_method):
    """Monkeypatch ``GenerationMixin`` for the selected CD method.

    ``none`` installs nothing (stock greedy/sampling). VCD/ICD/SID patch both the
    greedy and sampling search loops; APC patches the sampling loop only (it is a
    sampling-oriented method), mirroring ``inference/mme_infer_{cd,apc}.py``.
    """
    if cd_method == "vcd":
        evolve_vcd_greedy_search()
        evolve_vcd_sampling()
    elif cd_method == "icd":
        evolve_icd_greedy_search()
        evolve_icd_sampling()
    elif cd_method == "sid":
        evolve_sid_greedy_search()
        evolve_sid_sampling()
    elif cd_method == "apc":
        evolve_apc_sampling()
    elif cd_method == "none":
        pass
    else:
        raise ValueError(
            f"Unknown cd_method {cd_method!r}; choose one of {CD_METHODS}."
        )


class LLaVA_CD(object):
    """LLaVA-v1.5 evaluatee with optional contrastive decoding.

    Configured entirely from ``args`` (set by the ``LLaVA_CD`` subparser added to
    ``throne_generate.py``):
      ``model_path``, ``model_base``, ``conv_template_name``, ``cd_method``,
      ``temperature``, ``top_p``, ``noise_step``.
    """

    def __init__(self, args):
        self.args = args
        self.cd_method = (getattr(args, "cd_method", "none") or "none").lower()
        if self.cd_method not in CD_METHODS:
            raise ValueError(
                f"--cd_method must be one of {CD_METHODS}, got {self.cd_method!r}. "
                "OLM and PBA are intentionally unsupported for free-form THRONE "
                "generation (see README_THRONE.md)."
            )
        self.temperature = float(getattr(args, "temperature", 0.2) or 0.0)
        self.top_p = getattr(args, "top_p", None)
        self.noise_step = int(getattr(args, "noise_step", 900) or 900)
        self.conv_template_name = getattr(args, "conv_template_name", "vicuna_v1")

        # Import this repo's LLaVA lazily so the module also imports cleanly when
        # only inspecting CD_METHODS / install_cd_patches.
        import llava.constants
        import llava.conversation
        import llava.mm_utils
        import llava.model.builder

        self.llava = llava
        self._raw_prompt = None  # set by format_prompt

    # -- Evaluatee interface ---------------------------------------------------

    def load(self):
        from llava.utils import disable_torch_init

        disable_torch_init()
        # Patches monkeypatch global GenerationMixin methods; install before the
        # first generate(), mirroring mme_infer_*.py (which patch before load).
        install_cd_patches(self.cd_method)

        model_path = os.path.expanduser(self.args.model_path)
        self.model_name = self.llava.mm_utils.get_model_name_from_path(model_path)
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = self.llava.model.builder.load_pretrained_model(
            model_path, self.args.model_base, self.model_name, device="cuda"
        )

    def format_prompt(self, prompt):
        """Build the LLaVA conversation prompt for the free-form question.

        Called by THRONE *before* :meth:`load`, so it relies only on the
        conversation template + constants (no model). Matches the image-token
        handling used by ``inference/mme_infer_*.py`` for llava-v1.5
        (``mm_use_im_start_end`` is False for llava-v1.5-7b).
        """
        self._raw_prompt = prompt
        return self._build_prompt(prompt, system=None)

    def tokenize_prompt(self, formatted_prompt):
        input_ids = self.llava.mm_utils.tokenizer_image_token(
            formatted_prompt,
            self.tokenizer,
            self.llava.constants.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        return input_ids  # shape [L] (CPU); moved to cuda per-image in generate

    def generate_batch(self, input_ids, imgs):
        """Decode each image in ``imgs`` independently (batch size 1)."""
        do_sample = self.temperature > 0
        return [self._generate_one(input_ids, img, do_sample) for img in imgs]

    # -- internals -------------------------------------------------------------

    def _build_prompt(self, raw_prompt, system=None):
        qs = self.llava.constants.DEFAULT_IMAGE_TOKEN + "\n" + raw_prompt
        conv = self.llava.conversation.conv_templates[self.conv_template_name].copy()
        if system is not None:
            conv.system = system  # ICD negative system prompt
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _generate_one(self, input_ids, img, do_sample):
        ids = input_ids.unsqueeze(0).to("cuda")  # [1, L]
        image_tensor = self.image_processor.preprocess(img, return_tensors="pt")[
            "pixel_values"
        ][0]
        images = image_tensor.unsqueeze(0).half().cuda()

        # Per-method CD kwargs, each matching the corresponding mme_infer_*.py call.
        gen_kwargs = dict(
            images=images,
            do_sample=do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True,
        )
        if self.cd_method == "vcd":
            gen_kwargs["images_cd"] = (
                add_diffusion_noise(image_tensor, self.noise_step)
                .unsqueeze(0)
                .half()
                .cuda()
            )
        elif self.cd_method == "icd":
            icd_prompt = get_random_icd_prompt()
            prompt_cd = self._build_prompt(self._raw_prompt, system=icd_prompt)
            gen_kwargs["input_ids_cd"] = (
                self.llava.mm_utils.tokenizer_image_token(
                    prompt_cd,
                    self.tokenizer,
                    self.llava.constants.IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .cuda()
            )
        elif self.cd_method == "sid":
            gen_kwargs["use_sid"] = True
        elif self.cd_method == "apc":
            gen_kwargs["use_apc"] = True

        with torch.inference_mode():
            output_ids = self.model.generate(ids, **gen_kwargs)

        # This repo's llava-v1.5 generate returns input + new tokens, so slice off
        # the prompt before decoding (matching inference/mme_infer_*.py).
        input_token_len = ids.shape[1]
        text = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        return text.strip()
