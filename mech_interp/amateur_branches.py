#!/usr/bin/env python3
"""Per-method amateur (contrastive) forward-pass constructors for VCD / ICD / SID.

Every contrastive-decoding method in this repo forms the SAME logit algebra

    z_CD = (1 + alpha) * z_expert  -  alpha * z_amateur          (+ plausibility mask)

and differs ONLY in how the *amateur* forward pass is built. This module exposes that
one degree of freedom so the mechanistic battery (logit lens, subspace alignment,
patching) can be run method-agnostically: pick a method, get its amateur pass, keep
everything else identical to the real inference pipeline.

The three amateurs, verified against the repo's own inference code
(`inference/pope_infer_cd.py`, `cd_utils/*`, `llava/model/language_model/*`):

  * VCD  : same text prompt, image replaced by `add_diffusion_noise(image, 900)`.
           (pope_infer_cd.py: `image_tensor_cd = add_diffusion_noise(...)`)
  * ICD  : clean image, system prompt replaced by an adversarial instruction.
           (pope_infer_cd.py: `conv_cd.system = icd_prompt`)
  * SID  : clean image + `use_sid=True`, masking attention to the LEAST-important
           vision tokens ranked by the model's own attention at layer AGG_LAYER-1
           (SID's CT2S strategy, arXiv 2408.02032). Single forward pass; the ranking
           reads one layer's attention internally. See `sid_correct.py`.

Design notes / deliberate deviations (documented for the writeup):
  * ICD's shipped `get_random_icd_prompt()` picks uniformly from four instructions each
    decoding step. For a controlled, deterministic mechanistic contrast we FIX the
    amateur to the canonical adversarial negative ("confused ... fuzzy overview"),
    the one that actually degrades grounding. This is the standard ICD negative prompt
    and removes per-sample sampling noise from Delta.
  * SID: the repo's shipped `use_sid` path selects vision tokens with `torch.randperm`
    (random dropout), which is NOT SID. We install a faithful CT2S forward at runtime
    (`sid_correct.install_correct_sid`) -- attention-ranked least-important tokens, at
    SID's official config (agg_layer=2, attention_rank=100) -- without editing any repo
    file. The corrected amateur is deterministic given the image (no RNG).
"""
import os
import sys

import torch

# Make the repo + its `inference/` (where cd_utils lives) importable, mirroring
# extract_activations.py so representations match the real pipeline exactly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "inference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llava.constants import (  # noqa: E402
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import tokenizer_image_token  # noqa: E402

from cd_utils.vcd_utils import add_diffusion_noise  # noqa: E402

METHODS = ("vcd", "icd", "sid")

ANSWER_SUFFIX = " Answer the question using a single word or phrase."

# Canonical ICD adversarial negative (the grounding-degrading instruction from the
# repo's icd_utils.get_random_icd_prompt list). Fixed for a deterministic contrast.
ICD_NEG_PROMPT = (
    "You are a confused objects detector to provide a fuzzy overview or impression "
    "of the image."
)


def _question_with_image(model, question_text: str) -> str:
    """Prompt body with image token(s) + answer suffix, matching pope_infer_base.py."""
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question_text
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question_text
    return qs + ANSWER_SUFFIX


def build_icd_input_ids(tokenizer, model, question_text: str, conv_mode: str):
    """input_ids for the ICD amateur: adversarial system prompt, same question/image.

    Replicates inference/pope_infer_cd.py's ICD branch (`conv_cd.system = icd_prompt`).
    """
    conv = conv_templates[conv_mode].copy()
    conv.system = ICD_NEG_PROMPT
    conv.append_message(conv.roles[0], _question_with_image(model, question_text))
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )


def build_amateur_image(method: str, clean_image_tensor, noise_step: int):
    """Return the image tensor the amateur pass should see (clean for icd/sid)."""
    if method == "vcd":
        return add_diffusion_noise(clean_image_tensor, noise_step)
    return clean_image_tensor  # icd, sid use the clean image


def amateur_forward_kwargs(method: str):
    """Extra kwargs the amateur forward() needs beyond (input_ids, images)."""
    return {"use_sid": True} if method == "sid" else {}


def build_amateur_inputs(
    method: str,
    tokenizer,
    model,
    expert_input_ids,
    question_text: str,
    clean_image_tensor,
    conv_mode: str,
    noise_step: int,
):
    """One place that turns (method, sample) -> everything the amateur pass needs.

    Returns dict with:
      input_ids   : LongTensor on cuda (ICD differs from expert; VCD/SID reuse expert)
      image_tensor: CPU float tensor to be .unsqueeze(0).half().cuda() by the caller
      fwd_kwargs  : extra forward() kwargs (use_sid for SID)
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if method == "icd":
        input_ids = build_icd_input_ids(tokenizer, model, question_text, conv_mode)
    else:
        input_ids = expert_input_ids
    return {
        "input_ids": input_ids,
        "image_tensor": build_amateur_image(method, clean_image_tensor, noise_step),
        "fwd_kwargs": amateur_forward_kwargs(method),
    }
