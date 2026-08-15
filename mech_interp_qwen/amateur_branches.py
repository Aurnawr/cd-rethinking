#!/usr/bin/env python3
"""Per-method amateur (contrastive) forward passes for VCD / ICD / SID on Qwen2.5-VL.

Every contrastive-decoding method here forms the SAME logit algebra

    z_CD = (1 + alpha) * z_expert  -  alpha * z_amateur          (+ plausibility mask)

and differs ONLY in how the *amateur* pass is built. Isolating that one degree of freedom is what
lets the mechanistic battery run method-agnostically: swap the amateur, keep everything else
identical to the expert pass.

  VCD : same prompt, `add_diffusion_noise(pixel_values, 900)`   (arXiv 2311.16922)
  ICD : clean image, adversarial system message                  (arXiv 2403.18715)
  SID : clean image + CT2S attention masking                     (arXiv 2408.02032, see sid_ct2s.py)

Deliberate deviation, carried over from `mech_interp/amateur_branches.py`
------------------------------------------------------------------------
The repo's shipped `get_random_icd_prompt()` samples one of five instructions per decoding step.
For a controlled mechanistic contrast the amateur is FIXED to the canonical adversarial negative
-- the one that actually degrades grounding -- which removes per-sample sampling noise from Delta.
The SID paper quotes this same family of negatives when describing ICD.
"""
import contextlib

from qwen_runtime import build_inputs, vision_positions
from vcd_noise import add_diffusion_noise

METHODS = ("vcd", "icd", "sid")

ICD_NEG_PROMPT = (
    "You are a confused objects detector to provide a fuzzy overview or impression "
    "of the image."
)


def build_amateur_inputs(method: str, processor, expert_inputs, image, question: str,
                         noise_step: int, device: str = "cuda"):
    """Model inputs for `method`'s amateur pass on one sample.

    VCD/SID reuse the expert's tokenization (same prompt); only VCD's pixels differ, and SID's
    difference lives entirely in the attention mask (see `amateur_context`). ICD retokenizes
    because its system message changes.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if method == "icd":
        return build_inputs(processor, image, question, system=ICD_NEG_PROMPT, device=device)
    inputs = dict(expert_inputs)
    if method == "vcd":
        inputs["pixel_values"] = add_diffusion_noise(expert_inputs["pixel_values"], noise_step)
    return inputs


@contextlib.contextmanager
def amateur_context(method: str, inputs, sid=None, img_token_id: int = None):
    """Wrap the amateur forward. Only SID needs to install anything; the others are a no-op."""
    if method != "sid":
        yield None
        return
    if sid is None or img_token_id is None:
        raise ValueError("SID amateur requires a SidCT2S instance and the image token id")
    with sid.session(vision_positions(inputs["input_ids"], img_token_id)) as session:
        yield session
