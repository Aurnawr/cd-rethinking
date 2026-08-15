#!/usr/bin/env python3
"""Qwen2.5-VL runtime: loading, prompting, forward passes, and the logit lens.

This is the Qwen counterpart of what `mech_interp/extract_activations.py` gets from this repo's
vendored LLaVA. Nothing here imports `llava` or `cd_utils` -- the vendored stack is welded to
transformers 4.31 and cannot coexist with the version Qwen2.5-VL needs.

Everything downstream (extractor, probes, figures) depends only on the four contracts below:

  build_inputs()    prompt construction, identical modulo chat template to pope_infer_base.py
  forward_hiddens() one forward -> last-token hidden state per layer + final logits
  lens_margins()    frozen-head logit lens -> Yes/No decision margin per layer
  yes_no_token_ids()

Module-path robustness
----------------------
Qwen2.5-VL's submodule layout moved in transformers 4.52 (`model.model.layers` ->
`model.model.language_model.layers`). Rather than branch on version we locate the text stack by
*shape*: the unique submodule owning both a `layers` ModuleList and a `norm`. Same for the head.
"""
import os
from typing import List, Optional

import torch

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# The answer-format suffix the repo's POPE pipeline appends to every question
# (inference/pope_infer_base.py). Kept byte-identical so the task is the same task.
ANSWER_SUFFIX = " Answer the question using a single word or phrase."

# Qwen's own default system message. The expert branch uses it; ICD's amateur replaces it.
DEFAULT_SYSTEM = "You are a helpful assistant."


# --------------------------------------------------------------------------- loading
def _resolve_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def pick_device(pref: str = "auto") -> str:
    """'auto' -> cuda when available, else cpu (CPU is for the tiny-model tests, not real runs)."""
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    print("[device] no CUDA device visible -- falling back to CPU (unusably slow for a real run)")
    return "cpu"


def load_model(model_path: str = MODEL_ID, dtype: str = "bfloat16", attn_impl: str = "eager",
               min_pixels: Optional[int] = None, max_pixels: Optional[int] = None,
               device: str = "cuda"):
    """Load Qwen2.5-VL + its processor.

    attn_impl defaults to "eager" because SID's CT2S needs real attention weights out of one
    decoder layer; sdpa returns None for them. See sid_ct2s.py.

    min_pixels/max_pixels bound the dynamic-resolution vision-token count. Left at None the
    processor's own defaults apply, which for COCO val2014 (~640x480) yields ~390 vision tokens
    -- comfortably cheap, so we do not cap by default.
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc_kwargs = {}
    if min_pixels is not None:
        proc_kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        proc_kwargs["max_pixels"] = int(max_pixels)
    processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)

    torch_dtype = _resolve_dtype(dtype)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, attn_implementation=attn_impl,
        )
    except TypeError:  # transformers >= 5 renamed torch_dtype -> dtype
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch_dtype, attn_implementation=attn_impl,
        )
    model.to(device).eval()
    return processor, model


# --------------------------------------------------------------------------- module accessors
def text_model(model):
    """The decoder stack: the submodule owning both `.layers` (ModuleList) and `.norm`."""
    base = getattr(model, "model", model)
    for cand in (getattr(base, "language_model", None), base, model):
        if cand is None:
            continue
        if isinstance(getattr(cand, "layers", None), torch.nn.ModuleList) and hasattr(cand, "norm"):
            return cand
    for _, mod in model.named_modules():
        if isinstance(getattr(mod, "layers", None), torch.nn.ModuleList) and hasattr(mod, "norm"):
            return mod
    raise RuntimeError(
        "could not locate the Qwen text decoder stack (a module with .layers and .norm); "
        "the transformers layout changed -- update text_model() in qwen_runtime.py"
    )


def decoder_layers(model) -> torch.nn.ModuleList:
    return text_model(model).layers


def final_norm(model):
    return text_model(model).norm


def lm_head_weight(model) -> torch.Tensor:
    """The output embedding matrix [V, d], tied or untied."""
    head = getattr(model, "lm_head", None)
    if head is not None and hasattr(head, "weight"):
        return head.weight
    emb = model.get_output_embeddings()
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight
    raise RuntimeError("could not locate the lm_head weight")


def image_token_id(model, processor) -> int:
    """Token id of the per-vision-token placeholder (`<|image_pad|>`, 151655 for Qwen2.5-VL).

    The processor expands one `<|image_pad|>` into N of them, one per post-merge vision token,
    so `input_ids == image_token_id` marks exactly the vision band -- no hardcoded system-prompt
    length or fixed 576 offset (which is what `mech_interp/sid_correct.py` had to fall back on).
    """
    for attr in ("image_token_id", "image_token_index"):
        val = getattr(model.config, attr, None)
        if isinstance(val, int):
            return val
    tid = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if tid is None or tid < 0:
        raise RuntimeError("could not determine the image token id")
    return int(tid)


def n_hidden_layers(model) -> int:
    cfg = model.config
    return int(getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers)


# --------------------------------------------------------------------------- prompting
def build_messages(question: str, system: str = DEFAULT_SYSTEM):
    """POPE question -> Qwen chat messages, with the repo's single-word answer suffix."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question + ANSWER_SUFFIX},
        ]},
    ]


def build_inputs(processor, image, question: str, system: str = DEFAULT_SYSTEM,
                 device: str = "cuda"):
    """Tokenized+preprocessed model inputs for one (image, question).

    Two-step apply_chat_template(tokenize=False) -> processor(...) rather than the one-shot
    tokenizing form: it is the version-stable route and keeps the rendered prompt inspectable.
    """
    text = processor.apply_chat_template(
        build_messages(question, system), tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def yes_no_token_ids(tokenizer):
    """Token ids that can begin a 'Yes' / 'No' answer (cased and space-prefixed variants)."""
    yes_ids, no_ids = set(), set()
    for word, bucket in (("Yes", yes_ids), ("yes", yes_ids), ("No", no_ids), ("no", no_ids)):
        for variant in (word, " " + word):
            ids = tokenizer(variant, add_special_tokens=False).input_ids
            if ids:
                bucket.add(int(ids[0]))
    if not yes_ids or not no_ids:
        raise RuntimeError("tokenizer produced no Yes/No first-token ids")
    return sorted(yes_ids), sorted(no_ids)


def label_to_int(label: str) -> int:
    """POPE label 'yes'/'no' -> 1 (object present) / 0 (absent)."""
    return 1 if str(label).strip().lower() == "yes" else 0


def decode_is_yes(tokenizer, token_id: int) -> int:
    return 1 if tokenizer.decode([int(token_id)]).strip().lower().startswith("yes") else 0


# --------------------------------------------------------------------------- forward + lens
@torch.inference_mode()
def forward_hiddens(model, inputs):
    """One forward pass.

    Returns (hs [L+1, d] on device in model dtype, last_logits [V] float32 on device), where
    hs[i] is the last-token hidden state at layer i (0 = embeddings, L = post-final-norm).
    """
    out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    hs = torch.stack([h[0, -1, :] for h in out.hidden_states], dim=0)
    return hs, out.logits[0, -1, :].float()


@torch.inference_mode()
def lens_margins(model, hs, yes_ids: torch.Tensor, no_ids: torch.Tensor):
    """Logit-lens Yes-No margin per layer. hs [L+1, d] -> np.float32 [L+1].

    HuggingFace's Qwen2 decoder appends the final hidden state AFTER `self.norm`, so
    hidden_states[L] is already the lm_head input. Layers 0..L-1 are pre-norm residuals: push
    them through the frozen final RMSNorm; leave layer L untouched so the readout margin is
    *exactly* the true margin (asserted by selfcheck.py's lens-fidelity check).
    """
    W = lm_head_weight(model)
    h = hs.to(W.dtype).clone()
    h[:-1] = final_norm(model)(h[:-1])
    logits = h @ W.t()  # [L+1, V]
    yes = logits[:, yes_ids].max(dim=1).values
    no = logits[:, no_ids].max(dim=1).values
    return (yes - no).float().cpu().numpy()


def vision_positions(input_ids: torch.Tensor, img_token_id: int) -> torch.Tensor:
    """Key positions of the vision band in a [1, seq] input_ids."""
    return (input_ids[0] == img_token_id).nonzero(as_tuple=True)[0]


# --------------------------------------------------------------------------- misc
def load_questions(pope_dir: str, split: str):
    """POPE split jsonl (one json object per line), same layout download_pope_data.py writes."""
    import json
    path = os.path.join(pope_dir, f"coco_pope_{split}.json")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
