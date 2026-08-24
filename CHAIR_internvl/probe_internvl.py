"""
probe_internvl.py -- discover the InternVL3-8B facts that SID and the
two-branch KV cache depend on, before writing code that assumes them.

Guessing any of these produces silently wrong results rather than a crash --
exactly the failure mode that put a wrong APC criterion into this project's
appendix figures -- so they get checked against the real model first.

What it reports:
  1. module path to the language-model decoder layers (for the SID hooks)
  2. the image-token id and how many image tokens a real image expands to
  3. whether eager attention actually returns attention weights (SID needs the
     attention matrix; SDPA never materialises it)
  4. the shape of the returned attention weights
  5. whether a two-branch KV-cached continuation reproduces a stateless
     recompute (the assumption the whole cached design rests on)

Usage:
  python probe_internvl.py --model-path ~/models/InternVL3-8B-hf --image <path>
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

SYSTEM_PROMPT = (
    "You are a helpful assistant. Always respond in English only, in plain text "
    "without any markdown formatting (no bold, no headers, no bullet symbols)."
)
PROMPT = "Describe this image in detail."


def find_layers(model):
    """Locate the decoder-layer ModuleList without assuming a path."""
    hits = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
            child = mod[0]
            if any(hasattr(child, a) for a in ("self_attn", "attention", "attn")):
                hits.append((name, len(mod), type(child).__name__))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    print("== loading with attn_implementation='eager' (SID needs real attention weights) ==",
          flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path, crop_to_patches=True, min_patches=1, max_patches=12)
    ip = processor.image_processor
    ip.crop_to_patches, ip.min_patches, ip.max_patches = True, 1, 12
    print(f"   crop_to_patches={ip.crop_to_patches} (must be True)")
    print(f"   model class: {type(model).__name__}")
    print(f"   dtype: {model.dtype}   device: {model.device}")

    print("\n== 1. decoder-layer ModuleList candidates ==", flush=True)
    for name, n, cls in find_layers(model):
        print(f"   {name}  ({n} layers, {cls})")

    print("\n== 2. image token id / expansion ==", flush=True)
    img = Image.open(args.image).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=img, text=[text], return_tensors="pt").to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    ids = inputs["input_ids"][0]
    counts = torch.bincount(ids)
    top = torch.topk(counts, 3)
    print(f"   input_ids length: {len(ids)}")
    print(f"   pixel_values shape: {tuple(inputs['pixel_values'].shape)}  "
          f"(tiles, 3, 448, 448)")
    for cnt, tid in zip(top.values.tolist(), top.indices.tolist()):
        tokstr = processor.tokenizer.decode([tid])
        print(f"   most frequent id {tid}: {cnt} occurrences -> {tokstr!r}")
    for name in ("<IMG_CONTEXT>", "<img>", "</img>", "<image>"):
        tid = processor.tokenizer.convert_tokens_to_ids(name)
        print(f"   token {name!r} -> id {tid}")
    print(f"   inputs keys: {sorted(inputs.keys())}")

    print("\n== 3/4. does eager attention return weights? ==", flush=True)
    layers_name, layers = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8 and hasattr(mod[0], "self_attn"):
            layers_name, layers = name, mod
            break
    if layers is None:
        print("   !! could not locate decoder layers -- SID hooks cannot be written blind")
        return
    print(f"   using: {layers_name} ({len(layers)} layers)")

    captured = {}

    def pre_hook(mod, a, kw):
        kw["output_attentions"] = True
        return a, kw

    def post_hook(mod, a, kw, out):
        captured["out"] = out
        return out

    h1 = layers[2].self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = layers[2].self_attn.register_forward_hook(post_hook, with_kwargs=True)
    with torch.no_grad():
        out = model(**inputs, use_cache=True, return_dict=True)
    h1.remove(); h2.remove()

    got = captured.get("out")
    if isinstance(got, tuple):
        print(f"   layer-2 self_attn returned a {len(got)}-tuple")
        for i, el in enumerate(got):
            print(f"     [{i}] {type(el).__name__}"
                  + (f" shape={tuple(el.shape)}" if hasattr(el, "shape") else ""))
        w = got[1] if len(got) > 1 else None
        if w is None:
            print("   !! attention weights are None even under eager -- SID needs another route")
        else:
            print(f"   OK attention weights shape {tuple(w.shape)} "
                  f"(batch, heads, q_len, kv_len)")
    else:
        print(f"   unexpected return type: {type(got)}")

    print("\n== 5. KV-cache continuation vs stateless recompute ==", flush=True)
    logits_full = out.logits[0, -1].float()
    cache = out.past_key_values
    nxt = int(logits_full.argmax())
    new_tok = torch.tensor([[nxt]], device=model.device)

    with torch.no_grad():
        cached = model(
            input_ids=new_tok,
            attention_mask=torch.ones((1, ids.shape[0] + 1), dtype=torch.long, device=model.device),
            past_key_values=cache, use_cache=True, return_dict=True)
    cached_logits = cached.logits[0, -1].float()

    full_ids = torch.cat([inputs["input_ids"], new_tok], dim=1)
    with torch.no_grad():
        stateless = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            pixel_values=inputs["pixel_values"], use_cache=False, return_dict=True)
    stateless_logits = stateless.logits[0, -1].float()

    diff = (cached_logits - stateless_logits).abs().max().item()
    print(f"   max |cached - stateless| = {diff:.5f}")
    print(f"   argmax agree: {int(cached_logits.argmax()) == int(stateless_logits.argmax())}")
    print("   (a few logit units at bf16 is float noise; argmax must agree)")

    print("\nprobe complete.")


if __name__ == "__main__":
    main()
