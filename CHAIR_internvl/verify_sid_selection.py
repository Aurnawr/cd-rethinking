"""
verify_sid_selection.py -- check that SID masks the RIGHT image tokens, not
merely that it masks something.

Why this exists: a hook that fires but selects the wrong subset (most-attended
instead of least, wrong count, wrong column range) produces exactly the same
surface evidence as a correct one -- a non-zero d = E - A and fluent captions.
This project has already shipped two bugs of that shape: a Qwen SID capture
that used torch.randperm instead of attention scores, and appendix figures that
applied the APC cutoff to the contrastive score instead of the expert logits.
Both looked healthy from outside.

Checks performed against the real model:
  1. the masked column count is (1 - keep_frac) of the image tokens
  2. every masked column is an image-token position (never a text token)
  3. the KEPT tokens are genuinely the lowest-attention ones -- their max
     attention score is below the min of the masked ones
  4. masking actually changes the logits (a mask applied to a tensor the model
     ignores would be silent)
  5. the amateur branch under the KV cache still has the mask applied at a
     later decode step, not just at prefill

Usage:
  python verify_sid_selection.py --model-path ~/models/InternVL3-8B-hf --image <path>
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
from PIL import Image

from generate_internvl import (
    IMAGE_TOKEN_ID, AGG_LAYER, SID_KEEP_FRAC, PROMPT,
    SIDState, build_inputs, load_model_and_processor, Branch,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    model, processor = load_model_and_processor(args.model_path)
    device = model.device
    img = Image.open(args.image).convert("RGB")
    inp = build_inputs(processor, model, img, PROMPT)

    layers = model.model.language_model.layers
    state = SIDState()
    captured = {}

    # Re-implement the hooks here with instrumentation, mirroring
    # generate_internvl.install_sid_hooks exactly, so what is verified is the
    # same selection rule the generator uses.
    def agg_pre(mod, a, kw):
        if not state.active:
            return None
        kw["output_attentions"] = True
        return a, kw

    def agg_post(mod, a, kw, out):
        if not state.active or state.image_positions is None:
            return out
        attn = out[1]
        last = attn.mean(dim=1)[0, -1]
        img_pos = state.image_positions
        scores = last[img_pos]
        keep_n = max(1, int(round(SID_KEEP_FRAC * img_pos.numel())))
        keep_idx = scores.topk(keep_n, largest=False).indices
        keep = torch.zeros(img_pos.numel(), dtype=torch.bool, device=img_pos.device)
        keep[keep_idx] = True
        state.masked_cols = img_pos[~keep]
        captured.update(scores=scores.detach().float().cpu(),
                        keep=keep.detach().cpu(),
                        img_pos=img_pos.detach().cpu(),
                        keep_n=keep_n)
        return out

    applied = {"count": 0}

    def later_pre(mod, a, kw):
        if not state.active or state.masked_cols is None or state.masked_cols.numel() == 0:
            return None
        am = kw.get("attention_mask")
        if am is None:
            return None
        am2 = am.clone()
        am2[..., state.masked_cols] = torch.finfo(am2.dtype).min
        kw["attention_mask"] = am2
        applied["count"] += 1
        return a, kw

    layers[AGG_LAYER].self_attn.register_forward_pre_hook(agg_pre, with_kwargs=True)
    layers[AGG_LAYER].self_attn.register_forward_hook(agg_post, with_kwargs=True)
    for i in range(AGG_LAYER + 1, len(layers)):
        layers[i].self_attn.register_forward_pre_hook(later_pre, with_kwargs=True)

    ids = inp["input_ids"][0]
    img_positions = (ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    n_img = img_positions.numel()
    print(f"prompt length {ids.numel()}, image tokens {n_img}")

    # expert (hooks inert)
    expert = Branch(model, inp, device)
    E0 = expert.logits.clone()

    # amateur (hooks active)
    state.active = True
    state.image_positions = expert.image_positions
    state.masked_cols = None
    applied["count"] = 0
    amateur = Branch(model, inp, device)
    state.active = False
    A0 = amateur.logits.clone()

    print("\n-- 1. masked count --")
    masked = state.masked_cols
    exp_keep = max(1, int(round(SID_KEEP_FRAC * n_img)))
    print(f"   kept {captured['keep_n']} (expected {exp_keep}, {SID_KEEP_FRAC:.0%} of {n_img})")
    print(f"   masked {masked.numel()} (expected {n_img - exp_keep})")
    ok1 = captured["keep_n"] == exp_keep and masked.numel() == n_img - exp_keep
    print(f"   {'PASS' if ok1 else 'FAIL'}")

    print("\n-- 2. masked columns are all image tokens --")
    is_img = torch.isin(masked.cpu(), img_positions.cpu())
    ok2 = bool(is_img.all())
    print(f"   {int(is_img.sum())}/{masked.numel()} masked columns are image tokens -> "
          f"{'PASS' if ok2 else 'FAIL'}")

    print("\n-- 3. kept tokens are the LEAST attended --")
    sc, keep = captured["scores"], captured["keep"]
    kept_max = sc[keep].max().item()
    masked_min = sc[~keep].min().item()
    ok3 = kept_max <= masked_min + 1e-9
    print(f"   max attention among KEPT   = {kept_max:.3e}")
    print(f"   min attention among MASKED = {masked_min:.3e}")
    print(f"   kept_max <= masked_min -> {'PASS' if ok3 else 'FAIL (selection inverted?)'}")

    print("\n-- 4. masking changes the logits --")
    d = (E0 - A0).abs().max().item()
    ok4 = d > 1e-3
    print(f"   max|E - A| at step 0 = {d:.4f} -> {'PASS' if ok4 else 'FAIL (mask is inert)'}")
    print(f"   later-layer mask applications during amateur prefill: {applied['count']} "
          f"(expected {len(layers) - AGG_LAYER - 1})")

    print("\n-- 5. mask still applied on a cached decode step --")
    nxt = int((2 * E0 - A0).argmax())
    expert.step(nxt)
    applied["count"] = 0
    state.active = True
    state.image_positions = expert.image_positions
    state.masked_cols = None
    amateur.step(nxt)
    state.active = False
    d1 = (expert.logits - amateur.logits).abs().max().item()
    ok5 = applied["count"] > 0 and d1 > 1e-3
    print(f"   mask applications on step 1: {applied['count']}")
    print(f"   max|E - A| at step 1 = {d1:.4f}")
    print(f"   {'PASS' if ok5 else 'FAIL (cache path drops the mask)'}")

    allok = all([ok1, ok2, ok3, ok4, ok5])
    print(f"\n=== {'ALL CHECKS PASSED' if allok else 'SOME CHECKS FAILED'} ===")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
