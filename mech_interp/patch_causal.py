#!/usr/bin/env python3
"""Activation patching: the causal per-layer effect of a CD method's perturbation.

For each hallucinated sample (H = object absent, expert says Yes) we:
  1. run the AMATEUR pass and capture, per block b, the last-token residual output A[b];
  2. run the CLEAN pass -> baseline decision margin m0 (real logits, not the lens);
  3. for each block b, re-run the CLEAN pass with ONE hook that overwrites the last-token
     residual at block b's output with A[b], and read the resulting margin m_b.

effect[b] = m_b - m0 is the *causal* contribution of injecting the amateur's state at
block b, with no linearity assumption (unlike the logit lens). A genuine correction would
drive the margin negative (Yes -> No) at the layer where presence is decided; we report
the mean effect and the flip-to-No fraction per block over H.

This is the experiment a mech-interp reviewer asks for; it also locates the true causal
depth (the logit lens can misplace it). Default method: vcd (same prompt length as the
clean pass; ICD/SID also supported, patched at the -1 decision position).

Output: <out-dir>/patch_effect_<method>.npz + patch_effect_<method>.csv
  layer (1..32), effect_mean_H, m_patched_mean_H, flip_frac_H, m0_mean_H, n_H
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "inference"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.mm_utils import get_model_name_from_path  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402

from extract_activations import (  # noqa: E402
    build_input_ids, label_to_int, load_questions, yes_no_token_ids,
)
from amateur_branches import METHODS, build_amateur_inputs  # noqa: E402

SPLITS = ("random", "popular", "adversarial")


@torch.inference_mode()
def final_margin(model, input_ids, image_tensor, yes_t, no_t, use_sid=None):
    out = model(input_ids, images=image_tensor.unsqueeze(0).half().cuda(),
                use_cache=False, return_dict=True, use_sid=use_sid)
    lg = out.logits[0, -1, :].float()
    return float(lg[yes_t].max() - lg[no_t].max())


@torch.inference_mode()
def capture_amateur_residuals(model, input_ids, image_tensor, n_blocks, use_sid=None):
    """Run amateur pass; return A[b] = last-token residual out of block b, for b=0..n-1."""
    cache = [None] * n_blocks
    handles = []

    def mk(b):
        def hook(_m, _i, out):
            cache[b] = out[0][:, -1, :].detach().clone()
        return hook

    for b in range(n_blocks):
        handles.append(model.model.layers[b].register_forward_hook(mk(b)))
    try:
        model(input_ids, images=image_tensor.unsqueeze(0).half().cuda(),
              use_cache=False, return_dict=True, use_sid=use_sid)
    finally:
        for h in handles:
            h.remove()
    return cache


@torch.inference_mode()
def patched_margin(model, input_ids, clean_image, block, residual, yes_t, no_t):
    """Clean pass with block's last-token residual overwritten by `residual`."""
    def hook(_m, _i, out):
        h = out[0]
        h[:, -1, :] = residual.to(h.dtype)
        return (h,) + tuple(out[1:])

    handle = model.model.layers[block].register_forward_hook(hook)
    try:
        out = model(input_ids, images=clean_image.unsqueeze(0).half().cuda(),
                    use_cache=False, return_dict=True)
        lg = out.logits[0, -1, :].float()
        return float(lg[yes_t].max() - lg[no_t].max())
    finally:
        handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="vcd", choices=METHODS)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--conv-mode", default="vicuna_v1")
    ap.add_argument("--noise-step", type=int, default=900)
    ap.add_argument("--noise-seed", type=int, default=1234)
    ap.add_argument("--max-samples", type=int, default=0, help="0=all; cap of H per split")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    args = ap.parse_args()

    disable_torch_init()
    name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, name)
    n_blocks = model.config.num_hidden_layers  # 32
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    yes_t = torch.tensor(yes_ids, device="cuda")
    no_t = torch.tensor(no_ids, device="cuda")

    pope_dir = os.path.join(args.data_dir, "pope", "coco")
    images_dir = os.path.join(pope_dir, "images")

    eff = np.zeros((n_blocks,), np.float64)
    mpatch = np.zeros((n_blocks,), np.float64)
    flip = np.zeros((n_blocks,), np.float64)
    m0_sum = 0.0
    n_H = 0
    row = 0

    for split in args.splits:
        questions = load_questions(pope_dir, split)
        cap = args.max_samples if args.max_samples > 0 else 10**9
        seen = 0
        for line in tqdm(questions, desc=f"patch:{args.method}:{split}"):
            if seen >= cap:
                break
            try:
                image = Image.open(os.path.join(images_dir, line["image"])).convert("RGB")
            except (FileNotFoundError, OSError):
                continue
            input_ids = build_input_ids(tokenizer, model, line["text"], args.conv_mode)
            clean_img = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

            m0 = final_margin(model, input_ids, clean_img, yes_t, no_t)
            pred_yes = m0 > 0
            gt = label_to_int(line["label"])
            row += 1
            if not (gt == 0 and pred_yes):  # restrict to hallucinations H
                continue

            torch.manual_seed(args.noise_seed + row)
            am = build_amateur_inputs(args.method, tokenizer, model, input_ids,
                                      line["text"], clean_img, args.conv_mode, args.noise_step)
            A = capture_amateur_residuals(model, am["input_ids"], am["image_tensor"],
                                          n_blocks, **am["fwd_kwargs"])
            for b in range(n_blocks):
                mb = patched_margin(model, input_ids, clean_img, b, A[b], yes_t, no_t)
                eff[b] += (mb - m0)
                mpatch[b] += mb
                flip[b] += 1.0 if mb < 0 else 0.0
            m0_sum += m0
            n_H += 1
            seen += 1

    denom = max(n_H, 1)
    layers = np.arange(1, n_blocks + 1)  # block b -> hidden index b+1
    effect_mean = eff / denom
    csv = os.path.join(args.out_dir, f"patch_effect_{args.method}.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(csv, "w") as f:
        f.write("layer,effect_mean_H,m_patched_mean_H,flip_frac_H,m0_mean_H,n_H\n")
        for i, l in enumerate(layers):
            f.write(f"{l},{effect_mean[i]:.6f},{mpatch[i]/denom:.6f},{flip[i]/denom:.6f},"
                    f"{m0_sum/denom:.6f},{n_H}\n")
    np.savez_compressed(
        os.path.join(args.out_dir, f"patch_effect_{args.method}.npz"),
        layer=layers, effect_mean_H=effect_mean, m_patched_mean_H=mpatch / denom,
        flip_frac_H=flip / denom, m0_mean_H=m0_sum / denom, n_H=n_H,
    )
    b_star = int(np.argmin(effect_mean))  # most suppressive (most negative) block
    print(f"[patch:{args.method}] n_H={n_H}  m0_mean={m0_sum/denom:+.2f}")
    print(f"[patch:{args.method}] most-suppressive block={b_star+1} effect={effect_mean[b_star]:+.2f}  "
          f"readout-block effect={effect_mean[-1]:+.2f} flip={flip[-1]/denom:.3f}")
    print(f"[patch:{args.method}] wrote {csv}")


if __name__ == "__main__":
    main()
