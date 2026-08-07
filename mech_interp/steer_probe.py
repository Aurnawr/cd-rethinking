#!/usr/bin/env python3
"""The correction CD isn't: probe-direction steering at the formation layer.

CD's shift is blind (non-selective) and one-sided. As a counterfactual we take the
probe's presence/hallucination direction w_H[L] (from fit_probe_dirs) and, on hallucinated
samples, subtract beta * w_H[L] from the last-token residual at hidden-layer L (i.e. the
output of block L-1). This is a *targeted* intervention in exactly the subspace CD ignores.

We report, per (layer L, strength beta):
  H_flip_frac : fraction of hallucinations (absent, wrongly Yes) driven to No  -> correction
  T_flip_frac : fraction of correct positives (present, Yes) driven to No       -> collateral

A good correction has high H_flip with low T_flip -- showing suppression was cheaply
available at the very layer CD skips. Defaults steer at the presence-formation layer
(first with probe AUC >= 0.95) and a couple of layers deeper.

Output: <out-dir>/steer_sweep.csv (layer, beta, n_H, H_flip_frac, n_T, T_flip_frac)
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

SPLITS = ("random", "popular", "adversarial")


@torch.inference_mode()
def steered_margin(model, input_ids, clean_image, block, vec, yes_t, no_t):
    """Clean pass with `vec` added to the last-token residual at `block` output.

    block=None -> no steering (baseline). `vec` already includes the sign/scale.
    """
    handle = None
    if block is not None:
        def hook(_m, _i, out):
            h = out[0]
            h[:, -1, :] = h[:, -1, :] + vec.to(h.dtype)
            return (h,) + tuple(out[1:])
        handle = model.model.layers[block].register_forward_hook(hook)
    try:
        out = model(input_ids, images=clean_image.unsqueeze(0).half().cuda(),
                    use_cache=False, return_dict=True)
        lg = out.logits[0, -1, :].float()
        return float(lg[yes_t].max() - lg[no_t].max())
    finally:
        if handle is not None:
            handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--probe-dirs", required=True, help="probe_dirs.npz from fit_probe_dirs.py")
    ap.add_argument("--conv-mode", default="vicuna_v1")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="hidden-layer indices to steer at (default: formation layer + a couple)")
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[2.0, 4.0, 8.0, 16.0, 32.0])
    ap.add_argument("--t-cap", type=int, default=400, help="max correct positives for collateral")
    ap.add_argument("--max-samples", type=int, default=0, help="0=all; cap questions per split")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    args = ap.parse_args()

    disable_torch_init()
    name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, name)
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    yes_t = torch.tensor(yes_ids, device="cuda")
    no_t = torch.tensor(no_ids, device="cuda")

    probe = np.load(args.probe_dirs)
    W = torch.tensor(probe["w_H"], device="cuda")  # [33,4096] unit, absent->present
    auc = probe["auc_H"]
    if args.layers is None:
        form = int(np.argmax(auc >= 0.95)) if (auc >= 0.95).any() else 8
        args.layers = sorted({form, min(form + 4, 32), min(form + 8, 32)})
    print(f"[steer] layers={args.layers}  betas={args.betas}  (formation by AUC>=0.95)")

    pope_dir = os.path.join(args.data_dir, "pope", "coco")
    images_dir = os.path.join(pope_dir, "images")

    # accumulators keyed by (layer, beta): flips on H and on T
    H_flip = {(l, b): 0 for l in args.layers for b in args.betas}
    T_flip = {(l, b): 0 for l in args.layers for b in args.betas}
    n_H = 0
    n_T = 0

    for split in args.splits:
        questions = load_questions(pope_dir, split)
        if args.max_samples > 0:
            questions = questions[: args.max_samples]
        for line in tqdm(questions, desc=f"steer:{split}"):
            try:
                image = Image.open(os.path.join(images_dir, line["image"])).convert("RGB")
            except (FileNotFoundError, OSError):
                continue
            input_ids = build_input_ids(tokenizer, model, line["text"], args.conv_mode)
            clean_img = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            m0 = steered_margin(model, input_ids, clean_img, None, None, yes_t, no_t)
            pred_yes = m0 > 0
            gt = label_to_int(line["label"])
            is_H = (gt == 0 and pred_yes)
            is_T = (gt == 1 and pred_yes)
            if not (is_H or is_T):
                continue
            if is_T and n_T >= args.t_cap:
                continue
            for l in args.layers:            # hidden layer L -> steer block L-1
                block = max(l - 1, 0)
                for b in args.betas:
                    vec = -b * W[l]          # subtract presence direction -> toward No
                    m = steered_margin(model, input_ids, clean_img, block, vec, yes_t, no_t)
                    if m < 0:                # flipped to No
                        if is_H:
                            H_flip[(l, b)] += 1
                        else:
                            T_flip[(l, b)] += 1
            if is_H:
                n_H += 1
            else:
                n_T += 1

    os.makedirs(args.out_dir, exist_ok=True)
    csv = os.path.join(args.out_dir, "steer_sweep.csv")
    with open(csv, "w") as f:
        f.write("layer,beta,n_H,H_flip_frac,n_T,T_flip_frac\n")
        for l in args.layers:
            for b in args.betas:
                hf = H_flip[(l, b)] / max(n_H, 1)
                tf = T_flip[(l, b)] / max(n_T, 1)
                f.write(f"{l},{b},{n_H},{hf:.4f},{n_T},{tf:.4f}\n")
                print(f"  L={l:2d} beta={b:5.1f}  H_flip={hf:.3f} (corrected)  T_flip={tf:.3f} (collateral)")
    print(f"[steer] n_H={n_H}  n_T={n_T}  wrote {csv}")


if __name__ == "__main__":
    main()
