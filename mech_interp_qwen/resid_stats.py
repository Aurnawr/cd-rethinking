#!/usr/bin/env python3
"""Per-layer residual-stream magnitude stats, per CD method, for the logit-lens figures.

On the hallucination set H (ground truth absent, expert answered Yes) we report, per layer:

  cd_raw_mean            mean_H ||h_expert - h_amateur||                 raw residual change
  clean_state_norm_mean  mean_H ||h_expert||                             the clean-norm artifact
  cd_relative_mean       mean_H ||h_expert - h_amateur|| / ||h_expert||  de-confounded

The raw curve is confounded: residual norms grow with depth in every transformer, so a raw-L2
peak can be an artifact of ||h|| rather than of the intervention. The relative curve divides it
out, per sample, before averaging.

Unlike `mech_interp/resid_stats.py`, which re-derived these from a multi-GB fp16 dump of the
H-union-T hidden states, this reads the small per-layer norms that extract_cd.py already recorded
in fp32 during the forward pass. Same definition, exact arithmetic, no hidden-tensor dump.

Output: <results-dir>/per_layer_resid_<method>.csv  (the --resid-csv of logit_lens_figures.py)
"""
import argparse
import csv
import glob
import os

import numpy as np


def load_h_population(act_dir: str, splits, method: str):
    """Concatenate cd_change_<method> and clean_norm over the hallucination population H."""
    cd, cn = [], []
    for split in splits:
        for fp in sorted(glob.glob(os.path.join(act_dir, f"{split}_shard*.npz"))):
            z = np.load(fp)
            key = f"cd_change_{method}"
            if key not in z.files:
                raise SystemExit(f"{fp} has no '{key}' -- rerun extract_cd.py including {method}")
            gt, pred = z["gt"].astype(int), z["pred"].astype(int)
            H = (gt == 0) & (pred == 1)
            if H.any():
                cd.append(z[key].astype(np.float32)[H])
                cn.append(z["clean_norm"].astype(np.float32)[H])
    if not cd:
        raise SystemExit(f"[resid_stats] no hallucinations found under {act_dir}")
    return np.concatenate(cd), np.concatenate(cn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--splits", nargs="+", default=["random", "popular", "adversarial"])
    args = ap.parse_args()

    cd, cn = load_h_population(args.act_dir, args.splits, args.method)  # [n_H, L+1] each
    cd_raw = cd.mean(axis=0)
    clean_state = cn.mean(axis=0)
    cd_rel = (cd / np.clip(cn, 1e-6, None)).mean(axis=0)

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, f"per_layer_resid_{args.method}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["layer", "cd_raw_mean", "clean_state_norm_mean", "cd_relative_mean"])
        for l in range(cd_raw.shape[0]):
            w.writerow([l, float(cd_raw[l]), float(clean_state[l]), float(cd_rel[l])])
    print(f"[resid_stats:{args.method}] n_H={cd.shape[0]}  layers={cd_raw.shape[0]}  wrote {out}")


if __name__ == "__main__":
    main()
