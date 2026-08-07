#!/usr/bin/env python3
"""Per-layer residual-stream magnitude stats for the logit-lens figures.

Reads the H u T hidden states saved by extract_cd_generic (`hiddens_ht_<method>.npz`)
and writes a SMALL per-layer CSV so the residual-magnitude figures (cd_normalized_curves,
the right half of logit_lens_curves) can be plotted locally without pulling the ~GB
hidden tensors.

On the hallucination set H (gt==0 among the saved H u T population) we report, per layer:
  cd_raw_mean         mean_H ||h_expert - h_amateur||            (raw CD residual change)
  clean_state_norm    mean_H ||h_expert||                        (the clean-norm artifact)
  cd_relative_mean    mean_H ||h_expert - h_amateur|| / ||h_expert||   (de-confounded)

Matches the columns the original VCD `per_layer_normalized.csv` exposes, so the same
plotting code renders VCD and the new methods identically.
"""
import argparse
import csv
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--method", required=True)
    args = ap.parse_args()

    path = os.path.join(args.results_dir, f"hiddens_ht_{args.method}.npz")
    z = np.load(path)
    exp = z["expert_hidden"].astype(np.float32)   # [n, 33, 4096]
    ama = z["amateur_hidden"].astype(np.float32)  # [n, 33, 4096]
    gt = z["gt"].astype(int)                        # [n]  1=present, 0=absent
    H = gt == 0                                     # hallucinations within H u T
    if H.sum() == 0:
        raise SystemExit(f"[resid_stats] no hallucinations in {path}")

    dh = exp[H] - ama[H]                            # [n_H, 33, 4096]
    dh_norm = np.linalg.norm(dh, axis=2)           # [n_H, 33]
    clean_norm = np.linalg.norm(exp[H], axis=2)    # [n_H, 33]

    cd_raw = dh_norm.mean(axis=0)                   # [33]
    clean_state = clean_norm.mean(axis=0)           # [33]
    cd_rel = (dh_norm / np.clip(clean_norm, 1e-6, None)).mean(axis=0)  # [33]

    out = os.path.join(args.results_dir, f"per_layer_resid_{args.method}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "cd_raw_mean", "clean_state_norm_mean", "cd_relative_mean"])
        for l in range(cd_raw.shape[0]):
            w.writerow([l, float(cd_raw[l]), float(clean_state[l]), float(cd_rel[l])])
    print(f"[resid_stats:{args.method}] n_H={int(H.sum())}  wrote {out}")


if __name__ == "__main__":
    main()
