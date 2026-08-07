#!/usr/bin/env python3
"""Subspace-alignment test: is each CD method's perturbation aimed at the hallucination?

For every method we have, per layer l and per H u T sample, the contrastive perturbation
   dh[l] = h_expert[l] - h_amateur[l]
and the probe's presence/hallucination direction w_H[l] (unit, points absent->present).

A *selective* correction on a hallucinated Yes (H = absent) must move the residual AWAY
from "present": its projection onto w_H should be large and negative. We measure

  proj_H[l]  = mean over H of  dh[l] . w_H[l]       (signed; corrective => negative)
  |dh|_H[l]  = mean over H of  ||dh[l]||            (raw perturbation magnitude)
  cos_H[l]   = mean over H of  (dh[l].w_H[l]) / ||dh[l]||   (fraction along the direction)

If |dh| is large while cos ~ 0, the perturbation is ORTHOGONAL to the hallucination
subspace: CD moves the stream a lot, but not in the direction that would fix the error.
This upgrades "wrong layer / wrong magnitude" to "wrong subspace". T is reported as a
same-population reference.

Runs over every hiddens_ht_<method>.npz found. Writes only small summaries:
  <results-dir>/subspace_align.csv   (method, layer, proj_H, proj_T, dh_H, dh_T, cos_H, cos_T)
  <results-dir>/subspace_align.npz
"""
import argparse
import glob
import os

import numpy as np


def per_layer_stats(dh, w, mask):
    """dh:[n,4096] f32, w:[4096] unit, mask: bool over n -> (proj_mean, dhnorm_mean, cos_mean)."""
    d = dh[mask]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    proj = d @ w                      # [m]
    nrm = np.linalg.norm(d, axis=1)   # [m]
    cos = proj / (nrm + 1e-8)
    return float(proj.mean()), float(nrm.mean()), float(cos.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()

    probe = np.load(os.path.join(args.results_dir, "probe_dirs.npz"))
    W = probe["w_H"].astype(np.float32)  # [33,4096] unit
    L = W.shape[0]

    rows = []
    npz = {}
    for path in sorted(glob.glob(os.path.join(args.results_dir, "hiddens_ht_*.npz"))):
        method = os.path.basename(path)[len("hiddens_ht_"):-len(".npz")]
        z = np.load(path)
        gt = z["gt"].astype(int)
        H = gt == 0
        T = gt == 1
        exp = z["expert_hidden"]
        ama = z["amateur_hidden"]
        proj_H, proj_T, dh_H, dh_T, cos_H, cos_T = (np.zeros(L) for _ in range(6))
        for l in range(L):
            dh = (exp[:, l, :].astype(np.float32) - ama[:, l, :].astype(np.float32))
            proj_H[l], dh_H[l], cos_H[l] = per_layer_stats(dh, W[l], H)
            proj_T[l], dh_T[l], cos_T[l] = per_layer_stats(dh, W[l], T)
            rows.append((method, l, proj_H[l], proj_T[l], dh_H[l], dh_T[l], cos_H[l], cos_T[l]))
        npz[f"{method}_proj_H"] = proj_H
        npz[f"{method}_dh_H"] = dh_H
        npz[f"{method}_cos_H"] = cos_H
        npz[f"{method}_proj_T"] = proj_T
        npz[f"{method}_dh_T"] = dh_T
        npz[f"{method}_cos_T"] = cos_T
        print(f"[{method}] |dh|_H readout={dh_H[-1]:.2f}  cos_H readout={cos_H[-1]:+.3f}  "
              f"proj_H readout={proj_H[-1]:+.2f}  (corrective would be strongly negative)")

    csv = os.path.join(args.results_dir, "subspace_align.csv")
    with open(csv, "w") as f:
        f.write("method,layer,proj_H,proj_T,dh_H,dh_T,cos_H,cos_T\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f},{r[6]:.6f},{r[7]:.6f}\n")
    np.savez_compressed(os.path.join(args.results_dir, "subspace_align.npz"), **npz)
    print(f"[align] wrote {csv}")


if __name__ == "__main__":
    main()
