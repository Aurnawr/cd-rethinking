#!/usr/bin/env python3
"""Per-layer hallucination-direction probes from the saved H u T hidden states.

Within the model-says-Yes population (H = absent, T = present), separating H from T IS
object presence -- the exact signal a *selective* correction would need. We fit a linear
(logistic) probe per layer on the clean/expert residual and keep, for each layer:

  * w_H[l]   : the raw-hidden-space unit direction from H(absent) toward T(present).
               Steering subtracts beta * w_H[l] to push a hallucinated Yes back to No.
  * auc_H[l] : out-of-fold AUC (how decodable presence is at that layer).

The expert hidden states are identical across CD methods (they are the amateur-free
pass), so these directions are method-independent; we read them from whichever
hiddens_ht_<method>.npz is present (default: vcd).

Output: <results-dir>/probe_dirs.npz
  w_H   [33,4096] f32 (unit-norm, raw hidden space)
  auc_H [33] f32
  mean  [33,4096] f32   feature means (standardization reference)
  scale [33,4096] f32   feature stds
  n, n_H, n_T
"""
import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


def rank_auc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels).astype(bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ss = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--from-method", default="vcd",
                    help="which hiddens_ht_<method>.npz to read the expert hiddens from")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    path = os.path.join(args.results_dir, f"hiddens_ht_{args.from_method}.npz")
    z = np.load(path)
    X = z["expert_hidden"].astype(np.float32)  # [n,33,4096]
    gt = z["gt"].astype(int)                    # 1=present(T), 0=absent(H)
    n, L, D = X.shape
    print(f"[fit] {path}  n={n}  L={L}  D={D}  T={int(gt.sum())}  H={int((gt==0).sum())}")

    w_H = np.zeros((L, D), np.float32)
    auc_H = np.zeros(L, np.float32)
    mean = np.zeros((L, D), np.float32)
    scale = np.zeros((L, D), np.float32)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for l in range(L):
        Xl = X[:, l, :]
        scaler = StandardScaler().fit(Xl)
        Xs = scaler.transform(Xl)
        clf = LogisticRegression(C=args.C, max_iter=2000)
        oof = cross_val_predict(clf, Xs, gt, cv=cv, method="predict_proba")[:, 1]
        auc_H[l] = rank_auc(oof, gt)
        clf.fit(Xs, gt)  # final direction on all data
        # map scaled-space coef back to raw hidden space, unit-normalize
        w_raw = (clf.coef_[0] / scaler.scale_).astype(np.float32)
        nrm = np.linalg.norm(w_raw) + 1e-8
        w_H[l] = w_raw / nrm
        mean[l] = scaler.mean_.astype(np.float32)
        scale[l] = scaler.scale_.astype(np.float32)
        print(f"  layer {l:2d}  AUC={auc_H[l]:.3f}")

    out = os.path.join(args.results_dir, "probe_dirs.npz")
    np.savez_compressed(
        out, w_H=w_H, auc_H=auc_H, mean=mean, scale=scale,
        n=n, n_H=int((gt == 0).sum()), n_T=int(gt.sum()),
    )
    print(f"[fit] presence decodable (AUC>=0.95) from layer "
          f"{int(np.argmax(auc_H >= 0.95)) if (auc_H >= 0.95).any() else -1}")
    print(f"[fit] wrote {out}")


if __name__ == "__main__":
    main()
