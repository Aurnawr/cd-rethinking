#!/usr/bin/env python3
"""Cross-method analysis + figures for the CD mechanistic battery.

Consumes whatever is present under --results-dir (robust to partial results):
  logit_lens_per_sample_<method>.npz   -> Delta battery per method (selectivity/sign/R^2)
  subspace_align.npz                   -> orthogonality (cos of perturbation vs direction)
  patch_effect_<method>.npz            -> causal per-layer effect
  steer_sweep.csv                      -> constructive counterfactual
  probe_dirs.npz                       -> presence-formation layer

Delta_l = clean_margin_l - amateur_margin_l is each method's per-unit-alpha shift toward
Yes. On the model-says-Yes population (H=absent, T=present) we test whether Delta carries
hallucination-specific info (selectivity AUC), its sign (frac Delta>0 on H), and how much
of its variance hallucination status explains (R^2). The claim generalizes iff every
method is non-selective (AUC~0.5), one-sided (frac~1), and R^2~0.

Writes figures (png) + summary_all.{json,csv} to --results-dir.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_ORDER = ["vcd", "icd", "sid"]
COLORS = {"vcd": "C0", "icd": "C1", "sid": "C2"}


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


def battery(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    cm, am = z["clean_margin"].astype(float), z["amateur_margin"].astype(float)
    gt, pred = z["gt"].astype(int), z["pred"].astype(int)
    delta = cm - am
    L = cm.shape[1]
    H = (gt == 0) & (pred == 1)
    T = (gt == 1) & (pred == 1)
    pos = H | T
    yH = H[pos].astype(int)
    layers = np.arange(L)
    sel = np.array([rank_auc(-delta[pos, l], yH) for l in layers])
    frac = np.array([(delta[H, l] > 0).mean() for l in layers])
    r2 = np.array([
        np.corrcoef(delta[pos, l], yH)[0, 1] ** 2 if np.std(delta[pos, l]) > 0 else np.nan
        for l in layers
    ])
    mean_H = np.array([delta[H, l].mean() for l in layers])
    return {
        "layers": layers, "sel": sel, "frac": frac, "r2": r2, "mean_H": mean_H,
        "n_H": int(H.sum()), "n_T": int(T.sum()),
    }


def plot_battery(results, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (a1, a2), (a3, a4) = axes
    for m in METHOD_ORDER:
        if m not in results:
            continue
        r = results[m]
        c = COLORS[m]
        a1.plot(r["layers"], r["sel"], "-o", ms=3, color=c, label=m.upper())
        a2.plot(r["layers"], r["frac"], "-o", ms=3, color=c, label=m.upper())
        a3.plot(r["layers"], r["r2"], "-o", ms=3, color=c, label=m.upper())
        a4.plot(r["layers"], r["mean_H"], "-o", ms=3, color=c, label=m.upper())
    a1.axhline(0.5, color="0.6", ls=":"); a1.set_ylim(0.3, 1.0)
    a1.set_title("1. Selectivity AUC of $-\\Delta_l$ (H vs T)  - chance = blind")
    a1.set_ylabel("AUC"); a1.legend(fontsize=9)
    a2.axhline(0.5, color="0.6", ls=":"); a2.set_ylim(0, 1.04)
    a2.set_title("2. Fraction of H with $\\Delta_l>0$  - 1 = one-sided (reinforces Yes)")
    a2.set_ylabel("fraction"); a2.legend(fontsize=9)
    a3.set_ylim(0, max(0.05, max(np.nanmax(results[m]["r2"]) for m in results) * 1.15))
    a3.set_title("3. $R^2$ of $\\Delta_l$ by hallucination status  - ~0 = uninformative")
    a3.set_ylabel("$R^2$"); a3.set_xlabel("layer"); a3.legend(fontsize=9)
    a4.axhline(0.0, color="0.6", ls=":")
    a4.set_title("4. Mean shift $\\overline{\\Delta}_l$ on H (log-odds)  - >0 = toward Yes")
    a4.set_ylabel("log-odds"); a4.set_xlabel("layer"); a4.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("Cross-method: every CD variant's shift is blind and one-sided at all depths",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=150)
    print("wrote", out)


def plot_alignment(results_dir, out):
    path = os.path.join(results_dir, "subspace_align.npz")
    if not os.path.exists(path):
        return
    z = np.load(path)
    methods = sorted({k.split("_")[0] for k in z.files})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for m in methods:
        if f"{m}_cos_H" not in z:
            continue
        c = COLORS.get(m, None)
        L = len(z[f"{m}_cos_H"]); x = np.arange(L)
        a1.plot(x, z[f"{m}_cos_H"], "-o", ms=3, color=c, label=m.upper())
        a2.plot(x, z[f"{m}_dh_H"], "-o", ms=3, color=c, label=m.upper())
    a1.axhline(0.0, color="0.6", ls=":"); a1.set_ylim(-1, 1)
    a1.set_title("cos(perturbation, hallucination direction) on H\n~0 => orthogonal (wrong subspace)")
    a1.set_ylabel("cosine"); a1.set_xlabel("layer"); a1.legend(fontsize=9); a1.grid(alpha=0.25)
    a2.set_title("perturbation magnitude $\\|\\Delta h_l\\|$ on H\n(large while cosine ~ 0)")
    a2.set_ylabel("L2 norm"); a2.set_xlabel("layer"); a2.legend(fontsize=9); a2.grid(alpha=0.25)
    fig.suptitle("Subspace alignment: CD moves the residual a lot, but not toward the fix",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def plot_patch(results_dir, out):
    paths = sorted(glob.glob(os.path.join(results_dir, "patch_effect_*.npz")))
    if not paths:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for p in paths:
        m = os.path.basename(p)[len("patch_effect_"):-len(".npz")]
        z = np.load(p); c = COLORS.get(m, None)
        a1.plot(z["layer"], z["effect_mean_H"], "-o", ms=3, color=c, label=m.upper())
        a2.plot(z["layer"], z["flip_frac_H"], "-o", ms=3, color=c, label=m.upper())
    a1.axhline(0.0, color="0.6", ls=":")
    a1.set_title("Causal patching: effect of injecting amateur state at block b\n(<0 = suppresses Yes)")
    a1.set_ylabel("$\\Delta$ margin (log-odds)"); a1.set_xlabel("block"); a1.legend(fontsize=9)
    a2.set_title("Fraction of H causally flipped to No by the patch")
    a2.set_ylabel("flip fraction"); a2.set_xlabel("block"); a2.legend(fontsize=9)
    for ax in (a1, a2):
        ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print("wrote", out)


def summarize(results, results_dir):
    rows = []
    for m in METHOD_ORDER:
        if m not in results:
            continue
        r = results[m]
        rows.append({
            "method": m, "n_H": r["n_H"], "n_T": r["n_T"],
            "selectivity_readout": float(r["sel"][-1]),
            "selectivity_min": float(np.nanmin(r["sel"])),
            "selectivity_max": float(np.nanmax(r["sel"])),
            "frac_pos_readout": float(r["frac"][-1]),
            "r2_max": float(np.nanmax(r["r2"])),
            "mean_delta_H_readout": float(r["mean_H"][-1]),
        })
    with open(os.path.join(results_dir, "summary_all.json"), "w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        keys = list(rows[0].keys())
        with open(os.path.join(results_dir, "summary_all.csv"), "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
    for r in rows:
        print(f"[{r['method']}] sel_readout={r['selectivity_readout']:.3f} "
              f"frac={r['frac_pos_readout']:.3f} R2max={r['r2_max']:.4f} "
              f"meanD_H={r['mean_delta_H_readout']:+.2f}  (H={r['n_H']} T={r['n_T']})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()

    results = {}
    for p in sorted(glob.glob(os.path.join(args.results_dir, "logit_lens_per_sample_*.npz"))):
        m = os.path.basename(p)[len("logit_lens_per_sample_"):-len(".npz")]
        results[m] = battery(p)

    if results:
        plot_battery(results, os.path.join(args.results_dir, "cross_method_battery.png"))
        summarize(results, args.results_dir)
    plot_alignment(args.results_dir, os.path.join(args.results_dir, "subspace_alignment.png"))
    plot_patch(args.results_dir, os.path.join(args.results_dir, "patch_causal.png"))
    print("[analyze] done")


if __name__ == "__main__":
    main()
