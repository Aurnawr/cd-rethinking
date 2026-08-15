#!/usr/bin/env python3
"""Cross-method analysis for the Qwen2.5-VL CD battery.

Consumes every `logit_lens_per_sample_<method>.npz` under --results-dir and answers one question
per panel, for all three CD variants on the same axes.

Delta_l = clean_margin_l - amateur_margin_l is each method's per-unit-alpha shift toward "Yes".
On the model-says-Yes population (H = absent, T = present) we test whether Delta carries
hallucination-specific information at all:

  selectivity AUC of -Delta   ~0.5  => blind: the shift cannot tell H from T
  fraction of H with Delta>0  ~1    => one-sided: it reinforces Yes on the very samples it
                                       should suppress
  R^2 by hallucination status ~0    => essentially none of the shift is "about" hallucination
  mean Delta on H             >0    => the sign of the anti-correction

The thesis generalizes across CD methods iff all three variants look the same on all four panels.

Writes cross_method_battery.png + summary_all.{json,csv} to --results-dir. Port of
`mech_interp/analyze_all.py`; the subspace-alignment and activation-patching panels are dropped
because this branch does not run the causal battery.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
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
    if H.sum() == 0 or T.sum() == 0:
        print(f"[warn] {os.path.basename(npz_path)}: H={int(H.sum())} T={int(T.sum())} "
              f"-- selectivity is undefined with an empty class")
    yH = H[pos].astype(int)
    layers = np.arange(L)
    return {
        "layers": layers,
        "sel": np.array([rank_auc(-delta[pos, l], yH) for l in layers]),
        "frac": np.array([(delta[H, l] > 0).mean() if H.sum() else np.nan for l in layers]),
        "r2": np.array([np.corrcoef(delta[pos, l], yH)[0, 1] ** 2
                        if np.std(delta[pos, l]) > 0 else np.nan for l in layers]),
        "mean_H": np.array([delta[H, l].mean() if H.sum() else np.nan for l in layers]),
        "n_H": int(H.sum()), "n_T": int(T.sum()),
    }


def plot_battery(results, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (a1, a2), (a3, a4) = axes
    n_states = max(len(r["layers"]) for r in results.values())
    for m in METHOD_ORDER:
        if m not in results:
            continue
        r, c = results[m], COLORS[m]
        a1.plot(r["layers"], r["sel"], "-o", ms=3, color=c, label=m.upper())
        a2.plot(r["layers"], r["frac"], "-o", ms=3, color=c, label=m.upper())
        a3.plot(r["layers"], r["r2"], "-o", ms=3, color=c, label=m.upper())
        a4.plot(r["layers"], r["mean_H"], "-o", ms=3, color=c, label=m.upper())
    a1.axhline(0.5, color="0.6", ls=":")
    a1.set_ylim(0.3, 1.0)
    a1.set_title("1. Selectivity AUC of $-\\Delta_l$ (H vs T)  - chance = blind")
    a1.set_ylabel("AUC")
    a1.legend(fontsize=9)
    a2.axhline(0.5, color="0.6", ls=":")
    a2.set_ylim(0, 1.04)
    a2.set_title("2. Fraction of H with $\\Delta_l>0$  - 1 = one-sided (reinforces Yes)")
    a2.set_ylabel("fraction")
    a2.legend(fontsize=9)
    r2_max = max((np.nanmax(r["r2"]) for r in results.values() if np.isfinite(np.nanmax(r["r2"]))),
                 default=0.05)
    a3.set_ylim(0, max(0.05, r2_max * 1.15))
    a3.set_title("3. $R^2$ of $\\Delta_l$ by hallucination status  - ~0 = uninformative")
    a3.set_ylabel("$R^2$")
    a3.set_xlabel(f"layer (0 = embeddings, 1..{n_states - 1} = blocks)")
    a3.legend(fontsize=9)
    a4.axhline(0.0, color="0.6", ls=":")
    a4.set_title("4. Mean shift $\\overline{\\Delta}_l$ on H (log-odds)  - >0 = toward Yes")
    a4.set_ylabel("log-odds")
    a4.set_xlabel(f"layer (0 = embeddings, 1..{n_states - 1} = blocks)")
    a4.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("Qwen2.5-VL cross-method: every CD variant's shift is blind and one-sided at all depths",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=150)
    plt.close(fig)
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
    if not results:
        raise SystemExit(f"no logit_lens_per_sample_*.npz under {args.results_dir}")

    plot_battery(results, os.path.join(args.results_dir, "cross_method_battery.png"))
    summarize(results, args.results_dir)
    print("[analyze] done")


if __name__ == "__main__":
    main()
