#!/usr/bin/env python3
"""Scalar-bias experiment: is CD's POPE effect reproducible by one constant?

On a binary decision the whole contrastive term reduces to one number per sample, the shift
Delta added to the Yes-No margin (m_CD = m_E + beta * Delta). If that shift carried
hallucination-specific information, replacing it with a *single constant* shared by all 9,000
questions would have to cost accuracy. This sweeps

    decision(b) = [ m_E_readout + b > 0 ]

over b and compares the resulting POPE accuracy against what CD actually realizes,
[ m_E + beta * Delta > 0 ], on the identical questions and the identical readout.

Four reference points are reported:

  expert   b = 0, the model's own greedy decision
  CD       the realized per-sample contrastive decision (no APC; see the caveat below)
  b_equiv  b = mean Delta over all questions, i.e. CD's shift flattened to its mean
  b*       the accuracy-maximizing constant

Caveat worth stating in any writeup: this isolates the contrastive term. A deployed decoder
also applies the adaptive plausibility constraint, which is amateur-free and known to account
for most of the reported gain. The point here is precisely that what is *left* after removing
APC is a translation, and a translation is fully described by one scalar.

    python mech_interp_calib/scalar_bias.py \
        --npz results/qwen2.5-vl-7b/logit_lens_per_sample_vcd.npz \
        --model-tag qwen2.5-vl-7b --method vcd --out-dir results/calibration
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from calib_common import READOUT, Lens, model_label, pope_scores  # noqa: E402


def sweep(margin, gt, b_grid):
    return [dict(b=float(b), **pope_scores(margin + b > 0, gt)) for b in b_grid]


def reference_points(lens, beta):
    m = lens.clean[:, READOUT]
    d = lens.delta[:, READOUT]
    return {
        "expert": {"b": 0.0, **pope_scores(m > 0, lens.gt)},
        "cd_realized": {"beta": beta, **pope_scores(m + beta * d > 0, lens.gt)},
        "greedy_recorded": pope_scores(lens.pred == 1, lens.gt),
    }


def match_bias(rows, target_acc):
    """Smallest |b| whose accuracy reaches `target_acc`, or None if the sweep never does."""
    hits = [r for r in rows if r["accuracy"] >= target_acc]
    return min(hits, key=lambda r: abs(r["b"])) if hits else None


def run(lens, model_tag, method, beta, b_max, b_step):
    m = lens.clean[:, READOUT]
    d = lens.delta[:, READOUT]
    b_grid = np.arange(-b_max, b_max + 0.5 * b_step, b_step)
    rows = sweep(m, lens.gt, b_grid)

    refs = reference_points(lens, beta)
    best = max(rows, key=lambda r: r["accuracy"])
    b_equiv = float(beta * d.mean())
    equiv = dict(b=b_equiv, **pope_scores(m + b_equiv > 0, lens.gt))
    matched = match_bias(rows, refs["cd_realized"]["accuracy"])

    per_split = {}
    for s in sorted(set(lens.split.tolist())):
        sel = lens.split == s
        srows = sweep(m[sel], lens.gt[sel], b_grid)
        sbest = max(srows, key=lambda r: r["accuracy"])
        per_split[s] = {
            "n": int(sel.sum()),
            "expert": {"b": 0.0, **pope_scores(m[sel] > 0, lens.gt[sel])},
            "cd_realized": pope_scores(m[sel] + beta * d[sel] > 0, lens.gt[sel]),
            "best": sbest,
            "b_equiv": dict(b=b_equiv, **pope_scores(m[sel] + b_equiv > 0, lens.gt[sel])),
        }

    return {
        "model_tag": model_tag,
        "model_label": model_label(model_tag),
        "method": method,
        "beta": beta,
        "n": int(lens.n),
        "n_H": lens.n_H,
        "n_T": lens.n_T,
        "expert": refs["expert"],
        "greedy_recorded": refs["greedy_recorded"],
        "cd_realized": refs["cd_realized"],
        "b_equiv": equiv,
        "best": best,
        "b_matching_cd": matched,
        "gain_cd_over_expert": refs["cd_realized"]["accuracy"] - refs["expert"]["accuracy"],
        "gain_best_over_expert": best["accuracy"] - refs["expert"]["accuracy"],
        "gain_equiv_over_expert": equiv["accuracy"] - refs["expert"]["accuracy"],
        "mean_delta_all": float(d.mean()),
        "mean_delta_H": float(d[lens.H].mean()),
        "mean_delta_T": float(d[lens.T].mean()),
        "per_split": per_split,
        "rows": rows,
    }


def plot(res, out_png):
    rows = res["rows"]
    b = np.array([r["b"] for r in rows])
    acc = 100 * np.array([r["accuracy"] for r in rows])
    exp_acc = 100 * res["expert"]["accuracy"]
    cd_acc = 100 * res["cd_realized"]["accuracy"]

    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.plot(b, acc, color="0.15", lw=2.6, label="expert margin + constant bias $b$")
    ax.set_ylim(acc.min() - 1.5, max(acc.max(), cd_acc, exp_acc) + 2.5)
    ax.axvline(0, color="0.7", ls=":", lw=1.2)
    # Reference points go in the legend rather than inline: b* and b_equiv can land arbitrarily
    # close together (Qwen) or far apart (LLaVA), and inline labels collide in the first case.
    ax.axhline(cd_acc, color="#d4604a", ls="--", lw=2.0,
               label=f"{res['method'].upper()} realized: {cd_acc:.2f}%")
    ax.plot([0], [exp_acc], "o", ms=12, color="#1f8a4c", zorder=5,
            label=f"expert ($b$=0): {exp_acc:.2f}%")
    ax.plot([res["best"]["b"]], [100 * res["best"]["accuracy"]], "*", ms=19, color="0.15",
            zorder=5, label=f"best scalar $b^*$={res['best']['b']:+.2f}: "
                            f"{100 * res['best']['accuracy']:.2f}%")
    ax.plot([res["b_equiv"]["b"]], [100 * res["b_equiv"]["accuracy"]], "D", ms=11,
            color="#d4604a", zorder=5,
            label=f"$b_{{\\rm equiv}}$ = mean $\\Delta$ = {res['b_equiv']['b']:+.2f}: "
                  f"{100 * res['b_equiv']['accuracy']:.2f}%")
    ax.legend(fontsize=10.5, loc="lower center", framealpha=0.95)

    ax.set_xlabel("Constant bias $b$ added to every Yes$-$No margin (log-odds)", fontsize=12)
    ax.set_ylabel("POPE accuracy (%)", fontsize=12)
    verdict = ("matches/exceeds" if res["best"]["accuracy"] >= res["cd_realized"]["accuracy"]
               else "falls short of")
    ax.set_title(f"A single scalar threshold shift {verdict} {res['method'].upper()}\n"
                 f"{res['model_label']}", fontsize=13)
    ax.grid(alpha=0.2)
    fig.text(0.5, 0.008,
             f"{res['model_label']}, POPE-COCO pooled ({res['n']:,} Q), "
             f"beta = {res['beta']:g}. Higher = better.",
             ha="center", fontsize=10, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("wrote", out_png)


def write_csv(res, path):
    keys = ["b", "accuracy", "f1", "precision", "recall", "yes_rate"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in res["rows"]:
            w.writerow({k: r[k] for k in keys})
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--b-max", type=float, default=6.0)
    ap.add_argument("--b-step", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lens = Lens(args.npz)
    res = run(lens, args.model_tag, args.method, args.beta, args.b_max, args.b_step)

    stem = f"{args.model_tag}_{args.method}"
    plot(res, os.path.join(args.out_dir, f"fig_scalar_bias_sweep_{stem}.png"))
    write_csv(res, os.path.join(args.out_dir, f"scalar_bias_curve_{stem}.csv"))
    payload = {k: v for k, v in res.items() if k != "rows"}
    with open(os.path.join(args.out_dir, f"scalar_bias_{stem}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[bias {stem}] expert={100 * res['expert']['accuracy']:.2f}% "
          f"cd={100 * res['cd_realized']['accuracy']:.2f}% "
          f"b*={res['best']['b']:+.2f} -> {100 * res['best']['accuracy']:.2f}% "
          f"b_equiv={res['b_equiv']['b']:+.2f} -> {100 * res['b_equiv']['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
