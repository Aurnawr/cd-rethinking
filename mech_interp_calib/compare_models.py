#!/usr/bin/env python3
"""Cross-model comparison of the two calibration experiments.

Reads every `oracle_<model>_<method>.json` and `scalar_bias_<model>_<method>.json` produced by
the two experiment scripts and puts LLaVA-1.5-7B and Qwen2.5-VL-7B side by side.

Depth is plotted on a *relative* axis (layer / last layer) because the two stacks differ:
LLaVA-1.5-7B has 33 hidden states, Qwen2.5-VL-7B has 29. Absolute layer indices are not
comparable across models; fractional depth is.

    python mech_interp_calib/compare_models.py --dir results/calibration
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHODS = ["vcd", "icd", "sid"]
METHOD_COLORS = {"vcd": "#1f6fb4", "icd": "#e08214", "sid": "#2b8a3e"}
MODEL_STYLE = {"llava1.5-7b": dict(ls="--", marker="o"),
               "qwen2.5-vl-7b": dict(ls="-", marker="s")}


def load(directory, prefix):
    out = {}
    for p in sorted(glob.glob(os.path.join(directory, f"{prefix}_*.json"))):
        d = json.load(open(p))
        out[(d["model_tag"], d["method"])] = d
    # model first, then the canonical VCD / ICD / SID order rather than alphabetical
    order = sorted(out, key=lambda k: (k[0], METHODS.index(k[1]) if k[1] in METHODS else 99))
    return {k: out[k] for k in order}


def rel_depth(n):
    return np.arange(n) / (n - 1)


def fig_selectivity(oracle, out_png):
    models = sorted({k[0] for k in oracle})
    fig, axes = plt.subplots(1, len(METHODS), figsize=(5 * len(METHODS), 4.6), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), METHODS):
        for model in models:
            d = oracle.get((model, m))
            if not d:
                continue
            sel = np.array(d["sel_by_layer"])
            ax.plot(rel_depth(len(sel)), sel, color=METHOD_COLORS[m], lw=2.0,
                    **MODEL_STYLE.get(model, {}), ms=3.5, label=d["model_label"])
            ax.fill_between(rel_depth(len(sel)), d["sel_ci_lo"], d["sel_ci_hi"],
                            color=METHOD_COLORS[m], alpha=0.13)
        ax.axhline(0.5, color="0.5", ls=":", lw=1.2)
        ax.set_title(m.upper(), fontsize=13)
        ax.set_xlabel("relative depth (layer / last)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel("Selectivity AUC of $-\\Delta_\\ell$")
    fig.suptitle("Selectivity of CD's shift by depth, LLaVA-1.5-7B vs Qwen2.5-VL-7B "
                 "(shaded = 95% bootstrap CI)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("wrote", out_png)


def fig_oracle(oracle, out_png):
    models = sorted({k[0] for k in oracle})
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.8), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        label = None
        for m in METHODS:
            d = oracle.get((model, m))
            if not d:
                continue
            label = d["model_label"]
            ax.plot(d["s_grid"], d["auc_grid"], color=METHOD_COLORS[m], lw=2.2,
                    label=f"{m.upper()} (CD {d['selectivity_cd']:.2f}, "
                          f"$s^*$={d['s_star']:.2f})" if d["s_star"] else m.upper())
            ax.axhline(d["detection_threshold"], color=METHOD_COLORS[m], ls=":", lw=1.0)
            ax.plot([0], [d["selectivity_cd"]], "o", ms=8, color=METHOD_COLORS[m])
        ax.axhline(0.5, color="0.6", ls="--", lw=1.0)
        ax.set_title(label or model, fontsize=13)
        ax.set_xlabel("injected selective shift $s$ (log-odds)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="lower right")
    np.atleast_1d(axes)[0].set_ylabel("Selectivity AUC")
    fig.suptitle("Graded oracle: how much selectivity the test would have caught", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("wrote", out_png)


def fig_scalar_bias(bias, directory, out_png):
    models = sorted({k[0] for k in bias})
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.8))
    for ax, model in zip(np.atleast_1d(axes), models):
        label = None
        for m in METHODS:
            d = bias.get((model, m))
            if not d:
                continue
            label = d["model_label"]
            curve = os.path.join(directory, f"scalar_bias_curve_{model}_{m}.csv")
            if os.path.exists(curve) and m == METHODS[0]:
                rows = list(csv.DictReader(open(curve)))
                ax.plot([float(r["b"]) for r in rows],
                        [100 * float(r["accuracy"]) for r in rows],
                        color="0.15", lw=2.4, label="expert + constant bias $b$")
            ax.axhline(100 * d["cd_realized"]["accuracy"], color=METHOD_COLORS[m], ls="--",
                       lw=1.8, label=f"{m.upper()} realized "
                                     f"{100 * d['cd_realized']['accuracy']:.2f}%")
            ax.plot([d["b_equiv"]["b"]], [100 * d["b_equiv"]["accuracy"]], "D", ms=9,
                    color=METHOD_COLORS[m])
        d0 = bias[(model, METHODS[0])]
        ax.plot([0], [100 * d0["expert"]["accuracy"]], "o", ms=11, color="#1f8a4c",
                label=f"expert {100 * d0['expert']['accuracy']:.2f}%", zorder=5)
        ax.plot([d0["best"]["b"]], [100 * d0["best"]["accuracy"]], "*", ms=17, color="0.15",
                label=f"best $b^*$={d0['best']['b']:+.2f} "
                      f"({100 * d0['best']['accuracy']:.2f}%)", zorder=5)
        ax.set_title(label or model, fontsize=13)
        ax.set_xlabel("constant bias $b$ on the Yes$-$No margin (log-odds)")
        ax.set_ylabel("POPE accuracy (%)")
        ax.set_ylim(45, 95)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8.5, loc="lower center")
    fig.suptitle("One constant on the margin vs the full per-sample contrastive term",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("wrote", out_png)


ORACLE_COLS = [
    ("model_label", "model"), ("method", "method"), ("n_H", "n_H"), ("n_T", "n_T"),
    ("selectivity_cd", "sel AUC"), ("sel_ci", "sel 95% CI"),
    ("sel_resid", "sel AUC resid."), ("cd_above_chance", "above chance"),
    ("oracle_ceiling", "oracle ceiling"), ("cd_mean_abs_delta", "mean abs D"),
    ("s_star", "s*"), ("detection_factor", "mean abs D / s*"),
    ("cd_mean_delta_H", "mean D on H"), ("cd_mean_delta_T", "mean D on T"),
    ("cd_frac_delta_pos_H", "frac D>0 on H"),
]

BIAS_COLS = [
    ("model_label", "model"), ("method", "method"),
    ("expert_acc", "expert acc"), ("cd_acc", "CD acc"), ("cd_gain", "CD - expert"),
    ("b_equiv", "b_equiv"), ("b_equiv_acc", "acc(b_equiv)"),
    ("b_star", "b*"), ("b_star_acc", "acc(b*)"), ("b_star_gain", "b* - expert"),
    ("expert_yes", "expert yes-rate"), ("cd_yes", "CD yes-rate"),
]


def bias_row(d):
    return {
        "model_label": d["model_label"], "method": d["method"],
        "expert_acc": 100 * d["expert"]["accuracy"],
        "cd_acc": 100 * d["cd_realized"]["accuracy"],
        "cd_gain": 100 * d["gain_cd_over_expert"],
        "b_equiv": d["b_equiv"]["b"], "b_equiv_acc": 100 * d["b_equiv"]["accuracy"],
        "b_star": d["best"]["b"], "b_star_acc": 100 * d["best"]["accuracy"],
        "b_star_gain": 100 * d["gain_best_over_expert"],
        "expert_yes": 100 * d["expert"]["yes_rate"],
        "cd_yes": 100 * d["cd_realized"]["yes_rate"],
    }


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def oracle_row(d):
    row = {k: d.get(k) for k, _ in ORACLE_COLS}
    lo, hi = d["selectivity_cd_ci"]
    row["sel_ci"] = f"[{lo:.2f}, {hi:.2f}]"
    row["sel_resid"] = d["sel_resid_by_layer"][-1]
    return row


def write_tables(oracle, bias, directory):
    orows = [oracle_row(d) for d in oracle.values()]
    brows = [bias_row(d) for d in bias.values()]
    for name, rows, cols in (("comparison_oracle", orows, ORACLE_COLS),
                             ("comparison_scalar_bias", brows, BIAS_COLS)):
        path = os.path.join(directory, f"{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[k for k, _ in cols], lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print("wrote", path)

    md = ["# Calibration experiments: LLaVA-1.5-7B vs Qwen2.5-VL-7B", "",
          "All numbers at the readout layer, beta = 1, POPE-COCO pooled over the random, "
          "popular and adversarial splits.", "",
          "## Experiment 3: oracle calibration of the selectivity metric", ""]
    md += markdown_table(orows, ORACLE_COLS)
    md += ["", "### Per-pair verdict", ""]
    md += [f"- **{d['model_label']} / {d['method'].upper()}**: {d['verdict']}"
           for d in oracle.values()]
    md += ["", "## Scalar-bias experiment (POPE-COCO pooled, 9,000 questions)", "",
           "Accuracies in percent. `b_equiv` is CD's own mean shift flattened to a constant; "
           "`b*` is the accuracy-maximising constant.", ""]
    md += markdown_table(brows, BIAS_COLS)
    path = os.path.join(directory, "comparison.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", path)


def markdown_table(rows, cols):
    head = "| " + " | ".join(h for _, h in cols) + " |"
    rule = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(fmt(r.get(k)) for k, _ in cols) + " |" for r in rows]
    return [head, rule] + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    oracle = load(args.dir, "oracle")
    bias = load(args.dir, "scalar_bias")
    if not oracle or not bias:
        raise SystemExit(f"no oracle_*.json / scalar_bias_*.json under {args.dir}")

    fig_selectivity(oracle, os.path.join(args.dir, "fig_compare_selectivity.png"))
    fig_oracle(oracle, os.path.join(args.dir, "fig_compare_oracle.png"))
    fig_scalar_bias(bias, args.dir, os.path.join(args.dir, "fig_compare_scalar_bias.png"))
    write_tables(oracle, bias, args.dir)
    print("[compare] done")


if __name__ == "__main__":
    main()
