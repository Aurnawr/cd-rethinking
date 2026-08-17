"""
plot_d_per_step.py -- per-sample contrastive adjustment d = E - A bar plots,
matching the paper's Figure "Per-time step contrastive adjustment d = E - A
over the expert's top-10 object candidates" (main_resolved.tex,
ssec:contrastive-adjustment-results). This is NOT a histogram of d values --
each sample gets its own thin vertical bar at its index, colored red if
d > 0 (amplified) or blue if d < 0 (suppressed), with a dashed line at the
group mean and a stats box (n, mean, %pos, %neg).

One figure per model: an (n_methods x 2) grid, rows = method (vcd, sid, and
now icd -- pooled over the 5 disturbance-prompt passes, extending the
paper's original VCD/SID-only grid), columns = (Real Objects, Hallucinated
Objects). Population is the expert's top-10 object candidates (matching the
paper's "top10" figure, not "emitted").

Usage:
  python plot_d_per_step.py --model llava --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0 --outdir CHAIR_ANALYSIS_ICDSAMPLING
"""
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
CAPTURES = REPO / "outputs" / "captures"
FIGURES = REPO / "figures"
MODEL_DIR = {"llava": "llava-v1.5-7b", "qwen": "Qwen2.5-VL-7B-Instruct"}
MODEL_TITLE = {"llava": "LLaVA-1.5-7B", "qwen": "Qwen2.5-VL-7B"}
METHOD_TITLE = {"vcd": "VCD", "sid": "SID", "icd": "ICD"}


def group_name(method):
    return "icd" if method.startswith("icd_") else method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llava", "qwen"], required=True)
    ap.add_argument("--methods", nargs="+",
                     default=["vcd", "sid", "icd_p1", "icd_p2", "icd_n1", "icd_n2", "icd_p3"])
    ap.add_argument("--method-order", nargs="+", default=["vcd", "sid", "icd"],
                     help="row order in the grid")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--outdir", default=str(FIGURES))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from transformers import AutoTokenizer
    import object_mentions as chair_mod
    from eval.chair import singularize
    from contrastive_analysis import d_stats

    slow = args.model == "llava"
    tok = AutoTokenizer.from_pretrained(str(REPO / "models" / MODEL_DIR[args.model]), use_fast=not slow)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        groups = {}
        for method in args.methods:
            f = CAPTURES / f"{args.model}_{method}_seed{seed}.jsonl"
            if not f.exists():
                print(f"[skip] {f} not found")
                continue
            pops = d_stats(f, tok, chair_mod, singularize)
            g = groups.setdefault(group_name(method), {"real": [], "hall": []})
            g["real"].extend(pops["top10"]["real"])
            g["hall"].extend(pops["top10"]["hall"])

        rows = [m for m in args.method_order if m in groups]
        if not rows:
            print(f"[skip] no data for seed={seed}")
            continue

        fig, axes = plt.subplots(len(rows), 2, figsize=(16, 4.2 * len(rows)))
        if len(rows) == 1:
            axes = axes.reshape(1, 2)

        for r, method in enumerate(rows):
            for c, (col_label, tag) in enumerate((("Real Objects", "real"), ("Hallucinated Objects", "hall"))):
                ax = axes[r][c]
                vals = np.array(groups[method][tag])
                n = len(vals)
                idx = np.arange(n)
                mean = vals.mean() if n else 0.0
                pos_frac = (vals > 0).mean() * 100 if n else 0.0
                neg_frac = 100 - pos_frac if n else 0.0

                colors = np.where(vals > 0, "red", "blue")
                ax.vlines(idx, 0, vals, colors=colors, linewidth=0.4)
                ax.axhline(mean, color="black", linewidth=1.0, linestyle="--")

                ax.set_title(f"{MODEL_TITLE[args.model]}  {METHOD_TITLE[method]}  top-10\n{col_label}",
                             fontweight="bold")
                ax.set_xlabel("index (n)")
                ax.set_ylabel("d = expert - amateur logit")
                ax.text(0.98, 0.95, f"n = {n}\nmean = {mean:+.3f}\n%pos = {pos_frac:.0f}%   %neg = {neg_frac:.0f}%",
                        transform=ax.transAxes, ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="0.92", edgecolor="0.6"))

        fig.tight_layout()
        out_path = outdir / f"{args.model}_top10_seed{seed}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot] {out_path}")


if __name__ == "__main__":
    main()
