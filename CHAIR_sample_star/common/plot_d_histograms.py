"""
plot_d_histograms.py -- simple matplotlib histograms of d = E - A for
hallucinated vs. real objects, direct-sampling edition.

For every decoding step, looks at the expert's top-10 candidate tokens; for
each candidate that is an MSCOCO object word, computes d = E - A and buckets
it as "hallucinated" or "real" against that image's ground-truth objects.
TWO separate histograms per method-group x seed -- one for real objects, one
for hallucinated objects (not overlaid on one axes) -- for two-branch
methods only (vcd, sid, icd). The 5 ICD disturbance-prompt runs (icd_p1,
icd_p2, icd_n1, icd_n2, icd_p3) are POOLED into a single "icd" distribution
rather than 5 separate ones -- they're 5 passes of the same method, not 5
different methods, so one combined d-distribution is the natural summary.

2 models x 3 method-groups (vcd, sid, icd) x 2 (real, hallucinated) = 12
plots for the full LLaVA+Qwen sweep.

Deliberately plain: default matplotlib style, default colors, simple axis
labels, no smoothing/KDE -- a plot a person would make by hand, not a
styled dashboard figure.

Usage:
  python plot_d_histograms.py --model llava --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
"""
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
CAPTURES = REPO / "outputs" / "captures"
FIGURES = REPO / "figures"
MODEL_DIR = {"llava": "llava-v1.5-7b", "qwen": "Qwen2.5-VL-7B-Instruct"}


def group_name(method):
    """icd_p1/icd_p2/icd_n1/icd_n2/icd_p3 all pool into "icd"; everything
    else (vcd, sid) is its own group."""
    return "icd" if method.startswith("icd_") else method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llava", "qwen"], required=True)
    ap.add_argument("--methods", nargs="+",
                     default=["vcd", "sid", "icd_p1", "icd_p2", "icd_n1", "icd_n2", "icd_p3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--population", choices=["top10", "top30", "emitted"], default="top10")
    ap.add_argument("--outdir", default=str(FIGURES))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    import object_mentions as chair_mod
    from eval.chair import singularize
    from contrastive_analysis import d_stats

    slow = args.model == "llava"
    tok = AutoTokenizer.from_pretrained(str(REPO / "models" / MODEL_DIR[args.model]), use_fast=not slow)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        # group -> {"real": [...], "hall": [...], "members": [method, ...]}
        groups = {}
        for method in args.methods:
            f = CAPTURES / f"{args.model}_{method}_seed{seed}.jsonl"
            if not f.exists():
                print(f"[skip] {f} not found")
                continue
            pops = d_stats(f, tok, chair_mod, singularize)
            g = groups.setdefault(group_name(method), {"real": [], "hall": [], "members": []})
            g["real"].extend(pops[args.population]["real"])
            g["hall"].extend(pops[args.population]["hall"])
            g["members"].append(method)

        for gname, g in groups.items():
            pooled_note = f" (pooled over {len(g['members'])} prompts: {', '.join(g['members'])})" if len(g["members"]) > 1 else ""

            for label, color, tag in (("real", "tab:blue", "real"), ("hallucinated", "tab:red", "hall")):
                vals = g[tag]

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(vals, bins=60, color=color, label=f"{label} (n={len(vals)})")
                ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
                ax.set_xlabel("d = E - A")
                ax.set_ylabel("count")
                ax.set_title(f"{args.model}  {gname}  {label}  seed={seed}  "
                             f"({args.population} object candidates){pooled_note}")
                ax.legend()
                fig.tight_layout()

                out_path = outdir / f"d_hist_{args.model}_{gname}_{tag}_seed{seed}.png"
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"[plot] {out_path}  (n={len(vals)}){pooled_note}")


if __name__ == "__main__":
    main()
