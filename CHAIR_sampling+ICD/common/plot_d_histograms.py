"""
plot_d_histograms.py -- simple matplotlib histograms of d = E - A for
hallucinated vs. real objects, direct-sampling edition.

For every decoding step, looks at the expert's top-10 candidate tokens; for
each candidate that is an MSCOCO object word, computes d = E - A and buckets
it as "hallucinated" or "real" against that image's ground-truth objects.
One histogram per method x seed (two-branch methods only: vcd, sid, icd_*).

Deliberately plain: default matplotlib style, default colors, simple axis
labels, no smoothing/KDE -- a plot a person would make by hand, not a
styled dashboard figure.

Usage:
  python plot_d_histograms.py --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0 1
"""
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
CAPTURES = REPO / "outputs" / "captures"
FIGURES = REPO / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                     default=["vcd", "sid", "icd_p1", "icd_p2", "icd_n1", "icd_n2", "icd_p3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--population", choices=["top10", "top30", "emitted"], default="top10")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    import object_mentions as chair_mod
    from eval.chair import singularize
    from contrastive_analysis import d_stats

    tok = AutoTokenizer.from_pretrained(str(REPO / "models" / "llava-v1.5-7b"), use_fast=False)
    FIGURES.mkdir(exist_ok=True)

    for method in args.methods:
        for seed in args.seeds:
            f = CAPTURES / f"llava_{method}_seed{seed}.jsonl"
            if not f.exists():
                print(f"[skip] {f} not found")
                continue
            pops = d_stats(f, tok, chair_mod, singularize)
            real = pops[args.population]["real"]
            hall = pops[args.population]["hall"]

            fig, ax = plt.subplots(figsize=(6, 4))
            bins = 60
            ax.hist(real, bins=bins, alpha=0.6, label=f"real (n={len(real)})", color="tab:blue")
            ax.hist(hall, bins=bins, alpha=0.6, label=f"hallucinated (n={len(hall)})", color="tab:red")
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("d = E - A")
            ax.set_ylabel("count")
            ax.set_title(f"{method}  seed={seed}  ({args.population} object candidates)")
            ax.legend()
            fig.tight_layout()

            out_path = FIGURES / f"d_hist_{method}_seed{seed}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[plot] {out_path}  (real n={len(real)}, hall n={len(hall)})")


if __name__ == "__main__":
    main()
