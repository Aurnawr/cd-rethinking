"""
bucket_ci_paired.py -- PAIRED bootstrap CIs for the branch differences in the
bucket table (B = 10000, image-level resampling).

Why this is needed: Expert / Amateur / Contrastive are measured on the SAME
object steps of the SAME images. Their marginal 95% CIs therefore overlap
heavily even when the branch-to-branch difference is systematic, because most of
the interval width is shared image-sampling variance that cancels in a paired
comparison. Comparing overlapping marginal CIs would be the wrong test.

Here each bootstrap replicate resamples images ONCE and recomputes all three
branch statistics on that same resample, then takes the differences. A CI on the
difference that excludes 0 means the branches genuinely differ.

Reports: Contrastive - Expert, and Amateur - Expert (REAL mass).
Output: bucket_ci_paired.json
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bucket_ci import CFG, BRANCHES, per_image, LLAVA_ROOT, B


def main():
    sys.path.insert(0, str(LLAVA_ROOT))
    import llava_logit_eda.object_mentions as chair_mod
    from eval.chair import singularize
    from transformers import AutoTokenizer

    out = {}
    for model in ["LLaVA", "Qwen"]:
        cfg = CFG[model]
        tok = AutoTokenizer.from_pretrained(str(cfg["tok"]), use_fast=not cfg["slow"])
        for method in ["vcd", "sid"]:
            print(f"processing {model}/{method} ...", flush=True)
            pim = per_image(model, method, tok, chair_mod, singularize)
            E, A, C = pim["Expert"], pim["Amateur"], pim["Contrastive"]
            n = len(E)
            rng = np.random.default_rng(0)

            def stat(arr, idx):
                a = arr[idx]
                return a[:, 0].sum() / a[:, 1].sum()

            base = np.arange(n)
            pt_CE = stat(C, base) - stat(E, base)
            pt_AE = stat(A, base) - stat(E, base)
            d_CE = np.empty(B); d_AE = np.empty(B)
            for i in range(B):
                idx = rng.integers(0, n, n)          # SAME resample for all branches
                e = stat(E, idx)
                d_CE[i] = stat(C, idx) - e
                d_AE[i] = stat(A, idx) - e

            def summ(pt, d):
                lo, hi = np.percentile(d, [2.5, 97.5])
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                return {"diff": float(pt), "lo": float(lo), "hi": float(hi),
                        "excludes_zero": bool(lo > 0 or hi < 0),
                        "p_boot": float(max(p, 1.0 / B))}

            out[f"{model}|{method}"] = {"Contrastive_minus_Expert": summ(pt_CE, d_CE),
                                        "Amateur_minus_Expert": summ(pt_AE, d_AE)}
    json.dump(out, open(HERE / "bucket_ci_paired.json", "w"), indent=2)

    print(f"\n=== PAIRED differences in REAL mass (B={B}, same image resample) ===")
    print(f"{'setting':16} {'comparison':26} {'diff':>9} {'95% CI':>22} {'sig?':>6} {'p':>8}")
    for k, v in out.items():
        for comp, s in v.items():
            print(f"{k:16} {comp:26} {s['diff']:+9.5f} "
                  f"[{s['lo']:+.5f},{s['hi']:+.5f}] {str(s['excludes_zero']):>6} {s['p_boot']:8.4f}")


if __name__ == "__main__":
    main()
