"""
agreement_analysis.py -- agreement & overlap, direct-sampling edition.

Reads the per-step capture files produced by generate_llava.py/generate_llava_icd.py
(LLaVA) or qwen/generate_qwen.py/generate_qwen_icd.py (Qwen), both with
--decode sample (outputs/captures/{model}_{method}_seed{N}.jsonl), and
reports, with 95% image-level bootstrap confidence intervals (B configurable),
PER SEED (not pooled across seeds -- steps within a seed's 500 images share
that seed's noise draw / sampling draws, so pooling seeds would be
pseudoreplication):

  (1) Intervention rate: how often the contrastive output equals the expert's
      own top-1 candidate by raw logit (i.e. contrastive decoding + sampling
      landed back on the expert's favorite token), over all steps and split
      by real / hallucinated object steps.

  (2) Top-10 candidate-set overlap: expert vs amateur (E&A) and expert vs
      contrastive (E&C), all steps and split real / hallucinated.

Only applies to two-branch methods (vcd, sid, icd_*) -- baseline has no
amateur/contrastive branch to compare against.

Usage:
  python agreement_analysis.py --model llava --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
  python agreement_analysis.py --model qwen  --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent          # common/
REPO = HERE.parent                              # CHAIR_sampling+ICD/
sys.path.insert(0, str(HERE))
COCO = REPO / "data" / "coco" / "annotations"
CAPTURES = REPO / "outputs" / "captures"
TOPN = 10
MODEL_DIR = {"llava": "llava-v1.5-7b", "qwen": "Qwen2.5-VL-7B-Instruct"}


def setov(a, b):
    return len(set(a[:TOPN]) & set(b[:TOPN]))


def per_image(path, tok, chair_mod):
    recs = [json.loads(l) for l in open(path)]
    chair = chair_mod.load_chair([r["image_id"] for r in recs], str(COCO))
    rows = []
    for r in recs:
        steps = r["steps"]
        chosen = [s["chosen_id"] for s in steps]
        eq = [int(s["chosen_id"] == s["expert_top_ids"][0]) for s in steps]
        label = {}
        for m in chair_mod.all_mentions(chair, tok, r["image_id"], chosen):
            if m["category"] == "object":
                si = m["step_indices"][0]
                if si < len(steps):
                    label.setdefault(si, "hall" if m["hallucinated"] else "real")
        acc = {"n": len(steps), "eq": sum(eq),
               "real_n": 0, "real_eq": 0, "hall_n": 0, "hall_eq": 0,
               "ea_all": 0, "ec_all": 0,
               "ea_real": 0, "ec_real": 0, "n_real": 0,
               "ea_hall": 0, "ec_hall": 0, "n_hall": 0}
        for i, s in enumerate(steps):
            ea = setov(s["expert_top_ids"], s["amateur_top_ids"])
            ec = setov(s["expert_top_ids"], s["cd_top_ids"])
            acc["ea_all"] += ea; acc["ec_all"] += ec
            lab = label.get(i)
            if lab == "real":
                acc["real_n"] += 1; acc["real_eq"] += eq[i]
                acc["ea_real"] += ea; acc["ec_real"] += ec; acc["n_real"] += 1
            elif lab == "hall":
                acc["hall_n"] += 1; acc["hall_eq"] += eq[i]
                acc["ea_hall"] += ea; acc["ec_hall"] += ec; acc["n_hall"] += 1
        rows.append(acc)
    return rows


def boot_ratio(rows, num_key, den_key, B, rng):
    num = np.array([r[num_key] for r in rows], float)
    den = np.array([r[den_key] for r in rows], float)
    n = len(rows)
    pt = num.sum() / den.sum() if den.sum() else float("nan")
    d = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        s = den[idx].sum()
        d[i] = num[idx].sum() / s if s else np.nan
    d = d[~np.isnan(d)]
    return pt, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llava", "qwen"], required=True)
    ap.add_argument("--methods", nargs="+", default=["vcd", "sid", "icd_p1", "icd_p2", "icd_n1", "icd_n2", "icd_p3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import object_mentions as chair_mod
    slow = args.model == "llava"
    tok = AutoTokenizer.from_pretrained(str(REPO / "models" / MODEL_DIR[args.model]), use_fast=not slow)

    print(f"\n===== agreement & overlap (direct-sampling) : {args.model} =====")
    for method in args.methods:
        for seed in args.seeds:
            f = CAPTURES / f"{args.model}_{method}_seed{seed}.jsonl"
            if not f.exists():
                print(f"[skip] {f} not found")
                continue
            rows = per_image(f, tok, chair_mod)

            def pct(nk, dk):
                p, lo, hi = boot_ratio(rows, nk, dk, args.B, np.random.default_rng(0))
                return f"{100*p:.1f} [{100*lo:.1f}, {100*hi:.1f}]"

            def ov(nk, dk):
                p, lo, hi = boot_ratio(rows, nk, dk, args.B, np.random.default_rng(0))
                return f"{p:.2f} [{lo:.2f}, {hi:.2f}]"

            print(f"\n--- {method.upper()} seed={seed} (n={len(rows)} images) ---")
            print("Intervention rate (CD-sampled output == expert top-1), % [95% CI]:")
            print(f"  all steps        : {pct('eq','n')}")
            print(f"  real-object steps: {pct('real_eq','real_n')}")
            print(f"  hall-object steps: {pct('hall_eq','hall_n')}")
            print("Top-10 set overlap (out of 10) [95% CI]:")
            print(f"  E&A  all / real / hall : {ov('ea_all','n')} / {ov('ea_real','n_real')} / {ov('ea_hall','n_hall')}")
            print(f"  E&C  all / real / hall : {ov('ec_all','n')} / {ov('ec_real','n_real')} / {ov('ec_hall','n_hall')}")


if __name__ == "__main__":
    main()
