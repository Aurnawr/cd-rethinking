"""
bootstrap_exp23.py -- image-level (cluster) bootstrap for Experiments 2 & 3.

Decoding steps are NOT independent: they are nested within captions, which are
nested within images. A per-step Wilson/z interval assumes i.i.d. Bernoulli
draws and therefore understates uncertainty when some images are systematically
harder (more hallucination-prone AND more agreement-prone) than others.

This script recomputes every Table 1 / Table 2 quantity from the raw top-30
captures, aggregating per IMAGE, then bootstraps by resampling the 500 IMAGES
with replacement (B=10000) and recomputing each statistic as a ratio of sums.
That is the same item-level resampling unit already used elsewhere in the paper.

Stage 1 (per-image aggregation) is cached to per_image_agg.json so the bootstrap
can be re-run instantly.

Outputs: per_image_agg.json, bootstrap_exp23.json
"""
import json, sys, argparse
from pathlib import Path
import numpy as np

REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
LLAVA_ROOT = REPO / "llava_logit_eda" / "CHAIR-llava-detailed"
HERE = Path(__file__).resolve().parent

CFG = {
    "LLaVA": {
        "root": LLAVA_ROOT,
        "topk": LLAVA_ROOT / "llava_logit_eda" / "outputs_full_topk",
        "tok": LLAVA_ROOT / "llava_logit_eda" / "models" / "llava-v1.5-7b",
        "slow": True,
    },
    "Qwen": {
        "root": LLAVA_ROOT,  # only used for eval.chair + coco annotations
        "topk": REPO / "CHAIR-qwen-detailed" / "outputs_full_topk",
        "tok": None,  # resolved below
        "slow": False,
    },
}
COCO = LLAVA_ROOT / "data" / "coco" / "annotations"
TOPN = 10  # top-10 overlap, matching Table 2


def resolve_qwen_tok():
    for c in [REPO / "CHAIR-qwen-detailed" / "models" / "Qwen2.5-VL-7B-Instruct",
              LLAVA_ROOT / "llava_logit_eda" / "models" / "Qwen2.5-VL-7B-Instruct"]:
        if c.exists():
            return c
    raise SystemExit("Qwen tokenizer not found")


def overlap_stats(e_ids, x_ids):
    """set overlap (order-insensitive) and exact-position match, over top-N."""
    e, x = e_ids[:TOPN], x_ids[:TOPN]
    return len(set(e) & set(x)), sum(1 for a, b in zip(e, x) if a == b)


def aggregate(model):
    """Per-image aggregates for both methods of one model."""
    sys.path.insert(0, str(LLAVA_ROOT))
    from llava_logit_eda.object_mentions import all_mentions, load_chair
    from transformers import AutoTokenizer

    cfg = CFG[model]
    tok_path = cfg["tok"] or resolve_qwen_tok()
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=not cfg["slow"])

    out = {}
    for method in ["vcd", "sid"]:
        path = cfg["topk"] / f"captions_topk_{method}.jsonl"
        recs = [json.loads(l) for l in open(path)]
        image_ids = [r["image_id"] for r in recs]
        chair = load_chair(image_ids, str(COCO))

        rows = []
        for i, r in enumerate(recs):
            steps = r["steps"]
            chosen = [s["chosen_id"] for s in steps]
            # --- Exp 2: CD output == expert's own greedy top-1, per step ---
            eq = [int(s["chosen_id"] == s["expert_top_ids"][0]) for s in steps]
            # --- object attribution at the CAPTION level (never per-token) ---
            ms = [m for m in all_mentions(chair, tokenizer, r["image_id"], chosen)
                  if m["category"] == "object"]
            real_n = real_eq = hall_n = hall_eq = 0
            for m in ms:
                si = m["step_indices"][0]          # step emitting the mention's first token
                if si >= len(steps):
                    continue
                if m["hallucinated"]:
                    hall_n += 1; hall_eq += eq[si]
                else:
                    real_n += 1; real_eq += eq[si]
            # --- Exp 3: top-10 overlap, every step ---
            ea = ea_pos = ec = ec_pos = 0
            id_ec_ord = id_ec_set = id_ea_set = id_ea_ord = 0
            for s in steps:
                E, A, C = s["expert_top_ids"][:TOPN], s["amateur_top_ids"][:TOPN], s["cd_top_ids"][:TOPN]
                a, b = overlap_stats(E, A); ea += a; ea_pos += b
                c, d = overlap_stats(E, C); ec += c; ec_pos += d
                id_ec_ord += int(E == C)
                id_ec_set += int(set(E) == set(C))
                id_ea_ord += int(E == A)
                id_ea_set += int(set(E) == set(A))
            rows.append({
                "image_id": r["image_id"], "n_steps": len(steps), "eq": sum(eq),
                "real_n": real_n, "real_eq": real_eq, "hall_n": hall_n, "hall_eq": hall_eq,
                "ea": ea, "ea_pos": ea_pos, "ec": ec, "ec_pos": ec_pos,
                "id_ec_ord": id_ec_ord, "id_ec_set": id_ec_set,
                "id_ea_ord": id_ea_ord, "id_ea_set": id_ea_set,
            })
            if (i + 1) % 100 == 0:
                print(f"  {model}/{method}: {i+1}/{len(recs)}", flush=True)
        out[method] = rows
        print(f"  {model}/{method}: {len(rows)} images aggregated", flush=True)
    return out


def ratio(num, den):
    d = den.sum()
    return float(num.sum() / d) if d > 0 else float("nan")


def boot(rows, B=10000, seed=0):
    """Cluster bootstrap: resample IMAGES with replacement; each statistic is a
    ratio of sums, so it is recomputed from the resampled totals."""
    rng = np.random.default_rng(seed)
    A = {k: np.array([r[k] for r in rows], dtype=float)
         for k in ["n_steps", "eq", "real_n", "real_eq", "hall_n", "hall_eq",
                   "ea", "ea_pos", "ec", "ec_pos",
                   "id_ec_ord", "id_ec_set", "id_ea_ord", "id_ea_set"]}
    n = len(rows)

    def stats(idx):
        g = {k: v[idx] for k, v in A.items()}
        real = ratio(g["real_eq"], g["real_n"])
        hall = ratio(g["hall_eq"], g["hall_n"])
        return {
            "all": ratio(g["eq"], g["n_steps"]),
            "real": real, "hall": hall, "gap": real - hall,
            "ea": ratio(g["ea"], g["n_steps"]), "ea_pos": ratio(g["ea_pos"], g["n_steps"]),
            "ec": ratio(g["ec"], g["n_steps"]), "ec_pos": ratio(g["ec_pos"], g["n_steps"]),
            "id_ec_ord": ratio(g["id_ec_ord"], g["n_steps"]),
            "id_ec_set": ratio(g["id_ec_set"], g["n_steps"]),
            "id_ea_ord": ratio(g["id_ea_ord"], g["n_steps"]),
            "id_ea_set": ratio(g["id_ea_set"], g["n_steps"]),
        }

    point = stats(np.arange(n))
    draws = {k: np.empty(B) for k in point}
    for b in range(B):
        idx = rng.integers(0, n, n)
        s = stats(idx)
        for k in point:
            draws[k][b] = s[k]

    res = {}
    for k, v in point.items():
        d = draws[k][~np.isnan(draws[k])]
        res[k] = {"point": v, "lo": float(np.percentile(d, 2.5)),
                  "hi": float(np.percentile(d, 97.5)), "se": float(d.std(ddof=1))}
    # bootstrap p-value for gap > 0 (two-sided), by proportion of draws <= 0
    g = draws["gap"][~np.isnan(draws["gap"])]
    p = 2 * min((g <= 0).mean(), (g >= 0).mean())
    res["gap"]["p_boot"] = float(max(p, 1.0 / len(g)))
    res["gap"]["p_is_bound"] = bool((g <= 0).sum() == 0 or (g >= 0).sum() == 0)
    res["n_images"] = n
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--skip-agg", action="store_true")
    args = ap.parse_args()

    agg_path = HERE / "per_image_agg.json"
    if args.skip_agg and agg_path.exists():
        agg = json.load(open(agg_path))
    else:
        agg = {}
        for model in ["LLaVA", "Qwen"]:
            print(f"aggregating {model}...", flush=True)
            agg[model] = aggregate(model)
        json.dump(agg, open(agg_path, "w"))
        print(f"wrote {agg_path}", flush=True)

    out = {}
    for model in ["LLaVA", "Qwen"]:
        for method in ["vcd", "sid"]:
            rows = agg[model][method]
            print(f"bootstrapping {model}/{method} (B={args.B}, {len(rows)} images)...", flush=True)
            out[f"{model}_{method}"] = boot(rows, B=args.B)
    json.dump(out, open(HERE / "bootstrap_exp23.json", "w"), indent=2)

    print("\n=== Table 1: intervention rate (image-level bootstrap 95% CI) ===")
    for k, v in out.items():
        print(f"{k:12s} all={v['all']['point']*100:5.1f} [{v['all']['lo']*100:.1f},{v['all']['hi']*100:.1f}]  "
              f"real={v['real']['point']*100:5.1f} [{v['real']['lo']*100:.1f},{v['real']['hi']*100:.1f}]  "
              f"hall={v['hall']['point']*100:5.1f} [{v['hall']['lo']*100:.1f},{v['hall']['hi']*100:.1f}]  "
              f"gap={v['gap']['point']*100:5.1f} [{v['gap']['lo']*100:.1f},{v['gap']['hi']*100:.1f}] "
              f"p{'<' if v['gap']['p_is_bound'] else '='}{v['gap']['p_boot']:.4f}")
    print("\n=== Table 2: top-10 overlap (image-level bootstrap 95% CI) ===")
    for k, v in out.items():
        print(f"{k:12s} EA={v['ea']['point']:.2f} [{v['ea']['lo']:.2f},{v['ea']['hi']:.2f}]  "
              f"EA_pos={v['ea_pos']['point']:.2f} [{v['ea_pos']['lo']:.2f},{v['ea_pos']['hi']:.2f}]  "
              f"EC={v['ec']['point']:.2f} [{v['ec']['lo']:.2f},{v['ec']['hi']:.2f}]  "
              f"EC_pos={v['ec_pos']['point']:.2f} [{v['ec_pos']['lo']:.2f},{v['ec_pos']['hi']:.2f}]")
    print("\n=== Identical-% variants (image-level bootstrap 95% CI) ===")
    for k, v in out.items():
        print(f"{k:12s} EC_ord={v['id_ec_ord']['point']*100:5.1f} [{v['id_ec_ord']['lo']*100:.1f},{v['id_ec_ord']['hi']*100:.1f}]  "
              f"EC_set={v['id_ec_set']['point']*100:5.1f} [{v['id_ec_set']['lo']*100:.1f},{v['id_ec_set']['hi']*100:.1f}]  "
              f"EA_ord={v['id_ea_ord']['point']*100:5.1f} [{v['id_ea_ord']['lo']*100:.1f},{v['id_ea_ord']['hi']*100:.1f}]  "
              f"EA_set={v['id_ea_set']['point']*100:5.1f} [{v['id_ea_set']['lo']*100:.1f},{v['id_ea_set']['hi']*100:.1f}]")


if __name__ == "__main__":
    main()
