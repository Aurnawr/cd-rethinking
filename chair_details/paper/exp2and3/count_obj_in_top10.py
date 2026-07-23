"""
count_obj_in_top10.py -- at every object-emitting step, count how many of the
top-10 candidate tokens are REAL-object tokens and how many are HALLUCINATED-
object tokens (non-object tokens are dropped -- "only object tokens"). Reported
separately for steps that EMIT a real object vs a hallucinated object, for the
expert / amateur / contrastive(2E-A) top-10.

Candidate classification (same rule as the bucket experiment):
  * the emitted token (== chosen_id) uses the caption-level CHAIR mention label
    (its real/hall status is known exactly);
  * every other candidate is classified by single-token vocab match: decode ->
    word-initial only -> singularize -> CHAIR synonym dict; if it maps to an
    MSCOCO-80 category it is an object (real iff that category is in the image
    ground truth), else dropped.

Image-level bootstrap 95% CIs (B=10000). Runs on the stored top-30 captures.
Output: count_obj_in_top10.json (+ printed tables)
"""
import json, sys, math
from pathlib import Path
import numpy as np

REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
LLAVA_ROOT = REPO / "llava_logit_eda" / "CHAIR-llava-detailed"
HERE = Path(__file__).resolve().parent
COCO = LLAVA_ROOT / "data" / "coco" / "annotations"
TOP = 10

CFG = {
    "LLaVA": {"topk": LLAVA_ROOT / "llava_logit_eda" / "outputs_full_topk",
              "tok": LLAVA_ROOT / "llava_logit_eda" / "models" / "llava-v1.5-7b", "slow": True},
    "Qwen":  {"topk": REPO / "CHAIR-qwen-detailed" / "outputs_full_topk",
              "tok": REPO / "CHAIR-qwen-detailed" / "models" / "Qwen2.5-VL-7B-Instruct", "slow": False},
}
BRANCHES = {"Expert": "expert_top_ids", "Amateur": "amateur_top_ids", "Contrastive": "cd_top_ids"}


def classify(tok, tid, chair, singularize, gt, known):
    if known is not None:
        return known
    s = tok.decode([tid])
    if not (s.startswith(" ") or s.startswith("▁")):
        return None
    w = "".join(c for c in s.lower() if c.isalpha())
    if not w:
        return None
    w = singularize(w)
    if w not in chair.mscoco_objects:
        return None
    return "real" if chair.inverse_synonym_dict[w] in gt else "hall"


def aggregate(model, method, tok, chair_mod, singularize):
    cfg = CFG[model]
    recs = [json.loads(l) for l in open(cfg["topk"] / f"captions_topk_{method}.jsonl")]
    chair = chair_mod.load_chair([r["image_id"] for r in recs], str(COCO))
    # per-image: emit_type -> branch -> [sum_real, sum_hall, n_steps]
    rows = []
    for r in recs:
        steps = r["steps"]
        chosen = [s["chosen_id"] for s in steps]
        gt = chair.imid_to_objects.get(r["image_id"], set())
        lbl = {}
        for m in chair_mod.all_mentions(chair, tok, r["image_id"], chosen):
            if m["category"] == "object":
                si = m["step_indices"][0]
                if si < len(steps):
                    lbl.setdefault(si, "hall" if m["hallucinated"] else "real")
        acc = {et: {b: [0, 0, 0] for b in BRANCHES} for et in ("real", "hall")}
        for si, et in lbl.items():
            s = steps[si]
            for b, key in BRANCHES.items():
                rc = hc = 0
                for tid in s[key][:TOP]:
                    c = classify(tok, tid, chair, singularize, gt,
                                 et if tid == s["chosen_id"] else None)
                    if c == "real": rc += 1
                    elif c == "hall": hc += 1
                acc[et][b][0] += rc; acc[et][b][1] += hc; acc[et][b][2] += 1
        rows.append(acc)
    return rows


def boot(rows, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    ets, bs = ("real", "hall"), list(BRANCHES)
    A = {et: {b: np.array([[r[et][b][0], r[et][b][1], r[et][b][2]] for r in rows], float)
              for b in bs} for et in ets}
    n = len(rows)

    def stat(idx):
        o = {}
        for et in ets:
            o[et] = {}
            for b in bs:
                m = A[et][b][idx]
                N = m[:, 2].sum()
                o[et][b] = (m[:, 0].sum() / N, m[:, 1].sum() / N) if N else (np.nan, np.nan)
        return o

    pt = stat(np.arange(n))
    dr = {et: {b: {"real": np.empty(B), "hall": np.empty(B)} for b in bs} for et in ets}
    for k in range(B):
        idx = rng.integers(0, n, n)
        s = stat(idx)
        for et in ets:
            for b in bs:
                dr[et][b]["real"][k], dr[et][b]["hall"][k] = s[et][b]
    res = {}
    for et in ets:
        res[et] = {}
        for b in bs:
            res[et][b] = {}
            for j, kk in enumerate(("real", "hall")):
                d = dr[et][b][kk][~np.isnan(dr[et][b][kk])]
                res[et][b][kk] = {"pt": pt[et][b][j],
                                  "lo": float(np.percentile(d, 2.5)),
                                  "hi": float(np.percentile(d, 97.5))}
            res[et][b]["n_steps"] = int(A[et][b][:, 2].sum())
    return res


def main():
    sys.path.insert(0, str(LLAVA_ROOT))
    import llava_logit_eda.object_mentions as chair_mod
    from eval.chair import singularize
    from transformers import AutoTokenizer

    results = {}
    for model in ["LLaVA", "Qwen"]:
        cfg = CFG[model]
        tok = AutoTokenizer.from_pretrained(str(cfg["tok"]), use_fast=not cfg["slow"])
        for method in ["vcd", "sid"]:
            key = f"{model}_{method}"
            print(f"processing {key} ...", flush=True)
            results[key] = boot(aggregate(model, method, tok, chair_mod, singularize))
    json.dump(results, open(HERE / "count_obj_in_top10.json", "w"), indent=2)

    def cell(x):
        return f"{x['pt']:.2f} [{x['lo']:.2f},{x['hi']:.2f}]"
    for et, ttl in [("hall", "STEPS THAT EMIT A HALLUCINATED OBJECT"),
                    ("real", "STEPS THAT EMIT A REAL OBJECT")]:
        print("\n" + "=" * 74)
        print(f"{ttl}: avg # of top-10 candidates that are real / hallucinated objects")
        print("=" * 74)
        for key, r in results.items():
            b = r[et]["Expert"]
            print(f"\n### {key}  (n={b['n_steps']} steps)")
            print(f"  {'branch':<12} {'#real-obj':>18} {'#hall-obj':>18}")
            for br in BRANCHES:
                d = r[et][br]
                print(f"  {br:<12} {cell(d['real']):>18} {cell(d['hall']):>18}")


if __name__ == "__main__":
    main()
