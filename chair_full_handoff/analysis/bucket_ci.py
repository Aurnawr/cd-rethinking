"""
bucket_ci.py -- 95% bootstrap CIs for the normalized-probability-mass bucket
table (REAL vs HALL), B = 10000.

Statistic (identical to bucket_top5.py, so point estimates reproduce exactly):
  For every object-emitting step, take a branch's top-5 candidates and their
  softmax probabilities (softmax over the stored top-30). Keep only object
  tokens, split into real / hallucinated, and form
        real_frac = sum(P_real) / (sum(P_real) + sum(P_hall))
  The reported REAL value is the mean of real_frac over all contributing steps;
  HALL = 1 - REAL by construction.

Resampling unit = IMAGE (500 clusters). Steps within an image are correlated
(same picture, same ground truth), so a per-step bootstrap would understate
uncertainty. We therefore store, per image and per branch, the SUM of real_frac
and the COUNT of contributing steps, and each bootstrap replicate recomputes the
statistic as a ratio of sums over the resampled images. This matches the
image-level bootstrap used elsewhere in the paper.

NOTE: because real_frac + hall_frac = 1 at every step, the HALL interval is the
exact mirror of the REAL interval (lo_H = 1 - hi_R, hi_H = 1 - lo_R). It is not
an independent estimate.

Output: bucket_ci.json (+ printed table and LaTeX rows)
"""
import json, math, sys
from pathlib import Path
import numpy as np

REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
LLAVA_ROOT = REPO / "llava_logit_eda" / "CHAIR-llava-detailed"
HERE = Path(__file__).resolve().parent
COCO = LLAVA_ROOT / "data" / "coco" / "annotations"
TOP5 = 5
B = 10000

CFG = {
    "LLaVA": {"topk": LLAVA_ROOT / "llava_logit_eda" / "outputs_full_topk",
              "tok": LLAVA_ROOT / "llava_logit_eda" / "models" / "llava-v1.5-7b", "slow": True},
    "Qwen":  {"topk": REPO / "CHAIR-qwen-detailed" / "outputs_full_topk",
              "tok": REPO / "CHAIR-qwen-detailed" / "models" / "Qwen2.5-VL-7B-Instruct", "slow": False},
}
BRANCHES = {
    "Expert":      ("expert_top_ids", "expert_top_logits"),
    "Amateur":     ("amateur_top_ids", "amateur_top_logits"),
    "Contrastive": ("cd_top_ids", "cd_top_pre_apc_logits"),
}


def softmax(logits):
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    z = sum(e)
    return [x / z for x in e]


def classify_token(tok, tid, chair, singularize, gt, mention_label):
    if mention_label is not None:
        return mention_label
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


def per_image(model, method, tok, chair_mod, singularize):
    """returns {branch: np.array([[sum_real_frac, n_steps], ...per image])}"""
    cfg = CFG[model]
    recs = [json.loads(l) for l in open(cfg["topk"] / f"captions_topk_{method}.jsonl")]
    chair = chair_mod.load_chair([r["image_id"] for r in recs], str(COCO))
    rows = {b: [] for b in BRANCHES}

    for r in recs:
        steps = r["steps"]
        chosen = [s["chosen_id"] for s in steps]
        gt = chair.imid_to_objects.get(r["image_id"], set())
        step_label = {}
        for m in chair_mod.all_mentions(chair, tok, r["image_id"], chosen):
            if m["category"] != "object":
                continue
            si = m["step_indices"][0]
            if si < len(steps):
                step_label.setdefault(si, "hall" if m["hallucinated"] else "real")

        acc = {b: [0.0, 0] for b in BRANCHES}
        for si, mlabel in step_label.items():
            s = steps[si]
            for b, (id_key, lg_key) in BRANCHES.items():
                ids = s[id_key][:TOP5]
                probs = softmax(s[lg_key])[:TOP5]
                real_p = hall_p = 0.0
                for tid, p in zip(ids, probs):
                    lbl = classify_token(tok, tid, chair, singularize, gt,
                                         mlabel if tid == s["chosen_id"] else None)
                    if lbl == "real":
                        real_p += p
                    elif lbl == "hall":
                        hall_p += p
                den = real_p + hall_p
                if den == 0:
                    continue
                acc[b][0] += real_p / den
                acc[b][1] += 1
        for b in BRANCHES:
            rows[b].append(acc[b])
    return {b: np.array(v, dtype=float) for b, v in rows.items()}


def boot(arr, seed=0):
    """arr: [n_images, 2] = (sum_real_frac, n_steps). Ratio of sums."""
    rng = np.random.default_rng(seed)
    n = len(arr)
    point = arr[:, 0].sum() / arr[:, 1].sum()
    draws = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        a = arr[idx]
        draws[i] = a[:, 0].sum() / a[:, 1].sum()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"real": float(point), "real_lo": float(lo), "real_hi": float(hi),
            "hall": float(1 - point), "hall_lo": float(1 - hi), "hall_hi": float(1 - lo),
            "se": float(draws.std(ddof=1)), "n_steps": int(arr[:, 1].sum()),
            "n_images": int(n)}


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
            for b in BRANCHES:
                out[f"{model}|{method}|{b}"] = boot(pim[b])
    json.dump(out, open(HERE / "bucket_ci.json", "w"), indent=2)

    # ---- verify point estimates against the original bucket_top5_results.json ----
    print("\n=== point-estimate check vs bucket_top5_results.json ===")
    try:
        old = json.load(open(HERE / "bucket_top5_results.json"))
        for k, v in out.items():
            model, method, b = k.split("|")
            o = old[f"{model}_{method}"]["summary"][b]["all"]["real"]
            flag = "MATCH" if abs(o - v["real"]) < 1e-9 else f"DIFF ({o:.6f})"
            print(f"  {k:34} {v['real']:.5f}  {flag}")
    except FileNotFoundError:
        print("  (bucket_top5_results.json not found, skipping check)")

    print(f"\n=== 95% bootstrap CIs (B={B}, image-level resampling) ===")
    print(f"{'setting':34} {'REAL [95% CI]':>28} {'HALL [95% CI]':>28}")
    for k, v in out.items():
        print(f"{k:34} {v['real']:.5f} [{v['real_lo']:.5f},{v['real_hi']:.5f}] "
              f" {v['hall']:.5f} [{v['hall_lo']:.5f},{v['hall_hi']:.5f}]")

    print("\n=== LaTeX cells (REAL then HALL) ===")
    for k, v in out.items():
        print(f"% {k}")
        print(f"& ${v['real']:.5f}$ \\tiny{{[{v['real_lo']:.5f}, {v['real_hi']:.5f}]}} "
              f"& ${v['hall']:.5f}$ \\tiny{{[{v['hall_lo']:.5f}, {v['hall_hi']:.5f}]}} \\\\")


if __name__ == "__main__":
    main()
