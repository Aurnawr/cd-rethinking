"""
bucket_top5.py -- "top-5 object-probability-mass" bucket experiment.

For every decoding step whose FINAL emitted token is a CHAIR object mention
(real or hallucinated), take a branch's top-5 candidate tokens and their
softmax probabilities P (over the stored top-30). Among those 5, keep only the
tokens that are object words, split into REAL / HALLUCINATED, and compute
    real_frac = sum(P over real object tokens)   / sum(P over ALL object tokens in top-5)
    hall_frac = sum(P over hallucinated tokens)  / sum(P over ALL object tokens in top-5)
(so real_frac + hall_frac = 1 for that step). Average over all such steps.

Token classification within the top-5:
  * the emitted token (== chosen_id) is labelled by the caption-level CHAIR
    mention that selected the step (its real/hall status is known exactly);
  * every OTHER candidate is classified by single-token vocab match: decode ->
    word-initial only -> singularize -> look up in the CHAIR synonym dict; if it
    maps to an MSCOCO-80 category it is an object (real iff that category is in
    the image ground truth), else it is dropped as a non-object.

Branches: Expert (E), Amateur (A), Contrastive 2E-A pre-APC (C).
Runs on the stored top-30 captures -- no GPU. Covers all 500 images.

Output: bucket_top5_results.json  (+ printed tables)
"""
import argparse, json, sys, math
from pathlib import Path
from collections import defaultdict

REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
LLAVA_ROOT = REPO / "llava_logit_eda" / "CHAIR-llava-detailed"
HERE = Path(__file__).resolve().parent
COCO = LLAVA_ROOT / "data" / "coco" / "annotations"
TOP5 = 5

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
    """Return 'real' | 'hall' | None(=non-object) for a single candidate token.
    `mention_label` is 'real'/'hall' when this token is the emitted object token
    (known exactly from caption-level CHAIR), else None -> vocab match."""
    if mention_label is not None:
        return mention_label
    s = tok.decode([tid])
    if not (s.startswith(" ") or s.startswith("▁")):   # word-initial only
        return None
    w = "".join(c for c in s.lower() if c.isalpha())
    if not w:
        return None
    w = singularize(w)
    if w not in chair.mscoco_objects:
        return None
    cat = chair.inverse_synonym_dict[w]
    return "real" if cat in gt else "hall"


def run(model, method, tok, chair_mod):
    from transformers import AutoTokenizer
    cfg = CFG[model]
    path = cfg["topk"] / f"captions_topk_{method}.jsonl"
    recs = [json.loads(l) for l in open(path)]
    chair = chair_mod.load_chair([r["image_id"] for r in recs], str(COCO))
    from eval.chair import singularize
    all_mentions = chair_mod.all_mentions

    # accumulators: branch -> list of (real_frac, split_key)
    acc = {b: [] for b in BRANCHES}
    n_steps_total = 0
    n_no_object_in_top5 = {b: 0 for b in BRANCHES}

    for r in recs:
        steps = r["steps"]
        chosen = [s["chosen_id"] for s in steps]
        gt = chair.imid_to_objects.get(r["image_id"], set())
        # object-emitting steps -> (step_index -> mention label)
        step_label = {}
        for m in all_mentions(chair, tok, r["image_id"], chosen):
            if m["category"] != "object":
                continue
            si = m["step_indices"][0]
            if si < len(steps):
                step_label.setdefault(si, "hall" if m["hallucinated"] else "real")

        for si, mlabel in step_label.items():
            s = steps[si]
            n_steps_total += 1
            for b, (id_key, lg_key) in BRANCHES.items():
                ids = s[id_key][:TOP5]
                probs = softmax(s[lg_key])[:TOP5]
                real_p = hall_p = 0.0
                for tid, p in zip(ids, probs):
                    lbl = classify_token(
                        tok, tid, chair, singularize, gt,
                        mlabel if tid == s["chosen_id"] else None)
                    if lbl == "real":
                        real_p += p
                    elif lbl == "hall":
                        hall_p += p
                denom = real_p + hall_p
                if denom == 0:
                    n_no_object_in_top5[b] += 1
                    continue
                acc[b].append((real_p / denom, mlabel))
    return acc, n_steps_total, n_no_object_in_top5


def summarize(acc):
    def stats(vals):
        if not vals:
            return None
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
        return {"n": n, "real": mean, "hall": 1 - mean, "sd": math.sqrt(var)}
    out = {}
    for b, rows in acc.items():
        allv = [rf for rf, _ in rows]
        realsteps = [rf for rf, k in rows if k == "real"]
        hallsteps = [rf for rf, k in rows if k == "hall"]
        out[b] = {"all": stats(allv),
                  "at_real_steps": stats(realsteps),
                  "at_hall_steps": stats(hallsteps)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["LLaVA", "Qwen"])
    ap.add_argument("--methods", nargs="+", default=["vcd", "sid"])
    args = ap.parse_args()

    sys.path.insert(0, str(LLAVA_ROOT))
    import llava_logit_eda.object_mentions as chair_mod
    from transformers import AutoTokenizer

    results = {}
    for model in args.models:
        cfg = CFG[model]
        tok = AutoTokenizer.from_pretrained(str(cfg["tok"]), use_fast=not cfg["slow"])
        for method in args.methods:
            key = f"{model}_{method}"
            print(f"processing {key} ...", flush=True)
            acc, ntot, nno = run(model, method, tok, chair_mod)
            results[key] = {"summary": summarize(acc), "n_object_steps": ntot,
                            "n_no_object_in_top5": nno}

    json.dump(results, open(HERE / "bucket_top5_results.json", "w"), indent=2)

    print("\n" + "=" * 78)
    print("TOP-5 OBJECT-MASS BUCKETS  (real_frac | hall_frac), averaged over object steps")
    print("=" * 78)
    for key, r in results.items():
        print(f"\n### {key}   ({r['n_object_steps']} object-emitting steps)")
        print(f"  {'branch':<12} {'n':>6}  {'REAL':>7}  {'HALL':>7}")
        for b in BRANCHES:
            st = r["summary"][b]["all"]
            print(f"  {b:<12} {st['n']:>6}  {st['real']*100:>6.1f}%  {st['hall']*100:>6.1f}%")
    print("\n" + "=" * 78)
    print("SPLIT BY EMITTED-MENTION TYPE  (real_frac of top-5 object mass)")
    print("=" * 78)
    for key, r in results.items():
        print(f"\n### {key}")
        print(f"  {'branch':<12} {'@real-steps':>22}   {'@hall-steps':>22}")
        for b in BRANCHES:
            rs = r["summary"][b]["at_real_steps"]
            hs = r["summary"][b]["at_hall_steps"]
            rss = f"{rs['real']*100:.1f}% real (n={rs['n']})" if rs else "-"
            hss = f"{hs['real']*100:.1f}% real (n={hs['n']})" if hs else "-"
            print(f"  {b:<12} {rss:>22}   {hss:>22}")


if __name__ == "__main__":
    main()
