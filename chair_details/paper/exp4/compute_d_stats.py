"""
compute_d_stats.py -- d = E - A statistics for the exp-4 write-up.

For both models and both methods, over three candidate populations
(emitted object word / expert top-10 object candidates / expert top-30 object
candidates), split real vs hallucinated, report:
  n, mean d, and % of points with d > 0 (the sign-frequency the barplots show).

d = E - A is the amount contrastive decoding adds to the plain expert logit
(C = 2E - A = E + d). d > 0 means CD amplifies the token; d < 0 means it
suppresses it. A working suppressor would give hallucinations negative d.

emitted:  object mentions labelled at the caption level (fragment-proof); the
          emitted first token's E and A are read from the stored top-30 lists.
top10/30: object candidates in the expert's top-K, word-initial + COCO match,
          real/hall by ground-truth membership (candidate-level, <1% multiword
          caveat). A candidate is skipped if it is absent from the saved top-30
          amateur list (documented, small).
Output: d_stats.json (+ printed tables)
"""
import json, sys
from pathlib import Path

REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
LLAVA_ROOT = REPO / "llava_logit_eda" / "CHAIR-llava-detailed"
HERE = Path(__file__).resolve().parent
COCO = LLAVA_ROOT / "data" / "coco" / "annotations"

CFG = {
    "LLaVA": {"topk": LLAVA_ROOT / "llava_logit_eda" / "outputs_full_topk",
              "tok": LLAVA_ROOT / "llava_logit_eda" / "models" / "llava-v1.5-7b", "slow": True},
    "Qwen":  {"topk": REPO / "CHAIR-qwen-detailed" / "outputs_full_topk",
              "tok": REPO / "CHAIR-qwen-detailed" / "models" / "Qwen2.5-VL-7B-Instruct", "slow": False},
}


def summ(vals):
    if not vals:
        return {"n": 0, "mean": None, "pct_pos": None}
    n = len(vals)
    return {"n": n, "mean": sum(vals) / n,
            "pct_pos": 100.0 * sum(1 for v in vals if v > 0) / n}


def run(model, method, tok, chair_mod, singularize):
    cfg = CFG[model]
    recs = [json.loads(l) for l in open(cfg["topk"] / f"captions_topk_{method}.jsonl")]
    chair = chair_mod.load_chair([r["image_id"] for r in recs], str(COCO))
    pops = {p: {"real": [], "hall": []} for p in ("emitted", "top10", "top30")}

    for r in recs:
        steps = r["steps"]
        gt = chair.imid_to_objects.get(r["image_id"], set())
        chosen = [s["chosen_id"] for s in steps]
        emap = {s["step"]: dict(zip(s["expert_top_ids"], s["expert_top_logits"])) for s in steps}
        amap = {s["step"]: dict(zip(s["amateur_top_ids"], s["amateur_top_logits"])) for s in steps}

        # emitted object words (caption-level labels)
        for m in chair_mod.all_mentions(chair, tok, r["image_id"], chosen):
            if m["category"] != "object":
                continue
            si = m["step_indices"][0]
            if si >= len(steps):
                continue
            tid = steps[si]["chosen_id"]
            E, A = emap[si].get(tid), amap[si].get(tid)
            if E is None or A is None:
                continue
            pops["emitted"]["hall" if m["hallucinated"] else "real"].append(E - A)

        # top-K expert object candidates (candidate-level labels)
        for s in steps:
            ei, el = s["expert_top_ids"], s["expert_top_logits"]
            am = amap[s["step"]]
            for K, key in ((10, "top10"), (30, "top30")):
                for tid, E in zip(ei[:K], el[:K]):
                    txt = tok.decode([tid])
                    if not (txt.startswith(" ") or txt.startswith("▁")):
                        continue
                    w = "".join(c for c in txt.lower() if c.isalpha())
                    if not w:
                        continue
                    w = singularize(w)
                    if w not in chair.mscoco_objects:
                        continue
                    A = am.get(tid)
                    if A is None:
                        continue
                    cat = chair.inverse_synonym_dict[w]
                    pops[key]["hall" if cat not in gt else "real"].append(E - A)
    return {p: {lbl: summ(v) for lbl, v in d.items()} for p, d in pops.items()}


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
            key = f"{model}_{method}"
            print(f"processing {key} ...", flush=True)
            out[key] = run(model, method, tok, chair_mod, singularize)
    json.dump(out, open(HERE / "d_stats.json", "w"), indent=2)

    for pop in ("emitted", "top10", "top30"):
        print(f"\n===== population: {pop}  (mean d | %d>0 | n) =====")
        print(f"{'setting':<11} {'real mean':>10} {'real %+':>8} {'real n':>7}"
              f" {'hall mean':>10} {'hall %+':>8} {'hall n':>7}")
        for key, r in out.items():
            re_, ha = r[pop]["real"], r[pop]["hall"]
            print(f"{key:<11} {re_['mean']:>10.3f} {re_['pct_pos']:>7.1f}% {re_['n']:>7}"
                  f" {ha['mean']:>10.3f} {ha['pct_pos']:>7.1f}% {ha['n']:>7}")


if __name__ == "__main__":
    main()
