"""
overlap_by_segment.py -- top-10 SET overlap (order-insensitive) of the expert
top-10 with the amateur top-10 (E∩A) and the contrastive top-10 (E∩C),
reported separately over: ALL steps, REAL-object steps, HALLUCINATED-object
steps. Image-level bootstrap 95% CIs (B=10000), consistent with the rest.

Object steps are the steps that emit a CHAIR object mention (first token),
labelled real/hallucinated at the caption level.
Output: overlap_by_segment.json (+ printed LaTeX rows)
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
LLAVA_ROOT = REPO / "llava_logit_eda" / "CHAIR-llava-detailed"
HERE = Path(__file__).resolve().parent
COCO = LLAVA_ROOT / "data" / "coco" / "annotations"
TOPN = 10

CFG = {
    "LLaVA": {"topk": LLAVA_ROOT / "llava_logit_eda" / "outputs_full_topk",
              "tok": LLAVA_ROOT / "llava_logit_eda" / "models" / "llava-v1.5-7b", "slow": True},
    "Qwen":  {"topk": REPO / "CHAIR-qwen-detailed" / "outputs_full_topk",
              "tok": REPO / "CHAIR-qwen-detailed" / "models" / "Qwen2.5-VL-7B-Instruct", "slow": False},
}


def setov(a, b):
    return len(set(a[:TOPN]) & set(b[:TOPN]))


def aggregate(model, method, tok, chair_mod):
    cfg = CFG[model]
    recs = [json.loads(l) for l in open(cfg["topk"] / f"captions_topk_{method}.jsonl")]
    chair = chair_mod.load_chair([r["image_id"] for r in recs], str(COCO))
    rows = []
    for r in recs:
        steps = r["steps"]
        chosen = [s["chosen_id"] for s in steps]
        lbl = {}   # step -> 'real'/'hall'
        for m in chair_mod.all_mentions(chair, tok, r["image_id"], chosen):
            if m["category"] == "object":
                si = m["step_indices"][0]
                if si < len(steps):
                    lbl.setdefault(si, "hall" if m["hallucinated"] else "real")
        acc = {seg: {"n": 0, "ea": 0, "ec": 0} for seg in ("all", "real", "hall")}
        for i, s in enumerate(steps):
            ea = setov(s["expert_top_ids"], s["amateur_top_ids"])
            ec = setov(s["expert_top_ids"], s["cd_top_ids"])
            for seg in ("all", lbl.get(i)):
                if seg:
                    acc[seg]["n"] += 1; acc[seg]["ea"] += ea; acc[seg]["ec"] += ec
        rows.append(acc)
    return rows


def boot(rows, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    segs = ("all", "real", "hall")
    arr = {seg: {k: np.array([r[seg][k] for r in rows], float) for k in ("n", "ea", "ec")}
           for seg in segs}
    n = len(rows)

    def stat(idx):
        out = {}
        for seg in segs:
            N = arr[seg]["n"][idx].sum()
            out[seg] = (arr[seg]["ea"][idx].sum() / N if N else np.nan,
                        arr[seg]["ec"][idx].sum() / N if N else np.nan)
        return out

    point = stat(np.arange(n))
    draws = {seg: {"ea": np.empty(B), "ec": np.empty(B)} for seg in segs}
    for b in range(B):
        idx = rng.integers(0, n, n)
        s = stat(idx)
        for seg in segs:
            draws[seg]["ea"][b], draws[seg]["ec"][b] = s[seg]
    res = {}
    for seg in segs:
        res[seg] = {}
        for j, k in enumerate(("ea", "ec")):
            d = draws[seg][k][~np.isnan(draws[seg][k])]
            res[seg][k] = {"pt": point[seg][j],
                           "lo": float(np.percentile(d, 2.5)),
                           "hi": float(np.percentile(d, 97.5))}
        res[seg]["n_steps_total"] = int(arr[seg]["n"].sum())
    return res


def main():
    sys.path.insert(0, str(LLAVA_ROOT))
    import llava_logit_eda.object_mentions as chair_mod
    from transformers import AutoTokenizer

    results = {}
    for model in ["LLaVA", "Qwen"]:
        cfg = CFG[model]
        tok = AutoTokenizer.from_pretrained(str(cfg["tok"]), use_fast=not cfg["slow"])
        for method in ["vcd", "sid"]:
            key = f"{model}_{method}"
            print(f"processing {key} ...", flush=True)
            results[key] = boot(aggregate(model, method, tok, chair_mod))
    json.dump(results, open(HERE / "overlap_by_segment.json", "w"), indent=2)

    def c(x):
        return f"${x['pt']:.2f}$ \\tiny{{[{x['lo']:.2f}, {x['hi']:.2f}]}}"
    name = {"LLaVA": "LLaVA-1.5-7B", "Qwen": "Qwen2.5-VL-7B"}
    print("\n% ---- LaTeX rows (E∩A and E∩C set overlap, /10) ----")
    for model in ["LLaVA", "Qwen"]:
        for i, meth in enumerate(["vcd", "sid"]):
            r = results[f"{model}_{meth}"]
            lead = f"\\multirow{{2}}{{*}}{{{name[model]}}}\n & " if i == 0 else " & "
            print(f"{lead}{meth.upper()} & {c(r['all']['ea'])} & {c(r['all']['ec'])} "
                  f"& {c(r['real']['ea'])} & {c(r['real']['ec'])} "
                  f"& {c(r['hall']['ea'])} & {c(r['hall']['ec'])} \\\\")
        if model == "LLaVA":
            print("\\midrule")
    print("\n% step counts (all / real / hall):")
    for key, r in results.items():
        print(f"%  {key}: {r['all']['n_steps_total']} / {r['real']['n_steps_total']} / {r['hall']['n_steps_total']}")


if __name__ == "__main__":
    main()
