"""
contrastive_analysis.py -- "contrastive adjustment" experiments, direct-sampling
edition. Reads outputs/captures/llava_{method}_seed{N}.jsonl (method in
sample_star, vcd, sid, icd_p1, icd_p2, icd_n1, icd_n2, icd_p3) and reports, PER
SEED (not pooled -- avoids pseudoreplication across independent noise/sampling
draws on the same 500 images):

  (1) d = E - A on emitted object words and on the expert's top-10/top-30
      object candidates, split by real / hallucinated object. Only for
      two-branch methods (vcd, sid, icd_*).

  (2) The bucket view: normalized real/hallucinated probability mass among
      the top-5 candidates at each object step, per branch, with 95%
      image-level bootstrap CIs. Only for two-branch methods.

  (3) CHAIR-S / CHAIR-I for every method x seed present, including sample_star
      -- this is the direct-sampling replacement for the old greedy
      sample_star row, so it can be compared against VCD/SID/ICD under the
      SAME (sampling) decoding rule.

Usage:
  python contrastive_analysis.py --methods sample_star vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0 1
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
COCO = REPO / "data" / "coco" / "annotations"
CAPTURES = REPO / "outputs" / "captures"
MODEL_DIR = {"llava": "llava-v1.5-7b", "qwen": "Qwen2.5-VL-7B-Instruct"}

TWO_BRANCH_METHODS = {"vcd", "sid"}  # icd_* are matched by prefix below


def is_two_branch(method):
    return method in TWO_BRANCH_METHODS or method.startswith("icd_")


def softmax(logits):
    m = max(logits); e = [math.exp(x - m) for x in logits]; z = sum(e)
    return [x / z for x in e]


WORD_INITIAL_MARKERS = ("▁", "Ġ")  # SentencePiece "▁" (LLaVA), GPT2-BPE "Ġ" (Qwen)


def word_initial_text(tok, tid):
    """Returns the clean decoded text for token id `tid` if it's a
    word-initial piece, else None. Uses convert_ids_to_tokens (the RAW
    subword, which still carries the word-boundary marker) rather than
    decode() -- this transformers version's single-token decode() silently
    strips the leading-space marker even for word-initial tokens, which
    would otherwise make every top-10/top-30 candidate look like a
    mid-word fragment and silently empty out these populations."""
    piece = tok.convert_ids_to_tokens([tid])[0]
    if not piece.startswith(WORD_INITIAL_MARKERS):
        return None
    return tok.convert_tokens_to_string([piece])


# ---------- (1) d = E - A ----------
def d_stats(path, tok, chair_mod, singularize):
    recs = [json.loads(l) for l in open(path)]
    chair = chair_mod.load_chair([r["image_id"] for r in recs], str(COCO))
    pops = {p: {"real": [], "hall": []} for p in ("emitted", "top10", "top30")}
    for r in recs:
        steps = r["steps"]
        gt = chair.imid_to_objects.get(r["image_id"], set())
        chosen = [s["chosen_id"] for s in steps]
        emap = {s["step"]: dict(zip(s["expert_top_ids"], s["expert_top_logits"])) for s in steps}
        amap = {s["step"]: dict(zip(s["amateur_top_ids"], s["amateur_top_logits"])) for s in steps}
        for m in chair_mod.all_mentions(chair, tok, r["image_id"], chosen):
            if m["category"] != "object":
                continue
            si = m["step_indices"][0]
            if si >= len(steps):
                continue
            tid = steps[si]["chosen_id"]
            E, A = emap[si].get(tid), amap[si].get(tid)
            if E is not None and A is not None:
                pops["emitted"]["hall" if m["hallucinated"] else "real"].append(E - A)
        for s in steps:
            am = amap[s["step"]]
            for K, key in ((10, "top10"), (30, "top30")):
                for tid, E in zip(s["expert_top_ids"][:K], s["expert_top_logits"][:K]):
                    txt = word_initial_text(tok, tid)
                    if txt is None:
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
                    lab = "hall" if chair.inverse_synonym_dict[w] not in gt else "real"
                    pops[key][lab].append(E - A)

    def summ(v):
        if not v:
            return "n=0"
        a = np.array(v)
        return f"mean {a.mean():+.3f}  %d>0 {100*(a>0).mean():.1f}  (n={len(v)})"

    print("  d = E - A:")
    for p in ("emitted", "top10", "top30"):
        print(f"    {p:8} real: {summ(pops[p]['real'])}")
        print(f"    {p:8} hall: {summ(pops[p]['hall'])}")
    return pops


# ---------- (2) bucket mass with CI ----------
BRANCHES = {"Expert": ("expert_top_ids", "expert_top_logits"),
            "Amateur": ("amateur_top_ids", "amateur_top_logits"),
            "Contrastive": ("cd_top_ids", "cd_top_pre_apc_logits")}


def bucket(path, tok, chair_mod, singularize, B):
    recs = [json.loads(l) for l in open(path)]
    chair = chair_mod.load_chair([r["image_id"] for r in recs], str(COCO))
    rows = {b: [] for b in BRANCHES}
    for r in recs:
        steps = r["steps"]; chosen = [s["chosen_id"] for s in steps]
        gt = chair.imid_to_objects.get(r["image_id"], set())
        lab = {}
        for m in chair_mod.all_mentions(chair, tok, r["image_id"], chosen):
            if m["category"] == "object":
                si = m["step_indices"][0]
                if si < len(steps):
                    lab.setdefault(si, "hall" if m["hallucinated"] else "real")
        acc = {b: [0.0, 0] for b in BRANCHES}
        for si, ml in lab.items():
            s = steps[si]
            for b, (ik, lk) in BRANCHES.items():
                ids, probs = s[ik][:5], softmax(s[lk])[:5]
                rp = hp = 0.0
                for tid, p in zip(ids, probs):
                    if tid == s["chosen_id"]:
                        c = ml
                    else:
                        txt = word_initial_text(tok, tid)
                        if txt is None:
                            continue
                        w = "".join(ch for ch in txt.lower() if ch.isalpha())
                        if not w:
                            continue
                        w = singularize(w)
                        if w not in chair.mscoco_objects:
                            continue
                        c = "real" if chair.inverse_synonym_dict[w] in gt else "hall"
                    if c == "real": rp += p
                    elif c == "hall": hp += p
                if rp + hp > 0:
                    acc[b][0] += rp / (rp + hp); acc[b][1] += 1
        for b in BRANCHES:
            rows[b].append(acc[b])
    print("  normalized object-mass buckets (REAL share) [95% CI]:")
    for b in BRANCHES:
        arr = np.array(rows[b], float)
        pt = arr[:, 0].sum() / arr[:, 1].sum()
        rng = np.random.default_rng(0); n = len(arr); d = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, n, n); d[i] = arr[idx, 0].sum() / arr[idx, 1].sum()
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"    {b:12} REAL {pt:.5f} [{lo:.5f}, {hi:.5f}]   HALL {1-pt:.5f} [{1-hi:.5f}, {1-lo:.5f}]")


# ---------- (3) CHAIR ----------
def chair_score_file(path):
    from eval.chair import CHAIR
    caps = [json.loads(l) for l in open(path)]
    ids = {c["image_id"] for c in caps}
    ch = CHAIR(ids, str(COCO)); ch.get_annotations()
    res = ch.compute_chair(caps)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llava", "qwen"], required=True)
    ap.add_argument("--methods", nargs="+",
                     default=["sample_star", "vcd", "sid", "icd_p1", "icd_p2", "icd_n1", "icd_n2", "icd_p3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import object_mentions as chair_mod
    from eval.chair import singularize
    slow = args.model == "llava"
    tok = AutoTokenizer.from_pretrained(str(REPO / "models" / MODEL_DIR[args.model]), use_fast=not slow)

    print(f"\n===== contrastive adjustment (direct-sampling) : {args.model} =====")
    print("\n--- CHAIR-S / CHAIR-I, per method x seed (lower is better) ---")
    for method in args.methods:
        for seed in args.seeds:
            f = CAPTURES / f"{args.model}_{method}_seed{seed}.jsonl"
            if not f.exists():
                continue
            res = chair_score_file(f)
            print(f"  {method:10} seed{seed}  CHAIR-S {res['CHAIR_s']*100:5.1f}  "
                  f"CHAIR-I {res['CHAIR_i']*100:5.1f}  (n={res['num_captions']})")

    for method in args.methods:
        if not is_two_branch(method):
            continue
        for seed in args.seeds:
            f = CAPTURES / f"{args.model}_{method}_seed{seed}.jsonl"
            if not f.exists():
                print(f"\n[skip] {f} not found")
                continue
            print(f"\n--- {method.upper()} seed={seed} ---")
            d_stats(f, tok, chair_mod, singularize)
            bucket(f, tok, chair_mod, singularize, args.B)


if __name__ == "__main__":
    main()
