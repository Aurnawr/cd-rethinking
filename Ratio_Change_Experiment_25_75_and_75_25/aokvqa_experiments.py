"""
AOKVQA 25/75 and 75/25 stress tests for LLaVA-v1.5-7B and Qwen2.5-VL-7B.

Label structure (same as GQA):
  Random/Popular : YYYYYY NNNNNN per image
  Adversarial    : YNYNYN per image

For 25/75: delete 2 of 3 YES questions per image → 1 YES + 3 NO = 25/75
For 75/25: delete 2 of 3 NO  questions per image → 3 YES + 1 NO = 75/25

Deletion mask built from the RANDOM split (canonical YES/NO ordering).
Same mask used across all variants and all methods. Seed = 42.
"""

import json, os, random, shutil
from collections import defaultdict, Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW     = "/teamspace/studios/this_studio/cd_rethink/raw_aokvqa_outputs"
ANNO    = "/teamspace/studios/this_studio/cd_rethink/LLava1.5-7B/data/aokvqa"
EVAL_SRC= "/teamspace/studios/this_studio/cd_rethink/LLava1.5-7B/eval/pope_eval_base.py"
EXPBASE = "/teamspace/studios/this_studio/cd_rethink"

VARIANTS = ["random", "popular", "adversarial"]
METHODS  = ["baseline", "vcd", "sid", "icd", "pba", "olm"]
MODELS   = [
    ("llava", "LLava7B"),
    ("qwen",  "qwen7B"),
]
RATIOS   = ["25_75", "75_25"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def evaluate(ref_data, res_data):
    tp = tn = fp = fn = 0
    for ref, res in zip(ref_data, res_data):
        if ref["question_id"] != res["question_id"]:
            raise ValueError(f"ID mismatch: {ref['question_id']} vs {res['question_id']}")
        gt  = ref["label"].strip().lower()
        ans = res["text"].strip().lower()
        if gt == "yes":
            if "yes" in ans: tp += 1
            else:            fn += 1
        else:
            if "no" in ans:  tn += 1
            else:            fp += 1
    total  = tp + tn + fp + fn
    acc    = (tp + tn) / total * 100
    yes_pct= (tp + fp) / total * 100
    return tp, tn, fp, fn, acc, yes_pct

# ── Build deletion masks (one per ratio, shared across models & variants) ─────
print("=== Building deletion masks from AOKVQA Random split ===")

random_anno = load_jsonl(os.path.join(ANNO, "aokvqa_pope_random.json"))
image_order = []
image_qs    = defaultdict(list)
for q in random_anno:
    img = q["image"]
    if img not in image_qs:
        image_order.append(img)
    image_qs[img].append(q)

assert len(image_order) == 500

rng = random.Random(42)

# 25/75: delete 2 YES ordinals
del_yes = {}
for img in image_order:
    yes_qs = [q for q in image_qs[img] if q["label"] == "yes"]
    assert len(yes_qs) == 3
    del_yes[img] = set(rng.sample([0, 1, 2], 2))

# 75/25: delete 2 NO ordinals (re-seed for independence)
rng2 = random.Random(42)
del_no = {}
for img in image_order:
    no_qs = [q for q in image_qs[img] if q["label"] == "no"]
    assert len(no_qs) == 3
    del_no[img] = set(rng2.sample([0, 1, 2], 2))

print(f"  25/75 deletion combos: {dict(Counter(tuple(sorted(v)) for v in del_yes.values()))}")
print(f"  75/25 deletion combos: {dict(Counter(tuple(sorted(v)) for v in del_no.values()))}")


# ── Main experiment loop ───────────────────────────────────────────────────────
results = {}   # results[(ratio, model_key, method, variant)] = (tp,tn,fp,fn,acc,yes_pct)

for ratio in RATIOS:
    is_25_75 = (ratio == "25_75")
    del_mask = del_yes if is_25_75 else del_no
    del_label = "yes" if is_25_75 else "no"

    for (model_raw, model_tag) in MODELS:
        exp_dir = os.path.join(EXPBASE, f"{ratio}_experiments", f"{ratio}exp_{model_tag}_AOKVQA")
        print(f"\n{'='*60}")
        print(f"  {ratio}  |  {model_tag}  →  {exp_dir}")
        print(f"{'='*60}")

        # ── Compute keep sets per variant ──────────────────────────────────
        keep_ids = {}
        for variant in VARIANTS:
            anno = load_jsonl(os.path.join(ANNO, f"aokvqa_pope_{variant}.json"))
            img_qs_v = defaultdict(list)
            for q in anno:
                img_qs_v[q["image"]].append(q)

            keep_set = set()
            for img in image_order:
                qs = img_qs_v[img]
                yes_qs = [q for q in qs if q["label"] == "yes"]
                no_qs  = [q for q in qs if q["label"] == "no"]
                assert len(yes_qs) == 3 and len(no_qs) == 3, \
                    f"{img} {variant}: YES={len(yes_qs)} NO={len(no_qs)}"

                mask = del_mask[img]
                if is_25_75:
                    # keep 1 YES, all 3 NO
                    for i, q in enumerate(yes_qs):
                        if i not in mask:
                            keep_set.add(q["question_id"])
                    for q in no_qs:
                        keep_set.add(q["question_id"])
                else:
                    # keep all 3 YES, 1 NO
                    for q in yes_qs:
                        keep_set.add(q["question_id"])
                    for i, q in enumerate(no_qs):
                        if i not in mask:
                            keep_set.add(q["question_id"])

            keep_ids[variant] = keep_set
            y = sum(1 for q in anno if q["question_id"] in keep_set and q["label"] == "yes")
            n = sum(1 for q in anno if q["question_id"] in keep_set and q["label"] == "no")
            print(f"  [{variant}] keep={len(keep_set)}  YES={y}  NO={n}  YES%={y/(y+n)*100:.1f}%")

        # ── Write filtered annotation files ────────────────────────────────
        for variant in VARIANTS:
            anno = load_jsonl(os.path.join(ANNO, f"aokvqa_pope_{variant}.json"))
            filtered = [q for q in anno if q["question_id"] in keep_ids[variant]]
            dst = os.path.join(exp_dir, "data", "aokvqa", f"aokvqa_pope_{variant}.json")
            write_jsonl(filtered, dst)

        # ── Write filtered output files ────────────────────────────────────
        for method in METHODS:
            for variant in VARIANTS:
                src = os.path.join(RAW, model_raw, method, f"{variant}.jsonl")
                if not os.path.exists(src):
                    print(f"  MISSING: {src}")
                    continue
                data     = load_jsonl(src)
                filtered = [q for q in data if q["question_id"] in keep_ids[variant]]
                dst = os.path.join(exp_dir, "outputs", "pope", method,
                                   f"aokvqa-{variant}-greedy.jsonl")
                write_jsonl(filtered, dst)

        # ── Copy eval script ───────────────────────────────────────────────
        dst_eval = os.path.join(exp_dir, "eval", "pope_eval_base.py")
        os.makedirs(os.path.dirname(dst_eval), exist_ok=True)
        shutil.copy2(EVAL_SRC, dst_eval)

        # ── Evaluate ───────────────────────────────────────────────────────
        print(f"\n  {'Method':<10} {'Split':<12} {'FN→TP':>7} {'TN→FP':>7} {'Yes%':>7} {'Acc%':>7}")
        print(f"  {'-'*50}")

        for variant in VARIANTS:
            ref_path = os.path.join(exp_dir, "data", "aokvqa", f"aokvqa_pope_{variant}.json")
            ref_data = load_jsonl(ref_path)

            baseline_tp = baseline_tn = baseline_fp = baseline_fn = None

            for method in METHODS:
                res_path = os.path.join(exp_dir, "outputs", "pope", method,
                                        f"aokvqa-{variant}-greedy.jsonl")
                if not os.path.exists(res_path):
                    print(f"  MISSING result: {res_path}")
                    continue
                res_data = load_jsonl(res_path)

                tp, tn, fp, fn, acc, yes_pct = evaluate(ref_data, res_data)
                results[(ratio, model_tag, method, variant)] = (tp, tn, fp, fn, acc, yes_pct)

                if method == "baseline":
                    baseline_tp, baseline_tn = tp, tn
                    baseline_fp, baseline_fn = fp, fn
                    fn_tp_str = "ref"
                    tn_fp_str = "ref"
                else:
                    d_fn_tp = (tp - baseline_tp)
                    d_tn_fp = (fp - baseline_fp)
                    fn_tp_str = f"{d_fn_tp:+d}"
                    tn_fp_str = f"{d_tn_fp:+d}"

                print(f"  {method:<10} {variant:<12} {fn_tp_str:>7} {tn_fp_str:>7} "
                      f"{yes_pct:>6.1f}% {acc:>6.2f}%")
            print()

print("\n=== ALL DONE ===")
