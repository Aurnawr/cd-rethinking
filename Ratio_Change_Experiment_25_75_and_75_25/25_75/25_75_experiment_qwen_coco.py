"""
Creates the 25/75 YES/NO stress-test experiment for Qwen2.5-7B COCO POPE.
Same deletion mask (seed=42) as all other experiments.
"""

import json, os, random, shutil
from collections import defaultdict

SRC_INPUT  = "/teamspace/studios/this_studio/cd_rethink/qwen2.5-7B/data/coco"
SRC_OUTPUT = "/teamspace/studios/this_studio/cd_rethink/25_75exp_qwen7B_COCO/outputs/pope"
DST_BASE   = "/teamspace/studios/this_studio/cd_rethink/25_75exp_qwen7B_COCO"

VARIANTS = ["random", "popular", "adversarial"]
METHODS  = ["baseline", "vcd", "sid", "icd", "pba", "olm"]

INPUT_TEMPLATE  = os.path.join(SRC_INPUT,  "coco_pope_{variant}.json")
OUTPUT_TEMPLATE = os.path.join(SRC_OUTPUT, "{method}/qwen-7b-coco-{variant}-greedy.jsonl")
DST_INPUT_TMPL  = os.path.join(DST_BASE,   "data/coco/coco_pope_{variant}.json")
DST_OUTPUT_TMPL = os.path.join(DST_BASE,   "outputs/pope/{method}/qwen-7b-coco-{variant}-greedy.jsonl")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── Step 1: Deletion mask (seed=42, same as all other experiments) ────────────
print("=== Step 1: Building deletion mask (seed=42) ===")

random_input = load_jsonl(INPUT_TEMPLATE.format(variant="random"))
image_order, image_questions = [], defaultdict(list)
for q in random_input:
    img = q["image"]
    if img not in image_questions:
        image_order.append(img)
    image_questions[img].append(q)

assert len(image_order) == 500

rng = random.Random(42)
deletion_mask = {}
for img in image_order:
    yes_qs = [q for q in image_questions[img] if q["label"] == "yes"]
    assert len(yes_qs) == 3
    deletion_mask[img] = set(rng.sample([0, 1, 2], 2))

from collections import Counter
print(f"  Deletion combos: {dict(Counter(tuple(sorted(v)) for v in deletion_mask.values()))}")


# ── Step 2: Keep sets per variant ─────────────────────────────────────────────
print("\n=== Step 2: Computing keep sets ===")

keep_ids = {}
for variant in VARIANTS:
    data = load_jsonl(INPUT_TEMPLATE.format(variant=variant))
    img_qs = defaultdict(list)
    for q in data: img_qs[q["image"]].append(q)

    keep_set = set()
    for img in image_order:
        yes_qs = [q for q in img_qs[img] if q["label"] == "yes"]
        no_qs  = [q for q in img_qs[img] if q["label"] == "no"]
        for ordinal, q in enumerate(yes_qs):
            if ordinal not in deletion_mask[img]:
                keep_set.add(q["question_id"])
        for q in no_qs:
            keep_set.add(q["question_id"])

    keep_ids[variant] = keep_set
    yes_kept = sum(1 for q in data if q["question_id"] in keep_set and q["label"] == "yes")
    no_kept  = sum(1 for q in data if q["question_id"] in keep_set and q["label"] == "no")
    print(f"  {variant:12s}: {len(keep_set)} questions | YES={yes_kept} NO={no_kept} | YES%={yes_kept/(yes_kept+no_kept)*100:.1f}%")


# ── Step 3: Write filtered input files ───────────────────────────────────────
print("\n=== Step 3: Writing filtered input files ===")
for variant in VARIANTS:
    data     = load_jsonl(INPUT_TEMPLATE.format(variant=variant))
    filtered = [q for q in data if q["question_id"] in keep_ids[variant]]
    dst      = DST_INPUT_TMPL.format(variant=variant)
    write_jsonl(filtered, dst)
    print(f"  {len(filtered)} lines → {dst}")


# ── Step 4: Write filtered output files ──────────────────────────────────────
print("\n=== Step 4: Writing filtered output files ===")
for method in METHODS:
    for variant in VARIANTS:
        src = OUTPUT_TEMPLATE.format(method=method, variant=variant)
        if not os.path.exists(src):
            print(f"  MISSING: {src}"); continue
        data     = load_jsonl(src)
        filtered = [q for q in data if q["question_id"] in keep_ids[variant]]
        dst      = DST_OUTPUT_TMPL.format(method=method, variant=variant)
        write_jsonl(filtered, dst)
        print(f"  {method:8s} / {variant:11s}: {len(filtered)} lines")


# ── Step 5: Copy eval script ──────────────────────────────────────────────────
dst_eval = os.path.join(DST_BASE, "eval/pope_eval_base.py")
os.makedirs(os.path.dirname(dst_eval), exist_ok=True)
shutil.copy2("/teamspace/studios/this_studio/cd_rethink/LLava1.5-7B/eval/pope_eval_base.py", dst_eval)
print(f"\n=== Copied eval script ===")


# ── Step 6: Verification ──────────────────────────────────────────────────────
print("\n=== Step 6: Verification ===")
all_ok = True
for variant in VARIANTS:
    in_data = load_jsonl(DST_INPUT_TMPL.format(variant=variant))
    in_ids  = [q["question_id"] for q in in_data]
    yes_count = sum(1 for q in in_data if q["label"] == "yes")
    no_count  = sum(1 for q in in_data if q["label"] == "no")
    print(f"\n  [{variant}] {len(in_data)} lines | YES={yes_count} NO={no_count}")
    for method in METHODS:
        out_data = load_jsonl(DST_OUTPUT_TMPL.format(method=method, variant=variant))
        out_ids  = [q["question_id"] for q in out_data]
        ok = in_ids == out_ids
        if not ok: all_ok = False; print(f"    !! {method}: MISMATCH")
        else: print(f"    {method:8s}: {len(out_data)} lines | q_ids match ✓")

print("\n" + ("=== ALL CHECKS PASSED ===" if all_ok else "=== SOME CHECKS FAILED ==="))
