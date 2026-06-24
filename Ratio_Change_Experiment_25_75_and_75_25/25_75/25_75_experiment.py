"""
Creates the 25/75 YES/NO stress-test experiment for LLaVA-7B GQA POPE.

For each image (500 total), we have 3 YES questions and 3 NO questions per
POPE variant (random, popular, adversarial). We randomly delete 2 of the 3 YES
questions, keeping 1 YES + 3 NO = 25% YES / 75% NO.

The same 2 YES ordinals (0th, 1st, 2nd within each image's YES set) are deleted
across ALL variants and ALL decoding methods, ensuring consistency.

Fixed seed = 42 for full reproducibility.
"""

import json
import os
import random
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────────
SRC_BASE = "/teamspace/studios/this_studio/cd_rethink/LLava1.5-7B"
DST_BASE = "/teamspace/studios/this_studio/cd_rethink/25_75exp_LLava7B_GQA"

VARIANTS = ["random", "popular", "adversarial"]
METHODS  = ["baseline", "vcd", "sid", "icd", "pba", "olm"]

INPUT_TEMPLATE  = os.path.join(SRC_BASE, "data/gqa/gqa_pope_{variant}.json")
OUTPUT_TEMPLATE = os.path.join(SRC_BASE, "outputs/pope/{method}/llava-7b-gqa-{variant}-greedy.jsonl")

DST_INPUT_TEMPLATE  = os.path.join(DST_BASE, "data/gqa/gqa_pope_{variant}.json")
DST_OUTPUT_TEMPLATE = os.path.join(DST_BASE, "outputs/pope/{method}/llava-7b-gqa-{variant}-greedy.jsonl")


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── Step 1: Build deletion mask from RANDOM input (canonical YES ordering) ──
print("=== Step 1: Building deletion mask ===")

random_input = load_jsonl(INPUT_TEMPLATE.format(variant="random"))

# Group by image, preserving order
image_order = []
image_questions = defaultdict(list)
for q in random_input:
    img = q["image"]
    if img not in image_questions:
        image_order.append(img)
    image_questions[img].append(q)

assert len(image_order) == 500, f"Expected 500 images, got {len(image_order)}"

# For each image, randomly choose 2 of the 3 YES ordinals to delete
rng = random.Random(42)
# deletion_mask[image] = set of YES ordinals to delete (e.g. {0, 2})
deletion_mask = {}
for img in image_order:
    qs = image_questions[img]
    yes_qs = [q for q in qs if q["label"] == "yes"]
    assert len(yes_qs) == 3, f"Image {img} has {len(yes_qs)} YES questions, expected 3"
    ordinals_to_delete = set(rng.sample([0, 1, 2], 2))
    deletion_mask[img] = ordinals_to_delete

# Sanity check: show deletion distribution
from collections import Counter
combo_counts = Counter(tuple(sorted(v)) for v in deletion_mask.values())
print(f"  Deletion combos (YES ordinals removed): {dict(combo_counts)}")
# Each combo means: which ordinal is KEPT
keep_counts = Counter(({0,1,2} - v).pop() for v in deletion_mask.values())
print(f"  Which YES ordinal is kept: {dict(keep_counts)}")


# ── Step 2: Build per-variant keep sets (sets of question_ids to retain) ────
print("\n=== Step 2: Computing keep sets per variant ===")

keep_ids = {}  # keep_ids[variant] = set of question_ids to keep

for variant in VARIANTS:
    data = load_jsonl(INPUT_TEMPLATE.format(variant=variant))

    # Group by image
    img_qs = defaultdict(list)
    for q in data:
        img_qs[q["image"]].append(q)

    keep_set = set()
    for img in image_order:
        qs = img_qs[img]
        yes_qs = [q for q in qs if q["label"] == "yes"]
        no_qs  = [q for q in qs if q["label"] == "no"]
        assert len(yes_qs) == 3 and len(no_qs) == 3

        del_ordinals = deletion_mask[img]
        for ordinal, q in enumerate(yes_qs):
            if ordinal not in del_ordinals:
                keep_set.add(q["question_id"])
        for q in no_qs:
            keep_set.add(q["question_id"])

    keep_ids[variant] = keep_set
    yes_kept = sum(1 for q in data if q["question_id"] in keep_set and q["label"] == "yes")
    no_kept  = sum(1 for q in data if q["question_id"] in keep_set and q["label"] == "no")
    print(f"  {variant:12s}: keep {len(keep_set)} questions | YES={yes_kept} NO={no_kept} "
          f"| YES%={yes_kept/(yes_kept+no_kept)*100:.1f}%")


# ── Step 3: Write filtered input files ──────────────────────────────────────
print("\n=== Step 3: Writing filtered input files ===")

for variant in VARIANTS:
    data = load_jsonl(INPUT_TEMPLATE.format(variant=variant))
    filtered = [q for q in data if q["question_id"] in keep_ids[variant]]
    dst = DST_INPUT_TEMPLATE.format(variant=variant)
    write_jsonl(filtered, dst)
    print(f"  Wrote {len(filtered)} lines → {dst}")


# ── Step 4: Write filtered output files ─────────────────────────────────────
print("\n=== Step 4: Writing filtered output files ===")

for method in METHODS:
    for variant in VARIANTS:
        src = OUTPUT_TEMPLATE.format(method=method, variant=variant)
        if not os.path.exists(src):
            print(f"  MISSING: {src}")
            continue
        data = load_jsonl(src)
        filtered = [q for q in data if q["question_id"] in keep_ids[variant]]
        dst = DST_OUTPUT_TEMPLATE.format(method=method, variant=variant)
        write_jsonl(filtered, dst)
        print(f"  {method:8s} / {variant:11s}: {len(filtered)} lines → {dst}")


# ── Step 5: Copy eval script ─────────────────────────────────────────────────
import shutil
src_eval = os.path.join(SRC_BASE, "eval/pope_eval_base.py")
dst_eval = os.path.join(DST_BASE, "eval/pope_eval_base.py")
os.makedirs(os.path.dirname(dst_eval), exist_ok=True)
shutil.copy2(src_eval, dst_eval)
print(f"\n=== Copied eval script → {dst_eval} ===")


# ── Step 6: Verification ─────────────────────────────────────────────────────
print("\n=== Step 6: Verification ===")

all_ok = True
for variant in VARIANTS:
    in_path  = DST_INPUT_TEMPLATE.format(variant=variant)
    in_data  = load_jsonl(in_path)
    in_ids   = [q["question_id"] for q in in_data]
    yes_count = sum(1 for q in in_data if q["label"] == "yes")
    no_count  = sum(1 for q in in_data if q["label"] == "no")
    print(f"\n  [{variant}] input: {len(in_data)} lines | YES={yes_count} NO={no_count}")

    for method in METHODS:
        out_path = DST_OUTPUT_TEMPLATE.format(method=method, variant=variant)
        out_data = load_jsonl(out_path)
        out_ids  = [q["question_id"] for q in out_data]
        id_match = in_ids == out_ids
        if not id_match:
            print(f"    !! {method}: question_id MISMATCH")
            all_ok = False
        else:
            print(f"    {method:8s}: {len(out_data)} lines | q_ids match ✓")

print("\n" + ("=== ALL CHECKS PASSED ===" if all_ok else "=== SOME CHECKS FAILED ==="))


# ── Step 7: Sample spot-check ────────────────────────────────────────────────
print("\n=== Step 7: Spot-check (first 3 images, RANDOM variant) ===")

orig = load_jsonl(INPUT_TEMPLATE.format(variant="random"))
filt = load_jsonl(DST_INPUT_TEMPLATE.format(variant="random"))

orig_by_img = defaultdict(list)
for q in orig: orig_by_img[q["image"]].append(q)

filt_by_img = defaultdict(list)
for q in filt: filt_by_img[q["image"]].append(q)

for img in image_order[:3]:
    print(f"\n  Image: {img}")
    print(f"    Original ({len(orig_by_img[img])} qs): "
          + " ".join(f"q{q['question_id']}[{q['label'].upper()[0]}]" for q in orig_by_img[img]))
    print(f"    Filtered ({len(filt_by_img[img])} qs): "
          + " ".join(f"q{q['question_id']}[{q['label'].upper()[0]}]" for q in filt_by_img[img]))
    del_ords = deletion_mask[img]
    kept_ord = ({0,1,2} - del_ords).pop()
    print(f"    Deleted YES ordinals: {sorted(del_ords)} | Kept YES ordinal: {kept_ord}")
