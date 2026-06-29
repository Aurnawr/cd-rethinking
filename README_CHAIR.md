# CHAIR Evaluation for LLaVA-v1.5

This document explains how to compute **CHAIR** (Caption Hallucination Assessment
with Image Relevance) object-hallucination metrics for LLaVA-v1.5-7b in this repo.

CHAIR is from [Rohrbach et al., EMNLP 2018, "Object Hallucination in Image
Captioning"](https://aclanthology.org/D18-1437.pdf). It is a **different** metric
from THRONE (which lives in `third_party/THRONE/` and uses a Flan-T5 evaluator).
CHAIR instead matches object words in a generated caption against COCO
ground-truth objects using a fixed synonym vocabulary.

## Metrics

Both are lower-is-better:

- **CHAIR_s** (per-sentence) = fraction of captions that mention at least one
  hallucinated object.
- **CHAIR_i** (per-instance) = fraction of all mentioned object instances that are
  hallucinated.

An object word is *hallucinated* when it is not in the image's ground-truth
object set. The ground truth for an image is the union of:

1. objects in the COCO **instance segmentation** annotations, and
2. objects mentioned in the **5 human reference captions**.

## Files

```
inference/chair_generate.py   # step 1: LLaVA-v1.5 generates COCO captions
eval/chair.py                 # CHAIR metric implementation (Rohrbach et al. 2018)
eval/chair_eval.py            # step 2: score a caption file -> CHAIR-S / CHAIR-I
scripts/chair_generate.sh     # convenience wrapper for step 1
scripts/chair_eval.sh         # convenience wrapper for step 2
```

`eval/chair.py` is dependency-light (standard library only) and embeds the
canonical COCO `synonyms.txt`, so no external data files beyond the COCO
annotations are required.

## Prerequisites

- LLaVA-v1.5-7b checkpoint at `models/llava-v1.5-7b` (override with `MODEL_PATH`).
- COCO val2017 images at `data/coco/val2017`.
- COCO annotations `instances_val2017.json` and `captions_val2017.json` at
  `data/coco/annotations`.

## Running

### Step 1 — generate captions

```bash
bash scripts/chair_generate.sh
```

This describes a deterministic random sample of **500** COCO val2017 images
(seed 42) with **greedy** decoding (`--temperature 0`) and writes
`outputs/chair/llava-7b-greedy/captions.jsonl` as `{"image_id", "caption"}` lines.
Generation appends, so re-running resumes (already-written image ids are skipped).

Override defaults via environment variables:

```bash
MODEL_PATH=/path/to/llava-v1.5-7b NUM_SAMPLES=500 TEMPERATURE=0 \
    TAG=llava-7b-greedy bash scripts/chair_generate.sh
```

Or call the script directly:

```bash
python inference/chair_generate.py \
    --model-path      models/llava-v1.5-7b \
    --image-folder    data/coco/val2017 \
    --annotation-file data/coco/annotations/instances_val2017.json \
    --answers-file    outputs/chair/llava-7b-greedy/captions.jsonl \
    --num-samples 500 --seed 42 --temperature 0 --conv-mode vicuna_v1
```

### Step 2 — score

```bash
bash scripts/chair_eval.sh
```

Prints CHAIR-S / CHAIR-I (and average objects/length) and writes a detailed
`captions_chair.json` (per-sentence hallucinated objects) next to the caption
file. Run directly with:

```bash
python eval/chair_eval.py \
    --cap-file  outputs/chair/llava-7b-greedy/captions.jsonl \
    --coco-path data/coco/annotations
```

The evaluator also accepts a THRONE-style `responses.json`
(`{"responses": [[prompt_idx, image_id, text], ...]}`), so existing THRONE
captions can be scored with CHAIR without regenerating.

## Notes

- Metrics are reported as percentages (e.g. `CHAIR-S 21.6`, `CHAIR-I 7.2`).
- Results depend on the image sample, prompt, and decoding settings; keep `seed`,
  `num-samples`, and `temperature` fixed when comparing methods.
- Tokenisation uses a simple `\w+` splitter plus a hand-written singulariser in
  place of the original code's `nltk` + `pattern.en` dependencies, matching the
  original behaviour for the COCO object vocabulary.
