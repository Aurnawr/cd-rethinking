# THRONE Evaluation for Contrastive-Decoding Methods

This document explains how to run the [THRONE](https://github.com/amazon-science/THRONE)
object-hallucination benchmark for the contrastive-decoding (CD) methods in this repo.

THRONE evaluates free-form VLM captions for object hallucination against COCO
ground-truth annotations, reporting Precision / Recall / F1 / F0.5.

---

## How THRONE works

THRONE is a three-step pipeline:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `throne_generate_cd.py` | VLM generates "Describe this image in detail." captions for COCO val2017 images |
| 2 | `throne_aqa_evaluation.py` | Flan-T5 ensemble answers yes/no per COCO category from each caption |
| 3 | `throne_score_aqa.py` | Majority voting + Precision/Recall/F0.5 vs COCO ground truth |

**Evaluator LLM**: three Flan-T5 models (`google/flan-t5-base`, `large`, `xl`) × 3
question formats = 9 yes/no votes per object category per image.  Majority thresholds
5/8/9 are reported.

---

## File layout

```
inference/
  throne_cd_evaluatee.py      # LLaVA_CD class implementing THRONE's Evaluatee interface
third_party/THRONE/throne/
  throne_generate_cd.py       # NEW: standalone step-1 driver (no registry changes)
  throne_generate.py          # original THRONE step 1 (unmodified)
  throne_aqa_evaluation.py    # step 2 — unmodified
  throne_score_aqa.py         # step 3 — unmodified
  evaluated_models.py         # unmodified (LLaVA_CD is NOT registered here)
scripts/
  throne_setup_env.sh         # install pycocotools, sentencepiece
  throne_fetch_coco.sh        # download COCO val2017 + annotations
  throne_generate.sh          # run step 1 for all CD methods
  throne_eval.sh              # run steps 2 + 3 for all CD methods
  throne_run_all.sh           # download + run all steps end-to-end (for tmux)
  throne_progress.sh          # live progress dashboard (done/remaining/ETA)
  throne_slurm.sh             # SLURM batch wrapper for the full pipeline
```

The vendored THRONE tree (`third_party/THRONE/`) is **completely unmodified**.

---

## Prerequisites

### 1. Environment setup

```bash
bash scripts/throne_setup_env.sh
```

This adds `pycocotools`, `sentencepiece`, and `protobuf` to the existing environment.
It does **not** reinstall `torch` or `transformers`; the repo's pinned versions are
kept intact.

### 2. COCO data

```bash
bash scripts/throne_fetch_coco.sh
```

Downloads COCO val2017 images (~1 GB) and `instances_val2017.json` (~241 MB) to
`data/coco/`.  If you already have COCO, point the scripts at it:

```bash
export COCO_FILE=/path/to/instances_val2017.json
export COCO_IMAGE_DIR=/path/to/val2017
```

### 3. LLaVA checkpoint

```bash
export MODEL_PATH=/path/to/llava-v1.5-7b
```

The default path is `/teamspace/studios/this_studio/models/llava-v1.5-7b`.

---

## Running the full pipeline

### Option A — one command (downloads everything, runs all steps)

```bash
tmux new -s throne
bash scripts/throne_run_all.sh 2>&1 | tee outputs/throne/run.log
```

This downloads the model + COCO if missing, then runs steps 1–3 for all five
methods.  Generation is **checkpointed every 100 images** — if interrupted, just
re-run the same command and it resumes from the last checkpoint, skipping
finished methods.

Watch progress in a second tmux window:

```bash
bash scripts/throne_progress.sh
```

### Option B — SLURM

```bash
sbatch scripts/throne_slurm.sh
```

Same pipeline, wrapped as a batch job.  Re-submitting resumes from checkpoints.

### Option C — step by step

```bash
bash scripts/throne_generate.sh    # step 1: captions for all methods
bash scripts/throne_eval.sh        # steps 2 + 3: AQA eval + scoring
```

To run a single method manually:

```bash
export PYTHONPATH=inference:third_party/THRONE/throne

python third_party/THRONE/throne/throne_generate_cd.py \
    --coco_file      data/coco/annotations/instances_val2017.json \
    --coco_image_dir data/coco/val2017 \
    --save_path      outputs/throne/vcd/responses.json \
    LLaVA_CD \
        --model_path   /path/to/llava-v1.5-7b \
        --cd_method    vcd
```

Available `--cd_method` values: `none`, `vcd`, `icd`, `sid`, `apc`.

---

## Resuming an interrupted run

Generation writes `outputs/throne/<method>/responses.ckpt.json` every 100 images.
On restart, `throne_generate_cd.py`:

1. Skips any method whose final `responses.json` already exists.
2. Loads the checkpoint, collects already-processed COCO IDs, and only generates
   the remaining images.
3. Writes the final `responses.json` and removes the checkpoint when complete.

So `bash scripts/throne_run_all.sh` (or `sbatch scripts/throne_slurm.sh`) can be
re-run any number of times and will always pick up where it stopped.

---

## CD methods included

| Method | Flag | Description |
|--------|------|-------------|
| `none` | — | Standard greedy / sampling baseline |
| `vcd` | `--cd_method vcd` | Visual Contrastive Decoding — noisy-image negative |
| `icd` | `--cd_method icd` | Instruction Contrastive Decoding — random system-prompt negative |
| `sid` | `--cd_method sid` | Self-Introspective Decoding — model-internal contrastive layers |
| `apc` | `--cd_method apc` | Adaptive Plausibility Constraint — sampling-only spurious mitigation |

### Excluded methods

**OLM** and **PBA** are intentionally not evaluated under THRONE:

- **OLM** (`olm_utils.py`) hard-codes `YES_TOKEN_ID = 3869` and `NO_TOKEN_ID = 1939`
  into its greedy-search patch.  It is designed exclusively for binary yes/no QA
  (selecting the more-plausible token from a two-token vocabulary).  Applied to
  free-form captioning, it would force every generated token to be either "Yes" or
  "No" — producing nonsense output rather than a description.

- **PBA** (`pba_utils.py`) appends *"Answer yes whenever possible"* as a prompt
  suffix.  This is a prompt hack for yes/no QA tasks; it has no meaningful semantics
  for a free-form "Describe this image" prompt and would merely inject spurious text
  into every caption.

---

## How the bridge works

`inference/throne_cd_evaluatee.py` implements THRONE's `Evaluatee` protocol using
this repo's LLaVA loader and CD monkeypatches:

```
THRONE Evaluatee interface          LLaVA_CD implementation
─────────────────────────────────── ───────────────────────────────────────────
format_prompt(prompt)            →  builds DEFAULT_IMAGE_TOKEN + prompt via
                                    conv_templates[vicuna_v1]
load()                           →  install_cd_patches() + load_pretrained_model()
tokenize_prompt(formatted)       →  tokenizer_image_token() → CPU tensor [L]
generate_batch(input_ids, imgs)  →  per-image _generate_one() with CD kwargs:
                                      VCD:  images_cd = add_diffusion_noise(img)
                                      ICD:  input_ids_cd = tokenized icd prompt
                                      SID:  use_sid = True
                                      APC:  use_apc = True
```

Generation is done **one image at a time** (batch size 1) to keep CD math identical
to the MME/POPE runs in `inference/mme_infer_*.py`.

`third_party/THRONE/throne/throne_generate_cd.py` drives `LLaVA_CD` directly
(bypassing THRONE's `get_evaluated_model` registry) so `evaluated_models.py` is
not touched.

The evaluatee module also installs a few **transformers-version compatibility
shims** at import time (deregistering the native `llava` config, restoring
`_expand_mask` / `_make_causal_mask` for bloom/opt, and an SDPA→eager attention
fallback) so the repo's custom LLaVA-v1.5 stack imports cleanly on newer
transformers builds.

---

## Interpreting results

THRONE reports scores at majority-voting thresholds **5**, **8**, and **9** (out of
9 total votes from 3 models × 3 question formats).

- **Precision**: among objects predicted present, how many actually appear in the image.
- **Recall**: among objects actually present, how many were correctly predicted.
- **F0.5**: weighted harmonic mean favouring precision (β = 0.5), the primary THRONE metric.

Lower hallucination → higher Precision → higher F0.5.  Compare `none` (baseline) to
`vcd`, `icd`, `sid`, `apc` to see which CD method most reduces hallucination.
