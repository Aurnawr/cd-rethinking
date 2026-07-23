# Requirements Document

## Introduction

This feature integrates the [THRONE](https://github.com/amazon-science/THRONE) object-hallucination
benchmark into the cd-rethinking evaluation pipeline so that each contrastive-decoding method
(VCD, ICD, SID, APC, and the unmodified baseline) can be scored under the same THRONE P/R/F0.5
metric, enabling a direct comparison supplementary to the existing MME and POPE benchmarks.

THRONE evaluates free-form captions against COCO ground-truth annotations using a Flan-T5
ensemble as an extractive QA judge, so it measures hallucination in a setting that is harder to
game than yes/no probing benchmarks.

The implementation must not modify any existing file in the repository or in the vendored
`third_party/THRONE/` tree.  All new code is isolated to new files only.

## Glossary

- **Evaluatee**: THRONE's abstract interface (`load`, `format_prompt`, `tokenize_prompt`,
  `generate_batch`) declared in `third_party/THRONE/throne/evaluated_models.py`.
- **LLaVA_CD**: The new `Evaluatee` implementation in `inference/throne_cd_evaluatee.py` that
  drives this repo's LLaVA-v1.5 model with the selected CD method.
- **CD method**: One of `none` (baseline), `vcd`, `icd`, `sid`, `apc`.
  OLM and PBA are excluded (see Requirement 6).
- **THRONE step 1**: Caption generation — the VLM describes each COCO image.
- **THRONE step 2**: AQA evaluation — Flan-T5 answers yes/no per COCO category from each caption.
- **THRONE step 3**: Scoring — majority voting + P/R/F1/F0.5 computation.
- **Responses file**: JSON with `{"prompts": [...], "responses": [[prompt_idx, coco_id, text], ...]}`.
- **Generation driver**: `third_party/THRONE/throne/throne_generate_cd.py` — the new standalone
  step-1 script that uses `LLaVA_CD` directly without touching the THRONE registry.
- **Checkpoint**: `responses.ckpt.json` written every N images so generation is resumable.

## Requirements

### Requirement 1: Implement the THRONE Evaluatee interface for LLaVA + CD

**User Story:** As a researcher, I want to run THRONE step 1 with any of the repo's CD methods
so I can compare hallucination rates across methods on the same COCO evaluation set.

**Acceptance Criteria:**

1. A new file `inference/throne_cd_evaluatee.py` defines `LLaVA_CD(object)`.
2. `LLaVA_CD` implements `format_prompt`, `load`, `tokenize_prompt`, and `generate_batch`
   matching THRONE's `Evaluatee` contract.
3. `load()` calls `install_cd_patches(cd_method)` before the first `generate()` call,
   mirroring the patch-before-load ordering in `inference/mme_infer_*.py`.
4. `generate_batch` processes one image at a time (batch size 1) so that per-image
   CD kwargs (`images_cd`, `input_ids_cd`, `use_sid`, `use_apc`) are computed correctly.
5. The class constructor raises `ValueError` for any `cd_method` not in `CD_METHODS`.
6. No existing file is modified.

### Requirement 2: Standalone step-1 generation script with checkpointing

**User Story:** As a researcher, I want a single script to run THRONE step 1 for any CD
method without modifying THRONE's model registry, and I want it to resume if interrupted.

**Acceptance Criteria:**

1. A new file `third_party/THRONE/throne/throne_generate_cd.py` exists (new file, not
   a modification of `throne_generate.py`).
2. The script drives `LLaVA_CD` directly — it does not call `get_evaluated_model` and
   does not modify `evaluated_models.py`.
3. The script accepts the same core args as `throne_generate.py` plus a `LLaVA_CD`
   subcommand with `--model_path`, `--cd_method`, `--temperature`, `--top_p`, `--noise_step`.
4. The script works in both single-GPU and multi-GPU (`torchrun`) modes, writing the
   standard THRONE responses JSON.
5. The script writes a checkpoint every N images, skips already-processed COCO IDs on
   resume, and skips entirely if the final responses.json exists.
6. The output JSON is accepted unchanged by `throne_aqa_evaluation.py` (step 2).

### Requirement 3: Environment setup script

**User Story:** As a researcher setting up a new machine, I want a single script to install
THRONE's additional Python dependencies without breaking the repo's pinned torch/transformers.

**Acceptance Criteria:**

1. `scripts/throne_setup_env.sh` installs `pycocotools`, `sentencepiece`, and `protobuf`
   without version pins that conflict with the repo's torch/transformers.
2. The script verifies that `pycocotools` and `sentencepiece` are importable after install.
3. The script does not install THRONE's `requirements.txt`.

### Requirement 4: COCO data download script

**User Story:** As a researcher, I want a single script to download the COCO val2017 images
and annotations needed for THRONE evaluation.

**Acceptance Criteria:**

1. `scripts/throne_fetch_coco.sh` downloads `val2017.zip` and
   `annotations_trainval2017.zip` from the official COCO URLs.
2. The script extracts files to `data/coco/val2017/` and
   `data/coco/annotations/instances_val2017.json`.
3. The script is idempotent — re-running it skips already-downloaded files.
4. The `DATA_ROOT` default can be overridden via environment variable.

### Requirement 5: Orchestration scripts for all three THRONE steps

**User Story:** As a researcher, I want shell scripts that run the complete THRONE pipeline
for all five CD methods with a single command, locally and on SLURM.

**Acceptance Criteria:**

1. `scripts/throne_generate.sh` iterates over methods `none vcd icd sid apc` and calls
   `throne_generate_cd.py` for each.
2. `scripts/throne_eval.sh` iterates over the same five methods, runs
   `throne_aqa_evaluation.py` with all three Flan-T5 evaluators, then `throne_score_aqa.py`.
3. `scripts/throne_run_all.sh` downloads model + COCO then runs steps 1–3 end-to-end,
   suitable for a tmux session, with checkpoint-based resume.
4. `scripts/throne_slurm.sh` wraps the full pipeline as an `sbatch` job that resumes on
   re-submission.
5. `scripts/throne_progress.sh` shows a live dashboard of done/remaining/ETA per method.
6. All scripts set `PYTHONPATH` to include `inference/` and `third_party/THRONE/throne/`
   so imports resolve without `pip install`.

### Requirement 6: Document excluded methods (OLM, PBA)

**User Story:** As a reviewer reading the results, I want to understand why OLM and PBA
are absent from the THRONE comparison table.

**Acceptance Criteria:**

1. `README_THRONE.md` contains a dedicated "Excluded methods" section explaining that:
   - OLM hard-codes yes/no token IDs and cannot produce free-form captions.
   - PBA appends a yes-biasing prompt suffix that is meaningless for descriptive generation.
2. The same explanation appears in the docstring of `inference/throne_cd_evaluatee.py`.
3. `LLaVA_CD.__init__` raises `ValueError` with a clear message if `cd_method` is `olm`
   or `pba`, directing the user to `README_THRONE.md`.

### Requirement 7: Documentation

**User Story:** As a researcher reproducing the paper's THRONE results, I want a README
that walks through the full pipeline end-to-end.

**Acceptance Criteria:**

1. `README_THRONE.md` documents: prerequisites, env setup, COCO download, step 1
   generation, step 2+3 evaluation, resume behavior, interpreting P/R/F0.5 scores.
2. The file includes a table of included CD methods and a table of excluded methods with
   reasons.
3. The file includes a section explaining how the bridge works (Evaluatee interface,
   PYTHONPATH strategy, batch-size-1 rationale, transformers compat shims).
