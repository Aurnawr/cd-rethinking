# Implementation Plan

## Task 1: Create `inference/throne_cd_evaluatee.py`

**Status**: Complete

- [x] Define `CD_METHODS = ("none", "vcd", "icd", "sid", "apc")`
- [x] transformers compat shims (native llava config dereg; bloom/opt mask helpers; SDPA fallback)
- [x] `install_cd_patches`: route to the correct `evolve_*` functions
- [x] `LLaVA_CD.__init__`: validate `cd_method`, store hyperparams, lazy-import llava
- [x] `LLaVA_CD.format_prompt` / `load` / `tokenize_prompt` / `generate_batch`
- [x] `_generate_one`: VCD / ICD / SID / APC kwargs, slice output tokens, decode
- [x] Docstring explaining OLM/PBA exclusion

## Task 2: Keep `evaluated_models.py` unmodified

**Status**: Complete — `LLaVA_CD` is imported directly, no registry shim added.

## Task 3: Create `third_party/THRONE/throne/throne_generate_cd.py`

**Status**: Complete

- [x] Inline `InferenceSampler`, `COCOImageDataset`, `simple_collate`, `batch_generate`
- [x] Checkpointing: `_load_checkpoint` / `_save_checkpoint`, resume-by-COCO-ID, atomic writes
- [x] `generate_responses`: skip-if-final-exists, single-GPU + DDP paths
- [x] argparse with `LLaVA_CD` subparser; lazy `CUDA_VISIBLE_DEVICES` in DDP branch

## Task 4: Create `scripts/throne_setup_env.sh`

**Status**: Complete

## Task 5: Create `scripts/throne_fetch_coco.sh`

**Status**: Complete

## Task 6: Create `scripts/throne_generate.sh`

**Status**: Complete

## Task 7: Create `scripts/throne_eval.sh`

**Status**: Complete

## Task 8: Create `scripts/throne_run_all.sh`

**Status**: Complete — download + steps 1–3, checkpoint-resumable, tmux-friendly.

## Task 9: Create `scripts/throne_progress.sh`

**Status**: Complete — live dashboard parsing tqdm from run.log + checkpoints, with ETA.

## Task 10: Create `scripts/throne_slurm.sh`

**Status**: Complete — `sbatch` wrapper, resumes on re-submission.

## Task 11: Create `README_THRONE.md`

**Status**: Complete — prerequisites, run options (tmux / SLURM / step-by-step),
resume behavior, included/excluded methods, bridge + compat-shim explanation, score interpretation.

## Task 12: Create Kiro spec

**Status**: Complete — requirements, design, tasks, `.config.kiro`.
