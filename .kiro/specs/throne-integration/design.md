# Design Document

## Overview

The THRONE integration adds free-form hallucination measurement to the cd-rethinking
evaluation pipeline.  The design follows a **vendor + bridge** pattern: THRONE is cloned
into `third_party/THRONE/` as a read-only dependency, and a thin bridge layer in
`inference/` connects this repo's LLaVA + CD stack to THRONE's `Evaluatee` interface.
No existing file is modified.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  scripts/throne_run_all.sh  /  throne_slurm.sh                  │
│    sets PYTHONPATH = inference/:third_party/THRONE/throne:...   │
│    loops over methods: none vcd icd sid apc                     │
│    calls throne_generate_cd.py LLaVA_CD --cd_method <m>        │
│    throne_progress.sh shows live done/remaining/ETA            │
└───────────────────────┬─────────────────────────────────────────┘
                        │ runs
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  third_party/THRONE/throne/throne_generate_cd.py  (NEW FILE)   │
│    imports LLaVA_CD from throne_cd_evaluatee                   │
│    replicates COCOImageDataset / InferenceSampler inline        │
│    drives LLaVA_CD through THRONE's Evaluatee protocol         │
│    checkpoints every 100 images → responses.ckpt.json          │
│    writes {"prompts": [...], "responses": [...]} JSON          │
└───────────────────────┬─────────────────────────────────────────┘
                        │ imports
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  inference/throne_cd_evaluatee.py  (NEW FILE)                  │
│    transformers compat shims (import-time):                    │
│      - deregister native "llava" AutoConfig                    │
│      - restore _expand_mask/_make_causal_mask (bloom, opt)     │
│      - SDPA → eager attention fallback                         │
│    LLaVA_CD(object)                                            │
│      format_prompt(): DEFAULT_IMAGE_TOKEN + conv template       │
│      load():          install_cd_patches() +                   │
│                       load_pretrained_model()                   │
│      tokenize_prompt(): tokenizer_image_token() → CPU [L]      │
│      generate_batch():  per-image _generate_one()              │
│        VCD: gen_kwargs["images_cd"] = add_diffusion_noise(img) │
│        ICD: gen_kwargs["input_ids_cd"] = tokenized ICD prompt  │
│        SID: gen_kwargs["use_sid"] = True                       │
│        APC: gen_kwargs["use_apc"] = True                       │
└────────┬──────────────────────────┬────────────────────────────┘
         │ imports                  │ imports
         ▼                          ▼
┌─────────────────────┐  ┌──────────────────────────┐
│  inference/         │  │  llava/ (existing)        │
│  cd_utils/          │  │    model/builder.py        │
│  spurious_utils/    │  │    language_model/...       │
└─────────────────────┘  └──────────────────────────┘
```

### What is NOT modified

| File | Reason kept unmodified |
|------|------------------------|
| `third_party/THRONE/throne/evaluated_models.py` | `LLaVA_CD` is imported directly in `throne_generate_cd.py`, not via the registry |
| `third_party/THRONE/throne/throne_generate.py` | Replaced by `throne_generate_cd.py` for CD experiments |
| `third_party/THRONE/throne/throne_aqa_evaluation.py` | Used as-is for step 2 |
| `third_party/THRONE/throne/throne_score_aqa.py` | Used as-is for step 3 |
| All `inference/mme_infer_*.py` files | Unrelated to THRONE; not touched |
| `pyproject.toml` | Version pins preserved |

## Component Details

### `inference/throne_cd_evaluatee.py`

**Key design decisions:**

1. **Per-image generation (batch size 1)**: THRONE's dataloader `batch_size` controls
   how many images arrive in `generate_batch`, but `_generate_one` processes each image
   independently.  This ensures the CD noise image (VCD), the contrastive prompt (ICD),
   and the internal model state (SID/APC) are all computed relative to the current image
   only — matching `inference/mme_infer_*.py` exactly.

2. **Patch-before-load ordering**: `install_cd_patches` is called inside `load()`, before
   `load_pretrained_model`, mirroring `mme_infer_cd.py`.

3. **Lazy llava imports**: The `llava.*` imports are deferred to `__init__` so the module
   can be imported (e.g., to inspect `CD_METHODS`) without requiring the full LLaVA stack.

4. **transformers compatibility shims** (import-time, before llava import): newer
   transformers builds (≥4.36) (a) ship a native `llava` AutoConfig that collides with
   this repo's `AutoConfig.register("llava", ...)`, (b) removed `_expand_mask` /
   `_make_causal_mask` from `bloom`/`opt` modeling modules that
   `llava/.../mpt/hf_prefixlm_converter.py` still imports, and (c) default to SDPA
   attention which `LlavaLlamaModel` doesn't declare.  The module deregisters the native
   config, restores the four mask helpers, and wraps `_autoset_attn_implementation` to
   fall back to `eager`.  All three are best-effort try/except blocks so the module still
   imports on the repo's originally-pinned transformers 4.31.

### `third_party/THRONE/throne/throne_generate_cd.py`

**Key design decisions:**

1. **No import from `throne_generate.py`**: That file sets
   `os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]` at module level,
   which crashes any single-GPU launch where `LOCAL_RANK` is not set.  The needed
   utilities are reproduced inline.

2. **Checkpointing**: every `CHECKPOINT_EVERY` (100) images the partial responses are
   written atomically to `responses.ckpt.json` (write tmp + `os.replace`).  On startup
   the driver loads the checkpoint, builds the set of done COCO IDs, and only enqueues
   the remaining images.  When all images finish it writes `responses.json` and removes
   the checkpoint.  A pre-existing `responses.json` short-circuits the whole method.

3. **CUDA_VISIBLE_DEVICES set lazily**: in DDP mode the assignment is deferred to inside
   `generate_responses`, after `RANK` is confirmed present, so the script can also run as
   a plain `python` invocation on a single GPU.

4. **Subparser pattern**: the `LLaVA_CD` subparser mirrors `throne_generate.py`'s
   subparsers so the CLI is familiar and the namespace is compatible with `LLaVA_CD.__init__`.

### Orchestration scripts

| Script | Role |
|--------|------|
| `throne_setup_env.sh` | one-time pip installs (pycocotools, sentencepiece, protobuf) |
| `throne_fetch_coco.sh` | idempotent COCO val2017 + annotations download |
| `throne_generate.sh` | step 1 for all methods (single- or multi-GPU) |
| `throne_eval.sh` | steps 2 + 3 for all methods |
| `throne_run_all.sh` | download + steps 1–3 end-to-end, checkpoint-resumable, for tmux |
| `throne_slurm.sh` | `sbatch` wrapper of the full pipeline, resumes on re-submit |
| `throne_progress.sh` | live dashboard parsing tqdm from run.log + checkpoints |

### PYTHONPATH strategy

Both orchestration scripts prepend two directories to `PYTHONPATH`:

| Directory | Provides |
|-----------|----------|
| `inference/` | `throne_cd_evaluatee`, `cd_utils.*`, `spurious_utils.*` |
| `third_party/THRONE/throne/` | `throne_aqa_evaluation`, `throne_score_aqa`, `throne_constants`, `evaluated_models` |

This avoids any `pip install -e` or `setup.py` invocations against the vendored tree.

## Data Flow

```
COCO val2017 images (5000)
        │
        ▼
throne_generate_cd.py  ── checkpoint every 100 → responses.ckpt.json
  LLaVA_CD.generate_batch()  ── CD method kwargs ── llava_llama.generate()
        │
        ▼
outputs/throne/<method>/responses.json
        │
        ▼  (for each of 3 Flan-T5 models)
throne_aqa_evaluation.py
        │
        ▼
outputs/throne/<method>/eval/aqa_answers_<model>.json
        │
        ▼
throne_score_aqa.py  → outputs/throne/<method>/scores.txt
```

## Excluded Methods

### OLM

`olm_utils.py` monkeypatches the greedy-search loop to replace the standard next-token
logit distribution with a binary decision between `YES_INDEX = 3869` and `NO_INDEX = 1939`.
Every generated token is forced to be "yes" or "no" — correct for POPE-style yes/no QA but
incoherent for free-form captioning.

### PBA

PBA appends *"Answer yes whenever possible"* to the user prompt.  For a "Describe this image
in detail." prompt this suffix is semantically incoherent and has no decoding-time effect on
hallucination in free-form text.
