#!/bin/bash
# ---------------------------------------------------------------------------
# Central configuration for the LLaVA-v1.5-13B reproduction pipeline.
#
# Every driver script in repro_scripts/ sources this file, so all paths live
# in ONE place. Override any value by exporting it BEFORE running setup.sh or
# run_master.sh, e.g.:
#
#     export STORAGE_ROOT=/mnt/data
#     export COCO_IMAGES=/mnt/data/coco/val2014
#     bash run_master.sh
#
# Nothing here downloads or runs anything; it only defines variables and one
# helper. Sourcing it is side-effect free (it does not change your CWD).
# ---------------------------------------------------------------------------

# Resolve the repository root from this file's location (repro_scripts/ -> repo).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export REPO_ROOT

# --- Python interpreter -----------------------------------------------------
# The pinned stack (transformers==4.31.0 + tokenizers<0.14) only has wheels for
# Python <=3.11. setup.sh provisions a Python 3.10 env and records its
# interpreter path in repro_scripts/.py_env; everything runs through PY_BIN so
# it works regardless of the studio's default Python. Override by exporting
# PY_BIN yourself.
: "${PY_BIN:=}"
if [ -z "${PY_BIN}" ]; then
    if [ -f "${REPO_ROOT}/repro_scripts/.py_env" ]; then
        PY_BIN="$(cat "${REPO_ROOT}/repro_scripts/.py_env")"
    else
        PY_BIN="python"
    fi
fi
export PY_BIN
REPRO_ENV_NAME="${REPRO_ENV_NAME:-cd_rethink}"
export REPRO_ENV_NAME

# `:=` assigns the default only when the variable is unset/empty, so any value
# you export beforehand wins.
: "${STORAGE_ROOT:=/teamspace/lightning_storage}"

# --- Model ------------------------------------------------------------------
: "${HF_MODEL_ID:=liuhaotian/llava-v1.5-13b}"   # HuggingFace repo id
: "${MODEL_13B:=${STORAGE_ROOT}/models/llava-v1.5-13b}"  # local checkpoint dir
: "${MODEL_BASE:=None}"                          # base model (only for LoRA)
: "${CONV_MODE:=vicuna_v1}"                      # 13B uses the same conv as 7B

# --- Dataset selection ------------------------------------------------------
# Which POPE datasets to run. Default is AOKVQA only. Override to run more, e.g.
#   export DATASETS="coco gqa aokvqa"
#   export POPE_DATASET="coco"
# DATASETS drives the multi-dataset sweeps (amateur-logits, attention).
# POPE_DATASET is the single dataset used for the POPE Table-3 pipeline
# (baseline / vcd / icd / sid / pba / olm + eval).
: "${DATASETS:=aokvqa}"
: "${POPE_DATASET:=aokvqa}"

# --- Image datasets ---------------------------------------------------------
# COCO val2014 also serves AOKVQA (AOKVQA reuses COCO images).
: "${COCO_IMAGES:=${STORAGE_ROOT}/datasets/coco/val2014}"
: "${GQA_IMAGES:=${STORAGE_ROOT}/datasets/gqa/images}"

# --- Repo-local paths -------------------------------------------------------
: "${DATA_DIR:=${REPO_ROOT}/data}"               # POPE question JSONs (gitignored)
: "${OUT_ROOT:=${REPO_ROOT}/repro_outputs}"      # inference + eval outputs
: "${LOG_DIR:=${OUT_ROOT}/logs}"                 # per-step logs from run_master

# --- Optional: Qwen-2.5-VL-7B track (separate model, off by default) --------
: "${QWEN_MODEL:=${STORAGE_ROOT}/model/Qwen2.5_7b}"

# Model label used in output directory / file names (keeps 13B results separate
# from any committed 7B results).
: "${MODEL_TAG:=llava-v1.5-13b}"
: "${FILE_TAG:=llava-13b}"

export HF_MODEL_ID MODEL_13B MODEL_BASE CONV_MODE DATASETS POPE_DATASET
export COCO_IMAGES GQA_IMAGES DATA_DIR OUT_ROOT LOG_DIR QWEN_MODEL MODEL_TAG FILE_TAG

# image_folder_for <dataset> -> prints the image directory for that dataset.
image_folder_for() {
    case "$1" in
        coco|aokvqa) echo "${COCO_IMAGES}" ;;
        gqa)         echo "${GQA_IMAGES}" ;;
        *) echo "[config] unknown dataset: $1" >&2; return 1 ;;
    esac
}
