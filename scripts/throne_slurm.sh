#!/bin/bash
#SBATCH --job-name=throne_cd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/throne/slurm_throne_%j.log
#SBATCH --error=outputs/throne/slurm_throne_%j.log

# ──────────────────────────────────────────────────────────────────────────────
# THRONE object-hallucination benchmark for contrastive-decoding methods.
#
# Runs the SAME pipeline as scripts/throne_run_all.sh:
#   Step 0: download LLaVA-v1.5-7b + COCO val2017 (skipped if present)
#   Step 1: generate captions for each method (none, vcd, icd, sid, apc)
#           — checkpointed every 100 images, fully resumable
#   Step 2: Flan-T5 AQA evaluation (base/large/xl) per method
#   Step 3: P/R/F1/F0.5 scoring per method
#
# RESUME: just re-submit this script. Step 1 reads each method's
#   responses.ckpt.json and skips already-generated images; finished methods
#   (responses.json present) are skipped entirely.
#
# Submit with:   sbatch scripts/throne_slurm.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Activate environment ──────────────────────────────────────────────────────
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate cd_rethink

# ── Repo root ─────────────────────────────────────────────────────────────────
# Edit this to your cluster checkout path.
REPO_ROOT="${REPO_ROOT:-/mnt/home2/home/ankur_d/sm/cd-rethinking}"
cd "${REPO_ROOT}"

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p outputs/throne/logs
LOG_FILE="outputs/throne/logs/run_throne_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "===== THRONE experiment started at $(date) ====="
echo "Logging to: ${LOG_FILE}"
echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

# ── Paths ─────────────────────────────────────────────────────────────────────
export MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/llava-v1.5-7b}"
COCO_ROOT="${REPO_ROOT}/data/coco"
OUT_ROOT="${REPO_ROOT}/outputs/throne"
THRONE_DIR="${REPO_ROOT}/third_party/THRONE/throne"
THRONE_SCRIPT="${THRONE_DIR}/throne_generate_cd.py"
ANNO_FILE="${COCO_ROOT}/annotations/instances_val2017.json"
IMG_DIR="${COCO_ROOT}/val2017"
export REPO_ROOT

# inference/ + THRONE on PYTHONPATH so throne_generate_cd.py resolves its imports.
export PYTHONPATH="${REPO_ROOT}/inference:${THRONE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# Quiet the transformers/hub deprecation spam in the log.
export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning"

echo "Model : ${MODEL_DIR}"
echo "COCO  : ${COCO_ROOT}"
echo "Output: ${OUT_ROOT}"
echo ""

# ── Step 0a: download LLaVA-v1.5-7b ───────────────────────────────────────────
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo ">>> Downloading llava-v1.5-7b from HuggingFace …"
    python - << PYEOF
import os
from huggingface_hub import snapshot_download
model_dir = os.environ["MODEL_DIR"]
os.makedirs(model_dir, exist_ok=True)
snapshot_download(
    repo_id="liuhaotian/llava-v1.5-7b",
    local_dir=model_dir,
    ignore_patterns=["*.md", "*.txt"],
)
print(f"Model saved to {model_dir}")
PYEOF
else
    echo ">>> Model already present at ${MODEL_DIR}"
fi

# ── Step 0b: download COCO val2017 ────────────────────────────────────────────
if [ ! -f "${ANNO_FILE}" ]; then
    echo ">>> Downloading COCO annotations (~241 MB) …"
    mkdir -p "${COCO_ROOT}"
    curl -L "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
         -o "${COCO_ROOT}/annotations.zip"
    unzip -q "${COCO_ROOT}/annotations.zip" -d "${COCO_ROOT}"
    rm "${COCO_ROOT}/annotations.zip"
fi
if [ ! -d "${IMG_DIR}" ] || [ -z "$(ls -A "${IMG_DIR}" 2>/dev/null)" ]; then
    echo ">>> Downloading COCO val2017 images (~778 MB) …"
    curl -L "http://images.cocodataset.org/zips/val2017.zip" \
         -o "${COCO_ROOT}/val2017.zip"
    unzip -q "${COCO_ROOT}/val2017.zip" -d "${COCO_ROOT}"
    rm "${COCO_ROOT}/val2017.zip"
fi
echo ">>> Data ready."
echo ""

# ── Step 1: generate captions for each CD method (checkpointed / resumable) ───
for METHOD in none vcd icd sid apc; do
    OUT_DIR="${OUT_ROOT}/${METHOD}"
    mkdir -p "${OUT_DIR}"

    echo ">>> Step 1 [${METHOD}]: generating captions …  $(date)"
    python "${THRONE_SCRIPT}" \
        --coco_file        "${ANNO_FILE}" \
        --coco_image_dir   "${IMG_DIR}" \
        --save_path        "${OUT_DIR}/responses.json" \
        --per_device_batch_size 1 \
        LLaVA_CD \
            --model_path          "${MODEL_DIR}" \
            --conv_template_name  vicuna_v1 \
            --cd_method           "${METHOD}"
    echo ">>> Step 1 [${METHOD}]: done.  $(date)"
    echo ""
done

# ── Step 2: AQA evaluation (Flan-T5 ensemble) ─────────────────────────────────
EVALUATORS=("google/flan-t5-base" "google/flan-t5-large" "google/flan-t5-xl")

for METHOD in none vcd icd sid apc; do
    RESPONSES="${OUT_ROOT}/${METHOD}/responses.json"
    EVAL_DIR="${OUT_ROOT}/${METHOD}/eval"
    mkdir -p "${EVAL_DIR}"

    if [ ! -f "${RESPONSES}" ]; then
        echo "[skip] ${RESPONSES} not found — generation incomplete for ${METHOD}"
        continue
    fi

    for EVALUATOR in "${EVALUATORS[@]}"; do
        EVAL_FILE="${EVAL_DIR}/aqa_answers_${EVALUATOR//\//_}.json"
        if [ -f "${EVAL_FILE}" ]; then
            echo ">>> Step 2 [${METHOD}/${EVALUATOR}]: already done — skipping"
            continue
        fi
        echo ">>> Step 2 [${METHOD}/${EVALUATOR}]: running AQA eval …  $(date)"
        LOCAL_RANK=0 WORLD_SIZE=1 RANK=0 \
        python "${THRONE_DIR}/throne_aqa_evaluation.py" \
            --response_file        "${RESPONSES}" \
            --coco_file            "${ANNO_FILE}" \
            --evaluator_model_path "${EVALUATOR}" \
            --save_path            "${EVAL_DIR}/aqa_answers.json" \
            --M                    3 \
            --per_device_batch_size 8
        echo ">>> Step 2 [${METHOD}/${EVALUATOR}]: done.  $(date)"
    done
    echo ""
done

# ── Step 3: scoring (P/R/F1/F0.5) ─────────────────────────────────────────────
echo "===== THRONE SCORES ====="
for METHOD in none vcd icd sid apc; do
    EVAL_DIR="${OUT_ROOT}/${METHOD}/eval"
    SCORE_PATH="${EVAL_DIR}"
    [ -d "${EVAL_DIR}/combined" ] && SCORE_PATH="${EVAL_DIR}/combined"

    if [ ! -d "${SCORE_PATH}" ] || [ -z "$(ls -A "${SCORE_PATH}"/*.json 2>/dev/null)" ]; then
        echo "[skip] no eval outputs for ${METHOD}"
        continue
    fi

    echo "----- ${METHOD} -----"
    python "${THRONE_DIR}/throne_score_aqa.py" \
        --model_eval_path "${SCORE_PATH}" \
        --coco_file       "${ANNO_FILE}" \
        | tee "${OUT_ROOT}/${METHOD}/scores.txt"
    echo ""
done

echo "===== THRONE experiment finished at $(date) ====="
