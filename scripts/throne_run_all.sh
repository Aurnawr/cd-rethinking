#!/usr/bin/env bash
# Full THRONE experiment pipeline — downloads model + COCO then runs all steps.
# Designed to run inside a tmux session:
#   tmux new -s throne
#   bash scripts/throne_run_all.sh 2>&1 | tee outputs/throne/run.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export MODEL_DIR="${MODEL_DIR:-/teamspace/studios/this_studio/models/llava-v1.5-7b}"
COCO_ROOT="${REPO_ROOT}/data/coco"
OUT_ROOT="${REPO_ROOT}/outputs/throne"
export REPO_ROOT

export PYTHONPATH="${REPO_ROOT}/inference:${REPO_ROOT}/third_party/THRONE/throne${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning"

echo "======================================================="
echo " THRONE EXPERIMENT — $(date)"
echo " Model : ${MODEL_DIR}"
echo " COCO  : ${COCO_ROOT}"
echo " Output: ${OUT_ROOT}"
echo "======================================================="

# ── Step 0a: download LLaVA-v1.5-7b ────────────────────────────────────────
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo ""
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

# ── Step 0b: download COCO val2017 ──────────────────────────────────────────
ANNO_FILE="${COCO_ROOT}/annotations/instances_val2017.json"
IMG_DIR="${COCO_ROOT}/val2017"

if [ ! -f "${ANNO_FILE}" ]; then
    echo ""
    echo ">>> Downloading COCO annotations (~121 MB) …"
    mkdir -p "${COCO_ROOT}"
    curl -L "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
         -o "${COCO_ROOT}/annotations.zip"
    unzip -q "${COCO_ROOT}/annotations.zip" -d "${COCO_ROOT}"
    rm "${COCO_ROOT}/annotations.zip"
fi

if [ ! -d "${IMG_DIR}" ] || [ -z "$(ls -A "${IMG_DIR}" 2>/dev/null)" ]; then
    echo ""
    echo ">>> Downloading COCO val2017 images (~1 GB) …"
    curl -L "http://images.cocodataset.org/zips/val2017.zip" \
         -o "${COCO_ROOT}/val2017.zip"
    unzip -q "${COCO_ROOT}/val2017.zip" -d "${COCO_ROOT}"
    rm "${COCO_ROOT}/val2017.zip"
fi

echo ""
echo ">>> Data ready.  Model: ${MODEL_DIR}  COCO: ${COCO_ROOT}"

# ── Step 1: generate captions for each CD method ────────────────────────────
THRONE_SCRIPT="${REPO_ROOT}/third_party/THRONE/throne/throne_generate_cd.py"

for METHOD in none vcd icd sid apc; do
    OUT_DIR="${OUT_ROOT}/${METHOD}"
    mkdir -p "${OUT_DIR}"

    echo ""
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
done

# ── Step 2: AQA evaluation (Flan-T5 ensemble) ───────────────────────────────
EVALUATORS=("google/flan-t5-base" "google/flan-t5-large" "google/flan-t5-xl")
THRONE_DIR="${REPO_ROOT}/third_party/THRONE/throne"

for METHOD in none vcd icd sid apc; do
    RESPONSES="${OUT_ROOT}/${METHOD}/responses.json"
    EVAL_DIR="${OUT_ROOT}/${METHOD}/eval"
    mkdir -p "${EVAL_DIR}"

    for EVALUATOR in "${EVALUATORS[@]}"; do
        # throne_aqa_evaluation.py outputs: aqa_answers_{model_name}.json
        EVAL_FILE="${EVAL_DIR}/aqa_answers_${EVALUATOR//\//_}.json"
        if [ -f "${EVAL_FILE}" ]; then
            echo ">>> Step 2 [${METHOD}/${EVALUATOR}]: already exists — skipping"
            continue
        fi
        echo ""
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
done

# ── Step 3: score ────────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo " THRONE SCORES"
echo "======================================================="

for METHOD in none vcd icd sid apc; do
    EVAL_DIR="${OUT_ROOT}/${METHOD}/eval"
    SCORE_PATH="${EVAL_DIR}"
    [ -d "${EVAL_DIR}/combined" ] && SCORE_PATH="${EVAL_DIR}/combined"

    echo ""
    echo "--- ${METHOD} ---"
    python "${THRONE_DIR}/throne_score_aqa.py" \
        --model_eval_path "${SCORE_PATH}" \
        --coco_file       "${ANNO_FILE}" \
        | tee "${OUT_ROOT}/${METHOD}/scores.txt"
done

echo ""
echo "======================================================="
echo " ALL DONE — $(date)"
echo "======================================================="
