#!/usr/bin/env bash
# THRONE step 1: generate free-form captions for all CD methods.
#
# Runs throne_generate_cd.py for each method:
#   none (baseline), vcd, icd, sid, apc
#
# Each method writes its responses to outputs/throne/<method>/responses.json
# in the THRONE format:
#   {"prompts": [...], "responses": [[prompt_idx, coco_id, text], ...]}
#
# Generation is checkpointed every 100 images (responses.ckpt.json); re-running
# resumes where it left off and skips methods whose responses.json exists.
#
# Override variables via environment:
#   MODEL_PATH=/path/to/llava-v1.5-7b bash scripts/throne_generate.sh
#
# Multi-GPU: set NGPU > 1 to use torchrun (default: single GPU).
#   NGPU=8 bash scripts/throne_generate.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH=${MODEL_PATH:-/teamspace/studios/this_studio/models/llava-v1.5-7b}
CONV_MODE=${CONV_MODE:-vicuna_v1}
COCO_FILE=${COCO_FILE:-"${REPO_ROOT}/data/coco/annotations/instances_val2017.json"}
COCO_IMAGE_DIR=${COCO_IMAGE_DIR:-"${REPO_ROOT}/data/coco/val2017"}
OUT_ROOT=${OUT_ROOT:-"${REPO_ROOT}/outputs/throne"}
NGPU=${NGPU:-1}

THRONE_SCRIPT="${REPO_ROOT}/third_party/THRONE/throne/throne_generate_cd.py"

# inference/ must be on PYTHONPATH so throne_generate_cd.py can import
# throne_cd_evaluatee and the cd_utils / spurious_utils packages.
export PYTHONPATH="${REPO_ROOT}/inference:${REPO_ROOT}/third_party/THRONE/throne${PYTHONPATH:+:${PYTHONPATH}}"

run_generate() {
    local method="$1"
    local out_dir="${OUT_ROOT}/${method}"
    mkdir -p "${out_dir}"
    echo "=== THRONE generate: method=${method} ==="

    if [ "${NGPU}" -le 1 ]; then
        python "${THRONE_SCRIPT}" \
            --coco_file        "${COCO_FILE}" \
            --coco_image_dir   "${COCO_IMAGE_DIR}" \
            --save_path        "${out_dir}/responses.json" \
            --per_device_batch_size 1 \
            LLaVA_CD \
                --model_path          "${MODEL_PATH}" \
                --conv_template_name  "${CONV_MODE}" \
                --cd_method           "${method}"
    else
        torchrun --nproc_per_node "${NGPU}" "${THRONE_SCRIPT}" \
            --coco_file        "${COCO_FILE}" \
            --coco_image_dir   "${COCO_IMAGE_DIR}" \
            --save_path        "${out_dir}/responses.json" \
            --per_device_batch_size 1 \
            LLaVA_CD \
                --model_path          "${MODEL_PATH}" \
                --conv_template_name  "${CONV_MODE}" \
                --cd_method           "${method}"
    fi

    echo "Saved: ${out_dir}/responses.json"
}

for method in none vcd icd sid apc; do
    run_generate "${method}"
done

echo ""
echo "All THRONE generation runs complete."
echo "Next: bash scripts/throne_eval.sh"
