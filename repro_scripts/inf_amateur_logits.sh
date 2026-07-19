#!/bin/bash
# LLaVA-v1.5-13B amateur-logit audit (VCD & ICD) on POPE coco/gqa/aokvqa.
# Records the amateur branch's Yes/No logit deltas for each question.
# Outputs: llava_amateur_logits/<dataset>/{vcd,icd}/<dataset>-<type>-amateur-deltas.jsonl
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${REPO_ROOT}"

datasets=(${DATASETS})
types=(random popular adversarial)
methods=(vcd icd)

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        for type in "${types[@]}"; do
            echo "[amateur] method=${method} dataset=${dataset} type=${type}"
            "${PY_BIN}" ./inference/llava_amateur_logits.py \
                --model-path "${MODEL_13B}" \
                --question-file "${DATA_DIR}/${dataset}/${dataset}_pope_${type}.json" \
                --image-folder "$(image_folder_for "${dataset}")" \
                --answers-file "${REPO_ROOT}/llava_amateur_logits/${dataset}/${method}/${dataset}-${type}-amateur-deltas.jsonl" \
                --conv-mode "${CONV_MODE}" \
                --use-${method}
        done
    done
done
