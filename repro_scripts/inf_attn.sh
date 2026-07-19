#!/bin/bash
# LLaVA-v1.5-13B attention / visual-grounding experiment (VCD, ICD, SID)
# on POPE coco/gqa/aokvqa. Logs per-layer attention mass over image tokens.
# Outputs: attn/{ans,eval,warnings}/<dataset>/<method>/<dataset>-<type>.jsonl
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${REPO_ROOT}"

datasets=(${DATASETS})
types=(random popular adversarial)
methods=(vcd icd sid)

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        for type in "${types[@]}"; do
            echo "[attn] method=${method} dataset=${dataset} type=${type}"
            "${PY_BIN}" ./inference/attention_exp_inf.py \
                --model-path "${MODEL_13B}" \
                --question-file "${DATA_DIR}/${dataset}/${dataset}_pope_${type}.json" \
                --image-folder "$(image_folder_for "${dataset}")" \
                --answers-file "${REPO_ROOT}/attn/ans/${dataset}/${method}/${dataset}-${type}.jsonl" \
                --attn-file "${REPO_ROOT}/attn/eval/${dataset}/${method}/${dataset}-${type}.jsonl" \
                --warnings-file "${REPO_ROOT}/attn/warnings/${dataset}/${method}/${dataset}-${type}.jsonl" \
                --conv-mode "${CONV_MODE}" \
                --use-${method}
        done
    done
done
