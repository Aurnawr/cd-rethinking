#!/bin/bash
# LLaVA-v1.5-13B baseline (greedy) inference on POPE-COCO.
# Outputs: repro_outputs/coco/llava-v1.5-13b/baseline/llava-13b-coco-<type>-greedy.jsonl
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${REPO_ROOT}"

datasets=(coco)
types=(random popular adversarial)

for dataset in "${datasets[@]}"; do
    for type in "${types[@]}"; do
        echo "[baseline] dataset=${dataset} type=${type}"
        python ./inference/pope_infer_base.py \
            --model-path "${MODEL_13B}" \
            --question-file "${DATA_DIR}/${dataset}/${dataset}_pope_${type}.json" \
            --image-folder "$(image_folder_for "${dataset}")" \
            --answers-file "${OUT_ROOT}/${dataset}/${MODEL_TAG}/baseline/${FILE_TAG}-${dataset}-${type}-greedy.jsonl" \
            --temperature 0 \
            --conv-mode "${CONV_MODE}"
    done
done
