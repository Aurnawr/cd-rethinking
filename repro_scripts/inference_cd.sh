#!/bin/bash
# LLaVA-v1.5-13B contrastive-decoding inference on POPE-COCO: VCD, ICD, SID.
# Outputs: repro_outputs/coco/llava-v1.5-13b/{vcd,icd,sid}/llava-13b-coco-<type>-greedy.jsonl
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${REPO_ROOT}"

datasets=(${POPE_DATASET})
types=(random popular adversarial)
methods=(vcd icd sid)

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        for type in "${types[@]}"; do
            echo "[cd] method=${method} dataset=${dataset} type=${type}"
            "${PY_BIN}" ./inference/pope_infer_cd.py \
                --model-path "${MODEL_13B}" \
                --question-file "${DATA_DIR}/${dataset}/${dataset}_pope_${type}.json" \
                --image-folder "$(image_folder_for "${dataset}")" \
                --answers-file "${OUT_ROOT}/${dataset}/${MODEL_TAG}/${method}/${FILE_TAG}-${dataset}-${type}-greedy.jsonl" \
                --temperature 0 \
                --conv-mode "${CONV_MODE}" \
                --use-${method}
        done
    done
done
