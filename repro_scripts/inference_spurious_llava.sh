#!/bin/bash
# LLaVA-v1.5-13B "spurious improvement" methods on POPE-COCO:
#   PBA (Prompt-Based Adjustment) and OLM (Output-Layer Modification).
# Outputs: repro_outputs/<dataset>/llava-v1.6-vicuna-13b/{pba,olm}/llava-16-13b-<dataset>-<type>-greedy.jsonl
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${REPO_ROOT}"

datasets=(${POPE_DATASET})
types=(random popular adversarial)

## PBA -- prompt-based, no extra flag
for dataset in "${datasets[@]}"; do
    for type in "${types[@]}"; do
        echo "[pba] dataset=${dataset} type=${type}"
        "${PY_BIN}" ./inference/pope_infer_pba.py \
            --model-path "${MODEL_13B}" \
            --question-file "${DATA_DIR}/${dataset}/${dataset}_pope_${type}.json" \
            --image-folder "$(image_folder_for "${dataset}")" \
            --answers-file "${OUT_ROOT}/${dataset}/${MODEL_TAG}/pba/${FILE_TAG}-${dataset}-${type}-greedy.jsonl" \
            --temperature 0 \
            --conv-mode "${CONV_MODE}"
    done
done

## OLM -- output-layer modification, needs --use-olm
for dataset in "${datasets[@]}"; do
    for type in "${types[@]}"; do
        echo "[olm] dataset=${dataset} type=${type}"
        "${PY_BIN}" ./inference/pope_infer_olm.py \
            --model-path "${MODEL_13B}" \
            --question-file "${DATA_DIR}/${dataset}/${dataset}_pope_${type}.json" \
            --image-folder "$(image_folder_for "${dataset}")" \
            --answers-file "${OUT_ROOT}/${dataset}/${MODEL_TAG}/olm/${FILE_TAG}-${dataset}-${type}-greedy.jsonl" \
            --temperature 0 \
            --conv-mode "${CONV_MODE}" \
            --use-olm
    done
done
