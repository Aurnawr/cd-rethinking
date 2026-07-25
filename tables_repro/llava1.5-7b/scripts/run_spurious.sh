#!/usr/bin/env bash
# Naive yes-inflating controls on POPE: PBA (prompt) and OLM (token-logit push).
# Env: MODEL_PATH, GQA_IMAGE_DIR, COCO_IMAGE_DIR   (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?set MODEL_PATH to the LLaVA-1.5-7B weights}"
: "${GQA_IMAGE_DIR:?set GQA_IMAGE_DIR}"
: "${COCO_IMAGE_DIR:?set COCO_IMAGE_DIR}"

datasets=(gqa coco aokvqa)
types=(random popular adversarial)
img_dir () { case "$1" in gqa) echo "$GQA_IMAGE_DIR";; *) echo "$COCO_IMAGE_DIR";; esac; }

for dataset in "${datasets[@]}"; do
  for type in "${types[@]}"; do
    python ./inference/pope_infer_pba.py \
      --model-path "$MODEL_PATH" \
      --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
      --image-folder "$(img_dir "$dataset")" \
      --answers-file ./outputs/pope/pba/llava-7b-${dataset}-${type}-greedy.jsonl \
      --temperature 0 --conv-mode vicuna_v1

    python ./inference/pope_infer_olm.py \
      --model-path "$MODEL_PATH" \
      --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
      --image-folder "$(img_dir "$dataset")" \
      --answers-file ./outputs/pope/olm/llava-7b-${dataset}-${type}-greedy.jsonl \
      --temperature 0 --conv-mode vicuna_v1 --use-olm
  done
done
