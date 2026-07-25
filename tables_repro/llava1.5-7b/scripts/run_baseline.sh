#!/usr/bin/env bash
# Baseline greedy decoding on POPE (GQA, COCO, AOKVQA x random/popular/adversarial).
# Env: MODEL_PATH, GQA_IMAGE_DIR, COCO_IMAGE_DIR   (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?set MODEL_PATH to the LLaVA-1.5-7B weights}"
: "${GQA_IMAGE_DIR:?set GQA_IMAGE_DIR to the GQA images folder}"
: "${COCO_IMAGE_DIR:?set COCO_IMAGE_DIR to the COCO val2014 images folder}"

datasets=(gqa coco aokvqa)
types=(random popular adversarial)
img_dir () { case "$1" in gqa) echo "$GQA_IMAGE_DIR";; *) echo "$COCO_IMAGE_DIR";; esac; }

for dataset in "${datasets[@]}"; do
  for type in "${types[@]}"; do
    python ./inference/pope_infer_base.py \
      --model-path "$MODEL_PATH" \
      --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
      --image-folder "$(img_dir "$dataset")" \
      --answers-file ./outputs/pope/baseline/llava-7b-${dataset}-${type}-greedy.jsonl \
      --temperature 0 --conv-mode vicuna_v1
  done
done
