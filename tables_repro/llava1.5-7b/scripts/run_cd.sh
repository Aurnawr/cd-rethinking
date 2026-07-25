#!/usr/bin/env bash
# Contrastive decoding on POPE: VCD, ICD, SID.
# Env: MODEL_PATH, GQA_IMAGE_DIR, COCO_IMAGE_DIR   (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?set MODEL_PATH to the LLaVA-1.5-7B weights}"
: "${GQA_IMAGE_DIR:?set GQA_IMAGE_DIR}"
: "${COCO_IMAGE_DIR:?set COCO_IMAGE_DIR}"

datasets=(gqa coco aokvqa)
types=(random popular adversarial)
img_dir () { case "$1" in gqa) echo "$GQA_IMAGE_DIR";; *) echo "$COCO_IMAGE_DIR";; esac; }

# method -> flag
declare -A FLAG=( [vcd]="--use-vcd" [icd]="--use-icd" [sid]="--use-sid" )

for method in vcd icd sid; do
  for dataset in "${datasets[@]}"; do
    for type in "${types[@]}"; do
      python ./inference/pope_infer_cd.py \
        --model-path "$MODEL_PATH" \
        --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
        --image-folder "$(img_dir "$dataset")" \
        --answers-file ./outputs/pope/${method}/llava-7b-${dataset}-${type}-greedy.jsonl \
        --temperature 0 --conv-mode vicuna_v1 ${FLAG[$method]}
    done
  done
done
