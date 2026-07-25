#!/usr/bin/env bash
# Score every method's answers against the POPE ground truth.
# Prints accuracy, precision/recall/F1, and yes-proportion per (method, dataset, split).
set -euo pipefail
cd "$(dirname "$0")/.."

methods=(baseline vcd icd sid pba olm)
datasets=(gqa coco aokvqa)
types=(random popular adversarial)

for method in "${methods[@]}"; do
  for dataset in "${datasets[@]}"; do
    for type in "${types[@]}"; do
      echo "=== $method / $dataset / $type ==="
      python ./eval/pope_eval_base.py \
        --ref-files ./data/${dataset}/${dataset}_pope_${type}.json \
        --res-files ./outputs/pope/${method}/llava-7b-${dataset}-${type}-greedy.jsonl
    done
  done
done
