#!/usr/bin/env bash
# CHAIR step 2: score generated captions for object hallucination.
#
# Prints CHAIR-S / CHAIR-I and writes <cap>_chair.json next to the captions.
#
# Override via environment, e.g.:
#   CAP_FILE=outputs/chair/llava-7b-greedy/captions.jsonl bash scripts/chair_eval.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG=${TAG:-llava-7b-greedy}
CAP_FILE=${CAP_FILE:-"${REPO_ROOT}/outputs/chair/${TAG}/captions.jsonl"}
COCO_PATH=${COCO_PATH:-"${REPO_ROOT}/data/coco/annotations"}

python "${REPO_ROOT}/eval/chair_eval.py" \
    --cap-file  "${CAP_FILE}" \
    --coco-path "${COCO_PATH}"
