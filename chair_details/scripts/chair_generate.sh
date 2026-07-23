#!/usr/bin/env bash
# CHAIR step 1: generate free-form COCO captions with LLaVA-v1.5-7b.
#
# Supports baseline greedy decoding and the VCD / SID contrastive-decoding
# methods. Writes outputs/chair/<TAG>/captions.jsonl as {"image_id", "caption"}
# lines. Generation appends, so re-running resumes (skips finished ids).
#
# Select the method with CD_METHOD (none|vcd|sid). VCD/SID are sampling-based,
# so they default to temperature=1, top_p=1; greedy uses temperature=0.
#
# Examples:
#   CD_METHOD=none bash scripts/chair_generate.sh
#   CD_METHOD=vcd  bash scripts/chair_generate.sh
#   CD_METHOD=sid  bash scripts/chair_generate.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH=${MODEL_PATH:-"${REPO_ROOT}/models/llava-v1.5-7b"}
CONV_MODE=${CONV_MODE:-vicuna_v1}
IMAGE_DIR=${IMAGE_DIR:-"${REPO_ROOT}/data/coco/val2017"}
ANNOTATION_FILE=${ANNOTATION_FILE:-"${REPO_ROOT}/data/coco/annotations/instances_val2017.json"}
NUM_SAMPLES=${NUM_SAMPLES:-500}
SEED=${SEED:-42}
CD_METHOD=${CD_METHOD:-none}

# Decoding defaults per method: greedy for baseline, sampling for VCD/SID.
if [ "${CD_METHOD}" = "none" ]; then
    TEMPERATURE=${TEMPERATURE:-0}
    TOP_P=${TOP_P:-}
    DEFAULT_TAG="llava-7b-greedy"
else
    TEMPERATURE=${TEMPERATURE:-1.0}
    TOP_P=${TOP_P:-1.0}
    DEFAULT_TAG="llava-7b-${CD_METHOD}"
fi
TAG=${TAG:-${DEFAULT_TAG}}
ANSWERS_FILE=${ANSWERS_FILE:-"${REPO_ROOT}/outputs/chair/${TAG}/captions.jsonl"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CMD=(python "${REPO_ROOT}/inference/chair_generate.py"
    --model-path      "${MODEL_PATH}"
    --image-folder    "${IMAGE_DIR}"
    --annotation-file "${ANNOTATION_FILE}"
    --answers-file    "${ANSWERS_FILE}"
    --num-samples     "${NUM_SAMPLES}"
    --seed            "${SEED}"
    --temperature     "${TEMPERATURE}"
    --conv-mode       "${CONV_MODE}"
    --cd-method       "${CD_METHOD}")
if [ -n "${TOP_P}" ]; then
    CMD+=(--top_p "${TOP_P}")
fi

"${CMD[@]}"

echo "Saved captions: ${ANSWERS_FILE}"
echo "Next: TAG=${TAG} bash scripts/chair_eval.sh"
