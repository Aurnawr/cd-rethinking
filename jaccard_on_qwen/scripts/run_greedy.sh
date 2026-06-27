#!/usr/bin/env bash
# Greedy search (do_sample=False, num_beams=1)
# Image resolution: 256–1280 visual tokens per image (Qwen2.5-VL official recommendation)

set -e
cd "$(dirname "$0")/.."

MODEL_PATH=/teamspace/lightning_storage/model/Qwen2.5_7b
QUESTION_FILE=../jaccard_experiment/data/llava_bench/questions.jsonl
IMAGE_FOLDER=../jaccard_experiment/data/llava_bench/images
ANSWERS_FILE=./outputs/llava_bench/greedy/qwen2.5-7b-llava_bench-greedy.jsonl

python ./inference/qwen_bench_infer_greedy.py \
    --model-path    "$MODEL_PATH"    \
    --question-file "$QUESTION_FILE" \
    --image-folder  "$IMAGE_FOLDER"  \
    --answers-file  "$ANSWERS_FILE"  \
    --min-pixels    $((256 * 28 * 28))  \
    --max-pixels    $((1280 * 28 * 28))

echo "Greedy done. Output: $ANSWERS_FILE"
