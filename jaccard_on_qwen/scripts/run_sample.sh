#!/usr/bin/env bash
# Direct sampling (do_sample=True, temperature=1.0, top_p=1.0)

set -e
cd "$(dirname "$0")/.."

MODEL_PATH=/teamspace/lightning_storage/model/Qwen2.5_7b
QUESTION_FILE=../jaccard_experiment/data/llava_bench/questions.jsonl
IMAGE_FOLDER=../jaccard_experiment/data/llava_bench/images
ANSWERS_FILE=./outputs/llava_bench/sample/qwen2.5-7b-llava_bench-sample.jsonl

python ./inference/qwen_bench_infer_sample.py \
    --model-path    "$MODEL_PATH"    \
    --question-file "$QUESTION_FILE" \
    --image-folder  "$IMAGE_FOLDER"  \
    --answers-file  "$ANSWERS_FILE"  \
    --temperature   1.0              \
    --top_p         1.0              \
    --min-pixels    $((256 * 28 * 28))  \
    --max-pixels    $((1280 * 28 * 28))

echo "Sample done. Output: $ANSWERS_FILE"
