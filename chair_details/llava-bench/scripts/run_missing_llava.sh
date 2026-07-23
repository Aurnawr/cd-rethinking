#!/usr/bin/env bash
# Fill the 3 LLaVA-1.5-7B rows missing for Mirage Table 4 / Table 6.
# Requires an attached GPU (L4 ok: ~14GB weights + contrastive branch).
# max-new-tokens=512 to match the existing greedy/sample/vcd/apc outputs.
set -e
cd "$(dirname "$0")/.."

MODEL_PATH="./models/llava-v1.5-7b"
Q="./data/llava_bench/questions.jsonl"
IMG="./data/llava_bench/images"
INFER=".venv-llava/bin/python ./inference/llava_bench_infer_cd.py --model-path $MODEL_PATH --question-file $Q --image-folder $IMG --conv-mode vicuna_v1 --max-new-tokens 512"

echo "===== [1/3] VCD (greedy base) — Table 4 ====="
$INFER --use-vcd --temperature 0 --top_p 1.0 --noise-step 900 --cd-alpha 1.0 --cd-beta 0.1 \
       --answers-file ./outputs/llava/vcd_greedy.jsonl

echo "===== [2/3] SID (greedy base) — Table 4 ====="
$INFER --use-sid --temperature 0 --top_p 1.0 --cd-alpha 1.0 --cd-beta 0.1 \
       --answers-file ./outputs/llava/sid_greedy.jsonl

echo "===== [3/3] SID (sampling base) — Table 6 ====="
$INFER --use-sid --temperature 1.0 --top_p 1.0 --cd-alpha 1.0 --cd-beta 0.1 \
       --answers-file ./outputs/llava/sid_sample.jsonl

echo "All missing LLaVA runs complete."
