#!/bin/bash
#SBATCH --job-name=llava_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/llava_bench/slurm_%j.log
#SBATCH --error=outputs/llava_bench/slurm_%j.log

# Activate environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate cd_rethink

cd /mnt/home2/home/ankur_d/sm/cd-rethinking

# Log file with timestamp
LOG_FILE="outputs/llava_bench/logs/run_bench_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs/llava_bench/logs

# Redirect all stdout and stderr to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== LLaVA-Bench inference started at $(date) ====="
echo "Logging to: $LOG_FILE"
echo "Node: $(hostname)  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

MODEL=./pretrained_models/llava-v1.5-7b
BENCH_IMAGES=./data/llava_bench/images
BENCH_QS=./data/llava_bench/questions.jsonl
OUT=./outputs/llava_bench

# ── Step 1a: Greedy baseline ─────────────────────────────────────────────────
echo "----- Running Greedy baseline -----"
python eval/llava_bench_infer.py \
    --model-path  "${MODEL}" \
    --image-folder "${BENCH_IMAGES}" \
    --question-file "${BENCH_QS}" \
    --answers-file  "${OUT}/answers_greedy.jsonl" \
    --conv-mode vicuna_v1 \
    --temperature 0

# ── Step 1b: VCD (greedy, noise_step=900) ────────────────────────────────────
echo "----- Running VCD -----"
python eval/llava_bench_infer_cd.py \
    --model-path  "${MODEL}" \
    --image-folder "${BENCH_IMAGES}" \
    --question-file "${BENCH_QS}" \
    --answers-file  "${OUT}/answers_vcd.jsonl" \
    --conv-mode vicuna_v1 \
    --temperature 0 \
    --use-vcd \
    --noise-step 900

# ── Step 1c: SID (greedy) ─────────────────────────────────────────────────────
echo "----- Running SID -----"
python eval/llava_bench_infer_cd.py \
    --model-path  "${MODEL}" \
    --image-folder "${BENCH_IMAGES}" \
    --question-file "${BENCH_QS}" \
    --answers-file  "${OUT}/answers_sid.jsonl" \
    --conv-mode vicuna_v1 \
    --temperature 0 \
    --use-sid

echo ""
echo "===== Inference complete at $(date) ====="
echo ""

# ── Step 2 + 3: GPT-4 review + summarise (requires OPENAI_API_KEY) ───────────
# Uncomment and set OPENAI_API_KEY before running these steps.
#
# if [ -z "${OPENAI_API_KEY}" ]; then
#     echo "OPENAI_API_KEY not set; skipping GPT-4 review."
# else
#     for METHOD in greedy vcd sid; do
#         echo "----- GPT-4 review: ${METHOD} -----"
#         python eval/llava_bench_gpt_review.py \
#             --question    "${BENCH_QS}" \
#             --context     ./data/llava_bench/context.jsonl \
#             --rule        ./data/llava_bench/rule.json \
#             --answer-ref  ./data/llava_bench/answers_gpt4.jsonl \
#             --answer-model "${OUT}/answers_${METHOD}.jsonl" \
#             --output      "${OUT}/review_${METHOD}.jsonl"
#
#         echo "----- Scores: ${METHOD} -----"
#         python eval/llava_bench_summarize.py --review "${OUT}/review_${METHOD}.jsonl"
#         echo ""
#     done
# fi

echo "===== LLaVA-Bench job finished at $(date) ====="
