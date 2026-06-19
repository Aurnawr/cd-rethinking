#!/bin/bash
#SBATCH --job-name=bench_review
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/llava_bench/slurm_review_%j.log
#SBATCH --error=outputs/llava_bench/slurm_review_%j.log

# Activate environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate cd_rethink

cd /mnt/home2/home/ankur_d/sm/cd-rethinking

# Log file with timestamp
LOG_FILE="outputs/llava_bench/logs/run_review_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs/llava_bench/logs

# Redirect all stdout and stderr to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== LLaVA-Bench Gemini review started at $(date) ====="
echo "Logging to: $LOG_FILE"
echo ""

# ── Load credentials ─────────────────────────────────────────────────────────
# Store your key in ~/.llava_bench_credentials (one-time setup):
#   echo 'export GEMINI_API_KEY="AIza..."' > ~/.llava_bench_credentials
#   chmod 600 ~/.llava_bench_credentials
CREDS_FILE="${HOME}/.llava_bench_credentials"
if [ -f "${CREDS_FILE}" ]; then
    source "${CREDS_FILE}"
fi

# ── Require GEMINI_API_KEY ─────────────────────────────────────────────────
if [ -z "${GEMINI_API_KEY}" ]; then
    echo "ERROR: GEMINI_API_KEY is not set."
    echo "Create ~/.llava_bench_credentials with:"
    echo "  echo 'export GEMINI_API_KEY=\"AIza...\"' > ~/.llava_bench_credentials"
    echo "  chmod 600 ~/.llava_bench_credentials"
    exit 1
fi

BENCH=./data/llava_bench
OUT=./outputs/llava_bench

# ── Step 2: Gemini review for each method ────────────────────────────────────
for METHOD in greedy vcd sid; do
    ANS="${OUT}/answers_${METHOD}.jsonl"
    REVIEW="${OUT}/review_${METHOD}.jsonl"

    if [ ! -f "${ANS}" ]; then
        echo "[skip] ${ANS} not found — inference not done for ${METHOD}"
        continue
    fi

    echo "----- Gemini review: ${METHOD} -----"
    python eval/llava_bench_gemini_review.py \
        --question     "${BENCH}/questions.jsonl" \
        --context      "${BENCH}/context.jsonl" \
        --rule         "${BENCH}/rule.json" \
        --answer-ref   "${BENCH}/answers_gpt4.jsonl" \
        --answer-model "${ANS}" \
        --output       "${REVIEW}" \
        --gemini-model gemini-1.5-flash
    echo ""
done

# ── Step 3: Per-method score summary ─────────────────────────────────────────
echo "===== Per-method score summary ====="
for METHOD in greedy vcd sid; do
    REVIEW="${OUT}/review_${METHOD}.jsonl"
    if [ -f "${REVIEW}" ]; then
        echo "----- ${METHOD} -----"
        python eval/llava_bench_summarize.py --review "${REVIEW}"
        echo ""
    fi
done

# ── Step 4: Save Table 4 (LLaVA-Bench column) ──────────────────────────────
echo "----- Saving Table 4 (LLaVA-Bench) -----"
python eval/save_llava_bench_table.py \
    --results-dir "${OUT}" \
    --methods greedy vcd sid

echo ""
echo "===== LLaVA-Bench review + table saved at $(date) ====="
