#!/bin/bash
# Run LLaVA-Bench Gemini review + save results table directly on the login node.
# No GPU required — this is only API calls to Gemini.
#
# Usage:
#   bash scripts/run_llava_bench_review.sh

cd /mnt/home2/home/ankur_d/sm/cd-rethinking

LOG_FILE="outputs/llava_bench/logs/run_review_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs/llava_bench/logs

exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== LLaVA-Bench Gemini review started at $(date) ====="
echo "Logging to: $LOG_FILE"
echo ""

# ── Load API key ──────────────────────────────────────────────────────────────
CREDS="${HOME}/.llava_bench_credentials"
[ -f "${CREDS}" ] && source "${CREDS}"

if [ -z "${GEMINI_API_KEY}" ]; then
    echo "ERROR: GEMINI_API_KEY not set."
    echo "Run once:  echo 'export GEMINI_API_KEY=\"AIza...\"' > ~/.llava_bench_credentials"
    exit 1
fi

# ── Activate env ─────────────────────────────────────────────────────────────
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate cd_rethink

BENCH=./data/llava_bench
OUT=./outputs/llava_bench

# ── Step 2: Gemini review for each method ────────────────────────────────────
for METHOD in greedy vcd sid; do
    ANS="${OUT}/answers_${METHOD}.jsonl"
    REVIEW="${OUT}/review_${METHOD}.jsonl"

    if [ ! -f "${ANS}" ]; then
        echo "[skip] ${ANS} not found"
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
        --gemini-model gemini-2.5-flash-lite
    echo ""
done

# ── Step 3: Per-method score summary ─────────────────────────────────────────
echo "===== Per-method scores ====="
for METHOD in greedy vcd sid; do
    REVIEW="${OUT}/review_${METHOD}.jsonl"
    if [ -f "${REVIEW}" ]; then
        echo "----- ${METHOD} -----"
        python eval/llava_bench_summarize.py --review "${REVIEW}"
        echo ""
    fi
done

# ── Step 4: Save Table 4 ─────────────────────────────────────────────────────
echo "----- Saving Table 4 (LLaVA-Bench) -----"
python eval/save_llava_bench_table.py \
    --results-dir "${OUT}" \
    --methods greedy vcd sid

echo ""
echo "===== Done at $(date) ====="
