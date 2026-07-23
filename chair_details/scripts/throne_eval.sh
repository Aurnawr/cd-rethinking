#!/usr/bin/env bash
# THRONE steps 2 + 3: Flan-T5 AQA evaluation and P/R/F0.5 scoring.
#
# Step 2 (throne_aqa_evaluation.py):
#   For each method, runs three Flan-T5 evaluator models (base / large / xl)
#   in parallel (one torchrun per model). Each outputs a per-image answer JSON
#   to outputs/throne/<method>/eval/.
#
# Step 3 (throne_score_aqa.py):
#   Aggregates the three evaluator JSONs via majority voting (threshold 5/8/9)
#   and reports Precision / Recall / F1 / F0.5 (micro + classwise).
#
# Override defaults via environment:
#   OUT_ROOT=/my/outputs bash scripts/throne_eval.sh
#   NGPU=4 bash scripts/throne_eval.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COCO_FILE=${COCO_FILE:-"${REPO_ROOT}/data/coco/annotations/instances_val2017.json"}
OUT_ROOT=${OUT_ROOT:-"${REPO_ROOT}/outputs/throne"}
THRONE_DIR="${REPO_ROOT}/third_party/THRONE/throne"
NGPU=${NGPU:-1}

# Flan-T5 evaluator models (matching THRONE's one-click.sh defaults)
EVALUATORS=(
    "google/flan-t5-base"
    "google/flan-t5-large"
    "google/flan-t5-xl"
)

export PYTHONPATH="${THRONE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

run_aqa() {
    local method="$1"
    local responses_file="${OUT_ROOT}/${method}/responses.json"
    local eval_dir="${OUT_ROOT}/${method}/eval"
    mkdir -p "${eval_dir}"

    if [ ! -f "${responses_file}" ]; then
        echo "WARN: ${responses_file} not found — skipping method ${method}"
        return
    fi

    echo "=== THRONE AQA eval: method=${method} ==="
    for evaluator in "${EVALUATORS[@]}"; do
        echo "  evaluator: ${evaluator}"
        if [ "${NGPU}" -le 1 ]; then
            # Single-GPU: set required dist env vars for non-torchrun execution
            LOCAL_RANK=0 WORLD_SIZE=1 RANK=0 \
            python "${THRONE_DIR}/throne_aqa_evaluation.py" \
                --response_file   "${responses_file}" \
                --coco_file       "${COCO_FILE}" \
                --evaluator_model_path "${evaluator}" \
                --save_path       "${eval_dir}/aqa_answers.json" \
                --M               3 \
                --per_device_batch_size 8
        else
            torchrun --nproc_per_node "${NGPU}" \
                "${THRONE_DIR}/throne_aqa_evaluation.py" \
                --response_file   "${responses_file}" \
                --coco_file       "${COCO_FILE}" \
                --evaluator_model_path "${evaluator}" \
                --save_path       "${eval_dir}/aqa_answers.json" \
                --M               3 \
                --per_device_batch_size 8
        fi
    done
}

run_score() {
    local method="$1"
    local eval_dir="${OUT_ROOT}/${method}/eval"
    local combined_dir="${eval_dir}/combined"

    # throne_score_aqa.py globs *.json from model_eval_path; when torchrun was
    # used the combined/ subdir holds the merged files, otherwise they're in eval_dir.
    local score_path="${eval_dir}"
    [ -d "${combined_dir}" ] && score_path="${combined_dir}"

    echo "=== THRONE score: method=${method} ==="
    python "${THRONE_DIR}/throne_score_aqa.py" \
        --model_eval_path "${score_path}" \
        --coco_file       "${COCO_FILE}" \
        | tee "${OUT_ROOT}/${method}/scores.txt"
    echo "  Scores written to ${OUT_ROOT}/${method}/scores.txt"
}

for method in none vcd icd sid apc; do
    run_aqa   "${method}"
    run_score "${method}"
done

echo ""
echo "All THRONE evaluation runs complete."
echo "Results:"
for method in none vcd icd sid apc; do
    score_file="${OUT_ROOT}/${method}/scores.txt"
    if [ -f "${score_file}" ]; then
        echo "--- ${method} ---"
        cat "${score_file}"
    fi
done
