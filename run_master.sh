#!/bin/bash
# ===========================================================================
# run_master.sh -- run the entire LLaVA-v1.5-13B reproduction end to end.
#
# By default it runs EVERYTHING: baseline + VCD/ICD/SID + PBA/OLM inference on
# POPE-COCO, the POPE Table-3 eval, the amateur-logit audit + eval, the
# attention/visual-grounding experiment, and the KLD experiment + plots.
#
# Each step logs to repro_outputs/logs/<step>.log and runs in isolation: if one
# fails the rest continue, and a status summary is printed at the end.
#
# Usage:
#   bash run_master.sh                # everything (assumes setup.sh already ran)
#   bash run_master.sh --setup        # run setup.sh first, then everything
#   bash run_master.sh --skip-attn --skip-kld   # just the POPE + amateur tables
#   bash run_master.sh --only pope    # only baseline+cd+spurious+eval
#
# Flags:
#   --setup           run setup.sh (deps/model/data/images) before experiments
#   --skip-inference  reuse existing inference outputs; run evals/plots only
#   --skip-attn       skip the (heavy) attention experiment
#   --skip-kld        skip the (heavy, downloads VisIT-Bench) KLD experiment
#   --with-qwen       also run the Qwen-2.5-VL-7B baseline/spurious track
#   --only <group>    run one group only: pope | amateur | attn | kld
# ===========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/repro_scripts/config.sh"
cd "${REPO_ROOT}"
RS="${REPO_ROOT}/repro_scripts"

# ---- flags ----------------------------------------------------------------
DO_SETUP=0; SKIP_INF=0; SKIP_ATTN=0; SKIP_KLD=0; WITH_QWEN=0; ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --setup)          DO_SETUP=1 ;;
        --skip-inference) SKIP_INF=1 ;;
        --skip-attn)      SKIP_ATTN=1 ;;
        --skip-kld)       SKIP_KLD=1 ;;
        --with-qwen)      WITH_QWEN=1 ;;
        --only)           ONLY="${2:-}"; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[master] unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

want() {  # want <group> -> true if this group should run
    [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]
}

# ---- logging + step runner ------------------------------------------------
mkdir -p "${LOG_DIR}"
info() { printf '\n\033[1;34m[master]\033[0m %s\n' "$*"; }
okln() { printf '\033[1;32m[master]\033[0m %s\n' "$*"; }
wrn()  { printf '\033[1;33m[master]\033[0m %s\n' "$*" >&2; }

declare -a STEP_ORDER=()
declare -A STEP_STATUS=()

run_step() {  # run_step <name> <cmd...>
    local name="$1"; shift
    local logf="${LOG_DIR}/${name}.log"
    STEP_ORDER+=("$name")
    info "START ${name}  (log: ${logf})"
    local start=$SECONDS
    if "$@" >"$logf" 2>&1; then
        local dt=$((SECONDS - start))
        STEP_STATUS["$name"]="OK (${dt}s)"
        okln "DONE  ${name}  (${dt}s)"
    else
        local rc=$?
        STEP_STATUS["$name"]="FAILED rc=${rc} (tail: ${logf})"
        wrn "FAILED ${name} (rc=${rc}) -- continuing. Last lines:"
        tail -n 5 "$logf" | sed 's/^/    /' >&2 || true
    fi
}

# ---------------------------------------------------------------------------
# 0. Optional setup + preflight
# ---------------------------------------------------------------------------
if [ "$DO_SETUP" -eq 1 ]; then
    info "Running setup.sh"
    bash "${SCRIPT_DIR}/setup.sh" || { wrn "setup.sh reported problems -- review before continuing"; }
fi

preflight_ok=1
[ -f "${MODEL_13B}/config.json" ] || { wrn "model missing at ${MODEL_13B} (run with --setup)"; preflight_ok=0; }
[ -s "${DATA_DIR}/coco/coco_pope_random.json" ] || { wrn "POPE data missing in ${DATA_DIR} (run with --setup)"; preflight_ok=0; }
if [ -n "$(find "${COCO_IMAGES}" -maxdepth 1 -name '*.jpg' -print -quit 2>/dev/null)" ]; then :; else
    wrn "COCO images missing at ${COCO_IMAGES}"; preflight_ok=0; fi

if [ "$SKIP_INF" -eq 0 ] && [ "$preflight_ok" -eq 0 ]; then
    die_msg="Preflight failed and inference is requested. Fix the above (e.g. bash run_master.sh --setup) or pass --skip-inference."
    wrn "$die_msg"; exit 1
fi

START_TS=$SECONDS

# ---------------------------------------------------------------------------
# 1. POPE Table-3 group: baseline + CD (vcd/icd/sid) + spurious (pba/olm) + eval
# ---------------------------------------------------------------------------
if want pope; then
    if [ "$SKIP_INF" -eq 0 ]; then
        run_step "pope_baseline" bash "${RS}/inference_base_llava.sh"
        run_step "pope_cd"       bash "${RS}/inference_cd.sh"
        run_step "pope_spurious" bash "${RS}/inference_spurious_llava.sh"
    fi
    run_step "pope_eval" bash "${RS}/eval_base.sh"
fi

# ---------------------------------------------------------------------------
# 2. Amateur-logit audit + eval
# ---------------------------------------------------------------------------
if want amateur; then
    [ "$SKIP_INF" -eq 0 ] && run_step "amateur_infer" bash "${RS}/inf_amateur_logits.sh"
    run_step "amateur_eval" bash "${RS}/llava_eval_amateur_deltas.sh"
fi

# ---------------------------------------------------------------------------
# 3. Attention / visual-grounding experiment  (HEAVY: output_attentions)
# ---------------------------------------------------------------------------
if want attn && [ "$SKIP_ATTN" -eq 0 ]; then
    [ "$SKIP_INF" -eq 0 ] && run_step "attn_infer" bash "${RS}/inf_attn.sh"
    run_step "attn_eval" python "${REPO_ROOT}/eval/flipped_attn.py"
elif [ "$SKIP_ATTN" -eq 1 ]; then
    wrn "skipping attention experiment (--skip-attn)"
fi

# ---------------------------------------------------------------------------
# 4. KLD experiment + plots  (HEAVY: downloads + iterates VisIT-Bench)
# ---------------------------------------------------------------------------
if want kld && [ "$SKIP_KLD" -eq 0 ]; then
    [ "$SKIP_INF" -eq 0 ] && run_step "kld_infer" bash "${RS}/run_kld.sh"
    run_step "kld_plot"     bash "${RS}/plot_kld.sh"
    run_step "kld_coverage" bash "${RS}/analyse_kld_coverage.sh"
elif [ "$SKIP_KLD" -eq 1 ]; then
    wrn "skipping KLD experiment (--skip-kld)"
fi

# ---------------------------------------------------------------------------
# 5. Optional Qwen-2.5-VL-7B track
# ---------------------------------------------------------------------------
if [ "$WITH_QWEN" -eq 1 ]; then
    if [ -f "${QWEN_MODEL}/config.json" ]; then
        run_step "qwen_baseline" bash "${RS}/inference_baseline.sh"
        run_step "qwen_spurious" bash "${RS}/inference_spurious.sh"
    else
        wrn "--with-qwen requested but no model at ${QWEN_MODEL} (skipping; run setup.sh --with-qwen)"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$((SECONDS - START_TS))
info "==================== RUN SUMMARY ===================="
for name in "${STEP_ORDER[@]}"; do
    printf '  %-16s %s\n' "$name" "${STEP_STATUS[$name]}"
done
printf '  %-16s %s\n' "total_time" "${TOTAL}s"
echo
echo "Result artifacts:"
echo "  ${OUT_ROOT}/table3_results.csv                 (POPE accuracy/F1/Yes-ratio)"
echo "  ${OUT_ROOT}/llava_amateur_deltas_results.csv   (amateur logit deltas)"
echo "  ${REPO_ROOT}/attn/visual_grounding_flipped_metrics.csv   (attention flips)"
echo "  ${OUT_ROOT}/kld_experiment/visit-bench/*.png   (KLD plots)"
echo
# Non-zero exit if any step failed, so CI / callers notice.
for name in "${STEP_ORDER[@]}"; do
    case "${STEP_STATUS[$name]}" in FAILED*) exit 1 ;; esac
done
okln "All requested steps completed successfully."
