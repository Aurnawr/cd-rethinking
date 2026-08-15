#!/usr/bin/env bash
# =============================================================================
#  Qwen2.5-VL CD mechanistic-interpretability battery - end-to-end driver
#  Target: a Lightning.ai Studio with a GPU attached (L4 / A10G / L40S / A100).
#
#  One command, bare Studio to finished figures:
#      bash mech_interp_qwen/run_all_lightning.sh
#
#  Smoke test first (~10 min, 40 questions/split) - ALWAYS do this before the full run:
#      MAX_SAMPLES=40 bash mech_interp_qwen/run_all_lightning.sh
#
#  Every stage is idempotent: weights, images and completed splits are skipped on re-run.
#  Everything lands under $ROOT (persistent Studio storage), so a restart loses nothing.
# =============================================================================
set -euo pipefail

# ----------------------------------------------------------------- configuration
ROOT="${ROOT:-/teamspace/studios/this_studio/cd_pope_mech_interp_qwen}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"

MODELS_DIR="${ROOT}/models"
DATA_DIR="${ROOT}/data"
ACT_DIR="${ROOT}/activations"
RESULTS_DIR="${ROOT}/results"
REGEN_DIR="${RESULTS_DIR}/regen"
VENV_DIR="${VENV_DIR:-${ROOT}/venv}"
LOG_DIR="${ROOT}/logs"

MAX_SAMPLES="${MAX_SAMPLES:-0}"          # 0 = all ~3000 questions per split
SPLITS="${SPLITS:-random popular adversarial}"
METHODS="${METHODS:-vcd icd sid}"
PRIMARY_METHOD="${PRIMARY_METHOD:-vcd}"  # whose residual curve goes in per_layer.csv
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-eager}"          # must stay eager: SID needs real attention weights
NOISE_STEP="${NOISE_STEP:-900}"          # VCD diffusion step, this repo's default
SID_RANK_LAYER="${SID_RANK_LAYER:-2}"    # 0-indexed; SID paper's 1-indexed "Layer i=3"
SID_KEEP_RATIO="${SID_KEEP_RATIO:-0.10}" # SID paper: keep the 10% least-attended vision tokens
SID_KEEP_TOKENS="${SID_KEEP_TOKENS:-}"   # optional absolute override (released-code parity: 100)
MAX_PIXELS="${MAX_PIXELS:-}"             # optional vision-token cap; empty = Qwen defaults
SHARD_SIZE="${SHARD_SIZE:-500}"
PROBE_JOBS="${PROBE_JOBS:--1}"   # CV parallelism; lower it if the probe stage runs out of RAM
USE_VENV="${USE_VENV:-1}"
RESUME="${RESUME:-1}"
STAGES="${STAGES:-env model data selfcheck extract probes resid figures package}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="${REPO_ROOT}/mech_interp_qwen"
cd "${REPO_ROOT}"

# Local path or hub id? A directory with a config.json is used as-is.
if [ -d "${MODEL_ID}" ] && [ -f "${MODEL_ID}/config.json" ]; then
    MODEL_PATH="${MODEL_ID}"
else
    MODEL_PATH="${MODELS_DIR}/$(basename "${MODEL_ID}")"
fi

mkdir -p "${MODELS_DIR}" "${DATA_DIR}" "${ACT_DIR}" "${RESULTS_DIR}" "${REGEN_DIR}" "${LOG_DIR}"

# Keep every HF cache on persistent storage, not the ephemeral root disk.
export HF_HOME="${HF_HOME:-${ROOT}/hf_home}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }
banner() { echo; echo "=============== $* ==============="; date '+%Y-%m-%d %H:%M:%S'; }

echo "=========================================================="
echo " repo         : ${REPO_ROOT}"
echo " root         : ${ROOT}"
echo " model        : ${MODEL_ID}"
echo " model path   : ${MODEL_PATH}"
echo " methods      : ${METHODS}"
echo " splits       : ${SPLITS}"
echo " max samples  : ${MAX_SAMPLES}  (0 = all)"
echo " dtype / attn : ${DTYPE} / ${ATTN_IMPL}"
echo " SID          : rank_layer=${SID_RANK_LAYER} keep_ratio=${SID_KEEP_RATIO} keep_tokens=${SID_KEEP_TOKENS:-<ratio>}"
echo " stages       : ${STAGES}"
echo "=========================================================="
nvidia-smi || echo "[warn] nvidia-smi unavailable - attach a GPU to this Studio before 'extract'"

# ----------------------------------------------------------------- stage: env
# A venv with --system-site-packages reuses the Studio's CUDA-matched torch (reinstalling torch
# from PyPI is the classic way to end up CPU-only) while keeping transformers==4.51.3 out of the
# base environment, where this repo's LLaVA stack pins transformers==4.31.0.
if [ "${USE_VENV}" = "1" ]; then
    PY="${VENV_DIR}/bin/python"
    if [ ! -x "${PY}" ] && ! has_stage env; then
        echo "[warn] ${PY} does not exist and the 'env' stage is not in STAGES - using system python3"
        PY="$(command -v python3)"
    fi
else
    PY="$(command -v python3)"
fi

if has_stage env; then
    banner "1. environment"
    if [ "${USE_VENV}" = "1" ] && [ ! -x "${PY}" ]; then
        echo "[env] creating venv (system site packages) at ${VENV_DIR}"
        python3 -m venv --system-site-packages "${VENV_DIR}"
    fi
    "${PY}" - <<'PYCHECK'
import sys
try:
    import torch
except ImportError:
    sys.exit("[env] FATAL: torch is not available. Attach a GPU Studio image that ships torch, "
             "or install a CUDA build matching this machine before rerunning.")
major, minor = (int(x) for x in torch.__version__.split(".")[:2])
if (major, minor) < (2, 1):
    sys.exit(f"[env] FATAL: torch {torch.__version__} is too old for transformers 4.51 (need >= 2.1)")
print(f"[env] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("[env] WARNING: no CUDA device visible - the extract stage will fail")
PYCHECK
    "${PY}" -m pip install --quiet --upgrade pip
    "${PY}" -m pip install --quiet -r "${HERE}/requirements.txt"
    "${PY}" -c "import transformers; print('[env] transformers', transformers.__version__)"
fi

# hf_transfer gives a much faster 16 GB download, but only if it actually imported.
if "${PY}" -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi

# ----------------------------------------------------------------- stage: model
if has_stage model; then
    banner "2. model weights"
    if [ -f "${MODEL_PATH}/config.json" ]; then
        echo "[model] already present: ${MODEL_PATH}"
    else
        "${PY}" "${HERE}/download_model.py" --repo-id "${MODEL_ID}" --target "${MODEL_PATH}"
    fi
fi

# ----------------------------------------------------------------- stage: data
if has_stage data; then
    banner "3. POPE-COCO questions + images"
    # Reused verbatim from the LLaVA experiment: pure requests/json, no model dependency.
    # Fetches the 3 COCO splits and only the unique val2014 images they reference.
    "${PY}" "${REPO_ROOT}/mech_interp/download_pope_data.py" --data-dir "${DATA_DIR}"
fi

# ----------------------------------------------------------------- shared flags
SID_FLAGS=(--sid-rank-layer "${SID_RANK_LAYER}" --sid-keep-ratio "${SID_KEEP_RATIO}")
if [ -n "${SID_KEEP_TOKENS}" ]; then
    SID_FLAGS+=(--sid-keep-tokens "${SID_KEEP_TOKENS}")
fi

EXTRACT_FLAGS=("${SID_FLAGS[@]}")
if [ -n "${MAX_PIXELS}" ]; then
    EXTRACT_FLAGS+=(--max-pixels "${MAX_PIXELS}")
fi
if [ "${RESUME}" = "1" ]; then
    EXTRACT_FLAGS+=(--resume)
fi

# ----------------------------------------------------------------- stage: selfcheck
if has_stage selfcheck; then
    banner "4. selfcheck (fail-fast validation)"
    "${PY}" "${HERE}/selfcheck.py" \
        --model-path "${MODEL_PATH}" \
        --data-dir "${DATA_DIR}" \
        --dtype "${DTYPE}" --attn-impl "${ATTN_IMPL}" \
        --noise-step "${NOISE_STEP}" \
        "${SID_FLAGS[@]}" \
        2>&1 | tee "${LOG_DIR}/selfcheck.log"
fi

# ----------------------------------------------------------------- stage: extract
if has_stage extract; then
    banner "5. extraction (expert + 3 amateurs, one pass)"
    "${PY}" "${HERE}/extract_cd.py" \
        --model-path "${MODEL_PATH}" \
        --data-dir "${DATA_DIR}" \
        --act-dir "${ACT_DIR}" \
        --results-dir "${RESULTS_DIR}" \
        --methods ${METHODS} \
        --splits ${SPLITS} \
        --dtype "${DTYPE}" --attn-impl "${ATTN_IMPL}" \
        --noise-step "${NOISE_STEP}" \
        --shard-size "${SHARD_SIZE}" \
        --max-samples "${MAX_SAMPLES}" \
        "${EXTRACT_FLAGS[@]}" \
        2>&1 | tee "${LOG_DIR}/extract.log"
fi

# ----------------------------------------------------------------- stage: probes
if has_stage probes; then
    banner "6. per-layer probes (CPU, the slow analysis stage)"
    "${PY}" "${HERE}/train_probes.py" \
        --act-dir "${ACT_DIR}" \
        --results-dir "${RESULTS_DIR}" \
        --method "${PRIMARY_METHOD}" \
        --splits ${SPLITS} \
        --n-jobs "${PROBE_JOBS}" \
        2>&1 | tee "${LOG_DIR}/probes.log"
fi

# ----------------------------------------------------------------- stage: resid
if has_stage resid; then
    banner "7. per-method residual-magnitude stats"
    for m in ${METHODS}; do
        "${PY}" "${HERE}/resid_stats.py" \
            --act-dir "${ACT_DIR}" --results-dir "${RESULTS_DIR}" \
            --method "${m}" --splits ${SPLITS}
    done
fi

# ----------------------------------------------------------------- stage: figures
if has_stage figures; then
    banner "8. figures"
    for m in ${METHODS}; do
        "${PY}" "${HERE}/logit_lens_figures.py" \
            --npz "${RESULTS_DIR}/logit_lens_per_sample_${m}.npz" \
            --method "${m}" \
            --probe-csv "${RESULTS_DIR}/per_layer_normalized.csv" \
            --resid-csv "${RESULTS_DIR}/per_layer_resid_${m}.csv" \
            --out-dir "${REGEN_DIR}"
    done
    "${PY}" "${HERE}/analyze_all.py" --results-dir "${RESULTS_DIR}"
fi

# ----------------------------------------------------------------- stage: package
if has_stage package; then
    banner "9. package"
    EXPECTED=0
    MISSING=0
    for m in ${METHODS}; do
        for f in halluc_auc_vs_cd_effect layerwise_battery logit_lens_curves cd_normalized_curves; do
            EXPECTED=$((EXPECTED + 1))
            if [ ! -f "${REGEN_DIR}/${f}_${m}.png" ]; then
                echo "[package] MISSING ${REGEN_DIR}/${f}_${m}.png"
                MISSING=$((MISSING + 1))
            fi
        done
    done
    echo "[package] regen figures: $((EXPECTED - MISSING))/${EXPECTED} present"

    BUNDLE="${ROOT}/cd_pope_mech_interp_qwen_results.tar.gz"
    # everything in results/ is small now (the multi-GB hidden dumps are gone), so ship it all
    tar -czf "${BUNDLE}" -C "${ROOT}" results
    echo "[package] artifacts (csv/png/json/npz) -> ${BUNDLE}"
    echo
    echo "results   : ${RESULTS_DIR}"
    echo "figures   : ${REGEN_DIR}"
    echo "bundle    : ${BUNDLE}"
    [ "${MISSING}" -eq 0 ] || { echo "[package] FAILED: ${MISSING} figure(s) missing"; exit 1; }
fi

banner "DONE"
