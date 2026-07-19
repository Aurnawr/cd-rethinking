#!/bin/bash
# ===========================================================================
# setup.sh -- one-shot, idempotent environment bootstrap for the
# LLaVA-v1.5-13B reproduction of "The Mirage of Performance Gains".
#
# It installs Python deps, downloads the 13B checkpoint, fetches the POPE
# question files, and prepares the COCO / GQA image datasets. Every step
# checks for existing artifacts and skips work already done, so re-running is
# safe and cheap.
#
# Usage:
#   bash setup.sh                 # do everything that's missing
#   bash setup.sh --skip-images   # skip the large COCO/GQA downloads
#   bash setup.sh --skip-deps --skip-model
#
# Override any location by exporting it first (see repro_scripts/config.sh):
#   export STORAGE_ROOT=/mnt/data
#   export COCO_IMAGES=/existing/coco/val2014   # point at images you already have
#
# Flags: --skip-deps  --skip-model  --skip-data  --skip-images  --with-qwen
# ===========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/repro_scripts/config.sh"
cd "${REPO_ROOT}"

# ---- flags ----------------------------------------------------------------
SKIP_DEPS=0; SKIP_MODEL=0; SKIP_DATA=0; SKIP_IMAGES=0; WITH_QWEN=0
for arg in "$@"; do
    case "$arg" in
        --skip-deps)   SKIP_DEPS=1 ;;
        --skip-model)  SKIP_MODEL=1 ;;
        --skip-data)   SKIP_DATA=1 ;;
        --skip-images) SKIP_IMAGES=1 ;;
        --with-qwen)   WITH_QWEN=1 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[setup] unknown flag: $arg" >&2; exit 2 ;;
    esac
done

# ---- helpers --------------------------------------------------------------
log()  { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ok:\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  warn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# Download a URL to a path (resumable), preferring curl then wget.
fetch() {  # fetch <url> <dest>
    local url="$1" dest="$2"
    mkdir -p "$(dirname "$dest")"
    if have curl; then
        curl -fL --retry 3 -C - -o "$dest" "$url"
    elif have wget; then
        wget -c -O "$dest" "$url"
    else
        die "neither curl nor wget is available"
    fi
}

UPSTREAM_RAW="https://raw.githubusercontent.com/ustc-hyin/cd_rethink/main"

# Datasets actually in play = union of the sweep list and the POPE-table dataset.
NEEDED_DATASETS="$(printf '%s\n' ${DATASETS} ${POPE_DATASET} | sort -u | tr '\n' ' ')"
NEED_COCO=0; NEED_GQA=0
for d in ${NEEDED_DATASETS}; do
    case "$d" in
        coco|aokvqa) NEED_COCO=1 ;;   # AOKVQA reuses COCO val2014 images
        gqa)         NEED_GQA=1 ;;
    esac
done

# ---------------------------------------------------------------------------
log "Resolved configuration"
cat <<EOF
  REPO_ROOT   = ${REPO_ROOT}
  MODEL_13B   = ${MODEL_13B}   (from ${HF_MODEL_ID})
  DATASETS    = ${NEEDED_DATASETS} (need COCO imgs=${NEED_COCO}, GQA imgs=${NEED_GQA})
  COCO_IMAGES = ${COCO_IMAGES}
  GQA_IMAGES  = ${GQA_IMAGES}
  DATA_DIR    = ${DATA_DIR}
  OUT_ROOT    = ${OUT_ROOT}
EOF

# ---------------------------------------------------------------------------
# 1. Python environment + dependencies
#
# The reproduction stack pins transformers==4.31.0 (REQUIRED: the VCD/ICD/SID
# utils monkey-patch transformers.generation.utils internals that exist ONLY in
# that release). transformers 4.31 needs tokenizers<0.14, which -- like the
# paper's torch==2.0.1 -- has no wheels for Python >=3.12. So if the current
# Python is too new we build an isolated Python 3.10 env (conda > uv > venv) and
# run everything through it. We do NOT pin torch to 2.0.1; instead we install a
# modern CUDA-capable torch that still works with transformers 4.31.
# ---------------------------------------------------------------------------
ENVFILE="${REPO_ROOT}/repro_scripts/.py_env"

py_ok() {  # py_ok <python> -> true if version is 3.8..3.11
    "$1" -c 'import sys; raise SystemExit(0 if (3,8) <= sys.version_info[:2] <= (3,11) else 1)' 2>/dev/null
}

# Each _try_* sets PY_BIN and returns 0 on success, non-zero to fall through.
VENV_DIR="${REPO_ROOT}/.venv-cdr"

_resolve_uv() {  # print a usable uv path, or nothing
    local u
    u="$(command -v uv 2>/dev/null)" && { echo "$u"; return 0; }
    u="$(python -c 'import sysconfig,os;print(os.path.join(sysconfig.get_path("scripts"),"uv"))' 2>/dev/null)"
    [ -x "$u" ] && { echo "$u"; return 0; }
    return 1
}

_try_uv() {
    # uv self-downloads a standalone CPython 3.10 -- works on Lightning where
    # `conda create` is blocked and no system python3.10 exists.
    local UV
    UV="$(_resolve_uv)" || {
        log "installing uv (portable Python/venv manager)"
        python -m pip install --quiet uv >&2 2>/dev/null || pip install --quiet uv >&2 2>/dev/null || return 1
        UV="$(_resolve_uv)" || return 1
    }
    log "creating Python 3.10 venv via uv: ${VENV_DIR}"
    "$UV" venv --seed --python 3.10 "${VENV_DIR}" >&2 || return 1
    PY_BIN="${VENV_DIR}/bin/python"
    py_ok "$PY_BIN"
}

_try_python310() {
    have python3.10 || return 1
    log "creating Python 3.10 venv via python3.10: ${VENV_DIR}"
    python3.10 -m venv "${VENV_DIR}" >&2 || return 1
    PY_BIN="${VENV_DIR}/bin/python"
    py_ok "$PY_BIN"
}

_try_conda() {
    have conda || return 1
    log "creating conda env ${REPRO_ENV_NAME} (python=3.10)"
    conda create -y -n "${REPRO_ENV_NAME}" python=3.10 >&2 || return 1
    PY_BIN="$(conda run -n "${REPRO_ENV_NAME}" python -c 'import sys; print(sys.executable)' 2>/dev/null)" || return 1
    py_ok "$PY_BIN"
}

ensure_python_env() {
    # Reuse a previously recorded interpreter if it still works.
    if [ -f "$ENVFILE" ] && py_ok "$(cat "$ENVFILE" 2>/dev/null)"; then
        PY_BIN="$(cat "$ENVFILE")"; export PY_BIN
        ok "reusing repro Python: ${PY_BIN}"
        return 0
    fi
    # If the current python is already 3.8-3.11, just use it.
    if have python && py_ok python; then
        PY_BIN="$(command -v python)"; echo "$PY_BIN" > "$ENVFILE"; export PY_BIN
        ok "using current Python: ${PY_BIN} ($($PY_BIN -V 2>&1))"
        return 0
    fi
    warn "current Python is >=3.12; building a Python 3.10 env (uv > python3.10 > conda)"
    # Order matters: uv is the most portable (and the only one that works on a
    # Lightning studio, where conda create is disallowed). Fall through on
    # failure instead of dying, so a blocked/broken method doesn't abort setup.
    if _try_uv || _try_python310 || _try_conda; then
        echo "$PY_BIN" > "$ENVFILE"; export PY_BIN
        ok "created repro Python: ${PY_BIN} ($($PY_BIN -V 2>&1))"
        return 0
    fi
    die "could not build a Python<=3.11 env (tried uv, python3.10, conda). Run 'pip install uv' then re-run, or export PY_BIN to a compatible interpreter."
}

if [ "$SKIP_DEPS" -eq 0 ]; then
    log "Preparing Python environment"
    ensure_python_env
    PIP="${PY_BIN} -m pip"

    log "Installing dependencies into ${PY_BIN}"
    $PIP install --upgrade pip setuptools wheel >/dev/null 2>&1 || warn "pip self-upgrade failed (continuing)"

    # Modern CUDA-capable torch (default PyPI wheel supports Ampere/Ada/Hopper).
    # Override with TORCH_SPEC / TORCH_INDEX_URL for a specific CUDA build, e.g.
    #   export TORCH_SPEC="torch==2.2.2 torchvision==0.17.2"
    #   export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
    TORCH_SPEC="${TORCH_SPEC:-torch==2.2.2 torchvision==0.17.2}"
    if [ -n "${TORCH_INDEX_URL:-}" ]; then
        $PIP install ${TORCH_SPEC} --index-url "${TORCH_INDEX_URL}" || die "torch install failed"
    else
        $PIP install ${TORCH_SPEC} || die "torch install failed"
    fi

    # transformers is pinned; the rest are the runtime deps the inference/eval
    # code actually imports. Installed with explicit versions known to co-exist
    # with transformers 4.31 on Python 3.10.
    $PIP install \
        "transformers==4.31.0" "tokenizers>=0.13,<0.14" "sentencepiece==0.1.99" \
        "accelerate==0.21.0" "huggingface_hub[cli]>=0.16,<0.25" \
        "numpy<2" "scikit-learn" "shortuuid" "protobuf" \
        "einops==0.6.1" "einops-exts==0.0.4" "timm==0.6.13" \
        "pillow" "requests" "tqdm" "datasets>=2.14,<3" \
        || die "dependency install failed"

    # Register the local `llava` package WITHOUT its stale pinned deps
    # (pyproject pins torch==2.0.1 etc., which we deliberately override above).
    $PIP install -e "${REPO_ROOT}" --no-deps || die "editable install of llava failed"
    ok "dependencies installed"
else
    warn "skipping dependency install (--skip-deps)"
    [ -f "$ENVFILE" ] && { PY_BIN="$(cat "$ENVFILE")"; export PY_BIN; }
fi
# Make sure PY_BIN is defined for the download/check steps below even when deps
# were skipped (falls back to config's resolution / plain python).
PY_BIN="${PY_BIN:-python}"

# ---------------------------------------------------------------------------
# 2. LLaVA-v1.5-13B checkpoint
# ---------------------------------------------------------------------------
if [ "$SKIP_MODEL" -eq 0 ]; then
    if [ -f "${MODEL_13B}/config.json" ]; then
        ok "model already present at ${MODEL_13B}"
    else
        log "Downloading ${HF_MODEL_ID} -> ${MODEL_13B} (~26 GB, one-time)"
        mkdir -p "${MODEL_13B}"
        HF_MODEL_ID="${HF_MODEL_ID}" MODEL_13B="${MODEL_13B}" "${PY_BIN}" - <<'PY' || die "model download failed"
import os
from huggingface_hub import snapshot_download
repo = os.environ["HF_MODEL_ID"]
dest = os.environ["MODEL_13B"]
kwargs = dict(repo_id=repo, local_dir=dest,
              ignore_patterns=["*.gif", "*.png", "*.md", ".gitattributes"])
try:
    # older hub versions copy via symlinks unless told otherwise
    snapshot_download(local_dir_use_symlinks=False, **kwargs)
except TypeError:
    snapshot_download(**kwargs)
print("downloaded", repo, "->", dest)
PY
        [ -f "${MODEL_13B}/config.json" ] || die "download finished but config.json missing in ${MODEL_13B}"
        ok "model ready at ${MODEL_13B}"
    fi
else
    warn "skipping model download (--skip-model)"
fi

# ---------------------------------------------------------------------------
# 3. POPE question files  (data/<dataset>/<dataset>_pope_<type>.json)
# ---------------------------------------------------------------------------
if [ "$SKIP_DATA" -eq 0 ]; then
    log "Fetching POPE question files into ${DATA_DIR} (datasets: ${NEEDED_DATASETS})"
    missing=0
    for dataset in ${NEEDED_DATASETS}; do
        for type in random popular adversarial; do
            dest="${DATA_DIR}/${dataset}/${dataset}_pope_${type}.json"
            if [ -s "$dest" ]; then
                continue
            fi
            if fetch "${UPSTREAM_RAW}/data/${dataset}/${dataset}_pope_${type}.json" "$dest"; then
                :
            else
                warn "could not fetch ${dataset}_pope_${type}.json"
                missing=1
            fi
        done
    done
    [ "$missing" -eq 0 ] && ok "POPE question files present" || warn "some POPE files are missing (see above)"
else
    warn "skipping POPE data download (--skip-data)"
fi

# ---------------------------------------------------------------------------
# 4. Image datasets (COCO val2014 + GQA). AOKVQA reuses COCO images.
# ---------------------------------------------------------------------------
dir_has_jpg() { [ -d "$1" ] && [ -n "$(find "$1" -maxdepth 1 -name '*.jpg' -print -quit 2>/dev/null)" ]; }

if [ "$SKIP_IMAGES" -eq 0 ]; then
    # ---- COCO val2014 (needed by coco and aokvqa) ----
    if [ "$NEED_COCO" -eq 0 ]; then
        ok "COCO images not needed for datasets: ${NEEDED_DATASETS}"
    elif dir_has_jpg "${COCO_IMAGES}"; then
        ok "COCO images already present at ${COCO_IMAGES}"
    else
        log "Downloading COCO val2014 (~6.5 GB) -> ${COCO_IMAGES}"
        coco_parent="$(dirname "${COCO_IMAGES}")"   # unzip creates val2014/ here
        mkdir -p "${coco_parent}"
        zip="${coco_parent}/val2014.zip"
        fetch "http://images.cocodataset.org/zips/val2014.zip" "$zip" || die "COCO download failed"
        have unzip || die "unzip not found (needed to extract COCO)"
        unzip -q -o "$zip" -d "${coco_parent}" || die "COCO unzip failed"
        rm -f "$zip"
        dir_has_jpg "${COCO_IMAGES}" && ok "COCO images ready" \
            || warn "COCO extracted but ${COCO_IMAGES} has no jpgs -- check COCO_IMAGES path"
    fi

    # ---- GQA images ----
    if [ "$NEED_GQA" -eq 0 ]; then
        ok "GQA images not needed for datasets: ${NEEDED_DATASETS}"
    elif dir_has_jpg "${GQA_IMAGES}"; then
        ok "GQA images already present at ${GQA_IMAGES}"
    else
        log "Downloading GQA images (~20 GB) -> ${GQA_IMAGES}"
        gqa_parent="$(dirname "${GQA_IMAGES}")"     # unzip creates images/ here
        mkdir -p "${gqa_parent}"
        zip="${gqa_parent}/gqa_images.zip"
        fetch "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" "$zip" || die "GQA download failed"
        have unzip || die "unzip not found (needed to extract GQA)"
        unzip -q -o "$zip" -d "${gqa_parent}" || die "GQA unzip failed"
        rm -f "$zip"
        dir_has_jpg "${GQA_IMAGES}" && ok "GQA images ready" \
            || warn "GQA extracted but ${GQA_IMAGES} has no jpgs -- check GQA_IMAGES path"
    fi
else
    warn "skipping image downloads (--skip-images)"
    [ "$NEED_COCO" -eq 1 ] && { dir_has_jpg "${COCO_IMAGES}" || warn "COCO images NOT found at ${COCO_IMAGES}"; }
    [ "$NEED_GQA" -eq 1 ]  && { dir_has_jpg "${GQA_IMAGES}"  || warn "GQA images NOT found at ${GQA_IMAGES}"; }
    true
fi

# ---------------------------------------------------------------------------
# 5. Optional: Qwen-2.5-VL-7B checkpoint (separate track)
# ---------------------------------------------------------------------------
if [ "$WITH_QWEN" -eq 1 ]; then
    if [ -f "${QWEN_MODEL}/config.json" ]; then
        ok "Qwen model already present at ${QWEN_MODEL}"
    else
        log "Downloading Qwen2.5-VL-7B -> ${QWEN_MODEL}"
        mkdir -p "${QWEN_MODEL}"
        QWEN_MODEL="${QWEN_MODEL}" "${PY_BIN}" - <<'PY' || warn "Qwen download failed (llava track unaffected)"
import os
from huggingface_hub import snapshot_download
dest = os.environ["QWEN_MODEL"]
try:
    snapshot_download(repo_id="Qwen/Qwen2.5-VL-7B-Instruct", local_dir=dest, local_dir_use_symlinks=False)
except TypeError:
    snapshot_download(repo_id="Qwen/Qwen2.5-VL-7B-Instruct", local_dir=dest)
PY
    fi
fi

# ---------------------------------------------------------------------------
# 6. Preflight sanity checks
# ---------------------------------------------------------------------------
log "Preflight checks"
if have nvidia-smi; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
        | sed 's/^/  GPU: /' || warn "nvidia-smi present but query failed"
else
    warn "nvidia-smi not found -- no CUDA GPU? Inference needs a GPU."
fi

"${PY_BIN}" - <<'PY' 2>/dev/null || warn "python import check failed -- review dependency install"
import importlib
mods = ["torch", "transformers", "datasets", "PIL"]
for m in mods:
    importlib.import_module(m)
import torch, transformers
print(f"  torch {torch.__version__}, cuda avail={torch.cuda.is_available()}")
print(f"  transformers {transformers.__version__}")
assert transformers.__version__.startswith("4.31"), \
    "transformers must be 4.31.x for the CD monkey-patches"
print("  import check OK")
PY

log "Setup complete. Next: bash run_master.sh"
