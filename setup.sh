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

# ---------------------------------------------------------------------------
log "Resolved configuration"
cat <<EOF
  REPO_ROOT   = ${REPO_ROOT}
  MODEL_13B   = ${MODEL_13B}   (from ${HF_MODEL_ID})
  COCO_IMAGES = ${COCO_IMAGES}
  GQA_IMAGES  = ${GQA_IMAGES}
  DATA_DIR    = ${DATA_DIR}
  OUT_ROOT    = ${OUT_ROOT}
EOF

# ---------------------------------------------------------------------------
# 1. Python dependencies
# ---------------------------------------------------------------------------
if [ "$SKIP_DEPS" -eq 0 ]; then
    log "Installing Python dependencies"
    have python || die "python not found on PATH"
    PIP="python -m pip"
    $PIP install --upgrade pip >/dev/null 2>&1 || warn "pip self-upgrade failed (continuing)"
    # The package pins transformers==4.31.0 -- REQUIRED: the VCD/ICD/SID utils
    # monkey-patch transformers.generation.utils internals that only exist in
    # that release. Do not bump transformers.
    $PIP install -e "${REPO_ROOT}" || die "pip install -e . failed"
    # `datasets` is imported by kld_experiment.py but is not a declared dep.
    $PIP install "datasets>=2.14,<3" "huggingface_hub[cli]>=0.20" || die "extra deps failed"
    ok "dependencies installed"
else
    warn "skipping dependency install (--skip-deps)"
fi

# ---------------------------------------------------------------------------
# 2. LLaVA-v1.5-13B checkpoint
# ---------------------------------------------------------------------------
if [ "$SKIP_MODEL" -eq 0 ]; then
    if [ -f "${MODEL_13B}/config.json" ]; then
        ok "model already present at ${MODEL_13B}"
    else
        log "Downloading ${HF_MODEL_ID} -> ${MODEL_13B} (~26 GB, one-time)"
        mkdir -p "${MODEL_13B}"
        HF_MODEL_ID="${HF_MODEL_ID}" MODEL_13B="${MODEL_13B}" python - <<'PY' || die "model download failed"
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
    log "Fetching POPE question files into ${DATA_DIR}"
    missing=0
    for dataset in coco gqa aokvqa; do
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
    # ---- COCO val2014 ----
    if dir_has_jpg "${COCO_IMAGES}"; then
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
    if dir_has_jpg "${GQA_IMAGES}"; then
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
    dir_has_jpg "${COCO_IMAGES}" || warn "COCO images NOT found at ${COCO_IMAGES}"
    dir_has_jpg "${GQA_IMAGES}"  || warn "GQA images NOT found at ${GQA_IMAGES}"
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
        QWEN_MODEL="${QWEN_MODEL}" python - <<'PY' || warn "Qwen download failed (llava track unaffected)"
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

python - <<'PY' 2>/dev/null || warn "python import check failed -- review dependency install"
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
