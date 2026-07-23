#!/usr/bin/env bash
# =============================================================================
# install.sh  --  LLaVA-1.5-7B logit EDA dependencies
#
# PLACEMENT: drop this folder (llava_logit_eda/) INSIDE the cd-rethinking repo,
# so the directory tree looks like:
#
#   cd-rethinking/
#   ├── llava/
#   ├── inference/
#   ├── eval/
#   ├── outputs/
#   └── llava_logit_eda/      <-- this folder
#       ├── install.sh        <-- you are here
#       ├── run_eda.py
#       └── image_ids.json
#
# RUN FROM INSIDE llava_logit_eda/:
#   cd llava_logit_eda
#   bash install.sh
#
# Then:
#   python run_eda.py --model-path liuhaotian/llava-v1.5-7b
#
# Lightning AI notes:
#   - PyTorch is pre-installed; skip the torch block below if CUDA version
#     does not match (change cu118 -> cu121 for CUDA 12.1).
#   - If you already have the right torch, comment out the first pip block.
# =============================================================================

set -e

echo "=== [1/4] PyTorch ==="
python -c "import torch; print('  already installed:', torch.__version__, '| CUDA:', torch.version.cuda)" 2>/dev/null && TORCH_OK=1 || TORCH_OK=0

if [ "$TORCH_OK" -eq 0 ]; then
    echo "  Installing PyTorch 2.0.1 + cu118 ..."
    pip install torch==2.0.1 torchvision==0.15.2 \
        --index-url https://download.pytorch.org/whl/cu118 \
        --quiet
else
    echo "  Skipping (already present)."
fi

echo ""
echo "=== [2/4] Core NLP / vision dependencies ==="
# transformers==4.31.0 requires tokenizers<0.14, but tokenizers 0.13.x has no
# Python 3.12 pre-built wheel and fails to compile from source (needs Rust).
# transformers>=4.35.0 allows tokenizers>=0.15.0 which HAS Python 3.12 wheels.
# The llava custom LLaMA model is a local copy so it doesn't break when
# transformers is newer; we do manual generation so generate() API changes
# don't matter; add_diffusion_noise is inlined so vcd_utils.py is never loaded.
pip install \
    "transformers>=4.35.0" \
    "tokenizers>=0.15.0" \
    "sentencepiece" \
    "accelerate>=0.21.0" \
    "einops==0.6.1" \
    "einops-exts==0.0.4" \
    "timm==0.6.13" \
    "shortuuid" \
    "Pillow" \
    "tqdm" \
    "requests" \
    "numpy" \
    --quiet

echo ""
echo "=== [3/4] Local llava package (from parent repo) ==="
# The llava package lives one directory up (the cd-rethinking repo root).
# --no-deps skips the pyproject.toml dependency pins (torch==2.0.1, gradio, etc.)
# which would conflict with Lightning AI's pre-installed packages.
# All required dependencies were already installed in step 2.
pip install -e .. --no-deps --quiet

echo ""
echo "=== [4/4] Verify ==="
python - <<'PYCHECK'
import torch, transformers
print("  torch       :", torch.__version__)
print("  transformers:", transformers.__version__)
cuda = torch.cuda.is_available()
print("  CUDA        :", "YES -" if cuda else "NO (CPU-only run will be very slow)", torch.version.cuda if cuda else "")

# 1. Remove built-in "llava" from CONFIG_MAPPING so our local package can
#    re-register it (transformers>=4.36 ships its own LLaVA).
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if "llava" in CONFIG_MAPPING._mapping:
        del CONFIG_MAPPING._mapping["llava"]
except Exception:
    pass

# 2. Stub the MPT module before importing llava.  llava/model/__init__.py
#    imports LlavaMPTForCausalLM, which pulls in a vendored MPT model that
#    uses _expand_mask from transformers.models.bloom — removed in 4.36+.
import sys, types
_mpt = types.ModuleType("llava.model.language_model.llava_mpt")
_mpt.LlavaMPTForCausalLM = type("LlavaMPTForCausalLM", (), {})
_mpt.LlavaMPTConfig      = type("LlavaMPTConfig", (), {})
sys.modules["llava.model.language_model.llava_mpt"] = _mpt

try:
    import llava  # noqa
    print("  llava       : OK")
except Exception as e:
    print("  llava       : FAILED --", e)
    raise
PYCHECK

echo ""
echo "All done. Run: python run_eda.py --model-path liuhaotian/llava-v1.5-7b"
