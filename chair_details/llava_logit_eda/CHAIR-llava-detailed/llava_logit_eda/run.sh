#!/usr/bin/env bash
# =============================================================================
# run.sh  --  launch the LLaVA-1.5-7B logit EDA experiment
#
# USAGE:
#   bash run.sh                                      # downloads model from HF
#   bash run.sh liuhaotian/llava-v1.5-7b             # explicit HF model id
#   bash run.sh /path/to/local/llava-v1.5-7b         # local checkpoint
#   bash run.sh liuhaotian/llava-v1.5-7b greedy vcd  # only those methods
#
# WHAT THIS DOES:
#   For each of the 10 CHAIR benchmark images (IDs in image_ids.json), runs
#   GREEDY, VCD, and SID decoding on LLaVA-1.5-7B.  At every token generation
#   step it captures the full logit distributions at 4 stages:
#     1. expert_logit      - raw output with real image
#     2. amateur_logit     - raw output with degraded image (VCD: diffusion noise,
#                            SID: 504/576 image tokens masked at layer 2+)
#     3. cd_logit_pre_apc  - 2*expert - amateur  (alpha=1)
#     4. cd_logit_post_apc - stage 3 masked to -inf where expert < log(0.2)+max
#   Top-100 tokens by expert logit are saved per step with all 4 stage values,
#   probabilities, and APC survival flags.
#
# OUTPUTS (written to outputs/ inside this folder):
#   outputs/greedy/img_{id}_eda.json   (10 files)
#   outputs/vcd/img_{id}_eda.json      (10 files)
#   outputs/sid/img_{id}_eda.json      (10 files)
#   outputs/summary.csv                aggregate stats per (image, method)
#   outputs/coco_images/               auto-downloaded COCO val2017 images
#
# REQUIREMENTS: run bash install.sh first
# GPU: L4 24GB recommended (~8-12 min). T4 16GB is risky (only 2GB margin).
# =============================================================================

set -e

# Default model path
MODEL_PATH="${1:-liuhaotian/llava-v1.5-7b}"
shift 2>/dev/null || true   # shift past model arg; remaining args are methods

# Change to this script's directory so relative paths work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Sanity check: must be inside the cd-rethinking repo
if [ ! -d "../llava" ] || [ ! -d "../inference" ]; then
    echo "ERROR: llava_logit_eda/ must be placed INSIDE the cd-rethinking repo."
    echo ""
    echo "Expected layout:"
    echo "  cd-rethinking/"
    echo "  ├── llava/"
    echo "  ├── inference/"
    echo "  └── llava_logit_eda/   <-- this folder"
    echo ""
    echo "Move this folder inside the repo root and try again."
    exit 1
fi

# Check Python and key deps
echo "=== Pre-flight checks ==="
python - <<'PREFLIGHT'
import sys

# torch / transformers / PIL can be checked without side effects
for pkg, imp in [("torch", "torch"), ("transformers", "transformers"), ("PIL", "PIL")]:
    try:
        __import__(imp)
    except ImportError:
        print(f"  MISSING: {pkg}  --  run: bash install.sh")
        sys.exit(1)

import torch
cuda = torch.cuda.is_available()
print(f"  torch       : {torch.__version__}")
print(f"  CUDA        : {'YES  device=' + torch.cuda.get_device_name(0) if cuda else 'NO  (will be very slow)'}")
import transformers
print(f"  transformers: {transformers.__version__}")

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
import types
_mpt = types.ModuleType("llava.model.language_model.llava_mpt")
_mpt.LlavaMPTForCausalLM = type("LlavaMPTForCausalLM", (), {})
_mpt.LlavaMPTConfig      = type("LlavaMPTConfig", (), {})
sys.modules["llava.model.language_model.llava_mpt"] = _mpt

try:
    import llava  # noqa
    print("  llava       : OK")
except Exception as e:
    print(f"  llava       : FAILED ({e})")
    print("  Run:  bash install.sh  first.")
    sys.exit(1)
PREFLIGHT

echo ""
echo "====================================="
echo "  LLaVA-1.5-7B Logit EDA Experiment"
echo "====================================="
echo "  Model  : $MODEL_PATH"
if [ "$#" -gt 0 ]; then
    echo "  Methods: $@"
else
    echo "  Methods: greedy  vcd  sid  (all)"
fi
echo "  Output : $SCRIPT_DIR/outputs/"
echo "====================================="
echo ""

# Build method args
if [ "$#" -gt 0 ]; then
    METHOD_ARGS="--methods $@"
else
    METHOD_ARGS=""
fi

python run_eda.py \
    --model-path "$MODEL_PATH" \
    $METHOD_ARGS

echo ""
echo "====================================="
echo "  Done.  Results in: outputs/"
echo "====================================="
