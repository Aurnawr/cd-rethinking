#!/bin/bash
# KLD experiment for VCD, ICD, SID on VisIT-Bench with LLaVA-v1.5-13B.
# Downloads VisIT-Bench via the `datasets` library at runtime (needs internet;
# set HF_TOKEN if the dataset requires authentication).
# Outputs: repro_outputs/kld_experiment/visit-bench/{vcd,icd,sid}_kld.json
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${REPO_ROOT}"

OUTPUT_DIR="${OUT_ROOT}/kld_experiment/visit-bench"
mkdir -p "${OUTPUT_DIR}"

## vcd (noise-step drives the diffusion corruption of the amateur view)
python ./inference/kld_experiment.py \
    --model-path "${MODEL_13B}" \
    --conv-mode  "${CONV_MODE}" \
    --method     vcd \
    --output-file "${OUTPUT_DIR}/vcd_kld.json" \
    --noise-step 900 \
    --temperature 0.2 \
    --max-new-tokens 128

## icd
python ./inference/kld_experiment.py \
    --model-path "${MODEL_13B}" \
    --conv-mode  "${CONV_MODE}" \
    --method     icd \
    --output-file "${OUTPUT_DIR}/icd_kld.json" \
    --temperature 0.2 \
    --max-new-tokens 128

## sid
python ./inference/kld_experiment.py \
    --model-path "${MODEL_13B}" \
    --conv-mode  "${CONV_MODE}" \
    --method     sid \
    --output-file "${OUTPUT_DIR}/sid_kld.json" \
    --temperature 0.2 \
    --max-new-tokens 128
