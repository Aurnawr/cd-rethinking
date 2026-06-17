#!/bin/bash
# Runs every Qwen2.5-VL POPE inference script in sequence:
#   1. baseline / greedy   (pope_infer_base.py)
#   2. VCD / ICD / SID     (pope_infer_cd.py)
#   3. PBA / OLM           (pope_infer_spurious.py)
#
# Usage (from anywhere):
#   bash scripts/run_all_infer.sh
#
# Designed to be run inside tmux so it survives a disconnect:
#   tmux new -s qwen_run
#   bash scripts/run_all_infer.sh
#   (Ctrl+B then D to detach, `tmux attach -t qwen_run` to come back)

set -euo pipefail  # stop immediately on any real failure, including inside a `| tee` pipe

# Resolve the repo root regardless of where this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

mkdir -p logs

log() {
    echo ""
    echo "================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "================================================================"
}

START_TIME=$(date +%s)

# Uncomment if you ever need to (re)install deps as part of this run:
# log "Installing dependencies (pip install -e .)"
# pip install -e . 2>&1 | tee logs/install.log

log "STEP 1/3: Baseline / greedy inference (pope_infer_base.py)"
bash scripts/pope_infer_base.sh 2>&1 | tee logs/infer_base.log

log "STEP 2/3: Contrastive decoding -- VCD / ICD / SID (pope_infer_cd.py)"
bash scripts/pope_infer_cd.sh 2>&1 | tee logs/infer_cd.log

log "STEP 3/3: Spurious-improvement methods -- PBA / OLM (pope_infer_spurious.py)"
bash scripts/pope_infer_spurious.sh 2>&1 | tee logs/infer_spurious.log

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

log "ALL INFERENCE RUNS COMPLETE. Total time: $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"
echo "Answer files are in: outputs/pope/{baseline,vcd,icd,sid,pba,olm}/"
echo "Logs are in: logs/{infer_base,infer_cd,infer_spurious}.log"
