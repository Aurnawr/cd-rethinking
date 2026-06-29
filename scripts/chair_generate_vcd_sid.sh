#!/usr/bin/env bash
# Generate CHAIR captions for VCD then SID sequentially (single GPU).
# Each method writes outputs/chair/llava-7b-<method>/captions.jsonl and resumes
# from any existing partial file.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for method in vcd sid; do
    echo "=== CHAIR generate: ${method} ==="
    CD_METHOD="${method}" bash "${REPO_ROOT}/scripts/chair_generate.sh"
done

echo "VCD + SID generation complete."
