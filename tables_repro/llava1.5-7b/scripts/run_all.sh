#!/usr/bin/env bash
# Full reproduction: generate every method's answers, then score them.
set -euo pipefail
here="$(dirname "$0")"
bash "$here/run_baseline.sh"
bash "$here/run_cd.sh"
bash "$here/run_spurious.sh"
bash "$here/run_eval.sh"
