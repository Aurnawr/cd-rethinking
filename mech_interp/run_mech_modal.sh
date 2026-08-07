#!/usr/bin/env bash
# Drive the full CD mechanistic battery on Modal (LLaVA-1.5-7B, POPE-COCO).
#
# The Modal CLI is not on PATH; invoke via uvx (auth from ~/.modal.toml).
# Each step commits to the `cd-mech` Volume; the final step pulls the small result
# files to ~/Downloads for offline plotting. GPU steps use an L4 (~$0.80/hr).
#
# Run steps individually (recommended, so you can watch each) or `bash run_mech_modal.sh all`.
set -euo pipefail

M() { uvx --from modal modal "$@"; }
APP="mech_interp/modal_app.py"
OUT="${HOME}/Downloads/cd_pope_mech_interp_results"

setup() {            # one-time: image build + model + data (CPU/idempotent)
  M run "${APP}::download_model"
  M run "${APP}::download_data"
}

smoke() { M run "${APP}::smoke"; }   # 8-sample env validation

# --- Phase A: per-method logit-lens battery ---
# Sequential (one L4 at a time): spend ramps gradually and stays visible against the
# workspace limit, and a failure loses only the method in flight (each method writes its
# npz atomically at the end). Flip to parallel only with headroom on the spend limit.
extract_all() {
  for m in vcd icd sid; do
    echo "=== extract ${m} ==="
    M run "${APP}::extract" --method "$m"
  done
}

# --- Phase B: directions, causal, constructive (some depend on A) ---
analyze_phase() {
  M run "${APP}::fit_probes"                 # probe_dirs.npz (needs any hiddens_ht)
  M run "${APP}::align"                       # subspace_align.csv/npz (needs hiddens_ht_*)
  M run "${APP}::patch" --method vcd &        # causal patching on H (GPU)
  M run "${APP}::steer" &                      # constructive counterfactual (needs probe_dirs)
  wait
  M run "${APP}::analyze"                      # cross-method figures + summary tables
}

pull() {             # bring the small result files (npz/csv/png/json) local
  mkdir -p "${OUT}"
  M volume get cd-mech results "${OUT}" --force
  echo "pulled results -> ${OUT}"
}

case "${1:-}" in
  setup) setup ;;
  smoke) smoke ;;
  extract) extract_all ;;
  analyze) analyze_phase ;;
  pull) pull ;;
  all) setup; extract_all; analyze_phase; pull ;;
  *) echo "usage: $0 {setup|smoke|extract|analyze|pull|all}"; exit 1 ;;
esac
