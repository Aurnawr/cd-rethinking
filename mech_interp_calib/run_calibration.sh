#!/usr/bin/env bash
# Run both calibration experiments for every (model, method) pair that has stored margins,
# then build the cross-model comparison. CPU only, no GPU and no model weights: everything
# reads the per-sample logit-lens margins already committed under results/<model>/.
#
#   bash mech_interp_calib/run_calibration.sh
#   MODELS="qwen2.5-vl-7b" bash mech_interp_calib/run_calibration.sh   # one model
#
# Requires numpy, scikit-learn and matplotlib. Roughly 4 minutes for 6 pairs; the bootstrap
# CIs (2,000 replicates at the readout plus 500 per layer) dominate the runtime.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-python3}"

MODELS="${MODELS:-llava1.5-7b qwen2.5-vl-7b}"
METHODS="${METHODS:-vcd icd sid}"
OUT="${OUT:-$ROOT/results/calibration}"

export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
mkdir -p "$OUT"

for model in $MODELS; do
  for method in $METHODS; do
    npz="$ROOT/results/$model/logit_lens_per_sample_${method}.npz"
    if [[ ! -f "$npz" ]]; then
      echo "[skip] $model/$method: no $npz"
      continue
    fi
    echo "=== $model / $method ==="
    "$PY" "$HERE/oracle_calibration.py" --npz "$npz" \
        --model-tag "$model" --method "$method" --out-dir "$OUT"
    "$PY" "$HERE/scalar_bias.py" --npz "$npz" \
        --model-tag "$model" --method "$method" --out-dir "$OUT"
  done
done

echo "=== comparison ==="
"$PY" "$HERE/compare_models.py" --dir "$OUT"
echo "[done] artifacts in $OUT"
