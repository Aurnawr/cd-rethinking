#!/usr/bin/env bash
# =====================================================================
# run_all.sh -- multi-seed CHAIR runs for the Gaussian noise-proxy study.
#
# DEFAULT PLAN (fits a ~7 GPU-hour budget on one L4):
#   For each model (LLaVA-1.5-7B, Qwen2.5-VL-7B):
#     greedy   (deterministic baseline, 1 run)      -- cheap, optional
#     proxy    (full-vocabulary Gaussian noise)      x 3 seeds
#   then scores every run with CHAIR and reports mean/range across seeds.
#
# VCD and SID are intentionally NOT run here: they are expensive (Qwen's
# amateur branch is cache-less, hours per seed) and their single-run CHAIR
# numbers + image-level bootstrap CIs already exist in the paper. This package
# only produces the NEW result -- the seeded full-vocabulary proxy.
#
# WHY: the proxy result (amateur branch is replaceable by matched noise) was a
# single unseeded run; 3 seeds give it reproducibility and kill the "n=1"
# objection, at only a few GPU-hours because the proxy is greedy-speed.
#
# USAGE:
#   bash run_all.sh              # smoke test (2 imgs) then full proxy runs
#   bash run_all.sh --smoke      # only the 2-image smoke test
# TUNING (edit below): SEEDS, MODELS, RUN_GREEDY, PROXY_STATS.
# If GPU time runs short, set SEEDS=(0 1) or RUN_GREEDY=0.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
SCRIPTS="$HERE/scripts"
COCO="$HERE/data/coco/annotations"
CAPS="$HERE/outputs/caps"
mkdir -p "$CAPS"

# ---- configuration ----
SEEDS=(0 1 2)                 # 3 seeds for the proxy; drop to (0 1) if time is short
MODELS=(llava qwen)           # both models
RUN_GREEDY=1                  # 1 = also run greedy baseline (cheap); 0 = skip
PROXY_STATS=vcd               # proxy noise matched to VCD's measured d (or: sid)
FULL_N=500
SMOKE_N=2
PY=${PY:-python}              # override:  PY=/path/to/python bash run_all.sh
# -----------------------

gen () {  # gen <model> <method> <seed> <n> <out>
  local model=$1 method=$2 seed=$3 n=$4 out=$5
  local extra=""
  [ "$method" = "proxy" ] && extra="--stats-source $PROXY_STATS"
  "$PY" "$SCRIPTS/gen_${model}.py" --method "$method" --seed "$seed" --n-images "$n" --out "$out" $extra
}
score () { "$PY" "$SCRIPTS/eval/chair_eval.py" --cap-file "$1" --coco-path "$COCO"; }

# ---------- smoke test (fast sanity on 2 images: base path + noise path) ----------
echo "########## SMOKE TEST (${SMOKE_N} images) ##########"
SMOKE="$HERE/outputs/smoke"; mkdir -p "$SMOKE"
for model in "${MODELS[@]}"; do
  for method in greedy proxy; do
    f="$SMOKE/${model}_${method}.jsonl"; rm -f "$f"
    echo ">>> smoke ${model} ${method}"
    gen "$model" "$method" 0 "$SMOKE_N" "$f"
    [ "$(wc -l < "$f")" -ge 1 ] || { echo "SMOKE FAILED: $f empty"; exit 1; }
  done
done
echo "########## SMOKE TEST PASSED ##########"
echo "   (per-image time above x 500 = approx per-seed cost; confirm it fits your budget)"
[ "${1:-}" = "--smoke" ] && { echo "smoke-only mode, stopping."; exit 0; }

# ---------- full runs: greedy (once) + proxy (N seeds) ----------
echo "########## FULL RUNS (${FULL_N} images) ##########"
for model in "${MODELS[@]}"; do
  if [ "$RUN_GREEDY" = "1" ]; then
    out="$CAPS/${model}_greedy_seed0.jsonl"
    if [ ! -f "${out%.jsonl}_chair.json" ]; then
      echo ">>> ${model} greedy (baseline)"; gen "$model" greedy 0 "$FULL_N" "$out"; score "$out"
    fi
  fi
  for seed in "${SEEDS[@]}"; do
    out="$CAPS/${model}_proxy_${PROXY_STATS}_seed${seed}.jsonl"
    if [ -f "${out%.jsonl}_chair.json" ]; then echo "skip (done): $out"; continue; fi
    echo ">>> ${model} proxy seed ${seed}"
    gen "$model" proxy "$seed" "$FULL_N" "$out"
    score "$out"
  done
done

# ---------- aggregate ----------
echo "########## AGGREGATE ##########"
"$PY" "$SCRIPTS/aggregate_seeds.py"
echo "ALL DONE. Send back: outputs/aggregate.json  and  outputs/caps/*_chair.json"

# ---------------------------------------------------------------------
# To ALSO seed VCD/SID (expensive; not recommended under a 7h budget), add
# them to a seed loop, e.g.:  for m in vcd sid; do gen "$model" "$m" "$seed" ...
# gen_{llava,qwen}.py already implement vcd and sid.
# ---------------------------------------------------------------------
