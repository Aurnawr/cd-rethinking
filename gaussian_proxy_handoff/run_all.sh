#!/usr/bin/env bash
# =====================================================================
# run_all.sh -- multi-seed CHAIR runs for the Gaussian noise-proxy study.
#
# For each model (LLaVA-1.5-7B, Qwen2.5-VL-7B) it generates 500-image CHAIR
# captions under:
#   greedy  (deterministic baseline, 1 run)
#   vcd     (real Visual Contrastive Decoding)          x N seeds
#   sid     (real Self-Introspective Decoding)          x N seeds
#   proxy   (full-vocabulary Gaussian noise, no amateur) x N seeds
# then scores every run with CHAIR and reports mean +/- std across seeds.
#
# WHY: the proxy result (amateur branch is replaceable by matched noise) was a
# single unseeded run; multiple seeds give it error bars and protect it from the
# "n=1" objection. Seeds also test whether Qwen's proxy == greedy is coincidence.
#
# USAGE:
#   bash run_all.sh              # smoke test (2 imgs) then full 500-img runs
#   bash run_all.sh --smoke      # only the 2-image smoke test
#   Edit SEEDS / MODELS / METHODS below to control cost.
#
# COST WARNING: Qwen vcd/sid recompute the amateur branch cache-less at every
# step and are SLOW (hours per seed on 500 images). LLaVA is much faster.
# Start with fewer seeds if GPU time is limited. Runs are resumable: finished
# images are skipped, and a run whose _chair.json already exists is skipped.
# =====================================================================
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"
SCRIPTS="$HERE/scripts"
COCO="$HERE/data/coco/annotations"
CAPS="$HERE/outputs/caps"
mkdir -p "$CAPS"

# ---- configuration (edit these) ----
SEEDS=(0 1 2)                         # add 3 4 for five seeds
MODELS=(llava qwen)                   # comment out one to skip a model
SEED_METHODS=(vcd sid proxy)          # methods that vary with seed
PROXY_STATS=vcd                       # proxy noise matched to vcd's d (or: sid)
FULL_N=500
SMOKE_N=2
PY=${PY:-python}                      # override with:  PY=/path/to/python bash run_all.sh
# ------------------------------------

gen () {  # gen <model> <method> <seed> <n> <out>
  local model=$1 method=$2 seed=$3 n=$4 out=$5
  local script="$SCRIPTS/gen_${model}.py"
  local extra=""
  [ "$method" = "proxy" ] && extra="--stats-source $PROXY_STATS"
  "$PY" "$script" --method "$method" --seed "$seed" --n-images "$n" --out "$out" $extra
}

score () {  # score <capfile>
  "$PY" "$SCRIPTS/eval/chair_eval.py" --cap-file "$1" --coco-path "$COCO"
}

# ---------- smoke test (fast sanity check on 2 images) ----------
echo "########## SMOKE TEST (${SMOKE_N} images) ##########"
SMOKE="$HERE/outputs/smoke"; mkdir -p "$SMOKE"
for model in "${MODELS[@]}"; do
  for method in greedy "${SEED_METHODS[@]}"; do
    f="$SMOKE/${model}_${method}.jsonl"; rm -f "$f"
    echo ">>> smoke ${model} ${method}"
    gen "$model" "$method" 0 "$SMOKE_N" "$f"
    [ "$(wc -l < "$f")" -ge 1 ] || { echo "SMOKE FAILED: $f empty"; exit 1; }
  done
done
echo "########## SMOKE TEST PASSED ##########"
[ "${1:-}" = "--smoke" ] && { echo "smoke-only mode, stopping."; exit 0; }

# ---------- full runs ----------
echo "########## FULL RUNS (${FULL_N} images) ##########"
for model in "${MODELS[@]}"; do
  # greedy: deterministic, one run (seed 0)
  out="$CAPS/${model}_greedy_seed0.jsonl"
  if [ ! -f "${out%.jsonl}_chair.json" ]; then
    echo ">>> ${model} greedy (seed 0)"; gen "$model" greedy 0 "$FULL_N" "$out"; score "$out"
  fi
  for seed in "${SEEDS[@]}"; do
    for method in "${SEED_METHODS[@]}"; do
      tag="${method}"; [ "$method" = "proxy" ] && tag="proxy_${PROXY_STATS}"
      out="$CAPS/${model}_${tag}_seed${seed}.jsonl"
      if [ -f "${out%.jsonl}_chair.json" ]; then echo "skip (done): $out"; continue; fi
      echo ">>> ${model} ${method} seed ${seed}"
      gen "$model" "$method" "$seed" "$FULL_N" "$out"
      score "$out"
    done
  done
done

# ---------- aggregate ----------
echo "########## AGGREGATE ##########"
"$PY" "$SCRIPTS/aggregate_seeds.py"
echo "ALL DONE. See outputs/aggregate.json"
