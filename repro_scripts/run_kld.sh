#!/bin/bash
# run_kld_experiment.sh
# Runs the KLD experiment for VCD, ICD, and SID on LLaVA-Bench,
# then plots the results.

MODEL_PATH="/teamspace/lightning_storage/data/models/llava-v1.5-7b"   # <-- update this
MODEL_BASE=None
CONV_MODE="vicuna_v1"
OUTPUT_DIR="./repro_outputs/kld_experiment/visit-bench"

mkdir -p "$OUTPUT_DIR"

## vcd
python ./inference/kld_experiment.py \
    --model-path "$MODEL_PATH" \
    --conv-mode  "$CONV_MODE" \
    --method     vcd \
    --output-file "$OUTPUT_DIR/vcd_kld.json" \
    --noise-step 900 \
    --temperature 0.2 \
    --max-new-tokens 128

## icd
python ./inference/kld_experiment.py \
    --model-path "$MODEL_PATH" \
    --conv-mode  "$CONV_MODE" \
    --method     icd \
    --output-file "$OUTPUT_DIR/icd_kld.json" \
    --temperature 0.2 \
    --max-new-tokens 128

## sid
python ./inference/kld_experiment.py \
    --model-path "$MODEL_PATH" \
    --conv-mode  "$CONV_MODE" \
    --method     sid \
    --output-file "$OUTPUT_DIR/sid_kld.json" \
    --temperature 0.2 \
    --max-new-tokens 128