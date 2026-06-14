#!/bin/bash
# Exit immediately if any script fails
set -e 

# echo "Starting Baseline..."
# bash ./repro_scripts/inference_baseline.sh

echo "Starting VCD and SID..."
bash ./repro_scripts/inference_cd.sh 

echo "Starting Spurious..."
bash ./repro_scripts/inference_spurious.sh 

echo "All inference runs completed successfully!"