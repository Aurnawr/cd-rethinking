#!/bin/bash
cd /teamspace/studios/this_studio/cd-rethinking/CHAIR-qwen-detailed
for M in greedy vcd sid; do
  echo "===== START $M $(date) ====="
  python3 run_qwen_capture.py --method $M --n-images 500
  echo "===== END $M $(date) ====="
done
echo "ALL METHODS COMPLETE $(date)"
