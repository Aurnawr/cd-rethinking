#!/bin/bash
cd /teamspace/studios/this_studio/cd-rethinking/CHAIR-qwen-detailed
# wait until the capture chain reports completion
until grep -q "ALL METHODS COMPLETE" run_all_qwen.log 2>/dev/null; do sleep 300; done
sleep 30   # let GPU free after the last python exits
echo "=== running post-run verification $(date) ===" > verify.log
python3 verify_qwen.py >> verify.log 2>&1
echo "=== VERIFY WAITER DONE $(date) ===" >> verify.log
