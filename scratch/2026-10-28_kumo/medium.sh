#!/bin/bash
# Experiment 118's medium again: check 155 with kumo_medium (training rows drawn by weight
# as kumo_small), one random draw, one rating process: two filled the 8 GB of GPU memory.
#   nohup bash scratch/2026-10-28_kumo/medium.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/../.."

LOG=outputs/screen/screen112/kumo_medium.log
echo "=== $(date) kumo_medium, draw 0" >> $LOG
.venv/Scripts/python -u scripts/checks/155_input_screen.py run --tag screen112 --variant kumo_medium \
  --base-seeds 1 --workers 1 --fitters 4 >> $LOG 2>&1 || echo "=== $(date) FAILED" >> $LOG
