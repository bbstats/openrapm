#!/bin/bash
# Experiment 118's second look: check 155 with kumo_small_equal (every training row
# the same chance to be shown), one random draw, one rating process: two filled the 8 GB of GPU memory with medium.
#   nohup bash scratch/2026-10-28_kumo/equal.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/../.."

LOG=outputs/screen/screen112/kumo_small_equal.log
echo "=== $(date) kumo_small_equal, draw 0" >> $LOG
.venv/Scripts/python -u scripts/checks/155_input_screen.py run --tag screen112 --variant kumo_small_equal \
  --base-seeds 1 --workers 1 --fitters 4 >> $LOG 2>&1 || echo "=== $(date) FAILED" >> $LOG
