#!/bin/bash
# Experiment 118's second look, once medium's first draw is done: check 155 with kumo_small_equal (every training row
# the same chance to be shown), one random draw.
#   nohup bash scratch/2026-10-28_kumo/equal.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/../.."
while ps -ef | grep -q "[v]ariant kumo_medium"; do sleep 30; done
LOG=outputs/screen/screen112/kumo_small_equal.log
echo "=== $(date) kumo_small_equal, draw 0" >> $LOG
.venv/Scripts/python -u scripts/checks/155_input_screen.py run --tag screen112 --variant kumo_small_equal \
  --base-seeds 1 --workers 2 --fitters 4 >> $LOG 2>&1 || echo "=== $(date) FAILED" >> $LOG
