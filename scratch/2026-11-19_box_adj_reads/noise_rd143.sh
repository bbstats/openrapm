#!/usr/bin/env bash
# The noise control on the incumbent of experiment 143, noise_rd143: the chain with --fold-assignment 1 against
# rd143, on the shipped settings, which have named the line with the two defense inputs since the switch (22ebe90).
# Log: outputs/noise_rd143_chain.log; outputs/rd143/noise_rd143.out says how it ended.
#     nohup bash scratch/2026-11-19_box_adj_reads/noise_rd143.sh > /dev/null 2>&1 &
cd /a/code/spmm || exit 1
export PYTHONIOENCODING=utf-8
export INC_NAME=rd143 INC_SYSTEM=rd143 INC_FOLDS=rd143_early,rd143_within CONTROL_NAME=noise_rd143
OUT=outputs/rd143/noise_rd143.out
echo "=== $(date '+%F %I:%M:%S %p') start" >> "$OUT"
REUSE=${REUSE:-0} bash scripts/experiment_chain.sh noise_rd143 points_with_expected_opponent_threes_late_as_scored \
    "--fold-assignment 1" >> "$OUT" 2>&1
echo "=== $(date '+%F %I:%M:%S %p') ended, status $?" >> "$OUT"
