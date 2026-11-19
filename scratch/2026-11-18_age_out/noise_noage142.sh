#!/usr/bin/env bash
# The noise control on the incumbent of experiment 142, noise_noage142: the chain with --fold-assignment 1 against
# noage142, on the shipped settings, which have named the line without age since the switch (b6e62be).
# Log: outputs/noise_noage142_chain.log; outputs/age_out/noise_noage142.out says how it ended.
#     nohup bash scratch/2026-11-18_age_out/noise_noage142.sh > /dev/null 2>&1 &
cd /a/code/spmm || exit 1
export PYTHONIOENCODING=utf-8
export INC_NAME=noage142 INC_SYSTEM=noage142 INC_FOLDS=noage142_early,noage142_within CONTROL_NAME=noise_noage142
OUT=outputs/age_out/noise_noage142.out
echo "=== $(date '+%F %I:%M:%S %p') start" >> "$OUT"
REUSE=${REUSE:-0} bash scripts/experiment_chain.sh noise_noage142 points_with_expected_opponent_threes_late_as_scored \
    "--fold-assignment 1" >> "$OUT" 2>&1
echo "=== $(date '+%F %I:%M:%S %p') ended, status $?" >> "$OUT"
