#!/usr/bin/env bash
# The rebuild's test tables and its two readings (notes/DECISIONS.md, "The rebuild read on the two tests, its noise
# control, and the chain's incumbent", 2026-11-18): step 4 and step 5 at the test setting on the rebuild's step-3
# table, as scripts/experiment_chain.sh builds a candidate's, then check 151 and check 63 on the rebuild alone, for
# its absolute scores.  Log: outputs/rebuilt_scores.log.
#     nohup bash scratch/2026-11-18_age_out/rebuilt_scores.sh > /dev/null 2>&1 &
set -uo pipefail
cd /a/code/spmm || exit 1
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
LOG=outputs/rebuilt_scores.log
FOLDS=rebuilt_early,rebuilt_within
say() { echo "=== $(date '+%F %I:%M:%S %p') $*" >> "$LOG"; }
say "start"
$PY -u -m openrapm shrink --base season_ratings_rebuilt_step3 --setting test --folds $FOLDS --extrapolate-folds= \
    --name rebuilt >> "$LOG" 2>&1 || { say "FAILED shrink"; exit 1; }
say "shrink done"
$PY -u -m openrapm swap --base season_ratings_rebuilt_step4 --setting test --folds $FOLDS --extrapolate-folds= \
    --name rebuilt --types >> "$LOG" 2>&1 || { say "FAILED swap"; exit 1; }
say "swap done"
$PY -u scripts/checks/151_wowy.py --systems=rebuilt --ref=rebuilt --tag=rebuilt >> "$LOG" 2>&1 || say "FAILED 151"
say "151 done"
$PY -u scripts/checks/63_yoy.py --rankings=rebuilt=outputs/season_ratings_rebuilt.parquet --ref=rebuilt --tag=rebuilt \
    --splits= >> "$LOG" 2>&1 || say "FAILED 63"
say "finished"
