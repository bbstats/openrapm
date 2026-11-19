#!/usr/bin/env bash
# Experiment 143's full build (notes/DECISIONS.md, "Experiment 143"; the owner, 2026-11-19: "Two inputs first"):
#   1. Box Adj's line on the list with the two defense inputs, rd143_line, from rebuilt_base's held-back games
#   2. its check against the read: the screen's sums, and the formula's -0.102
#   3. the chain against noage142, noise_noage142 beside it (the chain's defaults)
#   4. the year-over-year test with each side's width matched to the incumbent's
# A step whose marker is in outputs/rd143/done/ is skipped on a rerun; a chain stopped half way goes on with REUSE=1.
# Log: outputs/rd143/driver.log; the chain writes outputs/rd143_chain.log.
#     nohup bash scratch/2026-11-19_box_adj_reads/build143.sh > /dev/null 2>&1 &
set -uo pipefail
cd /a/code/spmm || exit 1
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
DIR=outputs/rd143
mkdir -p "$DIR/done"
LOG=$DIR/driver.log
TARGET=points_with_expected_opponent_threes_late_as_scored
LIST=context_types_no_age_team_zones_defence_only_centred
say() { echo "=== $(date '+%F %I:%M:%S %p') $*" >> "$LOG"; }

say "driver start, $(git rev-parse --short HEAD)"
if [ ! -f "$DIR/done/line" ]; then
  say "start line"
  $PY -u -m openrapm second-prior --models rebuilt_base --folds rebuilt_base_early,rebuilt_base_within \
      --tag rd143_line --target rating --inputs $LIST >> "$LOG" 2>&1 || { say "FAILED line"; exit 1; }
  touch "$DIR/done/line"; say "done line"
fi
if [ ! -f "$DIR/done/check" ]; then
  $PY -u scratch/2026-11-19_box_adj_reads/line143_check.py >> "$LOG" 2>&1 || { say "FAILED check"; exit 1; }
  touch "$DIR/done/check"; say "done check"
fi
if [ ! -f "$DIR/done/rd143" ]; then
  reuse=0; [ -e outputs/rd143_chain.log ] && reuse=1
  say "start chain REUSE=$reuse"
  REUSE=$reuse bash scripts/experiment_chain.sh rd143 $TARGET "--second-prior rd143_line --second-prior-inputs $LIST"
  status=$?
  if ! tail -n 1 outputs/rd143_chain.log | grep -q "chain done"; then say "FAILED rd143: the build stopped"; exit 1; fi
  [ $status -eq 0 ] || say "rd143: a reading failed, see the last line of outputs/rd143_chain.log"
  touch "$DIR/done/rd143"; say "done rd143"
fi
if [ ! -f "$DIR/done/matched" ]; then
  say "start matched"
  $PY -u scratch/2026-11-19_box_adj_reads/matched143.py >> "$LOG" 2>&1 || { say "FAILED matched"; exit 1; }
  $PY -u scripts/checks/63_yoy.py --rankings=matched143=outputs/season_ratings_matched143.parquet,incumbent=outputs/season_ratings_noage142.parquet \
      --ref=incumbent --tag=matched143 --splits= >> "$LOG" 2>&1 || { say "FAILED matched year over year"; exit 1; }
  touch "$DIR/done/matched"; say "done matched"
fi
say "driver finished"
