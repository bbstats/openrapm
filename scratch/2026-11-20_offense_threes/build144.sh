#!/usr/bin/env bash
# Experiment 144's full build (notes/DECISIONS.md, "Experiment 144"): the offense on the defense's target, three quarters
# of the shooters' three-point luck taken out, by README.md's passes:
#   1. ratings without Box Adj on the new target, saved models off3_144_base, and their held-back games
#   2. Box Adj's line off3_144_line on today's input list, fitted on those held-back games
#   3. the chain off3_144 against rd143, noise_rd143 beside it (the chain's defaults)
#   4. the year-over-year test with each side's width matched to the incumbent's
# A step whose marker is in outputs/off3_144/done/ is skipped on a rerun; a chain stopped half way goes on with REUSE=1.
# Log: outputs/off3_144/driver.log; the chain writes outputs/off3_144_chain.log.
#     nohup bash scratch/2026-11-20_offense_threes/build144.sh > /dev/null 2>&1 &
set -uo pipefail
cd /a/code/spmm || exit 1
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
DIR=outputs/off3_144
mkdir -p "$DIR/done"
LOG=$DIR/driver.log
TARGET=points_with_expected_opponent_threes_late_as_scored
LIST=context_types_no_age_team_zones_defence_only_centred
say() { echo "=== $(date '+%F %I:%M:%S %p') $*" >> "$LOG"; }
step() {
  local name=$1; shift
  [ -f "$DIR/done/$name" ] && { say "skip $name"; return 0; }
  say "start $name: $*"
  $PY -u -m openrapm "$@" >> "$LOG" 2>&1 || { say "FAILED $name"; exit 1; }
  touch "$DIR/done/$name"; say "done $name"
}
say "driver start, $(git rev-parse --short HEAD)"
step base_rate   rate --setting test --seasons 1997-2026 --second-prior "" --offensive-target $TARGET \
                 --name off3_144_base --save-models off3_144_base --jobs 4
step base_early  folds --models off3_144_base --base season_ratings_off3_144_base_step3 --seasons 1997-2016 \
                 --tag off3_144_base_early --jobs 4
step base_within folds --models off3_144_base --base season_ratings_off3_144_base_step3 --seasons 2017-2026 \
                 --tag off3_144_base_within --jobs 4
step line        second-prior --models off3_144_base --folds off3_144_base_early,off3_144_base_within \
                 --tag off3_144_line --target rating --inputs $LIST
if [ ! -f "$DIR/done/chain" ]; then
  reuse=0; [ -e outputs/off3_144_chain.log ] && reuse=1
  say "start chain REUSE=$reuse"
  REUSE=$reuse bash scripts/experiment_chain.sh off3_144 $TARGET \
      "--offensive-target $TARGET --second-prior off3_144_line --second-prior-inputs $LIST"
  status=$?
  if ! tail -n 1 outputs/off3_144_chain.log | grep -q "chain done"; then say "FAILED chain: the build stopped"; exit 1; fi
  [ $status -eq 0 ] || say "chain: a reading failed, see the last line of outputs/off3_144_chain.log"
  touch "$DIR/done/chain"; say "done chain"
fi
if [ ! -f "$DIR/done/matched" ]; then
  say "start matched"
  $PY -u scratch/2026-11-20_offense_threes/matched144.py >> "$LOG" 2>&1 || { say "FAILED matched"; exit 1; }
  $PY -u scripts/checks/63_yoy.py --rankings=matched144=outputs/season_ratings_matched144.parquet,incumbent=outputs/season_ratings_rd143.parquet \
      --ref=incumbent --tag=matched144 --splits= >> "$LOG" 2>&1 || { say "FAILED matched year over year"; exit 1; }
  touch "$DIR/done/matched"; say "done matched"
fi
say "driver finished"
