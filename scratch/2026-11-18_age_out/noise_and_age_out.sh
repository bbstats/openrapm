#!/usr/bin/env bash
# After rebuilt_scores.sh (notes/DECISIONS.md, "The rebuild read on the two tests" and "Experiment 142"):
#   1. the noise control on the rebuild, noise_rebuilt: the chain with --fold-assignment 1 against rebuilt
#   2. Box Adj's line on the list without age, noage142_line, from rebuilt_base's held-back games
#   3. experiment 142, noage142: the chain against rebuilt with noise_rebuilt beside it
#   4. the year-over-year test again with each side's width matched to the incumbent's
# A step whose marker is in outputs/age_out/done/ is skipped on a rerun; a chain stopped half way goes on with REUSE=1.
# The chain's incumbent is named here and not in scripts/experiment_chain.sh, which no one edits while a build runs.
# Log: outputs/age_out/driver.log; each chain writes outputs/<name>_chain.log.
#     nohup bash scratch/2026-11-18_age_out/noise_and_age_out.sh > /dev/null 2>&1 &
set -uo pipefail
cd /a/code/spmm || exit 1
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
DIR=outputs/age_out
mkdir -p "$DIR/done"
LOG=$DIR/driver.log
TARGET=points_with_expected_opponent_threes_late_as_scored
LIST=context_types_no_age_team_zones_centred
export INC_NAME=rebuilt INC_SYSTEM=rebuilt INC_FOLDS=rebuilt_early,rebuilt_within CONTROL_NAME=noise_rebuilt
say() { echo "=== $(date '+%F %I:%M:%S %p') $*" >> "$LOG"; }

chain() {             # chain NAME "EXTRA RATE FLAGS": one full build and its readings, resumed if it was stopped
  local name=$1 extra=$2 reuse=0
  [ -f "$DIR/done/$name" ] && { say "skip $name"; return 0; }
  [ -e "outputs/${name}_chain.log" ] && reuse=1
  say "start $name ($extra) REUSE=$reuse"
  REUSE=$reuse bash scripts/experiment_chain.sh "$name" $TARGET "$extra"
  local status=$?
  if ! tail -n 1 "outputs/${name}_chain.log" | grep -q "chain done"; then say "FAILED $name: the build stopped"; exit 1; fi
  [ $status -eq 0 ] || say "$name: a reading failed, see the last line of outputs/${name}_chain.log"
  touch "$DIR/done/$name"; say "done $name"
}

say "driver start, $(git rev-parse --short HEAD)"
chain noise_rebuilt "--fold-assignment 1"
if [ ! -f "$DIR/done/line" ]; then
  say "start line"
  $PY -u -m openrapm second-prior --models rebuilt_base --folds rebuilt_base_early,rebuilt_base_within \
      --tag noage142_line --target rating --inputs $LIST >> "$LOG" 2>&1 || { say "FAILED line"; exit 1; }
  touch "$DIR/done/line"; say "done line"
fi
chain noage142 "--second-prior noage142_line --second-prior-inputs $LIST"
if [ ! -f "$DIR/done/matched" ]; then
  say "start matched"
  $PY -u scratch/2026-11-18_age_out/noage142_matched.py >> "$LOG" 2>&1 || { say "FAILED matched"; exit 1; }
  $PY -u scripts/checks/63_yoy.py --rankings=matched142=outputs/season_ratings_matched142.parquet,incumbent=outputs/season_ratings_rebuilt.parquet \
      --ref=incumbent --tag=matched142 --splits= >> "$LOG" 2>&1 || { say "FAILED matched year over year"; exit 1; }
  touch "$DIR/done/matched"; say "done matched"
fi
say "driver finished"
