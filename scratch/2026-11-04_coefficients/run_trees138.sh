#!/bin/bash
# Experiment 138 (notes/DECISIONS.md): defense's boosters at 330 trees, offense's as they are.  Stages, each skipped on a
# rerun when its output exists:
#   1. the full build trees138 on today's Box Adj line, for its boosters and within-season folds
#   2. Box Adj's line refitted on them: `openrapm second-prior --target rating`
#   3. the full build trees138_line on that line, the candidate
#   nohup bash scratch/2026-11-04_coefficients/run_trees138.sh > scratch/2026-11-04_coefficients/run_trees138.log 2>&1 &
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
NAME=trees138
LINE=trees138_line
INPUTS=context_types_team_rates_centred
TARGET=points_with_expected_opponent_threes_late_as_scored
export OPENRAPM_JOBS=4
stage() { echo "=== $(date) $1"; }

if [ ! -f outputs/csv/wowy_${NAME}_eras.csv ]; then
  stage "the boosters at 330 trees and their folds"
  bash scripts/experiment_chain.sh $NAME $TARGET "--booster-params $NAME" \
      > outputs/${NAME}_chain.nohup 2>&1 || stage "the first build reports a failure"
fi
if [ ! -f outputs/prior_models_${NAME}.pkl ] || [ ! -d outputs/within/${NAME}_early ] || [ ! -d outputs/within/${NAME}_within ]; then
  stage "FAILED: the first build left no boosters or folds"
  exit 1
fi
if [ ! -f outputs/second_prior/$LINE/run.json ]; then
  stage "Box Adj's line"
  $PY -u -m openrapm second-prior --models $NAME --folds ${NAME}_early,${NAME}_within --tag $LINE \
      --inputs $INPUTS --target rating > outputs/${LINE}_second_prior.log 2>&1 || { stage "FAILED: the line"; exit 1; }
fi
if [ ! -f outputs/csv/wowy_${LINE}_eras.csv ]; then
  stage "the candidate's full build"
  bash scripts/experiment_chain.sh $LINE $TARGET "--booster-params $NAME --second-prior $LINE --second-prior-inputs $INPUTS" \
      > outputs/${LINE}_chain.nohup 2>&1 || stage "the candidate's build reports a failure"
fi
stage "all done"
