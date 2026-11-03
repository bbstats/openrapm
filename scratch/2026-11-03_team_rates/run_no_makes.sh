#!/bin/bash
# Experiments 129-131 (notes/DECISIONS.md), one after another, each line fitted on creation110's boosters and folds as
# ctx110 was, then a full build.  After a stop: rerun only the arms not finished, a chain that had started with REUSE=1,
# and stop its python children first.  Log: scratch/2026-11-03_team_rates/run_no_makes.log; each chain's outputs/<name>_chain.log.
#   nohup bash scratch/2026-11-03_team_rates/run_no_makes.sh > scratch/2026-11-03_team_rates/run_no_makes.log 2>&1 &
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
TARGET=points_with_expected_opponent_threes_late_as_scored
PROFILE=context_types_team_profile_centred
TODAY=context_and_types_centred

arm() {             # arm NAME INPUTS LINE_TARGET
  local name=$1 inputs=$2 line_target=$3
  echo "=== $(date) $name: the line, inputs $inputs, target $line_target"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -u -m openrapm second-prior --models creation110 \
    --folds creation110_early,creation110_within --tag "$name" --inputs "$inputs" --target "$line_target" \
    > "outputs/${name}_second_prior.log" 2>&1 || { echo "=== $name: the line FAILED"; exit 1; }
  echo "=== $(date) $name: the full build"
  bash scripts/experiment_chain.sh "$name" $TARGET "--second-prior $name --second-prior-inputs $inputs" \
    > "outputs/${name}_chain.nohup" 2>&1 || echo "=== $name: the chain reports a failed reading"
  echo "=== $(date) $name: done"
}

arm nomakes129 $PROFILE whole_rating
arm nomakesout130 $PROFILE prior_part
arm ctxout131 $TODAY prior_part
echo "=== $(date) all three done"
