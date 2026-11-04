#!/bin/bash
# Experiments 132-133 (notes/DECISIONS.md): full builds on Box Adj lines fitted through the rating, one after the other.
#   nohup bash scratch/2026-11-03_direct/run_direct.sh > scratch/2026-11-03_direct/run_direct.log 2>&1 &
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
TARGET=points_with_expected_opponent_threes_late_as_scored

arm() {             # arm NAME INPUTS
  local name=$1 inputs=$2
  echo "=== $(date) $name: the full build, inputs $inputs"
  bash scripts/experiment_chain.sh "$name" $TARGET "--second-prior $name --second-prior-inputs $inputs" \
    > "outputs/${name}_chain.nohup" 2>&1 || echo "=== $name: the chain reports a failed reading"
  echo "=== $(date) $name: done"
}

arm direct132 context_and_types_centred
arm direct133 context_types_team_rates_centred
echo "=== $(date) both done"
