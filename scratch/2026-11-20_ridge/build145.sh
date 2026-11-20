#!/usr/bin/env bash
# Experiment 145 (notes/DECISIONS.md): four full builds at other step-3 ridge penalties, Box Adj's line held, each
# against rd143 with noise_rd143 beside it (the chain's defaults).  A finished penalty leaves a marker in
# outputs/ridge145/done/ and is skipped on a rerun; a chain stopped half way goes on with REUSE=1.
# Log: outputs/ridge145/driver.log; each chain writes outputs/ridge<penalty>_chain.log.
#     nohup bash scratch/2026-11-20_ridge/build145.sh > /dev/null 2>&1 &
set -uo pipefail
cd /a/code/spmm || exit 1
export PYTHONIOENCODING=utf-8
DIR=outputs/ridge145
mkdir -p "$DIR/done"
LOG=$DIR/driver.log
TARGET=points_with_expected_opponent_threes_late_as_scored
say() { echo "=== $(date '+%F %I:%M:%S %p') $*" >> "$LOG"; }
say "driver start, $(git rev-parse --short HEAD)"
for penalty in 6500 4400 9000 18000; do
  name=ridge$penalty
  [ -f "$DIR/done/$name" ] && { say "skip $name"; continue; }
  reuse=0; [ -e outputs/${name}_chain.log ] && reuse=1
  say "start $name REUSE=$reuse"
  REUSE=$reuse bash scripts/experiment_chain.sh $name $TARGET "--ridge-penalty $penalty"
  status=$?
  if ! tail -n 1 outputs/${name}_chain.log | grep -q "chain done"; then say "FAILED $name: the build stopped"; exit 1; fi
  [ $status -eq 0 ] || say "$name: a reading failed, see the last line of outputs/${name}_chain.log"
  touch "$DIR/done/$name"; say "done $name"
done
say "driver finished"
