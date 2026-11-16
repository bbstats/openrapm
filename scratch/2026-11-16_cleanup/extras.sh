#!/usr/bin/env bash
# The after side's two extras: a saved game-by-game model is refused by name; a fresh two-season save reproduces.
# Run from A:/code/spmm_cleanup once side a has finished.
export PYTHONPATH=/a/code/spmm_cleanup/src
export OPENRAPM_CONFIG=A:/code/spmm/outputs/cleanup_proof/config.yaml
PY=/a/code/spmm/.venv/Scripts/python
LOG=/a/code/spmm/outputs/cleanup_proof/extras.log
: > "$LOG"

echo "== 1. net1, a game-by-game build, must be refused" >> "$LOG"
$PY -m openrapm folds --models net1 --base season_ratings_net1_step3 --seasons 2026 --tag cu_a_net1 \
  --reproduce-only >> "$LOG" 2>&1
echo "exit $?" >> "$LOG"

echo "== 2. a fresh two-season save, then its reproduction" >> "$LOG"
$PY -m openrapm rate --setting test --seasons 2001,2026 --name cu_a_save --save-models --jobs 2 >> "$LOG" 2>&1
echo "rate exit $?" >> "$LOG"
$PY -m openrapm folds --models cu_a_save --base season_ratings_cu_a_save_step3 --seasons 2001,2026 \
  --tag cu_a_save --reproduce-only >> "$LOG" 2>&1
echo "folds exit $?" >> "$LOG"
echo "extras finished" >> "$LOG"
