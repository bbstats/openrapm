#!/bin/bash
# Experiment 123 (the owner, 2026-10-30, the bug-fix pass's proposal: "steadier multiplier"): the shrink's within-team
# layer fitted on every fold season the test rule allows instead of the 8 nearest.  Only the shrink changes, so the
# candidate takes today's ratings before the shrink (creation110's test table) and today's held-back game sets, and
# its comparison with today's carries no regrouping noise.  A finished stage is skipped on a rerun.
#   1. the shrink with the setting off reproduces creation110's test tables (the control)
#   2. the candidate: --within-nearest 0
#   3. the within-season WOWY test by era, both systems on creation110's folds, each with its own shrink
#   4. the year-over-year test; 5. the 2026 top 20
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
LOG=outputs/steady123_chain.log
FOLDS=creation110_early,creation110_within
FLAGS="--within-nearest 0"
export OPENRAPM_JOBS=4
echo "=== $(date) chain start: steady123 (experiment 123) FLAGS=$FLAGS" >> $LOG
if [ ! -f outputs/season_ratings_control123_step4.parquet ]; then
  echo "=== $(date) the control: the shrink at today's settings on creation110's table" >> $LOG
  $PY -u -m openrapm shrink --base season_ratings_creation110_step3 --setting test --folds $FOLDS --extrapolate-folds= \
      --name control123 >> $LOG 2>&1 || exit 1
fi
if [ ! -f outputs/season_ratings_steady123.parquet ]; then
  echo "=== $(date) the candidate's shrink and swap step" >> $LOG
  $PY -u -m openrapm shrink --base season_ratings_creation110_step3 --setting test --folds $FOLDS --extrapolate-folds= \
      $FLAGS --name steady123 >> $LOG 2>&1 || exit 1
  $PY -u -m openrapm swap --base season_ratings_steady123_step4 --setting test --folds $FOLDS --extrapolate-folds= \
      $FLAGS --name steady123 --types >> $LOG 2>&1 || exit 1
fi
echo "=== $(date) the within-season WOWY test (151)" >> $LOG
$PY -u scripts/checks/151_wowy.py --systems=creation110,steady123=creation110 --ref=creation110 --tag=steady123 \
    "--shrink-flags=steady123:$FLAGS" >> $LOG 2>&1 || exit 1
echo "=== $(date) the year-over-year test (63)" >> $LOG
$PY -u scripts/checks/63_yoy.py --rankings=steady123=outputs/season_ratings_steady123.parquet,incumbent=outputs/season_ratings_creation110.parquet \
    --ref=incumbent --tag=steady123 --splits= >> $LOG 2>&1 || exit 1
echo "=== $(date) top 20" >> $LOG
$PY -u scripts/checks/66_compare.py incumbent=outputs/season_ratings_creation110.parquet \
    steady123=outputs/season_ratings_steady123.parquet --season=2026 --top=20 --yoy=outputs/yoy_steady123.parquet \
    --ref=incumbent --open=0 >> $LOG 2>&1 || exit 1
echo "=== $(date) chain done" >> $LOG
