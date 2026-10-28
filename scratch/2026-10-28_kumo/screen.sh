#!/bin/bash
# Experiment 118's screen: check 155's within-season WOWY test on the 10 test seasons with the Kumo prior in place of
# LightGBM, small and medium, one random draw of each first, then two more of each, then every variant against the
# shipped prior.  Each run resumes, skipping the trials already in trials_<variant>.csv.
#   nohup bash scratch/2026-10-28_kumo/screen.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
OUT=outputs/screen/screen112
for draws in 1 3; do
  for size in small medium; do
    echo "=== $(date) kumo_$size, draws 0-$((draws - 1))" >> $OUT/kumo_$size.log
    $PY -u scripts/checks/155_input_screen.py run --tag screen112 --variant kumo_$size --base-seeds $draws \
      --workers 2 --fitters 4 >> $OUT/kumo_$size.log 2>&1 || { echo "=== $(date) FAILED" >> $OUT/kumo_$size.log; exit 1; }
  done
done
$PY -u scripts/checks/155_input_screen.py variants --tag screen112 > $OUT/kumo_variants.log 2>&1
echo "=== $(date) done" >> $OUT/kumo_variants.log
