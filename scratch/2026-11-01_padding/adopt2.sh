#!/bin/bash
# Adoption of experiment 127, after the crash of 2026-11-01 (about 6:04 PM, a memory stick came unseated; rebooted
# 6:31 PM).  The panels finished before it and match the candidate's to the bit, also when rebuilt after it.  The
# unfinished tenth folds are set aside and built fresh on the clean machine, and compared with what the crashed run
# had finished, which says whether the numbers of the hours before the crash can be trusted; the reproduction check of
# every season proves the candidate's boosters and step-3 table.  Then today's published tables are kept aside, the
# full build of 1997-2027, the site's files.  A finished stage is skipped on a rerun.  Log: outputs/adopt127.log.
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
LOG=outputs/adopt127.log
export OPENRAPM_JOBS=4
stage() { echo "=== $(date) $1" >> $LOG; }

if [ -d outputs/within/padbins127_k10 ] && [ ! -d outputs/within/padbins127_k10_crash ]; then
  stage "the crashed run's tenth folds set aside as padbins127_k10_crash"
  mv outputs/within/padbins127_k10 outputs/within/padbins127_k10_crash
fi
if [ ! -f outputs/within/padbins127_k10/summary.parquet ]; then
  stage "the tenth folds padbins127_k10, fresh"
  $PY -u -m openrapm folds --models padbins127 --base season_ratings_padbins127_step3 --seasons 1997-2026 \
      --tag padbins127_k10 --folds-per-repeat 10 --jobs 4 > outputs/padbins127_k10.log 2>&1 || { stage "tenth folds FAILED"; exit 1; }
fi
stage "the crashed run against the fresh one"
$PY -u scratch/2026-11-01_padding/compare_k10.py >> $LOG 2>&1 || { stage "the crashed run's numbers DIFFER: stopped"; exit 1; }
if [ ! -f outputs/season_ratings_product_before_padbins127.parquet ]; then
  stage "the published tables kept as season_ratings_product_before_padbins127"
  for suffix in _step3 _step4 ""; do
    cp outputs/season_ratings_product$suffix.parquet outputs/season_ratings_product_before_padbins127$suffix.parquet
  done
  cp outputs/csv/season_ratings_product_step4_multipliers.csv outputs/csv/season_ratings_product_before_padbins127_step4_multipliers.csv
fi
if [ outputs/season_ratings_product.parquet -ot outputs/within/padbins127_k10/summary.parquet ]; then
  stage "the full build of 1997-2027"
  $PY -u -m openrapm build > outputs/build_publish_2026-11-02.log 2>&1 || { stage "the full build FAILED"; exit 1; }
fi
stage "the site's files"
$PY -u -m openrapm site --bias --openrapm-decomposition --similarity --power > outputs/site_publish_2026-11-02.log 2>&1 || { stage "the site FAILED"; exit 1; }
stage "adoption's builds done"
