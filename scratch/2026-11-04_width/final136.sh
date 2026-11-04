#!/bin/bash
# The final tables at the owner's choice of 2026-11-04: experiment 134's line with the halfway width settings, run from a
# copy of the code whose defaults already are those settings, so the regular-season stage's child processes read them too.
# Nothing here touches the published tables or the working tree's settings, which the morning job runs on.
cd /a/code/spmm
export PYTHONPATH="C:/Users/Nathan/AppData/Local/Temp/claude/A--code-spmm/5aa8b876-8611-5f52-a8a1-983a72c7748d/scratchpad/final136/src" PYTHONIOENCODING=utf8 OPENRAPM_JOBS=4
PY=.venv/Scripts/python
stage() { echo "=== $(date) $1"; }
stage "shrink and swap"
cp outputs/season_ratings_product134_step3.parquet outputs/season_ratings_product136_step3.parquet
$PY -u -m openrapm shrink --base season_ratings_product136_step3 --setting product --name product136 > outputs/build_product136.log 2>&1 || { stage "shrink FAILED"; exit 1; }
$PY -u -m openrapm swap --base season_ratings_product136_step4 --setting product --name product136 >> outputs/build_product136.log 2>&1 || { stage "swap FAILED"; exit 1; }
stage "the regular seasons, then the wins"
$PY -u -m openrapm wins --name product136 --rebuild-regular --jobs 4 >> outputs/build_product136.log 2>&1 || { stage "wins FAILED"; exit 1; }
stage "the site's files into a scratch folder"
mkdir -p outputs/site_check136
$PY -u -m openrapm site --ratings outputs/season_ratings_product136.parquet --openrapm-decomposition --power --out-dir outputs/site_check136 > outputs/site_check136.log 2>&1 || { stage "site FAILED"; exit 1; }
stage "all done"
