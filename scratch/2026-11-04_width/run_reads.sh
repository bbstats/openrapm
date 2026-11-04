#!/bin/bash
# The within-season WOWY test, in the chain's form, of the two width settings read on experiment 134's line.
#   bash scratch/2026-11-04_width/run_reads.sh
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf8 OPENRAPM_JOBS=2 OMP_NUM_THREADS=2
FLAGS135="--within-factors 0.7826,0.9935 --defence-factor 0.8464,2017:0.9007"
FLAGS136="--within-factors 0.7663,0.9968 --defence-factor 0.8682,2017:0.8704"
$PY -u scripts/checks/151_wowy.py --systems=padbins127,rating134 --ref=padbins127 --tag=refit135 "--shrink-flags=rating134:$FLAGS135" > outputs/refit135_wowy.log 2>&1 || echo "135 FAILED"
$PY -u scripts/checks/151_wowy.py --systems=padbins127,rating134 --ref=padbins127 --tag=mid136 "--shrink-flags=rating134:$FLAGS136" > outputs/mid136_wowy.log 2>&1 || echo "136 FAILED"
date
tail -n 12 outputs/refit135_wowy.log | cut -c1-130
tail -n 12 outputs/mid136_wowy.log | cut -c1-130
