#!/bin/bash
# The halfway width settings as the owner saw them, 0.77 and 1.0 within teams and 0.87 on defense in every season, on
# experiment 134's line: the within-season WOWY test, the year-over-year test, the consensus.  Then the route the final
# build takes, shrink then swap on a finished step-3 table, checked at today's width against the full build's table.
#   bash scratch/2026-11-04_width/run_width136.sh
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf8 OPENRAPM_JOBS=2 OMP_NUM_THREADS=2
FOLDS=rating134_early,rating134_within
FLAGS="--within-factors 0.77,1 --defence-factor 0.87"
$PY -u -m openrapm shrink --base season_ratings_rating134_step3 --setting test --folds $FOLDS --extrapolate-folds= $FLAGS --name width136 > outputs/width136_chain.log 2>&1 || echo "shrink FAILED"
$PY -u -m openrapm swap --base season_ratings_width136_step4 --setting test --folds $FOLDS --extrapolate-folds= $FLAGS --name width136 >> outputs/width136_chain.log 2>&1 || echo "swap FAILED"
$PY -u scripts/checks/151_wowy.py --systems=padbins127,rating134 --ref=padbins127 --tag=width136 "--shrink-flags=rating134:$FLAGS" >> outputs/width136_chain.log 2>&1 || echo "151 FAILED"
$PY -u scripts/checks/63_yoy.py --rankings=rating134=outputs/season_ratings_width136.parquet,incumbent=outputs/season_ratings_padbins127.parquet --ref=incumbent --tag=width136 --splits= >> outputs/width136_chain.log 2>&1 || echo "yoy FAILED"
$PY -u scripts/checks/64_consensus_report.py outputs/season_ratings_width136.parquet outputs/season_ratings_padbins127.parquet >> outputs/width136_chain.log 2>&1 || echo "consensus FAILED"
$PY -u scripts/checks/63_yoy.py --rankings=rating134=outputs/season_ratings_rating134_refit156_step4.parquet,incumbent=outputs/season_ratings_padbins127.parquet --ref=incumbent --tag=refit135 --splits= > outputs/refit135_chain.log 2>&1 || echo "yoy 135 FAILED"
$PY -u -m openrapm shrink --base season_ratings_product134_step3 --setting product --folds $FOLDS --extrapolate-folds rating134_k10 --name routecheck > outputs/routecheck.log 2>&1 || echo "route shrink FAILED"
$PY -u -m openrapm swap --base season_ratings_routecheck_step4 --setting product --folds $FOLDS --extrapolate-folds rating134_k10 --name routecheck >> outputs/routecheck.log 2>&1 || echo "route swap FAILED"
date
