#!/bin/bash
# Adoption of experiment 141 (the owner, 2026-11-10: "adopt and update the site"): everything before the switch.
# Nothing here touches the published tables or the settings the morning job runs on: the full build goes under its own
# name, product141, its half-life and folds named by flag.  A finished stage is skipped on a rerun.
#   nohup bash scratch/2026-11-10_adopt141/adopt141.sh > scratch/2026-11-10_adopt141/adopt141.log 2>&1 &
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
NAME=fade141
OUT=product141
export OPENRAPM_JOBS=4
stage() { echo "=== $(date) $1"; }

if [ ! -f outputs/within/${NAME}_k10/summary.parquet ]; then
  stage "the tenth folds ${NAME}_k10, which read step 4's multipliers to a full season"
  $PY -u -m openrapm folds --models $NAME --base season_ratings_${NAME}_step3 --seasons 1997-2026 --tag ${NAME}_k10 \
      --folds-per-repeat 10 --jobs 4 > outputs/${NAME}_k10.log 2>&1 || { stage "the tenth folds FAILED"; exit 1; }
fi
# the season so far at seven points of the season, for the Power table's weights (check 166): the prior's weight held
# in full through a quarter of the season, then on the line to its fitted one at three quarters
for CUT in 0715 10 175; do
  TAG=${NAME}_cut${CUT}_held
  if [ ! -f outputs/within/$TAG/summary.parquet ]; then
    stage "the season so far at 0.$CUT, the prior's weight held"
    $PY -u -m openrapm folds --models $NAME --base season_ratings_${NAME}_step3 --seasons 1997-2026 --tag $TAG \
        --cut 0.$CUT --hold-prior-scale season_ratings_${NAME}_step3 --jobs 4 > outputs/${TAG}.log 2>&1 \
        || { stage "the folds at 0.$CUT FAILED"; exit 1; }
  fi
done
for CUT in 25 50 625 75; do
  TAG=${NAME}_cut${CUT}_line
  if [ ! -f outputs/within/$TAG/summary.parquet ]; then
    stage "the season so far at 0.$CUT, the prior's weight on the line from a quarter to three quarters"
    $PY -u -m openrapm folds --models $NAME --base season_ratings_${NAME}_step3 --seasons 1997-2026 --tag $TAG \
        --cut 0.$CUT --hold-prior-scale season_ratings_${NAME}_step3 --blend-through 0.75 --blend-from 0.25 --jobs 4 \
        > outputs/${TAG}.log 2>&1 || { stage "the folds at 0.$CUT FAILED"; exit 1; }
  fi
done
if [ ! -f outputs/season_ratings_${OUT}.parquet ]; then
  stage "the full build of 1997-2027 at the product setting, under the name $OUT"
  $PY -u -m openrapm build --name $OUT --second-prior-halflife 8 \
      --folds ${NAME}_early,${NAME}_within --extrapolate-folds ${NAME}_k10 > outputs/build_${OUT}.log 2>&1 \
      || { stage "the full build FAILED"; exit 1; }
fi
stage "all done"
