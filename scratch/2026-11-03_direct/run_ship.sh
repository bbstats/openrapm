#!/bin/bash
# Experiment 134 (notes/DECISIONS.md): the line fitted on the rating, in the shipped code, on today's boosters and folds.
# Stages, each skipped on a rerun when its output exists:
#   1. the line: `openrapm second-prior --target rating` on padbins127's boosters and folds
#   2. the whole-season chain against padbins127
#   3. each season rated from its first 7.15% (2027's 88 games), 10%, 25% and 50% of games, the prior scales held at the 8
#      nearest seasons' mean, then the within-season WOWY test of the rest of each season against padbins127's same folds
#   4. 2027 today under both, into copies of the chain tables, like for like
#   nohup bash scratch/2026-11-03_direct/run_ship.sh > scratch/2026-11-03_direct/run_ship.log 2>&1 &
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
NAME=rating134
INPUTS=context_types_team_rates_centred
TARGET=points_with_expected_opponent_threes_late_as_scored
export OPENRAPM_JOBS=4
stage() { echo "=== $(date) $1"; }

if [ ! -f outputs/second_prior/$NAME/run.json ]; then
  stage "the line"
  $PY -u -m openrapm second-prior --models padbins127 --folds padbins127_early,padbins127_within --tag $NAME \
      --inputs $INPUTS --target rating > outputs/${NAME}_second_prior.log 2>&1 || { stage "the line FAILED"; exit 1; }
fi
if [ ! -f outputs/csv/wowy_${NAME}_eras.csv ]; then
  stage "the whole-season chain"
  bash scripts/experiment_chain.sh $NAME $TARGET "--second-prior $NAME --second-prior-inputs $INPUTS" \
      > outputs/${NAME}_chain.nohup 2>&1 || stage "the chain reports a failed reading"
fi
for CUT in 0715 10 25 50; do
  TAG=${NAME}_cut${CUT}_held
  if [ ! -f outputs/within/$TAG/summary.parquet ]; then
    stage "the season so far at 0.$CUT"
    $PY -u -m openrapm folds --models $NAME --base season_ratings_${NAME}_step3 --seasons 1997-2026 --tag $TAG \
        --cut 0.$CUT --hold-prior-scale season_ratings_${NAME}_step3 --jobs 4 > outputs/${TAG}.log 2>&1 \
        || { stage "the folds FAILED"; exit 1; }
  fi
  if [ ! -f outputs/csv/wowy_${NAME}_cut${CUT}_eras.csv ]; then
    stage "the within-season WOWY test of the rest of each season at 0.$CUT"
    $PY -u scripts/checks/151_wowy.py --systems=today=padbins127_cut${CUT}_held,$NAME=$TAG --ref=today \
        --step4-from=today:padbins127,$NAME:$NAME --tag=${NAME}_cut${CUT} > outputs/${NAME}_wowy_cut${CUT}.log 2>&1 \
        || { stage "the test FAILED"; exit 1; }
  fi
done
for ARM in padbins127 $NAME; do
  OUT=${ARM}now27
  [ -f outputs/season_ratings_${OUT}.parquet ] && continue
  stage "2027 today under $ARM"
  for PART in "" _step3 _step4; do
    cp outputs/season_ratings_${ARM}${PART}.parquet outputs/season_ratings_${OUT}${PART}.parquet
  done
  if [ "$ARM" = "$NAME" ]; then
    FLAGS="--second-prior $NAME --second-prior-inputs $INPUTS --folds ${NAME}_early,${NAME}_within"
  else
    FLAGS="--folds padbins127_early,padbins127_within"
  fi
  $PY -u -m openrapm build --setting test --name $OUT --seasons 2027 --replace-seasons $FLAGS --extrapolate-folds= \
      > outputs/${OUT}.log 2>&1 || { stage "the 2027 build FAILED"; exit 1; }
done
stage "all done"
