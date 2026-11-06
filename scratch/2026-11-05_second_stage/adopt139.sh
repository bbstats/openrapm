#!/bin/bash
# Adoption of experiment 139 (the owner, 2026-11-06: "Adopt but don't publish until tomorrows run"): everything before the
# switch.  Nothing here touches the published tables or the settings the morning job runs on: the stints are the zones
# folder through its config, the full build goes under its own name, product139, every line and fold named by flag, and
# the site's files into a scratch folder.  A finished stage is skipped on a rerun.
#   nohup bash scratch/2026-11-05_second_stage/adopt139.sh > scratch/2026-11-05_second_stage/adopt139.log 2>&1 &
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
NAME=zones139
LINE=zones139_line
INPUTS=context_types_team_zones_centred
OUT=product139
export OPENRAPM_JOBS=4
export OPENRAPM_CONFIG=scratch/2026-11-05_second_stage/config_zones139.yaml
stage() { echo "=== $(date) $1"; }

stage "2027's stints with the zones, through yesterday's games"
$PY -u -m openrapm data stints 2027 2027 RS,PO --force > outputs/stints2027_${NAME}.log 2>&1 || { stage "2027's stints FAILED"; exit 1; }
if [ ! -f outputs/within/${NAME}_k10/summary.parquet ]; then
  stage "the tenth folds ${NAME}_k10"
  $PY -u -m openrapm folds --models $NAME --base season_ratings_${NAME}_step3 --seasons 1997-2026 --tag ${NAME}_k10 \
      --folds-per-repeat 10 --jobs 4 > outputs/${NAME}_k10.log 2>&1 || { stage "the tenth folds FAILED"; exit 1; }
fi
if [ ! -f outputs/season_ratings_${OUT}.parquet ]; then
  stage "the full build of 1997-2027 at the product setting, under the name $OUT"
  $PY -u -m openrapm build --name $OUT --second-prior $LINE --second-prior-inputs $INPUTS \
      --folds ${NAME}_early,${NAME}_within --extrapolate-folds ${NAME}_k10 > outputs/build_${OUT}.log 2>&1 \
      || { stage "the full build FAILED"; exit 1; }
fi
if [ ! -f outputs/site_check139/data/ratings.json ]; then
  stage "the site's files into a scratch folder"
  mkdir -p outputs/site_check139
  $PY -u -m openrapm site --ratings outputs/season_ratings_${OUT}.parquet --openrapm-decomposition --power \
      --out-dir outputs/site_check139 > outputs/site_check139.log 2>&1 || { stage "the site's files FAILED"; exit 1; }
fi
stage "all done"
