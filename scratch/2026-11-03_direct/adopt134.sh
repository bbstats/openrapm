#!/bin/bash
# Adoption of experiment 134 (the owner, 2026-11-04: "Adopt, then show me"), the part before the owner's word to publish.
# Nothing here touches the published tables or the settings the morning job runs on: the full build goes under its own
# name, product134, every line and fold named by flag, and the site's files into a scratch folder.  On the word:
# settings.py names them, the tables are copied over the published ones, main is fast-forwarded and pushed.
# A finished stage is skipped on a rerun.
#   nohup bash scratch/2026-11-03_direct/adopt134.sh > scratch/2026-11-03_direct/adopt134.log 2>&1 &
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
NAME=rating134
INPUTS=context_types_team_rates_centred
OUT=product134
export OPENRAPM_JOBS=4
stage() { echo "=== $(date) $1"; }

if [ ! -f outputs/within/${NAME}_k10/summary.parquet ]; then
  stage "the tenth folds ${NAME}_k10"
  $PY -u -m openrapm folds --models $NAME --base season_ratings_${NAME}_step3 --seasons 1997-2026 --tag ${NAME}_k10 \
      --folds-per-repeat 10 --jobs 4 > outputs/${NAME}_k10.log 2>&1 || { stage "the tenth folds FAILED"; exit 1; }
fi
if [ ! -f outputs/season_ratings_${OUT}.parquet ]; then
  stage "the full build of 1997-2027 at the product setting, under the name $OUT"
  $PY -u -m openrapm build --name $OUT --second-prior $NAME --second-prior-inputs $INPUTS \
      --folds ${NAME}_early,${NAME}_within --extrapolate-folds ${NAME}_k10 > outputs/build_${OUT}.log 2>&1 \
      || { stage "the full build FAILED"; exit 1; }
fi
if [ ! -f outputs/site_check134/data/ratings.json ]; then
  stage "the site's files into a scratch folder"
  mkdir -p outputs/site_check134
  $PY -u -m openrapm site --ratings outputs/season_ratings_${OUT}.parquet --openrapm-decomposition --power \
      --out-dir outputs/site_check134 > outputs/site_check134.log 2>&1 || { stage "the site's files FAILED"; exit 1; }
fi
stage "all done"
