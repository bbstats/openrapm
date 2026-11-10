#!/bin/bash
# Check 166's season-so-far ratings (notes/DECISIONS.md): each season 1997-2026 rated from its first 25%, 50%, 62.5% and
# 75% of games on the published build, the prior's weight held in full through a quarter of the season and then on the
# straight line to the season's fitted one at three quarters, as the published season in progress is rated.  The 7.15%,
# 10% and 17.5% folders of scratch/2026-11-08_power_next10/build_cuts.sh stand: there the weight is held in full.
# A finished stage is skipped on a rerun.  New folders only.
#     bash scratch/2026-11-09_improvements/build_line_cuts.sh
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
NAME=zones139
export OPENRAPM_JOBS=4
for CUT in 25 50 625 75; do
  TAG=${NAME}_cut${CUT}_line
  if [ ! -f outputs/within/$TAG/summary.parquet ]; then
    echo "=== $(date) the season so far at 0.$CUT, the prior's weight on the line from a quarter to three quarters"
    $PY -u -m openrapm folds --models $NAME --base season_ratings_${NAME}_step3 --seasons 1997-2026 --tag $TAG \
        --cut 0.$CUT --hold-prior-scale season_ratings_${NAME}_step3 --blend-through 0.75 --blend-from 0.25 --jobs 4 \
        > outputs/${TAG}.log 2>&1 || { echo "the folds at 0.$CUT FAILED"; exit 1; }
  fi
done
echo "=== $(date) all done"
