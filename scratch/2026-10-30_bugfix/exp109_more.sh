#!/bin/bash
# Experiment 109 continued (the bug-fix pass of 2026-10-30; the owner: "test on, then adopt"): each season 1997-2026
# rated from its games through the day a quarter, a half and three quarters of its regular season had been played,
# the prior scales fitted on those games (fit) or held at the mean of the 8 nearest seasons' whole-season scales
# (held), then the within-season WOWY test of the rest of each season, both arms with the whole season's step 4, as
# experiment 109 did at 2%, 5% and 10% (scratch/2026-10-24_daily/exp109.sh).  Both arms are built fresh, with today's
# code: the folds at a quarter, a half and three quarters on disk are experiment 108's, from 2026-10-23.
# A finished stage is skipped on a rerun.
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
LOG=outputs/exp109_more.log
export OPENRAPM_JOBS=4
for CUT in 25 50 75; do
  for ARM in fit held; do
    TAG=parts107_cut${CUT}_${ARM}30; EXTRA=""
    if [ "$ARM" = "held" ]; then EXTRA="--hold-prior-scale season_ratings_parts107_step3"; fi
    if [ -f outputs/within/$TAG/summary.parquet ]; then continue; fi
    echo "=== $(date) folds at 0.$CUT, $ARM (tag $TAG)" >> $LOG
    $PY -u -m openrapm folds --models parts107 --base season_ratings_parts107_step3 --seasons 1997-2026 \
        --tag $TAG --cut 0.$CUT --jobs 4 $EXTRA >> $LOG 2>&1 || exit 1
  done
done
for CUT in 25 50 75; do
  echo "=== $(date) WOWY at 0.$CUT: held against fitted, the whole season's step 4" >> $LOG
  $PY -u scripts/checks/151_wowy.py --systems=fit=parts107_cut${CUT}_fit30,held=parts107_cut${CUT}_held30 \
      --ref=fit --step4-from=fit:parts107,held:parts107 --tag=exp109_cut${CUT} >> $LOG 2>&1 || exit 1
done
echo "=== $(date) experiment 109 continued, done" >> $LOG
