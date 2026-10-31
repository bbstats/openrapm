#!/bin/bash
# Experiment 109 blended (the owner, 2026-10-31: "If it's just two different shrinkages we should probably just blend
# between them", then "straight line"): each season 1997-2026 rated from its games through 10%, 25% and 50% of its
# regular season with the prior scales a straight line from held at the first game to fitted at 75% (blend), and
# under today's rule, held in full until half the season (held), then the within-season WOWY test of the rest of each
# season, both with the whole season's step 4.  The held arms at 25% and 50% are last night's, built with today's
# code; the one at 10% is built again.  At 50% the blend is also read against the fitted arm, the switch's other side.
# A finished stage is skipped on a rerun.
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
LOG=outputs/blend109.log
export OPENRAPM_JOBS=4
HOLD="--hold-prior-scale season_ratings_parts107_step3"
build() {   # build TAG CUT [EXTRA]
  if [ -f outputs/within/$1/summary.parquet ]; then return 0; fi
  echo "=== $(date) folds at $2 (tag $1) $3" >> $LOG
  $PY -u -m openrapm folds --models parts107 --base season_ratings_parts107_step3 --seasons 1997-2026 \
      --tag $1 --cut $2 --jobs 4 $HOLD $3 >> $LOG 2>&1
}
build parts107_cut10_held31 0.10 || exit 1
build parts107_cut10_blend31 0.10 "--blend-through 0.75" || exit 1
build parts107_cut25_blend31 0.25 "--blend-through 0.75" || exit 1
build parts107_cut50_blend31 0.50 "--blend-through 0.75" || exit 1
wowy() {    # wowy TAG SYSTEMS
  echo "=== $(date) WOWY $1" >> $LOG
  $PY -u scripts/checks/151_wowy.py --systems=$2 --ref=today --tag=$1 \
      --step4-from=today:parts107,blend:parts107 >> $LOG 2>&1
}
wowy blend109_cut10 today=parts107_cut10_held31,blend=parts107_cut10_blend31 || exit 1
wowy blend109_cut25 today=parts107_cut25_held30,blend=parts107_cut25_blend31 || exit 1
wowy blend109_cut50 today=parts107_cut50_held30,blend=parts107_cut50_blend31 || exit 1
wowy blend109_cut50_fit today=parts107_cut50_fit30,blend=parts107_cut50_blend31 || exit 1
echo "=== $(date) experiment 109 blended, done" >> $LOG
