#!/bin/bash
# One experiment's whole ratings chain, as HANDOFF.md "The test: year-over-year" describes it: 62 in three chunks of ten
# seasons (stitched), 99, 91, then the readings 63, 64, 66 (2026 top 20), 90, 88, 70 + 73 (on the incumbent's trade-loss
# target, and on the candidate's own as a sensitivity).  Everything not named is the incumbent's.
#   bash scripts/experiment_chain.sh NAME DEFENSIVE_TARGET ["EXTRA 62 FLAGS"]   (INC_NAME, BASE_FLAGS, SHRINK_TAG, SKIP_BUILD: see below)
#   bash scripts/experiment_chain.sh quality_logistic_pinned x3def_qL7_w0.25 --deal_target_def=x3def_w0.25   (42f)
#   bash scripts/experiment_chain.sh redeal_incumbent x3def_w0.25 --deal_seed=1                              (C1)
# A target experiment keeps the incumbent's training groupings with --deal_target_def=x3def_w0.25 (DECISIONS.md, 42).
# Log: outputs/NAME_chain.log.  One build at a time.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
NAME=$1
TARGET=$2
EXTRA=${3:-}          # extra flags for 62, e.g. --deal_seed=1 or --deal_target_def=x3def_w0.25
LOG=outputs/${NAME}_chain.log
# The incumbent since 2026-10-08: experiment 45, the LightGBM prior without the on-court inputs, with its own
# prior-shrink multipliers (lgb_noonc_rs).  Its 62 settings ride in BASE_FLAGS, AFTER the experiment's own EXTRA flags,
# because 62 reads the first occurrence of a flag: an experiment's --features or --booster_params wins.
INC_NAME=${INC_NAME:-lgb_noonc_rs}
INC=outputs/season_ratings_${INC_NAME}.parquet
BASE_FLAGS=${BASE_FLAGS:---booster_params=lgb1 --features=boruta_noonc}
[ -n "$NAME" ] && [ -n "$TARGET" ] || { echo "usage: scripts/experiment_chain.sh NAME TARGET [EXTRA 62 FLAGS]"; exit 1; }
# SKIP_BUILD=1: outputs/season_ratings_NAME_raw.parquet already exists -- start at the prior shrink.
# SHRINK_TAG=<tag>: the within-season folds 99's multipliers are read from (default the incumbent's, `lgb_noonc_within`; chimeraboost's were `within`); the
# LightGBM rematch (2026-10-08) refits them on the candidate's own priors (97 --tag=<tag>).
SHRINK_TAG=${SHRINK_TAG:-lgb_noonc_within}
if [ "${SKIP_BUILD:-0}" != "1" ]; then
for FIRST in 1997 2007 2017; do
  LAST=$((FIRST + 9))
  BOARDS=$(seq -s, $FIRST $LAST)
  echo "=== $(date) build $FIRST-$LAST ($TARGET $EXTRA)" >> $LOG
  $PY -u scripts/62_single_year_board.py --exclude_neighbours=1 --score=0 --target_def=$TARGET $EXTRA $BASE_FLAGS --boards=$BOARDS --out=season_ratings_${NAME}_$FIRST >> $LOG 2>&1 || exit 1
done
$PY -u -c "
import pandas as pd
parts = [pd.read_parquet(f'outputs/season_ratings_${NAME}_{f}.parquet') for f in (1997, 2007, 2017)]
out = pd.concat(parts, ignore_index=True)
assert out.season.nunique() == 30 and not out.duplicated(['player_id', 'season']).any()
out.to_parquet('outputs/season_ratings_${NAME}_raw.parquet', index=False)
print(f'stitched outputs/season_ratings_${NAME}_raw.parquet: {len(out)} rows, {out.season.nunique()} seasons')
" >> $LOG 2>&1 || exit 1
fi
echo "=== $(date) prior shrink (multipliers from outputs/within/$SHRINK_TAG)" >> $LOG
$PY -u scripts/99_prior_shrink.py --base=season_ratings_${NAME}_raw --out=season_ratings_${NAME}_shrunk_raw --rule=test --tag=$SHRINK_TAG >> $LOG 2>&1 || exit 1
echo "=== $(date) swap adjustment" >> $LOG
$PY -u scripts/91_swap_adjust.py --base=outputs/season_ratings_${NAME}_shrunk_raw.parquet --kappas=0.5 --taus= --hold_spread=within --tag=${NAME} >> $LOG 2>&1 || exit 1
echo "=== $(date) year-over-year" >> $LOG
$PY -u scripts/63_yoy.py --rankings=${NAME}=outputs/season_ratings_${NAME}.parquet,incumbent=$INC --ref=incumbent --tag=${NAME} --splits= >> $LOG 2>&1
echo "=== $(date) consensus" >> $LOG
$PY -u scripts/64_consensus_report.py outputs/season_ratings_${NAME}.parquet $INC >> $LOG 2>&1
echo "=== $(date) top 20" >> $LOG
$PY -u scripts/66_compare.py incumbent=$INC ${NAME}=outputs/season_ratings_${NAME}.parquet --season=2026 --top=20 --yoy=outputs/yoy_${NAME}.parquet --ref=incumbent --open=0 >> $LOG 2>&1
echo "=== $(date) swap test" >> $LOG
$PY -u scripts/90_swap_test.py --rankings=incumbent=$INC,${NAME}=outputs/season_ratings_${NAME}.parquet --contexts=nofatigue --checks=0 --tag=${NAME} >> $LOG 2>&1
echo "=== $(date) quality tiers, year-over-year" >> $LOG
$PY -u scripts/88_yoy_by_player.py --cands=${NAME}=outputs/season_ratings_${NAME}.parquet --ref=incumbent=$INC --tag=${NAME} >> $LOG 2>&1
echo "=== $(date) trade loss" >> $LOG
$PY -u scripts/70_tradeset.py --rankings=outputs/season_ratings_${NAME}.parquet --team_effects=team --out=tradeset_${NAME} >> $LOG 2>&1
$PY -u scripts/73_tradeloss.py --alphas=incumbent=outputs/tradeset_${INC_NAME}_alpha.parquet,${NAME}=outputs/tradeset_${NAME}_alpha.parquet --ref=incumbent >> $LOG 2>&1
$PY -u scripts/73_tradeloss.py --alphas=incumbent=outputs/tradeset_${INC_NAME}_alpha.parquet,${NAME}=outputs/tradeset_${NAME}_alpha.parquet --ref=incumbent --quality=$INC --tier=each >> $LOG 2>&1
echo "=== $(date) trade loss, the candidate's alphas on its own defensive target (sensitivity)" >> $LOG
$PY -u scripts/70_tradeset.py --rankings=outputs/season_ratings_${NAME}.parquet --team_effects=team --target_def=$TARGET --out=tradeset_${NAME}_owntarget >> $LOG 2>&1
$PY -u scripts/73_tradeloss.py --alphas=incumbent=outputs/tradeset_${INC_NAME}_alpha.parquet,${NAME}=outputs/tradeset_${NAME}_owntarget_alpha.parquet --ref=incumbent >> $LOG 2>&1
echo "=== $(date) chain done" >> $LOG
