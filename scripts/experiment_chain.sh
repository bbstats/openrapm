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
# The incumbent since 2026-10-09: experiment 46 (the Robustness pass's body-weight fix on experiment 45's LightGBM prior
# without the on-court inputs), with its own prior-shrink multipliers fitted on its own within-season folds
# (weightfix_within).  Its 62 settings ride in BASE_FLAGS, AFTER the experiment's own EXTRA flags, because 62 reads the
# first occurrence of a flag: an experiment's --features or --booster_params wins.  INC_FOLDS: the incumbent's folds.
INC_NAME=${INC_NAME:-weightfix}
INC_FOLDS=${INC_FOLDS:-weightfix_within}
INC=outputs/season_ratings_${INC_NAME}.parquet
BASE_FLAGS=${BASE_FLAGS:---booster_params=lgb1 --features=boruta_noonc}
[ -n "$NAME" ] && [ -n "$TARGET" ] || { echo "usage: scripts/experiment_chain.sh NAME TARGET [EXTRA 62 FLAGS]"; exit 1; }
# SKIP_BUILD=1: outputs/season_ratings_NAME_raw.parquet already exists -- start at the prior shrink.
# SHRINK_TAG=<tag>: the within-season folds 99's multipliers are read from (default the incumbent's, INC_FOLDS; experiment 45's were `lgb_noonc_within`, chimeraboost's `within`); the
# LightGBM rematch (2026-10-08) refits them on the candidate's own priors (97 --tag=<tag>).
# OWN_FOLDS=1 (the Robustness pass, 2026-10-08): save the prior models per chunk (--save_models=NAME_<first>), build the
# candidate's own within-season folds for 2017-2026 (97 --tag=NAME_within), fit the prior shrink on them, and keep the
# table on the INCUMBENT's multipliers beside it (season_ratings_NAME_incmult) -- both are scored.  Then the held-out
# audit (138) compares the candidate's folds with the incumbent's (and with CONTROL_TAG's, the noise control's).
# FIRST_PASS=1: build 2017-2026 only, its folds and the audit, and stop (about 18 minutes); a later full run reuses that
# chunk.  A chunk is reused only when its table and its saved models both exist.
OWN_FOLDS=${OWN_FOLDS:-0}
FIRST_PASS=${FIRST_PASS:-0}
CONTROL_TAG=${CONTROL_TAG:-}
if [ "$OWN_FOLDS" = "1" ]; then SHRINK_TAG=${SHRINK_TAG:-${NAME}_within}; else SHRINK_TAG=${SHRINK_TAG:-$INC_FOLDS}; fi
# 97 must be told the settings the models were saved at: the first --features / --booster_params 62 read
FEATS97=$(echo "$EXTRA $BASE_FLAGS" | grep -o -- '--features=[^ ]*' | head -1)
BOOSTER97=$(echo "$EXTRA $BASE_FLAGS" | grep -o -- '--booster_params=[^ ]*' | head -1)
CHUNKS="1997 2007 2017"
[ "$FIRST_PASS" = "1" ] && CHUNKS="2017"
if [ "${SKIP_BUILD:-0}" != "1" ]; then
for FIRST in $CHUNKS; do
  LAST=$((FIRST + 9))
  BOARDS=$(seq -s, $FIRST $LAST)
  SAVE=""
  [ "$OWN_FOLDS" = "1" ] && SAVE="--save_models=${NAME}_$FIRST"
  if [ "$OWN_FOLDS" = "1" ] && [ -f outputs/season_ratings_${NAME}_$FIRST.parquet ] && [ -f outputs/prior_models_${NAME}_$FIRST.pkl ]; then
    echo "=== $(date) reusing build $FIRST-$LAST (table and models exist)" >> $LOG
    continue
  fi
  echo "=== $(date) build $FIRST-$LAST ($TARGET $EXTRA)" >> $LOG
  $PY -u scripts/62_single_year_board.py --exclude_neighbours=1 --score=0 --target_def=$TARGET $EXTRA $BASE_FLAGS --boards=$BOARDS $SAVE --out=season_ratings_${NAME}_$FIRST >> $LOG 2>&1 || exit 1
done
if [ "$OWN_FOLDS" = "1" ]; then
  echo "=== $(date) the candidate's own within-season folds, 2017-2026 (tag ${NAME}_within)" >> $LOG
  $PY -u scripts/97_within_season.py --models=${NAME}_2017 --base=season_ratings_${NAME}_2017 --seasons=2017-2026 --tag=${NAME}_within $FEATS97 $BOOSTER97 >> $LOG 2>&1 || exit 1
  COMPARE="--reference=$INC_FOLDS"
  [ -n "$CONTROL_TAG" ] && COMPARE="$COMPARE --control=$CONTROL_TAG"
  echo "=== $(date) held-out audit against the incumbent's folds" >> $LOG
  $PY -u scripts/138_heldout_audit.py --random=${NAME}_within --mult_tag=${NAME}_within $COMPARE --draws=0 --threshold_from=audit_lgb_noonc_2017 --out=audit_${NAME} >> $LOG 2>&1 || exit 1
fi
if [ "$FIRST_PASS" = "1" ]; then
  echo "=== $(date) first pass done" >> $LOG
  exit 0
fi
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
if [ "$OWN_FOLDS" = "1" ]; then
  echo "=== $(date) the second arm: the incumbent's multipliers ($INC_FOLDS)" >> $LOG
  $PY -u scripts/99_prior_shrink.py --base=season_ratings_${NAME}_raw --out=season_ratings_${NAME}_incmult_shrunk_raw --rule=test --tag=$INC_FOLDS >> $LOG 2>&1 || exit 1
  $PY -u scripts/91_swap_adjust.py --base=outputs/season_ratings_${NAME}_incmult_shrunk_raw.parquet --kappas=0.5 --taus= --hold_spread=within --tag=${NAME}_incmult >> $LOG 2>&1 || exit 1
  $PY -u scripts/63_yoy.py --rankings=${NAME}_incmult=outputs/season_ratings_${NAME}_incmult.parquet,incumbent=$INC --ref=incumbent --tag=${NAME}_incmult --splits= >> $LOG 2>&1
fi
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
