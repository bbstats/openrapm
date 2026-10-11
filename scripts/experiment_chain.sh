#!/bin/bash
# One experiment's whole ratings chain, as HANDOFF.md "The test: year-over-year" describes it: steps 1-3 in one run
# (`openrapm rate --setting test`), step 4 (`openrapm shrink --setting test`), step 5 (`openrapm swap --setting test`),
# then the readings 63, 64, 66 (2026 top 20), 90, 88, 70 + 73 (on the
# incumbent's trade-loss target, and on the candidate's own as a sensitivity).  Everything not named is the incumbent's.
#   bash scripts/experiment_chain.sh NAME DEFENSIVE_TARGET ["EXTRA RATE FLAGS"]   (INC_NAME, BASE_FLAGS, SHRINK_TAG, SKIP_BUILD: see below)
#   bash scripts/experiment_chain.sh noise_control x3def_w0.25 "--fold-assignment 1"
# Log: outputs/NAME_chain.log.  One build at a time.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
NAME=$1
TARGET=$2
EXTRA=${3:-}          # extra flags for `openrapm rate`, e.g. --fold-assignment 1 or --features <set>
LOG=outputs/${NAME}_chain.log
# The incumbent since 2026-10-09: experiment 46 (the Robustness pass's body-weight fix on experiment 45's LightGBM prior
# without the on-court inputs), with its own prior-shrink multipliers fitted on its own within-season folds
# (weightfix_within).  Its rate settings ride in BASE_FLAGS, BEFORE the experiment's own EXTRA flags, because the
# command line keeps the LAST occurrence of a flag: an experiment's --features or --booster-params wins.
# INC_FOLDS: the incumbent's folds.
INC_NAME=${INC_NAME:-weightfix}
INC_FOLDS=${INC_FOLDS:-weightfix_within}
INC=outputs/season_ratings_${INC_NAME}.parquet
BASE_FLAGS=${BASE_FLAGS:---booster-params lgb1 --features boruta_noonc}
[ -n "$NAME" ] && [ -n "$TARGET" ] || { echo "usage: scripts/experiment_chain.sh NAME TARGET [EXTRA RATE FLAGS]"; exit 1; }
# SKIP_BUILD=1: outputs/season_ratings_NAME_step3.parquet already exists -- start at the prior shrink.
# SHRINK_TAG=<tag>: the within-season folds step 4's multipliers are read from (default the incumbent's, INC_FOLDS; experiment
# 45's were `lgb_noonc_within`, chimeraboost's `within`); the LightGBM rematch (2026-10-08) refits them on the
# candidate's own priors (`openrapm folds --tag=<tag>`).
# OWN_FOLDS=1 (the Robustness pass, 2026-10-08): save the candidate's boosters (--save-models NAME, every rated season),
# build its own within-season folds for 2017-2026 (`openrapm folds --tag NAME_within`), fit the prior shrink on them,
# and keep the table on the INCUMBENT's multipliers beside it (season_ratings_NAME_incmult) -- both are scored.  Then the
# held-out audit (138) compares the candidate's folds with the incumbent's (and with CONTROL_TAG's, the noise control's).
# FIRST_PASS=1: rate 2017-2026 only, its folds and the audit, and stop (about 18 minutes).  A later full run rates all
# thirty seasons in one run and keeps the boosters the first pass saved (outputs/prior_models_NAME.pkl).
OWN_FOLDS=${OWN_FOLDS:-0}
FIRST_PASS=${FIRST_PASS:-0}
CONTROL_TAG=${CONTROL_TAG:-}
if [ "$OWN_FOLDS" = "1" ]; then SHRINK_TAG=${SHRINK_TAG:-${NAME}_within}; else SHRINK_TAG=${SHRINK_TAG:-$INC_FOLDS}; fi
SEASONS=1997-2026
[ "$FIRST_PASS" = "1" ] && SEASONS=2017-2026
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  SAVE=""
  [ "$OWN_FOLDS" = "1" ] && [ ! -f outputs/prior_models_${NAME}.pkl ] && SAVE="--save-models ${NAME}"
  echo "=== $(date) rate $SEASONS ($TARGET $EXTRA)" >> $LOG
  $PY -u -m openrapm rate --setting test --defensive-target $TARGET $BASE_FLAGS $EXTRA --seasons $SEASONS --name $NAME $SAVE >> $LOG 2>&1 || exit 1
fi
if [ "$OWN_FOLDS" = "1" ]; then
  echo "=== $(date) the candidate's own within-season folds, 2017-2026 (tag ${NAME}_within)" >> $LOG
  $PY -u -m openrapm folds --models ${NAME} --base season_ratings_${NAME}_step3 --seasons 2017-2026 --tag ${NAME}_within >> $LOG 2>&1 || exit 1
  COMPARE="--reference=$INC_FOLDS"
  [ -n "$CONTROL_TAG" ] && COMPARE="$COMPARE --control=$CONTROL_TAG"
  echo "=== $(date) held-out audit against the incumbent's folds" >> $LOG
  $PY -u scripts/138_heldout_audit.py --random=${NAME}_within --mult_tag=${NAME}_within $COMPARE --draws=0 --threshold_from=audit_lgb_noonc_2017 --out=audit_${NAME} >> $LOG 2>&1 || exit 1
fi
if [ "$FIRST_PASS" = "1" ]; then
  echo "=== $(date) first pass done" >> $LOG
  exit 0
fi
echo "=== $(date) prior shrink (multipliers from outputs/within/$SHRINK_TAG)" >> $LOG
$PY -u -m openrapm shrink --base season_ratings_${NAME}_step3 --setting test --folds $SHRINK_TAG --name ${NAME} >> $LOG 2>&1 || exit 1
echo "=== $(date) swap adjustment" >> $LOG
$PY -u -m openrapm swap --base season_ratings_${NAME}_step4 --setting test --name ${NAME} --types >> $LOG 2>&1 || exit 1
echo "=== $(date) year-over-year" >> $LOG
$PY -u scripts/63_yoy.py --rankings=${NAME}=outputs/season_ratings_${NAME}.parquet,incumbent=$INC --ref=incumbent --tag=${NAME} --splits= >> $LOG 2>&1
if [ "$OWN_FOLDS" = "1" ]; then
  echo "=== $(date) the second arm: the incumbent's multipliers ($INC_FOLDS)" >> $LOG
  $PY -u -m openrapm shrink --base season_ratings_${NAME}_step3 --setting test --folds $INC_FOLDS --name ${NAME}_incmult >> $LOG 2>&1 || exit 1
  $PY -u -m openrapm swap --base season_ratings_${NAME}_incmult_step4 --setting test --name ${NAME}_incmult --types >> $LOG 2>&1 || exit 1
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
