#!/bin/bash
# One experiment's whole ratings chain, as notes/HANDOFF.md "The test: year-over-year" describes it: steps 1-3 in one run
# (`openrapm rate --setting test`), step 4 (`openrapm shrink --setting test`), step 5 (`openrapm swap --setting test`),
# then the readings, each a check in scripts/checks/: 63, 64, 66 (2026 top 20), 90, 88, 70 + 73 (on the
# incumbent's trade-loss target, and on the candidate's own as a sensitivity).  Everything not named is the incumbent's.
#   bash scripts/experiment_chain.sh NAME DEFENSIVE_TARGET ["EXTRA RATE FLAGS"]   (INC_NAME, BASE_FLAGS, SHRINK_TAG, SKIP_BUILD: see below)
#   bash scripts/experiment_chain.sh noise_control points_with_expected_opponent_threes "--fold-assignment 1"
# Log: outputs/NAME_chain.log.  One build at a time.
# A reading that fails is logged and the chain goes on to the next (each reading stands alone); the last line of the
# log then names every reading that failed, and the chain exits with status 1.
cd "$(dirname "$0")/.."
# the environment's python: .venv/Scripts/python on Windows, .venv/bin/python on macOS and Linux; PY=... overrides
if [ -z "$PY" ]; then PY=.venv/bin/python; [ -x .venv/Scripts/python ] && PY=.venv/Scripts/python; fi
NAME=$1
TARGET=$2
EXTRA=${3:-}          # extra flags for `openrapm rate`, e.g. --fold-assignment 1 or --features <set>
LOG=outputs/${NAME}_chain.log
# The incumbent since 2026-10-09: experiment 46 (the Robustness pass's body-weight fix on experiment 45's LightGBM prior
# without the on-court inputs), with its own prior-shrink multipliers fitted on its own within-season folds
# (weightfix_within).  Its rate settings ride in BASE_FLAGS, BEFORE the experiment's own EXTRA flags, because the
# command line keeps the LAST occurrence of a flag: an experiment's --features or --booster-params wins.
# INC_FOLDS: the incumbent's folds.  Since 2026-10-12 the incumbent is experiment 55 (nearshrink): experiment 46's
# ratings with step 4's multipliers from the 8 fold seasons nearest each season, over its folds for all thirty
# seasons (the shipped default of `openrapm shrink`).  The held-out audit pairs a candidate with experiment 46's
# 2017-2026 folds (AUDIT_REF): the fold ratings come before step 4, so they are the same.  A candidate with
# OWN_FOLDS=1 has folds for 2017-2026 only; its step 4 then reads the 8 nearest of those.  EARLY_FOLDS=1 (with
# OWN_FOLDS=1; experiment 56) also builds its 1997-2016 folds (NAME_early), so its step 4 is the incumbent's rule on
# its own folds for all thirty seasons.
INC_NAME=${INC_NAME:-nearshrink}
INC_FOLDS=${INC_FOLDS:-weightfix_early,weightfix_within}
AUDIT_REF=${AUDIT_REF:-weightfix_within}
INC=outputs/season_ratings_${INC_NAME}.parquet
BASE_FLAGS=${BASE_FLAGS:---booster-params lgb1 --features box_score_and_bio}
[ -n "$NAME" ] && [ -n "$TARGET" ] || { echo "usage: scripts/experiment_chain.sh NAME TARGET [EXTRA RATE FLAGS]"; exit 1; }
# Every check the chain calls, looked for BEFORE the build: a missing one would otherwise surface only hours later.
READINGS="63_yoy 64_consensus_report 66_compare 70_tradeset 73_tradeloss 88_yoy_by_player 90_swap_test 138_heldout_audit"
for check in $READINGS; do
  [ -f scripts/checks/$check.py ] || { echo "scripts/checks/$check.py is missing; nothing was run"; exit 1; }
done
FAILED=""
reading() {            # reading LABEL SCRIPT [FLAGS...]: run one check into the log; a failure is noted, not fatal
  local label=$1; shift
  $PY -u "$@" >> $LOG 2>&1 || { echo "=== $(date) $label FAILED" >> $LOG; FAILED="$FAILED, $label"; }
}
# SKIP_BUILD=1: outputs/season_ratings_NAME_step3.parquet already exists -- start at the prior shrink.
# SHRINK_FLAGS: extra flags for `openrapm shrink`, e.g. "--fold-seasons 1997-2026 --nearest 8" (experiment 55).
# SHRINK_TAG=<tag>: the within-season folds step 4's multipliers are read from, outputs/within/<tag>/ (default the
# incumbent's, INC_FOLDS; with OWN_FOLDS=1 the candidate's own, NAME_within).  `openrapm folds --tag <tag>` makes them.
# OWN_FOLDS=1 (the Robustness pass, 2026-10-08): save the candidate's boosters (--save-models NAME, every rated season),
# build its own within-season folds for 2017-2026 (`openrapm folds --tag NAME_within`), fit the prior shrink on them,
# and keep the table on the INCUMBENT's multipliers beside it (season_ratings_NAME_incmult) -- both are scored.  Then the
# held-out audit (138) compares the candidate's folds with the incumbent's (and with CONTROL_TAG's, the noise control's).
# The audit draws no null of its own: --threshold_from=audit_lgb_noonc_2017 reads the threshold (3.64) and each test's
# null spread from outputs/audit_lgb_noonc_2017_thresholds.json and outputs/csv/audit_lgb_noonc_2017_null.csv, the
# 400-draw career null of the 2017-2026 audit of experiment 45 (2026-10-08), so every candidate is judged on the same
# frozen yardstick.  Both files must exist in outputs/.
# FIRST_PASS=1: rate 2017-2026 only, its folds and the audit, and stop.  A later full run rates all
# thirty seasons in one run and keeps the boosters the first pass saved (outputs/prior_models_NAME.pkl).
# EARLY_FOLDS=1 (needs OWN_FOLDS=1, refuses FIRST_PASS=1): after the 2017-2026 folds and the audit, also the 1997-2016
# folds (`openrapm folds --seasons 1997-2016 --tag NAME_early`, about 22 minutes), and step 4 reads both folders
# (SHRINK_TAG NAME_early,NAME_within) by the shipped rule: each season's multipliers from its 8 nearest fold seasons.
OWN_FOLDS=${OWN_FOLDS:-0}
FIRST_PASS=${FIRST_PASS:-0}
EARLY_FOLDS=${EARLY_FOLDS:-0}
CONTROL_TAG=${CONTROL_TAG:-}
if [ "$EARLY_FOLDS" = "1" ] && [ "$OWN_FOLDS" != "1" ]; then echo "EARLY_FOLDS=1 needs OWN_FOLDS=1"; exit 1; fi
if [ "$EARLY_FOLDS" = "1" ] && [ "$FIRST_PASS" = "1" ]; then echo "EARLY_FOLDS=1 rates all thirty seasons; FIRST_PASS=1 rates ten"; exit 1; fi
if [ "$EARLY_FOLDS" = "1" ]; then SHRINK_TAG=${SHRINK_TAG:-${NAME}_early,${NAME}_within}
elif [ "$OWN_FOLDS" = "1" ]; then SHRINK_TAG=${SHRINK_TAG:-${NAME}_within}; else SHRINK_TAG=${SHRINK_TAG:-$INC_FOLDS}; fi
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
  COMPARE="--reference=$AUDIT_REF"
  [ -n "$CONTROL_TAG" ] && COMPARE="$COMPARE --control=$CONTROL_TAG"
  echo "=== $(date) held-out audit against the incumbent's folds" >> $LOG
  $PY -u scripts/checks/138_heldout_audit.py --random=${NAME}_within --mult_tag=${NAME}_within $COMPARE --draws=0 --threshold_from=audit_lgb_noonc_2017 --out=audit_${NAME} >> $LOG 2>&1 || exit 1
fi
if [ "$EARLY_FOLDS" = "1" ]; then
  echo "=== $(date) the candidate's own within-season folds, 1997-2016 (tag ${NAME}_early)" >> $LOG
  $PY -u -m openrapm folds --models ${NAME} --base season_ratings_${NAME}_step3 --seasons 1997-2016 --tag ${NAME}_early >> $LOG 2>&1 || exit 1
fi
if [ "$FIRST_PASS" = "1" ]; then
  echo "=== $(date) first pass done" >> $LOG
  exit 0
fi
echo "=== $(date) prior shrink (multipliers from outputs/within/$SHRINK_TAG)" >> $LOG
$PY -u -m openrapm shrink --base season_ratings_${NAME}_step3 --setting test --folds $SHRINK_TAG $SHRINK_FLAGS --name ${NAME} >> $LOG 2>&1 || exit 1
echo "=== $(date) swap adjustment" >> $LOG
$PY -u -m openrapm swap --base season_ratings_${NAME}_step4 --setting test --name ${NAME} --types >> $LOG 2>&1 || exit 1
echo "=== $(date) year-over-year" >> $LOG
reading "year-over-year" scripts/checks/63_yoy.py --rankings=${NAME}=outputs/season_ratings_${NAME}.parquet,incumbent=$INC --ref=incumbent --tag=${NAME} --splits=
if [ "$OWN_FOLDS" = "1" ]; then
  echo "=== $(date) the second arm: the incumbent's multipliers ($INC_FOLDS)" >> $LOG
  $PY -u -m openrapm shrink --base season_ratings_${NAME}_step3 --setting test --folds $INC_FOLDS --name ${NAME}_incmult >> $LOG 2>&1 || exit 1
  $PY -u -m openrapm swap --base season_ratings_${NAME}_incmult_step4 --setting test --name ${NAME}_incmult --types >> $LOG 2>&1 || exit 1
  reading "year-over-year, second arm" scripts/checks/63_yoy.py --rankings=${NAME}_incmult=outputs/season_ratings_${NAME}_incmult.parquet,incumbent=$INC --ref=incumbent --tag=${NAME}_incmult --splits=
fi
echo "=== $(date) consensus" >> $LOG
reading "consensus" scripts/checks/64_consensus_report.py outputs/season_ratings_${NAME}.parquet $INC
echo "=== $(date) top 20" >> $LOG
reading "top 20" scripts/checks/66_compare.py incumbent=$INC ${NAME}=outputs/season_ratings_${NAME}.parquet --season=2026 --top=20 --yoy=outputs/yoy_${NAME}.parquet --ref=incumbent --open=0
echo "=== $(date) swap test" >> $LOG
reading "swap test" scripts/checks/90_swap_test.py --rankings=incumbent=$INC,${NAME}=outputs/season_ratings_${NAME}.parquet --contexts=nofatigue --checks=0 --tag=${NAME}
echo "=== $(date) quality tiers, year-over-year" >> $LOG
reading "quality tiers" scripts/checks/88_yoy_by_player.py --cands=${NAME}=outputs/season_ratings_${NAME}.parquet --ref=incumbent=$INC --tag=${NAME}
echo "=== $(date) trade loss" >> $LOG
reading "trade set" scripts/checks/70_tradeset.py --rankings=outputs/season_ratings_${NAME}.parquet --team_effects=team --out=tradeset_${NAME}
reading "trade loss" scripts/checks/73_tradeloss.py --alphas=incumbent=outputs/tradeset_${INC_NAME}_alpha.parquet,${NAME}=outputs/tradeset_${NAME}_alpha.parquet --ref=incumbent
reading "trade loss by quality tier" scripts/checks/73_tradeloss.py --alphas=incumbent=outputs/tradeset_${INC_NAME}_alpha.parquet,${NAME}=outputs/tradeset_${NAME}_alpha.parquet --ref=incumbent --quality=$INC --tier=each
echo "=== $(date) trade loss, the candidate's alphas on its own defensive target (sensitivity)" >> $LOG
reading "trade set, own target" scripts/checks/70_tradeset.py --rankings=outputs/season_ratings_${NAME}.parquet --team_effects=team --target_def=$TARGET --out=tradeset_${NAME}_owntarget
reading "trade loss, own target" scripts/checks/73_tradeloss.py --alphas=incumbent=outputs/tradeset_${INC_NAME}_alpha.parquet,${NAME}=outputs/tradeset_${NAME}_owntarget_alpha.parquet --ref=incumbent
if [ -n "$FAILED" ]; then
  echo "=== $(date) chain done; these readings FAILED: ${FAILED#, }" >> $LOG
  exit 1
fi
echo "=== $(date) chain done" >> $LOG
