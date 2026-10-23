#!/bin/bash
# One experiment's whole ratings chain, as notes/HANDOFF.md "The test" describes it: steps 1-3 in one run
# (`openrapm rate --setting test`), the candidate's own within-season folds for all thirty seasons, step 4 (`openrapm
# shrink --setting test`), step 5 (`openrapm swap --setting test`), then the three readings the owner judges on
# (2026-10-15: "too many tests ... within-season is truly the best metric, particularly WOWY"): 151 (the
# with-or-without-you test on the held-back games, the decider, with the noise control beside it), 63 (year over year,
# a guard) and 66 (the 2026 top 20).  FULL_READINGS=1 adds the older readings (138, the second arm, 64, 90, 88, 70 + 73).
# Everything not named is the incumbent's.
#   bash scripts/experiment_chain.sh NAME DEFENSIVE_TARGET ["EXTRA RATE FLAGS"]   (INC_NAME, BASE_FLAGS, SHRINK_TAG, SKIP_BUILD: see below)
#   bash scripts/experiment_chain.sh noise_control points_with_expected_opponent_threes_late_as_scored "--fold-assignment 1"
# Log: outputs/NAME_chain.log.  One build at a time.
# A reading that fails is logged and the chain goes on to the next (each reading stands alone); the last line of the
# log then names every reading that failed, and the chain exits with status 1.
cd "$(dirname "$0")/.."
# the environment's python: .venv/Scripts/python on Windows, .venv/bin/python on macOS and Linux; PY=... overrides
if [ -z "$PY" ]; then PY=.venv/bin/python; [ -x .venv/Scripts/python ] && PY=.venv/Scripts/python; fi
# rate and folds split their seasons over up to 4 processes, as many as free memory holds (commands/parallel.py;
# the owner, 2026-10-21: "You can do whatever you want as long as it's faster"); OPENRAPM_JOBS=1 runs one
export OPENRAPM_JOBS=${OPENRAPM_JOBS:-4}
NAME=$1
TARGET=$2
EXTRA=${3:-}          # extra flags for `openrapm rate`, e.g. --fold-assignment 1 or --features <set>
LOG=outputs/${NAME}_chain.log
# The incumbent since 2026-10-09: experiment 46 (the Robustness pass's body-weight fix on experiment 45's LightGBM prior
# without the on-court inputs), with its own prior-shrink multipliers fitted on its own within-season folds
# (weightfix_within).  Its rate settings ride in BASE_FLAGS, BEFORE the experiment's own EXTRA flags, because the
# command line keeps the LAST occurrence of a flag: an experiment's --features or --booster-params wins.
# INC_FOLDS: the incumbent's folds.  From 2026-10-12 the incumbent was experiment 55 (nearshrink): experiment 46's
# ratings with step 4's multipliers from the 8 fold seasons nearest each season, over its folds for all thirty
# seasons (the shipped default of `openrapm shrink`).  Since 2026-10-13 it is experiment 57 (fromdebut): the career
# inputs counted from each player's debut (--features career_from_debut), its own folds for all thirty seasons and
# step 4 by the same rule on them.  Since 2026-10-14 it is experiment 59 (nocareer): those career counts off the prior
# (--features no_career_counts), its own folds for all thirty seasons, step 4 by the same rule.  Since 2026-10-15
# it is experiment 63 (cols06): the same with each tree of the prior shown 60% of its inputs (--booster-params
# lgb1_cols06), its own folds for all thirty seasons; its noise control is noise_cols06.  Since 2026-10-15, later the
# same day, it is experiment 65 (gamesfree): cols06's steps 1-3 and folds, with step 4 multiplying the games' part too
# (settings.ShrinkSettings.free_games_part, the default of `openrapm shrink`).  Its folds are cols06's, so the decider
# reads them under INC_SYSTEM; the decider's step 4 follows the shipped setting for every system.  The held-out audit pairs a candidate with the incumbent's 2017-2026 folds
# (AUDIT_REF).  A candidate with OWN_FOLDS=1 has folds for 2017-2026 only; its step 4 then reads the 8 nearest of
# those.  EARLY_FOLDS=1 (with OWN_FOLDS=1; experiment 56) also builds its 1997-2016 folds (NAME_early), so its step 4
# is the incumbent's rule on its own folds for all thirty seasons.
# Since 2026-10-16 it is experiment 76 (tune2): the booster settings and step 4's defence factor from check 152's
# guarded search, with the hustle inputs; its own folds, its noise control noise_tune2.
# Since 2026-10-17 it is experiment 77 (nothreepct): tune2 with three-point % off the offensive booster, its own folds;
# no noise control on it yet, so noise_tune2 stands in.
# Since 2026-10-17, later the same day, it is experiment 78 (bags10): nothreepct with the prior averaged over 10
# bootstraps of the players (--bags 10, the shipped default of `openrapm rate`), its own folds; its noise control
# noise_bags10 (--fold-assignment 1).
# Since 2026-10-19 it is experiment 80 (crossfit80): bags10's steps 1-4 and folds, with step 5's type model trained on
# the folds' held-back games and seasons played before this one out of its inputs; its tables crossfit80, its folds
# still bags10's.  Every candidate's step 5 now reads its own folds (`openrapm swap --folds`).
# Since 2026-10-20 it is experiment 82 (ctx82): step 3 reads the second prior ctx82, step 5 off, step 4 on its own
# folds ctx82_early and ctx82_within, all three the shipped defaults; no noise control on it yet, so noise_bags10
# stands in.
# Since 2026-10-21 it is experiment 93 (ctx82c): experiment 82 with the line's inputs centred on the season
# (--second-prior-inputs context_and_types_centred, the shipped default), its own folds ctx82c_early and ctx82c_within.
# Since 2026-10-22 it is experiment 95 (heave95): experiment 93 on the stints rebuilt with the parser's fixes
# (experiment 94, parser94, published), with the threes of a period's last two seconds counted as they happened in the
# defensive target (points_with_expected_opponent_threes_late_as_scored, the shipped default), its own folds
# heave95_early and heave95_within; not published yet.
# Since later the same day it is experiment 98 (eras98): heave95's tables with the defense factor by era, 0.89
# before 2017 and 0.84 from 2017 (settings.ShrinkSettings.defence_factors, the shipped default), its folds still
# heave95's; 151 applies the same factors to every system it scores.
# Since later the same day it is experiment 99 (inputs99): eras98 with each season's own body listing and the pad k
# around the bins (config body_inputs season_listing, pad_k_around target, the shipped config), its own folds
# inputs99_early and inputs99_within; not published yet.
# Since 2026-10-23 it is experiment 106 (layers106): inputs99's steps 1-3 and folds, with step 4 in two layers (each
# player's deviation from his team's mean on held-back stints within games, settings.ShrinkSettings.within_team_layer,
# the shipped default of `openrapm shrink`); its tables season_ratings_layers106*, its folds still inputs99's, and 151
# scores every system's step 4 with the two layers.
INC_NAME=${INC_NAME:-layers106}
INC_SYSTEM=${INC_SYSTEM:-inputs99}              # the incumbent's folds (<name>_early, <name>_within) for 151
CONTROL_NAME=${CONTROL_NAME:-noise_bags10}       # the noise control's folds (<name>_early, <name>_within) for 151
INC_FOLDS=${INC_FOLDS:-inputs99_early,inputs99_within}
AUDIT_REF=${AUDIT_REF:-inputs99_within}
INC=outputs/season_ratings_${INC_NAME}.parquet
BASE_FLAGS=${BASE_FLAGS:---booster-params lgb1_tune2 --features hustle_no_three_pct}
[ -n "$NAME" ] && [ -n "$TARGET" ] || { echo "usage: scripts/experiment_chain.sh NAME TARGET [EXTRA RATE FLAGS]"; exit 1; }
# Every check the chain calls, looked for BEFORE the build: a missing one would otherwise surface only hours later.
READINGS="151_wowy 63_yoy 66_compare 64_consensus_report 70_tradeset 73_tradeloss 88_yoy_by_player 90_swap_test 138_heldout_audit"
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
# SWAP_FLAGS: extra flags for `openrapm swap`, e.g. "--applied-share 0" to leave step 4's ratings as they are
# (experiment 82, where the swaps' lesson moved into the second prior).
# DECIDER_FLAGS: extra flags for the decider (151), e.g. "--defence-factors=NAME:0.9" beside SHRINK_FLAGS'
# "--defence-factor 0.9", so the decider scores the step 4 that ships.
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
# since 2026-10-15 every experiment builds its own folds for all thirty seasons: the decider (151) scores them
OWN_FOLDS=${OWN_FOLDS:-1}
FIRST_PASS=${FIRST_PASS:-0}
EARLY_FOLDS=${EARLY_FOLDS:-1}
FULL_READINGS=${FULL_READINGS:-0}
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
  if [ "$FULL_READINGS" = "1" ]; then
    COMPARE="--reference=$AUDIT_REF"
    [ -n "$CONTROL_TAG" ] && COMPARE="$COMPARE --control=$CONTROL_TAG"
    echo "=== $(date) held-out audit against the incumbent's folds" >> $LOG
    $PY -u scripts/checks/138_heldout_audit.py --random=${NAME}_within --mult_tag=${NAME}_within $COMPARE --draws=0 --threshold_from=audit_lgb_noonc_2017 --out=audit_${NAME} >> $LOG 2>&1 || exit 1
  fi
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
# Candidates are judged without experiment 101's reading to a full season, against the incumbent's tables built the
# same way: it needs each system's own tenth folds, built once a candidate is adopted (notes/HANDOFF.md).
$PY -u -m openrapm shrink --base season_ratings_${NAME}_step3 --setting test --folds $SHRINK_TAG --extrapolate-folds= $SHRINK_FLAGS --name ${NAME} >> $LOG 2>&1 || exit 1
echo "=== $(date) swap adjustment" >> $LOG
$PY -u -m openrapm swap --base season_ratings_${NAME}_step4 --setting test --folds $SHRINK_TAG --extrapolate-folds= $SHRINK_FLAGS $SWAP_FLAGS --name ${NAME} --types >> $LOG 2>&1 || exit 1
if [ "$EARLY_FOLDS" = "1" ]; then
  echo "=== $(date) the decider: with or without you, stint by stint, on the held-back games (151)" >> $LOG
  reading "with or without you" scripts/checks/151_wowy.py --systems=${INC_SYSTEM},${CONTROL_NAME},${NAME} --ref=${INC_SYSTEM} --tag=${NAME} $DECIDER_FLAGS
fi
echo "=== $(date) the guard: year-over-year" >> $LOG
reading "year-over-year" scripts/checks/63_yoy.py --rankings=${NAME}=outputs/season_ratings_${NAME}.parquet,incumbent=$INC --ref=incumbent --tag=${NAME} --splits=
echo "=== $(date) top 20" >> $LOG
reading "top 20" scripts/checks/66_compare.py incumbent=$INC ${NAME}=outputs/season_ratings_${NAME}.parquet --season=2026 --top=20 --yoy=outputs/yoy_${NAME}.parquet --ref=incumbent --open=0
if [ "$FULL_READINGS" != "1" ]; then
  [ -n "$FAILED" ] && { echo "=== $(date) chain done; these readings FAILED: ${FAILED#, }" >> $LOG; exit 1; }
  echo "=== $(date) chain done" >> $LOG
  exit 0
fi
if [ "$OWN_FOLDS" = "1" ]; then
  echo "=== $(date) the second arm: the incumbent's multipliers ($INC_FOLDS)" >> $LOG
  $PY -u -m openrapm shrink --base season_ratings_${NAME}_step3 --setting test --folds $INC_FOLDS --extrapolate-folds= --name ${NAME}_incmult >> $LOG 2>&1 || exit 1
  $PY -u -m openrapm swap --base season_ratings_${NAME}_incmult_step4 --setting test --folds $INC_FOLDS --extrapolate-folds= --name ${NAME}_incmult --types >> $LOG 2>&1 || exit 1
  reading "year-over-year, second arm" scripts/checks/63_yoy.py --rankings=${NAME}_incmult=outputs/season_ratings_${NAME}_incmult.parquet,incumbent=$INC --ref=incumbent --tag=${NAME}_incmult --splits=
fi
echo "=== $(date) consensus" >> $LOG
reading "consensus" scripts/checks/64_consensus_report.py outputs/season_ratings_${NAME}.parquet $INC
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
