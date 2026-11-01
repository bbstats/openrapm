#!/bin/bash
# Experiment 127 (notes/DECISIONS.md, "Early-season padding"): each possession bin's target padded toward the league
# rate by the bin's own evidence (config pad_bin_target league, scratch/2026-11-01_padding/config_padbins.yaml).
# Stages, each skipped on a rerun when its output exists:
#   1. the candidate's season panel and its season-so-far panel, under the scratch config
#   2. the whole-season chain against ctx110 (within-season WOWY by era, year over year, the 2026 top 20)
#   3. each season 1997-2026 rated from its first 7.15% (2027's 88 games), 10%, 25% and 50% of games, the prior
#      scales held at the 8 nearest seasons' mean, candidate and incumbent built by the same code; then the
#      within-season WOWY test of the rest of each season, both with the whole season's step 4
#   4. 2027 today under the candidate, into its own tables
# Log: outputs/exp127.log.  Run: bash scratch/2026-11-01_padding/run.sh
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
LOG=outputs/exp127.log
CFG=scratch/2026-11-01_padding/config_padbins.yaml
export OPENRAPM_JOBS=4
stage() { echo "=== $(date) $1" >> $LOG; }

if [ ! -f outputs/season_panel_padbins127.parquet ]; then
  stage "the candidate's panel"
  OPENRAPM_CONFIG=$CFG $PY -u -m openrapm data panel --force >> $LOG 2>&1 || exit 1
fi
if [ ! -f outputs/season_panel_so_far_padbins127.parquet ]; then
  stage "the candidate's panel of each season so far"
  OPENRAPM_CONFIG=$CFG $PY -u -m openrapm data panel --so-far 0.25,0.5,0.75 >> $LOG 2>&1 || exit 1
fi
if [ ! -f outputs/csv/wowy_padbins127_seasons.csv ]; then
  stage "the whole-season chain"
  OPENRAPM_CONFIG=$CFG REUSE=1 bash scripts/experiment_chain.sh padbins127 \
      points_with_expected_opponent_threes_late_as_scored >> $LOG 2>&1 || echo "=== chain exit status $?" >> $LOG
fi
for CUT in 0715 10 25 50; do
  for ARM in padbins127 ctx110; do
    TAG=${ARM}_cut${CUT}_held
    [ -f outputs/within/$TAG/summary.parquet ] && continue
    stage "the season so far at 0.$CUT, $ARM (tag $TAG)"
    if [ "$ARM" = padbins127 ]; then export OPENRAPM_CONFIG=$CFG; else unset OPENRAPM_CONFIG; fi
    $PY -u -m openrapm folds --models $ARM --base season_ratings_${ARM}_step3 --seasons 1997-2026 --tag $TAG \
        --cut 0.$CUT --hold-prior-scale season_ratings_${ARM}_step3 --jobs 4 >> $LOG 2>&1 || exit 1
  done
  unset OPENRAPM_CONFIG
  if [ ! -f outputs/csv/wowy_exp127_cut${CUT}_eras.csv ]; then
    stage "the within-season WOWY test of the rest of each season at 0.$CUT"
    $PY -u scripts/checks/151_wowy.py --systems=today=ctx110_cut${CUT}_held,padbins127=padbins127_cut${CUT}_held \
        --ref=today --step4-from=today:ctx110,padbins127:padbins127 --tag=exp127_cut${CUT} >> $LOG 2>&1 || exit 1
  fi
done
stage "experiment 127's builds done"
