#!/bin/bash
# Adoption of experiment 127 (the owner, 2026-11-01: "Adopt"), the part before the full build: today's panels kept
# aside, both panels rebuilt under config.yaml with pad_bin_target league, then the shrink's tenth folds and the swap
# part of the prior refitted on the new boosters and folds, the two side by side.  Log: outputs/adopt127.log.
# Then: settings.py names them, `openrapm build --jobs 4` rebuilds 1997-2027, `openrapm site ...` the pages.
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
LOG=outputs/adopt127.log
export OPENRAPM_JOBS=4
stage() { echo "=== $(date) $1" >> $LOG; }

if [ ! -f outputs/season_panel_before_padbins127.parquet ]; then
  stage "today's panels kept as season_panel_before_padbins127 and season_panel_so_far_before_padbins127"
  cp outputs/season_panel.parquet outputs/season_panel_before_padbins127.parquet
  cp outputs/season_panel_so_far.parquet outputs/season_panel_so_far_before_padbins127.parquet
fi
if [ outputs/season_panel.parquet -ot config.yaml ]; then
  stage "the panel, every season, under the shipped config"
  $PY -u -m openrapm data panel --force >> $LOG 2>&1 || exit 1
fi
if [ outputs/season_panel_so_far.parquet -ot config.yaml ]; then
  stage "the panel of each season so far"
  $PY -u -m openrapm data panel --so-far 0.25,0.5,0.75 --force >> $LOG 2>&1 || exit 1
fi
stage "the tenth folds padbins127_k10 and the line ctx127, side by side"
if [ ! -f outputs/within/padbins127_k10/summary.parquet ]; then
  $PY -u -m openrapm folds --models padbins127 --base season_ratings_padbins127_step3 --seasons 1997-2026 \
      --tag padbins127_k10 --folds-per-repeat 10 --jobs 4 > outputs/padbins127_k10.log 2>&1 &
  K10=$!
fi
if [ ! -f outputs/second_prior/ctx127/run.json ]; then
  $PY -u -m openrapm second-prior --models padbins127 --folds padbins127_early,padbins127_within --tag ctx127 \
      --inputs context_and_types_centred > outputs/ctx127_line.log 2>&1 || { stage "the line FAILED"; exit 1; }
  stage "the line ctx127 done"
fi
if [ -n "$K10" ]; then wait $K10 || { stage "the tenth folds FAILED"; exit 1; }; fi
stage "the tenth folds done"
stage "adoption's builds before the full build done"
