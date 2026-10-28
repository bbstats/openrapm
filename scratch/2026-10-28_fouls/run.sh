#!/bin/bash
# Experiment 119: both panels with the fouls, goaltends, blocked shots and counts over expected, then the Boruta
# screen on them (check 155, tag screen119) and its table.  The cache step checks the shipped boosters refitted from
# the new panel give back creation110's saved priors, and the run checks seed 0's folds against creation110's.
cd /a/code/spmm
PY=.venv/Scripts/python
cp outputs/season_panel.parquet scratch/2026-10-28_fouls/season_panel_before.parquet
date; $PY -u -m openrapm data panel --force || exit 1
date; $PY -u -m openrapm data panel --so-far 0.25,0.5,0.75 --force || exit 1
date; $PY -u scripts/checks/155_input_screen.py cache --tag screen119 || exit 1
date; $PY -u scripts/checks/155_input_screen.py run --tag screen119 --workers 2 --fitters 6 || exit 1
date; $PY -u scripts/checks/155_input_screen.py table --tag screen119
date; echo "screen119 done"
