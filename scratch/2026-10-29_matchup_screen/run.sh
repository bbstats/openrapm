#!/bin/bash
# Experiment 120: the matchup inputs through the Boruta screen (check 155, tag screen120, seasons 2018-2025), the
# booster settings searched again with the matchup inputs in (study tune120, the ten tune seasons), then the matchup
# screen and experiment 119's screen again at the tuned settings, and each side by side with the shipped settings.
# Every step resumes; the cache step checks the shipped boosters refitted give back creation110's saved priors, the
# first screen trial checks seed 0's folds against creation110's, and the dry step checks screen119's cache still
# gives back creation110's 2025 folds with today's code.
cd /a/code/spmm
PY=.venv/Scripts/python
CHECK=scripts/checks/155_input_screen.py
date; $PY -u $CHECK cache --tag screen120 || exit 1
date; $PY -u $CHECK run --tag screen120 --workers 2 --fitters 6 || exit 1
date; $PY -u $CHECK table --tag screen120
date; echo "screen120 at the shipped settings done"
date; $PY -u $CHECK tune --tag screen120 --study tune120 --trials 80 --workers 2 --fitters 6 || exit 1
date; echo "tune120 done"
date; $PY -u $CHECK run --tag screen120 --variant tune120 --all-arms --workers 2 --fitters 6 || exit 1
date; $PY -u $CHECK table --tag screen120 --variant tune120
date; $PY -u $CHECK tandem --tag screen120 --variant tune120
date; echo "screen120 at the tuned settings done"
date; $PY -u $CHECK dry --tag screen119 --seasons 2025 || exit 1
date; $PY -u $CHECK run --tag screen119 --variant tune120 --all-arms --workers 2 --fitters 6 || exit 1
date; $PY -u $CHECK table --tag screen119 --variant tune120
date; $PY -u $CHECK tandem --tag screen119 --variant tune120
date; echo "experiment 120 done"
