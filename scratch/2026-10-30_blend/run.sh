#!/bin/bash
# Experiment 121: the line's penalties and fits (resumes), the quick screen of today's prior and five blends at two
# random draws, then the table.
cd /a/code/spmm
PY=.venv/Scripts/python
CHECK=scripts/checks/155_input_screen.py
date; $PY -u $CHECK linear --tag blend121 --processes 4 || exit 1
date; $PY -u $CHECK run --tag blend121 --seeds 2 --workers 2 --fitters 6 || exit 1
date; $PY -u $CHECK blends --tag blend121
date; echo "experiment 121 done"
