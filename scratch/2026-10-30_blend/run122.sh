#!/bin/bash
# Experiment 122: experiment 121's blend with the booster on what Boruta keeps, the line's shares 0, 0.1, 0.25, 0.5.
cd /a/code/spmm
PY=.venv/Scripts/python
CHECK=scripts/checks/155_input_screen.py
date; $PY -u $CHECK run --tag blend122 --seeds 2 --workers 2 --fitters 6 || exit 1
date; $PY -u $CHECK blends --tag blend122
date; echo "experiment 122 done"
