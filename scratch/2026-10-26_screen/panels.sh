#!/bin/bash
# Experiment 112's panels: self-creation's counts again with the candidates, then both panels.
cd /a/code/spmm
PY=.venv/Scripts/python
date; $PY -c "from openrapm.data.config import load_config; from openrapm.data.creation_counts import build_creation_counts; build_creation_counts(load_config(), range(1997, 2028), force=True)" || exit 1
date; $PY -u -m openrapm data panel --force || exit 1
date; $PY -u -m openrapm data panel --so-far 0.25,0.5,0.75 --force || exit 1
date; echo "panels done"
