#!/bin/bash
# Experiment 111's panels: the season panel and the season-so-far panel rebuilt with the two uncentred unassisted parts.
cd /a/code/spmm
PY=.venv/Scripts/python
date; $PY -u -m openrapm data panel --force || exit 1
date; $PY -u -m openrapm data panel --so-far 0.25,0.5,0.75 --force || exit 1
date; echo panels done
