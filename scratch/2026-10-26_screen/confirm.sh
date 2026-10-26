#!/bin/bash
# Experiment 112's confirming build, after the morning's daily update: the survivor's chain, a noise control on
# creation110 (re-dealt with --fold-assignment 1, the first on today's incumbent), then check 151 with both beside
# creation110 and check 154 on the survivor's folds.  Log outputs/screen112_confirm.log.
cd /a/code/spmm
PY=.venv/Scripts/python
TODAY=$(date +%F)
until grep -q "^$TODAY" outputs/daily/last.txt 2>/dev/null || [ "$(date +%H%M)" -ge 0800 ]; do sleep 60; done
date; echo "daily update: $(cat outputs/daily/last.txt)"
bash scripts/experiment_chain.sh screen112 points_with_expected_opponent_threes_late_as_scored "--features hustle_no_three_pct_self_creation_unassisted_inside"
echo "candidate chain exit $?"; date
bash scripts/experiment_chain.sh noise_creation110 points_with_expected_opponent_threes_late_as_scored "--fold-assignment 1"
echo "noise control chain exit $?"; date
$PY -u scripts/checks/151_wowy.py --systems=creation110,noise_creation110,screen112 --ref=creation110 --tag=screen112_confirm
echo "151 exit $?"; date
$PY -u scripts/checks/154_self_creation.py --folds=screen112 --tag=selfcreation_screen112 > outputs/selfcreation_screen112_154.log 2>&1
echo "154 exit $?"; date; echo "confirm done"
