#!/bin/bash
# Two more noise controls on creation110 (HANDOFF "START HERE", step 1): today's build with the box-score prior's
# players split into its five training groups a third and a fourth way (--fold-assignment 2 and 3), one after the
# other, then check 151 with all four beside creation110.  Log outputs/noise23.log.
cd /a/code/spmm
PY=.venv/Scripts/python
date
bash scripts/experiment_chain.sh noise2_creation110 points_with_expected_opponent_threes_late_as_scored "--fold-assignment 2"
echo "noise2 chain exit $?"; date
bash scripts/experiment_chain.sh noise3_creation110 points_with_expected_opponent_threes_late_as_scored "--fold-assignment 3"
echo "noise3 chain exit $?"; date
$PY -u scripts/checks/151_wowy.py --systems=creation110,noise_creation110,noise2_creation110,noise3_creation110 --ref=creation110 --tag=noise3
echo "151 exit $?"; date; echo "noise23 done"
