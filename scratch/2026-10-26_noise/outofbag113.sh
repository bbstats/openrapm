#!/bin/bash
# Step 4 of HANDOFF "START HERE" (2026-10-26): experiment 113, the screen's winner built in full (60 bags, each
# player's prior the mean of the bags that did not draw him), then three noise controls of its own (every bag's
# resample and seed redrawn, --bag-draw 1-3), then check 151 with today's build, its three noise controls, the
# candidate and its three.  Log outputs/outofbag113.log.
cd /a/code/spmm
PY=.venv/Scripts/python
FLAGS="--bags 60 --prediction out_of_bag"
date
bash scripts/experiment_chain.sh outofbag113 points_with_expected_opponent_threes_late_as_scored "$FLAGS"
echo "outofbag113 chain exit $?"; date
for draw in 1 2 3; do
  name=noise${draw}_outofbag113
  [ "$draw" = "1" ] && name=noise_outofbag113
  bash scripts/experiment_chain.sh $name points_with_expected_opponent_threes_late_as_scored "$FLAGS --bag-draw $draw"
  echo "$name chain exit $?"; date
done
$PY -u scripts/checks/151_wowy.py --systems=creation110,noise_creation110,noise2_creation110,noise3_creation110,outofbag113,noise_outofbag113,noise2_outofbag113,noise3_outofbag113 --ref=creation110 --tag=outofbag113_all
echo "151 exit $?"; date; echo "outofbag113 done"
