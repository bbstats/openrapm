#!/bin/bash
# Experiment 111: wait for the panels, then the chain, then check 154 with the new self-creation on its folds.
cd /a/code/spmm
PY=.venv/Scripts/python
until grep -q "panels done" outputs/creation111_panels.log; do
  if grep -q "Traceback" outputs/creation111_panels.log; then echo "panels failed"; exit 1; fi
  sleep 30
done
date; echo "chain start"
bash scripts/experiment_chain.sh creation111 points_with_expected_opponent_threes_late_as_scored "--features hustle_no_three_pct_self_creation_actual_makes"
echo "chain exit $?"; date
$PY -u scripts/checks/154_self_creation.py --folds=creation111 --creation=actual_makes --tag=selfcreation_creation111 \
  --save-rows=outputs/selfcreation_rows_creation111 > outputs/selfcreation_creation111_154.log 2>&1
echo "check 154 exit $?"; date; echo "queue done"
