#!/bin/bash
# Experiment 111, the estimated passes: both panels rebuilt with self-creation's counts, the chain, then check 154 with
# the new self-creation on the candidate's folds.  Log outputs/creation111_queue.log.
cd /a/code/spmm
PY=.venv/Scripts/python
cp outputs/season_panel.parquet outputs/season_panel_before_creation111b.parquet
cp outputs/season_panel_so_far.parquet outputs/season_panel_so_far_before_creation111b.parquet
date; echo "panel"
$PY -u -m openrapm data panel --force > outputs/creation111_panels.log 2>&1 || { echo "panel failed"; exit 1; }
date; echo "panel so far"
$PY -u -m openrapm data panel --so-far 0.25,0.5,0.75 --force >> outputs/creation111_panels.log 2>&1 || { echo "so-far panel failed"; exit 1; }
date; echo "chain start"
bash scripts/experiment_chain.sh creation111 points_with_expected_opponent_threes_late_as_scored "--features hustle_no_three_pct_self_creation_estimated"
echo "chain exit $?"; date
$PY -u scripts/checks/154_self_creation.py --folds=creation111 --creation=estimated --tag=selfcreation_creation111 \
  --save-rows=outputs/selfcreation_rows_creation111 > outputs/selfcreation_creation111_154.log 2>&1
echo "check 154 exit $?"; date; echo "queue done"
