#!/bin/bash
# After the publish of experiment 136: the morning job's dry run on the new settings, then the noise control on the new
# incumbent, the same build with the boosters' players dealt again.
cd "$(dirname "$0")/../.."
echo "=== $(date) the morning job's dry run"
PUBLISH=0 bash scripts/daily_update.sh; echo "dry run exit $?"
cat outputs/daily/last.txt
echo "=== $(date) the noise control"
OPENRAPM_JOBS=4 bash scripts/experiment_chain.sh noise_rating134 points_with_expected_opponent_threes_late_as_scored "--fold-assignment 1" > outputs/noise_rating134_chain.nohup 2>&1; echo "noise control exit $?"
echo "=== $(date) all done"
