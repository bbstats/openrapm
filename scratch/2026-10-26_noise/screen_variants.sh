#!/bin/bash
# Step 3 of HANDOFF "START HERE" (2026-10-26): after the two noise controls (noise23.sh) finish, bring the branch
# split-robust's prior settings into this tree (a fast-forward), check that today's prior at random draw 0 still
# gives back creation110's folds (check 155 dry, 2025), then check 155's base arm at five random draws for each
# proposal, the cheapest first, and the comparison.  Log outputs/screen_variants.log.
cd /a/code/spmm
PY=.venv/Scripts/python
until grep -q "^noise23 done" outputs/noise23.log 2>/dev/null; do sleep 60; done
date; echo "noise controls done"
git merge --ff-only split-robust || { echo "fast-forward failed"; exit 1; }
git log --oneline -1
$PY -u scripts/checks/155_input_screen.py dry --tag screen112 --seasons 2025 > outputs/screen_variants_dry.log 2>&1
cat outputs/screen_variants_dry.log
worst=$(grep "largest rating difference" outputs/screen_variants_dry.log | awk '{print $NF}')
awk -v w="$worst" 'BEGIN { exit !(w != "" && w + 0 <= 1e-6) }' || { echo "dry: today's prior misses creation110 by '$worst'"; exit 1; }
for variant in out_of_bag60 split_per_bag bags30; do
  date; echo "=== variant $variant"
  $PY -u scripts/checks/155_input_screen.py run --tag screen112 --workers 2 --fitters 6 --variant $variant
  echo "variant $variant exit $?"
  $PY -u scripts/checks/155_input_screen.py variants --tag screen112
done
date; echo "screen_variants done"
