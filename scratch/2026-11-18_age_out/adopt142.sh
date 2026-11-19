#!/usr/bin/env bash
# Adoption of experiment 142 (the owner, 2026-11-18, through the question tool: "Adopt and publish"): everything
# before the switch.  Nothing here touches the published tables or the settings the morning job runs on: the full build
# goes under its own name, product142, its line, list of inputs and folds named by flag.  A finished stage is skipped on
# a rerun.  After scratch/2026-11-10_adopt141/adopt141.sh; the season-so-far folders of the Power table's check 166 are
# not built, since no rating moves 0.1 (notes/DECISIONS.md, "Experiment 142: adopted").
#     nohup bash scratch/2026-11-18_age_out/adopt142.sh > /dev/null 2>&1 &
set -uo pipefail
cd /a/code/spmm || exit 1
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8 OPENRAPM_JOBS=4
NAME=noage142
OUT=product142
LIST=context_types_no_age_team_zones_centred
LOG=outputs/age_out/adopt142.log
stage() { echo "=== $(date '+%F %I:%M:%S %p') $*" >> "$LOG"; }

stage "adoption start, $(git rev-parse --short HEAD)"
if [ ! -f outputs/age_out/done/k10 ]; then
  stage "the tenth folds ${NAME}_k10, which read step 4's multipliers to a full season"
  $PY -u -m openrapm folds --models $NAME --base season_ratings_${NAME}_step3 --seasons 1997-2026 --tag ${NAME}_k10 \
      --folds-per-repeat 10 --jobs 4 > outputs/${NAME}_k10.log 2>&1 || { stage "the tenth folds FAILED"; exit 1; }
  touch outputs/age_out/done/k10; stage "the tenth folds done"
fi
if [ ! -f outputs/age_out/done/product142 ]; then
  stage "the full build of 1997-2027 at the product setting, under the name $OUT"
  $PY -u -m openrapm build --name $OUT --second-prior ${NAME}_line --second-prior-inputs $LIST \
      --folds ${NAME}_early,${NAME}_within --extrapolate-folds ${NAME}_k10 > outputs/build_${OUT}.log 2>&1 \
      || { stage "the full build FAILED"; exit 1; }
  touch outputs/age_out/done/product142; stage "the full build done"
fi
stage "adoption builds finished"
