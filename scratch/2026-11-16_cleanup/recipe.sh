#!/usr/bin/env bash
# README.md's model recipe, "Building the ratings", run as written on ten seasons, 2017-2026, under the names rc_*.
# Differences from the README, each only to fit ten seasons and stay off the published files: the early and within
# fold spans are 2017-2021 and 2022-2026; build names its tables rc and rates only these seasons; the site writes
# into outputs/rc_site from the rc table.  The data steps are not rerun: they rewrite the shared inputs.
set -uo pipefail
export PYTHONPATH=/a/code/spmm_cleanup/src
export OPENRAPM_CONFIG=A:/code/spmm/outputs/cleanup_proof/config.yaml
export PYTHONIOENCODING=utf-8
PY=/a/code/spmm/.venv/Scripts/python
LOG=/a/code/spmm/outputs/cleanup_proof/recipe.log
DONE=/a/code/spmm/outputs/cleanup_proof/done_recipe
mkdir -p "$DONE"
cd /a/code/spmm_cleanup || exit 1
say() { echo "=== $(date '+%F %I:%M:%S %p') $*" | tee -a "$LOG"; }
step() {
  local name=$1; shift
  [ -f "$DONE/$name" ] && { say "skip $name"; return 0; }
  say "start $name: $*"
  "$PY" -u -m openrapm "$@" >> "$LOG" 2>&1 || { say "FAILED $name"; exit 1; }
  touch "$DONE/$name"; say "done $name"
}
S=2017-2026; E=2017-2021; W=2022-2026
step base_rate   rate --setting test --seasons $S --second-prior "" --name rc_base --save-models rc_base --jobs 2
step base_early  folds --models rc_base --base season_ratings_rc_base_step3 --seasons $E --tag rc_base_early --jobs 2
step base_within folds --models rc_base --base season_ratings_rc_base_step3 --seasons $W --tag rc_base_within --jobs 2
step line        second-prior --models rc_base --folds rc_base_early,rc_base_within --tag rc_line --target rating
step mine_rate   rate --setting test --seasons $S --second-prior rc_line --name rc_mine --save-models rc_mine --jobs 2
step mine_early  folds --models rc_mine --base season_ratings_rc_mine_step3 --seasons $E --tag rc_mine_early --jobs 2
step mine_within folds --models rc_mine --base season_ratings_rc_mine_step3 --seasons $W --tag rc_mine_within --jobs 2
step mine_k10    folds --models rc_mine --base season_ratings_rc_mine_step3 --seasons $S --tag rc_mine_k10 \
                   --folds-per-repeat 10 --jobs 2
step build       build --seasons $S --name rc --second-prior rc_line --folds rc_mine_early,rc_mine_within \
                   --extrapolate-folds rc_mine_k10
step site        site --ratings /a/code/spmm/outputs/season_ratings_rc.parquet --openrapm-decomposition --similarity \
                   --power --bias --out-dir /a/code/spmm/outputs/rc_site
say "recipe finished"
