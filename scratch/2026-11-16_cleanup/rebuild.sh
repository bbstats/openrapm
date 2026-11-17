#!/usr/bin/env bash
# The ratings rebuilt after the deletion of 2026-11-16, by README.md's four passes on every season (the owner,
# 2026-11-16: "README recipe, all seasons").  Starts once refetch.sh has finished.  Names: rebuilt_base (pass 1),
# rebuilt_line (Box Adj's line), rebuilt (pass 3's test tables and saved models), rebuilt_early, rebuilt_within and
# rebuilt_k10 (its held-back games); the published tables under product.  A finished step leaves a marker in
# outputs/rebuild/done/ and is skipped on a rerun.  Nothing is published: the site step writes outputs/rebuild/site.
#     nohup bash scratch/2026-11-16_cleanup/rebuild.sh > /dev/null 2>&1 &
set -uo pipefail
ROOT=/a/code/spmm
cd "$ROOT" || exit 1
PY=$ROOT/.venv/Scripts/python
DIR=$ROOT/outputs/rebuild
mkdir -p "$DIR/done"
LOG=$DIR/rebuild.log
export PYTHONIOENCODING=utf-8
say() { echo "=== $(date '+%F %I:%M:%S %p') $*" | tee -a "$LOG"; }
FETCH=outputs/refetch/refetch.log
FROM=$(( $(wc -l < "$FETCH") + 1 ))   # a failure logged before this chain started was fixed and the fetch restarted
until tail -n +"$FROM" "$FETCH" | grep -q "refetch finished"; do
  tail -n +"$FROM" "$FETCH" | grep -q "FAILED" && { say "the refetch FAILED; nothing built"; exit 1; }
  sleep 120
done
step() {
  local name=$1; shift
  [ -f "$DIR/done/$name" ] && { say "skip $name"; return 0; }
  say "start $name: $*"
  "$PY" -u -m openrapm "$@" >> "$LOG" 2>&1 || { say "FAILED $name"; exit 1; }
  touch "$DIR/done/$name"; say "done $name"
}
say "rebuild from $(git rev-parse --short HEAD)"
S=1997-2026; E=1997-2016; W=2017-2026
# 1. ratings without Box Adj, at the test setting, keeping the boosters; their held-back games
step base_rate   rate --setting test --seasons $S --second-prior "" --name rebuilt_base --save-models rebuilt_base --jobs 4
step base_early  folds --models rebuilt_base --base season_ratings_rebuilt_base_step3 --seasons $E --tag rebuilt_base_early --jobs 4
step base_within folds --models rebuilt_base --base season_ratings_rebuilt_base_step3 --seasons $W --tag rebuilt_base_within --jobs 4
# 2. Box Adj's line, fitted on those held-back games through the ridge
step line        second-prior --models rebuilt_base --folds rebuilt_base_early,rebuilt_base_within --tag rebuilt_line --target rating
# 3. ratings with the line; their held-back games, at a quarter and at a tenth held back
step rate        rate --setting test --seasons $S --second-prior rebuilt_line --name rebuilt --save-models rebuilt --jobs 4
step early       folds --models rebuilt --base season_ratings_rebuilt_step3 --seasons $E --tag rebuilt_early --jobs 4
step within      folds --models rebuilt --base season_ratings_rebuilt_step3 --seasons $W --tag rebuilt_within --jobs 4
step k10         folds --models rebuilt --base season_ratings_rebuilt_step3 --seasons $S --tag rebuilt_k10 --folds-per-repeat 10 --jobs 4
# 4. the published tables, and the site's files into a folder of their own
export OPENRAPM_WRITE_PUBLISHED=1
step build       build --second-prior rebuilt_line --folds rebuilt_early,rebuilt_within --extrapolate-folds rebuilt_k10
step site        site --openrapm-decomposition --similarity --power --power-forecast --bias --out-dir "$DIR/site"
say "rebuild finished"
