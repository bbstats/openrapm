#!/usr/bin/env bash
# Before any push to main: what GitHub runs on main, run here first.
#
#     bash scripts/publish_check.sh        exit 0 when the commit about to be pushed would pass GitHub's three jobs
#
# GitHub's tests (.github/workflows/test.yml) run on a fresh clone with no scraped data: the lint check, every test,
# the tutorial, and the rules about how the code hangs together.  From 2026-11-07 they failed on 15 pushes to main in a
# row and nobody saw it: main's tests were older than the site files the working tree's code wrote, and the
# Similarity page's ratings had gone stale.  All of it would have failed here.
#
# The clone is of HEAD, the commit a push sends, so work not yet committed is not tested; the script says so.  The
# clone holds the site's data files, docs/data, as HEAD has them, which is what the site's own tests read.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=$ROOT/.venv/Scripts/python
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
  echo "note: uncommitted changes are not in the clone and are not tested"
fi
git clone -q "$ROOT" "$WORK/clone" || { echo "FAILED: the clone"; exit 1; }
cd "$WORK/clone" || exit 1
test ! -d data/raw && test ! -d data/stints || { echo "FAILED: the clone holds scraped data"; exit 1; }
# the clone's own code, not the working tree's the environment was installed from
export PYTHONPATH="$WORK/clone/src"
"$PY" -c "import openrapm, pathlib, sys; here = pathlib.Path(openrapm.__file__).resolve(); \
sys.exit(0 if pathlib.Path(sys.argv[1]).resolve() in here.parents else 1)" "$WORK/clone" \
  || { echo "FAILED: the tests would read the working tree's code, not the clone's"; exit 1; }

failed=""
run() {                       # $1: the job's name; the rest, its command
  local name=$1
  shift
  echo "=== $name"
  "$@" || failed="$failed; $name"
}
run "lint" uvx ruff@0.17.0 check .
run "tests" "$PY" -m pytest -q -p no:cacheprovider
run "tutorial" "$PY" examples/tutorial.py --small
if [ -n "$failed" ]; then
  echo "FAILED${failed#;}: do not push"
  exit 1
fi
echo "passed at $(git rev-parse --short HEAD): the lint check, the tests and the tutorial on a clone with no data"
