#!/usr/bin/env bash
# scripts/publish_check.sh's three steps for a commit of the cleanup checkout, which has no .venv of its own: the lint
# check, every test and the tutorial, on a fresh clone with no data, with the main checkout's python.
#     bash scratch/2026-11-16_cleanup/clean_check.sh
set -uo pipefail
PY=/a/code/spmm/.venv/Scripts/python
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git clone -q /a/code/spmm_cleanup "$WORK/clone" || { echo "FAILED: the clone"; exit 1; }
cd "$WORK/clone" || exit 1
test ! -d data/raw && test ! -d data/stints || { echo "FAILED: the clone holds scraped data"; exit 1; }
export PYTHONPATH="$WORK/clone/src"
failed=""
run() { local name=$1; shift; echo "=== $name"; "$@" || failed="$failed; $name"; }
run lint uvx ruff@0.17.0 check .
run tests "$PY" -m pytest -q -p no:cacheprovider
run tutorial "$PY" examples/tutorial.py --small
[ -z "$failed" ] || { echo "FAILED${failed#;}"; exit 1; }
echo "passed at $(git rev-parse --short HEAD): the lint check, the tests and the tutorial on a clone with no data"
