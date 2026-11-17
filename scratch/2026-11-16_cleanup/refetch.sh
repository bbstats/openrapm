#!/usr/bin/env bash
# The data again after the deletion of 2026-11-16 (notes/DECISIONS.md, "The cleanup of 2026-11-16"): every fetch and
# every built input, in the README's order.  A finished step leaves a marker in outputs/refetch/done/ and is skipped on
# a rerun; a fetch that fails is tried again after 5 minutes, up to 30 times, since every fetch keeps its files.
#     nohup bash scratch/2026-11-16_cleanup/refetch.sh > /dev/null 2>&1 &
set -uo pipefail
ROOT=/a/code/spmm
cd "$ROOT" || exit 1
PY=$ROOT/.venv/Scripts/python
DIR=$ROOT/outputs/refetch
mkdir -p "$DIR/done"
LOG=$DIR/refetch.log
export PYTHONIOENCODING=utf-8
say() { echo "=== $(date '+%F %I:%M:%S %p') $*" | tee -a "$LOG"; }
step() {
  local name=$1 tries=$2; shift 2
  [ -f "$DIR/done/$name" ] && { say "skip $name"; return 0; }
  local n
  for n in $(seq 1 "$tries"); do
    say "start $name, try $n: $*"
    if "$PY" -u -m openrapm "$@" >> "$LOG" 2>&1; then
      touch "$DIR/done/$name"; say "done $name"; return 0
    fi
    say "failed $name, try $n"
    [ "$n" -lt "$tries" ] && sleep 300
  done
  say "FAILED $name after $tries tries"; exit 1
}
say "refetch from $(git rev-parse --short HEAD)"
step ingest         30 data ingest 1997 2026 RS,PO
step ingest_2027    30 data ingest 2027 2027 RS,PO --refresh
step stints          1 data stints 1997 2027 RS,PO
step events          1 data events 1997 2027
step careers        30 data careers
step hustle_games   30 data hustle-games 2017 2026
step dashboards     30 data dashboards 2027 2027
step panel           1 data panel
step panel_so_far    1 data panel --so-far 0.25,0.5,0.75
step public_ratings  5 data public-ratings
say "refetch finished"
