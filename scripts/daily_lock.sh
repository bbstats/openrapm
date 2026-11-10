#!/usr/bin/env bash
# The morning job's lock, so two runs never write the same files.  Sourced, not run:
#
#     . scripts/daily_lock.sh
#     take_lock WHO      0 when this run now holds the lock, 1 when another run holds it
#     release_lock       the lock given back
#     lock_held          0 while another run holds a lock younger than LOCK_STALE_SECONDS
#     morning_window     0 from 5:50 to 7:30 AM, the hours the morning job has to itself
#
# scripts/daily_update.sh takes the lock before its first step and gives it back as it ends.  scripts/experiment_chain.sh
# waits while the lock is held or the morning's hours are on the clock, since a full build reads and writes the stints,
# the panel and the ratings tables the job is rebuilding.  A long command typed by hand that writes the published
# tables takes it the same way:
#
#     ( . scripts/daily_lock.sh; take_lock "wins rebuild" || exit 1; trap release_lock EXIT; COMMAND )
#
# The lock is a folder, outputs/daily/running.lock: mkdir either makes it or fails, as one step, so two runs cannot
# both take it.  A run that dies without giving it back leaves it behind; the scheduled task is stopped after 2 hours,
# so a lock older than 3 is taken over, by renaming it first, which again only one run can do.
#
# DAILY_LOCK, LOCK_STALE_SECONDS and MORNING_NOW (the clock as HHMM) are there for the tests.
DAILY_LOCK=${DAILY_LOCK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs/daily/running.lock}
LOCK_STALE_SECONDS=${LOCK_STALE_SECONDS:-10800}

lock_age() {                  # seconds since the lock was taken; a lock without its record is as old as can be
  local taken
  taken=$(cat "$DAILY_LOCK/taken" 2>/dev/null) || taken=0
  echo $(( $(date +%s) - ${taken:-0} ))
}

lock_held() { [ -d "$DAILY_LOCK" ] && [ "$(lock_age)" -lt "$LOCK_STALE_SECONDS" ]; }

_mark_lock() { date +%s > "$DAILY_LOCK/taken"; echo "$1" > "$DAILY_LOCK/owner"; }

take_lock() {                 # $1: who is taking it, for whoever finds it held
  mkdir -p "$(dirname "$DAILY_LOCK")"
  if mkdir "$DAILY_LOCK" 2>/dev/null; then _mark_lock "$1"; return 0; fi
  lock_held && return 1
  mv "$DAILY_LOCK" "$DAILY_LOCK.old.$$" 2>/dev/null || return 1
  rm -rf "$DAILY_LOCK.old.$$"
  mkdir "$DAILY_LOCK" 2>/dev/null || return 1
  _mark_lock "$1"
}

release_lock() { rm -rf "$DAILY_LOCK"; }

lock_owner() { cat "$DAILY_LOCK/owner" 2>/dev/null || echo "an unnamed run"; }

morning_window() {
  local now=${MORNING_NOW:-$(date +%H%M)}
  now=$((10#$now))
  [ "$now" -ge 550 ] && [ "$now" -lt 730 ]
}
