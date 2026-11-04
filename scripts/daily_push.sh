#!/usr/bin/env bash
# The morning's one commit pushed to main from the site worktree (scripts/daily_update.sh), tried for about an hour.
#
#     bash scripts/daily_push.sh SITE LOG     exit 0 pushed, 1 every try failed, 2 the rebase on origin/main failed
#
# Each try after the first waits, then puts the commit on top of whatever main holds now, since someone else may have
# pushed since the morning's fetch, and pushes again.  A try that cannot reach GitHub just waits for the next one: on
# 2026-11-04 GitHub's Git service was down from 4:49 to 6:50 AM Eastern, and the job's two pushes, two seconds apart at
# 6:17 AM, both failed.  With these waits the tries fall at 6:17, 6:18, 6:23, 6:38 and 7:18.
#
# PUSH_WAITS, the minutes before each try after the first, is there for the tests.
set -uo pipefail
SITE=$1
LOG=$2
WAITS=${PUSH_WAITS:-1 5 15 40}

git -C "$SITE" push -q origin HEAD:main >> "$LOG" 2>&1 && exit 0
for minutes in $WAITS; do
  echo "$(date '+%F %I:%M:%S %p') the push failed, trying again in $minutes minutes" >> "$LOG"
  sleep $((minutes * 60))
  if git -C "$SITE" fetch -q origin main >> "$LOG" 2>&1; then
    if ! git -C "$SITE" rebase -q origin/main >> "$LOG" 2>&1; then
      git -C "$SITE" rebase --abort >> "$LOG" 2>&1
      exit 2
    fi
  fi
  git -C "$SITE" push -q origin HEAD:main >> "$LOG" 2>&1 && exit 0
done
exit 1
