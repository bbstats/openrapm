#!/usr/bin/env bash
# The morning that never ran.  Task Scheduler runs this at 8:00 AM, two hours after scripts/daily_update.sh: when
# outputs/daily/last.txt does not open with today's date, the 6 AM job has not ended, and the owner's phone is told
# (scripts/daily_notify.sh).  A job that ends writes last.txt whether it published, found nothing new or failed, and a
# failure sends its own notice, so this speaks only when the job did not start, was stopped before its last line, or is
# still running after two hours.
#
# DAILY_FOLDER and TODAY are there for the tests.
exec < /dev/null
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FOLDER=${DAILY_FOLDER:-$ROOT/outputs/daily}
DAY=${TODAY:-$(date +%F)}
last=$(cat "$FOLDER/last.txt" 2>/dev/null)
case "$last" in
  "$DAY "*) exit 0 ;;
esac
# $BASH: the bash running this, not WSL's, which the name alone can mean on Windows
"$BASH" "$ROOT/scripts/daily_notify.sh" "OpenRAPM $DAY: the morning job has not finished. Last: ${last:-nothing on record}"
exit 0
