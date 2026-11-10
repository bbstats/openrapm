#!/usr/bin/env bash
# One line to the owner's phone (the owner, 2026-11-09: "Phone push").  scripts/daily_update.sh calls this when a
# morning ends FAILED, and scripts/daily_missed.sh when a morning never ran: until then a failure wrote a line to
# outputs/daily/last.txt and reached nobody, on 4 of the 10 mornings from 2026-10-31.
#
#     bash scripts/daily_notify.sh "OpenRAPM 2026-11-10 FAILED: data stints"
#
# Where it goes is one line of outputs/daily/notify.conf, which is not in git:
#
#     ntfy https://ntfy.sh/TOPIC
#
# ntfy.sh needs no account: whoever posts to a topic's address reaches the phones subscribed to it in the ntfy app, so
# the topic's name is the only secret and stays out of git.  Without the file nothing is sent.
#
# It always exits 0, and the caller runs it under a timeout: a notice that cannot be sent must never fail or hold the
# job.  NOTIFY_CONF and NOTIFY_CURL are there for the tests.
CONF=${NOTIFY_CONF:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs/daily/notify.conf}
CURL=${NOTIFY_CURL:-curl}
[ -f "$CONF" ] || exit 0
kind="" where=""
read -r kind where < "$CONF"
case "$kind" in
  ntfy)
    "$CURL" -fsS -m 15 --retry 2 -o /dev/null -H "Title: OpenRAPM morning job" -H "Priority: high" -d "${1:-}" "$where" \
      || echo "the notice was not sent"
    ;;
  *) echo "outputs/daily/notify.conf names no way to send a notice: its line starts '$kind'" ;;
esac
exit 0
