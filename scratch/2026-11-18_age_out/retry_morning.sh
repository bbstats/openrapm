#!/usr/bin/env bash
# 2026-11-19: the morning job stopped at its games check on 0022600265, ATL at OKC of 2026-11-18, whose play-by-play
# the league had posted with every description blank, so a substitution does not say who came in and the game builds
# no stints.  This asks the league for that one game's play-by-play again every 20 minutes, and runs the morning job
# once its substitutions carry their text.  It never runs the job while they are blank, so no failure notice is sent.
#     bash scratch/2026-11-18_age_out/retry_morning.sh [TRIES]
cd /a/code/spmm || exit 1
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
GAME=0022600265
FILE=data/raw/pbp/2027/$GAME.parquet
OUT=outputs/age_out/retry_morning.out
TRIES=${1:-6}
for try in $(seq 1 "$TRIES"); do
  rm -f "$FILE"
  $PY -u -m openrapm data ingest 2027 2027 RS,PO --refresh >> "$OUT" 2>&1
  filled=$($PY -I -c "
import pandas as pd, sys
try:
    t = pd.read_parquet('$FILE')
except Exception:
    print(0); sys.exit()
sub = t[t.actionType == 'Substitution']
print(int(sub.description.fillna('').str.contains('SUB:').sum()))")      # blank ones hold one space, not nothing
  echo "=== $(date '+%F %I:%M:%S %p') try $try: substitutions with their text: $filled" >> "$OUT"
  if [ "${filled:-0}" -gt 0 ]; then
    echo "=== $(date '+%F %I:%M:%S %p') the play-by-play is whole: running the morning job" >> "$OUT"
    bash scripts/daily_update.sh >> "$OUT" 2>&1
    echo "=== $(date '+%F %I:%M:%S %p') the job ended: $(cat outputs/daily/last.txt)" >> "$OUT"
    exit 0
  fi
  [ "$try" -lt "$TRIES" ] && sleep 1200
done
echo "=== $(date '+%F %I:%M:%S %p') still blank after $TRIES tries" >> "$OUT"
exit 2
