#!/usr/bin/env bash
# The season in progress, updated every morning and published (the owner, 2026-10-23: "we also need to pull data
# regularly(!)"; 2026-10-24: "You can publish", "6am is great").  Run by Windows Task Scheduler on this PC at 6 AM
# Eastern, after the night's games are final; no Claude session is needed.  Each step runs only if the one before it
# worked, and the first failure stops the morning, leaving yesterday's ratings on the site.
#
#   1. the game log fetched again, through yesterday in US Eastern time, and every new game's play-by-play and box
#      score (`openrapm data ingest --refresh`)
#   2. the season's stints built again, its play-by-play counts, the season-to-date dashboards, then the season panel
#      (which also rebuilds the season's roles rows and bio feed); the new games' per-game hustle box scores and
#      matchups are asked for between, not as steps, since nothing the morning publishes reads them
#   3. `openrapm build --seasons SEASON --replace-seasons`: the season rated and written into the published table in
#      place of yesterday's rows, every finished season as it was
#   4. `openrapm site --openrapm-decomposition --power` into the site worktree, then the three data files committed and
#      pushed to main, where GitHub Pages serves docs/, when one of them changed beyond the date it was built
#      (scripts/daily_files.py).  The ratings page opens on the season in progress only once its median player shown has
#      250 offensive possessions (openrapm site, default_season).  docs/data/power.json is each team's current ratings
#      times its players' shares of its possessions over its last 10 games.
#   5. not steps, so they can never hold the ratings back: the five outside rating systems asked for their tables
#      (`openrapm data public-ratings`) and the Bias page's file written from them (`openrapm site --bias`), which is
#      committed with the three files above on a morning they publish
#
# The site worktree, SITE, is a checkout of origin/main that only this job writes: it is reset to origin/main each
# morning, so the working tree in ROOT, its branch and its uncommitted work are never touched.  Made once with
#     git -C A:/code/spmm worktree add --detach A:/code/spmm_site origin/main
#
# Log: outputs/daily/<date>.log, one a day; outputs/daily/last.txt says how the last morning ended.
# By hand: bash scripts/daily_update.sh            (PUBLISH=0 bash scripts/daily_update.sh builds, writes the site's
#                                                   files to outputs/daily/site_dry and does not push)
set -uo pipefail
# Task Scheduler starts this under `conhost --headless`, which hands it no usable stdin: any Python that then starts a
# program stops on WinError 6, as scripts/daily_files.py did on 2026-11-03 after the morning's ratings were built.
exec < /dev/null

ROOT=/a/code/spmm
SITE=/a/code/spmm_site
SEASON=${SEASON:-2027}
PUBLISH=${PUBLISH:-1}
PY=$ROOT/.venv/Scripts/python
DAY=$(date +%F)               # this PC keeps US Eastern time (Git Bash cannot read TZ=America/New_York)
LOG=$ROOT/outputs/daily/$DAY.log
mkdir -p "$ROOT/outputs/daily"
cd "$ROOT" || exit 1

say() { echo "$(date '+%F %I:%M:%S %p') $*" >> "$LOG"; }
finish() { say "$1"; echo "$DAY $1" > "$ROOT/outputs/daily/last.txt"; exit "$2"; }
step() {
  say "start: $*"
  "$@" >> "$LOG" 2>&1 || finish "FAILED: $*" 1
}
# The Bias page's file, docs/data/bias.json, is never a step: it needs five outside systems' pages, and a site that is
# down or redesigned at 6 AM must not hold the ratings back.  A system that fails keeps its last stored table
# (data/public_ratings.py), and when the file itself cannot be written the published one stays.  Each call gets five
# minutes; /usr/bin/timeout by its full path, since Windows has a timeout of its own on the PATH.
BIAS=docs/data/bias.json
NOTE=""
bias_file() {                 # $1: the site folder written into; fails when the file was not written
  say "start: data public-ratings"
  /usr/bin/timeout 300 $PY -u -m openrapm data public-ratings >> "$LOG" 2>&1 \
    || NOTE="$NOTE; public ratings: a system failed, its last stored table is read"
  say "start: site --bias"
  /usr/bin/timeout 300 $PY -u -m openrapm site --bias --out-dir "$1" >> "$LOG" 2>&1
}

say "=== daily update of $SEASON"
step $PY -u -m openrapm data ingest $SEASON $SEASON RS,PO --refresh
step $PY -u -m openrapm data stints $SEASON $SEASON RS,PO --force
step $PY -u -m openrapm data events $SEASON $SEASON --force
# the per-game hustle box scores and matchups (data/dashboards.py, data/matchups.py), not steps: the season in progress
# reads its hustle from the league's season-to-date dashboards (data/panel.py) and no model reads the matchups yet, so
# a game the league has not processed by morning must not stop the publish.  Both are asked again tomorrow.
say "start: hustle-games"
$PY -u -m openrapm data hustle-games $SEASON $SEASON >> "$LOG" 2>&1 || say "hustle games: some failed, asked again tomorrow"
say "start: matchup-games"
$PY -u -m openrapm data matchup-games $SEASON $SEASON >> "$LOG" 2>&1 || say "matchup games: some failed, asked again tomorrow"
step $PY -u -m openrapm data dashboards $SEASON $SEASON --force
step $PY -u -m openrapm data panel --force
step $PY -u -m openrapm build --seasons $SEASON --replace-seasons

FILES="docs/data/ratings.json docs/data/decompose_openrapm.json docs/data/power.json"
if [ "$PUBLISH" != "1" ]; then
  # a dry run writes the site's files to a scratch folder, so a failure in the site step shows before a morning does
  DRY=$ROOT/outputs/daily/site_dry
  mkdir -p "$DRY"
  step $PY -u -m openrapm site --openrapm-decomposition --power --out-dir "$DRY"
  bias_file "$DRY" || NOTE="$NOTE; bias: not written"
  finish "built, not published (PUBLISH=0); the site's files in $DRY$NOTE" 0
fi
step git -C "$SITE" fetch origin main
step git -C "$SITE" checkout --detach --force origin/main
step $PY -u -m openrapm site --openrapm-decomposition --power --out-dir "$SITE/docs"
# The Bias page's file rides along once main has the page: the job never creates it there.  It is not one of FILES,
# whose change decides the publish and each of which must exist, so an outside system that moved publishes nothing
# on a morning without new games.
ADD=$FILES
if git -C "$SITE" cat-file -e "HEAD:$BIAS" 2>/dev/null; then
  if bias_file "$SITE/docs"; then
    ADD="$FILES $BIAS"
  else
    NOTE="$NOTE; bias: not written, the published table stays"
    git -C "$SITE" checkout -- "$BIAS" >> "$LOG" 2>&1
  fi
fi
# the files carry the date they were built, so they differ every morning; only a change beyond it is published
$PY scripts/daily_files.py changed "$SITE" $FILES >> "$LOG" 2>&1
case $? in
  0) ;;
  3) finish "nothing new to publish$NOTE" 0 ;;
  *) finish "FAILED: scripts/daily_files.py changed" 1 ;;
esac
LAST_GAME=$($PY scripts/daily_files.py last-game "$ROOT/data/raw/gamelog" $SEASON) || finish "FAILED: the last game's date" 1
step git -C "$SITE" add -- $ADD
step git -C "$SITE" commit -q -m "Daily update: $SEASON through the games of $LAST_GAME"
# tried for about an hour, on top of whatever main holds by then ($BASH: the bash running this job, not WSL's)
say "start: scripts/daily_push.sh"
"$BASH" "$ROOT/scripts/daily_push.sh" "$SITE" "$LOG"
case $? in
  0) finish "published $(git -C "$SITE" rev-parse --short HEAD)$NOTE" 0 ;;
  2) finish "FAILED: the rebase on origin/main, aborted; the day's commit is on $SITE's detached HEAD" 1 ;;
  *) finish "FAILED: every push to main over an hour; the day's commit is on $SITE's detached HEAD" 1 ;;
esac
