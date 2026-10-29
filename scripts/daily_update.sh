#!/usr/bin/env bash
# The season in progress, updated every morning and published (the owner, 2026-10-23: "we also need to pull data
# regularly(!)"; 2026-10-24: "You can publish", "6am is great").  Run by Windows Task Scheduler on this PC at 6 AM
# Eastern, after the night's games are final; no Claude session is needed.  Each step runs only if the one before it
# worked, and the first failure stops the morning, leaving yesterday's ratings on the site.
#
#   1. the game log fetched again, through yesterday in US Eastern time, and every new game's play-by-play and box
#      score (`openrapm data ingest --refresh`)
#   2. the season's stints built again, its play-by-play counts, the new games' hustle numbers, the season-to-date
#      dashboards, then the season panel (which also rebuilds the season's roles rows and bio feed)
#   3. `openrapm build --seasons SEASON --replace-seasons`: the season rated and written into the published table in
#      place of yesterday's rows, every finished season as it was
#   4. `openrapm site --openrapm-decomposition` into the site worktree, then the two data files committed and pushed to
#      main, where GitHub Pages serves docs/.  The ratings page opens on the season in progress only once its median
#      player has 250 offensive possessions (openrapm site, default_season).  Once the power ratings page is on main,
#      `--power` writes its file too, docs/data/power.json: each team's current ratings times its players' shares of
#      its possessions over its last 10 games, pushed with the other two.
#
# The site worktree, SITE, is a checkout of origin/main that only this job writes: it is reset to origin/main each
# morning, so the working tree in ROOT, its branch and its uncommitted work are never touched.  Made once with
#     git -C A:/code/spmm worktree add --detach A:/code/spmm_site origin/main
#
# Log: outputs/daily/<date>.log, one a day; outputs/daily/last.txt says how the last morning ended.
# By hand: bash scripts/daily_update.sh            (PUBLISH=0 bash scripts/daily_update.sh builds and does not push)
set -uo pipefail

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

say "=== daily update of $SEASON"
step $PY -u -m openrapm data ingest $SEASON $SEASON RS,PO --refresh
step $PY -u -m openrapm data stints $SEASON $SEASON RS,PO --force
step $PY -u -m openrapm data events $SEASON $SEASON --force
step $PY -u -m openrapm data hustle-games $SEASON $SEASON
# the per-game matchups (data/matchups.py), not a step: no model reads them yet, so a game the league has not processed
# by morning must not stop the publish.  It is asked again tomorrow, and the panel leaves it out meanwhile.
say "start: matchup-games"
$PY -u -m openrapm data matchup-games $SEASON $SEASON >> "$LOG" 2>&1 || say "matchup games: some failed, asked again tomorrow"
step $PY -u -m openrapm data dashboards $SEASON $SEASON --force
step $PY -u -m openrapm data panel --force
step $PY -u -m openrapm build --seasons $SEASON --replace-seasons

[ "$PUBLISH" = "1" ] || finish "built, not published (PUBLISH=0)" 0
step git -C "$SITE" fetch origin main
step git -C "$SITE" checkout --detach --force origin/main
FILES="docs/data/ratings.json docs/data/decompose_openrapm.json"
POWER=""
if [ -f "$SITE/docs/power.html" ]; then POWER="--power"; FILES="$FILES docs/data/power.json"; fi
step $PY -u -m openrapm site --openrapm-decomposition $POWER --out-dir "$SITE/docs"
if git -C "$SITE" diff --quiet -- $FILES; then
  finish "nothing new to publish" 0
fi
LAST_GAME=$($PY -c "import pandas as pd; print(pd.read_parquet('data/raw/gamelog/${SEASON}_RS.parquet').GAME_DATE.max())")
step git -C "$SITE" add -- $FILES
step git -C "$SITE" commit -q -m "Daily update: $SEASON through the games of $LAST_GAME"
# someone else may have pushed to main since the fetch: put this one commit on top of theirs and push again
if ! git -C "$SITE" push -q origin HEAD:main >> "$LOG" 2>&1; then
  step git -C "$SITE" fetch origin main
  step git -C "$SITE" rebase -q origin/main
  step git -C "$SITE" push -q origin HEAD:main
fi
finish "published $(git -C "$SITE" rev-parse --short HEAD)" 0
