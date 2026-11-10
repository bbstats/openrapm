#!/usr/bin/env bash
# The season in progress, updated every morning and published (the owner, 2026-10-23: "we also need to pull data
# regularly(!)"; 2026-10-24: "You can publish", "6am is great").  Run by Windows Task Scheduler on this PC at 6 AM
# Eastern, after the night's games are final; no Claude session is needed.  Each step runs only if the one before it
# worked, and the first failure stops the morning, leaving yesterday's ratings on the site.
#
#   1. the game log fetched again, through yesterday in US Eastern time, and every new game's play-by-play and box
#      score (`openrapm data ingest --refresh`).  Once the season's Finals are over the job stops here
#      (`openrapm data in-progress`): the season is then rated with every other, by a full build
#   2. the season's stints built again, then checked (`openrapm data check-games`): every game of the game log has
#      stints, and its play-by-play scores what its box score says.  A game that fails is fetched and built once more;
#      if it fails again the morning stops, since the ratings would be published without it
#   3. the season's play-by-play counts, the season-to-date dashboards, then the season panel (which also rebuilds the
#      season's roles rows and bio feed); the new games' per-game hustle box scores and matchups are asked for
#      between, not as steps, since nothing the morning publishes reads them
#   4. `openrapm build --seasons SEASON --replace-seasons`: the season rated and written into the published table in
#      place of yesterday's rows, every finished season as it was
#   5. `openrapm site --openrapm-decomposition --power` into the site worktree, then the four data files committed and
#      pushed to main, where GitHub Pages serves docs/, when one of them changed beyond the date it was built
#      (scripts/daily_files.py).  The ratings page opens on the season in progress only once its median player shown has
#      250 offensive possessions (openrapm site, default_season).  docs/data/power.json is each team's current ratings
#      times its players' shares of its possessions over its last 10 games.
#   6. not steps, so they can never hold the ratings back: the five outside rating systems asked for their tables
#      (`openrapm data public-ratings`) and the Bias page's file written from them (`openrapm site --bias`); and the
#      Similarity page's files, written again only on a morning when its player-seasons carry other ratings than the
#      published ones (`openrapm site --similarity`).  Both are committed with the four files above on a morning they
#      publish
#
# A morning that ends FAILED sends one line to the owner's phone (scripts/daily_notify.sh), and a second task at 8 AM
# sends one when the morning never ran (scripts/daily_missed.sh).  The job holds a lock while it runs
# (scripts/daily_lock.sh), which a full build waits for, and it alone says it means to write the published tables
# (OPENRAPM_WRITE_PUBLISHED, src/openrapm/commands/published.py).
#
# The site worktree, SITE, is a checkout of origin/main that only this job writes: it is reset to origin/main each
# morning, so the working tree in ROOT, its branch and its uncommitted work are never touched.  Made once with
#     git -C A:/code/spmm worktree add --detach A:/code/spmm_site origin/main
#
# Log: outputs/daily/<date>.log, one a day; outputs/daily/last.txt says how the last morning ended.
# By hand: bash scripts/daily_update.sh            (PUBLISH=0 bash scripts/daily_update.sh builds, writes the site's
#                                                   files to outputs/daily/site_dry and does not push; it says how it
#                                                   ended in outputs/daily/last_dry.txt and sends no notice)
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
# a dry run keeps its own record, so last.txt always says how the last real morning ended, which the 8 AM task reads
RECORD=$ROOT/outputs/daily/last.txt
[ "$PUBLISH" = "1" ] || RECORD=$ROOT/outputs/daily/last_dry.txt
mkdir -p "$ROOT/outputs/daily"
cd "$ROOT" || exit 1
export OPENRAPM_WRITE_PUBLISHED=1
. "$ROOT/scripts/daily_lock.sh"

say() { echo "$(date '+%F %I:%M:%S %p') $*" >> "$LOG"; }
# one line to the owner's phone; under a timeout and never able to fail, so it can neither hold nor change an ending
notice() { /usr/bin/timeout 30 "$BASH" "$ROOT/scripts/daily_notify.sh" "$1" >> "$LOG" 2>&1 || true; }
finish() {
  say "$1"
  echo "$DAY $1" > "$RECORD"
  [ "$PUBLISH" = "1" ] && case "$1" in FAILED*) notice "OpenRAPM $DAY $1" ;; esac
  exit "$2"
}
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
# The Similarity page's files, docs/data/similarity/, are never a step either.  Its pool holds finished seasons, whose
# ratings change only when a new build is published; on such a morning the pool is written again, and on every other
# morning it is left alone, since its files carry the date they were built.  Ten minutes at most.
SIMILAR=docs/data/similarity
similarity_files() {          # $1: the site folder written into; $2: the pool held against that folder's ratings
  $PY scripts/daily_files.py similarity-stale "$1/data/ratings.json" "$2" >> "$LOG" 2>&1
  case $? in                  # 3: the pool carries the published ratings, nothing to write; 0: it does not
    0) ;;
    3) return 3 ;;
    *) return 1 ;;
  esac
  say "start: site --similarity"
  /usr/bin/timeout 600 $PY -u -m openrapm site --similarity --out-dir "$1" >> "$LOG" 2>&1
}

# One run at a time.  A dry run that finds another at work stops at once and leaves every record alone; the morning
# waits up to half an hour for a build typed by hand to give the lock back.
if ! take_lock "the morning job, process $$"; then
  [ "$PUBLISH" = "1" ] || { echo "another run holds the lock: $(lock_owner)" >&2; exit 1; }
  say "waiting: $(lock_owner) holds the lock"
  waited=0
  until take_lock "the morning job, process $$"; do
    [ "$waited" -lt 1800 ] || finish "FAILED: $(lock_owner) held the lock for 30 minutes" 1
    sleep 60
    waited=$((waited + 60))
  done
fi
trap release_lock EXIT

say "=== daily update of $SEASON"
step $PY -u -m openrapm data ingest $SEASON $SEASON RS,PO --refresh
# Once a team has won the title the season is finished: it may be trained on, so rating it alone would publish it under
# another model than every other season's, from hustle files the league had not finished filling in.  The job stops,
# tells the owner once, and stops quietly every morning after until the full build (notes/HANDOFF.md, "June").
say "start: data in-progress"
$PY -u -m openrapm data in-progress $SEASON >> "$LOG" 2>&1
case $? in
  0) ;;
  3) grep -q "Finals are over" "$ROOT/outputs/daily/last.txt" 2>/dev/null \
       || [ "$PUBLISH" != "1" ] \
       || notice "OpenRAPM $DAY: $SEASON's Finals are over and the morning job has stopped; the full build is next"
     finish "stopped: $SEASON's Finals are over, so it is no longer rated alone; the full build is next" 0 ;;
  *) finish "FAILED: openrapm data in-progress $SEASON" 1 ;;
esac
step $PY -u -m openrapm data stints $SEASON $SEASON RS,PO --force
# A game whose play-by-play failed to load or to parse is left out of the stints with one printed line, and the
# ratings were then published without it.  The check removes such a game's cached files, so one more fetch and build
# can mend a file that was not ready at 6 AM; a game that fails twice stops the morning.
say "start: data check-games"
if ! $PY -u -m openrapm data check-games $SEASON >> "$LOG" 2>&1; then
  say "check-games failed: the games it named are fetched and built once more"
  step $PY -u -m openrapm data ingest $SEASON $SEASON RS,PO --refresh
  step $PY -u -m openrapm data stints $SEASON $SEASON RS,PO --force
  step $PY -u -m openrapm data check-games $SEASON
fi
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

# ratings_list.json is written with decompose_openrapm.json, cut from it: the ratings page's list (commands/site.py)
FILES="docs/data/ratings.json docs/data/decompose_openrapm.json docs/data/ratings_list.json docs/data/power.json"
if [ "$PUBLISH" != "1" ]; then
  # a dry run writes the site's files to a scratch folder, so a failure in the site step shows before a morning does
  DRY=$ROOT/outputs/daily/site_dry
  mkdir -p "$DRY"
  step $PY -u -m openrapm site --openrapm-decomposition --power --power-forecast --out-dir "$DRY"
  bias_file "$DRY" || NOTE="$NOTE; bias: not written"
  similarity_files "$DRY" "$SITE/$SIMILAR/player_seasons.json"
  case $? in 0|3) ;; *) NOTE="$NOTE; similarity: not written" ;; esac
  finish "built, not published (PUBLISH=0); the site's files in $DRY$NOTE" 0
fi
step git -C "$SITE" fetch origin main
step git -C "$SITE" checkout --detach --force origin/main
step $PY -u -m openrapm site --openrapm-decomposition --power --power-forecast --out-dir "$SITE/docs"
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
# The Similarity page's files ride along the same way, on a morning its pool is not in step with the ratings.  A
# rebuild that fails puts the published files back, and takes out any file it left half written.
if git -C "$SITE" cat-file -e "HEAD:$SIMILAR/player_seasons.json" 2>/dev/null; then
  similarity_files "$SITE/docs" "$SITE/$SIMILAR/player_seasons.json"
  case $? in
    0) ADD="$ADD $SIMILAR" ;;
    3) ;;
    *) NOTE="$NOTE; similarity: not written, the published lists stay"
       git -C "$SITE" checkout -- "$SIMILAR" >> "$LOG" 2>&1
       git -C "$SITE" clean -fdq -- "$SIMILAR" >> "$LOG" 2>&1 ;;
  esac
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
