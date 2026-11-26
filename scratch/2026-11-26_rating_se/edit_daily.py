"""One-off: the ratings' standard errors in the morning job, scripts/daily_update.sh (run once)."""
from edit_folds import edit

DAILY = [
    ('''#      Similarity page's files, written again only on a morning when its player-seasons carry other ratings than the
#      published ones (`openrapm site --similarity`).  Both are committed with the four files above on a morning they
#      publish
''',
     '''#      Similarity page's files, written again only on a morning when its player-seasons carry other ratings than the
#      published ones (`openrapm site --similarity`).  Both are committed with the four files above on a morning they
#      publish
#   7. not a step either: the ratings' standard errors (`rating_errors` below), between the build and the site's files
'''),
    ('''# One run at a time.  A dry run that finds another at work stops at once and leaves every record alone; the morning
''',
     '''# The ratings' standard errors (src/openrapm/rating_error.py) are never a step, so they can never hold the ratings
# back.  The season is rated 30 more times, each with a tenth of its games held back, by the boosters the build has
# just saved (`openrapm folds --in-progress`), and the spread of the 30 goes into outputs/rating_errors_product.parquet
# in place of yesterday's rows (`openrapm errors`); the site step reads that file.  All of it waits for the file to
# exist: `OPENRAPM_WRITE_PUBLISHED=1 openrapm errors` makes it, for the finished seasons, on the day the standard
# errors are published, and until then the morning is what it was.  The folds took 2 minutes on 2026-11-26 and get 15;
# a morning they fail keeps yesterday's standard errors beside today's ratings.
ERRORS=$ROOT/outputs/rating_errors_product.parquet
SAVE=""
[ -f "$ERRORS" ] && SAVE="--save-models daily"
rating_errors() {
  say "start: folds --in-progress"
  rm -rf "$ROOT/outputs/within/daily_k10"
  /usr/bin/timeout 900 $PY -u -m openrapm folds --models daily --base season_ratings_product_step3 --seasons $SEASON --tag daily_k10 --folds-per-repeat 10 --in-progress >> "$LOG" 2>&1 || return 1
  say "start: errors"
  /usr/bin/timeout 300 $PY -u -m openrapm errors --seasons $SEASON --folds daily_k10 >> "$LOG" 2>&1
}

# One run at a time.  A dry run that finds another at work stops at once and leaves every record alone; the morning
'''),
    ('''step $PY -u -m openrapm build --seasons $SEASON --replace-seasons
''',
     '''step $PY -u -m openrapm build --seasons $SEASON --replace-seasons $SAVE
if [ -f "$ERRORS" ]; then
  rating_errors || NOTE="$NOTE; standard errors: not written, yesterday's stay"
fi
'''),
]

TEST = [
    ('''def _job_lines() -> list:''',
     '''def test_the_standard_errors_can_never_hold_the_ratings_back():
    """The ratings' standard errors rate the season 30 more times.  Neither of their two calls is a step, each is
    capped, and both wait for the errors file `openrapm errors` makes on the day they are published; the build keeps
    its boosters for them only then."""
    text = (SCRIPT.parent / "daily_update.sh").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if not line.strip().startswith("#")]
    calls = [line for line in lines if "$PY" in line and ("openrapm folds" in line or "openrapm errors" in line)]
    assert len(calls) == 2 and all(line.startswith("/usr/bin/timeout ") for line in calls)
    assert "--in-progress" in calls[0] and "--folds-per-repeat 10" in calls[0] and "--tag daily_k10" in calls[0]
    assert "--folds daily_k10" in calls[1]
    assert 'ERRORS=$ROOT/outputs/rating_errors_product.parquet' in lines
    assert '[ -f "$ERRORS" ] && SAVE="--save-models daily"' in lines
    build = lines.index("step $PY -u -m openrapm build --seasons $SEASON --replace-seasons $SAVE")
    assert lines[build + 1] == 'if [ -f "$ERRORS" ]; then' and lines[build + 2].startswith("rating_errors || NOTE=")
    assert 'rm -rf "$ROOT/outputs/within/daily_k10"' in lines           # yesterday's folds are other games' folds


def _job_lines() -> list:'''),
]

if __name__ == "__main__":
    edit("scripts/daily_update.sh", DAILY)
    edit("tests/test_daily_files.py", TEST)
