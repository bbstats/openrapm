"""scripts/daily_files.py: the morning job publishes only a change beyond the built date, and names the last game
from both game logs."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "daily_files.py"


@pytest.fixture(scope="module")
def daily_files():
    spec = importlib.util.spec_from_file_location("daily_files", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(site: Path, *args) -> None:
    subprocess.run(["git", "-C", str(site), *args], check=True, capture_output=True)


def test_a_file_that_differs_only_by_its_built_date_is_not_a_change(daily_files, tmp_path):
    site = tmp_path / "site"
    (site / "docs" / "data").mkdir(parents=True)
    git(site, "init", "-q")
    git(site, "config", "user.email", "test@example.com")
    git(site, "config", "user.name", "test")
    path = site / "docs" / "data" / "ratings.json"
    path.write_text(json.dumps({"meta": {"built": "2026-10-31", "seasons": [2027]}, "rows": [[1, "Dončić", 2.5]]}),
                    encoding="utf-8")
    git(site, "add", "docs/data/ratings.json")
    git(site, "commit", "-q", "-m", "first")
    path.write_text(json.dumps({"meta": {"built": "2026-11-01", "seasons": [2027]}, "rows": [[1, "Dončić", 2.5]]}),
                    encoding="utf-8")
    assert daily_files.changed(site, ["docs/data/ratings.json"]) == []
    assert daily_files.main(["changed", str(site), "docs/data/ratings.json"]) == 3
    path.write_text(json.dumps({"meta": {"built": "2026-11-01", "seasons": [2027]}, "rows": [[1, "Dončić", 2.6]]}),
                    encoding="utf-8")
    assert daily_files.changed(site, ["docs/data/ratings.json"]) == ["docs/data/ratings.json"]
    assert daily_files.main(["changed", str(site), "docs/data/ratings.json"]) == 0
    # a file HEAD does not have is a change
    power = site / "docs" / "data" / "power.json"
    power.write_text('{"meta": {"built": "2026-11-01"}, "teams": []}', encoding="utf-8")
    assert daily_files.changed(site, ["docs/data/power.json"]) == ["docs/data/power.json"]
    with pytest.raises(SystemExit):                      # argparse: a command is required
        daily_files.main([])


def test_the_last_game_reads_the_playoff_log_too(daily_files, tmp_path):
    pd.DataFrame({"GAME_DATE": ["2027-04-10", "2027-04-12"]}).to_parquet(tmp_path / "2027_RS.parquet")
    assert daily_files.last_game(tmp_path, 2027) == "2027-04-12"
    pd.DataFrame({"GAME_DATE": ["2027-04-18", "2027-06-15"]}).to_parquet(tmp_path / "2027_PO.parquet")
    assert daily_files.last_game(tmp_path, 2027) == "2027-06-15"
    with pytest.raises(SystemExit, match="no game log of 2026"):
        daily_files.last_game(tmp_path, 2026)
    # a playoff log fetched before the playoffs: no dated row, passed over
    pd.DataFrame({"GAME_DATE": pd.Series([], dtype=object)}).to_parquet(tmp_path / "2026_PO.parquet")
    pd.DataFrame({"GAME_DATE": ["2026-04-12"]}).to_parquet(tmp_path / "2026_RS.parquet")
    assert daily_files.last_game(tmp_path, 2026) == "2026-04-12"
    assert daily_files.main(["last-game", str(tmp_path), "2027"]) == 0


def test_the_job_gives_its_programs_a_usable_stdin_before_it_starts_any():
    """Task Scheduler's headless console hands the job no usable stdin, and Python then cannot start a program."""
    lines = (SCRIPT.parent / "daily_update.sh").read_text(encoding="utf-8").splitlines()
    first_run = next(k for k, line in enumerate(lines) if "$PY " in line and not line.lstrip().startswith("#"))
    assert lines.index("exec < /dev/null") < first_run


def test_the_bias_file_can_never_hold_the_ratings_back():
    """The Bias page's file needs five outside systems' pages, and one may be down at 6 AM.  So neither of its two
    calls is a step, each is capped, the file is committed with the morning's three without being one of the files
    whose change decides the publish, and the job never creates it on main."""
    text = (SCRIPT.parent / "daily_update.sh").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if not line.strip().startswith("#")]
    calls = [line for line in lines if "$PY" in line and ("public-ratings" in line or "site --bias" in line)]
    assert len(calls) == 2 and all(line.startswith("/usr/bin/timeout 300 $PY") for line in calls)
    assert not any(line.startswith("step ") and "bias" in line.lower() for line in lines)
    files = next(line for line in lines if line.startswith("FILES="))
    assert "bias.json" not in files and 'BIAS=docs/data/bias.json' in lines
    assert 'daily_files.py changed "$SITE" $FILES' in text                 # the three ratings files alone decide
    assert 'if git -C "$SITE" cat-file -e "HEAD:$BIAS" 2>/dev/null; then' in lines
    assert 'git -C "$SITE" checkout -- "$BIAS" >> "$LOG" 2>&1' in lines     # a failed write leaves the published file
    assert 'step git -C "$SITE" add -- $ADD' in lines


def _job_lines() -> list:
    text = (SCRIPT.parent / "daily_update.sh").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if not line.strip().startswith("#")]


def _at(lines: list, start: str, after: int = 0) -> int:
    """Where the first line opening with `start` is, from line `after` on."""
    return next(k for k, line in enumerate(lines) if k >= after and line.startswith(start))


POOL_FIELDS = ["id", "n", "s", "p", "o", "d", "t"]


def test_the_similarity_pool_is_stale_when_a_rating_or_the_possessions_moved(daily_files, tmp_path):
    """2026-11-07: the ratings of finished seasons changed on the site and the Similarity page kept the old ones,
    Jokić 2026 at +9.2 where the Ratings page read +8.7."""
    ratings = {"meta": {}, "rows": [{"id": 203999, "s": 2026, "n": "Nikola Jokić", "o": 7.0, "d": 1.8, "t": 8.7,
                                     "p": 5269},
                                    {"id": 2, "s": 1997, "n": "Byron Scott", "o": 0.9, "d": -1.5, "t": -0.6,
                                     "p": 3000}]}

    def pool(*rows):
        return {"meta": {"fields": POOL_FIELDS}, "rows": [list(row) for row in rows]}
    scott = (2, "Byron Scott", 1997, 3000, 0.9, -1.5, -0.6)
    assert daily_files.stale_pool(ratings, pool(scott, (203999, "Nikola Jokić", 2026, 5269, 7.0, 1.8, 8.7))) == []
    assert daily_files.stale_pool(ratings, pool(scott, (203999, "Nikola Jokić", 2026, 5269, 7.0, 2.3, 9.2))) == \
        [[203999, "Nikola Jokić", 2026]]
    assert daily_files.stale_pool(ratings, pool((2, "Byron Scott", 1997, 3001, 0.9, -1.5, -0.6))) == []   # rounding
    assert daily_files.stale_pool(ratings, pool((2, "Byron Scott", 1997, 3010, 0.9, -1.5, -0.6))) == \
        [[2, "Byron Scott", 1997]]
    assert daily_files.stale_pool(ratings, pool((7, "Dan Schayes", 1997, 1148, -3.5, 2.1, -1.4))) == \
        [[7, "Dan Schayes", 1997]]                                          # a player-season the ratings do not hold
    published, stale, fresh = tmp_path / "ratings.json", tmp_path / "stale.json", tmp_path / "fresh.json"
    published.write_text(json.dumps(ratings), encoding="utf-8")
    stale.write_text(json.dumps(pool((203999, "Nikola Jokić", 2026, 5269, 7.0, 2.3, 9.2))), encoding="utf-8")
    fresh.write_text(json.dumps(pool(scott)), encoding="utf-8")
    assert daily_files.main(["similarity-stale", str(published), str(stale)]) == 0
    assert daily_files.main(["similarity-stale", str(published), str(fresh)]) == 3


def test_the_similarity_files_can_never_hold_the_ratings_back():
    """Never a step and capped at ten minutes; written only on a morning the pool is not in step; a rebuild that fails
    puts the published files back and takes out what it left; and the three ratings files still decide a publish."""
    lines = _job_lines()
    calls = [line for line in lines if "$PY" in line and "site --similarity" in line]
    assert calls == ['/usr/bin/timeout 600 $PY -u -m openrapm site --similarity --out-dir "$1" >> "$LOG" 2>&1']
    assert not any(line.startswith("step ") and "similarity" in line.lower() for line in lines)
    asked = _at(lines, '$PY scripts/daily_files.py similarity-stale "$1/data/ratings.json" "$2"')
    assert asked < lines.index(calls[0]) and "3) return 3 ;;" in lines[asked:lines.index(calls[0])]
    assert 'if git -C "$SITE" cat-file -e "HEAD:$SIMILAR/player_seasons.json" 2>/dev/null; then' in lines
    assert 'git -C "$SITE" checkout -- "$SIMILAR" >> "$LOG" 2>&1' in lines
    assert 'git -C "$SITE" clean -fdq -- "$SIMILAR" >> "$LOG" 2>&1 ;;' in lines
    files = next(line for line in lines if line.startswith("FILES="))
    assert "similarity" not in files and '0) ADD="$ADD $SIMILAR" ;;' in lines


def test_the_games_are_checked_after_the_stints_are_built_and_before_anything_reads_them():
    """A failed check gets one more fetch and build; a second failure is a step that fails, which stops the morning."""
    lines = _job_lines()
    stints = _at(lines, "step $PY -u -m openrapm data stints")
    check = _at(lines, "if ! $PY -u -m openrapm data check-games $SEASON")
    events = _at(lines, "step $PY -u -m openrapm data events")
    assert stints < check < events < _at(lines, "step $PY -u -m openrapm build")
    again = lines[check + 1:events]
    assert [line.split(" >>")[0] for line in again if line.startswith("step ")] == [
        "step $PY -u -m openrapm data ingest $SEASON $SEASON RS,PO --refresh",
        "step $PY -u -m openrapm data stints $SEASON $SEASON RS,PO --force",
        "step $PY -u -m openrapm data check-games $SEASON"]


def test_the_job_stops_after_the_ingest_once_the_seasons_finals_are_over():
    """The ingest is what brings the title-winning game in, so the question is asked right after it and before
    anything of the season is built; the stop is not a failure, and the owner is told once."""
    lines = _job_lines()
    ingest = _at(lines, "step $PY -u -m openrapm data ingest")
    asked = _at(lines, "$PY -u -m openrapm data in-progress $SEASON")
    assert ingest < asked < _at(lines, "step $PY -u -m openrapm data stints")
    stop = "\n".join(lines[asked:asked + 9])
    assert 'finish "stopped: $SEASON\'s Finals are over' in stop and stop.count('" 0 ;;') == 1
    assert 'grep -q "Finals are over" "$ROOT/outputs/daily/last.txt"' in stop and "notice " in stop
    assert '*) finish "FAILED: openrapm data in-progress $SEASON" 1 ;;' in stop


def test_a_dry_run_keeps_its_own_record_and_sends_no_notice():
    """outputs/daily/last.txt says how the last real morning ended, which the 8 AM task reads
    (scripts/daily_missed.sh)."""
    lines = _job_lines()
    assert 'RECORD=$ROOT/outputs/daily/last.txt' in lines
    assert '[ "$PUBLISH" = "1" ] || RECORD=$ROOT/outputs/daily/last_dry.txt' in lines
    assert 'echo "$DAY $1" > "$RECORD"' in lines
    assert '[ "$PUBLISH" = "1" ] && case "$1" in FAILED*) notice "OpenRAPM $DAY $1" ;; esac' in lines
