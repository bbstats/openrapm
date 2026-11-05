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
