"""scripts/checks/90_swap_test.py starts, answers to the flags the experiment chain types, and stops on a bad
argument before it reads a single season's stints (thirty seasons of them take a minute to read)."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "90_swap_test.py"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, encoding="utf-8",
                          cwd=ROOT, timeout=300)


@pytest.fixture
def table(tmp_path) -> Path:
    """A two-player ratings table in the published sign."""
    path = tmp_path / "ratings.parquet"
    pd.DataFrame({"player_id": [1, 2], "season": [2020, 2020], "rating_off": [1.0, -1.0],
                  "rating_def": [0.5, -0.5]}).to_parquet(path, index=False)
    return path


def test_help_lists_every_flag_the_chain_and_the_docstring_use():
    result = run("--help")
    assert result.returncode == 0, result.stderr
    for flag in ("--rankings", "--ref", "--plain_rapm", "--checks", "--seed", "--shift_sd", "--contexts", "--first",
                 "--last", "--tag"):
        assert flag in result.stdout


def test_no_rankings_prints_the_docstring():
    result = run()
    assert result.returncode == 1 and "The swap test" in result.stderr


@pytest.mark.parametrize("bad, message", [("--ref=other", "is not one of ['mine']"),
                                          ("--plain_rapm=0", "--plain_rapm must be positive"),
                                          ("--contexts=full,tired", "['tired'] are not among")])
def test_a_bad_argument_stops_the_run_before_the_stints(table, bad, message):
    result = run(f"--rankings=mine={table}", bad, "--tag=never_written")
    assert result.returncode == 1 and message in result.stderr, result.stderr


def test_a_misspelled_flag_stops_the_run(table):
    result = run(f"--rankings=mine={table}", "--chekcs=0")
    assert result.returncode == 2 and "unrecognized arguments: --chekcs=0" in result.stderr
