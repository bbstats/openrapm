"""scripts/checks/70_tradeset.py starts, answers to the flags the experiment chain and its docstring use, and stops
on a bad argument before it builds a single season's design."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "70_tradeset.py"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, encoding="utf-8",
                          cwd=ROOT, timeout=300)


def test_help_lists_every_flag_the_chain_and_the_docstring_use():
    result = run("--help")
    assert result.returncode == 0, result.stderr
    for flag in ("--rankings", "--first", "--last", "--seasons", "--lambdas", "--fill", "--target_off", "--target_def",
                 "--out", "--score", "--block", "--free_scale", "--team_effects"):
        assert flag in result.stdout


def test_an_unknown_team_term_stops_the_run():
    result = run("--team_effects=franchise")
    assert result.returncode == 2 and "invalid choice" in result.stderr
