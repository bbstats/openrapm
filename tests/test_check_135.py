"""scripts/checks/135_robust_review.py starts, answers to the flags notes/HANDOFF.md and its docstring use, and refuses
to run without a candidate."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "135_robust_review.py"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, encoding="utf-8",
                          cwd=ROOT, timeout=300)


def test_help_lists_every_flag_the_docstring_uses():
    result = run("--help")
    assert result.returncode == 0, result.stderr
    for flag in ("--cands", "--controls", "--ref", "--refresh", "--out"):
        assert flag in result.stdout


def test_no_candidate_stops_the_run():
    result = run()
    assert result.returncode == 1 and "--cands=<name>" in result.stderr
