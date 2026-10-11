"""scripts/checks/137_stat_calibration.py starts and answers to the flags its docstring and HANDOFF.md use: a renamed
import (the glossary, the feature lists) would otherwise surface only the next time someone draws the curves."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "137_stat_calibration.py"


def test_help_lists_every_flag():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    for flag in ("--alpha", "--tag", "--perms", "--bins"):
        assert flag in result.stdout
