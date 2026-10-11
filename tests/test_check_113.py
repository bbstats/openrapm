"""scripts/checks/113_swap_explain.py starts and answers to the flags HANDOFF.md uses
(`--team=BOS --player="Derrick White" --also=2024,2025`): a renamed step-5 function would otherwise surface only
the next time someone asks why a player's swap adjustment is what it is."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "113_swap_explain.py"


def test_help_lists_every_flag():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    for flag in ("--team", "--season", "--base", "--also", "--player"):
        assert flag in result.stdout
