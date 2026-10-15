"""scripts/checks/148_noise_column_screen.py (what check 146's placebo gained from): the arms and the noise columns'
names; and the script starts."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from openrapm import features

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "148_noise_column_screen.py"


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("noise_column_screen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_lists_its_flag():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "--jobs" in result.stdout


def test_the_arms_are_today_the_shuffled_dashboard_columns_and_plain_noise_columns_of_the_same_count(screen):
    arms = screen.arms()
    today = features.FEATURE_SETS["no_career_counts"]
    assert list(arms) == ["today", "noise_missing", "noise_dense"]
    for side in ("O", "D"):
        both = features.FEATURE_SETS["tracking_and_hustle"][side]
        assert arms["noise_dense"][0][side] == today[side] + screen.dense_names()
        assert len(arms["noise_missing"][0][side]) == len(both)
        assert len(arms["noise_dense"][0][side]) == len(today[side]) + 15
    assert screen.dense_names() == [f"noise_{i}" for i in range(15)] and screen.SEASONS == tuple(range(2018, 2027))
