"""scripts/checks/147_era_flag_screen.py (experiment 62's era-flag screen): the arms, the flags, and the gate that
decides what is built, on made-up per-season errors; and the script starts."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from openrapm import features

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "147_era_flag_screen.py"


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("era_flag_screen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_lists_its_flag():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "--jobs" in result.stdout


def test_the_arms_append_their_flags_to_the_incumbents_lists(screen):
    arms = screen.arms()
    today = features.FEATURE_SETS["no_career_counts"]
    assert list(arms) == ["today", "era_from_2005", "era_from_2014", "both_flags", "odd_seasons"]
    for side in ("O", "D"):
        assert arms["both_flags"][0][side] == today[side] + ["era_from_2005", "era_from_2014"]
        assert arms["odd_seasons"][0][side] == today[side] + ["odd_seasons"]
        assert arms["today"][0][side] == today[side]


def test_the_flags_are_zero_before_their_year_and_one_from_it_and_the_placebo_alternates(screen):
    panel = pd.DataFrame({"season": [2003, 2004, 2005, 2013, 2014, 2026]})
    out = screen.with_flags(panel)
    assert out.era_from_2005.tolist() == [0, 0, 1, 1, 1, 1]
    assert out.era_from_2014.tolist() == [0, 0, 0, 0, 1, 1]
    assert out.odd_seasons.tolist() == [1, 0, 1, 1, 0, 0]
    assert panel.columns.tolist() == ["season"]                       # the panel handed in is not changed


def _per_season(screen, changes: dict) -> pd.DataFrame:
    """Today's error 1.0 in every screen season; each arm's change from `changes` ({(arm, side, yardstick): a number}),
    plus a wobble of -0.002, 0 or +0.002 that cycles by season with a phase of its own, so no two arms differ by a
    constant (a paired t over seasons needs spread)."""
    rows = []
    for number, arm in enumerate(["today", *screen.GATE_ARMS]):
        for side in ("O", "D"):
            for yardstick in ("career", "neighbours"):
                for i, season in enumerate(screen.SCREEN.SCREEN):
                    wobble = 0.0 if arm == "today" else 0.002 * (((i + number) % 3) - 1)
                    change = changes.get((arm, side, yardstick), 0.0)
                    rows.append(dict(side=side, arm=arm, yardstick=yardstick, group="all", season=season,
                                     error=1.0 + change + wobble))
    return pd.DataFrame(rows)


def test_nothing_is_built_when_no_flag_beats_today(screen):
    decision = screen.decide(_per_season(screen, {}))
    assert decision["chosen"] is None and decision["placebo_clean"]


def test_a_flag_that_gains_as_much_as_the_placebo_is_not_built(screen):
    gains = {(arm, "O", yardstick): -0.02 for arm in ("era_from_2014", "odd_seasons")
             for yardstick in ("career", "neighbours")}
    decision = screen.decide(_per_season(screen, gains))
    assert not decision["placebo_clean"] and decision["chosen"] is None


def test_the_shortest_flag_list_within_one_standard_error_of_the_best_is_chosen(screen):
    gains = {(arm, "O", yardstick): -0.02 for arm in ("era_from_2014", "both_flags")
             for yardstick in ("career", "neighbours")}
    gains[("both_flags", "O", "career")] = -0.0201
    decision = screen.decide(_per_season(screen, gains))
    assert decision["placebo_clean"] and decision["chosen"] == "era_from_2014"
    assert decision["features"] == ["era_from_2014"]
