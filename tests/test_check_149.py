"""scripts/checks/149_matchup_screen.py (the matchup assignments as prior inputs): the arms and the placebo's names; and
the script starts."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from openrapm import features

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "149_matchup_screen.py"


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("matchup_screen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_lists_its_flag():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "--jobs" in result.stdout


def test_the_arms_are_today_one_assignment_per_side_and_the_same_assignments_shuffled(screen):
    arms = screen.arms()
    today = features.FEATURE_SETS["no_career_counts"]
    assert list(arms) == ["today", "matchups", "placebo"]
    assert arms["matchups"][0]["D"] == today["D"] + ["attackers_guarded_scoring_vs_league_avg"]
    assert arms["matchups"][0]["O"] == today["O"] + ["defenders_faced_stinginess_vs_league_avg"]
    assert arms["placebo"][0]["D"] == today["D"] + ["shuffled_attackers_guarded_scoring_vs_league_avg"]
    assert arms["placebo"][0]["O"] == today["O"] + ["shuffled_defenders_faced_stinginess_vs_league_avg"]
    assert screen.SEASONS == tuple(range(2018, 2027))


def _per_season(screen, changes: dict) -> pd.DataFrame:
    """Today's error 1.0 in every season; each arm's change from `changes` ({(arm, side, yardstick): a number}), plus a
    wobble of +-0.002 that alternates by season and averages out, its phase set by the arm."""
    rows = []
    for number, arm in enumerate(screen.arms()):
        for side in ("O", "D"):
            for yardstick in ("career", "neighbours"):
                for i, season in enumerate(screen.SEASONS):
                    wobble = 0.0 if arm == "today" else 0.002 * (1 if (i + number) % 2 == 0 else -1)
                    rows.append(dict(side=side, arm=arm, yardstick=yardstick, group="all", season=season,
                                     error=1.0 + changes.get((arm, side, yardstick), 0.0) + wobble))
    return pd.DataFrame(rows)


def test_a_pass_is_built_only_when_the_placebo_does_not_gain_too(screen):
    gain = {("matchups", "D", "career"): -0.02, ("matchups", "D", "neighbours"): -0.02}
    quiet = screen.decide(_per_season(screen, gain))
    assert quiet["placebo_clean"] and quiet["chosen"] == "matchups" and not quiet["blocked_by_placebo"]
    both = screen.decide(_per_season(screen, {**gain, ("placebo", "O", "career"): -0.02,
                                              ("placebo", "O", "neighbours"): -0.02}))
    assert both["verdicts"]["matchups"]["passes"] and not both["placebo_clean"]
    assert both["chosen"] is None and both["blocked_by_placebo"]


def test_nothing_is_built_when_nothing_beats_today(screen):
    decision = screen.decide(_per_season(screen, {}))
    assert decision["chosen"] is None and decision["best"] is None and decision["placebo_clean"]
