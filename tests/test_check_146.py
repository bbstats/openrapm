"""scripts/checks/146_dashboard_screen.py (experiment 62): the arms, the placebo's shuffle, and the gate that decides
what is built, on made-up per-season errors; and the script starts."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openrapm import features

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "146_dashboard_screen.py"


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("dashboard_screen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_lists_its_flags():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "--jobs" in result.stdout and "--first" in result.stdout


def test_the_arms_are_the_incumbents_lists_with_each_arms_inputs_appended(screen):
    arms = screen.arms()
    assert list(arms) == ["today", "tracking", "hustle", "tracking_and_hustle", "placebo"]
    today = features.FEATURE_SETS["no_career_counts"]
    for name in ("tracking", "hustle", "tracking_and_hustle"):
        assert arms[name][0] == features.FEATURE_SETS[name]
    for side in ("O", "D"):
        both, fake = arms["tracking_and_hustle"][0][side], arms["placebo"][0][side]
        assert fake[:len(today[side])] == today[side] and len(fake) == len(both)
        assert fake[len(today[side]):] == [f"shuffled_{name}" for name in both[len(today[side]):]]
    assert screen.added_inputs("today") == 0
    assert screen.added_inputs("tracking_and_hustle") == screen.added_inputs("tracking") + screen.added_inputs("hustle")


def test_the_shuffle_keeps_each_seasons_values_and_the_missing_ones_in_place(screen):
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({"season": np.repeat([2018, 2019], 40), "side": np.tile(["O", "D"], 40),
                          "x": rng.normal(size=80)})
    panel.loc[[3, 10, 55], "x"] = np.nan
    out = screen.shuffle_within(panel, ["x"])
    assert out.shuffled_x.isna().to_numpy().tolist() == panel.x.isna().to_numpy().tolist()
    for _, rows in panel.groupby(["season", "side"]).indices.items():
        assert np.sort(out.shuffled_x.to_numpy()[rows][~np.isnan(out.shuffled_x.to_numpy()[rows])]).tolist() == \
            pytest.approx(np.sort(panel.x.to_numpy()[rows][~np.isnan(panel.x.to_numpy()[rows])]).tolist())
    assert not np.allclose(out.shuffled_x.fillna(0), panel.x.fillna(0))             # it did move players


def _per_season(screen, changes: dict) -> pd.DataFrame:
    """Today's error 1.0 in every season; each arm's change from `changes` ({(arm, side, yardstick): a number}), plus a
    wobble of +-0.002 that alternates by season and averages out, its phase set by the arm."""
    rows = []
    for number, arm in enumerate(screen.arms()):
        for side in ("O", "D"):
            for yardstick in ("career", "neighbours"):
                for i, season in enumerate(screen.SCREEN_SEASONS):
                    wobble = 0.0 if arm == "today" else 0.002 * (1 if (i + number) % 2 == 0 else -1)
                    change = changes.get((arm, side, yardstick), 0.0)
                    rows.append(dict(side=side, arm=arm, yardstick=yardstick, group="all", season=season,
                                     error=1.0 + change + wobble))
    return pd.DataFrame(rows)


def test_nothing_is_built_when_no_arm_beats_today_and_the_placebo_is_clean(screen):
    decision = screen.decide(_per_season(screen, {}))
    assert decision["chosen"] is None and decision["placebo_clean"]
    assert all(not verdict["passes"] for verdict in decision["verdicts"].values())


def test_the_arm_with_the_fewest_added_inputs_within_one_standard_error_of_the_best_is_chosen(screen):
    both = _per_season(screen, {("tracking", "D", "career"): -0.020, ("tracking_and_hustle", "D", "career"): -0.0201,
                                ("hustle", "D", "career"): -0.010})
    decision = screen.decide(both)
    assert decision["verdicts"]["tracking"]["passes"] and decision["verdicts"]["tracking_and_hustle"]["passes"]
    assert decision["best"] == "tracking_and_hustle" and decision["chosen"] == "tracking"
    apart = _per_season(screen, {("tracking", "D", "career"): -0.010, ("tracking_and_hustle", "D", "career"): -0.030})
    assert screen.decide(apart)["chosen"] == "tracking_and_hustle"


def test_a_placebo_that_gains_is_flagged(screen):
    decision = screen.decide(_per_season(screen, {("placebo", "D", "career"): -0.02,
                                                  ("placebo", "D", "neighbours"): -0.02,
                                                  ("tracking", "D", "career"): -0.02,
                                                  ("tracking", "D", "neighbours"): -0.02}))
    assert not decision["placebo_clean"] and decision["verdicts"]["placebo"]["passes"] is True
    quiet = screen.decide(_per_season(screen, {("tracking", "D", "career"): -0.02,
                                               ("tracking", "D", "neighbours"): -0.02}))
    assert quiet["placebo_clean"] and quiet["chosen"] == "tracking"


def test_an_arm_worse_on_the_neighbours_yardstick_does_not_pass(screen):
    decision = screen.decide(_per_season(screen, {("tracking", "D", "career"): -0.02,
                                                  ("tracking", "D", "neighbours"): 0.01}))
    assert not decision["verdicts"]["tracking"]["passes"]
