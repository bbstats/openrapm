"""scripts/checks/142_era_screen.py (experiment 56): the error after the line, and the gate that decides what is
built, on made-up per-season errors; and the script starts."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "142_era_screen.py"


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("era_screen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_lists_its_flag():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "--jobs" in result.stdout


def test_the_error_is_read_after_the_line_fitted_on_every_rated_player(screen):
    rated = pd.DataFrame({"possessions": [3000.0, 2500.0, 800.0, 400.0], "career": [2.0, 0.0, -1.0, 1.0],
                          "neighbours": [1.0, np.nan, 0.5, -0.5], "neighbours_possessions": [2000.0, 0.0, 900.0, 50.0],
                          "pre1997": [True, False, False, False]})
    exact = screen.errors_of(rated, 3.0 + 2.0 * rated.career.to_numpy())       # any straight line misses nothing
    assert all(value == pytest.approx(0.0, abs=1e-20) for (yardstick, _), value in exact.items()
               if yardstick == "career")
    noisy = screen.errors_of(rated, np.array([1.0, 0.0, 0.0, 0.0]))
    label, prior, weight = rated.career.to_numpy(), np.array([1.0, 0.0, 0.0, 0.0]), rated.possessions.to_numpy()
    a, b = screen.fitted_line(label, prior, weight)
    miss = label - a - b * prior
    assert noisy[("career", "all")] == pytest.approx(np.sum(weight * miss ** 2) / weight.sum())
    bench = rated.possessions.to_numpy() < 1000
    assert noisy[("career", "bench")] == pytest.approx(np.sum((weight * miss ** 2)[bench]) / weight[bench].sum())
    assert noisy[("career", "pre1997")] == pytest.approx(miss[0] ** 2)


def _per_season(screen, changes: dict) -> pd.DataFrame:
    """Today's error 1.0 in every season; each arm's change from `changes` ({(arm, side, yardstick): a number or a
    function of the season}, 0 elsewhere) plus a wobble of +-0.002 that alternates by season and averages exactly 0
    over the ten, its phase set by the arm, so every paired t is finite and no mean moves."""
    rows = []
    for number, arm in enumerate(screen.arms()):
        for side in ("O", "D"):
            for yardstick in ("career", "neighbours"):
                for i, season in enumerate(screen.SCREEN):
                    change = changes.get((arm, side, yardstick), 0.0)
                    change = change(season) if callable(change) else change
                    wobble = 0.0 if arm == "today" else 0.002 * (1 if (i + number) % 2 == 0 else -1)
                    rows.append(dict(side=side, arm=arm, yardstick=yardstick, group="all", season=season,
                                     error=1.0 + change + wobble))
    return pd.DataFrame(rows)


def test_nothing_is_built_when_no_arm_beats_today(screen):
    decision = screen.decide(_per_season(screen, {}))
    assert decision["chosen"] is None and decision["flags"] is None
    assert all(not verdict["passes"] for verdict in decision["verdicts"].values())


def test_the_longest_half_life_within_one_standard_error_of_the_best_is_chosen(screen):
    close = _per_season(screen, {("half_life_8", "D", "career"): -0.020, ("half_life_16", "D", "career"): -0.0199})
    assert screen.decide(close)["half_life"] == 16
    apart = _per_season(screen, {("half_life_8", "D", "career"): -0.020, ("half_life_16", "D", "career"): -0.015})
    decision = screen.decide(apart)
    assert decision["half_life"] == 8 and decision["flags"] == "--era-halflife 8"
    assert decision["chosen"] == "half_life_8" and not decision["with_vs_league"]


def test_half_life_two_counts_only_when_four_beats_eight_and_sixteen(screen):
    shut = _per_season(screen, {("half_life_2", "D", "career"): -0.030, ("half_life_8", "D", "career"): -0.020})
    decision = screen.decide(shut)
    assert not decision["half_life_2_eligible"] and decision["half_life"] == 8
    open_ = _per_season(screen, {("half_life_2", "D", "career"): -0.0251, ("half_life_4", "D", "career"): -0.025,
                                 ("half_life_8", "D", "career"): -0.020, ("half_life_16", "D", "career"): -0.010})
    decision = screen.decide(open_)
    assert decision["half_life_2_eligible"] and decision["half_life"] == 4


def test_each_rule_of_the_gate_can_fail_an_arm(screen):
    worse_offence = _per_season(screen, {("half_life_8", "D", "career"): -0.02, ("half_life_8", "O", "career"): 0.02})
    assert "(b)" in screen.decide(worse_offence)["verdicts"]["half_life_8"]["why"]
    neighbours = _per_season(screen, {("half_life_8", "D", "career"): -0.02,
                                      ("half_life_8", "D", "neighbours"): 0.0006})
    assert "(c)" in screen.decide(neighbours)["verdicts"]["half_life_8"]["why"]
    one_era = _per_season(screen, {("half_life_8", "D", "career"): lambda season: 0.01 if season >= 2017 else -0.05})
    verdict = screen.decide(one_era)["verdicts"]["half_life_8"]
    assert not verdict["passes"] and "(d)" in verdict["why"] and "2017-2026" in verdict["why"]


def test_the_ratios_against_the_league_join_only_when_they_add_to_the_half_life(screen):
    adds = _per_season(screen, {("half_life_8", "D", "career"): -0.02,
                                ("vs_league_half_life_8", "D", "career"): -0.04})
    decision = screen.decide(adds)
    assert decision["with_vs_league"] and decision["flags"] == "--era-halflife 8 --features ratios_vs_league"
    same = _per_season(screen, {("half_life_8", "D", "career"): -0.02,
                                ("vs_league_half_life_8", "D", "career"): -0.02})
    assert screen.decide(same)["flags"] == "--era-halflife 8"
    alone = _per_season(screen, {("vs_league", "O", "career"): -0.02})
    decision = screen.decide(alone)
    assert decision["chosen"] == "vs_league" and decision["flags"] == "--features ratios_vs_league"
