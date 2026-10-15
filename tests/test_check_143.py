"""scripts/checks/143_warm_start_screen.py (the warm-start screen): the extra trees follow the rated era on made-up
rows, the base is left as shipped, and the arm the gate chooses is the registered one; and the script starts."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "143_warm_start_screen.py"
# the shipped learning rate, so the base under-fits the era's price and the extra trees have something to follow
PARAMS = {"n_estimators": 60, "num_leaves": 7, "learning_rate": 0.037, "min_child_samples": 20, "linear_tree": True,
          "linear_lambda": 1.0}


@pytest.fixture(scope="module")
def warm():
    spec = importlib.util.spec_from_file_location("warm_start_screen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_lists_its_flag():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "--jobs" in result.stdout


def test_the_arms_are_the_registered_two_by_two(warm):
    assert warm.warm_arms() == {"warm_h4_e25": ("kernel_4", 25), "warm_h4_e75": ("kernel_4", 75),
                                "warm_h8_e25": ("kernel_8", 25), "warm_h8_e75": ("kernel_8", 75)}
    assert set(warm.GATE_ARMS) == set(warm.warm_arms())


def _two_eras(seed=0, n=3000):
    """Rows of two eras whose price of the first input has opposite signs: the era is the last column."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    era = rng.integers(0, 2, n)
    x[:, 5] = era
    y = x[:, 0] * np.where(era == 1, 1.0, -1.0) + 0.3 * x[:, 1] + rng.normal(scale=0.3, size=n)
    return x, y, era


def _slope(predict, era):
    grid = np.zeros((5, 6))
    grid[:, 0] = np.linspace(-1, 1, 5)
    grid[:, 5] = era
    return float(np.polyfit(grid[:, 0], predict(grid), 1)[0])


def test_extra_trees_move_the_price_toward_the_rows_the_kernel_favours(warm):
    from openrapm.box_prior import fit_lightgbm
    x, y, era = _two_eras()
    w = np.ones(len(y))
    base = fit_lightgbm(PARAMS, x, y, w)
    start = base.predict(x)
    for favoured, sign in ((1, 1.0), (0, -1.0)):
        kernel = np.where(era == favoured, 1.0, 0.1)
        extra = warm.fit_extra(PARAMS, x, y, w, kernel, start, 25)
        warm_slope = _slope(lambda g: base.predict(g) + extra.predict(g), favoured)
        base_slope = _slope(base.predict, favoured)
        assert sign * warm_slope > sign * base_slope + 0.2, (favoured, base_slope, warm_slope)
        assert abs(warm_slope - sign) < abs(base_slope - sign)


def test_a_kernel_of_ones_changes_little_and_the_base_is_never_touched(warm):
    from openrapm.box_prior import fit_lightgbm
    x, y, era = _two_eras(seed=1)
    w = np.ones(len(y))
    base = fit_lightgbm(PARAMS, x, y, w)
    before = base.predict(x).copy()
    extra = warm.fit_extra(PARAMS, x, y, w, np.ones(len(y)), base.predict(x), 25)
    assert np.array_equal(base.predict(x), before)
    assert np.mean((y - base.predict(x) - extra.predict(x)) ** 2) <= np.mean((y - before) ** 2) + 1e-9


def _per_season(changes: dict) -> pd.DataFrame:
    """Today's error 1.0 in every season; each arm's change from `changes` ({(arm, side, yardstick): a number}, 0
    elsewhere) plus a wobble of +-0.002 alternating by season, so every paired t is finite and no mean moves."""
    rows = []
    spec = importlib.util.spec_from_file_location("warm_start_screen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for number, arm in enumerate(["today", *module.GATE_ARMS]):
        for side in ("O", "D"):
            for yardstick in ("career", "neighbours"):
                for i, season in enumerate(module.SCREEN.SCREEN):
                    change = changes.get((arm, side, yardstick), 0.0)
                    wobble = 0.0 if arm == "today" else 0.002 * (1 if (i + number) % 2 == 0 else -1)
                    rows.append(dict(side=side, arm=arm, yardstick=yardstick, group="all", season=season,
                                     error=1.0 + change + wobble))
    return pd.DataFrame(rows)


def test_nothing_is_chosen_when_no_arm_beats_today(warm):
    decision = warm.decide(_per_season({}))
    assert decision["chosen"] is None and decision["best"] is None
    assert all(not verdict["passes"] for verdict in decision["verdicts"].values())


def test_the_fewest_extra_trees_within_one_standard_error_of_the_best_is_chosen(warm):
    big = {("warm_h4_e75", "D", "career"): -0.030, ("warm_h4_e25", "D", "career"): -0.0299}
    decision = warm.decide(_per_season(big))
    assert decision["best"] == "warm_h4_e75" and decision["chosen"] == "warm_h4_e25"
    assert (decision["half_life"], decision["extra_trees"]) == (4, 25)
    apart = {("warm_h4_e75", "D", "career"): -0.030, ("warm_h4_e25", "D", "career"): -0.020}
    decision = warm.decide(_per_season(apart))
    assert decision["chosen"] == "warm_h4_e75" and decision["extra_trees"] == 75
    mild = {("warm_h8_e25", "D", "career"): -0.030, ("warm_h4_e25", "D", "career"): -0.0299}
    assert warm.decide(_per_season(mild))["chosen"] == "warm_h8_e25", "of equal trees, the longer half-life"
