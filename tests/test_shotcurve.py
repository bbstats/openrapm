"""The per-season shot-distance curve (src/openrapm/shotcurve.py) on shots simulated from a known one."""
import numpy as np
import pandas as pd
import pytest

from openrapm.shotcurve import (MAX_DIST, UNLOCATED, ShotCurve, calibration, fit_curve, logistic_irls, natural_spline,
                               shot_bin, shot_rows)


def _true_p(dist, three):
    if three:
        return 0.40 - 0.01 * (dist - 23)                       # 40% at the arc, drifting down
    return 0.66 * np.exp(-0.06 * dist) + 0.38 * (1 - np.exp(-0.06 * dist))   # rim 66% down to ~40%


def _simulate(n=60000, seed=5):
    rng = np.random.default_rng(seed)
    three = rng.random(n) < 0.35
    dist = np.where(three, rng.uniform(22, 30, n), rng.exponential(8, n).clip(0, 24)).round()
    p = np.array([_true_p(d, t) for d, t in zip(dist, three)])
    made = rng.random(n) < p
    # an unlocated share: 15% of threes at distance 0, and some twos at 0 with no rim word
    unloc = (three & (rng.random(n) < 0.15)) | (~three & (dist == 0) & (rng.random(n) < 0.3))
    desc = np.where(three, "X 25' 3PT Jump Shot", np.where(dist == 0, "X Layup", "X Jump Shot"))
    desc = np.where(unloc & ~three, "X Jump Shot", desc)
    dist = np.where(unloc, 0.0, dist)
    return pd.DataFrame({"actionType": np.where(made, "Made Shot", "Missed Shot"), "shotValue": np.where(three, 3, 2),
                         "shotDistance": dist, "description": desc}), p


def test_shot_bin_rules():
    assert shot_bin(-1.0, False) == UNLOCATED
    assert shot_bin(0.0, True) == UNLOCATED and shot_bin(15.0, True) == UNLOCATED
    assert shot_bin(0.0, False, "Hone Dunk (2 PTS)") == 0
    assert shot_bin(0.0, False, "Hone Jump Shot") == UNLOCATED
    assert shot_bin(25.4, True) == 25 and shot_bin(80.0, True) == MAX_DIST


def test_curve_recovers_the_truth():
    pbp, _ = _simulate()
    rows = shot_rows(pbp)
    assert set(rows.columns) == {"three", "made", "bin"}
    curve = fit_curve(rows, 2024)
    for d in (1, 5, 10, 15, 20):
        assert abs(curve.prob(d, False, "X Jump Shot" if d else "X Layup") - _true_p(d, False)) < 0.03, d
    for d in (23, 25, 28):
        assert abs(curve.prob(d, True) - _true_p(d, True)) < 0.03, d
    cal = calibration(curve, rows)
    assert cal.gap.abs().max() < 0.04
    # the unlocated cells carry their own rate, near the league rate of that shot value
    assert abs(curve.prob(0.0, True) - rows[rows.three].made.mean()) < 0.05
    assert curve.table.three.nunique() == 2 and (curve.table.bin == UNLOCATED).sum() == 2


def test_constant_curve_and_roundtrip():
    c = ShotCurve.constant(p2=0.5, p3=0.35)
    assert c.prob(7.0, False) == 0.5 and c.prob(24.0, True) == 0.35 and c.prob(-1.0, False) == 0.5
    again = ShotCurve(c.season, c.table.copy())
    assert again.prob(7.0, False) == 0.5


def test_the_ridge_takes_one_value_or_one_per_column():
    """`logistic_irls`, moved here from the retired shot-quality model with its test."""
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(4000), rng.standard_normal((4000, 5))])
    y = (rng.random(4000) < 1 / (1 + np.exp(-(X @ np.array([0.2, 1.0, -0.5, 0.3, 0.8, -1.0]))))).astype(float)
    assert np.array_equal(logistic_irls(X, y, ridge=2.0), logistic_irls(X, y, ridge=np.full(6, 2.0)))
    r = np.full(6, 1e-6)
    r[4] = 1e9                                          # one column held at zero, the rest all but free
    b = logistic_irls(X, y, ridge=r)
    free = logistic_irls(np.delete(X, 4, axis=1), y, ridge=1e-6)
    assert abs(b[4]) < 1e-5 and np.allclose(np.delete(b, 4), free, atol=1e-4)
    with pytest.raises(ValueError):
        logistic_irls(X, y, ridge=np.ones(5))


def test_the_natural_spline_is_linear_beyond_its_last_knot():
    """`natural_spline`: [1, x, K - 2 cubic terms], and natural means straight outside the boundary knots."""
    knots = (1.0, 3.0, 5.0, 8.0, 12.0)
    x = np.linspace(-5.0, 30.0, 141)
    B = natural_spline(x, knots)
    assert B.shape == (len(x), len(knots))
    assert np.allclose(B[:, 0], 1.0) and np.allclose(B[:, 1], x)
    out = x >= knots[-1]
    assert np.allclose(np.diff(B[out], n=2, axis=0), 0.0, atol=1e-6)     # zero second difference: linear
    assert np.allclose(B[x <= knots[0], 2:], 0.0)                         # and flat zero below the first knot
