"""The four-factor defence (fastfit.factor_defense, HANDOFF 3.2): four factor fits on the points fit's own layout,
the points prior shared out across them, the four recombined into points allowed.

The decisive check is an identity.  If the eFG% factor is built to be EXACTLY half the points response on the
same rows with the same weights, and the other three factors carry nothing, then the factor fit is the points
fit rescaled, its prior share must come out at one half, and the recombined defensive rating must equal the
points fit's to machine precision.  Anything in the plumbing -- the offset, the shares, the gradients, the side
halves of u -- would break it.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef.design import FEATURES, build_design
from eracoef.estimator import MixedModelRAPM, Moments, _layout
from eracoef.fastfit import FACTORS, factor_defense, factor_rows, points_per_factor
from eracoef.simulate import simulate

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500, "features": FEATURES, "pad_target": "league"}
LAM, RATIO = 300.0, 0.7


@pytest.fixture(scope="module")
def wd():
    sim = simulate(n_seasons=2, n_teams=6, players_per_team=8, games_per_season=40, stints_per_game=(15, 25), seed=5)
    w = build_design(sim["stints"], sim["box"], FEATURES, CFG)
    poss = w.rows["poss"].to_numpy(dtype=float)
    pts = w.y * poss / 100.0
    # the counters the four factors read, fabricated so that eFG% = y / 2 on every row with weight = poss
    # (fga = poss, fgm = pts / 2, no threes) and the other three rates are identically zero
    w.counters = pd.DataFrame({"poss": poss, "fga": poss, "fgm": pts / 2.0, "fg3m": 0.0, "tov": 0.0,
                               "reb_chance": 1.0, "reb_cont": 0.0, "att": 1.0, "fta": 0.0})
    return w


def _points_fit(wd, off):
    layout = _layout(wd.X, wd.spec, None, None, prior_offset=off)
    mm = MixedModelRAPM(lam=LAM, lam_ratio=RATIO, spec=wd.spec)
    mom = Moments(layout, wd.y, wd.w, mm._season_cols(), mm._scale())
    mom.want_edf = False
    return layout, np.asarray(mom.solve_chol(LAM).u, dtype=float)


def test_factor_rows_weights_by_the_denominator_and_drops_rows_without_one(wd):
    y, w = factor_rows(wd, "efg")
    assert np.allclose(y, wd.y / 2.0) and np.allclose(w, wd.rows["poss"].to_numpy())
    c = wd.counters.copy()
    c.loc[c.index[:3], "fga"] = 0.0
    wd2 = type(wd)(wd.X_src, wd.y, wd.w, wd.groups, wd.spec, wd.game_box, wd.game_poss, wd.rows, wd.games, c)
    y2, w2 = factor_rows(wd2, "efg")
    assert (w2[:3] == 0).all() and (y2[:3] == 0).all() and np.isfinite(y2).all()
    assert np.allclose(y2[3:], y[3:]) and np.allclose(w2[3:], w[3:])


def test_points_per_factor_recovers_a_planted_linear_law():
    rng = np.random.default_rng(0)
    n = 4000
    season = rng.choice([2001, 2002, 2003], n)
    level = {2001: 105.0, 2002: 108.0, 2003: 111.0}
    ys = {f: rng.normal(50, 5, n) for f in FACTORS}
    ws = {f: rng.uniform(1, 30, n) for f in FACTORS}
    truth = {"efg": 1.5, "tov": -1.0, "oreb": 0.6, "ftr": 0.3}
    y = np.array([level[s] for s in season]) + sum(truth[f] * ys[f] for f in FACTORS)
    ws["oreb"][:200] = 0.0                       # rows with no rebound chance: must be ignored, whatever y says there
    y[:200] = -1e6
    g, r2 = points_per_factor(y, ys, ws, rng.uniform(1, 50, n), season)
    for f in FACTORS:
        assert abs(g[f] - truth[f]) < 1e-8
    assert abs(r2 - 1.0) < 1e-9


def test_the_identity_the_efg_factor_reproduces_the_points_fit(wd):
    rng = np.random.default_rng(1)
    m = wd.spec.n_ps
    off = rng.normal(0, 1.0, 2 * m)                  # a per-player prior, both sides, in points per 100
    layout, u = _points_fit(wd, off)
    fd = factor_defense(wd, layout, off, wd.y, wd.y, wd.y, wd.w, LAM, RATIO, None,
                        factor_lams={"efg": (LAM, RATIO)})
    d = fd["diag"]
    assert abs(d["g"]["efg"] - 2.0) < 1e-9 and abs(d["r2"] - 1.0) < 1e-9
    assert abs(d["h_d"]["efg"] - 0.5) < 1e-9 and abs(d["h_o"]["efg"] - 0.5) < 1e-9
    for f in ("tov", "oreb", "ftr"):
        assert abs(d["h_d"][f]) < 1e-12 and abs(d["sd_u"][f]) < 1e-12
    expect = off[m:] + u[m:]
    assert np.allclose(fd["d"], expect, atol=1e-8)
    assert np.allclose(fd["u_d"], u[m:], atol=1e-8)
    assert np.allclose(fd["u_d_pts0"], _points_fit(wd, np.zeros(2 * m))[1][m:], atol=1e-8)


def test_a_wrong_share_would_not_reproduce_it(wd):
    """The same identity with the prior withheld from the factor fit: the shares are what carry it."""
    rng = np.random.default_rng(2)
    m = wd.spec.n_ps
    off = rng.normal(0, 1.0, 2 * m)
    layout, u = _points_fit(wd, off)
    fd = factor_defense(wd, layout, np.zeros(2 * m), wd.y, wd.y, wd.y, wd.w, LAM, RATIO, None,
                        factor_lams={"efg": (LAM, RATIO)})
    assert not np.allclose(fd["d"], off[m:] + u[m:], atol=1e-3)


def test_reml_reselects_each_ridge_inside_the_grid(wd):
    rng = np.random.default_rng(3)
    m = wd.spec.n_ps
    off = rng.normal(0, 1.0, 2 * m)
    layout, _ = _points_fit(wd, off)
    fd = factor_defense(wd, layout, off, wd.y, wd.y, wd.y, wd.w, LAM, RATIO, None,
                        factor_lams={"efg": (LAM, RATIO)}, reml=True)
    d = fd["diag"]
    assert set(d["lam"]) == set(FACTORS) and all(np.isfinite(v) and v > 0 for v in d["lam"].values())
    assert set(d["at_edge"]) == set(FACTORS)
    assert np.isfinite(fd["d"]).all()
    # the eFG% factor carries the whole signal here, so its REML choice is a real one: a value on the grid
    # around the a-priori ridge, flagged if it sits on an end
    assert LAM / 16.0 <= d["lam"]["efg"] <= 64.0 * LAM and isinstance(d["at_edge"]["efg"], bool)
