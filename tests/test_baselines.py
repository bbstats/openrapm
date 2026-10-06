"""Experiment 35's baselines: src/eracoef/vanilla.py (B2/B3 ridge) and src/eracoef/boxspm.py (B1 linear box SPM)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from eracoef import boxspm as bx
from eracoef import design
from eracoef import scorecard as sc
from eracoef import vanilla as va


def _stints(rng, m=30, rows=900, c=3):
    X = np.zeros((rows, 2 * m + c))
    for i in range(rows):
        X[i, rng.choice(m, 5, replace=False)] = 1.0
        X[i, m + rng.choice(m, 5, replace=False)] = 1.0
    X[:, 2 * m] = 1.0
    X[:, 2 * m + 1] = rng.choice([-1.0, 1.0], rows)
    X[:, 2 * m + 2] = 0.0                            # an unpenalized column no row touches (e.g. no playoff games)
    true = rng.normal(0, 3, 2 * m)
    y = X[:, :2 * m] @ true + 105 + rng.normal(0, 20, rows)
    w = rng.uniform(5, 15, rows)
    return sp.csr_matrix(X), y, w, m


def test_the_ridge_is_linear_in_the_prior_multipliers():
    rng = np.random.default_rng(0)
    X, y, w, m = _stints(rng)
    ids = np.arange(m) * 3 + 1
    ridge = va.Ridge(X, y, w, m, ids, np.asarray(X[:, :m].T @ w).ravel())
    p_o, p_d = rng.normal(0, 2, m), rng.normal(0, 2, m)
    comp = ridge.components(800.0, 1200.0, p_o, p_d)
    # direct solve at (m_o, m_d) = (0.6, 1.3), then centered the same way
    Xd = X.toarray()
    pen = np.r_[np.full(m, 800.0), np.full(m, 1200.0), np.zeros(3)]
    A = Xd.T @ (Xd * w[:, None]) + np.diag(pen)
    A[-1, -1] += 1.0
    b = Xd.T @ (w * y) + np.r_[0.6 * 800.0 * p_o, 1.3 * 1200.0 * p_d, np.zeros(3)]
    beta = np.linalg.solve(A, b)
    o, d = beta[:m], beta[m:2 * m]
    poss = ridge.poss
    o, d = o - np.average(o, weights=poss), d - np.average(d, weights=poss)
    t = comp.table(0.6, 1.3)
    assert np.allclose(t.o, o, atol=1e-8) and np.allclose(t.d, d, atol=1e-8)


def test_the_stored_quadratic_gives_the_scored_error_at_any_multiplier():
    rng = np.random.default_rng(1)
    X, y, w, m = _stints(rng)
    ids = np.arange(m) * 3 + 1
    ridge = va.Ridge(X, y, w, m, ids, np.asarray(X[:, :m].T @ w).ravel())
    comp = ridge.components(500.0, 500.0, rng.normal(0, 2, m), rng.normal(0, 2, m))
    # a held-out block with two unrated players
    n = m + 2
    rows = 200
    Z = np.zeros((rows, 2 * n))
    for i in range(rows):
        Z[i, rng.choice(n, 5, replace=False)] = 1.0
        Z[i, n + rng.choice(n, 5, replace=False)] = 1.0
    home = rng.choice([-1.0, 1.0], rows)
    block = sc.Block(key="x", season=2001, deal=0, Z=sp.csr_matrix(Z), y=rng.normal(108, 12, rows),
                     w=rng.uniform(90, 105, rows), F=np.column_stack([home, np.ones(rows)]), home=home,
                     player_ids=np.r_[ids, [9001, 9002]])
    q = pd.DataFrame([va.quadratic(block, comp, fill=True)])
    for m_o, m_d in ((0.0, 0.0), (0.7, 1.1), (1.0, 0.4)):
        table = comp.table(m_o, m_d)
        fill = va.stand_in(table[["o", "d"]].to_numpy(), table.poss.to_numpy())
        direct = sc.error(block, sc.parts(block, table, tuple(fill)).total)
        assert abs(va.error_at(q, m_o, m_d).iloc[0] - direct) < 1e-8


def test_the_ridge_keeps_only_the_named_fixed_effects():
    """A cached design carries the game index as a last column; left in, it was an unpenalized time trend (exp 37)."""
    rng = np.random.default_rng(4)
    X, y, w, m = _stints(rng)
    ids = np.arange(m) * 3 + 1
    poss = np.asarray(X[:, :m].T @ w).ravel()
    game_index = sp.csr_matrix(np.arange(X.shape[0], dtype=float)[:, None])
    with_index = sp.hstack([X, game_index], format="csr")
    a = va.Ridge(with_index, y, w, m, ids, poss, n_fixed=3)
    b = va.Ridge(X, y, w, m, ids, poss)
    assert a.c == 3 == b.c
    p = rng.normal(0, 1, m)
    ta, tb = a.components(900.0, 900.0, p, p).table(1.0, 1.0), b.components(900.0, 900.0, p, p).table(1.0, 1.0)
    assert np.allclose(ta.o, tb.o) and np.allclose(ta.d, tb.d)


def test_the_stand_in_is_the_replacement_fill():
    from eracoef import tradeset as ts
    rng = np.random.default_rng(2)
    t = pd.DataFrame({"o": rng.normal(-1, 1, 50), "d": rng.normal(1, 1, 50), "poss": rng.uniform(1, 3000, 50)})
    assert np.allclose(va.stand_in(t[["o", "d"]].to_numpy(), t.poss.to_numpy()), ts.replacement_fill(t))


def test_box_counts_are_the_design_features():
    assert bx.COUNTS == list(design.FEATURES)


def test_box_possessions_are_one_number_per_game_and_split_by_minutes():
    box = pd.DataFrame({"game_id": ["g"] * 4, "team_id": [1, 1, 2, 2], "player_id": [10, 11, 20, 21],
                        "minutes": [120.0, 120.0, 150.0, 90.0], **{c: [0.0] * 4 for c in bx.COUNTS}})
    box.loc[0, "fg2m"], box.loc[2, "fg2_miss"], box.loc[3, "tov"] = 50.0, 40.0, 10.0
    b = bx.with_possessions(box)
    assert np.isclose(b.game_poss.iloc[0], 50.0)                   # (50 + 50) / 2
    assert np.isclose(b.poss.sum(), 2 * 5 * 50.0)                  # five players' worth of possessions per team
    assert np.isclose(b.set_index("player_id").poss[20], 150.0 / 48.0 * 50.0)
