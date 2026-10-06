"""src/eracoef/heldoutprior.py: the held-out error is an exact quadratic in the prior's input weights."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from eracoef import heldoutprior as hp
from eracoef import scorecard as sc
from eracoef import vanilla as va


def _setup(seed=0, m=30, rows=900):
    rng = np.random.default_rng(seed)
    X = np.zeros((rows, 2 * m + 3))
    for i in range(rows):
        X[i, rng.choice(m, 5, replace=False)] = 1.0
        X[i, m + rng.choice(m, 5, replace=False)] = 1.0
    X[:, 2 * m] = 1.0
    X[:, 2 * m + 1] = rng.choice([-1.0, 1.0], rows)
    true = rng.normal(0, 3, 2 * m)
    y = X[:, :2 * m] @ true + 105 + rng.normal(0, 20, rows)
    w = rng.uniform(5, 15, rows)
    ids = np.arange(m) * 3 + 1
    poss = np.asarray(sp.csr_matrix(X)[:, :m].T @ w).ravel()
    poss[-1] = 0.0                                     # one player with no rating-game possessions: unrated
    ridge = va.Ridge(sp.csr_matrix(X), y, w, m, ids, poss)
    # a held-out block with two players the ratings never saw, and team ids
    n, nr = m + 2, 240
    Z = np.zeros((nr, 2 * n))
    for i in range(nr):
        Z[i, rng.choice(n, 5, replace=False)] = 1.0
        Z[i, n + rng.choice(n, 5, replace=False)] = 1.0
    home = rng.choice([-1.0, 1.0], nr)
    block = sc.Block(key="x", season=2001, deal=0, Z=sp.csr_matrix(Z), y=rng.normal(108, 12, nr),
                     w=rng.uniform(90, 105, nr), F=np.column_stack([home, np.ones(nr)]), home=home,
                     player_ids=np.r_[ids, [9001, 9002]])
    teams = (rng.integers(0, 6, nr), rng.integers(0, 6, nr))
    return rng, ridge, block, teams, m


def _direct_error(ridge, block, p_o, p_d, lam_o, lam_d, level="home", teams=None):
    """The prior p handed to the B3 ridge at multiplier 1, scored by the scorecard (or with team levels profiled)."""
    table = ridge.components(lam_o, lam_d, p_o, p_d).table(1.0, 1.0)
    fill = tuple(va.stand_in(table[["o", "d"]].to_numpy(), table.poss.to_numpy()))
    pred = sc.parts(block, table, fill).total
    if level == "home":
        return sc.error(block, pred)
    L = hp.level_matrix(block, "team", teams)
    r = sc._wls_residual(L, block.y - pred, block.w)
    return float(np.average(r ** 2, weights=block.w))


@pytest.mark.parametrize("level", ["home", "team"])
def test_the_quadratic_in_the_weights_is_the_direct_error(level):
    rng, ridge, block, teams, m = _setup()
    F_o, F_d = rng.normal(0, 1, (m, 3)), rng.normal(0, 1, (m, 2))
    ids, w, V_o, V_d = hp.responses(ridge, 700.0, 1100.0, F_o, F_d)
    C = hp.contributions(block, ids, w, V_o, V_d)
    G = hp.gram(block, C, level, teams)
    for _ in range(4):
        beta = rng.normal(0, 1, 5)
        direct = _direct_error(ridge, block, F_o @ beta[:3], F_d @ beta[3:], 700.0, 1100.0, level, teams)
        assert abs(hp.error_at(G, block.w.sum(), beta) - direct) < 1e-8


def test_one_input_per_side_is_b3s_quadratic():
    rng, ridge, block, teams, m = _setup(1)
    p_o, p_d = rng.normal(0, 2, m), rng.normal(0, 2, m)
    ids, w, V_o, V_d = hp.responses(ridge, 900.0, 900.0, p_o[:, None], p_d[:, None])
    G = hp.gram(block, hp.contributions(block, ids, w, V_o, V_d))
    q = va.quadratic(block, ridge.components(900.0, 900.0, p_o, p_d))
    assert np.allclose([G[0, 0], G[0, 1], G[0, 2], G[1, 1], G[1, 2], G[2, 2]],
                       [q["g00"], q["g0o"], q["g0d"], q["goo"], q["god"], q["gdd"]], rtol=1e-9, atol=1e-6)


def test_a_constant_input_moves_nothing():
    rng, ridge, block, teams, m = _setup(2)
    F_o = np.column_stack([np.full(m, 3.0), rng.normal(0, 1, m)])
    ids, w, V_o, V_d = hp.responses(ridge, 800.0, 800.0, F_o, np.zeros((m, 0)))
    assert np.abs(V_o[:, 1]).max() < 1e-8 and np.abs(V_d[:, 1]).max() < 1e-8


def test_team_mean_ratings_score_the_no_ratings_error_at_team_level():
    rng, ridge, block, teams, m = _setup(3)
    n = block.n_players
    t_off, t_def = teams
    # a rating that is the same for every player of a row's team is absorbed by the team levels
    G0 = hp.gram(block, np.zeros((len(block.y), 1)), "team", teams)
    const = np.asarray(5.0 * pd.Series(t_off).map({k: rng.normal() for k in range(6)}).to_numpy())[:, None]
    G1 = hp.gram(block, const, "team", teams)
    assert abs(G0[0, 0] - G1[0, 0]) < 1e-8


def test_drifting_weights_need_no_new_solve():
    rng, ridge, block, teams, m = _setup(4)
    F_o, F_d = rng.normal(0, 1, (m, 2)), rng.normal(0, 1, (m, 1))
    ids, w, V_o, V_d = hp.responses(ridge, 600.0, 600.0, F_o, F_d)
    G = hp.gram(block, hp.contributions(block, ids, w, V_o, V_d))
    beta, gamma, t = rng.normal(0, 1, 3), rng.normal(0, 1, 3), 0.7
    assert abs(hp.error_at(hp.drift(G, t), block.w.sum(), np.r_[beta, gamma])
               - hp.error_at(G, block.w.sum(), beta + gamma * t)) < 1e-8


def test_the_anchored_fit_with_heavy_corrections_penalty_is_the_anchor_fit():
    rng, ridge, block, teams, m = _setup(5)
    F_o, F_d = rng.normal(0, 1, (m, 3)), rng.normal(0, 1, (m, 3))
    ids, w, V_o, V_d = hp.responses(ridge, 900.0, 900.0, F_o, F_d)
    G = hp.gram(block, hp.contributions(block, ids, w, V_o, V_d))
    sw = block.w.sum()
    pen = np.array([0.0, 1e12, 1e12, 0.0, 1e12, 1e12])      # inputs 0 and 3 are the anchors
    beta = hp.fit(G, sw, pen)
    anchor = hp.fit(hp.subset(G, [0, 3]), sw, [0.0, 0.0])
    assert np.allclose(beta[[0, 3]], anchor, atol=1e-6) and np.abs(beta[[1, 2, 4, 5]]).max() < 1e-6


def test_banned_inputs_raise():
    with pytest.raises(ValueError):
        hp.GROUPS["oops"] = ["test_poss_off"]
        try:
            hp.columns_of(["oops"])
        finally:
            del hp.GROUPS["oops"]
    assert "b1" in hp.columns_of(["b1"])


def test_the_traded_split_collapses_to_the_plain_gram_and_scores_a_hybrid_exactly():
    rng, ridge, block, teams, m = _setup(4)
    lam_o, lam_d = 700.0, 1100.0
    F_o, F_d = rng.normal(0, 1, (m, 2)), rng.normal(0, 1, (m, 2))
    ids, w, V_o, V_d = hp.responses(ridge, lam_o, lam_d, F_o, F_d)
    C = hp.contributions(block, ids, w, V_o, V_d)
    # every third player "traded": his team in the rating games is one he never plays for here; the two players the
    # ratings never saw have no team
    nr = len(block.y)
    team_of = pd.Series(np.where(np.arange(m) % 3 == 0, 99.0, 0.0), index=block.player_ids[:m])
    Zt = sc.traded_entries(block, team_of, np.zeros(nr), np.zeros(nr))
    stay, traded = sc.split_traded(block, Zt)
    Gs = hp.split_gram(block, C, hp.contributions(traded, ids, w, V_o, V_d))
    K = 4
    assert np.allclose(hp.collapse_split(Gs, K), hp.gram(block, C), rtol=1e-10, atol=1e-6)

    def table(beta):
        t = ridge.components(lam_o, lam_d, F_o @ beta[:2], F_d @ beta[2:]).table(1.0, 1.0)
        return t, tuple(va.stand_in(t[["o", "d"]].to_numpy(), t.poss.to_numpy()))

    a, b = rng.normal(0, 1, K), rng.normal(0, 1, K)
    (ta, fa), (tb, fb) = table(a), table(b)
    pa_s, pa_t = sc.parts(stay, ta, fa).total, sc.parts(traded, ta, fa).total
    pb_s, pb_t = sc.parts(stay, tb, fb).total, sc.parts(traded, tb, fb).total
    sw = block.w.sum()
    # the traded players at weights b, everyone else at a
    assert abs(hp.error_at(Gs, sw, np.r_[a, b]) - sc.error(block, pa_s + pb_t)) < 1e-8
    Q = hp.increment_gram(hp.split_subset(Gs, np.arange(K), K), a, b)

    def err(x):
        x = np.asarray(x, dtype=float)
        return (Q[0, 0] - 2 * x @ Q[1:, 0] + x @ Q[1:, 1:] @ x) / sw

    assert abs(err([0, 0]) - sc.error(block, pa_s + pa_t)) < 1e-8
    assert abs(err([1, 1]) - sc.error(block, pb_s + pb_t)) < 1e-8
    assert abs(err([0, 1]) - sc.error(block, pa_s + pb_t)) < 1e-8
    assert abs(err([1, 0]) - sc.error(block, pb_s + pa_t)) < 1e-8
    # the blend weights are the regression of a's miss on b's change, split by stayed / traded
    direct = sc.solve([sc.normal(block, {"stayed": pb_s - pa_s, "traded": pb_t - pa_t}, offset=pa_s + pa_t,
                                 level="home")])
    assert np.allclose(np.linalg.solve(Q[1:, 1:], Q[1:, 0]), direct.to_numpy(), rtol=1e-8, atol=1e-10)
