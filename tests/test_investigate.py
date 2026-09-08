"""investigate.py: a planted per-player miss is recovered by the residual ridge, blamed on teammates by the
on-court mean, and found in its lineup."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from eracoef.investigate import lineups, on_court, pooled, residual_ridge, season_table


def _world(seed=0, m=40, n=6000):
    rng = np.random.default_rng(seed)
    rows_o = np.array([rng.choice(m, 5, replace=False) for _ in range(n)])
    rows_d = np.array([rng.choice(m, 5, replace=False) for _ in range(n)])
    Zo = sp.csr_matrix((np.ones(5 * n), (np.repeat(np.arange(n), 5), rows_o.ravel())), shape=(n, m))
    Zd = sp.csr_matrix((np.ones(5 * n), (np.repeat(np.arange(n), 5), rows_d.ravel())), shape=(n, m))
    miss_o, miss_d = np.zeros(m), np.zeros(m)
    miss_o[3], miss_d[7] = 4.0, 3.0            # player 3 is under-rated on offense, player 7 on defense
    w = rng.uniform(5, 40, n)
    r = Zo @ miss_o - Zd @ miss_d + rng.normal(0, 12, n) / np.sqrt(w / 20)
    return Zo, Zd, r, w, miss_o, miss_d


def test_the_ridge_recovers_a_planted_miss_and_its_sign():
    Zo, Zd, r, w, mo, md = _world()
    a, b, se_a, se_b = residual_ridge(Zo, Zd, r, w, lam=200.0)
    assert np.argmax(a) == 3 and 2.0 < a[3] < 5.0 and abs(a[3]) / se_a[3] > 3
    assert np.argmin(b) == 7 and -4.5 < b[7] < -1.5          # in the offense's sign; the board's flips it
    assert np.abs(np.delete(a, 3)).max() < 1.5 and np.abs(np.delete(b, 7)).max() < 1.5


def test_on_court_blames_the_teammates_too_and_the_season_table_flips_defense():
    Zo, Zd, r, w, mo, md = _world()
    mean_o, poss_o, mean_d, poss_d = on_court(Zo, Zd, r, w)
    assert np.argmax(mean_o) == 3 and mean_o[3] > 2.0 and abs(poss_o.sum() - 5 * w.sum()) < 1e-6 * w.sum()
    others = np.delete(mean_o, 3)
    assert others.mean() > 0.05                               # everyone who shared a floor with 3 gets some credit
    t = season_table(2001, np.arange(40) + 100, Zo, Zd, r, w, lam=200.0)
    assert t.miss_d.idxmax() == 7 and t.miss_d[7] > 1.5 and t.oncourt_d[7] > 0
    assert list(t.player_id[:3]) == [100, 101, 102]


def test_lineups_find_the_unit_and_pooled_is_inverse_variance_weighted():
    Zo, Zd, r, w, mo, md = _world()
    ids = np.arange(40) + 100
    u = lineups(Zo, Zd, ids, r, w, min_poss=1.0)
    with3 = u[(u.side == "O") & u.key.map(lambda k: 103 in k)]
    without = u[(u.side == "O") & ~u.key.map(lambda k: 103 in k)]
    assert with3.miss.mean() > without.miss.mean() + 2.0
    with7 = u[(u.side == "D") & u.key.map(lambda k: 107 in k)]
    assert with7.miss.mean() > 1.5                             # flipped: positive = the unit defends better than rated
    P = pd.concat([season_table(h, ids, Zo, Zd, r, w, lam=200.0) for h in (2001, 2002)], ignore_index=True)
    Q = pooled(P, min_poss=1.0)
    q3 = Q[(Q.player_id == 103) & (Q.side == "O")].iloc[0]
    assert q3.seasons == 2 and q3.z > 4 and abs(q3.se - P[P.player_id == 103].se_o.iloc[0] / np.sqrt(2)) < 1e-9


def test_attributable_finds_a_planted_miss_and_nothing_in_pure_noise():
    from eracoef.investigate import attributable
    Zo, Zd, r, w, mo, md = _world()
    a = attributable(Zo, Zd, r, w, lam=200.0)
    assert a["total"] > a["player"] > 0 and a["ss_o"] > 0 and a["ss_d"] > 0
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 12, len(w)) / np.sqrt(w / 20)
    b = attributable(Zo, Zd, noise, w, lam=200.0)
    assert b["player"] < 0.5 * a["player"]                 # the planted miss is most of what the ridge finds
