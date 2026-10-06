"""src/eracoef/portable.py: the split columns add back to each part's contribution, and the constrained fit recovers
planted tier slopes and traded-to-stayed ratios."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from eracoef import portable as pt
from eracoef import scorecard as sc


def _league(seed=0, n_teams=10, roster=12, rows=6000, noise=2.0, ratio=None, traded_level=(-1.0, 0.5)):
    rng = np.random.default_rng(seed)
    n = n_teams * roster
    ids = np.arange(1, n + 1) * 11
    team = np.repeat(np.arange(n_teams), roster)
    parts = pd.DataFrame({"player_id": ids, "o_prior": rng.normal(0, 1.5, n), "o_beyond": rng.normal(0, 1.0, n),
                          "d_prior": rng.normal(0, 1.0, n), "d_beyond": rng.normal(0, 0.8, n),
                          "tier": rng.integers(0, 3, n), "poss": rng.uniform(100, 3000, n)})
    traded = np.zeros(n, dtype=bool)
    traded[::3] = True                                  # their team in the rating games is one they never play for
    rating_team = pd.Series(np.where(traded, 99, team).astype(float), index=ids)
    beta = {"prior": np.array([0.8, 1.0, 1.2]), "beyond": np.array([1.5, 1.2, 1.0])}
    ratio = ratio or {("O", "prior"): 1.1, ("O", "beyond"): 0.5, ("D", "prior"): 0.9, ("D", "beyond"): 0.3}
    true = {}
    for s in ("O", "D"):
        v = np.zeros(n)
        for c in ("prior", "beyond"):
            b = beta[c][parts.tier.to_numpy()]
            v += b * np.where(traded, ratio[(s, c)], 1.0) * parts[f"{s.lower()}_{c}"].to_numpy()
        v += np.where(traded, traded_level[0 if s == "O" else 1], 0.0)
        true[s] = v
    Z = np.zeros((rows, 2 * n))
    t_off, t_def = rng.integers(0, n_teams, rows), rng.integers(0, n_teams, rows)
    for i in range(rows):
        Z[i, rng.choice(np.flatnonzero(team == t_off[i]), 5, replace=False)] = 1.0
        Z[i, n + rng.choice(np.flatnonzero(team == t_def[i]), 5, replace=False)] = 1.0
    home = rng.choice([-1.0, 1.0], rows)
    w = rng.uniform(90, 105, rows)
    y = 110 + 1.5 * home + Z[:, :n] @ true["O"] + Z[:, n:] @ true["D"] + noise * rng.normal(0, 1, rows) * 10 / np.sqrt(w)
    block = sc.Block(key="x", season=2001, deal=0, Z=sp.csr_matrix(Z), y=y, w=w,
                     F=np.column_stack([home, np.ones(rows)]), home=home, player_ids=ids)
    return block, parts, rating_team, t_off, t_def, ratio


def test_the_split_columns_add_back_to_each_parts_contribution():
    block, parts, rating_team, t_off, t_def, _ = _league(rows=400)
    Zt = sc.traded_entries(block, rating_team, t_off, t_def)
    cols = pt.columns(block, Zt, parts)
    n = block.n_players
    for s, lo in (("O", 0), ("D", n)):
        for c in pt.PARTS:
            got = sum(cols[f"{s} {c} t{k} {g}"] for k in pt.TIERS for g in ("stayed", "traded"))
            want = block.Z[:, lo:lo + n] @ parts[f"{s.lower()}_{c}"].to_numpy()
            assert np.allclose(got, want, atol=1e-10)


def test_the_fit_recovers_planted_ratios_and_the_traded_level():
    block, parts, rating_team, t_off, t_def, ratio = _league()
    Zt = sc.traded_entries(block, rating_team, t_off, t_def)
    eq = sc.normal(block, pt.columns(block, Zt, parts), level="full")
    f = pt.fit(eq.xx, eq.xy)
    for k, v in ratio.items():
        assert abs(f["keys"][k] - v) < 0.06, (k, f["keys"][k], v)
        assert all(f["ratio"][f"{k[0]} {k[1]} t{t}"] == f["keys"][k] for t in pt.TIERS)
    assert abs(f["levels"]["O traded level"] - (-1.0)) < 0.15
    assert abs(f["levels"]["D traded level"] - 0.5) < 0.15
    # the free fit agrees on the stayed slopes
    free = pt.free_fit(eq.xx, eq.xy)
    assert abs(free["O prior t1 stayed"] - 1.0) < 0.06


def test_a_ratio_of_one_is_the_rating_itself():
    _, parts, _, _, _, _ = _league(rows=10)
    ones = {nm: 1.0 for nm in pt.slope_names()}
    p = pt.portable(parts, ones)
    assert np.allclose(p.o, parts.o_prior + parts.o_beyond) and np.allclose(p.d, parts.d_prior + parts.d_beyond)


def test_held_and_tied_ratios():
    block, parts, rating_team, t_off, t_def, _ = _league(seed=2, ratio={("O", "prior"): 1.0, ("O", "beyond"): 0.4,
                                                                      ("D", "prior"): 1.0, ("D", "beyond"): 0.4})
    Zt = sc.traded_entries(block, rating_team, t_off, t_def)
    eq = sc.normal(block, pt.columns(block, Zt, parts), level="full")
    one = pt.fit(eq.xx, eq.xy, "beyond-one")
    assert set(one["keys"]) == {("both", "beyond")} and abs(one["keys"][("both", "beyond")] - 0.4) < 0.06
    assert one["ratio"]["O prior t0"] == 1.0
    tier = pt.fit(eq.xx, eq.xy, "box+beyond-tier")
    assert len(tier["keys"]) == 12
    # the four-ratio fit nests the held one: never a larger residual sum of squares
    assert pt.fit(eq.xx, eq.xy, "box+beyond")["ssr"] <= one["ssr"] + 1e-6 * abs(one["ssr"])


def test_the_portable_table_keeps_the_box_part_and_each_seasons_mean():
    rng = np.random.default_rng(4)
    n = 40
    t = pd.DataFrame({"player_id": np.arange(n), "season": np.repeat([2001, 2002], n // 2),
                      "prior_off": rng.normal(0, 1, n), "u_off": rng.normal(0, 0.5, n), "c_off": rng.normal(0, 0.3, n),
                      "prior_def": rng.normal(0, 1, n), "u_def": rng.normal(0, 0.5, n), "c_def": rng.normal(0, 0.3, n),
                      "poss_season": rng.uniform(100, 3000, n)})
    for s in ("off", "def"):
        t[f"rating_{s}"] = t[f"prior_{s}"] + t[f"u_{s}"] + t[f"c_{s}"]
    one = pt.portable_table(t, 1.0)
    assert np.allclose(one.portable_off, t.rating_off) and np.allclose(one.portable_def, t.rating_def)
    half = pt.portable_table(t, 0.5)
    for s, g in t.groupby("season"):
        w = g.poss_season
        assert abs(np.average(half.loc[g.index, "portable_off"], weights=w) - np.average(g.rating_off, weights=w)) < 1e-12
    d = half.portable_off - (t.prior_off + 0.5 * (t.u_off + t.c_off))
    assert np.allclose(d.groupby(t.season).std(), 0.0)          # one constant shift per season
