"""The team-game leave-one-out target: the closure identities, the constant, and the rebalancing claim.

Everything here is built by hand or simulated, so the tests state what the estimator is supposed to do
rather than what this season's data happens to say.  The one claim worth stating up front: the closure
of `team_design` is a TEAM-GAME identity, not a row identity (see its docstring), and the tests check it
at that level deliberately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eracoef import teamloo as tl


# --------------------------------------------------------------------------- fixtures
def simulate_season(n_teams=30, games=82, tau={"fg3": 0.030, "fg2": 0.022, "ft": 0.024},
                    mu={"fg3": 0.36, "fg2": 0.54, "ft": 0.78},
                    natt={"fg3": 33.0, "fg2": 55.0, "ft": 22.0}, def_tau=0.0, seed=0):
    """A season of team-games with KNOWN between-team true-rate standard deviations.

    Each team draws a true rate per component; every game draws attempts and then binomial makes.  The
    defence has `def_tau` of its own on threes, so the estimators can be asked to find a zero as well as
    a positive.  Points follow the counters exactly, so `pts100` is a real label.
    """
    rng = np.random.default_rng(seed)
    pi = {c: np.clip(rng.normal(mu[c], tau[c], n_teams), 0.05, 0.95) for c in tl.COMPONENTS}
    dpi = np.clip(rng.normal(0.0, def_tau, n_teams), -0.2, 0.2) if def_tau > 0 else np.zeros(n_teams)
    rows = []
    for g in range(games * n_teams // 2):
        a, b = rng.choice(n_teams, 2, replace=False)
        for off, dfn, is_home in ((a, b, True), (b, a, False)):
            r = dict(game_id=f"{g:07d}", is_home_off=is_home, team_id=int(off), opp_id=int(dfn),
                     home=1.0 if is_home else -1.0, season=2024, game_date=pd.Timestamp("2023-10-01") + pd.Timedelta(days=g // 8))
            pts = 0.0
            for c in tl.COMPONENTS:
                n = int(rng.poisson(natt[c]))
                p = float(np.clip(pi[c][off] + (dpi[dfn] if c == "fg3" else 0.0), 0.02, 0.98))
                m = int(rng.binomial(n, p))
                r[f"n_{c}"], r[f"m_{c}"] = float(n), float(m)
                r[f"p_{c}"] = m / n if n else np.nan
                pts += tl.VALUE[c] * m
            r["poss"] = 100.0
            r["pts"], r["pts100"] = pts, pts
            r["tov"] = float(rng.poisson(14))
            rows.append(r)
    tg = pd.DataFrame(rows).sort_values(["game_date", "game_id", "is_home_off"]).reset_index(drop=True)
    tg["rank"] = np.arange(len(tg))
    tg["ftm_tech"] = 0.0
    # The truth an estimator can reach is the variance of the teams that were actually DRAWN, weighted
    # the way the estimator weights them.  With 30 teams the sample variance sits about 26% away from
    # the population tau^2, so comparing to tau^2 would be testing the random number generator.
    truth = {}
    for c in tl.COMPONENTS:
        w = tg[f"n_{c}"].to_numpy(dtype=float)
        v = pi[c][tg["team_id"].to_numpy()]
        truth[("off", c)] = float(np.average((v - np.average(v, weights=w)) ** 2, weights=w))
    w3 = tg["n_fg3"].to_numpy(dtype=float)
    d = dpi[tg["opp_id"].to_numpy()]
    truth[("def", "fg3")] = float(np.average((d - np.average(d, weights=w3)) ** 2, weights=w3))
    return tg, truth


def hand_stints(n=8, seed=3):
    """A tiny counters frame plus the row metadata `team_design` reads, for the closure tests."""
    rng = np.random.default_rng(seed)
    fg3a = rng.integers(2, 12, n).astype(float)
    fg3m = np.array([rng.integers(0, int(a) + 1) for a in fg3a], dtype=float)
    fg2a = rng.integers(4, 20, n).astype(float)
    fg2m = np.array([rng.integers(0, int(a) + 1) for a in fg2a], dtype=float)
    fta = rng.integers(0, 8, n).astype(float)
    ftm = np.array([rng.integers(0, int(a) + 1) for a in fta], dtype=float)
    c = pd.DataFrame(dict(fg3a=fg3a, fg3m=fg3m, fga=fg2a + fg3a, fgm=fg2m + fg3m, fta=fta, ftm=ftm,
                          fta_tech=np.zeros(n), ftm_tech=np.zeros(n), xftm=ftm * 0.8, xftm_tech=np.zeros(n),
                          poss=rng.integers(8, 25, n).astype(float), pts_tech=np.zeros(n)))
    c["pts"] = 3 * fg3m + 2 * fg2m + ftm
    return c


# --------------------------------------------------------------------------- the partner rule
def test_partner_moves_the_mean_back_without_overshooting():
    label = np.array([100.0, 120.0, 90.0, 110.0, 80.0])
    w = np.ones(5)
    partner = tl.rebalance_partners(label, w)
    M = label.mean()
    for j, k in enumerate(partner):
        d1 = np.delete(label, j).mean() - M
        if k < 0:
            continue
        d2 = np.delete(label, [j, k]).mean() - M
        assert abs(d2) < abs(d1), "the partner must move the mean back toward the full mean"
        assert np.sign(d2) * np.sign(d1) >= 0, "and must not overshoot it"


def test_partner_is_minus_one_when_nothing_helps():
    assert (tl.rebalance_partners(np.ones(4), np.ones(4)) == -1).all()
    assert (tl.rebalance_partners(np.array([1.0, 2.0]), np.ones(2)) == -1).all()


def test_every_fold_drops_one_or_two_games():
    tg, _ = simulate_season(n_teams=6, games=20, seed=1)
    loo = tl.loo_rates(tg)
    n_all = tg.groupby("team_id")["n_fg3"].transform("count").to_numpy()
    dropped = np.where(loo["off_partner"].to_numpy() >= 0, 2, 1)
    assert set(np.unique(dropped)) <= {1, 2}
    assert (dropped < n_all).all()


# --------------------------------------------------------------------------- identity (A)
def test_plain_loo_is_an_exact_negative_multiple_of_the_deviation():
    """p_loo(j) - pbar == -n_j (p_j - pbar) / (S_n - n_j), the mechanical bias, exactly."""
    tg, _ = simulate_season(n_teams=8, games=30, seed=2)
    loo = tl.loo_rates(tg, rebalance="none")
    g = tg.groupby("team_id")[["n_fg3", "m_fg3"]].transform("sum")
    n, m = tg["n_fg3"].to_numpy(float), tg["m_fg3"].to_numpy(float)
    pbar = g["m_fg3"].to_numpy() / g["n_fg3"].to_numpy()
    lhs = loo["off_pl_fg3"].to_numpy() - pbar
    rhs = -n * (m / n - pbar) / (g["n_fg3"].to_numpy() - n)
    assert np.abs(lhs - rhs).max() < 1e-12


# --------------------------------------------------------------------------- the constant
@pytest.mark.parametrize("seed", [0, 7])
def test_estimate_k_recovers_the_true_between_team_variance(seed):
    tg, truth = simulate_season(seed=seed)
    kt = tl.k_table(tg, tl.loo_rates(tg))
    for c in tl.COMPONENTS:
        r = kt[(kt.component == c) & (kt.side == "off")].iloc[0]
        t = truth[("off", c)]
        # 35%: with 30 teams the estimate of a between-team variance is itself noisy, and the
        # independent split-half reference misses by the same amount in the same direction, so this
        # tolerance is the estimator's sampling error and not slack for a bias.
        assert abs(r["tau2"] - t) < 0.35 * t, f"{c}: rebalanced tau2 {r['tau2']:.2e} vs truth {t:.2e}"
        assert abs(r["tau2_half"] - t) < 0.45 * t, f"{c}: the split-half reference missed the truth"
        # k is in attempt units and must be near within/truth
        assert abs(r["k"] - r["within"] / t) < 0.6 * (r["within"] / t)


def test_a_defence_with_no_control_reads_as_no_control():
    tg, _ = simulate_season(def_tau=0.0, seed=5)
    kt = tl.k_table(tg, tl.loo_rates(tg))
    r = kt[(kt.component == "fg3") & (kt.side == "def")].iloc[0]
    off = kt[(kt.component == "fg3") & (kt.side == "off")].iloc[0]
    assert r["tau2_half"] < 0.25 * off["tau2_half"], "a defence that controls nothing must not read like an offence"


@pytest.mark.parametrize("seed", [0, 11])
def test_the_moment_estimate_barely_needs_rebalancing(seed):
    """Measured, not assumed, and it is the opposite of what the paper's headline suggests.

    For a MOMENT estimate of the between-team variance the mechanical term of identity (A) is very
    nearly cancelled by the sampling noise in the team's own mean, so the plain-LOO covariance is
    already close to the independent split-half reference and rebalancing moves it only a few percent.
    Rebalancing earns its keep in the REGRESSION (the test below), not here.  If this ever starts
    failing because the gap has grown, the estimate of `k` is the thing to re-derive.
    """
    tg, _ = simulate_season(seed=seed)
    kt = tl.k_table(tg, tl.loo_rates(tg))
    for c in tl.COMPONENTS:
        r = kt[(kt.component == c) & (kt.side == "off")].iloc[0]
        assert r["tau2_plain"] < r["tau2"], f"{c}: rebalancing must raise the covariance"
        assert r["bias_exact"] < 0.0, f"{c}: the mechanical term is negative by construction"
        assert abs(r["tau2"] - r["tau2_plain"]) < 0.10 * r["tau2"], \
            f"{c}: for tau2 the two must be within a few percent of each other"
        assert abs(r["tau2_plain"] - r["tau2_half"]) < 0.20 * r["tau2_half"], \
            f"{c}: plain LOO must track the independent split-half reference"


def test_rebalancing_de_attenuates_the_matchup_regression():
    """Where the paper's bias actually bites: the LOO regressor is mechanically anticorrelated with its
    own held-out label, so a plain-LOO fit attenuates."""
    tg, _ = simulate_season(seed=13)
    plain = tl.matchup_regression(tg, tl.loo_rates(tg, rebalance="none"), design="composite")
    reb = tl.matchup_regression(tg, tl.loo_rates(tg, rebalance="label"), design="composite")
    names = ("off_fg3", "off_fg2", "off_ft")
    for k in names:
        assert reb["coef"][k] > plain["coef"][k], f"{k} must de-attenuate"
    # the composite is points per possession, so an unattenuated coefficient is 100.  Rebalancing does
    # not land every component on it -- on this simulation it fixes most of a large attenuation on twos
    # and free throws and slightly OVERSHOOTS on threes -- so the claim tested is that it is closer on
    # average, which is what it is entitled to.
    err = lambda r: np.mean([abs(r["coef"][k] - 100.0) for k in names])
    assert err(reb) < 0.6 * err(plain)


# --------------------------------------------------------------------------- the regression
def test_realised_design_is_an_identity():
    tg, _ = simulate_season(n_teams=10, games=40, seed=4)
    r = tl.matchup_regression(tg, tl.loo_rates(tg), design="realised")
    assert r["r2"] > 1.0 - 1e-9
    assert np.abs(r["pred"] - tg["pts100"].to_numpy()).max() < 1e-8


def test_composite_design_predicts_something_and_prices_home():
    tg, _ = simulate_season(seed=6)
    r = tl.matchup_regression(tg, tl.loo_rates(tg), design="composite")
    assert 0.0 < r["r2"] < 0.6, "one game is mostly noise; a large R2 would mean a leak"
    assert abs(r["coef"]["home"]) < 2.0, "the simulation has no home edge"


# --------------------------------------------------------------------------- the adjusted rates
def test_k_zero_is_the_games_own_rate():
    tg, _ = simulate_season(n_teams=8, games=30, seed=8)
    loo = tl.loo_rates(tg)
    ks = {(r["side"], r["component"]): r for r in tl.k_table(tg, loo).to_dict("records")}
    adj = tl.adjusted_rates(tg, loo, ks, prior="off", k_fixed=0.0)
    for c in tl.COMPONENTS:
        own = np.clip(tg[f"m_{c}"] / tg[f"n_{c}"], *tl.P_CLIP)
        assert np.abs(adj[f"p_adj_{c}"].to_numpy() - own.to_numpy()).max() < 1e-12


def test_adjusted_rate_sits_between_the_game_and_the_prior():
    tg, _ = simulate_season(n_teams=8, games=30, seed=9)
    loo = tl.loo_rates(tg)
    ks = {(r["side"], r["component"]): r for r in tl.k_table(tg, loo).to_dict("records")}
    adj = tl.adjusted_rates(tg, loo, ks, prior="off")
    for c in tl.COMPONENTS:
        own = (tg[f"m_{c}"] / tg[f"n_{c}"]).to_numpy()
        prior = np.clip(loo[f"off_p_{c}"].to_numpy(), *tl.P_CLIP)
        p = adj[f"p_adj_{c}"].to_numpy()
        lo, hi = np.minimum(own, prior), np.maximum(own, prior)
        assert (p >= lo - 1e-9).all() and (p <= hi + 1e-9).all()
        assert np.abs(p - prior).mean() < np.abs(p - own).mean(), "k is large, so the prior dominates"


def test_prior_both_uses_both_sides():
    tg, _ = simulate_season(seed=10)
    loo = tl.loo_rates(tg)
    ks = {(r["side"], r["component"]): r for r in tl.k_table(tg, loo).to_dict("records")}
    a_off = tl.adjusted_rates(tg, loo, ks, prior="off")["p_adj_fg3"].to_numpy()
    a_both = tl.adjusted_rates(tg, loo, ks, prior="both")["p_adj_fg3"].to_numpy()
    assert np.abs(a_off - a_both).max() > 1e-6


# --------------------------------------------------------------------------- the stint target
class _FakeWD:
    """Just the four things `team_design` reads off a WindowData."""

    def __init__(self, counters, rows, games, y, w):
        self.counters, self.rows, self.games, self.y, self.w = counters, rows, games, y, w
        self.handed = None

    def with_target(self, y, w=None):
        self.handed = y
        return self


def _fake_design(tg, per_game=3, seed=12):
    """Split each team-game's counters into `per_game` stints, so the row/team-game distinction is real."""
    rng = np.random.default_rng(seed)
    rows, cnt = [], []
    for i, r in tg.iterrows():
        frac = rng.dirichlet(np.ones(per_game))
        splits = {}
        for c in tl.COMPONENTS:
            n = int(r[f"n_{c}"])
            m = int(r[f"m_{c}"])
            # split makes first, then the misses, so every stint's makes <= its attempts
            sm = rng.multinomial(m, frac).astype(float)
            smiss = rng.multinomial(n - m, frac).astype(float)
            splits[c] = (sm + smiss, sm)
        for s in range(per_game):
            rows.append(dict(game_idx=int(i), season=int(r["season"]), is_home_off=bool(r["is_home_off"]),
                             poss=float(r["poss"]) / per_game))
            cnt.append(dict(fg3a=splits["fg3"][0][s], fg3m=splits["fg3"][1][s],
                            fga=splits["fg3"][0][s] + splits["fg2"][0][s],
                            fgm=splits["fg3"][1][s] + splits["fg2"][1][s],
                            fta=splits["ft"][0][s], ftm=splits["ft"][1][s],
                            fta_tech=0.0, ftm_tech=0.0, xftm=splits["ft"][1][s], xftm_tech=0.0,
                            pts_tech=0.0, poss=float(r["poss"]) / per_game))
    rows = pd.DataFrame(rows)
    cnt = pd.DataFrame(cnt)
    cnt["pts"] = 3 * cnt["fg3m"] + 2 * (cnt["fgm"] - cnt["fg3m"]) + cnt["ftm"]
    y = 100.0 * cnt["pts"].to_numpy() / cnt["poss"].to_numpy()
    games = tg[["game_id", "season"]].copy()
    games["game_idx"] = np.arange(len(tg))
    return _FakeWD(cnt, rows, games, y, cnt["poss"].to_numpy())


def _team_game_points(wd, y):
    key = pd.factorize(pd.MultiIndex.from_arrays([wd.rows["game_idx"], wd.rows["is_home_off"]]))[0]
    return np.bincount(key, weights=y * wd.rows["poss"].to_numpy() / 100.0)


def _patched(monkeypatch, tg):
    monkeypatch.setattr(tl, "team_games", lambda season, cfg, keep=None: tg)


def test_a_zero_reproduces_points_row_by_row(monkeypatch):
    tg, _ = simulate_season(n_teams=8, games=30, seed=14)
    _patched(monkeypatch, tg)
    wd = _fake_design(tg)
    tl.team_design([2024], None, wd, comps=("fg3", "fg2"), ft="none", a=0.0, calibrate=False)
    assert np.abs(wd.handed - wd.y).max() < 1e-12


def test_k_zero_closes_at_team_game_level(monkeypatch):
    """The closure identity: with no shrinkage the team-game total is actual points exactly, while the
    individual stints have moved -- the game's makes spread over its attempts."""
    tg, _ = simulate_season(n_teams=8, games=30, seed=15)
    _patched(monkeypatch, tg)
    wd = _fake_design(tg)
    tl.team_design([2024], None, wd, comps=("fg3", "fg2"), ft="none", a=1.0, k_fixed=0.0, calibrate=False)
    assert np.abs(_team_game_points(wd, wd.handed) - _team_game_points(wd, wd.y)).max() < 1e-9
    assert np.abs(wd.handed - wd.y).max() > 1.0, "the rows must move; that is what this target does"


def test_season_total_is_within_half_a_percent_before_alignment(monkeypatch):
    tg, _ = simulate_season(seed=16)
    _patched(monkeypatch, tg)
    wd = _fake_design(tg)
    _, rep = tl.team_design([2024], None, wd, comps=("fg3", "fg2"), ft="team", a=1.0, calibrate=False)
    g = rep["gates"].iloc[0]
    assert abs(g["pts"] - 1.0) < 0.005
    for col in ("r_fg3", "r_fg2", "r_ft"):
        assert abs(g[col] - 1.0) < 0.02
    assert rep["uncovered"] == 0.0


def test_the_target_has_less_variance_than_points(monkeypatch):
    tg, _ = simulate_season(seed=17)
    _patched(monkeypatch, tg)
    wd = _fake_design(tg)
    tl.team_design([2024], None, wd, comps=("fg3", "fg2"), ft="team", a=1.0, calibrate=False)
    assert wd.handed.std() < wd.y.std()


def test_partial_scalar_interpolates(monkeypatch):
    tg, _ = simulate_season(n_teams=10, games=40, seed=18)
    _patched(monkeypatch, tg)
    ys = {}
    for a in (0.0, 0.5, 1.0):
        wd = _fake_design(tg)
        tl.team_design([2024], None, wd, comps=("fg3", "fg2"), ft="none", a=a, calibrate=False)
        ys[a] = wd.handed
    assert np.abs(ys[0.5] - 0.5 * (ys[0.0] + ys[1.0])).max() < 1e-9


def test_missing_counters_are_named(monkeypatch):
    tg, _ = simulate_season(n_teams=6, games=20, seed=19)
    _patched(monkeypatch, tg)
    wd = _fake_design(tg)
    wd.counters = wd.counters.drop(columns=["xftm"])
    with pytest.raises(KeyError, match="xftm"):
        tl.team_design([2024], None, wd, comps=("fg3",), ft="shooter", calibrate=False)


# --------------------------------------------------------------------------- the cut
def test_a_cut_season_sees_only_the_kept_games(monkeypatch):
    """`keep` is `inseason.keep_games`: the fit may see the first part of the anchor season only."""
    tg, _ = simulate_season(n_teams=8, games=30, seed=20)
    kept = set(tg["game_id"].unique()[: len(tg["game_id"].unique()) // 2])

    def fake_team_games(season, cfg, keep=None):
        return tg if keep is None else tg[tg["game_id"].isin(set(keep))].reset_index(drop=True)

    monkeypatch.setattr(tl, "team_games", fake_team_games)
    full, _ = tl.adjusted_table([2024], None, keep=None, prior="off")
    cut, _ = tl.adjusted_table([2024], None, keep={2024: sorted(kept)}, prior="off")
    assert set(cut["game_id"]) == kept
    both = full.merge(cut, on=["season", "game_id", "is_home_off"], suffixes=("_f", "_c"))
    assert np.abs(both["p_adj_fg3_f"] - both["p_adj_fg3_c"]).max() > 1e-6, \
        "a cut fit must not reproduce the whole-season rates"


# --------------------------------------------------------------------------- the registries
def test_every_target_declares_its_counters():
    assert set(tl.TARGETS) == set(tl.TARGET_COLUMNS)
    for name, cols in tl.TARGET_COLUMNS.items():
        assert {"pts", "poss", "fg3a", "fg3m", "fga", "fgm"} <= cols, name


def test_the_ladder_is_registered_as_systems():
    from eracoef.config import load_config
    from eracoef.systems import registry
    S = registry(load_config())
    for base in ("tune501_b7_pasto_pOD", "ks52_lam05"):
        assert f"{base}_pts" in S
        for t in tl.TARGETS:
            assert f"{base}_{t}" in S, f"{base}_{t}"
    assert "ks52_lam05_tlfto_q75" in S
    s = S["tune501_b7_pasto_pOD_tlft32o"]
    assert s.off_target == "tlft32o" and s.def_target == "tlft32o"
    assert s.counter_columns() is not None, "an undeclared target makes the design build all 118 counters"
    assert S["tune501_b7_pasto_pOD_tlfto_O"].def_target == "x3def"
