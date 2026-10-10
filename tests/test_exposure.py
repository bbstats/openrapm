"""The padded per-100 box rates (src/openrapm/exposure.py) on the path that ships: spm.apm_fit -> cv.plugin_fit ->
BoxExposure in full mode with the possession-conditional padding target, whose season_rates_ become the panel's
13 rates (49_role_panel.py, seasoninputs.py).  The first four tests are test_core's padding tests, moved unchanged
when test_core was deleted in the cleanup's phase 1 (they move into test_box_rates in phase 9a)."""
import numpy as np

import pytest

from openrapm.cv import crossfit_beta
from openrapm.design import FEATURES, build_design
from openrapm.exposure import BoxExposure, split_half_k
from openrapm.simulate import simulate
from openrapm.spm import apm_fit

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500}


@pytest.fixture(scope="module")
def small():
    sim = simulate(n_seasons=2, n_teams=6, players_per_team=10, games_per_season=40, stints_per_game=(20, 30), seed=1)
    wd = build_design(sim["stints"], sim["box"], FEATURES, CFG)
    return sim, wd


def _exp(wd, **kw):
    kw.setdefault("pad_k", {f: 50.0 for f in FEATURES})
    kw.setdefault("center", False)
    return BoxExposure(wd.game_box, wd.game_poss, FEATURES, spec=wd.spec, game_half=wd.game_half, **kw)


def test_full_mode_test_rows_use_training_totals_only(small):
    _, wd = small
    game = wd.rows["game_idx"].to_numpy()
    held = np.unique(game)[:5]
    tr = ~np.isin(game, held); te = ~tr
    exp = _exp(wd, mode="full").fit(wd.X[tr])
    gb = wd.game_box[(wd.game_box.phase == "RS") & ~wd.game_box.game_idx.isin(held)]
    tot = np.zeros_like(exp.totals_)
    np.add.at(tot, gb.psx_idx.to_numpy(), gb[FEATURES].to_numpy())
    np.testing.assert_allclose(exp.totals_, tot)
    assert not np.isin(held, exp.covariate_games_).any()
    Xt = exp.transform(wd.X[te])
    nf = len(FEATURES)
    box_o = Xt[:, wd.spec.box_o(nf)].toarray()
    Zo = wd.X[te][:, wd.spec.zo].tocsr()
    r = 3
    po = Zo.indices[Zo.indptr[r]:Zo.indptr[r + 1]]
    np.testing.assert_allclose(box_o[r], exp.rates_[po].sum(0), rtol=1e-10)


def test_padding_and_centering_formulas(small):
    _, wd = small
    k = {f: 10.0 * (j + 1) for j, f in enumerate(FEATURES)}
    exp = _exp(wd, mode="full", pad_k=k, center=True)
    exp.fit(wd.X, sample_weight=wd.w)
    ps = 5
    s = wd.spec.season_of_ps[ps]
    kk = np.array([k[f] for f in FEATURES])
    expect = (100 * exp.totals_[ps] + kk * exp.league_rates_[s]) / (exp.poss_off_[ps] + kk)
    np.testing.assert_allclose(exp.rates_[ps], expect)
    np.testing.assert_allclose(exp.season_rates_[ps], expect)   # full mode: same thing
    Xt = exp.transform(wd.X)
    B = Xt[:, wd.spec.n_base:].toarray()
    np.testing.assert_allclose((wd.w[:, None] * B).sum(0) / wd.w.sum(), 0.0, atol=1e-9)
    exp2 = _exp(wd, mode="full", pad_k=k, pad_scale=2.0).fit(wd.X)
    np.testing.assert_allclose(exp2.pad_k_, 2 * exp.pad_k_)
    # crossfit: season_rates_ (reporting) use both halves, rates_ (covariates) only the other half
    m = wd.half_mask("A")
    exp3 = _exp(wd, mode="crossfit", half="A", pad_k=k).fit(wd.X[m])
    assert exp3.season_poss_off_.sum() > exp3.poss_off_.sum()


def test_split_half_k_recovers_truth():
    rng = np.random.default_rng(0)
    n_pl, n_games = 400, 60
    mu = rng.gamma(4.0, 2.0 / 4.0, size=(n_pl, 2))
    mu[:, 1] = rng.gamma(4.0, 6.0 / 4.0, size=n_pl)
    poss = rng.integers(20, 60, size=(n_pl, n_games)).astype(float)
    counts = rng.poisson(mu[:, None, :] * poss[:, :, None] / 100.0).astype(float)
    ps = np.repeat(np.arange(n_pl), n_games)
    g = np.tile(np.arange(n_games), n_pl)
    k, s2, t2, _ = split_half_k(ps, g, counts.reshape(-1, 2), poss.ravel(), np.zeros(n_pl, int), 1)
    k_true = 100 * mu.mean(0) / mu.var(0)
    np.testing.assert_allclose(k[0], k_true, rtol=0.2)


def test_pad_target_and_fixed_padding(small):
    _, wd = small
    a = crossfit_beta(wd, lam=8000.0)
    exp = a.fits["A"]["exposure"]
    # unit game weights: n_eff equals the possession count, so nothing changes
    np.testing.assert_allclose(exp.poss_off_eff_, exp.poss_off_, rtol=1e-10)
    # frozen padding constants reproduce the fit exactly
    frozen = {h: {"fixed_padding": a.fits[h]["exposure"].padding_state()} for h in ("A", "B")}
    c = crossfit_beta(wd, lam=8000.0, exposure_kw_by_half=frozen)
    np.testing.assert_allclose(a.beta_, c.beta_, rtol=1e-12)
    # possession-conditional target: one target per possession bin, different from the league mean
    b = crossfit_beta(wd, lam=8000.0, pad_target="poss_conditional")
    eb = b.fits["A"]["exposure"]
    assert eb.pad_bins_ is not None and eb.pad_target_.shape[1] >= 2
    assert not np.allclose(eb.pad_target_[0, 0], eb.pad_target_[0, -1])
    # a player with no covariate possessions gets the target, not a rate
    p = np.flatnonzero(eb.poss_off_ == 0)
    if len(p):
        np.testing.assert_allclose(eb.rates_[p[0]], eb._target(eb.poss_off_[p[:1]], wd.spec.season_of_ps[p[:1]])[0])


def test_apm_fit_returns_the_full_mode_possession_conditional_rates(small):
    """The call 49 and seasoninputs make, at config.yaml's pad_target and spm penalty: the rates come from a full-mode
    BoxExposure with one padding target per possession bin, and the centred rates are season_rates_ minus a fifth
    of the lineup means."""
    _, wd = small
    cfg = {"pad_target": "poss_conditional", "spm": {"apm_lam": 100, "apm_lam_ratio": 1.0}}
    a = apm_fit(wd, cfg)
    exp = a["pipe"]["exposure"]
    assert exp.mode == "full" and a["lam"] == 100.0
    assert exp.pad_bins_ is not None and exp.pad_target_.shape[1] >= 2
    np.testing.assert_allclose(exp.season_rates_, exp.rates_)                      # full mode: one set of rates
    np.testing.assert_allclose(a["ro"], exp.season_rates_ - exp.means_o_ / 5.0)
    np.testing.assert_allclose(a["rd"], exp.season_rates_d_ - exp.means_d_ / 5.0)
    np.testing.assert_allclose(a["poss_o"], exp.season_poss_off_)
    m = wd.spec.n_ps
    assert a["u_o"].shape == (m,) and a["u_d"].shape == (m,) and np.isfinite(a["u_o"]).all()
