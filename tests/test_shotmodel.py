"""The shot-quality model (eracoef.shotmodel): it recovers a known shot effect, stays shooter-neutral, never
prices a season it saw, and sets the level from the other half."""
import numpy as np
import pandas as pd
import pytest

from eracoef import shotmodel as sm


def _frame(seed=0, seasons=(2001, 2002, 2003, 2010, 2011), n=24000):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        k = n // len(seasons)
        value = np.where(rng.random(k) < 0.3, 3, 2)
        d = np.where(value == 3, rng.uniform(23.0, 27.0, k), rng.uniform(0.0, 20.0, k))
        shooter = rng.integers(0, 80, k)
        skill = np.linspace(-0.6, 0.6, 80)[shooter]
        # good shooters take the long twos: a model without the offset would read long twos as easier
        d = np.where((value == 2) & (skill > 0.3), np.clip(d + 6.0, 0, 21.9), d)
        start = rng.choice(["dreb", "made_fg", "steal"], k, p=[0.4, 0.45, 0.15])
        tposs = np.where(start == "steal", rng.uniform(1, 6, k), rng.uniform(4, 22, k))
        logit = 0.9 - 0.08 * d + 0.5 * (start == "steal") + skill - 0.4 * (value == 3)
        made = (rng.random(k) < 1 / (1 + np.exp(-logit))).astype(int)
        rows.append(pd.DataFrame(dict(season=s, phase="RS", half=np.where(np.arange(k) % 2 == 0, "A", "B"),
                                      shooter=shooter, arena=rng.integers(0, 30, k), value=value, made=made,
                                      dist_xy=d, angle=rng.uniform(0, 90, k), noloc=False, corner3=False,
                                      poss_start=start, secs_into_poss=tposs, true_logit=logit - skill)))
    return pd.concat(rows, ignore_index=True)


def test_the_model_reads_the_shot_and_not_the_shooter():
    f = _frame()
    with_off = sm.fit(f, ["spot", "start"], offsets=True)
    no_off = sm.fit(f, ["spot", "start"], offsets=False)
    test = _frame(seed=1, seasons=(2020,))
    eta_w = with_off.predict_raw(test)
    eta_n = no_off.predict_raw(test)
    long2 = (test.value == 2) & (test.dist_xy > 15)
    err_w = np.abs(eta_w - test.true_logit)[long2].mean()
    err_n = np.abs(eta_n - test.true_logit)[long2].mean()
    assert err_w < err_n                       # without the offset the long twos look too easy
    # the steal effect (+0.5 on the logit, carried by the start dummy and its early-seconds term together)
    steal = (test.poss_start == "steal").to_numpy()
    gap = (eta_w - test.true_logit.to_numpy())
    assert abs(gap[steal].mean() - gap[~steal].mean()) < 0.15


def test_a_model_refuses_a_season_it_saw_or_its_neighbour():
    f = _frame(seasons=(2001, 2002))
    m = sm.fit(f, ["spot"], offsets=False)
    with pytest.raises(ValueError):
        m.predict_raw(_frame(seed=2, seasons=(2003,)))
    m.predict_raw(_frame(seed=2, seasons=(2004,)))


def test_train_seasons_leave_out_the_block_and_its_neighbours_and_the_current_season():
    seasons = list(range(1997, 2027))
    tr = sm.train_seasons((2015, 2016, 2017), seasons)
    assert 2014 not in tr and 2018 not in tr and 2016 not in tr and 2026 not in tr and 2013 in tr and 2019 in tr


def test_relevel_sets_the_season_level_by_default_and_the_other_half_on_request():
    f = _frame(seasons=(2005,))
    eta = np.zeros(len(f))
    f.loc[f.half == "B", "made"] = 1                  # half B makes everything
    a = (f.half == "A").to_numpy()
    season, _ = sm.relevel(eta, f, k=0.0)
    cell = pd.Series(season).groupby([sm.band_of(f), f["half"].to_numpy()]).first().unstack()
    assert np.allclose(cell["A"], cell["B"])                       # one level per distance cell for the whole season
    half, _ = sm.relevel(eta, f, k=0.0, mode="half")              # the rejected rule, kept to rerun the record
    assert half[a].mean() > 5.0 and abs(half[~a].mean()) < 3.0


def test_the_update_moves_a_make_up_and_a_miss_down_by_the_capped_weight():
    q = np.array([0.4, 0.4, 0.4])
    after = sm.update_after(q, np.array([1, 0, 1]), np.array([0.006, 0.006, 1.0]))
    w = 0.006 / 0.24
    assert np.allclose(after[:2], [0.4 + 0.6 * w, 0.4 - 0.4 * w])
    assert np.isclose(after[2], 0.4 + 0.6 * 0.15)


# ------------------------------------------------------------------------------------------ the search's options
def _rich_frame(seed=0, seasons=(2001, 2002, 2003, 2010, 2011), n=24000, inter=0.0, trend=0.0, trend_mid=2008):
    """`_frame` plus every column the default blocks and the interactions read.  `inter` plants a late-clock long-shot
    penalty (logit + inter x [shot clock under 4 s] x distance / 20); `trend` a steal effect that grows by `trend`
    on the logit per decade from `trend_mid`."""
    f = _frame(seed, seasons, n)
    rng = np.random.default_rng(seed + 100)
    k = len(f)
    f["sc_eff"] = rng.uniform(0.0, 24.0, k)
    f["clock_off"] = rng.random(k) < 0.03
    f["reset_kind"] = rng.choice(["made_fg", "dreb", "oreb", "steal", "dfoul"], k)
    f["secs_since_oreb"] = np.where(rng.random(k) < 0.15, rng.uniform(0.0, 12.0, k), np.nan)
    f["period"] = rng.integers(1, 6, k)
    f["clock"] = rng.uniform(0.0, 720.0, k)
    f["margin"] = rng.integers(-25, 26, k)
    f["shooter_secs_on"] = np.where(rng.random(k) < 0.01, np.nan, rng.uniform(0.0, 1200.0, k))
    f["start_prev_value"] = rng.choice([0, 2, 3], k)
    f["start_prev_dist"] = np.where(rng.random(k) < 0.3, np.nan, rng.uniform(0.0, 28.0, k))
    f["start_prev_made"] = rng.choice([-1, 0, 1], k)
    f["start_prev_blocked"] = rng.choice([-1, 0, 1], k, p=[0.3, 0.65, 0.05])
    if inter or trend:
        skill = np.linspace(-0.6, 0.6, 80)[f["shooter"].to_numpy()]
        true = (f["true_logit"].to_numpy() + inter * (f["sc_eff"].to_numpy() < 4.0) * f["dist_xy"].to_numpy() / 20.0
                + trend * (f["season"].to_numpy() - trend_mid) / 10.0 * (f["poss_start"].to_numpy() == "steal"))
        f["made"] = (rng.random(k) < 1 / (1 + np.exp(-(true + skill)))).astype(int)
        f["true_logit"] = true
    return f


def _det_frame(n=6000, seasons=(2001, 2002, 2010)):
    """A frame with no random generator (Weyl sequences), so its numbers never move with numpy's streams."""
    i = np.arange(n)
    roots = np.sqrt([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0])

    def u(j):
        return (i * roots[j]) % 1.0

    value = np.where(u(0) < 0.3, 3, 2)
    d = np.where(value == 3, 23.0 + 5.0 * u(1), 21.0 * u(1))
    start = np.array(["made_fg", "dreb", "steal", "made_ft", "dead_tov"])[(u(2) * 5).astype(int)]
    t = 24.0 * u(3)
    sc = 24.0 * u(4)
    logit = 0.9 - 0.08 * d + 0.5 * (start == "steal") - 0.4 * (value == 3) - 0.3 * (sc < 4) + 0.2 * (t < 6)
    return pd.DataFrame(dict(
        season=np.asarray(seasons)[i % len(seasons)], phase="RS", half=np.where(i % 2 == 0, "A", "B"),
        shooter=(u(5) * 60).astype(int), arena=(u(6) * 20).astype(int), value=value,
        made=(u(7) < 1 / (1 + np.exp(-logit))).astype(int), dist_xy=d, angle=90.0 * u(8), noloc=False,
        corner3=(value == 3) & (u(9) < 0.2), poss_start=start, secs_into_poss=t, sc_eff=sc, clock_off=u(10) < 0.03,
        reset_kind=np.array(["made_fg", "dreb", "oreb", "dfoul"])[(u(11) * 4).astype(int)],
        secs_since_oreb=np.where(u(12) < 0.15, 12.0 * u(13), np.nan), period=1 + (u(14) * 5).astype(int),
        clock=720.0 * u(15), margin=np.round(50.0 * u(13) - 25.0), shooter_secs_on=1200.0 * u(12),
        start_prev_value=np.where(u(9) < 0.4, 3, np.where(u(9) < 0.8, 2, 0)), start_prev_dist=28.0 * u(8),
        start_prev_made=np.where(u(10) < 0.5, 0, 1), start_prev_blocked=np.where(u(11) < 0.05, 1, 0)))


# what the shipped fit gave on _det_frame BEFORE the search's options existed (all blocks, the 5-season closeness for
# 2005-07, every other argument at its default): per sub-model (columns, sum of beta, sum of beta^2), then the logit
# of 900 attempts of 2020.  Computed with the pre-search shotmodel.py, 2026-10-06.
_GOLDEN = {"rim": (52, 2.328644847566167, 3.6889949626971994),
           "mid": (55, 2.329349016103775, 1.3965399505378),
           "three": (56, -0.4189980144951056, 12.53771920420776)}
_GOLDEN_ETA = (-0.30810385342296476, 0.8905258571380323,
               (-0.20646776013775758, -0.19757606639923964, 0.441753172023327, -1.698062224004719))


def _legacy_irls(X, y, w, ridge, offset, beta0, iters=25):
    """The IRLS as it was before the search (one ridge, the weighted design built twice a step), verbatim."""
    n, p = X.shape
    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=float).copy()
    P = ridge * np.eye(p)
    P[0, 0] = 0.0
    for _ in range(iters):
        eta = X @ beta + offset
        mu = 1.0 / (1.0 + np.exp(-eta))
        s = np.maximum(mu * (1 - mu), 1e-6) * w
        z = X @ beta + (y - mu) / np.maximum(mu * (1 - mu), 1e-6)
        new = np.linalg.solve((X * s[:, None]).T @ X + P, (X * s[:, None]).T @ z)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def _legacy_fit(train, blocks, w_all, ridge=1.0, lam=50.0, rounds=4):
    """The backfit as it was before the search, verbatim apart from returning plain arrays."""
    sub = sm.submodel_of(train)
    out = {}
    for s in sm.SUBMODELS:
        m = sub == s
        f = train[m]
        y = f["made"].to_numpy(float)
        w = w_all[m]
        X = np.column_stack([np.ones((len(f), 1))] + [sm.block_columns(f, s, b)[0] for b in blocks])
        sh = pd.factorize(f["season"].astype(str) + ":" + f["shooter"].astype(str))
        ar = pd.factorize(f["season"].astype(str) + ":" + f["arena"].astype(str))
        d_sh, d_ar, beta = np.zeros(len(sh[1])), np.zeros(len(ar[1])), None
        for _ in range(rounds):
            off = d_sh[sh[0]] + d_ar[ar[0]]
            beta = _legacy_irls(X, y, w, ridge, off, beta)
            eta = X @ beta
            d_sh = sm._group_offsets(eta + d_ar[ar[0]], y, w, sh[0], len(sh[1]), lam)
            d_ar = sm._group_offsets(eta + d_sh[sh[0]], y, w, ar[0], len(ar[1]), lam)
        out[s] = (beta, d_sh, d_ar)
    return out


def test_the_default_fit_is_the_shipped_fit_bit_for_bit():
    f = _rich_frame()
    test = _rich_frame(seed=1, seasons=(2020,), n=6000)
    blocks = list(sm.BLOCKS)
    w = sm.closeness_weights(f["season"].to_numpy(), (2005, 2007))
    implicit = sm.fit(f, blocks, weights=w)
    explicit = sm.fit(f, blocks, weights=w, standardise=False, ridge_int=None, era_lambda=None, knots=None, inter=(),
                      half_life_past=None, half_life_future=None, block=None, era_clip=False)
    legacy = _legacy_fit(f, blocks, w)
    sub = sm.submodel_of(test)
    for s in sm.SUBMODELS:
        beta, d_sh, d_ar = legacy[s]
        for model in (implicit, explicit):
            assert np.array_equal(model.coef[s][1], beta)
            assert np.array_equal(model.offsets[s]["shooter"].to_numpy(), d_sh)
            assert np.array_equal(model.offsets[s]["arena"].to_numpy(), d_ar)
            assert model.knots == {} and model.scale == {} and model.era is None and model.inter == ()
        m = sub == s
        X = np.column_stack([np.ones((m.sum(), 1))] + [sm.block_columns(test[m], s, b)[0] for b in blocks])
        assert np.array_equal(implicit.predict_raw(test)[m], X @ beta)
    assert np.array_equal(implicit.predict_raw(test), explicit.predict_raw(test))
    assert np.array_equal(implicit.predict_raw(test, with_offsets=True), explicit.predict_raw(test, with_offsets=True))
    # and the numbers themselves, against those the pre-search code gave (blocks, knots and all)
    g = sm.fit(_det_frame(), blocks, weights=sm.closeness_weights(_det_frame()["season"].to_numpy(), (2005, 2007)))
    for s, (p, total, squares) in _GOLDEN.items():
        b = g.coef[s][1]
        assert len(b) == p and np.isclose(b.sum(), total, rtol=1e-9, atol=1e-11)
        assert np.isclose((b ** 2).sum(), squares, rtol=1e-9)
    eta = g.predict_raw(_det_frame(n=900, seasons=(2020,)))
    assert np.isclose(eta.mean(), _GOLDEN_ETA[0], rtol=1e-9) and np.isclose(eta.std(), _GOLDEN_ETA[1], rtol=1e-9)
    assert np.allclose(eta[:4], _GOLDEN_ETA[2], rtol=1e-9, atol=0)


def test_the_ridge_takes_one_value_or_one_per_column():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(4000), rng.standard_normal((4000, 5))])
    y = (rng.random(4000) < 1 / (1 + np.exp(-(X @ np.array([0.2, 1.0, -0.5, 0.3, 0.8, -1.0]))))).astype(float)
    assert np.array_equal(sm.logistic_irls(X, y, ridge=2.0), sm.logistic_irls(X, y, ridge=np.full(6, 2.0)))
    r = np.full(6, 1e-6)
    r[4] = 1e9                                          # one column held at zero, the rest all but free
    b = sm.logistic_irls(X, y, ridge=r)
    free = sm.logistic_irls(np.delete(X, 4, axis=1), y, ridge=1e-6)
    assert abs(b[4]) < 1e-5 and np.allclose(np.delete(b, 4), free, atol=1e-4)
    with pytest.raises(ValueError):
        sm.logistic_irls(X, y, ridge=np.ones(5))


def test_closeness_decays_apart_before_and_after_the_block():
    s = np.arange(1997, 2027)
    blk = (2009, 2011)
    dist = np.where(s < 2009, 2009 - s, np.where(s > 2011, s - 2011, 0))
    assert np.array_equal(sm.closeness_weights(s, blk), 0.5 ** (dist / 5.0))        # unchanged
    past = sm.closeness_weights(s, blk, half_life_past=2.0)
    assert np.allclose(past[s < 2009], 0.5 ** ((2009 - s[s < 2009]) / 2.0)) and np.all(past[s >= 2009] == 1.0)
    future = sm.closeness_weights(s, blk, half_life=1.0, half_life_future=4.0)      # `half_life` then unused
    assert np.allclose(future[s > 2011], 0.5 ** ((s[s > 2011] - 2011) / 4.0)) and np.all(future[s <= 2011] == 1.0)
    both = sm.closeness_weights(s, blk, half_life_past=3.0, half_life_future=float("inf"))
    assert np.allclose(both, np.where(s < 2009, 0.5 ** ((2009 - s) / 3.0), 1.0))


def test_the_fit_multiplies_the_half_lives_into_the_weights_and_needs_the_block():
    f = _rich_frame(n=9000)
    v = np.where(f["season"].to_numpy() == 2010, 2.0, 1.0)
    a = sm.fit(f, ["spot", "start"], weights=v, half_life_past=3.0, half_life_future=10.0, block=range(2005, 2008))
    w = v * sm.closeness_weights(f["season"].to_numpy(), (2005, 2007), half_life_past=3.0, half_life_future=10.0)
    b = sm.fit(f, ["spot", "start"], weights=w)
    for s in sm.SUBMODELS:
        assert np.array_equal(a.coef[s][1], b.coef[s][1])
    assert a.params["block"] == (2005, 2007) and a.params["half_life_past"] == 3.0
    with pytest.raises(ValueError):
        sm.fit(f, ["spot"], half_life_past=3.0)
    with pytest.raises(ValueError):
        sm.fit(f, ["spot"], era_lambda=10.0)
    with pytest.raises(ValueError):
        sm.fit(f, ["spot"], inter=("spot3d",))


def test_standardising_puts_the_ridge_on_sd_scaled_columns_and_prices_in_natural_units():
    f = _rich_frame(n=12000)
    blocks = ["spot", "start", "clock"]
    m = sm.fit(f, blocks, ridge=50.0, standardise=True, offsets=False)
    sub = sm.submodel_of(f)
    for s in sm.SUBMODELS:
        g = f[sub == s]
        X, names = sm.design(g, s, blocks)
        sd = X[:, 1:].std(axis=0)
        binary = ((X[:, 1:] == 0) | (X[:, 1:] == 1)).all(axis=0)       # yes/no columns keep scale 1 (the review)
        sd = np.where(binary | (sd <= 1e-12), 1.0, sd)
        assert np.allclose(m.scale[s], np.r_[1.0, sd])
        # the same fit: the plain design with a ridge of 50 x sd^2 on each column
        direct = sm.logistic_irls(X, g["made"].to_numpy(float), ridge=np.r_[0.0, 50.0 * sd ** 2])
        assert m.coef[s][0] == names and np.allclose(m.coef[s][1], direct, rtol=1e-6, atol=1e-8)
    # with next to no ridge, scaling the columns changes nothing
    a = sm.fit(f, blocks, ridge=1e-8, standardise=True, offsets=False)
    b = sm.fit(f, blocks, ridge=1e-8, offsets=False)
    test = _rich_frame(seed=1, seasons=(2020,), n=4000)
    assert np.max(np.abs(a.predict_raw(test) - b.predict_raw(test))) < 1e-5


def test_knots_sit_at_the_training_quantiles_and_travel_with_the_model():
    f = _rich_frame()
    m = sm.fit(f, ["spot", "start", "fatigue"], knots={"dist": 6, "tposs": 5, "on": 3}, offsets=False)
    sub = sm.submodel_of(f)
    for s in sm.SUBMODELS:
        g = f[sub == s]
        kp = m.knots[s]
        assert np.allclose(kp["dist"], np.quantile(np.clip(g["dist_xy"], 0, 40), sm.QUANTILE_PROBS[6]), atol=1e-6)
        assert len(kp["tposs"]) == 5 and len(kp["on"]) == 3
        names = m.coef[s][0]
        assert sum(n.startswith("dist") for n in names) == 5 and sum(n.startswith("tposs") for n in names) == 4
    # the priced rows never move the knots: pricing in two pieces gives the same logits
    test = _rich_frame(seed=1, seasons=(2020,), n=6000)
    whole = m.predict_raw(test)
    assert np.allclose(whole, np.concatenate([m.predict_raw(test.iloc[:1000]), m.predict_raw(test.iloc[1000:])]),
                       rtol=0, atol=1e-12)
    # repeated quantiles (whole seconds piled at 0) merge; fewer than three distinct keeps the fixed knots
    i = np.arange(len(f))
    piled = sm.knot_positions(f.assign(secs_into_poss=np.where(i % 4 == 0, (i % 30).astype(float), 0.0)), "mid",
                              {"tposs": 7})
    assert piled["tposs"][0] == 0.0 and len(piled["tposs"]) == 3
    assert "tposs" not in sm.knot_positions(f.assign(secs_into_poss=5.0), "mid", {"tposs": 5})


def test_an_interaction_block_finds_a_planted_interaction_and_its_ridge_can_switch_it_off():
    f = _rich_frame(n=40000, inter=-1.5)
    test = _rich_frame(seed=1, seasons=(2020,), n=20000, inter=-1.5)
    blocks = ["spot", "start", "clock"]
    base = sm.fit(f, blocks, standardise=True)
    withx = sm.fit(f, blocks, standardise=True, inter=("clock_x_dist",), ridge_int=1.0)
    # the same mid-range shots at 2 s and at 14 s on the clock: the planted gap grows with distance
    probe = test[sm.submodel_of(test) == "mid"]
    late, ref = probe.assign(sc_eff=2.0), probe.assign(sc_eff=14.0)
    planted = -1.5 * probe["dist_xy"].to_numpy() / 20.0

    def miss(m):
        return np.abs((m.predict_raw(late) - m.predict_raw(ref)) - planted).mean()

    assert miss(base) > 0.25 and miss(withx) < 0.3 * miss(base)        # 0.33 against 0.06 when written
    off = sm.fit(f, blocks, standardise=True, inter=("clock_x_dist",), ridge_int=1e12)
    assert np.max(np.abs(off.predict_raw(test) - base.predict_raw(test))) < 1e-3
    # every block builds finite, named columns for every sub-model; the putback one only at the rim
    allx = sm.fit(f, list(sm.BLOCKS), standardise=True, inter=sm.INTER_BLOCKS, rounds=2)
    for s in sm.SUBMODELS:
        names = allx.coef[s][0]
        assert np.all(np.isfinite(allx.coef[s][1])) and len(names) == len(set(names))
        assert any(n.startswith("putback_x_dist:") for n in names) == (s == "rim")
        assert all(any(n.startswith(b + ":") for n in names) for b in sm.INTER_BLOCKS if b != "putback_x_dist")
    assert np.all(np.isfinite(allx.predict_raw(test)))


def test_the_era_block_reads_the_main_effects_at_the_rated_blocks_own_era():
    # the steal effect grows by 1.0 on the logit per decade; rating 2024-26 from 1999-2001 and 2015-17 alone
    kw = dict(trend=1.0, trend_mid=2025)
    f = _rich_frame(seasons=(1999, 2000, 2001, 2015, 2016, 2017), n=48000, **kw)
    test = _rich_frame(seed=1, seasons=(2025,), n=20000, **kw)
    blocks = ["spot", "start"]
    plain = sm.fit(f, blocks, standardise=True)
    era = sm.fit(f, blocks, standardise=True, era_lambda=1.0, block=(2024, 2026))
    assert era.era["mid"] == 2025.0 and np.isclose(era.era["hi"], -0.8) and np.isclose(era.era["lo"], -2.6)
    names = era.coef["mid"][0]
    assert "era:intercept" in names and "era:start_steal" in names and not any(n.startswith("era:dist") for n in names)
    steal = (test["poss_start"] == "steal").to_numpy()

    def steal_bias(m):
        gap = m.predict_raw(test) - test["true_logit"].to_numpy()
        return abs(gap[steal].mean() - gap[~steal].mean())

    assert steal_bias(plain) > 0.5 and steal_bias(era) < 0.25
    # a rated season is read at its own era; `era_clip` holds it at the nearest training season instead
    t24, t26 = test.assign(season=2024), test.assign(season=2026)
    assert np.max(np.abs(era.predict_raw(t26) - era.predict_raw(t24))) > 1e-3
    clipped = sm.fit(f, blocks, standardise=True, era_lambda=1.0, block=(2024, 2026), era_clip=True)
    assert np.array_equal(clipped.predict_raw(t24), clipped.predict_raw(t26))


def test_margin_train_and_the_leakage_rule_hold_with_every_option():
    f = _rich_frame(n=12000)
    m = sm.fit(f, list(sm.BLOCKS), standardise=True, ridge=3.0, ridge_int=30.0, era_lambda=100.0, block=(2006, 2008),
               knots={"dist": 4, "tposs": 7, "on": 5}, inter=("spot2d", "start_x_time"), half_life_past=4.0,
               half_life_future=8.0, rounds=2)
    margin = m.margin_train(f)
    plain = m.margin_train(f, with_offsets=False)
    sub = sm.submodel_of(f)
    for s in sm.SUBMODELS:
        X, names = m.design(f[sub == s], s)
        assert names == m.coef[s][0] and np.allclose(plain[sub == s], X @ m.coef[s][1])
    assert np.std(margin - plain) > 0.0                    # the offsets are in the margin
    with pytest.raises(ValueError):
        m.predict_raw(f)                                   # its own training seasons
    with pytest.raises(ValueError):
        m.predict_raw(_rich_frame(seed=2, seasons=(2004,), n=2000))   # next to one
    with pytest.raises(ValueError):
        m.margin_train(_rich_frame(seed=2, seasons=(2020,), n=2000))
    m.predict_raw(_rich_frame(seed=2, seasons=(2007,), n=2000))
