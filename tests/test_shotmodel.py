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


def test_relevel_uses_the_other_half_only():
    f = _frame(seasons=(2005,))
    eta = np.zeros(len(f))
    f.loc[f.half == "B", "made"] = 1                  # half B makes everything
    out, shifts = sm.relevel(eta, f, k=0.0)
    a = (f.half == "A").to_numpy()
    assert out[a].mean() > 5.0                        # half A is priced from half B's makes
    assert abs(out[~a].mean()) < 3.0                  # half B from half A's, which are ordinary


def test_the_update_moves_a_make_up_and_a_miss_down_by_the_capped_weight():
    q = np.array([0.4, 0.4, 0.4])
    after = sm.update_after(q, np.array([1, 0, 1]), np.array([0.006, 0.006, 1.0]))
    w = 0.006 / 0.24
    assert np.allclose(after[:2], [0.4 + 0.6 * w, 0.4 - 0.4 * w])
    assert np.isclose(after[2], 0.4 + 0.6 * 0.15)
