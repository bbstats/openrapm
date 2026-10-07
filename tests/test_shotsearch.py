"""The search harness (eracoef.shotsearch): the registered split, the fast relevel, and SALL's ranking."""
import numpy as np
import pandas as pd
import pytest

from eracoef import shotmodel as sm
from eracoef import shotsearch as ss


def _frame(seed=0, seasons=(2003, 2004), n=30000):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        value = np.where(rng.random(n) < 0.3, 3, 2)
        d = np.where(value == 3, rng.uniform(23.0, 27.0, n), rng.uniform(0.0, 20.0, n))
        shooter = rng.integers(0, 120, n)
        skill = np.linspace(-0.7, 0.7, 120)[shooter]
        d = np.where((value == 2) & (skill > 0.3), np.clip(d + 7.0, 0, 21.9), d)       # good shooters take long twos
        true = 0.9 - 0.08 * d - 0.4 * (value == 3)
        made = (rng.random(n) < 1 / (1 + np.exp(-(true + skill)))).astype(int)
        rows.append(pd.DataFrame(dict(season=s, phase="RS", half=np.where(np.arange(n) % 2 == 0, "A", "B"),
                                      shooter=shooter, arena=rng.integers(0, 30, n), value=value, made=made,
                                      dist_xy=d, angle=rng.uniform(0, 90, n), noloc=False, corner3=False,
                                      true=true, skill=skill)))
    return pd.concat(rows, ignore_index=True)


def test_the_split_is_registered_and_enforced():
    assert not ss.SEARCH_SEASONS & ss.CONFIRM_SEASONS
    assert len(ss.SEARCH_SEASONS | ss.CONFIRM_SEASONS) == 30 and {2014, 2015, 2016, 2017, 2026} <= ss.CONFIRM_SEASONS
    ss.assert_phase([1998, 2019], "search")
    with pytest.raises(ValueError):
        ss.assert_phase([1998, 2015], "search")


def test_row_perm_nests_the_search_sample_in_the_full_one():
    p = ss.row_perm(200_000, 2010)
    assert set(p[:5_000]) <= set(p[:10_000]) and np.array_equal(p, ss.row_perm(200_000, 2010))
    assert not np.array_equal(p, ss.row_perm(200_000, 2011))


def test_relevel_fast_equals_relevel():
    f = _frame()
    f["phase"] = np.where(np.arange(len(f)) % 17 == 0, "PO", "RS")
    eta = np.random.default_rng(3).normal(0, 0.4, len(f))
    slow, _ = sm.relevel(eta, f)
    fast = ss.relevel_fast(eta, f["season"].to_numpy(), ss.band_codes(f), f["made"].to_numpy(), (f["phase"] == "RS").to_numpy())
    assert np.max(np.abs(slow - fast)) < 1e-12


def test_sall_ranks_the_shot_alone_above_a_model_of_who_shoots():
    """The planted league: good shooters take long twos.  A logit that learned that (the true shot plus the
    shooter's average selection) wins on raw log loss; SALL must prefer the true shot quality."""
    f = _frame(seasons=(2003,))
    truth = f["true"].to_numpy()
    # 'selection': the shot's logit shifted by the average skill of whoever takes that kind of shot
    band = pd.cut(f["dist_xy"], [0, 7, 14, 30]).astype(str) + f["value"].astype(str)
    selection = truth + f.groupby(band)["skill"].transform("mean").to_numpy()
    rs = np.ones(len(f), bool)
    lev = {k: ss.relevel_fast(e, f["season"].to_numpy(), ss.band_codes(f), f["made"].to_numpy(), rs)
           for k, e in (("truth", truth), ("selection", selection), ("flat", np.zeros(len(f))))}
    T = {k: ss.sall_table(e, f) for k, e in lev.items()}
    raw = {k: np.average(t["raw"], weights=t["n"]) for k, t in T.items()}
    sall = {k: np.average(t["sall"], weights=t["n"]) for k, t in T.items()}
    assert raw["selection"] < raw["truth"]                 # raw makes reward knowing who shoots
    assert sall["truth"] < sall["selection"] < sall["flat"]  # SALL does not
