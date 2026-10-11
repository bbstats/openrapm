"""Step 4, the prior shrink (src/openrapm/prior_shrink.py), and the fold reader it fits on (data/within_folds.py).

The multipliers are a weighted least squares that recovers a planted value; the rules leave out the right seasons;
the cache by fold seasons changes no bit; and, where the verification copies are on disk, the golden step-3 tables
rebuild the one-thread golden step-4 tables byte for byte (the pinned fold seasons matter: on 2026-10-04 the fold
folder grew to 1997-2026, and an unpinned rerun shrank harder than what ships -- experiment 34's first run).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from openrapm import prior_shrink as ps
from openrapm.data import within_folds as wf
from openrapm.data.config import load_config, resolve

VERIFY = Path(os.environ.get("OPENRAPM_VERIFY", "A:/code/spmm_verify"))


class _Fold:
    """The attributes `fit_multipliers` reads from a `within_folds.Fold`, planted with known multipliers."""

    def __init__(self, rng, season, m_off, m_def, n_players=40, n_rows=300):
        self.season = season
        rows = {}
        for side, prior, games in (("O", "prior_off", "u_off"), ("D", "prior_def", "u_def")):
            rows[side] = pd.DataFrame({prior: rng.normal(0, 1.5, n_players), games: rng.normal(0, 0.8, n_players)})
        self.rows = rows
        self.Z = {side: sp.csr_matrix(rng.dirichlet(np.ones(n_players) * 0.2, n_rows) * 5) for side in ("O", "D")}
        prior_o = self.Z["O"] @ rows["O"].prior_off.to_numpy()
        games_o = self.Z["O"] @ rows["O"].u_off.to_numpy()
        prior_d = self.Z["D"] @ (-rows["D"].prior_def.to_numpy())
        games_d = self.Z["D"] @ (-rows["D"].u_def.to_numpy())
        stand_in = rng.normal(0, 0.3, n_rows)
        self.F = np.column_stack([rng.choice([-1.0, 1.0], n_rows), np.ones(n_rows)])
        level = self.F @ np.array([1.5, 110.0])
        self.base = prior_o + games_o + prior_d + games_d + stand_in
        self.y = level + m_off * prior_o + games_o + m_def * prior_d + games_d + stand_in
        self.w = rng.uniform(80, 110, n_rows)


def test_the_multipliers_recover_a_planted_value_exactly_without_noise():
    rng = np.random.default_rng(7)
    folds = [_Fold(rng, season, 0.7, 0.95) for season in (2017, 2018, 2019)]
    m_off, m_def = ps.fit_multipliers(folds)
    assert abs(m_off - 0.7) < 1e-9 and abs(m_def - 0.95) < 1e-9


def test_the_test_rule_leaves_out_the_rated_season_and_its_neighbours():
    rng = np.random.default_rng(11)
    # a fold season's planted multiplier says which seasons were used: 2018 is the odd one out
    folds = [_Fold(rng, s, 0.5 if s == 2018 else 0.8, 0.9) for s in (2016, 2017, 2018, 2019, 2020)]
    out = ps.multipliers_by_season([2018, 2022], folds, "test").set_index("season")
    assert abs(out.at[2018, "multiplier_off"] - 0.8) < 1e-9    # 2017-2019 left out, so only 0.8 folds remain
    assert out.at[2018, "fold_seasons"] == 2
    assert out.at[2022, "fold_seasons"] == 5                   # far from every fold season: all of them
    assert 0.5 < out.at[2022, "multiplier_off"] < 0.8


def test_the_product_rule_leaves_out_the_rated_season_only():
    assert ps.unseen_fold_seasons(2020, "product") == {2020}
    assert ps.unseen_fold_seasons(2020, "test") == {2019, 2020, 2021}
    assert ps.fold_seasons_used(2016, range(2017, 2027), "test") == tuple(range(2018, 2027))
    assert ps.fold_seasons_used(2010, range(2017, 2027), "product") == tuple(range(2017, 2027))
    with pytest.raises(ValueError):
        ps.unseen_fold_seasons(2020, "neither")


def test_one_fit_serves_every_season_with_the_same_fold_seasons(monkeypatch):
    rng = np.random.default_rng(3)
    folds = [_Fold(rng, s, 0.7, 0.9) for s in (2017, 2018)]
    calls = []
    real = ps.fit_multipliers
    monkeypatch.setattr(ps, "fit_multipliers", lambda chosen: calls.append(len(chosen)) or real(chosen))
    out = ps.multipliers_by_season(range(2000, 2019), folds, "product")
    assert len(calls) == 3                                   # 2000-2016 share one fit; 2017 and 2018 one each
    early = out[out.season < 2017]
    assert early.multiplier_off.nunique() == 1 and early.multiplier_def.nunique() == 1
    assert early.multiplier_off.iloc[0] == real(folds)[0]   # the same bits as a refit


def test_the_shrink_multiplies_the_prior_part_recentres_and_leaves_the_games_part():
    table = pd.DataFrame({"player_id": [1, 2, 3, 1, 2], "season": [2025, 2025, 2025, 2026, 2026],
                          "prior_off": [2.0, -1.0, 0.5, 1.0, -1.0], "u_off": [0.5, 0.0, -0.5, 0.2, 0.1],
                          "prior_def": [1.0, 0.0, -1.0, 0.5, -0.5], "u_def": [0.0, 0.3, 0.1, -0.1, 0.1],
                          "poss_off": [3000.0, 1000.0, 500.0, 2000.0, 2000.0]})
    for column in ("rating_off", "rating_def", "rating_total", "prior_total", "u_total", "offense", "defense",
                   "prior_offense", "prior_defense"):
        table[column] = 0.0
    multipliers = pd.DataFrame({"season": [2025, 2026], "multiplier_off": [0.7, 0.5], "multiplier_def": [0.9, 1.0]})
    out = ps.shrink(table, multipliers)
    assert list(out.columns) == list(table.columns)
    assert np.allclose(out.u_off, table.u_off) and np.allclose(out.u_def, table.u_def)
    for season, rows in out.groupby("season"):
        for side in ("off", "def"):
            assert abs(np.average(rows[f"rating_{side}"], weights=rows.poss_off)) < 1e-12
        assert np.allclose(rows.rating_off, rows.prior_off + rows.u_off)
    first = out[out.season == 2025]
    gaps = first.prior_off.to_numpy() - first.prior_off.iloc[0]
    assert np.allclose(gaps, 0.7 * (table.prior_off[:3].to_numpy() - 2.0))     # a level shift after the multiply
    assert np.allclose(out.defense, -out.rating_def) and np.allclose(out.prior_offense, out.prior_off)


def _record(season=2026, repeat=0, fold=0, rate_from="rest", checks=None):
    return {"summary": {"season": season, "repeat": repeat, "fold": fold, "rate_from": rate_from},
            "checks": checks if checks is not None else {"a leak check": True, "a measured size": 0.4}}


def test_a_fold_of_another_size_or_with_a_failed_leak_check_is_refused():
    wf.check_fold_record(_record(), "2026_r0_f0")
    wf.check_fold_record(_record(rate_from=float("nan")), "2026_r0_f0")       # written before the key existed
    with pytest.raises(ValueError, match="rates from"):
        wf.check_fold_record(_record(rate_from="one"), "2026_r0_f0")
    with pytest.raises(ValueError, match="leak checks failed"):
        wf.check_fold_record(_record(checks={"a leak check": False}), "2026_r0_f0")


def test_every_repeat_must_have_its_four_folds():
    whole = {f"2026_r0_f{k}": _record(fold=k) for k in range(4)}
    wf.check_repeats(whole)
    with pytest.raises(ValueError, match="four folds"):
        wf.check_repeats({stem: record for stem, record in whole.items() if not stem.endswith("f3")})


_CFG = load_config()
_FOLDS = resolve(_CFG, "within") / "weightfix_within"
_REBUILDS = [("product", "outputs/season_ratings_product_weightfix_pre_swap.parquet", "onethread/gate0_ps.parquet"),
             ("test", "outputs/season_ratings_weightfix_raw.parquet", "onethread/gate0_test_shrunk.parquet")]


@pytest.mark.slow
@pytest.mark.parametrize("rule, step3, step4", _REBUILDS)
def test_the_golden_step3_tables_rebuild_the_one_thread_golden_step4_tables_exactly(rule, step3, step4, tmp_path):
    """The weightfix pair: the golden step-3 table through step 4 gives the one-thread golden step 4, byte for byte."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    if not (_FOLDS.exists() and (VERIFY / "golden" / step3).exists() and (VERIFY / "golden" / step4).exists()):
        pytest.skip("needs the weightfix_within folds and the verification copies (OPENRAPM_VERIFY)")
    with threadpoolctl.threadpool_limits(1):
        folds = wf.load_folds(_FOLDS, (2017, 2026))
        table = pd.read_parquet(VERIFY / "golden" / step3)
        got = ps.shrink(table, ps.multipliers_by_season(table.season.unique(), folds, rule))
    pd.testing.assert_frame_equal(got, pd.read_parquet(VERIFY / "golden" / step4), check_exact=True)
    got.to_parquet(tmp_path / "step4.parquet", index=False)
    assert (tmp_path / "step4.parquet").read_bytes() == (VERIFY / "golden" / step4).read_bytes()
