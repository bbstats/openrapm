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
    # experiment 55: the 8 nearest allowed fold seasons, the nearer first, an equal distance the earlier
    assert ps.fold_seasons_used(2010, range(1997, 2027), "test", nearest=8) == (2005, 2006, 2007, 2008, 2012, 2013,
                                                                               2014, 2015)
    assert ps.fold_seasons_used(1998, range(1997, 2027), "test", nearest=8) == tuple(range(2000, 2008))
    assert ps.fold_seasons_used(2026, range(1997, 2027), "product", nearest=8) == tuple(range(2018, 2026))
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


def test_each_multiplier_is_read_to_a_full_season_off_the_line_through_two_holdout_sizes():
    """Experiment 101: a quarter held back gives 1.41 and a tenth 1.35, so the line reads 1.31 at nothing held back;
    the prior part's 0.70 and 0.71 read 0.7167."""
    quarter = pd.DataFrame({"season": [2026], "multiplier_off": [0.70], "games_multiplier_off": [1.41],
                            "fold_seasons": [8]})
    tenth = pd.DataFrame({"season": [2026], "multiplier_off": [0.71], "games_multiplier_off": [1.35],
                          "fold_seasons": [8]})
    full = ps.read_to_full_season(quarter, tenth, 0.25, 0.10)
    assert full.games_multiplier_off.iloc[0] == pytest.approx(1.31)
    assert full.multiplier_off.iloc[0] == pytest.approx(0.71 + 0.10 * 0.01 / 0.15)
    assert full.fold_seasons.iloc[0] == 8
    assert ps.read_to_full_season(tenth, quarter, 0.10, 0.25).games_multiplier_off.iloc[0] == pytest.approx(1.31)
    with pytest.raises(ValueError):
        ps.read_to_full_season(quarter, tenth, 0.25, 0.25)


def test_the_held_back_share_is_one_over_the_folds_of_a_repeat():
    from types import SimpleNamespace
    quarter = [SimpleNamespace(season=2026, repeat=r, fold=k) for r in range(3) for k in range(4)]
    assert ps.held_back_share(quarter) == 0.25
    assert ps.held_back_share([SimpleNamespace(season=2026, repeat=0, fold=k) for k in range(10)]) == 0.10
    with pytest.raises(ValueError):
        ps.held_back_share(quarter + [SimpleNamespace(season=2025, repeat=0, fold=k) for k in range(10)])


def _record(season=2026, repeat=0, fold=0, rate_from="rest", checks=None, folds_per_repeat=None):
    summary = {"season": season, "repeat": repeat, "fold": fold, "rate_from": rate_from}
    if folds_per_repeat is not None:
        summary["folds_per_repeat"] = folds_per_repeat
    return {"summary": summary,
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
    with pytest.raises(ValueError, match="without exactly its folds"):
        wf.check_repeats({stem: record for stem, record in whole.items() if not stem.endswith("f3")})


def test_a_repeat_of_another_fold_count_needs_every_fold_its_records_name():
    """`openrapm folds --folds-per-repeat 2`: two folds a repeat, each record saying so."""
    halves = {f"2026_r{r}_f{k}": _record(repeat=r, fold=k, folds_per_repeat=2) for r in range(2) for k in range(2)}
    wf.check_repeats(halves)
    with pytest.raises(ValueError, match="without exactly its folds"):
        wf.check_repeats({stem: record for stem, record in halves.items() if stem != "2026_r1_f1"})
    mixed = dict(halves, **{"2026_r1_f1": _record(repeat=1, fold=1, folds_per_repeat=3)})
    with pytest.raises(ValueError, match="without exactly its folds"):
        wf.check_repeats(mixed)


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


def test_the_games_part_freed_recovers_both_planted_multipliers_and_is_applied():
    rng = np.random.default_rng(5)
    folds = [_Fold(rng, season, 0.7, 0.95) for season in (2017, 2018, 2019)]
    for fold in folds:                                   # the games' parts hold up 1.3 and 1.5 times
        games_o = fold.Z["O"] @ fold.rows["O"].u_off.to_numpy()
        games_d = fold.Z["D"] @ (-fold.rows["D"].u_def.to_numpy())
        fold.y = fold.y + 0.3 * games_o + 0.5 * games_d
    m_off, m_def, g_off, g_def = ps.fit_multipliers(folds, free_games=True)
    assert np.allclose([m_off, m_def, g_off, g_def], [0.7, 0.95, 1.3, 1.5], atol=1e-9)
    table = ps.multipliers_by_season([2022], folds, "test", free_games=True)
    assert np.allclose(table[["games_multiplier_off", "games_multiplier_def"]].to_numpy(), [[1.3, 1.5]])
    rows = pd.DataFrame({"player_id": [1, 2], "season": [2022, 2022], "prior_off": [1.0, -1.0], "u_off": [0.5, -0.5],
                         "prior_def": [0.5, -0.5], "u_def": [0.2, -0.2], "poss_off": [1000.0, 1000.0]})
    for column in ("rating_off", "rating_def", "rating_total", "prior_total", "u_total", "offense", "defense",
                   "prior_offense", "prior_defense"):
        rows[column] = 0.0
    out = ps.shrink(rows, table)
    assert np.allclose(out.u_off, [0.65, -0.65]) and np.allclose(out.u_def, [0.3, -0.3])
    assert np.allclose(out.rating_off, out.prior_off + out.u_off)


# ------------------------------------------------------------------ step 4 on one within-season fold (experiment 80)
def _fold_rows(n=30, seed=3):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({"player_id": np.arange(n, dtype=np.int64), "season": 2024,
                          "prior_off": rng.normal(0, 2, n), "prior_def": rng.normal(0, 1, n),
                          "u_off": rng.normal(0, 1, n), "u_def": rng.normal(0, 1, n),
                          "poss_off": rng.uniform(10, 3000, n), "poss_def": rng.uniform(10, 3000, n)})
    for side in ("off", "def"):
        table[f"rating_{side}"] = table[f"prior_{side}"] + table[f"u_{side}"]
    return table


def test_step_4_on_a_folds_rows_is_step_4_on_a_table():
    """Both parts multiplied and each side re-centred: prior_shrink.shrink on the same rows."""
    table = _fold_rows()
    mine = ps.shrink_fold({"O": table.assign(side="O"), "D": table.assign(side="D")}, 2024, (0.7, 0.8, 1.3, 1.4))
    multipliers = pd.DataFrame({"season": [2024], "multiplier_off": [0.7], "multiplier_def": [0.8],
                                "games_multiplier_off": [1.3], "games_multiplier_def": [1.4]})
    assert ps.season_factors(multipliers, 2024) == (0.7, 0.8, 1.3, 1.4)
    theirs = ps.shrink(table.assign(rating_total=0.0, offense=0.0, defense=0.0, prior_total=0.0, u_total=0.0,
                                    prior_offense=0.0, prior_defense=0.0), multipliers)
    for column in ("rating_off", "rating_def", "prior_off", "prior_def", "u_off", "u_def"):
        assert np.allclose(mine[column].to_numpy(), theirs[column].to_numpy(), atol=1e-12), column


def test_a_player_without_a_defence_row_is_shrunk_as_the_defence_rows_are():
    table = _fold_rows()
    defence = table.iloc[1:].assign(side="D")             # player 0: offensive possessions, no defence row
    mine = ps.shrink_fold({"O": table.assign(side="O"), "D": defence}, 2024, (0.7, 0.8, 1.3, 1.4))
    rest = ps.shrink_fold({"O": table.iloc[1:].assign(side="O"), "D": defence}, 2024, (0.7, 0.8, 1.3, 1.4))
    level = 0.8 * defence.prior_def.iloc[0] - rest.prior_def.iloc[0]
    assert np.isclose(mine.prior_def.iloc[0], 0.8 * table.prior_def.iloc[0] - level)
    assert np.isclose(mine.u_def.iloc[0], 1.4 * table.u_def.iloc[0])
    assert np.allclose(mine.rating_def.iloc[1:].to_numpy(), rest.rating_def.to_numpy())
