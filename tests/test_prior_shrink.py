"""scripts/99_prior_shrink.py: the prior-part multipliers are a least squares that recovers a planted value, and the
pinned fold seasons rebuild the two shipped tables exactly (the 2026-10-04 fold folder grew to 1997-2026, and an
unpinned rerun shrank harder than what ships -- experiment 34's first run)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("prior_shrink_99", ROOT / "scripts" / "99_prior_shrink.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Fold:
    """The attributes `fit_prior_multipliers` reads from a 98 `Fold`, planted with known multipliers."""

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
    m = _script()
    rng = np.random.default_rng(7)
    folds = [_Fold(rng, season, 0.7, 0.95) for season in (2017, 2018, 2019)]
    m_off, m_def = m.fit_prior_multipliers(folds)
    assert abs(m_off - 0.7) < 1e-9 and abs(m_def - 0.95) < 1e-9


def test_the_test_rule_leaves_out_the_rated_season_and_its_neighbours():
    m = _script()
    rng = np.random.default_rng(11)
    # a fold season's planted multiplier says which seasons were used: 2018 is the odd one out
    folds = [_Fold(rng, s, 0.5 if s == 2018 else 0.8, 0.9) for s in (2016, 2017, 2018, 2019, 2020)]
    out = m.multipliers_for([2018, 2022], folds, "test").set_index("season")
    assert abs(out.at[2018, "prior_off"] - 0.8) < 1e-9        # 2017-2019 left out, so only 0.8 folds remain
    assert out.at[2018, "fold_seasons"] == 2
    assert out.at[2022, "fold_seasons"] == 5                   # far from every fold season: all of them
    assert 0.5 < out.at[2022, "prior_off"] < 0.8


def test_fold_seasons_parse():
    m = _script()
    assert m.parse_seasons("2017-2026") == (2017, 2026)
    assert m.parse_seasons("all") is None


_FOLDS = ROOT / "outputs" / "within" / "within"
_NEEDED = [ROOT / "outputs" / f"season_ratings_{n}.parquet"
           for n in ("unshrinkdef", "priorshrink_raw", "product_pre_swap", "product_priorshrink_pre_swap")]


@pytest.mark.skipif(not _FOLDS.exists() or not all(p.exists() for p in _NEEDED),
                    reason="needs the within-season folds and the shipped tables")
@pytest.mark.parametrize("base,rule,shipped", [("unshrinkdef", "test", "priorshrink_raw"),
                                                ("product_pre_swap", "product", "product_priorshrink_pre_swap")])
def test_the_pinned_folds_rebuild_the_shipped_tables_exactly(base, rule, shipped):
    m = _script()
    folds = m.load_folds(_FOLDS, m.parse_seasons("2017-2026"))
    table = pd.read_parquet(ROOT / "outputs" / f"season_ratings_{base}.parquet")
    got = m.shrink(table, m.multipliers_for(table.season.unique(), folds, rule)).set_index(["player_id", "season"])
    ref = pd.read_parquet(ROOT / "outputs" / f"season_ratings_{shipped}.parquet").set_index(["player_id", "season"])
    cols = ["rating_off", "rating_def", "prior_off", "prior_def"]
    assert float((got[cols] - ref.loc[got.index, cols]).abs().to_numpy().max()) <= 1e-10
