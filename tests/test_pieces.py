"""The pieces of one season's vanilla RAPM (src/eracoef/pieces.py), on the simulated league.

The decomposition's two splits are identities, not estimates: each side's five by-player pieces and its four
possession groups must add up to that side's RAPM to rounding.  A failure there means a piece is mislabelled, and
nothing downstream -- the season panel, the prior, the cross-fit -- would notice.  The second claim is the one the
cross-fit rests on: a fold built from a design's own rows (a `WindowData.subset`, which has no stored parts and is
read back off its matrix) gives the same pieces as the whole design.
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from eracoef import pieces
from eracoef.design import FEATURES, build_design
from eracoef.simulate import simulate

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500, "features": FEATURES,
       "first_season": 2001, "last_season": 2001, "windows": [[2001, 2001]],
       "lam_plugin": 4000.0, "lam_ratio_plugin": 1.0, "pad_target": "league", "holdout": {}}
TOL = 1e-9


def _with_parts(wd):
    """The design as the cache hands it over (`designcache` stores its blocks; `build_design` does not)."""
    n, n_f = wd.spec.n_ps, len(wd.spec.f_names)
    X = sp.csr_matrix(wd.X)
    X.sort_indices()

    def lineup(block):
        return block.indices.reshape(-1, 5).astype(np.int64)

    Z = X[:, :2 * n].tocsr()
    parts = dict(Z=Z, F=np.asarray(X[:, 2 * n:2 * n + n_f].todense(), dtype=float),
                 lineup_o=lineup(X[:, :n].tocsr()), lineup_d=lineup(X[:, n:2 * n].tocsr()),
                 game_idx=wd.rows.game_idx.to_numpy(np.int64))
    return dataclasses.replace(wd, parts=parts)


@pytest.fixture(scope="module")
def season():
    sim = simulate(n_seasons=1, n_teams=6, players_per_team=10, games_per_season=40, stints_per_game=(15, 25),
                   seed=11, eps_var=0.2, leak=False)
    stints, box = sim["stints"], sim["box"]
    wd = _with_parts(build_design(stints, box, FEATURES, CFG))
    team = sim["truth"]["ps"].set_index("player_id").team
    # (game_id, player_id) -> team, as scripts/83_decompose_site.py's box_teams builds it from the box scores
    teams = pd.Series(team.reindex(box.player_id.to_numpy()).to_numpy(np.int64),
                      index=pd.MultiIndex.from_arrays([box.game_id.astype(str).to_numpy(),
                                                       box.player_id.to_numpy(np.int64)]))
    return wd, teams


def test_every_identity_holds_on_both_sides(season):
    wd, teams = season
    tab, miss = pieces.decompose_sides(pieces.fit(wd, teams, 3000.0))
    assert len(tab) > 40
    for name in ("by_player", "by_possession", "ridge", "weights"):
        assert miss[name] < TOL, f"{name} misses by {miss[name]:.1e}"
    for side in ("off", "def"):
        by_player = tab[[f"{k}_{side}" for k in pieces.BY_PLAYER]].sum(axis=1)
        by_possession = tab[[f"{k}_{side}" for k in pieces.BY_POSSESSION]].sum(axis=1)
        assert np.allclose(by_player, tab[f"rapm_{side}"], atol=TOL, rtol=0)
        assert np.allclose(by_possession, tab[f"rapm_{side}"], atol=TOL, rtol=0)


def test_a_fold_of_every_row_reproduces_the_whole_season(season):
    """The cross-fit's path (no stored parts) and the panel's path (stored parts) are the same computation."""
    wd, teams = season
    assert wd.parts is not None
    whole, _ = pieces.season_pieces(wd, wd, teams)
    every_row = wd.subset(np.ones(len(wd.y), dtype=bool))
    assert every_row.parts is None
    rebuilt, _ = pieces.season_pieces(every_row, every_row, teams)
    pd.testing.assert_frame_equal(whole, rebuilt, check_exact=False, atol=1e-12, rtol=0)


def test_a_fold_changes_the_pieces_and_keeps_the_identities(season):
    """Four fifths of the games: different pieces, the same identities, every column still present."""
    wd, teams = season
    games = wd.rows.game_idx.to_numpy()
    fold = wd.subset(games % 5 != 0)
    tab, miss = pieces.decompose_sides(pieces.fit(fold, teams, 3000.0))
    assert miss["by_player"] < TOL and miss["by_possession"] < TOL
    whole, _ = pieces.season_pieces(wd, wd, teams)
    part, _ = pieces.season_pieces(fold, fold, teams)
    assert list(part.columns) == ["player_id"] + pieces.COLUMNS
    joined = whole.merge(part, on="player_id", suffixes=("", "_fold"))
    assert not np.allclose(joined.pc_on_rtg_o, joined.pc_on_rtg_o_fold)


def test_the_prior_columns_are_padded_toward_zero(season):
    """Padded over the possessions behind them, so no piece is larger than its unpadded self."""
    wd, teams = season
    tab, _ = pieces.decompose_sides(pieces.fit(wd, teams, 3000.0))
    padded, _ = pieces.season_pieces(wd, wd, teams)
    ids = pieces.fit(wd, teams, 3000.0)["player_ids"][tab.i.to_numpy()]
    raw = tab.assign(player_id=ids).set_index("player_id")
    got = padded.set_index("player_id").reindex(raw.index)
    for k in pieces.PIECES:
        shrink = raw.poss_off / (raw.poss_off + 3000.0)
        assert np.allclose(got[f"pc_{k}_o"], raw[f"{k}_off"] * shrink, atol=1e-12)
        assert (got[f"pc_{k}_o"].abs() <= raw[f"{k}_off"].abs() + 1e-12).all()
