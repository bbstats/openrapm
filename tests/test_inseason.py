"""The in-season kernel and the game cut (src/eracoef/inseason.py).

The identities that have to hold, because they are what says the cut leaks nothing:

  * a flat kernel with no cut is the plain fit on the same seasons;
  * a cut of 1 is no cut;
  * a cut of 0 at anchor `a` leaves the fit knowing nothing about `a`;
  * the ridge rows and the games behind the padded rates carry the SAME weight, game by game.

The whole-fit identities against real data live in `scratch/inseason_ident.py` (they need the role
panel): the flat kernel reproduces the shipped board to 0.0e+00, and a cut of 0 reproduces the
shorter kernel on the earlier seasons to 5e-14 once the prior's window-exclusion set is matched --
a cut fit trains on the anchor season's window too, so it keeps the prior off one more window.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef.cv import make_exposure
from eracoef.design import FEATURES, build_design
from eracoef.holdout import RESULT_COLUMNS, SPLITS, Context, Holdout, Ratings, TableSystem, cut_season
from eracoef.inseason import (BlockSystem, KernelSystem, anchor_of, kernel_game_mult, kernel_seasons,
                              keep_games, season_frac, season_rank)
from eracoef.simulate import simulate

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500, "features": FEATURES,
       "first_season": 2001, "last_season": 2005, "windows": [[2001, 2003], [2004, 2006]],
       "lam_plugin": 4000.0, "lam_ratio_plugin": 1.0, "pad_target": "league",
       "holdout": {"first": 2003, "last": 2004, "ks": [2], "rank_bins": 5, "rank_min_poss": 1, "level": "full",
                   "exposure_edges": [0, 1, 300, 1000, 1e9]}}
CUTS = [0.0, 0.25, 0.5, 0.75]


@pytest.fixture(scope="module")
def world():
    sim = simulate(n_seasons=5, n_teams=8, players_per_team=10, games_per_season=40, stints_per_game=(15, 25),
                   rho=0.85, turnover=0.3, seed=5, eps_var=0.2, leak=False)
    st, box, truth = sim["stints"], sim["box"], sim["truth"]["ps"]

    def loader(seasons, cfg, target, phases=None, **kw):    # simulate() builds RS games only
        return build_design(st[st.season.isin(seasons)], box[box.season.isin(seasons)], FEATURES, cfg)

    ctx = Context(cfg=CFG, loader=loader)
    table = truth.rename(columns={"impact_O": "o", "impact_D": "d"})[["player_id", "season", "o", "d"]]
    return ctx, table


@pytest.fixture(scope="module")
def wd(world):
    ctx, _ = world
    return ctx.design([2002, 2003, 2004], "pts")


def test_rank_and_frac_are_chronological_within_each_season(wd):
    rank, frac = season_rank(wd.games), season_frac(wd.games)
    for season, g in wd.games[wd.games.phase == "RS"].groupby("season"):
        idx = np.sort(g.game_idx.to_numpy())
        assert list(rank[idx]) == list(range(len(idx)))          # 0..n-1 in game_idx order
        assert frac[idx].min() == 0.0 and frac[idx].max() < 1.0


def test_flat_kernel_is_all_ones_and_a_cut_of_one_is_no_cut(wd):
    flat = kernel_game_mult(wd, 2004, {0: 1.0, -1: 1.0, -2: 1.0}, None)
    assert np.allclose(flat, 1.0)
    assert np.allclose(kernel_game_mult(wd, 2004, {0: 1.0, -1: 1.0, -2: 1.0}, 1.0), flat)
    assert keep_games(wd, 2004, None) is None and keep_games(wd, 2004, 1.0) is None


def test_an_offset_the_kernel_does_not_name_is_zero(wd):
    gm = kernel_game_mult(wd, 2004, {0: 1.0, -1: 0.5}, None)
    season = wd.games.set_index("game_idx").season
    for gi, v in zip(season.index, gm[season.index]):
        assert v == {2004: 1.0, 2003: 0.5, 2002: 0.0}[int(season[gi])]


@pytest.mark.parametrize("q", CUTS)
def test_the_cut_zeroes_exactly_the_anchor_games_from_q_onward(wd, q):
    gm = kernel_game_mult(wd, 2004, {0: 1.0, -1: 1.0, -2: 1.0}, q)
    frac = season_frac(wd.games)
    g = wd.games[(wd.games.season == 2004) & (wd.games.phase == "RS")]
    idx = g.game_idx.to_numpy()
    assert np.all(gm[idx][frac[idx] >= q] == 0.0)
    assert np.all(gm[idx][frac[idx] < q] == 1.0)
    assert np.all(gm[wd.games[wd.games.season != 2004].game_idx.to_numpy()] == 1.0)
    keep = keep_games(wd, 2004, q)
    kept = set(map(str, keep[2004]))
    assert kept == set(g.game_id.astype(str).to_numpy()[frac[idx] < q])
    assert (len(kept) == 0) == (q == 0.0)


@pytest.mark.parametrize("q", CUTS)
def test_no_row_of_the_anchor_season_past_the_cut_carries_weight(wd, q):
    gm = kernel_game_mult(wd, 2004, {0: 1.0, -1: 0.5, -2: 0.25}, q)
    w = np.asarray(wd.w, dtype=float) * gm[wd.rows.game_idx.to_numpy()]
    frac = season_frac(wd.games)[wd.rows.game_idx.to_numpy()]
    late = (wd.rows.season.to_numpy() == 2004) & (frac >= q)
    assert not (w[late] > 0).any()
    assert (w[(wd.rows.season.to_numpy() == 2004) & (frac < q)] > 0).all() or q == 0.0


def test_the_exposure_possessions_are_the_kernel_weighted_ones(wd):
    kern = {0: 1.0, -1: 0.5, -2: 0.25}
    gm = kernel_game_mult(wd, 2004, kern, 0.5)
    w = np.asarray(wd.w, dtype=float) * gm[wd.rows.game_idx.to_numpy()]
    exp = make_exposure(wd, mode="full", pad_target="league", game_mult=gm).fit(wd.X, sample_weight=w)
    # by hand from game_poss: each player-season's RS offensive possessions times its game's weight
    gp = wd.game_poss
    rs = set(wd.games.loc[wd.games.phase == "RS", "game_idx"])
    gp = gp[gp.game_idx.isin(rs)]
    per_psx = np.zeros(wd.spec.n_psx)
    np.add.at(per_psx, gp.psx_idx.to_numpy(), gp.poss_off.to_numpy(dtype=float) * gm[gp.game_idx.to_numpy()])
    want = np.bincount(wd.spec.ps_of_psx, weights=per_psx, minlength=wd.spec.n_ps)
    assert np.allclose(exp.season_poss_off_, want, atol=1e-8)
    assert exp.season_poss_off_.sum() < np.asarray(wd.game_poss.poss_off).sum()      # the cut really cut


def test_cut_season_keeps_the_rows_after_the_cut_only(world):
    ctx, _ = world
    full = ctx.design([2004], "pts")
    for q in CUTS:
        sub = cut_season(full, 2004, q)
        frac = season_frac(full.games)[sub.rows.game_idx.to_numpy()]
        assert (frac >= q).all()
        assert len(sub.rows) < len(full.rows) or q == 0.0
    assert cut_season(full, 2004, 1.0) is full


def test_anchor_is_the_last_training_season():
    assert anchor_of([2002, 2003, 2004]) == 2004
    assert anchor_of([1997]) == 1997


def test_kernel_and_block_systems_name_their_own_training_seasons(world):
    ctx, table = world
    inner = TableSystem("t", table)
    inner.kernel = {0: 1.0, -1: 0.5, -2: 0.25}
    ks = KernelSystem("ks", inner, cut=0.5)
    assert ks.train_for(2004, ctx) == [2002, 2003, 2004]
    assert ks.train_for(2001, ctx) == [2001]                       # truncated at the first season
    blk = BlockSystem("blk", TableSystem("t", table), cut=0.5)
    assert blk.train_for(2004, ctx) == [2001, 2002, 2003]          # the last window that FINISHED before 2004
    assert blk.train_for(2002, ctx) is None                        # no window has finished yet


def test_a_zero_weight_season_is_not_a_training_season(world):
    """`kernel_game_mult` zeroes those games, which is enough for the design and the possessions -- and
    not enough for the inputs built from season tables (the shot-quality features, the role inputs),
    which are built over whatever the season LIST names.  A single-season kernel whose list still said
    three seasons moved a 2026 offensive rating by up to 0.61 per 100 that way, which is ruling 1's
    "that season's games only" broken through the feature side door."""
    ctx, table = world
    inner = TableSystem("t", table)
    inner.kernel = {0: 1.0, -1: 0.0, -2: 0.0}
    assert KernelSystem("ks00", inner).train_for(2004, ctx) == [2004]
    assert kernel_seasons(inner.kernel, 2004, 1997) == [2004]
    assert kernel_seasons({0: 1.0, -1: 0.5, -2: 0.0}, 2004, 1997) == [2003, 2004]
    assert kernel_seasons({0: 1.0, -1: 0.5, -2: 0.25}, 1998, 1997) == [1997, 1998]   # truncated, not padded
    assert kernel_seasons(None, 2004, 1997) == [2004]


def test_the_runner_scores_an_in_season_system_after_its_cut(world):
    ctx, table = world
    inner = TableSystem("t", table)
    inner.kernel = {0: 1.0, -1: 1.0, -2: 1.0}
    ho = Holdout.from_config(CFG)
    plain = TableSystem("plain", table)
    systems = [plain, KernelSystem("ks_q50", inner, cut=0.5), BlockSystem("blk_q50", plain, cut=0.5)]
    res = ho.run(systems, ctx, splits={"rookie": SPLITS["rookie"]}, verbose=False)
    assert list(res.columns) == RESULT_COLUMNS
    r = res[res.split == "all"].set_index(["system", "held_out"])
    for h in (2003, 2004):
        assert r.loc[("ks_q50", h), "train"] == f"{h - 2},{h - 1},{h}"
        assert r.loc[("ks_q50", h), "cut"] == 0.5
        assert np.isnan(r.loc[("plain", h), "cut"])
        # scored on half a season, so half the possessions of the full-season row
        assert 0.3 < r.loc[("ks_q50", h), "n"] / r.loc[("plain", h), "n"] < 0.7
    # the block baseline exists only where a window has FINISHED: 2004 yes, 2003 no
    assert r.loc[("blk_q50", 2004), "n"] == r.loc[("ks_q50", 2004), "n"]
    assert ("blk_q50", 2003) not in r.index
    assert set(res[res.split == "rookie"].group) != set()
