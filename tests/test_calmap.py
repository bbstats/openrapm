"""The calibration map (src/eracoef/calmap.py) against simulated seasons with a known truth."""
import numpy as np
import pytest

from eracoef.calmap import EXPOSURES, FAMILIES, SideMap, build_design, dump_systems, evaluate, fit_theta, load_frames, unmapped_rows
from eracoef.design import FEATURES, build_design as design_build
from eracoef.holdout import Context, Holdout, TableSystem, paired, pooled
from eracoef.simulate import simulate

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500, "features": FEATURES,
       "first_season": 2001, "last_season": 2005, "windows": [[2001, 2003], [2004, 2006]],
       "lam_plugin": 4000.0, "lam_ratio_plugin": 1.0, "pad_target": "league",
       "holdout": {"first": 2002, "last": 2004, "ks": [2], "rank_min_poss": 1, "level": "full"}}


@pytest.fixture(scope="module")
def world():
    sim = simulate(n_seasons=5, n_teams=8, players_per_team=10, games_per_season=60, stints_per_game=(20, 30),
                   rho=0.85, turnover=0.3, seed=11, eps_var=0.2, leak=False)
    st, box, truth = sim["stints"], sim["box"], sim["truth"]["ps"]

    def loader(seasons, cfg, target, phases=None, **kw):    # simulate() builds RS games only
        return design_build(st[st.season.isin(seasons)], box[box.season.isin(seasons)], FEATURES, cfg)

    ctx = Context(cfg=CFG, loader=loader)
    table = truth.rename(columns={"impact_O": "o", "impact_D": "d"})[["player_id", "season", "o", "d"]]
    table = table.assign(poss=200.0 + 900.0 * (table.player_id % 7))     # exposure must vary for the exposure terms
    ho = Holdout.from_config(CFG)
    systems = [TableSystem("true", table), TableSystem("double", table.assign(o=2 * table.o, d=2 * table.d))]
    dump = dump_systems(ho, systems, ctx, verbose=False)
    frames = load_frames(ctx, ho.seasons(), level=ho.level, verbose=False)
    return ctx, ho, dump, frames


def test_dump_schema(world):
    _, ho, dump, _ = world
    assert set(dump.system) == {"true", "double"}
    assert set(dump.held_out) == set(ho.seasons())
    assert dump.groupby(["system", "k", "held_out", "player_id"]).size().max() == 1


def test_linear_map_is_the_game_level_optimum(world):
    """The pooled WLS with the level profiled out lands on the same scalar a direct search over the
    criterion's own team-game score finds, and a doubled system wants exactly half the multiplier."""
    from eracoef.holdout import Ratings, predict_season, score
    from eracoef.calmap import ratings_for
    _, ho, dump, frames = world
    lin = SideMap.parse("linear")
    th = fit_theta(build_design(dump, frames, "true", 2, lin, lin, min_poss=1))
    th2 = fit_theta(build_design(dump, frames, "double", 2, lin, lin, min_poss=1))
    assert np.allclose(th2, th / 2.0, atol=1e-6)
    h = ho.seasons()[0]
    f = frames[h]
    th1 = fit_theta(build_design(dump, {h: f}, "true", 2, lin, lin, min_poss=1))
    rat = ratings_for(dump, "true", 2, h)
    grid = np.linspace(0.2, 1.4, 25)
    tg = [score(predict_season(Ratings(rat.df.assign(o=rat.df.o * a, d=rat.df.d * th1[1])), f.wd, level=ho.level))["tg"]
          for a in grid]
    assert abs(grid[int(np.argmin(tg))] - th1[0]) <= 0.05 + 1e-9


def test_mapped_beats_unmapped_at_game_level(world):
    _, _, dump, frames = world
    base = unmapped_rows(dump, frames, "double", 2)
    rows = [base]
    for fam in ("linear", "poly2", "expo", "linear+sat"):
        m = SideMap.parse(fam)
        r, p = evaluate(dump, frames, "double", 2, m, m, f"double_{fam}", min_poss=1)
        rows.append(r)
        assert set(p.held_out) == set(frames) | {-1}
    import pandas as pd
    R = pd.concat(rows, ignore_index=True)
    P = pooled(R).set_index("system")
    assert P.loc["double_linear", "game"] < P.loc["double", "game"]
    assert abs(P.loc["double_linear", "scale_off"] - 1.0) < abs(P.loc["double", "scale_off"] - 1.0)
    z = paired(R, "double", "tg").set_index("system")
    assert z.loc["double_linear", "wins"] == 3


def test_maps_reduce_to_identity_and_keep_unseen_at_zero():
    x = np.linspace(-6, 6, 25)
    poss = np.linspace(0, 20000, 25)
    for fam in FAMILIES:
        for expo in EXPOSURES:
            m = SideMap.parse(f"{fam}+{expo}")
            assert np.allclose(m.apply(x, poss, m.identity(), 2.0), x)
            assert m.basis(x, poss, 2.0).shape == (25, m.n_params)
            if expo not in ("unseen", "bins"):          # smooth exposure terms are 0 for a player never seen
                assert np.allclose(m.expo.basis(np.zeros(3)), 0.0)


def test_calmapped_system_applies_table(world):
    """CalMappedSystem reproduces evaluate()'s leave-one-out row for the held-out season it is scoring,
    and a player the block never saw stays at 0."""
    from eracoef.calmap import CalMappedSystem, mapped_ratings, ratings_for
    from eracoef.holdout import TableSystem
    ctx, ho, dump, frames = world
    m = SideMap.parse("linear+sat")
    _, p = evaluate(dump, frames, "double", 2, m, m, "double_linear+sat", min_poss=1)
    d = dump[(dump.system == "double") & (dump.k == 2)]
    inner = TableSystem("double", d.rename(columns={"held_out": "season"})[["player_id", "season", "o", "d", "poss"]])
    sysm = CalMappedSystem("double_linear+sat", inner, p, "double_linear+sat")
    ctx.current_h, ctx.current_k = 2003, 2
    r = sysm.fit([2002, 2004], ctx)
    ctx.current_h = ctx.current_k = None
    assert r.fill_o == 0.0 and r.fill_d == 0.0
    row = p[(p.held_out == 2003)].iloc[0]
    base = inner.fit([2002, 2004], ctx)
    th = np.array([row.o0, row.o1, row.d0, row.d1])
    want = mapped_ratings(base, th, m, m, row.scale_o, row.scale_d).df
    got = r.df.set_index("player_id").loc[want.player_id]
    assert np.allclose(got.o.to_numpy(), want.o.to_numpy()) and np.allclose(got.d.to_numpy(), want.d.to_numpy())


def test_splits_reach_the_calmap_scorer(world):
    """The criterion is a team-game number, so a 200-possession player is a rounding error in the pooled
    row and every change to the bottom of the board scores as a tie.  `evaluate` / `unmapped_rows` take
    `splits` (holdout.SPLITS) and score each held-out season again inside groups of rows, which is what lets
    the criterion itself -- not a side diagnostic -- answer the bench-player question.

    Two things have to hold: the pooled row must be untouched by asking for splits (otherwise every number
    ever read off this script moved), and the groups must actually partition the season's possessions.
    """
    import pandas as pd
    from eracoef.holdout import SPLITS
    ctx, _, dump, frames = world
    sp = {"exposure": SPLITS["exposure"]}
    plain = unmapped_rows(dump, frames, "true", 2)
    split = unmapped_rows(dump, frames, "true", 2, splits=sp, ctx=ctx)
    assert set(split.split) == {"all", "exposure"}
    pooled_only = split[split.split == "all"].reset_index(drop=True)
    pd.testing.assert_frame_equal(plain, pooled_only)
    for h, g in split[split.split == "exposure"].groupby("held_out"):
        assert g.group.nunique() == len(g), "a group is scored twice"
        assert abs(g.n.sum() - float(plain.loc[plain.held_out == h, "n"].iloc[0])) < 1e-6

    m = SideMap.parse("linear")
    r, _ = evaluate(dump, frames, "true", 2, m, m, "true_linear", min_poss=1, splits=sp, ctx=ctx)
    assert set(r.split) == {"all", "exposure"}
    r_plain, _ = evaluate(dump, frames, "true", 2, m, m, "true_linear", min_poss=1)
    pd.testing.assert_frame_equal(r[r.split == "all"].reset_index(drop=True), r_plain)
