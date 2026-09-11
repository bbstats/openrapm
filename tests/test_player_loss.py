"""The PLAYER-level loss (holdout.player_truth / player_scores), against the simulator's known truth.

The team-game criterion weights a player by how much he played, so a 200-possession player is a rounding
error in it.  These losses count every player once.  The tests below check the three things that make the
number trustworthy: the pair arithmetic is exact, a perfect board loses nothing and an inverted one loses
everything, and on simulated seasons a better board wins on tau and on dollars the way it wins on MSE.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef.design import FEATURES, build_design
from eracoef.holdout import (PLAYER_COLUMNS, Context, Holdout, PluginSystem, Ratings, TableSystem, _dollar_loss,
                             _tau_b, beta_none, paired, player_report, player_scores, player_truth)
from eracoef.simulate import simulate

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500, "features": FEATURES,
       "first_season": 2001, "last_season": 2005, "windows": [[2001, 2003], [2004, 2006]],
       "lam_plugin": 4000.0, "lam_ratio_plugin": 1.0, "pad_target": "league",
       "holdout": {"first": 2002, "last": 2004, "ks": [2], "level": "full",
                   "player_edges": [0.0, 1000.0, 1e9], "player_labels": ["low", "high"],
                   "player_top_k": 10, "points_per_win": 30.0, "dollars_per_win": 3.0e6}}
EDGES, LABELS = CFG["holdout"]["player_edges"], CFG["holdout"]["player_labels"]


@pytest.fixture(scope="module")
def world():
    sim = simulate(n_seasons=5, n_teams=8, players_per_team=10, games_per_season=60, stints_per_game=(20, 30),
                   rho=0.85, turnover=0.3, seed=11, eps_var=0.2, leak=False)
    st, box, truth = sim["stints"], sim["box"], sim["truth"]["ps"]

    def loader(seasons, cfg, target, phases=None, **kw):
        assert target == "pts"
        return build_design(st[st.season.isin(seasons)], box[box.season.isin(seasons)], FEATURES, cfg)

    ctx = Context(cfg=CFG, loader=loader)
    for season, g in truth.groupby("season"):        # rosters from the truth, not from box scores
        ctx._teams[int(season)] = dict(zip(g.player_id.astype(int), g.team.astype(int)))
    table = truth.rename(columns={"impact_O": "o", "impact_D": "d"})[["player_id", "season", "o", "d"]]
    return sim, ctx, table


@pytest.fixture(scope="module")
def results(world):
    sim, ctx, table = world
    rng = np.random.default_rng(3)
    noisy = table.assign(o=table.o + rng.normal(0, 3.0, len(table)), d=table.d + rng.normal(0, 3.0, len(table)))
    systems = [TableSystem("true", table), TableSystem("noisy", noisy), TableSystem("zero", table.assign(o=0.0, d=0.0)),
               TableSystem("shifted", table.assign(o=table.o + 25.0)), PluginSystem("rapm", beta=beta_none)]
    ho = Holdout.from_config(CFG)
    res = ho.run(systems, ctx, verbose=False, players=True)
    return ho, res


# ------------------------------------------------------------------------------ the pair arithmetic
def test_tau_b_is_exact():
    assert _tau_b(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert _tau_b(np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(-1.0)
    # a tie on the board leaves that pair out of BOTH sides of tau-b, so two concordant pairs over sqrt(2 * 3)
    assert _tau_b(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0 / np.sqrt(6.0))
    assert np.isnan(_tau_b(np.zeros(4), np.arange(4.0)))       # a board that says nothing has no ordering
    assert np.isnan(_tau_b(np.array([1.0]), np.array([1.0])))


def test_dollar_loss_charges_wrong_pairs_fully_and_ties_by_half():
    truth = np.array([0.0, 30.0, 90.0])
    assert _dollar_loss(truth, truth)["dollars"] == pytest.approx(0.0)
    # exactly inverted: every pair wrong, the mean pairwise gap (30 + 90 + 60) / 3
    assert _dollar_loss(-truth, truth)["dollars"] == pytest.approx(180.0 / 3.0)
    # the board has 0 and 1 the right way round and 2 misplaced: only the (0, 2) and (1, 2) gaps are charged
    s = _dollar_loss(np.array([0.0, 30.0, -500.0]), truth)
    assert s["discordant"] == pytest.approx(2.0 / 3.0)
    assert s["dollars"] == pytest.approx((90.0 + 60.0) / 3.0)
    # a board with nothing to say is indifferent about every pair: half the gap, the no-information point
    flat = _dollar_loss(np.zeros(3), truth)
    assert flat["dollars"] == pytest.approx(0.5 * 180.0 / 3.0)
    assert flat["discordant"] == pytest.approx(0.5)


# ------------------------------------------------------------------------------ perfect and inverted boards
def test_a_perfect_board_loses_nothing_and_an_inverted_one_loses_most(world):
    _, ctx, _ = world
    wd = ctx.design([2003], "pts")
    truth = player_truth(ctx, wd)
    perfect = player_scores(Ratings(truth.df), truth, edges=EDGES, labels=LABELS, top_k=10)
    a = perfect.set_index("group").loc["all"]
    assert a.tau == pytest.approx(1.0) and a.dollars_lost == pytest.approx(0.0) and a.top_k == pytest.approx(1.0)
    flipped = truth.df.assign(o=-truth.df.o, d=-truth.df.d)
    bad = player_scores(Ratings(flipped), truth, edges=EDGES, labels=LABELS, top_k=10).set_index("group").loc["all"]
    assert bad.tau == pytest.approx(-1.0) and bad.top_k == pytest.approx(0.0) and bad.dollars_lost > 0


def test_the_truth_is_prior_free_and_recovers_simulated_impact(world):
    sim, ctx, _ = world
    wd = ctx.design([2003], "pts")
    truth = player_truth(ctx, wd)
    t = sim["truth"]["ps"]
    t = t[t.season == 2003]
    j = truth.df.merge(t[["player_id", "impact_O", "impact_D"]], on="player_id")
    assert len(j) > 50
    assert np.corrcoef(j.o, j.impact_O)[0, 1] > 0.4
    assert np.corrcoef(j.d, j.impact_D)[0, 1] > 0.4


# ------------------------------------------------------------------------------ the runner
def test_player_rows_schema_and_groups(results):
    ho, _ = results
    pf = ho.player_
    assert list(pf.columns) == PLAYER_COLUMNS
    assert (pf[pf.group == "all"].tau_pairs > 0).all()        # the within-roster tau rests on real pairs
    assert set(pf.held_out) == {2002, 2003, 2004}
    assert set(pf.group) <= {"all", *LABELS}
    # the buckets partition the players: their headcounts add up to the "all" row
    for (h, k, s), d in pf.groupby(["held_out", "k", "system"]):
        n_all = float(d[d.group == "all"].n_players.iloc[0])
        assert float(d[d.group != "all"].n_players.sum()) == n_all
        assert float(d[d.group != "all"].poss.sum()) == pytest.approx(float(d[d.group == "all"].poss.iloc[0]))


def test_true_beats_noisy_beats_zero_on_players(results):
    ho, _ = results
    a = ho.player_[ho.player_.group == "all"].groupby("system")[["tau", "dollars_lost", "top_k"]].mean()
    # noisy is true talent plus sd-3 noise on a talent of sd ~5, so it need not beat a real fit; it must
    # beat saying nothing, and true must beat both.
    assert a.loc["true", "tau"] > a.loc["noisy", "tau"] > 0 and a.loc["true", "tau"] > a.loc["rapm", "tau"] > 0
    assert a.loc["true", "dollars_lost"] < a.loc["noisy", "dollars_lost"] < a.loc["zero", "dollars_lost"]
    assert a.loc["true", "dollars_lost"] < a.loc["rapm", "dollars_lost"] < a.loc["zero", "dollars_lost"]
    assert a.loc["true", "top_k"] > a.loc["zero", "top_k"]
    z = paired(ho.player_[ho.player_.group == "all"], ref="rapm", value="tau", by=("k", "lam", "group"))
    assert float(z.set_index("system").loc["true", "mean_diff"]) > 0
    assert "dollars_lost" in player_report(ho.player_, ref="rapm")


def test_a_zero_board_sits_at_the_no_information_point(results):
    """Every player scoring 0 is one giant tie: tau is undefined (there is no ordering to score), and the
    dollars charge is half the mean pairwise gap -- indifference means picking at random."""
    ho, _ = results
    pf = ho.player_[ho.player_.group == "all"]
    z = pf[pf.system == "zero"]
    assert z.tau.isna().all()
    assert np.allclose(z.discordant.to_numpy(), 0.5)
    for h, d in pf.groupby("held_out"):
        zero = float(d[d.system == "zero"].dollars_lost.iloc[0])
        assert float(d[d.system == "true"].dollars_lost.iloc[0]) < zero
        assert float(d[d.system == "rapm"].dollars_lost.iloc[0]) < zero


def test_money_skill_is_the_share_of_the_misallocation_avoided(results):
    """`dollars_flat` is what a board with nothing to say would lose, so money_skill = 1 - lost / flat is 1
    for a perfect board, 0 for a flat one and negative for one that is worse than saying nothing."""
    ho, _ = results
    pf = ho.player_[ho.player_.group == "all"]
    assert np.allclose(pf[pf.system == "zero"].money_skill.to_numpy(), 0.0)
    a = pf.groupby("system").money_skill.mean()
    assert a.loc["true"] > a.loc["noisy"] > 0 and a.loc["true"] > a.loc["rapm"] > 0
    # every candidate's flat reference is the same number: it depends only on the truth
    assert pf.groupby("held_out").dollars_flat.nunique().max() == 1


def test_the_losses_ignore_where_a_board_puts_its_zero(results):
    """`predict_season` refits the intercept on the held-out season, so the team-game criterion cannot see a
    constant shift in a board.  Neither may this one, or it would score the choice of zero (average player
    vs replacement level) instead of the ordering -- and it would score it HARD, because dollars are a
    rating times possessions and an all-positive board orders players by minutes."""
    ho, _ = results
    pf = ho.player_[ho.player_.group == "all"].set_index(["held_out", "k", "system"])
    for col in ("tau", "tau_o", "tau_d", "top_k", "dollars_lost", "discordant"):
        a = pf.xs("true", level="system")[col]
        b = pf.xs("shifted", level="system")[col]
        assert np.allclose(a.to_numpy(), b.reindex(a.index).to_numpy()), col


# ------------------------------------------------------------------ within a roster, across rosters
def test_rank_is_within_roster_and_ignores_team_level_error(world):
    """`tau` runs over teammate pairs, so giving every player on a team the same wrong bonus cannot touch
    it -- the team mean is not what an ordering is for.  The league-wide `tau_league` does move, which is
    the part of it that was scoring "is this a good team" rather than "who on this team is better"."""
    _, ctx, _ = world
    truth = player_truth(ctx, ctx.design([2003], "pts"))
    rng = np.random.default_rng(5)
    bump = dict(zip(sorted(set(truth.df.team)), rng.normal(0, 6.0, truth.df.team.nunique())))
    shifted = truth.df.assign(o=truth.df.o + truth.df.team.map(bump).to_numpy())
    a = player_scores(Ratings(truth.df), truth, edges=EDGES, labels=LABELS, top_k=10).set_index("group")
    b = player_scores(Ratings(shifted), truth, edges=EDGES, labels=LABELS, top_k=10).set_index("group")
    assert a.loc["all", "tau"] == pytest.approx(1.0)
    assert b.loc["all", "tau"] == pytest.approx(1.0)          # teammates still ordered perfectly
    assert b.loc["all", "tau_league"] < 0.95                  # the league-wide order is wrecked
    assert b.loc["all", "dollars_lost"] > 0                   # and so are the trades, which are cross-team


def test_the_trade_loss_is_cross_team_and_ignores_teammate_pairs(world):
    """A trade is between rosters.  Reordering a team INTERNALLY costs rank and costs no dollars."""
    _, ctx, _ = world
    truth = player_truth(ctx, ctx.design([2003], "pts"))
    d = truth.df
    # reverse each roster's own ordering, leaving every player's value where it was across teams
    swapped = d.copy()
    for tm, g in d.groupby("team"):
        swapped.loc[g.index, "o"] = g.o.to_numpy()[np.argsort(np.argsort(-g.o.to_numpy()))]
    a = player_scores(Ratings(d), truth, edges=EDGES, labels=LABELS, top_k=10).set_index("group")
    b = player_scores(Ratings(swapped), truth, edges=EDGES, labels=LABELS, top_k=10).set_index("group")
    assert b.loc["all", "tau"] < a.loc["all", "tau"]          # the roster order is now wrong
    assert b.loc["all", "dollars_lost"] > 0


def test_the_bench_is_visible_to_the_player_loss(results, world):
    """The point of the whole thing: corrupt only the LOW-possession players and the player loss moves in
    the bucket where they live, which is the population the team-game criterion calls a tie."""
    sim, ctx, table = world
    wd = ctx.design([2003], "pts")
    truth = player_truth(ctx, wd)
    rng = np.random.default_rng(7)
    low = truth.df.poss < EDGES[1]
    assert low.sum() >= 5 and (~low).sum() >= 5
    corrupt = truth.df.assign(o=np.where(low, truth.df.o + rng.normal(0, 5.0, len(truth.df)), truth.df.o))
    s = player_scores(Ratings(corrupt), truth, edges=EDGES, labels=LABELS, top_k=10).set_index("group")
    assert s.loc["low", "tau"] < 0.9              # the bench is now wrong
    assert s.loc["high", "tau"] == pytest.approx(1.0)   # and the starters are untouched
