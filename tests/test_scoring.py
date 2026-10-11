"""How ratings are scored on games they never saw (src/openrapm/scoring.py), against simulated seasons with a known
truth, and how a miss is reported.

The simulator runs with persistent talent and roster turnover, so the true impacts of one season carry information
about the next and players appear with new teammates: the two things the year-over-year test is built to see.
"""
import numpy as np
import pandas as pd
import pytest

from openrapm.design import FEATURES, build_design
from openrapm.scoring import (
    MAE_SCALE,
    QUALITY_LABELS,
    Prediction,
    Ratings,
    armse,
    game_bench_share_split,
    keeps_whole_team_games,
    movers_split,
    paired,
    pooled,
    predict_season,
    quality_tier,
    quality_tiers,
    replacement_level,
    score,
)
from openrapm.simulate import simulate

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500, "box_counts": FEATURES}


def test_armse_is_the_root_mean_square_on_the_mean_absolute_scale():
    assert MAE_SCALE == pytest.approx(np.sqrt(2.0 / np.pi))
    assert armse(100.0) == pytest.approx(10.0 * MAE_SCALE)
    sample = np.random.default_rng(0).normal(0.0, 3.0, 400_000)
    assert armse(float((sample ** 2).mean())) == pytest.approx(float(np.abs(sample).mean()), rel=0.01)


@pytest.fixture(scope="module")
def season():
    """One simulated season's design, the true impacts of its players and their main teams."""
    sim = simulate(n_seasons=2, n_teams=8, players_per_team=10, games_per_season=60, stints_per_game=(20, 30),
                   rho=0.85, turnover=0.3, seed=11, eps_var=0.2, leak=False)
    st, box, truth = sim["stints"], sim["box"], sim["truth"]["ps"]
    design = build_design(st[st.season == 2002], box[box.season == 2002], FEATURES, CFG)
    table = truth.rename(columns={"impact_O": "o", "impact_D": "d"})
    # the three deepest players of each team are barely seen (100 possessions), so team-games differ in how
    # much of them the bench played
    table = table.assign(poss=np.where(table.player_id % 1000 > 7, 100.0, 1000.0))
    teams = {int(s): dict(zip(g.player_id.astype(int), g.team.astype(int))) for s, g in truth.groupby("season")}
    return design, table, teams


def _ratings(table, season, scale=1.0, noise=0.0):
    rows = table[table.season == season][["player_id", "o", "d", "poss"]].reset_index(drop=True)
    rng = np.random.default_rng(3)
    return Ratings(rows.assign(o=scale * rows.o + rng.normal(0, noise, len(rows)) if noise else scale * rows.o,
                               d=scale * rows.d + rng.normal(0, noise, len(rows)) if noise else scale * rows.d))


def test_the_true_impacts_beat_noisy_ones_and_beat_none(season):
    design, table, _ = season
    true = score(predict_season(_ratings(table, 2002), design))
    noisy = score(predict_season(_ratings(table, 2002, noise=1.0), design))
    zero = score(predict_season(_ratings(table, 2002, scale=0.0), design))
    assert true["mse"] < noisy["mse"] < zero["mse"]
    assert true["tg"] < zero["tg"]
    assert true["mse"] < true["base"] and zero["mse"] == pytest.approx(zero["base"])
    assert true["covered"] == 1.0


def test_the_scale_reads_amplitude(season):
    """Ratings twice too wide want exactly half the multiplier the true ones want, on each side."""
    design, table, _ = season
    true = score(predict_season(_ratings(table, 2002), design))
    double = score(predict_season(_ratings(table, 2002, scale=2.0), design))
    assert abs(true["scale_off"] - 1.0) < 0.2 and abs(true["scale_def"] - 1.0) < 0.2
    assert double["scale_off"] == pytest.approx(true["scale_off"] / 2.0)
    assert double["scale_def"] == pytest.approx(true["scale_def"] / 2.0)
    assert double["calib_side"] == pytest.approx(true["calib_side"])


def test_a_player_with_no_row_scores_the_fill(season):
    design, table, _ = season
    rows = table[table.season == 2002][["player_id", "o", "d", "poss"]].reset_index(drop=True)
    some = Ratings(rows.iloc[: len(rows) // 2], fill_o=-1.0, fill_d=0.5)
    p = predict_season(some, design)
    missing = ~np.isin(p.rat.player_id, rows.player_id.iloc[: len(rows) // 2])
    assert (p.rat.o[missing] == -1.0).all() and (p.rat.d[missing] == 0.5).all() and (p.rat.poss[missing] == 0).all()
    assert score(p)["covered"] == pytest.approx(0.5, abs=0.05)


def test_the_two_splits_partition_the_rows(season):
    design, table, teams = season
    p = predict_season(_ratings(table, 2001), design)
    whole = score(p)["n"]
    movers = movers_split(design, teams[2002], teams[2001])
    shares = game_bench_share_split(p, design)
    for labels in (movers, shares):
        assert sum(score(p, labels == g)["n"] for g in pd.unique(labels)) == pytest.approx(whole)
    assert {"no movers", "1-2 movers", "3+ movers"} & set(movers)
    assert set(shares) <= {"0%-5%", "5%-15%", "15%-30%", "30%+"} and len(set(shares)) >= 2


def test_the_bench_share_split_recombines_to_the_pooled_team_game_score(season):
    """It labels whole team-games, so its groups' team-game errors add up to the pooled one."""
    design, table, _ = season
    p = predict_season(_ratings(table, 2001), design)
    whole = score(p)
    shares = game_bench_share_split(p, design)
    parts = [score(p, shares == g) for g in pd.unique(shares)]
    assert sum(s["tg"] * s["tg_n"] for s in parts) / whole["tg_n"] == pytest.approx(whole["tg"])


def test_nobody_moves_when_the_teams_are_the_same(season):
    design, _, teams = season
    assert set(movers_split(design, teams[2002], teams[2002])) == {"no movers"}
    assert set(movers_split(design, teams[2002], {})) == {"no movers"}


def test_team_game_score_is_nan_when_the_mask_cuts_a_team_game():
    """`tg` sums a team's points over its rows in a game.  On a mask that keeps some of a team-game's stints
    and not others, that sum is a PARTIAL point total scored against a level fitted on complete ones -- not
    the test restricted to those rows.  It read three times the pooled error on the 2026-09 ratings and
    flipped the sign of a comparison before this guard existed.

    Whole team-games kept: `tg` is the real number and the groups recombine.  Cut: NaN, read `mse` instead.
    """
    n = 12
    rng = np.random.default_rng(0)
    game_idx = np.repeat(np.arange(3), 4)
    is_home_off = np.tile(np.array([0, 0, 1, 1]), 3)
    p = Prediction(y=rng.normal(size=n), w=np.ones(n), poss=np.full(n, 50.0),
                   home=np.zeros(n), c_off=np.zeros(n), c_def=np.zeros(n),
                   pred=rng.normal(size=n), base_pred=np.zeros(n), game_idx=game_idx,
                   is_home_off=is_home_off, rat=pd.DataFrame({"poss": [1.0, 2.0]}), A=np.ones((n, 1)))

    whole = np.isin(game_idx, [0, 1])                  # both teams, every stint, of two games
    assert keeps_whole_team_games(p, np.flatnonzero(whole))
    assert np.isfinite(score(p, whole)["tg"])

    cut = np.zeros(n, dtype=bool)                      # one stint out of a team-game's two
    cut[[0, 2, 4]] = True
    assert not keeps_whole_team_games(p, np.flatnonzero(cut))
    assert np.isnan(score(p, cut)["tg"])
    assert np.isfinite(score(p, cut)["mse"]), "the stint-level score stays valid on any mask"
    assert np.isfinite(score(p)["tg"]), "no mask at all is still scored"


# ------------------------------------------------------------------- the pooled and paired summaries
def _results():
    """Three systems over four scored seasons with known errors: `better` is 1 lower than `ref` in three."""
    rows = []
    for season, (ref, better, worse) in zip(range(2001, 2005), [(10, 9, 12), (11, 10, 12), (12, 11, 13),
                                                                 (10, 10.5, 11)]):
        for system, mse in (("ref", ref), ("better", better), ("worse", worse)):
            rows.append(dict(held_out=season, k=1, lam=0.0, system=system, split="all", group="all", n=100.0,
                             mse=float(mse), base=20.0, calib_side=float(mse), scale_off=1.0, scale_def=1.0,
                             covered=1.0, tg=float(mse), tg_base=20.0, tg_n=50.0))
    return pd.DataFrame(rows)


def test_pooled_weights_by_rows_and_reports_on_the_mean_absolute_scale():
    p = pooled(_results()).set_index("system")
    assert p.loc["ref", "mse"] == pytest.approx(10.75)
    assert p.loc["ref", "vs_no_ratings"] == pytest.approx(1.0 - 10.75 / 20.0)
    assert p.loc["ref", "game_armse"] == pytest.approx(np.sqrt(10.75) * MAE_SCALE)
    assert p.loc["ref", "share"] == 1.0 and p.loc["ref", "seasons"] == 4


def test_paired_is_the_mean_difference_over_its_standard_error():
    t = paired(_results(), "ref", "tg").set_index("system")
    diff = np.array([-1.0, -1.0, -1.0, 0.5])
    assert t.loc["better", "mean_diff"] == pytest.approx(diff.mean())
    assert t.loc["better", "se"] == pytest.approx(diff.std(ddof=1) / 2.0)
    assert t.loc["better", "z"] == pytest.approx(diff.mean() / (diff.std(ddof=1) / 2.0))
    assert t.loc["better", "wins"] == 3 and t.loc["worse", "wins"] == 0
    assert "ref" not in t.index


# ------------------------------------------------------------------------------ the replacement level
# The year-over-year test's fill and the trade set's (`trade_set.replacement_fill`) are this one rule.
def _three_players(poss):
    return pd.DataFrame({"player_id": [1, 2, 3], "o": [-2.0, -4.0, 8.0], "d": [1.0, 3.0, -9.0],
                         "poss": list(poss)})


def test_the_replacement_level_is_the_low_exposure_mean():
    fill_o, fill_d = replacement_level(_three_players([100.0, 300.0, 5000.0]), max_poss=500.0, shrink=0.25)
    assert fill_o == pytest.approx(0.25 * np.average([-2.0, -4.0], weights=[100.0, 300.0]))
    assert fill_d == pytest.approx(0.25 * np.average([1.0, 3.0], weights=[100.0, 300.0]))
    assert replacement_level(_three_players([100.0, 300.0, 5000.0]))[0] == pytest.approx(-3.5)


def test_with_no_low_exposure_players_the_fill_is_the_average_player():
    assert replacement_level(_three_players([900.0, 1200.0, 5000.0]), shrink=0.25) == (0.0, 0.0)


def test_the_trade_set_uses_the_same_rule():
    from openrapm.trade_set import replacement_fill
    players = _three_players([100.0, 300.0, 5000.0])
    assert replacement_fill(players) == replacement_level(players, shrink=0.25)


# ---------------------------------------------------------------------------------- the quality tiers
def test_quality_tiers_follow_the_rank_within_each_season():
    assert list(quality_tier(np.array([1, 30, 31, 90, 91, 300, 301]))) == [
        "top 30", "top 30", "31-90", "31-90", "91-150", "151-300", "301+"]
    table = pd.DataFrame({"player_id": np.arange(400), "season": 2026, "rating_total": -np.arange(400.0)})
    tiers = quality_tiers(table)
    assert tiers.loc[(0, 2026)] == "top 30" and tiers.loc[(399, 2026)] == "301+"
    assert set(tiers) == set(QUALITY_LABELS)
