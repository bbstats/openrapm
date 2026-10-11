"""The swap test (src/openrapm/swap_test.py) on a synthetic league whose true ratings are known (conftest.py)."""
import itertools

import numpy as np
import pandas as pd
import pytest

from openrapm import swap_test as sw


def _season(league, seed=0):
    stints, truth, roster = league(seed)
    return sw.prepare(sw.season_rows(stints)), truth, roster


def test_fatigue_runs_while_on_court_and_resets():
    base = dict(game_id="g1", phase="RS", frac_rem=1.0, margin_h=0, is_gt=False, neutral=False,
                poss_h=1, poss_a=1, pts_h=1.0, pts_a=1.0, home_team_id=1, away_team_id=2)
    away = {f"a{i}": 10 + i for i in range(1, 6)}
    stints = pd.DataFrame([
        dict(base, period=1, start_clock=720.0, **{f"h{i}": i for i in range(1, 6)}, **away),
        dict(base, period=1, start_clock=660.0, h1=1, h2=2, h3=3, h4=4, h5=6, **away),    # 5 sits, 6 checks in
        dict(base, period=1, start_clock=600.0, **{f"h{i}": i for i in range(1, 6)}, **away),  # 5 back, rested
        dict(base, period=2, start_clock=720.0, **{f"h{i}": i for i in range(1, 6)}, **away),
    ])
    home, away_f = sw.fatigue(stints)
    # stint lengths 60, 60, 600 (the last runs to the end of the period) and 720; clocks read at the START,
    # so 1-4 have 60 s behind them in the second stint and 120 s in the third, and 5 comes back at zero
    assert np.allclose(home, [0.0, (4 * 60) / 300.0, (4 * 120) / 300.0, 0.0])
    assert np.allclose(away_f, [0.0, 1.0, 2.0, 0.0])                # a new period starts every clock at zero


def test_pairs_are_one_swap_on_one_team_each_once():
    rng = np.random.default_rng(1)
    lineups = {tuple(sorted(rng.choice(np.arange(team * 10, team * 10 + 7), 5, replace=False))) + (team,)
               for team in (1, 2) for _ in range(40)}
    lineups = sorted(lineups)
    team = np.array([x[-1] for x in lineups])
    five = np.array([x[:5] for x in lineups])
    got = sw.swap_pairs(team, five)
    want = {(i, j) for i, j in itertools.combinations(range(len(lineups)), 2)
            if team[i] == team[j] and len(set(five[i]) & set(five[j])) == 4}
    assert set(zip(got.lineup_a, got.lineup_b)) == want and len(got) == len(want)
    for r in got.itertuples():
        assert r.fifth_a in five[r.lineup_a] and r.fifth_a not in five[r.lineup_b]
        assert r.fifth_b in five[r.lineup_b] and r.fifth_b not in five[r.lineup_a]


def test_a_team_wide_constant_cannot_move_that_teams_swaps(lineup_league):
    season, truth, roster = _season(lineup_league)
    shifted = truth.copy()
    mine = shifted.player_id.isin(roster[1])
    shifted.loc[mine, "o"] += 3.0
    shifted.loc[mine, "d"] -= 2.0
    for side in sw.SIDES:
        pr = season.pairs[side]
        pr = pr[pr.team == 1]
        a, b = pr.lineup_a.to_numpy(), pr.lineup_b.to_numpy()
        gaps, calls = [], []
        for table in (truth, shifted):
            o, d = sw.aligned(season.ids, table)
            pred, _ = sw.predict(season, o, d, context="none")
            _, prd, _, own = sw.side_rates(season, pred, o, d)[side]
            gaps.append(prd[a] - prd[b])
            calls.append(own[a] - own[b])
        assert len(pr) > 50 and np.allclose(gaps[0], gaps[1]) and np.allclose(calls[0], calls[1])


def test_order_ignores_spread_and_gaps_does_not(lineup_league):
    season, truth, _ = _season(lineup_league)
    common = sw.plain_rapm(season, lam=50.0)
    wide = truth.assign(o=2 * truth.o, d=2 * truth.d)
    rated = set(truth.player_id)
    one, _ = sw.score_season(season, {"truth": truth}, rated, common, contexts=("none",))
    two, _ = sw.score_season(season, {"wide": wide}, rated, common, contexts=("none",))
    assert np.allclose(one.order.to_numpy(), two.order.to_numpy())
    assert np.all(two.gaps.to_numpy() > one.gaps.to_numpy())
    assert np.allclose(two.slope.to_numpy() * 2, one.slope.to_numpy(), rtol=1e-9)


def test_the_truth_beats_the_same_ratings_shuffled_inside_each_team(lineup_league):
    season, truth, roster = _season(lineup_league)
    rng = np.random.default_rng(7)
    shuffled = truth.copy()
    for players in roster.values():
        rows = shuffled.index[shuffled.player_id.isin(players)]
        shuffled.loc[rows, ["o", "d"]] = shuffled.loc[rng.permutation(rows), ["o", "d"]].to_numpy()
    scores, _ = sw.score_season(season, {"truth": truth, "shuffled": shuffled}, set(truth.player_id),
                                sw.plain_rapm(season, lam=50.0), contexts=("full", "home"))
    for (side, context), s in scores.groupby(["side", "context"]):
        s = s.set_index("ranking")
        assert s.order["truth"] > s.order["shuffled"] + 1.0, (side, context)
        assert s.gaps["truth"] < s.gaps["shuffled"], (side, context)


def test_pair_groups_split_the_pairs_by_the_rated_seasons_teams(lineup_league):
    season, truth, roster = _season(lineup_league)
    team_then = {int(p): t for t, players in roster.items() for p in players}
    team_then[101] = 99                                    # one player of team 1 was elsewhere the year before
    scores, _ = sw.score_season(season, {"truth": truth}, set(truth.player_id), sw.plain_rapm(season, lam=50.0),
                                contexts=("none",), team_then=team_then)
    for side, s in scores.groupby("side"):
        s = s.set_index("group")
        assert s.pairs["all"] == s.pairs["teammates then"] + s.pairs["not teammates then"]
        assert np.isclose(s.information["all"], s.information["teammates then"] + s.information["not teammates then"])
    pr = season.pairs["net"]
    moved = ((pr.fifth_a == 101) | (pr.fifth_b == 101)).sum()
    assert scores[(scores.side == "net") & (scores.group == "not teammates then")].pairs.iloc[0] == moved


def test_plain_rapm_recovers_the_truth_with_the_right_signs(lineup_league):
    season, truth, _ = _season(lineup_league)
    fit = sw.plain_rapm(season, lam=50.0).merge(truth, on="player_id", suffixes=("", "_true"))
    assert np.corrcoef(fit.o, fit.o_true)[0, 1] > 0.8
    assert np.corrcoef(fit.d, fit.d_true)[0, 1] > 0.6


def test_a_positive_good_table_is_read_in_raw_sign():
    table = pd.DataFrame({"player_id": [1, 2], "season": [2026, 2026], "rating_off": [2.0, -1.0],
                          "rating_def": [1.5, -0.5], "prior_off": [1.0, 0.0], "prior_def": [0.5, 0.0]})
    raw = sw.ratings_in_raw_sign(table)
    assert raw.o.tolist() == [2.0, -1.0] and raw.d.tolist() == [-1.5, 0.5]
    assert sw.ratings_in_raw_sign(table, "prior").d.tolist() == [-0.5, -0.0]
    with pytest.raises(ValueError, match="lacks"):
        sw.ratings_in_raw_sign(table.drop(columns="rating_def"))
