"""The swap test (src/eracoef/swaptest.py) on a synthetic league whose true ratings are known."""
import itertools

import numpy as np
import pandas as pd

from eracoef import swaptest as sw


def _league(seed=0, teams=4, players=8, games=240, per_period=6, poss=5, noise=1.2):
    """Stints of a league with known ratings: random lineups, five possessions a side per stint, points
    drawn around 1.1 a possession plus the ten players' ratings (offense adds, defense = points allowed)."""
    rng = np.random.default_rng(seed)
    roster = {t: np.arange(t * 100 + 1, t * 100 + 1 + players) for t in range(1, teams + 1)}
    every = np.concatenate(list(roster.values()))
    truth = pd.DataFrame({"player_id": every, "o": rng.normal(0, 3, len(every)), "d": rng.normal(0, 2, len(every))})
    o, d = dict(zip(truth.player_id, truth.o)), dict(zip(truth.player_id, truth.d))
    rows = []
    for g in range(games):
        h, a = rng.choice(np.arange(1, teams + 1), 2, replace=False)
        margin = 0
        for period in range(1, 5):
            for k in range(per_period):
                five_h = rng.choice(roster[h], 5, replace=False)
                five_a = rng.choice(roster[a], 5, replace=False)
                mu_h = 1.1 + (sum(o[p] for p in five_h) + sum(d[p] for p in five_a)) / 100.0
                mu_a = 1.1 + (sum(o[p] for p in five_a) + sum(d[p] for p in five_h)) / 100.0
                pts_h = poss * mu_h + rng.normal(0, noise * np.sqrt(poss))
                pts_a = poss * mu_a + rng.normal(0, noise * np.sqrt(poss))
                start = 720.0 - k * 720.0 / per_period
                rows.append(dict(game_id=f"g{g:04d}", phase="RS", period=period, start_clock=start,
                                 frac_rem=((4 - period) * 720.0 + start) / 2880.0, margin_h=margin, is_gt=False,
                                 neutral=False, poss_h=poss, poss_a=poss, pts_h=pts_h, pts_a=pts_a,
                                 home_team_id=h, away_team_id=a,
                                 **{f"h{i + 1}": int(p) for i, p in enumerate(five_h)},
                                 **{f"a{i + 1}": int(p) for i, p in enumerate(five_a)}))
                margin += int(round(pts_h - pts_a))
    return pd.DataFrame(rows), truth, roster


def _season(seed=0):
    stints, truth, roster = _league(seed)
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


def test_a_team_wide_constant_cannot_move_that_teams_swaps():
    season, truth, roster = _season()
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
            _, prd, _, own = sw._side_rates(season, pred, o, d)[side]
            gaps.append(prd[a] - prd[b])
            calls.append(own[a] - own[b])
        assert len(pr) > 50 and np.allclose(gaps[0], gaps[1]) and np.allclose(calls[0], calls[1])


def test_order_ignores_spread_and_gaps_does_not():
    season, truth, _ = _season()
    common = sw.plain_rapm(season, lam=50.0)
    wide = truth.assign(o=2 * truth.o, d=2 * truth.d)
    rated = set(truth.player_id)
    one, _ = sw.score_season(season, {"truth": truth}, rated, common, contexts=("none",))
    two, _ = sw.score_season(season, {"wide": wide}, rated, common, contexts=("none",))
    assert np.allclose(one.order.to_numpy(), two.order.to_numpy())
    assert np.all(two.gaps.to_numpy() > one.gaps.to_numpy())
    assert np.allclose(two.slope.to_numpy() * 2, one.slope.to_numpy(), rtol=1e-9)


def test_the_truth_beats_the_same_ratings_shuffled_inside_each_team():
    season, truth, roster = _season()
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


def test_pair_groups_split_the_pairs_by_the_rated_seasons_teams():
    season, truth, roster = _season()
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


def test_plain_rapm_recovers_the_truth_with_the_right_signs():
    season, truth, _ = _season()
    fit = sw.plain_rapm(season, lam=50.0).merge(truth, on="player_id", suffixes=("", "_true"))
    assert np.corrcoef(fit.o, fit.o_true)[0, 1] > 0.8
    assert np.corrcoef(fit.d, fit.d_true)[0, 1] > 0.6
