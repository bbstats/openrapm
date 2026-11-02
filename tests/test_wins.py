"""Wins over starter average (src/openrapm/wins.py) on hand-made stints and on leagues whose truth is planted."""
import numpy as np
import pandas as pd
import pytest

from openrapm import swap_test as sw
from openrapm import wins
from openrapm.simulate import simulate

HOME_FIVE = dict(h1=1, h2=2, h3=3, h4=4, h5=5)
AWAY_FIVE = dict(a1=11, a2=12, a3=13, a4=14, a5=15)
SLOTS = [*sw.HOME, *sw.AWAY]


def _stint(**values) -> dict:
    return dict(dict(game_id="g1", phase="RS", period=1, start_clock=720.0, margin_h=0, is_gt=False, neutral=False,
                     half="A", poss_h=10, poss_a=10, pts_h=10.0, pts_a=10.0, home_team_id=100, away_team_id=200,
                     **HOME_FIVE, **AWAY_FIVE), **values)


def _brownian_games(seed=0, n_games=800, n_teams=16, sigma=12.0, home_edge=3.0, strength_sd=4.0) -> pd.DataFrame:
    """Games whose margin moves by a normal step each stint: 24 stints of two minutes, a level of home edge plus the two
    teams' strength difference per full game, a variance of sigma squared per full game."""
    rng = np.random.default_rng(seed)
    strengths = rng.normal(0.0, strength_sd, n_teams)
    step = 120.0 / 2880.0
    rows = []
    for game in range(n_games):
        home, away = rng.choice(n_teams, 2, replace=False)
        moves = rng.normal((home_edge + strengths[home] - strengths[away]) * step, sigma * np.sqrt(step), 24)
        margin = np.concatenate([[0.0], np.cumsum(moves)[:-1]])
        for number, (move, level) in enumerate(zip(moves, margin)):
            period, clock = 1 + number // 6, 720.0 - 120.0 * (number % 6)
            rows.append(dict(game_id=f"g{game}", phase="RS", period=period, start_clock=clock, margin_h=level,
                             is_gt=False, neutral=False, half="AB"[game % 2], poss_h=4, poss_a=4,
                             pts_h=max(move, 0.0), pts_a=max(-move, 0.0),
                             home_team_id=int(home), away_team_id=int(away),
                             **{f"h{i + 1}": 1000 * (int(home) + 1) + i for i in range(5)},
                             **{f"a{i + 1}": 1000 * (int(away) + 1) + i for i in range(5)}))
    return pd.DataFrame(rows)


def _league(seed=3) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """A simulated season's stints, their seconds, each player's ratings and pace effect."""
    stints = simulate(n_seasons=1, n_teams=6, players_per_team=9, games_per_season=60, stints_per_game=(20, 30),
                      seed=seed)["stints"].copy()
    stints["home_team_id"], stints["away_team_id"] = stints.h1 // 1000, stints.a1 // 1000
    seconds = sw.stint_durations(stints)
    player_ids = wins.player_index(stints)[0]
    rng = np.random.default_rng(seed)
    offense, defense = rng.normal(0.0, 2.0, len(player_ids)), rng.normal(0.0, 1.5, len(player_ids))
    ratings = pd.DataFrame({"player_id": player_ids, "rating_off": offense, "rating_def": defense,
                            "rating_total": offense + defense})
    pace = pd.DataFrame({"player_id": player_ids, "pace_effect": rng.normal(0.0, 1.0, len(player_ids))})
    return stints, seconds, ratings, pace


# ------------------------------------------------------------------------------------------- the win model
def test_share_left_counts_regulation_then_an_overtimes_own_clock_and_never_reaches_zero():
    start, middle = wins.share_left([4, 2, 5, 4], [300.0, 300.0, 120.0, 10.0], [60.0, 0.0, 60.0, 10.0])
    assert start * 2880 == pytest.approx([300.0, 1740.0, 120.0, 10.0])
    assert middle * 2880 == pytest.approx([270.0, 1740.0, 90.0, 5.0])
    assert wins.share_left([4], [0.0], [0.0])[1] * 2880 == pytest.approx([1.0])


def test_the_games_margin_is_what_its_stints_add_up_to_and_a_tie_is_there_to_be_counted():
    stints = pd.DataFrame([_stint(pts_h=12.0, pts_a=9.0), _stint(pts_h=7.0, pts_a=8.0),
                           _stint(game_id="g2", pts_h=5.0, pts_a=5.0)])
    results = wins.game_results(stints)
    assert results.game_id.tolist() == ["g1", "g2"] and results.margin.tolist() == [2.0, 0.0]


def test_strength_leaves_the_game_itself_out():
    results = pd.DataFrame({"game_id": ["a", "b", "c"], "home_team_id": [1, 1, 2], "away_team_id": [2, 3, 3],
                            "neutral": False, "margin": [30.0, 10.0, -4.0]})
    # team 1 in game a: its other game is +10; team 2 there: its other game, at home against 3, is -4
    assert wins.strength_gap(results)[0] == pytest.approx(10.0 - (-4.0))
    # team 1 in game b: +30; team 3 there: +4, its win at team 2
    assert wins.strength_gap(results)[1] == pytest.approx(30.0 - 4.0)


def test_the_win_model_reads_how_much_of_a_lead_lasts():
    """A margin pulled back toward zero: the share of a lead that lasts falls with the time left, and sits near 1 at
    the end.  With no pull it is 1 throughout."""
    rng = np.random.default_rng(4)
    rows = []
    for game in range(1500):
        margin = 0.0
        for number in range(24):
            move = rng.normal(0.0, 2.5) - 0.03 * margin
            rows.append(dict(game_id=f"g{game}", phase="RS", period=1 + number // 6,
                             start_clock=720.0 - 120.0 * (number % 6), margin_h=margin, is_gt=False, neutral=False,
                             half="A", poss_h=4, poss_a=4,
                             pts_h=max(move, 0.0), pts_a=max(-move, 0.0), home_team_id=game % 8,
                             away_team_id=(game + 1) % 8))
            margin += move
    pulled = pd.DataFrame(rows)
    model = wins.fit_win_model(pulled, sw.stint_durations(pulled), 2001)
    assert wins.lead_kept(1.0 / 24.0, model) == pytest.approx(0.97, abs=0.05)
    assert wins.lead_kept(0.5, model) == pytest.approx(0.97 ** 12, abs=0.08)
    plain = _brownian_games(strength_sd=0.0)
    model = wins.fit_win_model(plain, sw.stint_durations(plain), 2001)
    assert model.lead_kept_at_end == pytest.approx(1.0, abs=0.05) and abs(model.lead_kept_per_game) < 0.2


def test_the_win_model_recovers_a_planted_home_edge_and_spread():
    stints = _brownian_games()
    seconds = sw.stint_durations(stints)
    model = wins.fit_win_model(stints, seconds, 2001)
    assert model.n_games == 800 and model.n_ties == 0
    assert model.home_edge == pytest.approx(3.0, abs=1.0)
    assert 0.7 < model.strength_slope < 1.1            # a season's net is a noisy strength: under 1
    # the rest of a game from a stint's start, read at its middle: sigma x sqrt(share left at the start)
    for seconds_left in (60.0, 1500.0, 2820.0):
        expected = 12.0 * np.sqrt((seconds_left + 60.0) / 2880.0)
        assert wins.spread_at(seconds_left / 2880.0, model) == pytest.approx(expected, rel=0.12)


def test_the_spread_never_shrinks_with_more_time_left():
    stints = _brownian_games(n_games=60)
    model = wins.fit_win_model(stints, sw.stint_durations(stints), 2001)
    assert np.all(np.diff(model.spread) >= 0.0)


def test_a_game_of_margin_zero_is_left_out_and_counted():
    stints = _brownian_games(n_games=40)
    tied = stints.assign(game_id="tied", pts_h=1.0, pts_a=1.0, margin_h=0.0).head(24)
    model = wins.fit_win_model(pd.concat([stints, tied], ignore_index=True),
                               sw.stint_durations(pd.concat([stints, tied], ignore_index=True)), 2001)
    assert model.n_games == 40 and model.n_ties == 1


def _model(edge=0.0, spread=12.0, kept=1.0) -> wins.WinModel:
    """A spread of `spread` x sqrt(share left), read on a fine grid; `kept` of a lead lasts at any time."""
    seconds_left = np.linspace(1.0, 2880.0, 400)
    return wins.WinModel(2001, edge, 1.0, kept, 0.0, seconds_left, spread * np.sqrt(seconds_left / 2880.0), 1, 0)


def test_leverage_is_highest_late_and_close_and_nothing_once_a_game_is_decided():
    model = _model()
    tied_late, tied_early = wins.leverage(0.0, 60.0 / 2880.0, model), wins.leverage(0.0, 1.0, model)
    assert tied_late == pytest.approx(0.228, abs=0.01)        # a spread of 1.73 with a minute left
    assert tied_early == pytest.approx(0.0331, abs=0.001)     # the tip-off: 1 / (12 x sqrt(2 pi))
    assert wins.leverage(20.0, 60.0 / 2880.0, model) < 1e-6
    assert wins.leverage(5.0, 0.1, model) == pytest.approx(wins.leverage(-5.0, 0.1, model))     # either team's point
    assert wins.leverage(0.0, 1.0 / 2880.0, model) <= 0.5


def test_a_neutral_site_takes_the_home_edge_away():
    model = _model(edge=6.0)
    assert wins.win_probability(0.0, 1.0, model) == pytest.approx(0.6915, abs=0.001)
    assert wins.win_probability(0.0, 1.0, model, neutral=True) == pytest.approx(0.5)


def test_a_lead_that_only_partly_lasts_is_worth_less_of_the_win_and_each_point_too():
    full, kept = _model(), _model(kept=0.5)
    assert wins.win_probability(6.0, 0.25, kept) == pytest.approx(wins.win_probability(3.0, 0.25, full))
    assert wins.leverage(0.0, 0.25, kept) < wins.leverage(0.0, 0.25, full)


def test_relative_leverage_averages_one_over_the_possessions():
    stints, seconds, _, _ = _league()
    relative = wins.relative_leverage(stints, seconds, wins.fit_win_model(stints, seconds, 2001))
    assert np.average(relative, weights=stints.poss_h + stints.poss_a) == pytest.approx(1.0)
    assert relative.max() > 1.5 and relative.min() < 0.5


def test_the_leverage_of_every_point_scored_adds_up_to_the_wins():
    """The win model's own test: a game's leverage times its points moves from the tip-off's chance to the result, so
    across games the sum follows the outcome with a slope of 1."""
    stints = _brownian_games(strength_sd=0.0, n_games=1500, home_edge=0.0)
    seconds = sw.stint_durations(stints)
    model = wins.fit_win_model(stints, seconds, 2001)
    left = wins.share_left(stints.period, stints.start_clock, seconds)[1]
    lever = wins.leverage(stints.margin_h.to_numpy(dtype=float), left, model)
    summed = stints.assign(value=lever * (stints.pts_h - stints.pts_a)).groupby("game_id").value.sum()
    won = (wins.game_results(stints).set_index("game_id").margin > 0.0).astype(float).reindex(summed.index)
    assert np.polyfit(summed, won, 1)[0] == pytest.approx(1.0, abs=0.2)


def test_the_standings_read_about_thirty_points_a_win_on_a_league_of_about_that_size():
    stints = _brownian_games(n_games=2000)
    assert wins.net_points_per_win(wins.game_results(stints)) == pytest.approx(30.0, abs=8.0)


def test_rating_points_per_win_is_the_slope_of_records_on_the_ratings_points():
    results_margin = {"g1": 10.0, "g2": 5.0, "g3": -5.0, "g4": 3.0}
    teams = {"g1": (1, 2), "g2": (1, 3), "g3": (2, 3), "g4": (3, 2)}
    stints = pd.DataFrame([_stint(game_id=game, pts_h=100.0 + value, pts_a=100.0, poss_h=100, poss_a=100,
                                  home_team_id=teams[game][0], away_team_id=teams[game][1])
                           for game, value in results_margin.items()])
    margin = stints.pts_h.to_numpy() - stints.pts_a.to_numpy()      # per 100 on 100 possessions: the points themselves
    got = wins.rating_points_per_win(stints, np.ones(4), margin)
    assert got == pytest.approx(wins.net_points_per_win(wins.game_results(stints)))
    assert wins.rating_points_per_win(stints, np.ones(4), margin * 0.5) == pytest.approx(got * 0.5)


# ------------------------------------------------------------------------------------------- the pace ridge
def _pace_league(seed=1, noise=2.0):
    """Stints whose pace is 100 plus the ten players' planted effects plus noise; every player both home and away."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1, 121)
    planted = pd.Series(rng.normal(0.0, 1.5, len(ids)), index=ids)
    rows = []
    for game in range(1500):
        ten = rng.choice(ids, 10, replace=False)
        seconds = float(rng.integers(60, 400))
        possessions = max(1, round((100.0 + planted[ten].sum() + rng.normal(0.0, noise)) * seconds / 2880.0))
        rows.append(dict(game_id=f"g{game}", period=1, start_clock=seconds, margin_h=0, is_gt=False, neutral=False,
                         poss_h=possessions, poss_a=possessions, pts_h=0.0, pts_a=0.0,
                         **{slot: int(player) for slot, player in zip(SLOTS, ten)}))
    return pd.DataFrame(rows), planted


def test_the_pace_ridge_recovers_planted_effects():
    stints, planted = _pace_league()
    effects = wins.pace_effects(stints, sw.stint_durations(stints), penalty=300.0, min_seconds=15.0)
    assert np.corrcoef(effects.pace_effect, planted.reindex(effects.player_id))[0, 1] > 0.8


def test_a_player_speeds_up_a_stint_from_either_bench():
    stints, _ = _pace_league()
    seconds = sw.stint_durations(stints)
    swapped = stints.rename(columns={**dict(zip(sw.HOME, sw.AWAY)), **dict(zip(sw.AWAY, sw.HOME))})
    one = wins.pace_effects(stints, seconds, 300.0, 15.0)
    other = wins.pace_effects(swapped, seconds, 300.0, 15.0)
    assert one.pace_effect.to_numpy() == pytest.approx(other.pace_effect.to_numpy(), abs=1e-8)


def test_a_stint_under_the_minimum_or_without_a_clock_leaves_the_fit_and_divides_nothing():
    stints = pd.DataFrame([_stint(start_clock=720.0), _stint(start_clock=700.0), _stint(start_clock=695.0),
                           _stint(start_clock=695.0), _stint(start_clock=300.0)])
    seconds = sw.stint_durations(stints)
    assert seconds.tolist() == [20.0, 5.0, 0.0, 395.0, 300.0]
    rows = wins.pace_rows(stints, seconds, 15.0)
    assert rows.weight.tolist() == [20.0, 395.0, 300.0]
    assert rows.target.tolist() == pytest.approx([2880.0 * 10.0 / 20.0, 2880.0 * 10.0 / 395.0, 2880.0 * 10.0 / 300.0])
    assert rows.players.shape == (3, 10) and rows.context.shape == (3, len(wins.PACE_CONTEXT))


def test_the_context_keeps_blowouts_and_period_ends_out_of_the_players():
    stints = pd.DataFrame([_stint(start_clock=720.0, margin_h=-40, is_gt=True), _stint(start_clock=100.0, margin_h=5)])
    rows = wins.pace_rows(stints, np.array([620.0, 100.0]), 15.0)
    assert rows.context[0].tolist() == [1.0, 1.0, 25.0, 25.0, 0.0]         # the margin clipped, the whole game left
    assert rows.context[1].tolist() == pytest.approx([1.0, 0.0, 5.0, 5.0 * 2260.0 / 2880.0, 1.0])


def test_held_back_error_prefers_the_truth_and_a_player_it_never_saw_reads_zero():
    stints, _ = _pace_league()
    seconds = sw.stint_durations(stints)
    first, second = stints.index < 1000, stints.index >= 1000
    fit_rows = wins.pace_rows(stints[first], seconds[first], 15.0)
    score_rows = wins.pace_rows(stints[second], seconds[second], 15.0)
    good = wins.pace_error(wins.solve_pace(fit_rows, 300.0), score_rows)
    overfit = wins.pace_error(wins.solve_pace(fit_rows, 1e-3), score_rows)
    flat = wins.pace_error(wins.solve_pace(fit_rows, 1e12), score_rows)
    assert good < flat and good < overfit
    short = wins.PaceFit(fit_rows.player_ids[:5], np.zeros(5), wins.solve_pace(fit_rows, 300.0).context)
    assert np.isfinite(wins.pace_error(short, score_rows))


# ------------------------------------------------------------------------------------------- the wins
def test_starts_are_opening_stints_and_minutes_are_the_seconds_played():
    stints = pd.DataFrame([_stint(start_clock=720.0), _stint(start_clock=660.0, h5=6),
                           _stint(game_id="g2", start_clock=720.0, h1=6), _stint(game_id="g2", start_clock=120.0)])
    table = wins.starts_and_minutes(stints, sw.stint_durations(stints)).set_index("player_id")
    assert table.loc[6, "starts_rs"] == 1 and table.loc[2, "starts_rs"] == 2 and table.loc[1, "starts_rs"] == 1
    assert table.loc[5, "minutes_rs"] == pytest.approx((60.0 + 600.0 + 120.0) / 60.0)
    assert table.loc[6, "minutes_rs"] == pytest.approx((660.0 + 600.0) / 60.0)


def _ranked(ids, starts, minutes, ratings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = pd.DataFrame({"player_id": ids, "starts_rs": starts, "minutes_rs": minutes})
    rated = pd.DataFrame({"player_id": ids, "rating_off": ratings, "rating_def": 1.0,
                          "rating_total": np.add(ratings, 1.0)})
    return table, rated, pd.DataFrame({"player_id": ids, "pace_effect": np.multiply(ratings, 10.0)})


def test_the_bar_is_the_most_games_started_then_the_most_minutes():
    table, rated, pace = _ranked([1, 2, 3, 4], [50, 50, 49, 80], [100.0, 900.0, 3000.0, 10.0], [1.0, 2.0, 40.0, 4.0])
    bar = wins.starter_bar(rated, table, pace, 2)                    # player 4 on starts, player 2 on minutes
    assert bar.off == pytest.approx(3.0) and bar.defense == pytest.approx(1.0) and bar.total == pytest.approx(4.0)
    assert bar.pace == pytest.approx(30.0) and bar.n == 2 and bar.fewest_starts == 50


def test_the_bar_takes_everyone_in_a_seasons_first_days_and_a_starter_without_a_pace_reads_zero():
    table, rated, pace = _ranked([1, 2], [1, 1], [20.0, 10.0], [1.0, 3.0])
    bar = wins.starter_bar(rated, table, pace[pace.player_id == 1], 150)
    assert bar.n == 2 and bar.off == pytest.approx(2.0) and bar.pace == pytest.approx(5.0)


def test_the_expected_margin_is_the_ten_ratings_and_the_home_edge_with_none_at_a_neutral_site():
    ratings = pd.DataFrame({"player_id": [*HOME_FIVE.values(), *AWAY_FIVE.values()],
                            "rating_total": [1.0] * 5 + [-1.0] * 5})
    stints = pd.DataFrame([_stint(pts_h=11.0, pts_a=10.0), _stint(neutral=True, pts_h=10.0, pts_a=10.0)])
    margin, edge = wins.expected_margins(stints, ratings)
    assert edge == pytest.approx(10.0 - 10.0)                        # +10 per 100 actual, +10 from the ratings
    stints = pd.DataFrame([_stint(pts_h=12.0, pts_a=10.0), _stint(neutral=True)])
    margin, edge = wins.expected_margins(stints, ratings)
    assert edge == pytest.approx(10.0) and margin.tolist() == pytest.approx([20.0, 10.0])


def _one_stint(seconds=2880.0, poss_h=100, poss_a=100):
    """Two fives over one whole stint, every rating 0 unless a test sets one, and a flat leverage."""
    stints = pd.DataFrame([_stint(start_clock=720.0, period=4, poss_h=poss_h, poss_a=poss_a, pts_h=100.0, pts_a=100.0)])
    ratings = pd.DataFrame({"player_id": [*HOME_FIVE.values(), *AWAY_FIVE.values()], "rating_off": 0.0,
                            "rating_def": 0.0, "rating_total": 0.0})
    pace = pd.DataFrame({"player_id": ratings.player_id, "pace_effect": 0.0})
    return stints, np.array([seconds]), ratings, pace


def _with(ratings: pd.DataFrame, player: int, off: float, deff: float) -> pd.DataFrame:
    ratings = ratings.copy()
    ratings.loc[ratings.player_id == player, ["rating_off", "rating_def", "rating_total"]] = [off, deff, off + deff]
    return ratings


BAR_ZERO = wins.Bar(0.0, 0.0, 0.0, 0.0, 150, 40)


def test_a_player_at_the_bar_reads_zero():
    stints, seconds, ratings, pace = _one_stint()
    bar = wins.Bar(0.5, 0.25, 0.75, 1.0, 150, 40)
    ratings = _with(ratings, 1, 0.5, 0.25)
    pace.loc[pace.player_id == 1, "pace_effect"] = 1.0
    got = wins.season_wins(stints, seconds, ratings, np.ones(1), pace, bar, 30.0).set_index("player_id").loc[1]
    assert got[["wosa_off", "wosa_def", "wosa_pace", "wosa"]].tolist() == pytest.approx([0.0] * 4, abs=1e-12)


def test_the_value_parts_are_his_rating_over_the_bar_on_the_possessions_at_each_end():
    stints, seconds, ratings, pace = _one_stint(poss_h=100, poss_a=80)
    ratings = _with(ratings, 1, 5.0, 2.0)
    got = wins.season_wins(stints, seconds, ratings, np.ones(1), pace, BAR_ZERO, 30.0).set_index("player_id")
    assert got.loc[1, "wosa_off"] == pytest.approx(5.0 * 100.0 / 100.0 / 30.0)
    assert got.loc[1, "wosa_def"] == pytest.approx(2.0 * 80.0 / 100.0 / 30.0)
    away = _with(ratings, 11, 5.0, 2.0)                               # on the visitors each end is the other count
    got = wins.season_wins(stints, seconds, away, np.ones(1), pace, BAR_ZERO, 30.0).set_index("player_id")
    assert got.loc[11, "wosa_off"] == pytest.approx(5.0 * 80.0 / 100.0 / 30.0)
    assert got.loc[11, "wosa_def"] == pytest.approx(2.0 * 100.0 / 100.0 / 30.0)


def test_the_pace_part_follows_the_lineups_margin_and_flips_with_it():
    stints, seconds, ratings, pace = _one_stint()
    pace.loc[pace.player_id == 1, "pace_effect"] = 3.0
    ahead = _with(ratings, 2, 6.0, 0.0)                 # a teammate at +6: player 1's lineup is +6 without any of his
    behind = _with(ratings, 11, 6.0, 0.0)               # an opponent at +6
    edge_free = {"pts_h": 103.0, "pts_a": 97.0}         # the stint's result equals the ratings' margin: no home edge
    up = wins.season_wins(stints.assign(**edge_free), seconds, ahead, np.ones(1), pace, BAR_ZERO, 30.0)
    down = wins.season_wins(stints.assign(pts_h=97.0, pts_a=103.0), seconds, behind, np.ones(1), pace, BAR_ZERO, 30.0)
    expected = 6.0 * 3.0 / 100.0 / 30.0                 # +6 per 100 on 3 extra possessions in the whole 48 minutes
    assert up.set_index("player_id").loc[1, "wosa_pace"] == pytest.approx(expected)
    assert down.set_index("player_id").loc[1, "wosa_pace"] == pytest.approx(-expected)


def test_his_own_surplus_is_not_paid_twice_on_the_possessions_he_adds():
    stints, seconds, ratings, pace = _one_stint()
    pace.loc[pace.player_id == 1, "pace_effect"] = 3.0
    star = _with(ratings, 1, 6.0, 0.0)                  # the lineup's whole margin is his
    got = wins.season_wins(stints.assign(pts_h=103.0, pts_a=97.0), seconds, star, np.ones(1), pace, BAR_ZERO,
                           30.0).set_index("player_id")
    assert got.loc[1, "wosa_pace"] == pytest.approx(0.0, abs=1e-12)


def test_a_constant_on_every_rating_moves_the_bar_by_it_and_no_win():
    stints, seconds, ratings, pace = _league()
    relative = wins.relative_leverage(stints, seconds, wins.fit_win_model(stints, seconds, 2001))
    starts = wins.starts_and_minutes(stints, seconds)
    bar = wins.starter_bar(ratings, starts, pace, 20)
    before = wins.season_wins(stints, seconds, ratings, relative, pace, bar, 27.0)
    moved = ratings.assign(rating_off=ratings.rating_off + 2.0, rating_total=ratings.rating_total + 2.0)
    moved_bar = wins.starter_bar(moved, starts, pace, 20)
    assert moved_bar.off == pytest.approx(bar.off + 2.0)
    after = wins.season_wins(stints, seconds, moved, relative, pace, moved_bar, 27.0)
    assert after.wosa.to_numpy() == pytest.approx(before.wosa.to_numpy(), abs=1e-10)


def test_the_parts_add_up_and_the_long_way_round_gives_the_same_wins():
    stints, seconds, ratings, pace = _league()
    relative = wins.relative_leverage(stints, seconds, wins.fit_win_model(stints, seconds, 2001))
    bar = wins.starter_bar(ratings, wins.starts_and_minutes(stints, seconds), pace, 20)
    got = wins.season_wins(stints, seconds, ratings, relative, pace, bar, 27.0).set_index("player_id")
    assert got.wosa.to_numpy() == pytest.approx((got.wosa_off + got.wosa_def + got.wosa_pace).to_numpy(), abs=1e-12)
    long_way = wins.counterfactual_wins(stints, seconds, ratings, relative, pace, bar, 27.0)
    assert long_way.reindex(got.index).to_numpy() == pytest.approx(got.wosa.to_numpy(), abs=1e-10)


def test_leverage_reads_one_for_everyone_when_it_is_flat_and_above_one_for_who_closes():
    stints, seconds, ratings, pace = _league()
    flat = wins.season_wins(stints, seconds, ratings, np.ones(len(stints)), pace, BAR_ZERO, 27.0)
    assert flat.leverage.to_numpy() == pytest.approx(1.0)
    lever = (stints.start_clock.to_numpy() < 300.0) * (stints.period.to_numpy() == 4) * 0.2 + 0.01
    closers = wins.season_wins(stints, seconds, ratings, lever, pace, BAR_ZERO, 27.0)
    assert closers.leverage.std() > 0.0 and np.average(closers.leverage, weights=closers.minutes_rs) == pytest.approx(
        1.0, abs=0.02)


def test_the_columns_go_after_the_tables_own_replacing_any_it_has_and_a_playoff_callup_reads_zero():
    rows = pd.DataFrame({"player_id": [1, 2, 3], "season": 2026, "rating_off": [1.0, 2.0, 3.0]})
    computed = pd.DataFrame({"player_id": [1, 2], **{column: [1.0, 2.0] for column in wins.WINS_COLUMNS}})
    joined = wins.wins_columns(rows, computed)
    assert list(joined.columns) == ["player_id", "season", "rating_off", *wins.WINS_COLUMNS]
    assert joined.player_id.tolist() == [1, 2, 3] and joined.wosa.tolist() == [1.0, 2.0, 0.0]
    assert joined.starts_rs.dtype.kind == "i" and joined[wins.WINS_COLUMNS].notna().all().all()
    again = wins.wins_columns(joined.assign(rating_off=9.0), computed.assign(wosa=[7.0, 8.0]))
    assert list(again.columns) == list(joined.columns) and again.wosa.tolist() == [7.0, 8.0, 0.0]
