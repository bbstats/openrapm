"""game_results.py on a simulated season with a known truth: a lean planted in the ratings handed in, by a made-up stat,
is read back with its sign on each side; a rating with no lean reads near 0; the table's defense goes in as points
allowed; and a stat no player on the court has reads as no lean at all."""
import numpy as np
import pandas as pd
import pytest

from openrapm import game_results, trade_set


@pytest.fixture(scope="module")
def season_world(synthetic_league, synthetic_design, small_cfg):
    sim = synthetic_league(n_seasons=1, n_teams=10, players_per_team=12, games_per_season=600, seed=23)
    design = synthetic_design(sim, cfg=small_cfg(player_unit="player"))
    truth = sim["truth"]["ps"].groupby(["player_id", "season"])[["impact_O", "impact_D"]].mean().reset_index()
    stat = np.random.default_rng(9).normal(0, 1, len(truth))       # a made-up stat, in standard deviations
    return design, truth, stat, _teams(sim, trade_set.team_game_design(design))


def _teams(sim, games) -> pd.DataFrame:
    """team_off and team_def per team-game, from the simulator's truth: the team of the players with a share on that
    side (every simulated player has one team a season)."""
    team = sim["truth"]["ps"].groupby("player_id").team.first()
    n = games.n_players
    out = {}
    for side, block in (("team_off", games.Z[:, :n]), ("team_def", games.Z[:, n:])):
        dense = np.asarray(block.todense())
        out[side] = [int(team.reindex(games.player_ids[row > 0]).mode().iloc[0]) for row in dense]
    return pd.DataFrame(out)


def _season(season_world, table):
    design, truth, _, teams = season_world
    # the simulated league has one target, points, so both sides read the same design
    return game_results.season_games({"offense": design, "defense": design}, table, int(truth.season.iloc[0]),
                                     lambda keys: teams)


def _lean(season_world, offense_lean: float, defense_lean: float) -> tuple[float, float]:
    """The game results' lean when the ratings handed in are the truth plus the planted leans times the stat: OpenRAPM
    rating players higher than the games by `offense_lean` and `defense_lean` per standard deviation."""
    _, truth, stat, _ = season_world
    table = pd.DataFrame({"season": truth.season, "player_id": truth.player_id,
                          "rating_off": truth.impact_O + offense_lean * stat,
                          # stored positive good: the simulator's defensive impact is points allowed
                          "rating_def": -truth.impact_D + defense_lean * stat, "poss_off": 1000.0})
    season = _season(season_world, table)
    z = pd.DataFrame({"player_id": truth.player_id, "made_up": stat})
    row = game_results.lean_table(season, z, ["made_up"]).iloc[0]
    return row.games, row.games_se


def test_the_tables_defense_goes_in_as_points_allowed():
    table = pd.DataFrame({"season": [2027, 2026], "player_id": [5, 5], "rating_off": [2.0, 9.0],
                          "rating_def": [1.5, 9.0], "poss_off": [1800.0, 10.0]})
    raw = game_results.raw_ratings(table, 2027)
    assert raw.to_dict("records") == [{"player_id": 5, "o": 2.0, "d": -1.5, "poss": 1800.0}]


def test_the_truth_reads_no_lean(season_world):
    value, error = _lean(season_world, 0.0, 0.0)
    assert error > 0 and abs(value) < 2.5 * error


@pytest.mark.parametrize("offense_lean, defense_lean", [(1.5, 0.0), (3.0, 0.0), (0.0, 3.0), (-3.0, 0.0), (2.0, -2.0)])
def test_a_planted_lean_is_read_back_at_the_games_width(season_world, offense_lean, defense_lean):
    """Rated 3.0 too high on offense per standard deviation: the column reads it above 0, OpenRAPM above the games;
    too low reads below 0, and leans on the two sides add, as the consensus column's total does.  It reads each side's
    lean at the games' width, as the consensus column reads OpenRAPM at the consensus's: the true ratings fit this
    simulated season at 0.865 on offense and 0.700 on defense, so a 3.0 plant on offense reads about 2.6 (2.81, its
    error 0.19) and on defense about 2.1 (2.30, error 0.32)."""
    _, truth, _, _ = season_world
    true_table = pd.DataFrame({"season": truth.season, "player_id": truth.player_id, "rating_off": truth.impact_O,
                               "rating_def": -truth.impact_D, "poss_off": 1000.0})
    width = game_results.multipliers(_season(season_world, true_table))
    expected = offense_lean * width["offense"] + defense_lean * width["defense"]
    value, error = _lean(season_world, offense_lean, defense_lean)
    assert abs(value - expected) < 2 * error, (value, expected, error)
    assert abs(expected) < 1 or (np.sign(value) == np.sign(expected) and abs(value) > 3 * error), (value, error)


def test_a_stat_no_player_on_the_court_has_reads_as_no_lean(season_world):
    _, truth, _, _ = season_world
    table = pd.DataFrame({"season": truth.season, "player_id": truth.player_id, "rating_off": truth.impact_O,
                          "rating_def": -truth.impact_D, "poss_off": 1000.0})
    season = _season(season_world, table)
    flat = game_results.lean_table(season, pd.DataFrame({"player_id": [-1], "nobody": [3.0]}), ["nobody"])
    assert flat.games.isna().all() and flat.games_se.isna().all()
    assert set(game_results.multipliers(season)) == {"offense", "defense"}
