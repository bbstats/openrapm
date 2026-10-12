"""The season panel of a simulated league (src/openrapm/simulate.py: player_profiles, simulated_panel): the role and
career columns counted by hand, the body drawn from its own generator, and a panel the shipped prior can read."""
import numpy as np
import pandas as pd
import pytest

from openrapm.box_rates import box_rates
from openrapm.features import SEASON_PANEL_COLUMNS, SHIPPED, add_features
from openrapm.simulate import PROFILE_COLUMNS, player_profiles, simulate, simulated_panel


@pytest.fixture(scope="module")
def league(synthetic_league, synthetic_design):
    """Three seasons with players moving teams, each season's design, and the profiles."""
    sim = synthetic_league(n_seasons=3, n_teams=6, games_per_season=30, seed=2)
    designs = {season: synthetic_design(sim, [season]) for season in (2001, 2002, 2003)}
    return sim, designs, player_profiles(sim, seed=2)


def test_five_players_share_every_team_possession(league):
    sim, _, profiles = league
    teams = sim["truth"]["ps"][["player_id", "season", "team"]]
    share = profiles.merge(teams, on=["player_id", "season"]).groupby(["season", "team"]).team_possession_share.sum()
    np.testing.assert_allclose(share.to_numpy(), 5.0)


def test_starts_are_the_first_stint_of_each_game(league):
    sim, _, profiles = league
    stints = sim["stints"]
    first = stints.drop_duplicates("game_id")
    one = profiles[profiles.season == 2001].set_index("player_id")
    player = int(first.h1.iloc[0])
    in_season = stints[stints.season == 2001]
    slots = [f"{side}{place}" for side in "ha" for place in range(1, 6)]
    games = in_season.loc[(in_season[slots] == player).any(axis=1), "game_id"].unique()
    starts = first[(first.season == 2001) & (first[slots] == player).any(axis=1)]
    assert one.loc[player, "games_started_share"] == pytest.approx(len(starts) / len(games))
    assert profiles.games_started_share.between(0.0, 1.0).all()


def test_the_career_counts_only_earlier_seasons(league):
    _, _, profiles = league
    assert (profiles[profiles.season == 2001][["seasons_before", "career_possessions_before_k"]] == 0.0).all().all()
    one = profiles[profiles.player_id == profiles.player_id.iloc[0]].sort_values("season")
    assert one.seasons_before.tolist() == [0.0, 1.0, 2.0]
    assert np.all(np.diff(one.career_possessions_before_k.to_numpy()) > 0)


def test_seasons_with_the_team_restart_after_a_move(league):
    sim, _, profiles = league
    teams = sim["truth"]["ps"][["player_id", "season", "team"]]
    merged = profiles.merge(teams, on=["player_id", "season"]).sort_values(["player_id", "season"])
    previous = merged.groupby("player_id").team.shift()
    moved = merged[(merged.season > 2001) & (merged.team != previous)]
    stayed = merged[(merged.season == 2002) & (merged.team == previous)]
    assert len(moved) and (moved.seasons_with_team == 1.0).all()
    assert len(stayed) and (stayed.seasons_with_team == 2.0).all()


def test_the_body_comes_from_its_own_generator(league):
    """The same seed gives the same heights, and simulate's own draws are untouched by asking for them."""
    sim, _, profiles = league
    again = player_profiles(sim, seed=2)
    pd.testing.assert_frame_equal(profiles, again)
    assert profiles.groupby("player_id").height_in.nunique().max() == 1
    fresh = simulate(n_seasons=1, n_teams=4, players_per_team=8, games_per_season=6, stints_per_game=(5, 6), seed=9)
    player_profiles(fresh, seed=9)
    replay = simulate(n_seasons=1, n_teams=4, players_per_team=8, games_per_season=6, stints_per_game=(5, 6), seed=9)
    pd.testing.assert_frame_equal(fresh["stints"], replay["stints"])


def test_the_panel_has_both_sides_in_the_panel_order_and_the_shipped_inputs_build(league):
    _, designs, profiles = league
    panel = simulated_panel(designs, profiles)
    assert list(panel.columns) == [column for column in SEASON_PANEL_COLUMNS if column in panel.columns]
    assert set(PROFILE_COLUMNS) <= set(panel.columns)
    rows = panel[(panel.side == "D") & (panel.season == 2002)]
    np.testing.assert_array_equal(rows.possessions.to_numpy(), box_rates(designs[2002]).defence.possessions)
    for side in ("O", "D"):
        add_features(panel[panel.side == side].copy(), SHIPPED[side])
