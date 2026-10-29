"""The matchup inputs (src/openrapm/data/matchups.py): who guarded whom, worked by hand on two teams and three games.

Team 10 (players 1, 2, 3, and 7 who has no panel row) attacks team 20 (players 4, 5, 6) and the reverse.  Player 1 is
team 10's top creator; on team 20, player 5 has the higher total creation per 100, 25 against player 4's 22, but on
half the possessions, so player 4 creates more shots a game and is the top.  With `pad=0` every input is its plain
weighted average or share.
"""
import numpy as np
import pandas as pd
import pytest

from openrapm.data import matchups
from openrapm.data.matchups import matchup_inputs, matchup_pairs, padded_average, top_creators

GAMES = ["0022500001", "0022500002", "0022500003"]
# (attacking team, attacker, defender, matchup possessions) in games 1 and 2; game 3 below
SAME = [(10, 1, 4, 20.0), (10, 1, 5, 5.0), (10, 2, 5, 15.0), (10, 2, 6, 10.0), (10, 3, 6, 15.0),
        (20, 4, 1, 10.0), (20, 4, 2, 10.0), (20, 5, 2, 30.0), (20, 6, 3, 20.0)]
# game 3: player 6 guards player 1, not player 3, and also player 7, whom the panel does not know
THIRD = [row for row in SAME if row[1] != 3] + [(10, 1, 6, 15.0), (10, 7, 6, 10.0)]


def pairs(games=GAMES) -> pd.DataFrame:
    rows = [(game, *row) for game in games for row in (THIRD if game == GAMES[2] else SAME)]
    return pd.DataFrame(rows, columns=["game_id", "attacker_team_id", "attacker_id", "defender_id",
                                       "matchup_possessions"])


def partners() -> pd.DataFrame:
    return pd.DataFrame({"height_in": [75.0, 80.0, 84.0, 74.0, 79.0, 85.0],
                         "self_creation_per100": [20.0, 10.0, 5.0, 18.0, 12.0, 3.0],
                         "total_creation_per100": [30.0, 14.0, 6.0, 22.0, 25.0, 4.0],
                         "possessions": [300.0, 300.0, 300.0, 300.0, 150.0, 300.0]}, index=[1, 2, 3, 4, 5, 6])


def test_each_input_is_the_weighted_average_or_share_of_the_pairs():
    out = matchup_inputs(pairs(), partners(), pad=0.0)
    # defender 6: 10 on player 2 at 80 in, 15 on player 3 at 84 in games 1-2, then 10 on 2 and 15 on 1 at 75 in
    # game 3; player 7 has no height, so his 10 are left out of the average
    assert out.loc[6, "attackers_guarded_height_in"] == pytest.approx(6045 / 75)
    assert out.loc[5, "attackers_guarded_height_in"] == pytest.approx(78.75)        # 5 on 1, 15 on 2, each game
    assert out.loc[2, "attackers_guarded_height_in"] == pytest.approx(77.75)        # 10 on 4, 30 on 5
    assert out.loc[1, "defenders_faced_height_in"] == pytest.approx(6900 / 90)      # 4 and 5, then 6 in game 3
    assert out.loc[7, "defenders_faced_height_in"] == pytest.approx(85.0)
    assert out.loc[5, "attackers_guarded_self_creation_per100"] == pytest.approx(12.5)    # (5 x 20 + 15 x 10) / 20
    assert out.loc[5, "attackers_guarded_total_creation_per100"] == pytest.approx(18.0)   # (5 x 30 + 15 x 14) / 20
    assert np.isnan(out.loc[7, "attackers_guarded_height_in"]), "player 7 never defended"


def test_the_top_creator_is_the_most_shots_a_game_not_the_highest_rate():
    out = matchup_inputs(pairs(), partners(), pad=0.0)
    assert out.loc[4, "top_self_creator_guarded_share"] == 1.0                      # all his defence on player 1
    assert out.loc[5, "top_self_creator_guarded_share"] == pytest.approx(0.25)      # 5 of 20 on player 1
    assert out.loc[6, "top_self_creator_guarded_share"] == pytest.approx(15 / 85)   # player 7 counts in the 85
    # team 20's top by total creation is player 4, 22 x 300 / 3 / 100 = 22 a game against player 5's 12.5
    assert out.loc[2, "top_total_creator_guarded_share"] == pytest.approx(0.25)     # 10 of 40 on player 4
    assert out[["top_self_creator_guarded_share_on_shared_floor"]].isna().all().all(), "no stints, no shared floor"


def test_fewer_fit_games_change_the_inputs_and_read_no_other_game():
    out = matchup_inputs(pairs(GAMES[:2]), partners(), pad=0.0)
    assert out.loc[6, "attackers_guarded_height_in"] == pytest.approx(82.4)         # (10 x 80 + 15 x 84) / 25
    assert out.loc[6, "top_self_creator_guarded_share"] == 0.0
    assert 7 not in out.index


def test_a_tie_goes_to_the_lower_id_and_an_unknown_attacker_is_never_the_top():
    two = pd.DataFrame({"game_id": "g", "attacker_team_id": 10, "attacker_id": [3, 2, 9], "defender_id": 4,
                        "matchup_possessions": 1.0})
    top = top_creators(two, pd.Series({2: 5.0, 3: 5.0}))
    assert top.top_id.tolist() == [2]
    assert top_creators(two, pd.Series({9: np.nan})).empty


def test_the_shared_floor_share_counts_only_the_possessions_with_the_top_creator_on():
    """One game: player 1, team 10's only player with a volume, plays the first of two 10-possession stints.  Team
    10's attackers hold 100 matchup possessions on 20 stint possessions, a scale of 1, so each opposing defender shares
    10 with him: defender 4 spent 6 of them on him, defender 5 4.  Of all their defence, 6 of 20 and 4 of 20."""
    game = "0022500001"
    rows = [(1, 4, 6.0), (1, 5, 4.0), (2, 5, 6.0), (2, 6, 14.0), (3, 6, 6.0), (3, 10, 14.0), (8, 10, 6.0),
            (8, 11, 14.0), (9, 11, 6.0), (9, 4, 14.0), (12, 5, 10.0)]
    one = pd.DataFrame([(game, 10, *row) for row in rows], columns=["game_id", "attacker_team_id", "attacker_id",
                                                                     "defender_id", "matchup_possessions"])
    known = pd.DataFrame({"height_in": 80.0, "self_creation_per100": 20.0, "total_creation_per100": 30.0,
                          "possessions": 300.0}, index=[1])
    stints = pd.DataFrame({"game_id": game, "h1": [1, 2], "h2": [2, 3], "h3": [3, 8], "h4": [8, 9], "h5": [9, 12],
                           "a1": 4, "a2": 5, "a3": 6, "a4": 10, "a5": 11, "poss_h": 10.0, "poss_a": 10.0})
    games = pd.DataFrame({"game_id": [game], "home_team_id": [10], "away_team_id": [20]})
    out = matchup_inputs(one, known, stints, games, pad=0.0)
    shared = out["top_self_creator_guarded_share_on_shared_floor"]
    assert (shared[4], shared[5], shared[6]) == (pytest.approx(0.6), pytest.approx(0.4), 0.0)
    assert out.loc[4, "top_self_creator_guarded_share"] == pytest.approx(0.3)
    assert out.loc[5, "top_self_creator_guarded_share"] == pytest.approx(0.2)
    assert out.loc[5, "top_total_creator_guarded_share_on_shared_floor"] == pytest.approx(0.4)


def test_padding_with_too_few_players_takes_the_fallback_pad_toward_the_season():
    """One game each, so no player has two halves: the pad is FALLBACK_PAD (50), toward the season's 39,100 / 500."""
    out = padded_average(np.array([0, 1]), np.array([0, 1]), np.array([400 * 76.5, 100 * 85.0]),
                         np.array([400.0, 100.0]), 3)
    assert out[0] == pytest.approx((400 * 76.5 + 50 * 78.2) / 450)
    assert out[1] == pytest.approx((100 * 85.0 + 50 * 78.2) / 150)
    assert np.isnan(out[2]), "no weight, no value"


def test_the_pairs_come_from_the_games_asked_for_and_a_missing_game_stops_the_run(tmp_path):
    cfg = {"_root": str(tmp_path), "paths": {"raw": str(tmp_path / "raw")}}
    folder = matchups.matchup_folder(cfg, 2026)
    folder.mkdir(parents=True)
    for game in GAMES[:2]:
        pd.DataFrame({"gameId": game, "teamId": [10, 20], "personIdOff": [1, 4], "personIdDef": [4, 1],
                      "partialPossessions": [12.5, 0.0]}).to_parquet(folder / f"{game}.parquet")
    got, read = matchup_pairs(cfg, 2026, GAMES[:1])
    assert read == {GAMES[0]} and got.game_id.tolist() == [GAMES[0]], "the pair with no possessions is left out"
    assert got.dtypes["attacker_id"] == np.int64 and got.matchup_possessions.tolist() == [12.5]
    with pytest.raises(FileNotFoundError, match="openrapm data matchup-games"):
        matchup_pairs(cfg, 2026, GAMES)
    assert matchup_pairs(cfg, 2017, GAMES) == (None, set())
    assert matchup_pairs(cfg, 2025, GAMES) == (None, set()), "no folder: a season not fetched reads missing"
