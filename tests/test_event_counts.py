"""The play-by-play parts of box counts (src/openrapm/data/event_counts.py): each rule on a hand-written game,
and the padding design built from them on a simulated league (the same games and rows the 13 box rates count)."""
import numpy as np
import pandas as pd
import pytest

from openrapm.box_rates import box_rates
from openrapm.data.event_counts import EVENT_COUNTS, event_design, game_event_counts, plain_name
from openrapm.data.panel import cut_design
from openrapm.design import FEATURES
from openrapm.features import EVENT_RATES
from openrapm.simulate import simulate

HOME, AWAY = 1610612760, 1610612745          # team ids, as the feed writes them


NAMES = {1: "One", 2: "Two", 9: "Nine"}
TEAM = {1: HOME, 2: HOME, 9: AWAY}


def _row(action, kind, text, person, value=0, x=0, y=0):
    """One row as the feed writes it: a player's row carries his team and both forms of his name; a team row
    (a team rebound) carries the team in personId and 0 in teamId."""
    name = NAMES.get(person, "")
    return dict(actionType=action, subType=kind, description=text, personId=person, teamId=TEAM.get(person, 0),
                playerName=name, playerNameI=f"{name[:1]}. {name}" if name else "", shotValue=value, xLegacy=x,
                yLegacy=y)


def _game() -> pd.DataFrame:
    """Player 1 (home) rebounds four misses of player 9 (away), one of each kind; then the steals, the makes and the
    threes.  Rows with no action type are the feed's STEAL and BLOCK rows."""
    rows = [
        _row("Missed Shot", "Jump Shot", "MISS Nine 25' 3PT Jump Shot", 9, 3, -179, 179),
        _row("Rebound", "Unknown", "One REBOUND (Off:0 Def:1)", 1),                       # of a three
        _row("Free Throw", "Free Throw 2 of 2", "MISS Nine Free Throw 2 of 2", 9),
        _row("Rebound", "Unknown", "One REBOUND (Off:0 Def:2)", 1),                       # of a free throw
        _row("Missed Shot", "Layup Shot", "MISS Nine  Layup", 9, 2, 0, 0),
        _row("", "", "Two BLOCK (1 BLK)", 2),                                             # skipped
        _row("Rebound", "Unknown", "One REBOUND (Off:0 Def:3)", 1),                       # of a rim two, no location
        _row("Missed Shot", "Jump Shot", "MISS Nine 15' Jump Shot", 9, 2, 72, 126),
        _row("Rebound", "Unknown", "One REBOUND (Off:0 Def:4)", 1),                       # of a two from 14.5 ft
        _row("Missed Shot", "Jump Shot", "MISS One 8' Jump Shot", 1, 2, -76, 33),
        _row("Rebound", "Unknown", "One REBOUND (Off:1 Def:4)", 1),                       # offensive: no part
        _row("Missed Shot", "Jump Shot", "MISS Nine 8' Jump Shot", 9, 2, 40, 20),
        _row("Rebound", "Unknown", "Rockets Rebound", HOME),                              # a team's: no part
        _row("Turnover", "Bad Pass", "Nine Bad Pass Turnover (P1.T1)", 9),
        _row("", "", "Two STEAL (1 STL)", 2),
        _row("Turnover", "Lost Ball", "Nine Lost Ball Turnover (P1.T2)", 9),
        _row("", "", "Two STEAL (2 STL)", 2),
        _row("Turnover", "Stolen Pass Turnover", "Nine Stolen Pass Turnover (P1.T3)", 9),
        _row("", "", "One STEAL (1 STL)", 1),
        _row("Made Shot", "Jump Shot", "Two 26' 3PT Jump Shot (3 PTS) (One 1 AST)", 2, 3, 164, 201),
        _row("Made Shot", "Jump Shot", "Two 22' 3PT Jump Shot (6 PTS)", 2, 3, 233, 0),     # corner, unassisted
        _row("Missed Shot", "Jump Shot", "MISS Two 23' 3PT Jump Shot", 2, 3, -225, 80),   # corner
        _row("Made Shot", "Dunk Shot", "Two  Dunk (8 PTS) (One 2 AST)", 2, 2, 0, 0),
        _row("Missed Shot", "Jump Shot", "MISS Two 3PT Jump Shot", 2, 3, 0, 0),           # no location: above
        _row("Missed Shot", "Layup Shot", "MISS Nine 5' Layup", 9, 2, 0, 50),
        _row("", "", "Two BLOCK (1 BLK)", 2),                                             # a rim block ...
        _row("Rebound", "Unknown", "Thunder Rebound", HOME),                              # ... his team's: Russell
        _row("Missed Shot", "Jump Shot", "MISS Nine 25' 3PT Jump Shot", 9, 3, -150, 200),
        _row("", "", "One BLOCK (1 BLK)", 1),                                             # a three blocked ...
        _row("Rebound", "Unknown", "Nine REBOUND (Off:1 Def:0)", 9),                      # ... Nine keeps it
        _row("Foul", "Offensive Charge", "Nine OFF.Foul (P1)", 9),
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def counts():
    return game_event_counts(_game()).set_index("player_id")


def test_each_defensive_rebound_is_filed_under_the_miss_it_ended(counts):
    one = counts.loc[1]
    assert (one.def_rebounds_of_threes, one.def_rebounds_of_free_throws, one.def_rebounds_of_rim_twos,
            one.def_rebounds_of_outside_twos) == (1, 1, 1, 1)
    assert HOME not in counts.index, "a team rebound is no player's"


def test_steals_split_by_the_turnover_they_took(counts):
    assert (counts.loc[2].bad_pass_steals, counts.loc[2].lost_ball_steals) == (1, 1)
    assert (counts.loc[1].bad_pass_steals, counts.loc[1].lost_ball_steals) == (1, 0), "a stolen pass is a pass"


def test_points_and_threes_split_by_the_assist_and_the_spot(counts):
    two = counts.loc[2]
    assert (two.assisted_points, two.unassisted_points) == (3 + 2, 3)
    assert (two.assisted_threes_made, two.unassisted_threes_made) == (1, 1)
    assert (two.corner_threes_attempted, two.above_break_threes_attempted) == (2, 2)


def test_blocks_split_by_the_shot_and_by_who_got_the_ball_back(counts):
    two, one = counts.loc[2], counts.loc[1]
    # the layup blocked earlier in the game (no location) is a rim block too, and One rebounded it: Russell
    assert (two.rim_blocks, two.mid_range_blocks, two.three_point_blocks, two.russell_blocks) == (2, 0, 0, 2)
    assert (one.rim_blocks, one.three_point_blocks, one.russell_blocks) == (0, 1, 0)


def test_turnovers_stolen_offensive_fouls_and_assists_by_zone(counts):
    nine, one = counts.loc[9], counts.loc[1]
    assert (nine.stolen_turnovers, nine.offensive_fouls) == (3, 1)
    assert (one.assists_to_above_break_threes, one.assists_to_rim, one.assists_to_mid_range,
            one.assists_to_corner_threes) == (1, 1, 0, 0), "credited to the passer, by the zone of the make"
    assert counts.loc[2][[name for name in EVENT_COUNTS if name.startswith("assists_to")]].sum() == 0


def test_the_counts_come_back_in_their_order_with_no_gaps(counts):
    assert list(counts.columns) == EVENT_COUNTS == [rate.removesuffix("_per100") for rate in EVENT_RATES]
    assert not counts.isna().any().any()
    assert set(counts.index) == {1, 2, 9}
    assert counts.loc[9].above_break_threes_attempted == 2, "his two missed threes"


def test_names_are_matched_without_accents_suffixes_or_case():
    assert plain_name("Luka Dončić") == plain_name("luka doncic") == "luka doncic"
    assert plain_name("Jaren Jackson Jr.") == "jaren jackson" and plain_name("Gary Payton II") == "gary payton"


@pytest.fixture(scope="module")
def league(synthetic_design):
    """A simulated league's design, and event counts invented from its box table: every steal a bad-pass steal
    in half-A-numbered games, a lost-ball steal otherwise, and a player outside the box table who never counts."""
    sim = simulate(n_seasons=1, n_teams=6, players_per_team=10, games_per_season=40, stints_per_game=(20, 30), seed=3)
    design = synthetic_design(sim)
    box = sim["box"].copy()
    box["game_id"] = box["game_id"].astype(str)
    odd = box.game_id.str[-1].astype(int) % 2 == 1
    counts = pd.DataFrame({"game_id": box.game_id, "season": box.season, "phase": box.phase,
                           "player_id": box.player_id})
    for name in EVENT_COUNTS:
        counts[name] = 0.0
    counts["bad_pass_steals"] = np.where(odd, box.stl, 0.0)
    counts["lost_ball_steals"] = np.where(odd, 0.0, box.stl)
    stranger = counts.iloc[[0]].assign(player_id=999_999, bad_pass_steals=50.0)
    return design, sim, pd.concat([counts, stranger], ignore_index=True)


def test_the_parts_are_counted_over_the_box_tables_rows(league):
    """The two steal parts add up to the box score's steals player by player, and the padding sees the same games
    and possessions as the 13 rates."""
    design, _, counts = league
    events, read = event_design(design, counts)
    parts, rates = box_rates(events), box_rates(design)
    steals = FEATURES.index("stl")
    np.testing.assert_allclose(parts.counts[:, 4] + parts.counts[:, 5], rates.counts[:, steals])
    np.testing.assert_array_equal(parts.offence.possessions, rates.offence.possessions)
    assert len(events.game_box) == len(design.game_box) and list(events.spec.box_counts) == EVENT_COUNTS
    assert read == set(design.games.game_id.astype(str))


def test_a_design_cut_to_some_games_reads_only_their_counts(league):
    design, _, counts = league
    keep = np.unique(design.rows.game_idx)[:20]
    events, read = event_design(cut_design(design, keep), counts)
    assert read == set(design.games.loc[design.games.game_idx.isin(keep), "game_id"].astype(str))
    assert set(events.game_box.game_idx) <= set(keep)
