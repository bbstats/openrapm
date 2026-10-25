"""Self-creation's counts (src/openrapm/data/creation_counts.py): the reader's possession rules on a hand-written
game, each shot's cell, and the per-game counts with the passes expected on misses and trips."""
import numpy as np
import pandas as pd
import pytest

from openrapm.data import creation_counts as cc
from openrapm.features import CREATION_RATES

HOME, AWAY = 1610612760, 1610612745          # team ids, as the feed writes them
TEAM = {1: HOME, 2: HOME, 9: AWAY}


def _row(action, kind, text, person, clock, value=0, x=0, y=0, period=1):
    """One row as the feed writes it; a team row carries the team in personId and 0 in teamId."""
    return dict(actionNumber=0, clock=f"PT{int(clock // 60):02d}M{clock % 60:05.2f}S", period=period,
                teamId=TEAM.get(person, 0), personId=person, actionType=action, subType=kind, description=text,
                shotValue=value, xLegacy=x, yLegacy=y)


def _game() -> pd.DataFrame:
    rows = [
        _row("period", "start", "Start of 1st Period", 0, 720),
        # a corner three 22 s into the period
        _row("Made Shot", "Jump Shot", "One 24' 3PT Jump Shot (3 PTS) (Two 1 AST)", 1, 698, 3, -235, 30),
        _row("Missed Shot", "Layup Shot", "MISS Nine 2' Layup", 9, 690, 2, 5, 10),     # 8 s after the make: s7_14
        _row("Rebound", "Unknown", "Nine REBOUND (Off:1 Def:0)", 9, 689),
        _row("Missed Shot", "Tip Shot", "MISS Nine 1' Tip", 9, 688, 2, 0, 5),          # 1 s after: putback
        _row("Rebound", "Unknown", "One REBOUND (Off:0 Def:1)", 1, 687),
        _row("Made Shot", "Layup Shot", "Two 1' Layup (2 PTS)", 2, 683, 2, 0, 8),      # 4 s after a dreb: transition
        _row("Free Throw", "Free Throw 1 of 1", "Two Free Throw 1 of 1 (3 PTS)", 2, 683),   # the and-one
        _row("Turnover", "Bad Pass", "Nine Bad Pass Turnover (P1.T1)", 9, 670),
        _row("", "", "One STEAL (1 STL)", 1, 670),
        _row("Missed Shot", "Jump Shot", "MISS One 15' Jump Shot", 1, 650, 2, 100, 110),   # 20 s after the steal
        _row("Rebound", "Unknown", "Hawks Rebound", HOME, 649),                       # a team's, offensive
        _row("Missed Shot", "Jump Shot", "MISS One 18' Jump Shot", 1, 640, 2, 150, 90),    # second chance
        _row("Rebound", "Unknown", "Nine REBOUND (Off:1 Def:1)", 9, 639),
        _row("Foul", "Shooting", "One S.FOUL (P1.T1)", 1, 630),
        _row("Free Throw", "Free Throw 1 of 2", "MISS Nine Free Throw 1 of 2", 9, 630),
        _row("Rebound", "Unknown", "Team Rebound", AWAY, 630),                         # dead ball: passed over
        _row("Free Throw", "Free Throw 2 of 2", "Nine Free Throw 2 of 2 (1 PTS)", 9, 630),
        _row("Free Throw", "Free Throw Technical", "Nine Free Throw Technical (2 PTS)", 9, 620),
        _row("Missed Shot", "Jump Shot", "MISS Two 25' 3PT Jump Shot", 2, 605, 3, 50, 245),  # 25 s after the made FT
    ]
    return pd.DataFrame(rows)


def test_the_reader_follows_the_ball_and_counts_trips_and_and_ones():
    shots, trips = cc.game_shots(_game())
    shots = cc.add_cells(shots)
    assert shots.shooter.tolist() == [1, 9, 9, 2, 1, 1, 2]
    assert shots.cell.tolist() == ["corner_three|s20_plus", "rim|s7_14", "rim|putback", "rim|transition",
                                   "mid|s14_20", "long_two|second_chance", "above_break_three|s20_plus"]
    assert shots.assisted.tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert shots.and_one.tolist() == [0, 0, 0, 1, 0, 0, 0]
    assert shots.start_kind.tolist() == ["period_start", "made_fg", "made_fg", "dreb", "steal", "steal", "made_ft"]
    by = trips.set_index("player_id")
    assert by.loc[2, ["trips", "and_one_trips", "technical"]].tolist() == [1, 1, 0]
    assert by.loc[9, ["trips", "and_one_trips", "technical"]].tolist() == [1, 0, 1]


def test_the_counts_add_a_cells_share_moved_by_its_miss_factor_and_the_trips_rim_chance():
    shots, trips = cc.game_shots(_game())
    shots = cc.add_cells(shots).assign(game_id="0022500001", phase="RS")
    trips = trips.assign(game_id="0022500001", phase="RS")
    counts = cc.season_creation_counts(shots, trips, 2026).set_index("player_id")
    assert list(counts.columns[3:]) == [rate.removesuffix("_per100") for rate in CREATION_RATES]
    share = cc.cell_shares(shots)
    # player 9's two rim misses: each cell's padded share moved by its factor
    expected = sum(cc.expit(cc.logit(share[cell]) + cc.MISS_FACTOR[cell]) for cell in ("rim|s7_14", "rim|putback"))
    assert counts.loc[9, "expected_passed_misses_inside"] == pytest.approx(expected)
    assert counts.loc[9, "misses_inside"] == 2
    # player 9's one trip without a make, at the mean chance of the game's rim misses x the trips' factor
    assert counts.loc[9, "expected_passed_trips"] == pytest.approx(expected / 2 * cc.FT_FACTOR)
    # player 2's only trip is an and-one, left to its make
    assert counts.loc[2, "trips_without_make"] == 0
    assert counts.loc[1, "assisted_makes_three"] == 1
    assert counts.loc[1, "expected_assisted_makes_three"] == pytest.approx(share["corner_three|s20_plus"])


def test_a_cells_share_is_padded_toward_its_prior_and_every_cell_has_a_factor():
    cells = [f"{spot}|{context}" for spot in cc.SPOTS for context in cc.CONTEXTS]
    assert sorted(cc.CELL_PRIOR) == sorted(cells) and sorted(cc.MISS_FACTOR) == sorted(cells)
    shots = pd.DataFrame({"cell": ["rim|s7_14"] * 50, "made": 1, "assisted": [1] * 50})
    share = cc.cell_shares(shots)
    assert share["rim|s7_14"] == pytest.approx((50 + 50 * cc.CELL_PRIOR["rim|s7_14"]) / 100)
    assert share["mid|early"] == pytest.approx(cc.CELL_PRIOR["mid|early"])
    # misses at the rim come off a pass less often than makes there, in every possession situation but putbacks
    assert all(cc.MISS_FACTOR[f"rim|{context}"] < 0 for context in cc.CONTEXTS)
    assert np.isfinite(cc.FT_FACTOR) and 0.3 < cc.FT_FACTOR < 1.0


def test_the_candidates_count_unassisted_makes_by_group_the_situations_and_the_and_ones():
    """Experiment 112's counts on the hand-written game: player 2's transition layup is an unassisted make inside and
    an and-one; player 9's tip a second after his offensive rebound is a putback; player 1's corner three 22 s into the
    period and player 2's three 25 s after the made free throw are late-clock attempts."""
    shots, trips = cc.game_shots(_game())
    shots = cc.add_cells(shots).assign(game_id="0022500001", phase="RS")
    counts = cc.season_creation_counts(shots, trips.assign(game_id="0022500001", phase="RS"), 2026)
    counts = counts.set_index("player_id")
    inside = ["unassisted_makes_inside", "and_ones", "transition_attempts", "late_clock_attempts"]
    assert counts.loc[2, inside].tolist() == [1, 1, 1, 1]
    assert counts.loc[1, ["unassisted_makes_three", "late_clock_attempts", "transition_attempts"]].tolist() == [0, 1, 0]
    assert counts.loc[9, ["putback_attempts", "unassisted_makes_inside", "and_ones"]].tolist() == [1, 0, 0]
