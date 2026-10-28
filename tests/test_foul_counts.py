"""The fouls, goaltends, shots blocked and three-shot trips read from the play-by-play
(src/openrapm/data/foul_counts.py), each rule on a hand-written game."""
import pandas as pd

from openrapm.data.foul_counts import FOUL_COUNTS, game_foul_counts

HOME, AWAY = 1610612760, 1610612745          # team ids, as the feed writes them
TEAM = {1: HOME, 2: HOME, 9: AWAY}


def _row(action, kind, text, person, value=0):
    """One row as the feed writes it; a team's row carries the team in personId and 0 in teamId."""
    return dict(actionType=action, subType=kind, description=text, personId=person, teamId=TEAM.get(person, 0),
                playerName="", playerNameI="", shotValue=value, xLegacy=0, yLegacy=0)


def _game() -> pd.DataFrame:
    rows = [
        _row("Violation", "Defensive Goaltending", "One Violation:Defensive Goaltending (R.Hussey)", 1),
        _row("Violation", "Defensive Goaltending", "76ERS Violation: Def. Goaltending Violation", HOME),  # a team's
        _row("Foul", "Shooting", "One S.FOUL (P1.T1)", 1),
        _row("Foul", "Shooting Block", "Two Shooting Block Foul (P1.T2)", 2),
        _row("Foul", "Loose Ball", "Two L.B.FOUL (P2.T3)", 2),
        _row("Foul", "Personal", "Two P.FOUL (P3.T4)", 2),                                               # not counted
        _row("Missed Shot", "Jump Shot", "MISS Nine 25' 3PT Jump Shot", 9, 3),
        _row("", "", "One BLOCK (1 BLK)", 1),                                                            # Nine's shot
        _row("Missed Shot", "Jump Shot", "MISS Nine 18' Jump Shot", 9, 2),                               # not blocked
        _row("Free Throw", "Free Throw 1 of 3", "Nine Free Throw 1 of 3 (1 PTS)", 9),
        _row("Free Throw", "Free Throw 2 of 3", "Nine Free Throw 2 of 3 (2 PTS)", 9),
        _row("Free Throw", "Free Throw Flagrant 1 of 3", "Nine Free Throw Flagrant 1 of 3 (3 PTS)", 9),  # not counted
    ]
    return pd.DataFrame(rows)


def test_each_count_reads_its_rows_and_only_a_named_player():
    counts = game_foul_counts(_game()).set_index("player_id")
    assert list(counts.columns) == FOUL_COUNTS
    assert counts.loc[1].to_dict() == dict(goaltends=1, shooting_fouls=1, loose_ball_fouls=0, shots_blocked=0,
                                           three_shot_trips=0)
    assert counts.loc[2].to_dict() == dict(goaltends=0, shooting_fouls=1, loose_ball_fouls=1, shots_blocked=0,
                                           three_shot_trips=0)
    assert counts.loc[9].to_dict() == dict(goaltends=0, shooting_fouls=0, loose_ball_fouls=0, shots_blocked=1,
                                           three_shot_trips=1)
    assert HOME not in counts.index
