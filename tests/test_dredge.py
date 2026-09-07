"""The Dredge counters (src/eracoef/dredge.py) and the features they become (gbdt_prior.add_dredge).

Every case here is built by hand from the shapes the real feed uses, taken verbatim from
`scratch/dredge_audit*.py`'s reading of 1997 and 2026: a BLOCK is its own row under the blocked shot, a
STEAL its own row beside the turnover, a team rebound carries the team in `personId` with `teamId` 0.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef.dredge import COUNTERS, DREDGE_LEAGUE_COLS, DREDGE_TOTAL_COLS, game_counts
from eracoef.gbdt_prior import DREDGE_ALL, DREDGE_RATES, DREDGE_SHARES, add_dredge

HOME, AWAY = 1610612740, 1610612753


def ev(team, person, action, sub="", desc="", value=0, dist=-1):
    return dict(teamId=team, personId=person, actionType=action, subType=sub, description=desc,
                shotValue=value, shotDistance=dist)


def frame(rows):
    return pd.DataFrame(rows)


def counts(rows):
    return game_counts(frame(rows)).set_index("player_id")


def test_assisted_and_unassisted_makes_split_on_the_description():
    c = counts([
        ev(HOME, 11, "Made Shot", "Jump Bank Shot", "Vucevic 19' Jump Bank Shot (2 PTS) (Payton 1 AST)", 2, 19),
        ev(HOME, 12, "Made Shot", "Dunk Shot", "Davis Dunk (2 PTS)", 2, 0),
        ev(AWAY, 21, "Made Shot", "Layup Shot", "Asik 2' Layup (2 PTS) (Evans 1 AST)", 2, 2),
        ev(AWAY, 22, "Missed Shot", "Jump Shot", "MISS Harris 16' Step Back Jump Shot", 2, 16),
    ])
    assert c.loc[11, "fgm_ast"] == 1 and c.loc[11, "fgm_unast"] == 0
    assert c.loc[12, "fgm_unast"] == 1 and c.loc[12, "fgm_ast"] == 0
    assert c.loc[21, "fgm_ast"] == 1
    assert 22 not in c.index or c.loc[22, ["fgm_ast", "fgm_unast"]].sum() == 0


def test_a_russell_is_a_block_the_blockers_team_rebounds():
    # blocker 12 (HOME) has three blocks: one his own team rebounds, one the shooting team rebounds,
    # and one that goes out of bounds and comes back as a TEAM rebound to the offence
    c = counts([
        ev(AWAY, 21, "Missed Shot", "Layup Shot", "MISS Payton 2' Layup", 2, 2),
        ev(HOME, 12, "", "", "Davis BLOCK (1 BLK)", 2),
        ev(HOME, 13, "Rebound", "Unknown", "Evans REBOUND (Off:0 Def:1)"),

        ev(AWAY, 21, "Missed Shot", "Jump Shot", "MISS Payton 18' Jump Shot", 2, 18),
        ev(HOME, 12, "", "", "Davis BLOCK (2 BLK)", 2),
        ev(AWAY, 22, "Rebound", "Unknown", "Harris REBOUND (Off:1 Def:0)"),

        ev(AWAY, 21, "Missed Shot", "Jump Shot", "MISS Payton 25' 3PT Jump Shot", 3, 25),
        ev(HOME, 12, "", "", "Davis BLOCK (3 BLK)", 3),
        ev(0, AWAY, "Rebound", "Normal Rebound", "MAGIC Rebound"),
    ])
    assert c.loc[12, "blk"] == 3
    assert c.loc[12, "blk_rus"] == 1                       # only the first: the other two go to the offence
    assert c.loc[12, "blk_rim"] == 1                       # the 2-foot layup; the 18-footer is not the rim
    assert c.loc[12, "blk_3"] == 1


def test_a_team_rebound_to_the_blockers_own_team_is_a_russell():
    c = counts([
        ev(AWAY, 21, "Missed Shot", "Layup Shot", "MISS Payton 2' Layup", 2, 2),
        ev(HOME, 12, "", "", "Davis BLOCK (1 BLK)", 2),
        ev(0, HOME, "Rebound", "Normal Rebound", "PELICANS Rebound"),
    ])
    assert c.loc[12, "blk_rus"] == 1


def test_a_1997_layup_at_distance_zero_still_counts_as_a_rim_block():
    # 1997 records layups and dunks at shotDistance 0; shotcurve.shot_bin reads the description for that,
    # and a distance-0 JUMP shot (a corner three whose distance was dropped) must not become a rim block
    c = counts([
        ev(AWAY, 21, "Missed Shot", "Layup Shot", "MISS Williams  Layup", 2, 0),
        ev(HOME, 12, "", "", "Ellison BLOCK (1 BLK)", 2),
        ev(HOME, 13, "Rebound", "Unknown", "Pippen REBOUND (Off:0 Def:1)"),
        ev(AWAY, 21, "Missed Shot", "Jump Shot", "MISS Williams Jump Shot", 2, 0),
        ev(HOME, 12, "", "", "Ellison BLOCK (2 BLK)", 2),
        ev(HOME, 13, "Rebound", "Unknown", "Pippen REBOUND (Off:0 Def:2)"),
    ])
    assert c.loc[12, "blk"] == 2 and c.loc[12, "blk_rim"] == 1


def test_a_steal_names_the_stealer_and_its_turnover_the_loser():
    for order in (0, 1):
        rows = [ev(HOME, 11, "Turnover", "Bad Pass", "Holiday Bad Pass Turnover (P1.T1)"),
                ev(AWAY, 21, "", "", "O'Quinn STEAL (1 STL)")]
        c = counts(rows[::-1] if order else rows)
        assert c.loc[21, "stl"] == 1 and c.loc[21, "tov_all"] == 0
        assert c.loc[11, "tov_all"] == 1 and c.loc[11, "tov_stolen"] == 1


def test_an_unstolen_turnover_is_counted_but_not_stolen():
    c = counts([ev(HOME, 11, "Turnover", "Traveling", "Fournier Traveling Turnover (P1.T1)")])
    assert c.loc[11, "tov_all"] == 1 and c.loc[11, "tov_stolen"] == 0


def test_fouls_go_to_the_committer_and_team_rows_are_dropped():
    c = counts([
        ev(HOME, 11, "Foul", "Loose Ball", "Holiday LOOSE.BALL.FOUL (P1.T1)"),
        ev(HOME, 11, "Foul", "Technical", "Holiday T.FOUL (P1.T1)"),
        ev(HOME, 11, "Foul", "Flagrant Type 1", "Holiday FLAGRANT.FOUL.TYPE1 (P2.T2)"),
        ev(HOME, 12, "Foul", "Offensive", "Asik OFF.Foul (P3)"),
        ev(HOME, 12, "Foul", "Offensive Charge", "Asik OFF.Foul (P4)"),
        ev(HOME, 12, "Foul", "Shooting", "Asik S.FOUL (P5.T3)"),
        ev(HOME, 12, "Foul", "Defense 3 Second", "Asik DEF.3.SEC (T4)"),
        ev(0, HOME, "Foul", "Technical", "PELICANS T.FOUL (Delay of Game)"),
    ])
    assert c.loc[11, "foul_loose"] == 1 and c.loc[11, "foul_tech"] == 2
    assert c.loc[12, "foul_off"] == 2                      # the aggregate: charges are never split out
    assert c.loc[12, "foul_tech"] == 0                     # defensive three seconds is not a feistiness proxy
    assert HOME not in c.index                             # the team technical belongs to nobody


def test_defensive_goaltending_is_counted_off_the_violation_row():
    c = counts([ev(HOME, 12, "Violation", "Defensive Goaltending", "Asik Violation:Defensive Goaltending"),
                ev(HOME, 12, "Violation", "Kicked Ball", "Asik Violation:Kicked Ball")])
    assert c.loc[12, "goaltend"] == 1


def _totals(**kw):
    """One player's block totals and a league that is ten of him, as add_dredge wants them."""
    base = dict.fromkeys(COUNTERS, 0.0) | dict(poss_off=1000.0, poss_def=1000.0) | kw
    row = {f"dr_{k}": v for k, v in base.items()}
    row |= {f"dr_lg_{k}": 10.0 * v for k, v in base.items()}
    return pd.DataFrame([row])


def test_every_feature_is_built_and_lands_between_the_player_and_his_league():
    d = add_dredge(_totals(blk=40.0, blk_rus=10.0, blk_rim=8.0, blk_3=2.0, fgm_unast=100.0, fgm_ast=100.0,
                           tov_all=50.0, tov_stolen=10.0, foul_loose=10.0, foul_tech=4.0, foul_off=6.0,
                           goaltend=2.0))
    assert set(DREDGE_ALL) <= set(d.columns)
    assert d[list(DREDGE_ALL)].notna().all().all()
    # this player's Russell share is 0.25 and the league's is 0.25, so padding cannot move it
    assert d["russsh"].iloc[0] == pytest.approx(0.25)
    assert d["blkrimsh"].iloc[0] == pytest.approx(0.20)
    assert d["unastsh"].iloc[0] == pytest.approx(0.50)
    # ... and a RATE is per 100 of its own denominator: 10 Russells in 1,000 defensive possessions
    assert d["russ"].iloc[0] == pytest.approx(1.0)
    assert d["loose"].iloc[0] == pytest.approx(0.5)        # 10 loose-ball fouls over 2,000 possessions


def test_padding_pulls_a_thin_row_all_the_way_to_the_league_and_leaves_a_thick_one_alone():
    lg = dict(blk=4000.0, blk_rus=2000.0, poss_def=400000.0)
    thin = _totals(blk=0.0, poss_def=0.0).assign(**{f"dr_lg_{k}": v for k, v in lg.items()})
    thick = _totals(blk=400.0, blk_rus=200.0, poss_def=40000.0).assign(
        **{f"dr_lg_{k}": v for k, v in lg.items()})
    a, b = add_dredge(thin), add_dredge(thick)
    league_rate = 100.0 * lg["blk_rus"] / lg["poss_def"]
    assert a["russ"].iloc[0] == pytest.approx(league_rate)                 # nothing of his own
    assert b["russ"].iloc[0] == pytest.approx(league_rate, rel=0.05)       # 10% of the league, 40k possessions
    assert a["russsh"].iloc[0] == pytest.approx(0.5)


def test_add_dredge_is_a_no_op_without_the_columns():
    df = pd.DataFrame({"season": [2000.0]})
    assert list(add_dredge(df).columns) == ["season"]


def test_the_stored_column_names_cover_every_counter_and_denominator():
    names = {c[3:] for c in DREDGE_TOTAL_COLS}
    assert names == set(COUNTERS) | {"poss_off", "poss_def"}
    assert [c[6:] for c in DREDGE_LEAGUE_COLS] == [c[3:] for c in DREDGE_TOTAL_COLS]
    for num, den, k in (*DREDGE_RATES.values(), *DREDGE_SHARES.values()):
        assert num in names and k > 0
        assert den in names or den in ("poss_all", "fgm_all")
