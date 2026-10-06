"""The shot logger (stints.GameParser, log_shots=True) and the shot frame's derived fields."""
import numpy as np
import pandas as pd

from eracoef.shotframe import derive, player_game_totals, recentre
from eracoef.stints import GameParser
from test_stints import AWAY, HOME, _box, synthetic_game


def _game():
    """The synthetic game with the feed's action numbers: a STEAL row shares its turnover's number."""
    g = synthetic_game()
    g["actionNumber"] = np.arange(1, len(g) + 1)
    tov = g.index[g.actionType == "Turnover"][0]
    g.loc[g.description.str.contains("STEAL"), "actionNumber"] = g.loc[tov, "actionNumber"]
    return g


def _parsed(log_shots):
    gp = GameParser(_game(), _box(), HOME, AWAY, "0020000001", log_shots=log_shots)
    p = gp.possessions()
    return gp, p, gp.stints(p)


def test_the_logger_changes_no_possession_and_no_stint():
    _, p0, s0 = _parsed(False)
    gp, p1, s1 = _parsed(True)
    pd.testing.assert_frame_equal(p0.drop(columns="shooters"), p1.drop(columns="shooters"))
    pd.testing.assert_frame_equal(s0, s1)
    assert len(gp.shot_log) == int(p1.fga.sum())


def test_one_row_per_attempt_with_its_possession():
    gp, p, _ = _parsed(True)
    f = pd.DataFrame(gp.shot_log)
    assert f["made"].sum() == p.fgm.sum() and (f["value"] == 3).sum() == p.fg3a.sum()
    # the opening possession follows the tip; the away make after its own offensive rebound
    assert f.iloc[0]["poss_start"] == "period_start"
    assert f.iloc[2]["events"] == "oreb@678" and f.iloc[2]["att_no"] == 2 and f.iloc[2]["first"] == 0
    # the and-one is flagged on its make, and only there
    assert f["post_and1"].tolist() == [0, 0, 0, 1, 0, 0]
    # a turnover with a STEAL row starts a live-ball possession; a team rebound a dead-ball one
    assert f.iloc[4]["poss_start"] == "steal"
    assert f.iloc[5]["poss_start"] == "dreb_team"
    assert f.iloc[5]["events"].startswith("foul:off:Technical@585")
    # the lead before the shot, from the shooting team's side
    assert f.iloc[1]["margin"] == -2 and f.iloc[3]["margin"] == -1
    # no scorer shot-type tag is ever stored (the owner's ruling, 2026-10-06)
    assert not any("sub" in c.lower() and "type" in c.lower() for c in f.columns)
    assert not {"subType", "description", "actionType"} & set(f.columns)


def test_derived_fields():
    gp, _, _ = _parsed(True)
    f = pd.DataFrame(gp.shot_log)
    f["x"], f["y"] = [0.0, 230.0, -10.0, 0.0, 150.0, 5.0], [0.0, 50.0, 5.0, 0.0, 120.0, 3.0]
    f["arena"], f["season"] = HOME, 2016
    d = derive(f)
    assert d.loc[2, "secs_since_oreb"] == 8.0 and np.isnan(d.loc[1, "secs_since_oreb"])
    assert d.loc[0, "noloc"] and not d.loc[2, "noloc"]
    assert d.loc[1, "corner3"] == (d.loc[1, "value"] == 3)
    assert abs(d.loc[4, "dist_xy"] - np.hypot(150, 120) / 10) < 1e-12
    assert (d["secs_into_poss"] >= 0).all()


def test_recentre_finds_a_shifted_rim():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(17.0, 3.0, n)          # an arena whose origin sits 1.7 ft left of and 2.4 ft short of the rim
    y = rng.normal(24.0, 3.0, n)
    far = rng.uniform(-50, 50, (n, 2))
    f = pd.DataFrame({"arena": 1, "season": 2024, "value": 2,
                      "x": np.r_[x, far[:, 0]], "y": np.r_[y, far[:, 1]]})
    off = recentre(f).iloc[0]
    assert abs(off.dx - 17.0) < 1.0 and abs(off.dy - 24.0) < 1.0


def test_player_game_totals():
    f = pd.DataFrame({"game_id": ["g"] * 4, "shooter": [1, 1, 2, 2], "made": [1, 0, 1, 1], "value": [3, 2, 3, 3]})
    t = player_game_totals(f).set_index("shooter")
    assert t.loc[1].tolist() == ["g", 2, 1, 1, 1] and t.loc[2, "fg3m"] == 2


def test_the_turnover_and_trip_log():
    gp, p, _ = _parsed(True)
    a = pd.DataFrame(gp.aux_log)
    tov = a[a.kind == "tov"]
    assert len(tov) == int(p.tov.sum()) and tov.iloc[0]["sub"] == "Bad Pass"
    trips = a[a.kind == "trip"]
    assert trips["sub"].tolist() == ["and1:1", "shooting:2"]


def test_the_shot_that_handed_the_possession_over_and_time_on_court():
    gp, _, _ = _parsed(True)
    f = pd.DataFrame(gp.shot_log)
    # the away miss at 680 came after the home make at 700: that make handed the ball over
    assert f.iloc[1]["start_prev_made"] == 1 and f.iloc[1]["start_prev_clock"] == 700.0
    # the home shot at 610 followed a steal: nothing handed over by a shot
    assert f.iloc[4]["start_prev_value"] == 0 and f.iloc[4]["start_prev_made"] == -1
    # Hsix came on at 590 and dunked at 570; Hone started the period and shot at 700
    assert f.iloc[5]["shooter_secs_on"] == 20.0 and f.iloc[0]["shooter_secs_on"] == 20.0


def test_rim_map_puts_a_deep_scorer_back_on_the_league_scale_and_keeps_the_home_team_style():
    rng = np.random.default_rng(3)
    rows = []
    for arena in range(6):
        for team in range(6):
            n = 600
            d = rng.gamma(2.0, 1.5, n).clip(0, 9.9)
            if arena == 0:                                       # its scorer logs close shots 1.5 ft deeper
                d = d + 1.5 * (1.0 - d / 10.0)
            ang = rng.uniform(0, np.pi, n)
            rows.append(pd.DataFrame(dict(arena=arena, season=2024, team=team, value=2,
                                          x=10 * d * np.cos(ang), y=10 * d * np.sin(ang))))
    f = pd.concat(rows, ignore_index=True)
    from eracoef.shotframe import rim_map
    rmap = rim_map(f)
    base = f.assign(events="", poss_start_clock=0.0, clock=0.0, prev_event_clock=0.0)
    raw, mapped = derive(base), derive(base, rmap=rmap)
    share = lambda d: d[d.arena == 0].dist_xy.lt(3.0).mean()
    lg = raw[raw.team != raw.arena].dist_xy.lt(3.0).mean()      # the league the map targets: every visitor
    assert abs(share(raw) - lg) > 0.1 and abs(share(mapped) - lg) < 0.03
    assert np.allclose(np.arctan2(mapped.y, mapped.x), np.arctan2(raw.y, raw.x))     # only the depth moves
