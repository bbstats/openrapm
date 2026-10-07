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


def test_the_anchor_is_the_last_row_that_cannot_belong_to_the_shot():
    gp, _, _ = _parsed(True)
    f = pd.DataFrame(gp.shot_log)
    got = list(zip(f["anchor_clock"], f["anchor_kind"]))
    assert got[0] == (720.0, "jump")                     # the opening tip
    assert got[2] == (678.0, "reb")[:1] + ("oreb",)       # the offensive rebound before the putback three
    assert got[4] == (630.0, "tov")                       # the turnover, never the blank STEAL row
    assert got[5] == (585.0, "ft")                        # the technical free throw, never the technical foul
    # and the shot's own stamp is never the anchor
    assert (f["anchor_clock"] >= f["clock"]).all()


def test_anchored_timing_never_reads_the_shots_own_stamp():
    from eracoef.shotframe import anchored
    f = pd.DataFrame({"clock": [500.0, 480.0, 300.0], "anchor_clock": [505.0, 486.0, 300.0],
                      "poss_start_clock": [510.0, 500.0, 320.0], "secs_into_poss": [10.0, 20.0, 20.0],
                      "secs_since_oreb": [3.0, np.nan, 0.0], "secs_since_timeout": [np.nan, 30.0, 1.0],
                      "shooter_secs_on": [2.0, 100.0, 50.0]})
    g = anchored(f)
    assert g["clock"].tolist() == [505.0, 486.0, 300.0]
    assert g["secs_into_poss"].tolist() == [5.0, 14.0, 20.0]
    assert g["secs_since_oreb"].iloc[2] == 0.0 and np.isnan(g["secs_since_oreb"].iloc[1])
    assert g["secs_since_timeout"].iloc[1] == 24.0
    assert g["shooter_secs_on"].tolist() == [0.0, 94.0, 50.0]          # entered between the anchor and the shot: 0
    # moving the shot's stamp (the leak) moves nothing that a model reads
    h = anchored(f.assign(clock=f["clock"] - 2.0, secs_into_poss=f["secs_into_poss"] + 2.0,
                          secs_since_oreb=f["secs_since_oreb"] + 2.0, secs_since_timeout=f["secs_since_timeout"] + 2.0,
                          shooter_secs_on=f["shooter_secs_on"] + 2.0))
    for c in ("clock", "secs_into_poss", "secs_since_oreb", "secs_since_timeout", "shooter_secs_on"):
        assert np.allclose(g[c], h[c], equal_nan=True), c


def test_events_after_the_anchor_are_only_fouls_and_violations():
    gp, _, _ = _parsed(True)
    f = pd.DataFrame(gp.shot_log)
    for ev, n in zip(f["events"].fillna(""), f["anchor_nev"]):
        toks = [t.rpartition("@")[0] for t in ev.split(";")] if ev else []
        assert 0 <= n <= len(toks)
        assert all(k.startswith(("foul:", "viol:")) for k in toks[n:]), (ev, n)


def test_the_spot_inside_ten_feet_is_coded_to_two_zones_and_the_logged_one_kept():
    from eracoef.shotframe import code_location
    f = pd.DataFrame({"dist_xy": [0.0, 1.2, 5.9, 6.1, 9.9, 10.0, 24.0], "xc": [0.0, 5.0, -40.0, 30.0, 90.0, 0.0, 220.0],
                      "yc": [0.0, 11.0, 45.0, 53.0, 40.0, 100.0, 80.0], "angle": [0.0, 24.0, 41.0, 29.0, 66.0, 0.0, 70.0],
                      "noloc": [True, False, False, False, False, False, False]})
    g = code_location(f)
    assert g["dist_xy"].tolist() == [0.0, 3.0, 3.0, 8.0, 8.0, 10.0, 24.0]     # unlocated and 10+ ft untouched
    assert g["xc"].tolist()[1:5] == [0.0] * 4 and g["angle"].tolist()[1:5] == [0.0] * 4
    assert np.allclose(np.hypot(g["xc"], g["yc"])[1:5] / 10.0, g["dist_xy"][1:5])   # a spot rebuilt from x, y agrees
    assert g["dist_xy_logged"].tolist() == f["dist_xy"].tolist() and g["xc_logged"].tolist() == f["xc"].tolist()
    # a made 4.3 ft shot logged at 3.5 ft no longer moves: both are the 0-6 zone
    assert code_location(f.iloc[[2]].assign(dist_xy=4.3))["dist_xy"].iloc[0] == code_location(f.iloc[[2]].assign(dist_xy=3.5))["dist_xy"].iloc[0]


def test_every_logged_attempt_lands_in_its_stint_slot():
    from eracoef.stints import stint_slots
    gp, p, st = _parsed(True)
    log = stint_slots(gp, p, pd.DataFrame(gp.shot_log))
    placed = log[log["stint_no"] >= 0]
    assert len(placed) > 0 and (log["stint_no"] < len(st)).all()
    for kind, v in (("2", 2), ("3", 3)):
        got = placed[placed["value"] == v].groupby(["stint_no", "stint_side", "stint_slot"]).size()
        for (no, side, slot), n in got.items():
            assert st[f"fg{kind}a_s{slot}_{side}"].iat[no] == n, (kind, no, side, slot)
        # and every attempt the stints counted is in the log
        total = sum(st[f"fg{kind}a_s{s}_{side}"].sum() for s in ("1", "2", "3", "4", "5", "x") for side in ("h", "a"))
        assert total == got.sum()


def test_an_attempt_whose_possession_never_closes_is_in_no_stint():
    # the feed has no end row for period 1, and the away team's last miss is still open when period 2 starts: the
    # parser drops that possession (no record), so the attempt must not be filed under period 2's first stint
    from eracoef.stints import stint_slots
    from test_stints import _ev
    g = synthetic_game()
    g = g[~((g.actionType == "period") & (g.subType == "end"))]
    extra = pd.DataFrame([
        _ev(1, 3, AWAY, 12, "Missed Shot", "Jump Shot", "MISS Atwo 24' 3PT Jump Shot", 3),
        _ev(2, 720, 0, 0, "period", "start", "Start of 2nd Period"),
        _ev(2, 700, HOME, 6, "Made Shot", "Jump Shot", "Hsix 15' Jump Shot (2 PTS)", 2, 9, 5),
        _ev(2, 680, AWAY, 12, "Missed Shot", "Jump Shot", "MISS Atwo 20' Jump Shot", 2),
        _ev(2, 678, HOME, 6, "Rebound", "Unknown", "Hsix REBOUND (Off:0 Def:1)"),
        _ev(2, 0, 0, 0, "period", "end", "End of 2nd Period")])
    g = pd.concat([g, extra], ignore_index=True)
    g["actionNumber"] = np.arange(1, len(g) + 1)
    gp = GameParser(g, _box(), HOME, AWAY, "0020000001", log_shots=True)
    p = gp.possessions()
    st = gp.stints(p)
    log = stint_slots(gp, p, pd.DataFrame(gp.shot_log))
    late = log[(log["period"] == 1) & (log["value"] == 3) & (log["shooter"] == 12) & (log["made"] == 0)]
    # the dropped possession appended no record, so the attempt has none (the old rule, len(recs) at the shot,
    # pointed it at period 2's first record) and sits in no stint
    assert len(late) == 1 and late["rec_idx"].iat[0] == -1 and late["stint_no"].iat[0] == -1
    assert (log.loc[log["rec_idx"] >= 0, "rec_idx"].map(p["period"]) == log.loc[log["rec_idx"] >= 0, "period"]).all()
    # whatever the parser does, the placed attempts rebuild every slot count exactly
    placed = log[log["stint_no"] >= 0]
    for kind, v in (("2", 2), ("3", 3)):
        got = placed[placed["value"] == v].groupby(["stint_no", "stint_side", "stint_slot"]).size()
        total = sum(st[f"fg{kind}a_s{s}_{side}"].sum() for s in ("1", "2", "3", "4", "5", "x") for side in ("h", "a"))
        assert total == got.sum(), kind
        for (no, side, slot), n in got.items():
            assert st[f"fg{kind}a_s{slot}_{side}"].iat[no] == n
