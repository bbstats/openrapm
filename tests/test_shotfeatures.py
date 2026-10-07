"""The search's new play-by-play features (eracoef.shotfeatures) and their same-second audit (scripts/128).

Each block builds the same columns whatever rows it is handed, reads the events string under the same-second rule,
takes the previous attempt from the same possession in game order, and plugs into shotmodel.fit; the boosters' inputs
refuse a banned name; the audit passes the rule and catches a feature that breaks it."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eracoef import shotfeatures as sf
from eracoef import shotmodel as sm
from eracoef.shotclock import ClockRules

ROOT = Path(__file__).resolve().parents[1]

BASE = dict(season=2010, phase="RS", game_id="0021000001", action_number=1, half="A", arena=1, team=1, opp=2,
            shooter=10, period=1, clock=600.0, value=2, made=0, xc=0.0, yc=100.0, margin=0, poss_start="made_fg",
            poss_no=1, att_no=1, fga_no=1, n_oreb=0, secs_into_poss=5.0, secs_since_oreb=np.nan,
            secs_since_timeout=np.nan, poss_start_clock=605.0, start_prev_value=0, start_prev_dist=np.nan,
            start_prev_made=-1, start_prev_blocked=-1, start_prev_x=np.nan, start_prev_y=np.nan,
            start_prev_clock=np.nan, events="", post_blocked=0, post_and1=0, sc_eff=12.0, clock_off=False,
            reset_kind="made_fg", since_reset=5.0, shooter_secs_on=100.0, heave=False)


def _geometry(f):
    x, y = f["xc"].to_numpy(float), f["yc"].to_numpy(float)
    return f.assign(dist_xy=np.hypot(x, y) / 10.0, angle=np.degrees(np.arctan2(np.abs(x), y)),
                    corner3=(f["value"].to_numpy() == 3) & (np.abs(x) >= 220) & (y <= 92.5), noloc=(x == 0) & (y == 0))


def _rows(*over):
    return _geometry(pd.DataFrame([{**BASE, **o} for o in over]))


def _spot(rng, value):
    if value == 3:
        if rng.random() < 0.25:
            return float(rng.choice([-1, 1]) * rng.uniform(221, 235)), float(rng.uniform(-30, 90))
        r, a = rng.uniform(237.5, 270), np.radians(rng.uniform(-68, 68))
    else:
        r, a = rng.uniform(0, 215), np.radians(rng.uniform(-90, 90))
    return float(r * np.sin(a)), float(r * np.cos(a))


def _table(seed=0, season=2010, games=12):
    """A season table with the possession structure the logger writes: attempts of a possession in clock order with
    offensive rebounds between them, earlier timeouts and fouls, and some events logged at the shot's own second.
    Shots after an offensive rebound are planted 0.8 harder on the logit.  Rows come back shuffled."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(games):
        gid = f"00{season % 100:02d}{seed:02d}{g:04d}"
        an, poss = 0, 0
        for period in (1, 2, 3, 4):
            clock, last = 720.0, None
            while clock > 20.0:
                poss += 1
                start = "period_start" if last is None else ("made_fg" if last[4] == 1 else "dreb")
                sp = last if start in ("dreb", "made_fg") else None
                start_clock = clock - (1.0 if start == "dreb" else 0.0)
                if start == "dreb" and rng.random() < 0.08:
                    start_clock = clock - 9.0                     # a missed free throw's rebound: a stale start
                margin = int(rng.integers(-8, 9))
                events = []
                if rng.random() < 0.15:
                    events.append(("timeout", start_clock - 1.0))
                c, oreb_at = start_clock, np.nan
                n_att = int(rng.choice([1, 2, 3], p=[0.7, 0.22, 0.08]))
                for k in range(n_att):
                    c -= float(rng.integers(2, 9) if k == 0 else rng.integers(0, 6))   # 0: a tip at the rebound's second
                    if c <= 0:
                        break
                    if rng.random() < 0.10:
                        events.append(("foul:def:Personal", c + 1.0))
                    if rng.random() < 0.04:
                        events.append(("foul:def:Shooting", c + 2.0))
                    if rng.random() < 0.08:
                        events.append(("foul:def:Personal", c))     # logged before the shot, at its own second
                    if rng.random() < 0.08:
                        events.append(("timeout", c))
                    value = 3 if rng.random() < 0.35 else 2
                    x, y = _spot(rng, value)
                    d = np.hypot(x, y) / 10.0
                    made = int(rng.random() < 1.0 / (1.0 + np.exp(-(0.3 - 0.05 * d - 0.8 * (k >= 1)))))
                    blocked = int(not made and rng.random() < 0.07)
                    tos = [ec for ek, ec in events if ek == "timeout"]
                    an += 1 + int(rng.integers(0, 3))
                    rows.append(dict(
                        season=season, phase="RS", game_id=gid, action_number=an, half="A" if g % 2 else "B",
                        arena=g % 6, team=int(rng.integers(1, 7)), opp=0, shooter=int(rng.integers(0, 60)),
                        period=period, clock=c, value=value, made=made, xc=x, yc=y, margin=margin, poss_start=start,
                        poss_no=poss, att_no=k + 1, fga_no=k + 1, n_oreb=k, secs_into_poss=start_clock - c,
                        secs_since_oreb=(oreb_at - c) if k else np.nan,
                        secs_since_timeout=(tos[-1] - c) if tos else np.nan, poss_start_clock=start_clock,
                        start_prev_value=sp[0] if sp else 0, start_prev_dist=round(sp[1]) if sp else np.nan,
                        start_prev_x=sp[2] if sp else np.nan, start_prev_y=sp[3] if sp else np.nan,
                        start_prev_made=sp[4] if sp else -1, start_prev_blocked=sp[5] if sp else -1,
                        start_prev_clock=sp[6] if sp else np.nan,
                        events=";".join(f"{ek}@{ec:g}" for ek, ec in events), post_blocked=blocked, post_and1=0,
                        sc_eff=float(np.clip(24.0 - (start_clock - c), 0, 24)), clock_off=False, reset_kind=start,
                        since_reset=start_clock - c, shooter_secs_on=float(rng.uniform(0, 900)), heave=False))
                    last = (value, d, x, y, made, blocked, c)
                    if made:
                        break
                    oreb_at = c - 1.0
                    if k < n_att - 1:
                        events.append(("oreb", oreb_at))
                        c = oreb_at
                clock = c
    f = _geometry(pd.DataFrame(rows))
    return f.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _script():
    spec = importlib.util.spec_from_file_location("feature_audit_128", ROOT / "scripts" / "128_feature_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------ the blocks
def test_every_block_builds_the_same_columns_whatever_rows_it_is_handed():
    a, b = sf.prepare(_table(0)), sf.prepare(_table(1, games=3))
    for name in sf.BLOCKS:
        for sub in sm.SUBMODELS:
            ma, mb = sm.submodel_of(a) == sub, sm.submodel_of(b) == sub
            Xa, na = sm.block_columns(a[ma], sub, name)            # shotmodel dispatches to shotfeatures
            Xb, nb = sf.BLOCKS[name](b[mb].head(3), sub)
            assert na == nb and len(set(na)) == len(na), (name, sub)
            assert Xa.shape == (int(ma.sum()), len(na)) and Xb.shape == (3, len(nb))
            assert np.isfinite(Xa).all() and np.isfinite(Xb).all(), (name, sub)
    assert sf.line(a, "rim")[1] == [] and "scr_kickout" in sf.scramble(a, "three")[1]
    assert "scr_kickout" not in sf.scramble(a, "mid")[1]
    names = [n for b in sf.BLOCKS for n in sf.BLOCKS[b](a, "three")[1]] + sm.design(a.head(5), "three", sm.BLOCKS)[1]
    assert len(names) == len(set(names))                         # no name clashes with the shipped blocks


def test_late_flags_the_last_minutes_of_a_close_game_and_the_two_for_one():
    f = _rows(dict(period=4, clock=100.0, margin=-2), dict(period=4, clock=250.0, margin=4),
              dict(period=3, clock=100.0, margin=0), dict(period=2, clock=35.0, secs_into_poss=5.0),
              dict(period=5, clock=60.0, margin=2), dict(period=1, clock=30.0, secs_into_poss=12.0))
    X, names = sf.late(f, "mid")
    got = pd.DataFrame(X, columns=names).astype(int)
    assert got.loc[0].to_dict() == dict(late5_c5=1, late5_c3=1, late2_c5=1, late2_c3=1, late2_trail13=1,
                                        late2_lead13=0, two_for_one=0, two_for_one_quick=0)
    assert got.loc[1].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert got.loc[2].sum() == 0                                  # the third quarter is not late
    assert got.loc[3].tolist() == [0] * 6 + [1, 1]
    assert got.loc[4, "late2_lead13"] == 1                        # overtime counts
    assert got.loc[5].tolist() == [0] * 6 + [1, 0]                # in the window, but not early in the possession


def test_the_events_are_read_under_the_same_second_rule():
    f = _rows(dict(clock=20.0, events="timeout@30;foul:def:Personal@25;foul:def:Shooting@24;viol:def:Kicked Ball@23;"
                                       "foul:def:Personal@20;timeout@20;viol:def:Defensive Goaltending@20;oreb@20"),
              dict(clock=20.0, events=""),
              dict(clock=5.0, events="foul:def:Loose Ball@9;foul:def:Personal@8;timeout@7.5"))
    ef = sf.event_fields(f)
    assert ef["timeout_secs"].iloc[0] == 10.0                    # the timeout at the shot's own second is ignored
    assert ef["n_dfoul"].iloc[0] == 1                            # not the shooting foul, the kick, or the same-second foul
    assert np.isnan(ef["timeout_secs"].iloc[1]) and ef["n_dfoul"].iloc[1] == 0
    assert ef["timeout_secs"].iloc[2] == 2.5 and ef["n_dfoul"].iloc[2] == 2
    A, an = sf.ato(f, "three")
    assert an == ["ato_0_6", "ato_6_12", "ato_12_24", "ato_24p"] and A.tolist() == [[0, 1, 0, 0], [0] * 4, [1, 0, 0, 0]]
    D, dn = sf.dfoul(f, "rim")
    assert dn == ["dfoul_1", "dfoul_2p"] and D.tolist() == [[1, 0], [0, 0], [0, 1]]
    # the precomputed fields are used when present, and agree
    assert np.array_equal(sf.ato(sf.prepare(f.assign(poss_no=[1, 2, 3])), "rim")[0], A)


def test_the_previous_attempt_is_the_last_one_of_the_same_possession_in_game_order():
    f = _rows(dict(game_id="g1", poss_no=4, period=2, clock=300.0, action_number=10, value=2, xc=0.0, yc=20.0, post_blocked=1),
              # the same second, the later attempt (fga_no 3) logged with the LOWER action number: order by the attempt count
              dict(game_id="g1", poss_no=4, period=2, clock=297.0, action_number=13, value=3, xc=0.0, yc=250.0, fga_no=2),
              dict(game_id="g1", poss_no=4, period=2, clock=297.0, action_number=12, value=2, xc=10.0, yc=0.0, fga_no=3),
              dict(game_id="g1", poss_no=5, period=2, clock=280.0, action_number=15),
              dict(game_id="g2", poss_no=4, period=2, clock=299.0, action_number=11))
    f = f.iloc[[3, 2, 0, 4, 1]].set_index(pd.Index([30, 20, 0, 40, 10]))      # shuffled, with its own index
    p = sf.add_previous_attempt(f)
    assert list(p.index) == [30, 20, 0, 40, 10] and list(p.columns[: f.shape[1]]) == list(f.columns)
    assert p.loc[0, "prev_att_value"] == 0 and np.isnan(p.loc[0, "prev_att_dist"]) and p.loc[0, "prev_att_blocked"] == -1
    assert p.loc[10, "prev_att_value"] == 2 and p.loc[10, "prev_att_dist"] == pytest.approx(2.0)
    assert p.loc[10, "prev_att_blocked"] == 1 and p.loc[10, "prev_att_clock"] == 300.0
    assert p.loc[20, "prev_att_value"] == 3 and p.loc[20, "prev_att_clock"] == 297.0     # same second: attempt order
    assert p.loc[30, "prev_att_value"] == 0 and p.loc[40, "prev_att_value"] == 0        # new possession; other game
    # the planted table: every later attempt finds the attempt before it
    t = sf.add_previous_attempt(_table(0))
    later = t["fga_no"] >= 2
    assert (t.loc[later, "prev_att_value"] > 0).all() and (t.loc[~later, "prev_att_value"] == 0).all()
    assert (t.loc[later, "prev_att_clock"] >= t.loc[later, "clock"]).all()


def test_the_previous_attempt_needs_post_blocked_and_the_scramble_block_needs_the_previous_attempt():
    f = _rows(dict(), dict(clock=595.0, action_number=2, fga_no=2)).drop(columns=["post_blocked"])
    with pytest.raises(KeyError, match="post_blocked"):
        sf.add_previous_attempt(f, require_blocked=True)
    p = sf.add_previous_attempt(f)                               # -1: not known, for every row
    assert p["prev_att_blocked"].tolist() == [-1, -1] and p["prev_att_value"].tolist() == [0, 2]
    with pytest.raises(KeyError, match="post_blocked"):           # the block that reads it refuses
        sf.scramble(p, "mid")
    assert sf.late(p, "mid")[0].shape == (2, 8)                   # the rest do not care
    with pytest.raises(KeyError, match="add_previous_attempt"):
        sf.scramble(_rows(dict()), "rim")
    with pytest.raises(KeyError, match="add_previous_attempt"):
        sf.tree_features(_rows(dict()))


def test_scramble_reads_the_rebounds_the_previous_attempt_and_the_kick_out():
    f = _rows(dict(poss_no=1, clock=400.0, action_number=1, value=2, xc=0.0, yc=150.0, post_blocked=1),
              dict(poss_no=1, clock=398.0, action_number=3, value=3, xc=0.0, yc=250.0, n_oreb=1, att_no=2, fga_no=2,
                   secs_since_oreb=1.0),
              dict(poss_no=1, clock=392.0, action_number=5, value=3, xc=0.0, yc=250.0, n_oreb=2, att_no=3, fga_no=3,
                   secs_since_oreb=5.0))
    p = sf.add_previous_attempt(f)
    X, names = sf.scramble(p, "three")
    got = pd.DataFrame(X, columns=names).astype(int)
    assert got.loc[0].sum() == 0
    assert got.loc[1].to_dict() == dict(scr_oreb1=1, scr_oreb2p=0, scr_att2p=1, scr_prev_rim=0, scr_prev_mid=1,
                                        scr_prev_three=0, scr_prev_blocked=1, scr_prev_0_3=1, scr_prev_3_8=0,
                                        scr_kickout=1)
    assert got.loc[2].to_dict() == dict(scr_oreb1=0, scr_oreb2p=1, scr_att2p=1, scr_prev_rim=0, scr_prev_mid=0,
                                        scr_prev_three=1, scr_prev_blocked=0, scr_prev_0_3=0, scr_prev_3_8=1,
                                        scr_kickout=0)


def test_the_line_is_measured_at_the_shot_s_own_angle_and_era():
    f = _rows(dict(value=3, xc=230.0, yc=50.0), dict(value=3, xc=0.0, yc=250.0),
              dict(value=3, xc=0.0, yc=230.0, season=1997), dict(value=2, xc=0.0, yc=230.0),
              dict(value=2, xc=-210.0, yc=40.0), dict(value=2, xc=0.0, yc=0.0), dict(value=2, xc=0.0, yc=215.0))
    b = sf.line_ft(f)
    assert b[:5] == pytest.approx([1.0, 1.25, 1.0, -0.75, -1.0]) and np.isnan(b[5])
    X, names = sf.line(f, "three")
    assert names == ["line_in", "line_0_05", "line_05_1", "line_1_2", "line_2_4"]
    assert X[0].tolist() == [0, 0, 0, 1, 0] and X[3].tolist() == [1, 0, 0, 0, 0] and X[5].sum() == 0
    M, mn = sf.line(f, "mid")
    assert mn == ["line_toe", "line_in_1_3"] and M[3:7].tolist() == [[1, 0], [1, 0], [0, 0], [0, 1]]
    assert sf.line(f, "rim")[0].shape == (len(f), 0)


def test_prevloc_reads_the_missed_shot_that_handed_the_ball_over_and_flags_a_stale_start():
    miss3 = dict(poss_start="dreb", start_prev_value=3, start_prev_made=0, start_prev_dist=25.0, start_prev_x=-150.0,
                 start_prev_y=190.0, start_prev_clock=500.0, poss_start_clock=499.0)
    f = _rows(dict(miss3, clock=495.0, xc=-50.0, yc=10.0),
              dict(miss3, clock=489.0, xc=50.0, yc=10.0, start_prev_value=2, start_prev_x=0.0, start_prev_y=0.0,
                   start_prev_dist=1.0),
              dict(miss3, clock=480.0, poss_start_clock=491.0),                   # 9 s later: a free throw's rebound
              dict(miss3, clock=480.0, start_prev_made=1),                        # a make cannot start a dreb possession
              dict(miss3, poss_start="made_fg", start_prev_made=1, clock=490.0))
    X, names = sf.prevloc(f, "rim")
    got = pd.DataFrame(X, columns=names).astype(int)
    assert got.loc[0].to_dict() == dict(pl_rim=0, pl_short=0, pl_long2=0, pl_three=1, pl_same_side=1, pl_tr_0_4=0,
                                        pl_tr_4_7=1, pl_tr_7_12=0, pl_stale=0)
    assert got.loc[1].to_dict() == dict(pl_rim=1, pl_short=0, pl_long2=0, pl_three=0, pl_same_side=0, pl_tr_0_4=0,
                                        pl_tr_4_7=0, pl_tr_7_12=1, pl_stale=0)
    assert got.loc[2].tolist() == [0] * 8 + [1] and got.loc[3].tolist() == [0] * 8 + [1]
    assert got.loc[4].sum() == 0


def test_side_is_the_sign_of_x_and_its_angle():
    f = _rows(dict(xc=-100.0, yc=0.0), dict(xc=100.0, yc=100.0), dict(xc=0.0, yc=0.0))
    X, names = sf.side(f, "mid")
    assert names == ["side", "side_angle"] and np.allclose(X, [[-1.0, -1.0], [1.0, 0.5], [0.0, 0.0]])


def test_the_new_blocks_fit_and_price_through_shotmodel_and_carry_their_signal():
    train = pd.concat([sf.prepare(_table(s, season=y, games=30)) for s, y in ((2, 2001), (3, 2002))], ignore_index=True)
    test = sf.prepare(_table(4, season=2010, games=30))
    after = (test["n_oreb"] >= 1).to_numpy()
    gaps = {}
    for blocks in (["spot"], ["spot"] + list(sf.BLOCKS)):
        model = sm.fit(train, blocks, rounds=2)
        eta = model.predict_raw(test)
        assert np.isfinite(eta).all()
        gaps[len(blocks)] = eta[after].mean() - eta[~after].mean()
    assert "late2_c3" in model.coef["mid"][0] and "scr_kickout" in model.coef["three"][0]
    assert abs(gaps[1]) < 0.2 and gaps[1 + len(sf.BLOCKS)] < -0.5        # planted -0.8 after an offensive rebound


def test_tree_features_are_a_whitelist_that_refuses_a_banned_name():
    t = sf.prepare(_table(0, games=2))
    X = sf.tree_features(t)
    assert list(X.columns) == list(sf.TREE_RAW) + list(sf.TREE_DERIVED) and len(X) == len(t)
    assert not [c for c in X.columns if sf.FORBIDDEN.search(c)]
    assert set(X["sub"]) <= {0, 1, 2} and (X["poss_start"] >= 0).all() and (X["reset_kind"] >= 0).all()
    assert X["prev_att_value"].max() in (2, 3) and np.isfinite(sf.tree_features(t, fill=-1.0).to_numpy()).all()
    for bad in ("post_and1", "arena", "x", "dist", "lp", "shooter", "season", "off1", "description"):
        with pytest.raises(ValueError, match="banned"):
            sf.tree_features(t.assign(**{bad: 0}), extra=[bad])
    assert "margin2" in sf.tree_features(t.assign(margin2=1), extra=["margin2"]).columns


# ------------------------------------------------------------------------------------------ the audit (scripts/128)
def test_stripping_keeps_offensive_rebounds_and_earlier_events():
    a = _script()
    f = _rows(dict(clock=20.0, events="timeout@25;oreb@20;foul:def:Personal@20;viol:def:Defensive Goaltending@20"))
    assert a.strip_same_second(f)["events"].iloc[0] == "timeout@25;oreb@20"
    assert a.strip_same_second(f, tol=5.0)["events"].iloc[0] == "oreb@20"


def test_the_audit_passes_the_rule_and_catches_features_that_break_it():
    a = _script()
    t = sf.prepare(_table(0))
    builders = a.builders_for(["late", "scramble", "ato", "dfoul", "line", "prevloc", "side", "putback"])
    rep = a.same_second_audit(t, builders)
    assert rep["changed"].sum() == 0 and set(rep["block"]) == set(builders)

    def leak_timeout(f, sub):                     # shotframe.derive's secs_since_timeout reads every timeout
        return sf._bands(f["secs_since_timeout"].to_numpy(float), (0.0, 6.0, 1e9)), ["to_0_6", "to_6p"]

    def leak_fouls(f, sub):                       # every defensive foul token, the shot's own second included
        return f["events"].fillna("").str.count("foul:def:Personal").to_numpy(float)[:, None], ["n_personal"]

    rep = a.same_second_audit(t, {"leak_timeout": leak_timeout, "leak_fouls": leak_fouls})
    assert (rep.groupby("block")["changed"].sum() > 0).all()
    # the shipped clock, rebuilt: since the review shotclock.rebuild drops a non-shooting foul logged at the shot's own
    # clock reading, so the clock block is same-second clean too -- 15 s into the possession, the foul at the shot
    late_foul = sf.prepare(_rows(dict(poss_start_clock=600.0, clock=585.0, secs_into_poss=15.0,
                                      events="foul:def:Personal@585")))
    rep = a.same_second_audit(late_foul, a.builders_for(["clock", "dfoul"]), rules=ClockRules())
    assert rep.groupby("block")["changed"].max().to_dict() == {"clock": 0, "dfoul": 0}
    # a wider window is a stress test: the rule's own features then move too
    assert a.same_second_audit(t, a.builders_for(["dfoul"]), tol=1.5)["changed"].sum() > 0


def test_a_season_s_audit_links_a_leak_to_the_result_and_not_the_rule_s_own_features():
    a = _script()
    t = sf.prepare(_table(0, games=30))
    own = np.array([any(k == "foul:def:Personal" and c == clk for k, c in sf._tokens(e))
                    for e, clk in zip(t["events"], t["clock"])])
    t["post_and1"] = (own & (t["made"] == 1)).astype(int)        # the and-one foul, logged at the make's own second

    def leak(f, sub):
        return (f["events"].fillna("").str.count("foul:def:Personal").to_numpy(float) > 0)[:, None].astype(float), ["leak"]

    rng = np.random.default_rng(1)
    read = rng.random(len(t)) < 0.5                                # the games whose play-by-play was read
    tags = pd.DataFrame({k: np.where(read, rng.random(len(t)) < 0.3, None) for k in a.TAGS})
    res = a.audit_season(t, {**a.builders_for(["dfoul", "late"]), "leak": leak}, 2010, q=rng.random(len(t)), tags=tags)
    L = res["link"][res["link"]["outcome"] == "post_and1"].set_index(["block", "column"]).sort_index()
    assert (L.loc[("leak", "leak"), "diff"] > 0.2).all()          # every sub-model
    ok = L.loc["dfoul"]
    assert (ok["diff"].abs() < 4 * ok["se"]).all(), ok
    changed = res["same_second"].groupby("block")["changed"].sum()
    assert changed["dfoul"] == 0 and changed["late"] == 0 and changed["leak"] > 0
    assert set(res["tagproxy"]["tag"]) == set(a.TAGS) and len(res["team"]) == len(res["link"]) // 2
    assert res["tagproxy"]["n_active"].max() < (t["n_dfoul"] > 0).sum()   # only the rows whose words were read


def test_the_links_and_the_team_share_find_what_was_planted():
    a = _script()
    rng = np.random.default_rng(0)
    n = 40000
    cells = rng.integers(0, 5, n)
    active = rng.random(n) < 0.3
    y = rng.random(n) < (0.1 + 0.05 * cells + 0.08 * active)
    r = a.stratified_diff(active, y, cells)
    assert abs(r["diff"] - 0.08) < 3 * r["se"] and r["n_active"] == int(active.sum())
    null = a.stratified_diff(active, rng.random(n) < 0.1 + 0.05 * cells, cells)
    assert abs(null["diff"]) < 3 * null["se"]
    teams = rng.integers(0, 30, n)
    habit = rng.random(n) < np.linspace(0.05, 0.6, 30)[teams]
    assert a.team_concentration(habit, teams)["real_share"] > 0.9
    assert a.team_concentration(rng.random(n) < 0.3, teams)["real_share"] < 0.3


def test_the_tag_reader_finds_the_scorer_s_words():
    a = _script()
    tg = a.tags_of(["Driving Layup Shot", "Dunk", "Tip Layup Shot", "Pullup Jump shot", None],
                   ["Smith 2' Driving Layup", "Jones Alley Oop Dunk", "", "", "Brown 3PT Step Back Jump Shot"])
    assert tg["layup"].tolist() == [True, False, True, False, False]
    assert tg["dunk"].tolist() == [False, True, False, False, False] and tg["alley_oop"].tolist()[1]
    assert tg["tip"].tolist()[2] and tg["jump"].tolist() == [False, False, False, True, True]


def test_no_feature_block_shadows_a_shipped_one_and_the_label_is_banned():
    from eracoef import shotmodel as sm
    assert not set(sf.BLOCKS) & set(sm.BLOCKS + sm.EXTRA_BLOCKS)
    for bad in ("made", "prev_event_clock", "secs_since_timeout", "home", "phase", "half", "sc_raw"):
        with pytest.raises(ValueError):
            sf.check_names([bad])


def test_the_anchors_own_timeout_counts_and_a_foul_after_the_anchor_does_not():
    f = pd.DataFrame({"events": ["timeout@500;foul:def:Personal@495"], "clock": [500.0], "anchor_nev": [1]})
    got = sf.event_fields(f)
    assert got.loc[0, "timeout_secs"] == 0.0 and got.loc[0, "n_dfoul"] == 0
    assert np.isnan(sf.event_fields(f.drop(columns="anchor_nev")).loc[0, "timeout_secs"])   # the old rule lost it


def test_the_logged_spot_and_the_stamp_timing_are_banned_inputs():
    for c in ("dist_xy_logged", "xc_logged", "angle_logged", "clock_stamp", "sc_eff_stamp"):
        assert sf.FORBIDDEN.search(c), c
    for c in ("dist_xy", "xc", "angle", "clock", "sc_eff"):
        assert not sf.FORBIDDEN.search(c), c
