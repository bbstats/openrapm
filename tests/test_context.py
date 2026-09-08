"""context.py: the destination's usage minutes and quality, every teammate measured in the feature window."""
import numpy as np
import pandas as pd

from eracoef.context import DEST, FLOOR, destination_features, destination_inputs, usage_minutes_panel

WINS = {"2000-2002": [2000, 2001, 2002], "2003-2005": [2003, 2004, 2005]}


def _panel():
    rows = []
    # window 1: three players; usage minutes 30 / 20 / 10, APM 5 / 0 / -5; window 2: only the first two
    for w, ids, ums, apms in (("2000-2002", [1, 2, 3], [30.0, 20.0, 10.0], [5.0, 0.0, -5.0]),
                              ("2003-2005", [1, 2], [25.0, 22.0], [4.0, 1.0])):
        for pid, um, a in zip(ids, ums, apms):
            r = dict(window=w, side="O", player_id=pid, poss=1000.0, poss_pct=0.5, apm=a)
            for c in ("fg2m", "fg2_miss", "fg3m", "fg3_miss", "ftm", "ft_miss", "tov", "blk", "orb", "drb"):
                r[f"raw_{c}"] = 0.0
            r["raw_tov"] = um / 0.5                                 # usage = raw_tov alone here, so um = usage x share
            rows.append(r)
    return pd.DataFrame(rows)


def _tm():
    # in 2003-2005 player 1 plays 300 shared possessions beside 2 and 100 beside 9 (never in the panel)
    return pd.DataFrame({"season": [2003, 2003, 2004], "player_id": [1, 1, 1], "teammate_id": [2, 9, 2], "shared": [200.0, 100.0, 100.0]})


def test_usage_minutes_and_destination_features():
    um = usage_minutes_panel(_panel())
    assert um[(um.window == "2000-2002") & (um.player_id == 1)].um.iloc[0] == 30.0
    keys = pd.DataFrame({"player_id": [1], "window": ["2000-2002"], "window_to": ["2003-2005"]})
    f = destination_features(um, _tm(), WINS, keys)
    assert set(DEST) <= set(f.columns) and f.own_um[0] == 30.0
    # teammates in the TARGET window (2 for 300, 9 for 100), each valued in the FEATURE window: 2 -> um 20, apm 0;
    # 9 is unseen -> the feature window's league values (um 20, apm 0, possession-weighted over the three)
    assert abs(f.dest_um[0] - FLOOR * (300 * 20.0 + 100 * 20.0) / 400) < 1e-9
    assert abs(f.dest_apm[0] - (300 * 0.0 + 100 * 0.0) / 400) < 1e-9
    # a player with no target-window teammates at all takes league values
    keys2 = pd.DataFrame({"player_id": [3], "window": ["2000-2002"], "window_to": ["2003-2005"]})
    g = destination_features(um, _tm(), WINS, keys2)
    assert g.own_um[0] == 10.0 and abs(g.dest_um[0] - FLOOR * 20.0) < 1e-9 and abs(g.dest_apm[0]) < 1e-9


def test_destination_inputs_actual_average_and_roster():
    block = pd.DataFrame({"player_id": [1, 2, 3], "um": [30.0, 20.0, 10.0], "apm_o": [5.0, 0.0, -5.0], "poss": [1000.0] * 3})
    tm = pd.DataFrame({"season": [2006, 2006], "player_id": [1, 1], "teammate_id": [3, 2], "shared": [300.0, 100.0]})
    a = destination_inputs(block, tm, [2006], [1, 2, 99])
    assert set(DEST) <= set(a.columns) and a.own_um.tolist() == [30.0, 20.0, 20.0]
    assert abs(a.dest_um[0] - FLOOR * (300 * 10 + 100 * 20) / 400) < 1e-9 and abs(a.dest_apm[0] - (300 * -5 + 100 * 0) / 400) < 1e-9
    assert abs(a.dest_um[1] - FLOOR * 20.0) < 1e-9                      # player 2 has no rows in tm: league values
    avg = destination_inputs(block, tm, [2006], [1, 2], override={"dest_um": 77.0, "dest_apm": 1.5})
    assert (avg.dest_um == 77.0).all() and (avg.dest_apm == 1.5).all() and avg.own_um.tolist() == [30.0, 20.0]
    ro = destination_inputs(block, tm, [2006], [1, 3], override={"roster": [1, 2, 3], "weights": [1.0, 1.0, 1.0]})
    assert abs(ro.dest_um[0] - FLOOR * (20 + 10) / 2) < 1e-9             # player 1 traded to the roster {1,2,3}: beside 2 and 3
    assert abs(ro.dest_um[1] - FLOOR * (30 + 20) / 2) < 1e-9             # player 3: beside 1 and 2
