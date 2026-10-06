"""The movement parser's release finder on a synthetic possession (eracoef.movement)."""
import numpy as np
import pandas as pd

from eracoef.movement import ReleaseRules, shot_rows


def _possession():
    """25 frames a second from game clock 700.0: the shooter (id 1) dribbles twice at (70, 25), rises and shoots at
    700.0 - 3.0; the ball flies 1 s to the hoop at (88.75, 25).  A defender (id 11) closes from 8 ft to 3 ft."""
    n = 6 * 25
    gc = 700.0 - np.arange(n) / 25.0
    release = 75                                              # frame of the release, game clock 697.0
    ball = np.zeros((n, 3))
    for i in range(n):
        if i <= release:
            ball[i] = (70.5, 25.0, 3.0 if i < 40 else 8.0)
            if i in (10, 25):                                 # two bounces
                ball[i, 2] = 0.8
        elif i <= release + 25:
            a = (i - release) / 25.0
            ball[i] = (70.5 + a * (88.75 - 70.5), 25.0, 8.0 + 6.0 * np.sin(np.pi * a) + 2.0 * a)
        else:
            ball[i] = (85.0, 20.0, 3.0)
    pid = np.tile(np.r_[1, 2, 3, 4, 5, 11, 12, 13, 14, 15], (n, 1))
    team = np.tile(np.r_[[100] * 5, [200] * 5], (n, 1))
    xy = np.zeros((n, 10, 2))
    xy[:, :, 0], xy[:, :, 1] = np.linspace(40, 60, 10), 10.0
    xy[:, 0] = (70.0, 25.0)
    close = np.clip(8.0 - 5.0 * np.arange(n) / release, 3.0, 8.0)
    xy[:, 5, 0], xy[:, 5, 1] = 70.0 + close, 25.0
    T = dict(gameid="g", per=np.ones(n, dtype=int), gc=gc, sc=np.full(n, 14.0) - np.arange(n) / 25.0,
             ball=ball, pid=pid, team=team, xy=xy, ts=np.arange(n))
    return T, release


def test_release_defender_clock_dribbles():
    T, release = _possession()
    shots = pd.DataFrame(dict(action_number=[7], period=[1], clock=[697.0 - 2.6], shooter=[1]))
    r = shot_rows(T, shots, ReleaseRules(release_back=0)).iloc[0]
    assert r.how == "rim"
    # the ball stays within 2.5 ft of the shooter for a frame or two after it leaves his hand: the finder runs up
    # to three frames late, which is what the calibrated `release_back` takes back
    assert 0.0 <= T["gc"][release] - r.mv_gc <= 0.121
    assert abs(r.def_dist - 3.0) < 0.3 and r.def_id == 11
    assert abs(r.mv_dist - 18.75) < 0.6
    assert r.dribbles == 2
    assert r.touch_time > 2.5
    earlier = shot_rows(T, shots, ReleaseRules(release_back=10)).iloc[0]
    assert earlier.def_dist > r.def_dist                          # earlier, the closeout has further to go
