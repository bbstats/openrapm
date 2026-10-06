"""The 2015-16 SportVU movement data, turned into one tracking row per field-goal attempt (stage 3b).

Source: github.com/sealneaward/nba-movement-data (MIT licence; a copy of neilmj/BasketballData), 636 7-zip
archives of one JSON game each, 2015-10-27 to 2016-01-23, five of them empty.  Downloaded whole into
data/tracking/movement_7z/ (the owner approved it, 2026-10-06: "yes, all fields").

A game's JSON is a list of play-by-play events (eventId == the play-by-play actionNumber) with their frames
(25 a second): period, a timestamp, the game clock, the SHOT CLOCK, and eleven entities -- the ball with its
height, then the ten players' x/y.  The events' frames overlap and are sometimes misaligned with their own
event (event 2 of 0021500492 carries frames 32 s after its shot), so the frames are pooled into one timeline
per game, deduplicated by timestamp, and each attempt is found by its game clock instead.

The release (`find_release`): within a window around the play-by-play time (which runs about 2.6 s behind
the true clock), the ball must reach the rim within 2.5 s of last being near the shooter; the release is
the last frame before that with the ball within 2.5 ft of the shooter (x/y), moved `release_back` frames
earlier.  The offset is calibrated on the tracking dashboards' own per-player defender bins (the STATS
definition of the moment of the shot), never on the attempts being labelled.

At the release: the closest defender and his distance from the SHOOTER (the dashboards' convention), the
second closest, the shot clock, the shooter's speed, and over the possession that ends in the shot (the
ball within `hold_ft` of the shooter) the touch time and the dribbles (bounces of the ball below
`bounce_ft`).  Court units are feet, x 0-94 along the court, hoops at x 5.25 and 88.75.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HOOPS = np.array([[5.25, 25.0], [88.75, 25.0]])
SEVEN_ZIP = r"C:\Program Files\7-Zip\7z.exe"


@dataclass
class ReleaseRules:
    window_before: float = 1.5        # seconds of game clock after the play-by-play time (it runs behind)
    window_after: float = 8.0         # seconds before it (the 95th percentile of the lag is 5.6 s)
    expected_lag: float = 2.6         # the median lag: of several shots in the window, the one nearest this
    near_ft: float = 4.0              # the ball counts as with the shooter within this (x/y)
    release_ft: float = 2.5           # the release: the last frame this close before the ball reaches the rim
    release_back: int = 3             # frames (40 ms each) the release is moved earlier (calibrated)
    rim_ft: float = 3.0               # the ball "at the rim": this close to the hoop in x/y ...
    rim_z: tuple = (7.0, 14.0)        # ... and this high
    max_flight: float = 2.5           # seconds from last near the shooter to the rim
    hold_ft: float = 3.5              # possession before the shot: the ball this close to the shooter
    hold_gap: int = 4                 # frames the ball may leave that radius inside one possession
    bounce_ft: float = 2.5            # a dribble: the ball's height has a local minimum below this


def timeline(path) -> dict:
    """A game's frames pooled across events, deduplicated by timestamp and sorted."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    seen = {}
    for e in d["events"]:
        for m in e["moments"]:
            ts = m[1]
            if ts in seen:
                continue
            ents = m[5]
            if len(ents) < 11 or ents[0][0] != -1:
                continue
            seen[ts] = m
    ks = sorted(seen)
    n = len(ks)
    per = np.empty(n, dtype=np.int16)
    gc = np.empty(n)
    sc = np.empty(n)
    ball = np.empty((n, 3))
    pid = np.empty((n, 10), dtype=np.int64)
    team = np.empty((n, 10), dtype=np.int64)
    xy = np.empty((n, 10, 2))
    for i, ts in enumerate(ks):
        m = seen[ts]
        per[i], gc[i] = m[0], m[2]
        sc[i] = np.nan if m[3] is None else m[3]
        b = m[5][0]
        ball[i] = b[2], b[3], b[4]
        for k, p in enumerate(m[5][1:11]):
            team[i, k], pid[i, k], xy[i, k, 0], xy[i, k, 1] = p[0], p[1], p[2], p[3]
    return dict(gameid=d["gameid"], gamedate=d.get("gamedate"), per=per, gc=gc, sc=sc, ball=ball, pid=pid,
                team=team, xy=xy, ts=np.asarray(ks, dtype=np.int64))


def _nearest_hoop(xy):
    dd = np.hypot(xy[:, None, 0] - HOOPS[None, :, 0], xy[:, None, 1] - HOOPS[None, :, 1])
    return HOOPS[np.argmin(dd, axis=1)]


def find_release(T: dict, period: int, clock: float, shooter: int, rules: ReleaseRules):
    """(frame indices of the window, the shooter's column per frame, the release position in the window, how)
    or None.  `how` is "rim" when the ball was followed to the rim, "last" when it was not (an airball, a
    gap in the data): the last frame near the shooter."""
    w = np.flatnonzero((T["per"] == period) & (T["gc"] >= clock - rules.window_before)
                       & (T["gc"] <= clock + rules.window_after))
    if len(w) < 10:
        return None
    on = (T["pid"][w] == shooter)
    w = w[on.any(axis=1)]
    if len(w) < 10:
        return None
    j = np.argmax(T["pid"][w] == shooter, axis=1)
    sxy = T["xy"][w, j]
    b = T["ball"][w]
    d = np.hypot(b[:, 0] - sxy[:, 0], b[:, 1] - sxy[:, 1])
    hoop = _nearest_hoop(sxy)
    hd = np.hypot(b[:, 0] - hoop[:, 0], b[:, 1] - hoop[:, 1])
    at_rim = (hd < rules.rim_ft) & (b[:, 2] > rules.rim_z[0]) & (b[:, 2] < rules.rim_z[1])
    near = np.flatnonzero(d <= rules.near_ft)
    if not len(near):
        return None
    gc = T["gc"][w]
    # every flight from the shooter to the rim in the window; a putback after his own miss gives two, so keep
    # the release nearest the play-by-play time plus its usual lag
    best, best_gap = None, np.inf
    for k in np.flatnonzero(at_rim):
        before = near[near < k]
        if not len(before) or gc[before[-1]] - gc[k] > rules.max_flight:
            continue
        c = np.flatnonzero(d[:k] <= rules.release_ft)
        if not len(c):
            continue
        r = int(c[-1])
        gap = abs(gc[r] - (clock + rules.expected_lag))
        if gap < best_gap:
            best, best_gap = r, gap
    if best is not None:
        return w, j, max(best - rules.release_back, 0), "rim", hoop
    r = int(near[np.argmin(np.abs(gc[near] - (clock + rules.expected_lag)))])
    return w, j, max(r - rules.release_back, 0), "last", hoop


def _possession(d: np.ndarray, r: int, hold_ft: float, gap: int) -> int:
    """The first frame of the possession that ends at the release: walking back while the ball stays within
    `hold_ft` of the shooter, allowing gaps of up to `gap` frames (a dribble's bounce can take it further)."""
    start, miss = r, 0
    for i in range(r, -1, -1):
        if d[i] <= hold_ft:
            start, miss = i, 0
        else:
            miss += 1
            if miss > gap:
                break
    return start


def shot_rows(T: dict, shots: pd.DataFrame, rules: ReleaseRules | None = None) -> pd.DataFrame:
    """One tracking row per attempt of `shots` (columns action_number, period, clock, shooter) found in T."""
    rules = rules or ReleaseRules()
    out = []
    for s in shots.itertuples(index=False):
        hit = find_release(T, int(s.period), float(s.clock), int(s.shooter), rules)
        if hit is None:
            continue
        w, j, r, how, hoop = hit
        fr = w[r]
        sxy = T["xy"][fr, j[r]]
        opp = T["team"][fr] != T["team"][fr, j[r]]
        dd = np.hypot(*(T["xy"][fr][opp] - sxy).T)
        order = np.argsort(dd)
        b = T["ball"][w]
        sx = T["xy"][w, j]
        d = np.hypot(b[:, 0] - sx[:, 0], b[:, 1] - sx[:, 1])
        st = _possession(d, r, rules.hold_ft, rules.hold_gap)
        z = b[st:r + 1, 2]
        bounces = int(np.sum((z[1:-1] < z[:-2]) & (z[1:-1] <= z[2:]) & (z[1:-1] < rules.bounce_ft))) if len(z) > 2 else 0
        k0, k1 = max(r - 5, 0), r
        dt = T["gc"][w[k0]] - T["gc"][w[k1]]
        speed = float(np.hypot(*(sx[k1] - sx[k0])) / dt) if dt > 0 else np.nan
        out.append(dict(action_number=int(s.action_number), how=how, mv_gc=float(T["gc"][fr]),
                        mv_lag=float(T["gc"][fr] - s.clock), mv_shot_clock=float(T["sc"][fr]),
                        mv_dist=float(np.hypot(*(sxy - hoop[r]))), mv_ball_z=float(T["ball"][fr, 2]),
                        def_id=int(T["pid"][fr][opp][order[0]]), def_dist=float(dd[order[0]]),
                        def_dist2=float(dd[order[1]]) if len(dd) > 1 else np.nan,
                        touch_time=float(T["gc"][w[st]] - T["gc"][fr]), dribbles=bounces, shooter_speed=speed,
                        window_start_clipped=int(st == 0)))
    return pd.DataFrame(out)


def game_id_of(archive: Path, tmp: Path) -> str | None:
    """Unpack one archive into `tmp` (emptied first) and return the path of its JSON, or None."""
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    subprocess.run([SEVEN_ZIP, "x", "-y", f"-o{tmp}", str(archive)], capture_output=True, check=False)
    js = glob.glob(str(tmp / "*.json"))
    return js[0] if js else None


def parse_archive(archive: Path, frames_by_game: dict, tmp: Path, rules: ReleaseRules | None = None):
    """(gameid, shot rows) for one archive; the unpacked JSON (~100 MB) is deleted before returning."""
    js = game_id_of(archive, tmp)
    if js is None:
        return None, pd.DataFrame()
    try:
        T = timeline(js)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    shots = frames_by_game.get(T["gameid"])
    if shots is None or not len(shots):
        return T["gameid"], pd.DataFrame()
    rows = shot_rows(T, shots, rules)
    if len(rows):
        rows.insert(0, "game_id", T["gameid"])
    return T["gameid"], rows


def archives(cfg) -> list[Path]:
    d = Path(cfg["_root"]) / "data" / "tracking" / "movement_7z"
    return sorted(p for p in d.glob("*.7z") if os.path.getsize(p) > 10_000)
