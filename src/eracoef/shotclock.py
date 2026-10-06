"""The shot clock, rebuilt from play-by-play (the shot-quality build, stage 3).

The feed never prints the shot clock, but every reset is an event it does print: the possession began (24),
the offence rebounded its own miss off the rim (24, or 14 from 2018-19), the defence fouled or kicked the ball
with the offence keeping it (back up to 14 if it was lower).  So at any moment

    shot clock = (value it was reset to) - (game seconds since that reset) + (a logging lag for that kind)

and when the game clock at the reset is shorter than the reset value, the clock is OFF: what is left is the
game clock.  The lags are fitted on the 2014-15 shot log's true SHOT_CLOCK (tracking.join_shotlog), never on
the attempts being priced; the rules are checked on every season by the shot-clock violations, which should
rebuild to zero (`violations`).

The rules, as written in the NBA rule book (rule 7) and the 2018-19 change:
    - 24 on every change of possession; the clock starts on the inbound touch, which after a made basket
      (game clock running, except the last two minutes of the fourth and overtime) is a few seconds after the
      basket -- the `made_fg` lag
    - an offensive rebound of a shot that hit the rim: 24 before 2018-19, 14 from then (`oreb_14_from`)
    - a non-shooting defensive foul, a kicked ball, a defensive three seconds: the larger of what is left and 14
    - a held ball the offence keeps: the larger of what is left and 5
    - a timeout: no reset

Model layer: frames in, arrays out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DFOUL_TYPES = frozenset({"Personal", "Loose Ball", "Personal Take", "Transition Take", "Personal Block",
                         "Away From Play", "Clear Path", "Flagrant Type 1", "Flagrant Type 2", "Defense 3 Second",
                         "Inbound", "Punching"})
KICK_TYPES = frozenset({"Kicked Ball"})
START_KINDS = ("made_fg", "made_ft", "dreb", "dreb_team", "steal", "dead_tov", "jump_ball", "period_start", "forced")
BINS = (0.0, 4.0, 7.0, 15.0, 18.0, 22.0, 24.01)          # the tracking dashboards' six shot-clock ranges


@dataclass
class ClockRules:
    oreb_14_from: int = 2019          # first season (by end year) of the 14-second offensive-rebound reset
    oreb_14_mode: str = "set"         # "set": exactly 14; "max": the larger of what is left and 14
    dfoul_from: int = 1997            # first season the defensive-foul / kicked-ball reset to 14 applies
    lag: dict = field(default_factory=dict)   # reset kind -> seconds added (fitted; 0 if absent)

    def oreb_value(self, season: int, remaining: float) -> float:
        if season >= self.oreb_14_from:
            return 14.0 if self.oreb_14_mode == "set" else max(remaining, 14.0)
        return 24.0


def _parse(events: str):
    if not events:
        return []
    out = []
    for tok in events.split(";"):
        k, _, c = tok.rpartition("@")
        try:
            out.append((k, float(c)))
        except ValueError:
            continue
    return out


def rebuild(frame: pd.DataFrame, rules: ClockRules | None = None) -> pd.DataFrame:
    """Per row: the rebuilt shot clock at the row's game clock (`sc`, before the off rule), the kind of the
    last reset, the seconds since it, whether the clock was off, and `sc_eff` = what the offence really had
    (the shot clock, or the game clock when that is shorter).

    Needs season, clock, poss_start, poss_start_clock, events (shotframe's columns)."""
    rules = rules or ClockRules()
    n = len(frame)
    sc = np.empty(n)
    kind = np.empty(n, dtype=object)
    since = np.empty(n)
    off = np.zeros(n, dtype=bool)
    seasons = frame["season"].to_numpy()
    clocks = frame["clock"].to_numpy(float)
    starts = frame["poss_start"].to_numpy()
    t0s = frame["poss_start_clock"].to_numpy(float)
    evs = frame["events"].fillna("").to_numpy()
    lag = rules.lag
    for i in range(n):
        season, t = int(seasons[i]), clocks[i]
        k0 = str(starts[i])
        t_reset, value, rk = t0s[i], 24.0, k0
        clock_off = t_reset < value
        for ek, ec in _parse(evs[i]):
            if ec < t:                      # an event after the shot (same-second ties stay in)
                break
            rem = value - (t_reset - ec) + lag.get(rk, 0.0)
            if ek in ("oreb", "oreb_team"):
                t_reset, value, rk = ec, rules.oreb_value(season, rem), "oreb"
            elif ek.startswith("foul:def:") or ek.startswith("viol:def:"):
                typ = ek.split(":", 2)[2]
                if season >= rules.dfoul_from and (typ in DFOUL_TYPES or typ in KICK_TYPES):
                    if rem < 14.0:
                        t_reset, value, rk = ec, 14.0, "dfoul"
            elif ek == "jump":
                if rem < 5.0:
                    t_reset, value, rk = ec, 5.0, "jump"
            else:
                continue
            clock_off = t_reset < value
        s = value - (t_reset - t) + lag.get(rk, 0.0)
        sc[i] = s
        kind[i] = rk
        since[i] = t_reset - t
        off[i] = clock_off
    out = pd.DataFrame({"sc": np.clip(sc, 0.0, 24.0), "sc_raw": sc, "reset_kind": kind, "since_reset": since,
                        "clock_off": off}, index=frame.index)
    out["sc_eff"] = np.minimum(out["sc"].to_numpy(), clocks)
    return out


def fit_lags(rebuilt: pd.DataFrame, truth: np.ndarray, min_n: int = 200) -> dict:
    """The median of (true - rebuilt) by reset kind, on rows where the true clock is on and the rebuilt one
    is not at a bound; kinds with fewer than `min_n` rows borrow the pooled median."""
    r = rebuilt.assign(truth=truth)
    ok = r["truth"].notna() & ~r["clock_off"] & (r["sc"] > 0.0) & (r["sc"] < 24.0)
    r = r[ok]
    res = r["truth"] - r["sc"]
    pooled = float(res.median()) if len(res) else 0.0
    out = {}
    for k, g in res.groupby(r["reset_kind"]):
        out[str(k)] = float(g.median()) if len(g) >= min_n else pooled
    out["_pooled"] = pooled
    return out


def accuracy(sc: np.ndarray, truth: np.ndarray, off: np.ndarray | None = None) -> dict:
    """Agreement with the true clock where it was on: median absolute error, share within 1 / 3 s, and share
    in the same dashboard bin."""
    sc, truth = np.asarray(sc, float), np.asarray(truth, float)
    keep = np.isfinite(truth)
    e = np.abs(sc[keep] - truth[keep])
    b1 = np.digitize(np.clip(sc[keep], 0, 24), BINS[1:-1])
    b2 = np.digitize(np.clip(truth[keep], 0, 24), BINS[1:-1])
    out = dict(n=int(keep.sum()), mae=float(np.mean(e)), median_abs=float(np.median(e)),
               within_1=float(np.mean(e <= 1.0)), within_3=float(np.mean(e <= 3.0)), same_bin=float(np.mean(b1 == b2)))
    if off is not None:
        off = np.asarray(off, bool)
        out["off_recall"] = float(off[~keep].mean()) if (~keep).any() else np.nan     # truth off, rebuilt off
        out["off_false"] = float(off[keep].mean())                                   # truth on, rebuilt off
    return out
