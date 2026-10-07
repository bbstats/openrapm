"""The shot-quality search's new play-by-play features, as shotmodel blocks (DECISIONS.md, "The shot-quality search").

The owner, 2026-10-06: "let's go bananas on making the model as good as possible".  What the play-by-play can still
add sits mostly in the possession around the shot, which the logger already writes and the model never read: how many
offensive rebounds came first, whether a timeout or a side inbound set the shot up, how late and close the game was,
where the shot stood against the three-point line, what the shot that handed the ball over looked like.  The design
round's held-out residuals put each at 0.00002-0.0001 log loss a shot (more on the team rows), so each one runs as an
add-one arm and a drop-one arm, and joins the search only if it passes the registered bar.

    late      Q4 or overtime with 5 and with 2 minutes or less left, within 5 and within 3 points; trailing or
              leading by 1-3 in the last 2 minutes (each sub-model has its own coefficient, so this is split by shot
              value); the two-for-one window (28-40 s left in any period), and that window early in the possession
    scramble  offensive rebounds so far (1, 2+); a second or later attempt; the previous attempt of this possession
              (rim / other two / three, blocked, seconds since); on threes, the kick-out (3 s or less after an
              offensive rebound)
    ato       a timeout in this possession, by seconds since it (0-6, 6-12, 12-24, 24+).  Yes or no only: timeout
              kinds are era artefacts (short until 2017, official TV gone after 2017, coach's challenge from 2020)
    dfoul     earlier non-shooting defensive fouls in this possession (1, 2+): the offence inbounding from the side
    line      feet beyond the three-point line at that angle, era-correct (22 ft all round in 1997, the season ending
              in 1997); on twos, the toe on the line (within 1 ft inside, or logged on it) and 1-3 ft inside
    prevloc   after a missed shot handed the ball over: that shot's distance class, whether it was on the same side
              as this one, how fast the ball came back (seconds from that shot to this one), and the stale start (a
              defensive-rebound start whose 'shot that handed it over' is an older one or a make: the rebound of a
              missed free throw, about 6% of such starts in 2016)
    side      the side of the floor (the sign of x), and side times angle

THE SAME-SECOND RULE.  The feed sometimes logs a foul before the made shot it belongs to (the and-one rate of mid and
three shots whose previous row is within 0.5 s is 3.8% against 0.8% in 2002, 8.1% against 1.1% in 2018), and a
defensive goaltending violation sits at the shot's own clock reading 593 times in 596 (2016).  So every feature read
from the events string ignores events at (or after) the shot's own clock reading -- except an offensive rebound, which
really does come first.  scripts/128_feature_audit.py checks it on the real tables: each block rebuilt with those
events removed must be unchanged.  `timeout_secs` here is the rule's version of shotframe.derive's
`secs_since_timeout`, which reads every timeout.

The PREVIOUS ATTEMPT of a possession is another row of the table, so it is built once per season table, before any row
is filtered or sampled (`add_previous_attempt`, or `prepare` for every precomputed field).  A block raises if it is
missing rather than read a sampled table's gaps as 'no previous attempt'.  Its blocked flag is the previous row's
post_blocked: a fact about an EARLIER shot, as the shipped `prev` block's start_prev_blocked is.  A table without
post_blocked gets -1 ('not known') there; the scramble block then refuses to build rather than read every previous
attempt as not blocked.

Never an input (the owner's rulings): a scorer shot-type tag or description word, an outcome of THIS shot (post_*), an
identity or trait of a player, team or arena.  `tree_features`, the boosters' inputs, is a whitelist checked against
FORBIDDEN, which raises.

Every block is fn(f, sub) -> (columns, names), shotmodel.block_columns' contract: no intercept, and the column set
depends on the sub-model only, never on the rows, so a model trained on one frame prices another.

Model layer: frames in, arrays out.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import shotmodel as sm
from .shotclock import DFOUL_TYPES

SAME_TOL = 0.005                    # two clock readings this close are the same moment (the feed's clock: tenths at best)
STALE_S = 5.0                       # a dreb start this long after its 'previous shot' began on a free throw's rebound
OREB_KINDS = ("oreb", "oreb_team")  # the one kind of event the same-second rule keeps
ARC_FT = 23.75                      # the arc's radius from the rim; the corner line is at |x| = CORNER_X
CORNER_X, CORNER_Y = 220.0, 92.5    # tenths of a foot: shotframe.derive's corner three
SHORT_LINE = {1997: 22.0}           # seasons (by end year) whose line was one radius all the way round
PREV_COLS = ("prev_att_dist", "prev_att_value", "prev_att_blocked", "prev_att_clock")
EVENT_COLS = ("timeout_secs", "n_dfoul")

FORBIDDEN = re.compile(r"post_|at_heave|^off\d|^def\d|^shooter$|^team$|^opp$|arena|^lp$|^dist$|^x$|^y$|subType|"
                       r"description|actionType|lineup|secs_since_prev|def_dist|true_sc|dribbles|touch_time|game_id|"
                       r"season|^made$|prev_event_clock|secs_since_timeout|^home$|^neutral$|^phase$|^half$|game_date|"
                       r"sc_raw|_stamp$|_logged$")

# the table columns each block reads (besides what `prepare` adds), for the driver's checks
NEEDS = {
    "late": ("period", "clock", "margin", "secs_into_poss"),
    "scramble": ("n_oreb", "att_no", "clock", "value", "secs_since_oreb") + PREV_COLS,
    "ato": ("events", "clock"),
    "dfoul": ("events", "clock"),
    "line": ("season", "value", "xc", "yc", "dist_xy", "noloc"),
    "prevloc": ("poss_start", "start_prev_made", "start_prev_value", "start_prev_dist", "start_prev_x",
                "start_prev_y", "start_prev_clock", "poss_start_clock", "clock", "xc"),
    "side": ("xc", "angle"),
}


def _need(f: pd.DataFrame, cols, block: str, hint: str = "") -> None:
    miss = [c for c in cols if c not in f.columns]
    if miss:
        raise KeyError(f"block {block!r} needs columns {miss}{hint}")


# ------------------------------------------------------------------------------------------ the events string
def _tokens(events: str):
    """(kind, clock) per token of shotframe's events string 'kind@clock;...', in the order the feed logged them."""
    if not events:
        return
    for tok in events.split(";"):
        k, _, c = tok.rpartition("@")
        try:
            yield k, float(c)
        except ValueError:
            continue


def before_shot(events: str, clock: float, tol: float = SAME_TOL, n=None) -> list:
    """The events of a row that came before its shot.  With `n` (the row's anchor_nev): the first n, the ones logged up
    to and including the anchor row -- no clock comparison at all.  Without it: the same-second rule, offensive
    rebounds always kept."""
    if n is not None:
        return list(_tokens(events))[:int(n)]
    return [(k, c) for k, c in _tokens(events) if c > clock + tol or k in OREB_KINDS]


def event_fields(f: pd.DataFrame, tol: float = SAME_TOL) -> pd.DataFrame:
    """Per row, under the same-second rule: `timeout_secs`, seconds since the latest timeout of the possession (NaN
    if none), and `n_dfoul`, the earlier non-shooting defensive fouls (shotclock.DFOUL_TYPES; kicked balls are
    violations and are not counted).  Row-local, so it is right on a sampled table too."""
    n = len(f)
    ev = f["events"].fillna("").to_numpy(object)
    clock = f["clock"].to_numpy(float)
    nev = f["anchor_nev"].to_numpy() if "anchor_nev" in f.columns else None
    to = np.full(n, np.nan)
    nd = np.zeros(n)
    for i in np.flatnonzero(ev != ""):
        last, k_d = np.nan, 0
        for k, c in before_shot(ev[i], clock[i], tol, None if nev is None else nev[i]):
            if k == "timeout":
                last = c                                    # the feed's order: the last one kept is the latest
            elif k.startswith("foul:def:") and k[len("foul:def:"):] in DFOUL_TYPES:
                k_d += 1
        to[i] = max(last - clock[i], 0.0) if np.isfinite(last) else np.nan
        nd[i] = k_d
    return pd.DataFrame({"timeout_secs": to, "n_dfoul": nd}, index=f.index)


def _events(f: pd.DataFrame, block: str) -> pd.DataFrame:
    if all(c in f.columns for c in EVENT_COLS):
        return f[list(EVENT_COLS)]
    _need(f, ("events", "clock"), block, " (or the precomputed timeout_secs, n_dfoul of shotfeatures.prepare)")
    return event_fields(f)


# ------------------------------------------------------------------------------------------ the previous attempt
def add_previous_attempt(t: pd.DataFrame, require_blocked: bool = False) -> pd.DataFrame:
    """The previous field-goal attempt of each row's possession, as four columns (same rows, same order):
    prev_att_dist (its dist_xy), prev_att_value (0 if none), prev_att_blocked (-1 if none), prev_att_clock (the earlier attempt's own stamp, clock_stamp, when present).

    Rows are put in game order -- game_id, period, clock descending, action_number -- and shifted within
    (game_id, poss_no).  Call it ONCE on a whole season table, before any filter or sample: the previous attempt is
    another row, and a missing row would read as 'no previous attempt'.  The blocked flag is the previous row's
    post_blocked (a fact about the earlier shot) where the table carries post_blocked, and -1 ('not known')
    where it does not -- the scramble block then raises; `require_blocked=True` raises here instead."""
    _need(t, ("game_id", "period", "clock", "action_number", "poss_no", "dist_xy", "value"), "add_previous_attempt")
    has_b = "post_blocked" in t.columns
    if require_blocked and not has_b:
        raise KeyError("add_previous_attempt needs post_blocked (the PREVIOUS row's, a fact about an earlier shot): "
                       "add it to 120's TABLE_COLS or merge it from the shot frame on game_id, action_number")
    n = len(t)
    g = pd.factorize(t["game_id"])[0]
    order = np.lexsort((t["fga_no"].to_numpy(), t["poss_no"].to_numpy(), g))
    s = pd.DataFrame({"g": g[order], "p": t["poss_no"].to_numpy()[order], "pos": np.arange(n)})
    prev_pos = s.groupby(["g", "p"], sort=False)["pos"].shift(1).to_numpy(float)
    has = np.isfinite(prev_pos)
    this, prev = order[has], order[prev_pos[has].astype(np.int64)]
    dist = np.full(n, np.nan)
    value = np.zeros(n, dtype=np.int64)
    blocked = np.full(n, -1, dtype=np.int64)
    clock = np.full(n, np.nan)
    dist[this] = t["dist_xy"].to_numpy(float)[prev]
    value[this] = t["value"].to_numpy(np.int64)[prev]
    if has_b:
        blocked[this] = t["post_blocked"].to_numpy(np.int64)[prev]
    # the earlier attempt's OWN stamp where the table keeps it: that row is at or before this shot's anchor (an attempt
    # is an anchor row), so its stamp is as safe as the anchor and keeps the miss-to-putback timing the anchor loses
    src = "clock_stamp" if "clock_stamp" in t.columns else "clock"
    clock[this] = t[src].to_numpy(float)[prev]
    return t.assign(prev_att_dist=dist, prev_att_value=value, prev_att_blocked=blocked, prev_att_clock=clock)


def prepare(t: pd.DataFrame, require_blocked: bool = False) -> pd.DataFrame:
    """Everything the blocks and tree_features precompute, once per season table (before any filter or sample):
    the previous attempt, and the events' fields under the same-second rule."""
    out = add_previous_attempt(t, require_blocked=require_blocked)
    ef = event_fields(out)
    return out.assign(**{c: ef[c].to_numpy() for c in EVENT_COLS})


# ------------------------------------------------------------------------------------------ geometry
def line_ft(f: pd.DataFrame) -> np.ndarray:
    """Signed feet beyond the three-point line (negative inside), at the shot's own angle; NaN without coordinates.

    Below the break (y <= 92.5 tenths, the corner three's own bound) the line is the straight corner segment at
    |x| = 22 ft, so the distance is horizontal, on either side of it; above, it is the arc, 23.75 ft from the rim.
    In 1997 (1996-97) the line was 22 ft all the way round."""
    x = np.abs(f["xc"].to_numpy(float))
    y = f["yc"].to_numpy(float)
    d = f["dist_xy"].to_numpy(float)
    b = np.where(y <= CORNER_Y, (x - CORNER_X) / 10.0, d - ARC_FT)
    season = f["season"].to_numpy()
    for s, r in SHORT_LINE.items():
        m = season == s
        b[m] = d[m] - r
    noloc = f["noloc"].to_numpy(bool) if "noloc" in f.columns else (x == 0) & (y == 0)
    b[noloc] = np.nan
    return b


def _prev_shot_ft(f: pd.DataFrame) -> np.ndarray:
    """The distance of the shot that handed the ball over, from its coordinates as dist_xy is (the feed's own
    integer distance only where it has none); (0, 0) reads 0, as an unlocated two reads at the rim."""
    px, py = f["start_prev_x"].to_numpy(float), f["start_prev_y"].to_numpy(float)
    d = np.hypot(px, py) / 10.0
    return np.where(np.isfinite(d), d, f["start_prev_dist"].to_numpy(float))


def _sign(v: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(v), np.sign(v), 0.0)


def _bands(x, edges) -> np.ndarray:
    """One column per band [e_i, e_{i+1}); NaN falls in no band (shotmodel's own rule)."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([((x >= a) & (x < b)).astype(float) for a, b in zip(edges[:-1], edges[1:])])


# ------------------------------------------------------------------------------------------ the blocks
def late(f: pd.DataFrame, sub: str):
    _need(f, NEEDS["late"], "late")
    per = f["period"].to_numpy(int)
    left = f["clock"].to_numpy(float)
    m = f["margin"].to_numpy(float)
    t = f["secs_into_poss"].to_numpy(float)
    in5 = (per >= 4) & (left <= 300.0)
    in2 = (per >= 4) & (left <= 120.0)
    c5, c3 = np.abs(m) <= 5.0, np.abs(m) <= 3.0
    w = (left >= 28.0) & (left <= 40.0)
    cols = [in5 & c5, in5 & c3, in2 & c5, in2 & c3, in2 & (m >= -3.0) & (m <= -1.0), in2 & (m >= 1.0) & (m <= 3.0),
            w, w & (t <= 8.0)]
    return np.column_stack(cols).astype(float), ["late5_c5", "late5_c3", "late2_c5", "late2_c3", "late2_trail13",
                                                 "late2_lead13", "two_for_one", "two_for_one_quick"]


def scramble(f: pd.DataFrame, sub: str):
    _need(f, PREV_COLS, "scramble", ": call shotfeatures.add_previous_attempt (or prepare) on the whole season "
                                    "table before any filter or sample")
    _need(f, ("att_no", "clock", "value", "secs_since_oreb"), "scramble")
    if "n_oreb" in f.columns:
        no = f["n_oreb"].to_numpy(float)
    else:
        _need(f, ("events",), "scramble")
        no = f["events"].fillna("").str.count(r"(?:^|;)oreb").to_numpy(float)
    att = f["att_no"].to_numpy(float)
    pv = f["prev_att_value"].to_numpy(float)
    pdist = f["prev_att_dist"].to_numpy(float)
    has = pv > 0
    gap = f["prev_att_clock"].to_numpy(float) - f["clock"].to_numpy(float)
    pb = f["prev_att_blocked"].to_numpy(float)
    if (has & (pb < 0)).any():
        raise KeyError("block 'scramble' needs the previous attempt's blocked flag, and this table had no post_blocked "
                       "when add_previous_attempt ran: add post_blocked to 120's TABLE_COLS (or merge it from the shot "
                       "frame on game_id, action_number) before prepare")
    cols = [no == 1, no >= 2, att >= 2,
            has & (pv == 2) & (pdist < sm.RIM_FT), has & (pv == 2) & (pdist >= sm.RIM_FT), has & (pv == 3),
            has & (pb == 1),
            has & (gap < 3.0), has & (gap >= 3.0) & (gap < 8.0)]
    names = ["scr_oreb1", "scr_oreb2p", "scr_att2p", "scr_prev_rim", "scr_prev_mid", "scr_prev_three",
             "scr_prev_blocked", "scr_prev_0_3", "scr_prev_3_8"]
    if sub == "three":
        so = f["secs_since_oreb"].to_numpy(float)
        cols.append((f["value"].to_numpy() == 3) & (so >= 0.0) & (so <= 3.0))
        names.append("scr_kickout")
    return np.column_stack(cols).astype(float), names


def ato(f: pd.DataFrame, sub: str):
    s = _events(f, "ato")["timeout_secs"].to_numpy(float)
    return _bands(s, (0.0, 6.0, 12.0, 24.0, 1e9)), ["ato_0_6", "ato_6_12", "ato_12_24", "ato_24p"]


def dfoul(f: pd.DataFrame, sub: str):
    k = _events(f, "dfoul")["n_dfoul"].to_numpy(float)
    return np.column_stack([k == 1, k >= 2]).astype(float), ["dfoul_1", "dfoul_2p"]


def line(f: pd.DataFrame, sub: str):
    if sub == "rim":
        return np.zeros((len(f), 0)), []
    _need(f, NEEDS["line"], "line")
    b = line_ft(f)
    if sub == "three":
        cols = [b < 0.0, (b >= 0.0) & (b < 0.5), (b >= 0.5) & (b < 1.0), (b >= 1.0) & (b < 2.0), (b >= 2.0) & (b < 4.0)]
        return np.column_stack(cols).astype(float), ["line_in", "line_0_05", "line_05_1", "line_1_2", "line_2_4"]
    two = f["value"].to_numpy() == 2
    return (np.column_stack([two & (b >= -1.0), two & (b >= -3.0) & (b < -1.0)]).astype(float),
            ["line_toe", "line_in_1_3"])


def prevloc(f: pd.DataFrame, sub: str):
    _need(f, NEEDS["prevloc"], "prevloc")
    dreb = np.isin(f["poss_start"].to_numpy(object), ["dreb", "dreb_team"])
    made = f["start_prev_made"].to_numpy(float)
    sp_clock = f["start_prev_clock"].to_numpy(float)
    fresh = (sp_clock - f["poss_start_clock"].to_numpy(float)) <= STALE_S
    after = dreb & (made == 0) & fresh
    stale = dreb & ~((made == 0) & fresh)
    pv = f["start_prev_value"].to_numpy(float)
    d = _prev_shot_ft(f)
    two = after & (pv == 2)
    same = after & (_sign(f["start_prev_x"].to_numpy(float)) * _sign(f["xc"].to_numpy(float)) > 0)
    tr = sp_clock - f["clock"].to_numpy(float)
    cols = [two & (d < 10.0), two & (d >= 10.0) & (d < 16.0), two & (d >= 16.0), after & (pv == 3), same,
            after & (tr < 4.0), after & (tr >= 4.0) & (tr < 7.0), after & (tr >= 7.0) & (tr < 12.0), stale]
    return np.column_stack(cols).astype(float), ["pl_rim", "pl_short", "pl_long2", "pl_three", "pl_same_side",
                                                 "pl_tr_0_4", "pl_tr_4_7", "pl_tr_7_12", "pl_stale"]


def side(f: pd.DataFrame, sub: str):
    _need(f, NEEDS["side"], "side")
    s = _sign(f["xc"].to_numpy(float))
    a = np.clip(f["angle"].to_numpy(float), 0.0, 90.0) / 90.0
    return np.column_stack([s, s * np.where(np.isfinite(a), a, 0.0)]), ["side", "side_angle"]


BLOCKS: dict = {"late": late, "scramble": scramble, "ato": ato, "dfoul": dfoul, "line": line, "prevloc": prevloc,
                "side": side}


# ------------------------------------------------------------------------------------------ the boosters' inputs
# raw table columns a booster may read: the spot (no raw x, y, dist or lp), the possession, the clock, the game
# situation, the shooter's time on court, the shot that handed the ball over, the possession's counts
TREE_RAW = ("value", "dist_xy", "angle", "corner3", "noloc", "xc", "yc", "secs_into_poss", "secs_since_oreb", "sc_eff",
            "clock_off", "since_reset", "period", "clock", "margin", "shooter_secs_on", "start_prev_value",
            "start_prev_made", "start_prev_blocked", "n_oreb", "att_no", "fga_no")
TREE_DERIVED = ("sub", "poss_start", "reset_kind", "timeout_secs", "n_dfoul", "prev_att_dist", "prev_att_value",
                "prev_att_blocked", "prev_att_secs", "start_prev_dist_xy", "trans_secs", "prev_stale", "line_ft", "side")
_SUB_CODE = {"rim": 0, "mid": 1, "three": 2}


def check_names(cols) -> None:
    """Raise on any banned input name: an outcome of the shot, an identity, a scorer tag, a tracking field."""
    bad = [str(c) for c in cols if FORBIDDEN.search(str(c))]
    if bad:
        raise ValueError(f"banned inputs {bad}: no identity, scorer tag, outcome of the shot or tracking field may enter")


def tree_features(f: pd.DataFrame, extra=(), fill: float | None = None) -> pd.DataFrame:
    """The boosters' inputs: TREE_RAW passed through, then TREE_DERIVED (categoricals as integer codes, -1 unknown;
    the events' fields under the same-second rule; the previous attempt), plus any `extra` columns of `f`.  Every
    name is checked against FORBIDDEN.  Needs the previous attempt (add_previous_attempt / prepare on the whole
    season table first).  NaN stays NaN unless `fill` is given."""
    extra = list(extra)
    check_names(list(TREE_RAW) + list(TREE_DERIVED) + extra)
    _need(f, TREE_RAW + tuple(extra) + ("poss_start", "reset_kind", "start_prev_x", "start_prev_y", "start_prev_dist",
                                        "start_prev_clock", "poss_start_clock", "season"), "tree_features")
    _need(f, PREV_COLS, "tree_features", ": call shotfeatures.add_previous_attempt (or prepare) on the whole season "
                                         "table before any filter or sample")
    out = {c: f[c].to_numpy(float) for c in TREE_RAW}
    out["sub"] = pd.Series(sm.submodel_of(f)).map(_SUB_CODE).to_numpy(np.int64)
    starts = {k: i for i, k in enumerate(sm.START_TYPES)}
    kinds = {k: i for i, k in enumerate(tuple(sm.START_TYPES) + tuple(sm.RESET_KINDS))}
    out["poss_start"] = f["poss_start"].map(starts).fillna(-1).to_numpy(np.int64)
    out["reset_kind"] = f["reset_kind"].map(kinds).fillna(-1).to_numpy(np.int64)
    ev = _events(f, "tree_features")
    out["timeout_secs"] = ev["timeout_secs"].to_numpy(float)
    out["n_dfoul"] = ev["n_dfoul"].to_numpy(float)
    clock = f["clock"].to_numpy(float)
    out["prev_att_dist"] = f["prev_att_dist"].to_numpy(float)
    out["prev_att_value"] = f["prev_att_value"].to_numpy(float)
    out["prev_att_blocked"] = f["prev_att_blocked"].to_numpy(float)
    out["prev_att_secs"] = f["prev_att_clock"].to_numpy(float) - clock
    # banded (under 10 / 10-16 / 16+ ft): the raw x/y of the shot that handed the ball over is not rim-mapped
    d_prev = _prev_shot_ft(f)
    out["start_prev_dist_xy"] = np.where(f["start_prev_value"].to_numpy() > 0,
                                         np.where(d_prev < 10.0, 5.0, np.where(d_prev < 16.0, 13.0, 20.0)), np.nan)
    out["trans_secs"] = f["start_prev_clock"].to_numpy(float) - clock
    dreb = np.isin(f["poss_start"].to_numpy(object), ["dreb", "dreb_team"])
    fresh = (f["start_prev_clock"].to_numpy(float) - f["poss_start_clock"].to_numpy(float)) <= STALE_S
    out["prev_stale"] = (dreb & ~((f["start_prev_made"].to_numpy(float) == 0) & fresh)).astype(float)
    out["line_ft"] = line_ft(f)
    out["side"] = _sign(f["xc"].to_numpy(float))
    for c in extra:
        out[c] = f[c].to_numpy(float)
    X = pd.DataFrame(out, index=f.index)
    return X if fill is None else X.fillna(fill)


__all__ = ["BLOCKS", "EVENT_COLS", "FORBIDDEN", "NEEDS", "PREV_COLS", "SAME_TOL", "STALE_S", "TREE_DERIVED", "TREE_RAW",
           "add_previous_attempt", "before_shot", "check_names", "event_fields", "line_ft", "prepare", "tree_features"]
