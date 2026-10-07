"""Shot-quality models: the probability a LEAGUE-AVERAGE shooter makes this exact shot (the shot-quality build).

The owner's rulings (2026-10-06): the quality describes the shot alone -- not who took it, not who defended it --
and no play-by-play scorer tag (`subType`) is ever an input.  Everything here is read from the shot frame
(shotframe.derive) and the rebuilt shot clock (shotclock.rebuild).

The model.  Three penalised logistic regressions -- shots at the rim (two-pointers within 4 ft, and the
pre-2011 twos with no coordinates), other twos, threes -- on BLOCKS of inputs, so the ablation can add one block
at a time:

    spot      distance (natural spline), angle off the rim's axis, corner three, the no-coordinates cell
    start     how the possession began, seconds into it (spline), and the early seconds of each start type
    putback   seconds since the offence's own offensive rebound, in bands
    clock     the rebuilt shot clock (what the offence really had: the game clock when shorter) in FOUR BANDS
              (0-4, 4-10, 10-18, 18-24 s), whether the clock was off, and the kind of its last reset.  Bands, not a
              spline: the rebuild is 0.8 s off at the median against the 2014-15 truth, but the shot-clock
              violations read only 68-74% within 2 s of zero in every season (stage-3 gate failed as written; the
              plan's fallback).  `clockfine` (the spline) is an ablation row only
    context   period, the last seconds of a period, score margin, garbage time
    fatigue   the shooter's game seconds on court since he last came on
    prev      for a possession that began with a defensive rebound or after a make, the shot that handed it
              over: a three, a long two, blocked

Two OFFSETS are fitted alongside and set to ZERO when pricing: one per shooter-season and one per arena-season
(ridge-penalised, by backfitting).  Fitted, they stop the inputs from absorbing who takes which shots (good
shooters take harder ones) and the arena's coordinate habits; zeroed, they leave a league-average shooter in a
league-average arena.

Training is on OTHER seasons only (`train_seasons`): a block of three rated seasons never sees itself or the
season on either side, nor the current season (ruling 3).  The level is then set per season, game half, shot
value and distance band from the OTHER half's makes, padded toward zero with `LEVEL_K` attempts (`relevel`):
the league's shooting moves year to year (2P% .465 in 2000-02, .547 in 2024-26).  The level is the season's own
(both halves), as the shipped curve's is: setting each half from the other half was tried first and rejected, because
the league's three-point rate moves between halves by as much as the spread of defences' shot quality.

The search's options (`fit`, the shot-quality search; every default reproduces the model above bit for bit):

    standardise      divide every non-intercept column by its training sd before the ridge.  A ridge of 1 on the raw
                     spline columns (cubes of feet) is no penalty at all, so without this an interaction menu overfits;
                     the coefficients are stored in natural units (the scale divided back out), the sds for the record
    ridge_int        the ridge on interaction columns, apart from the main columns' `ridge`
    knots            {"dist": 4|6|8, "tposs": 3|5|7, "on": 3|5}: that many spline knots at the training rows'
                     quantiles (per sub-model) instead of the fixed ones; the positions are stored with the model
    inter            interaction blocks (INTER_BLOCKS): distance x angle (spot2d), a possession-time curve per start
                     group (start_x_time), clock band x distance (clock_x_dist), putback band x distance at the rim
                     (putback_x_dist), late-game margin and end-of-period distance (context_x)
    era_lambda       an `era` block: the intercept and every main column outside `spot`, times (season - the rated
                     block's middle season) / 10, with its own ridge.  The main effects are then read at the rated
                     block's own era, and the drift the half-life can only down-weight is fitted instead.  Not the
                     spot columns (the level cells already reset distance every season), not the interactions
    half_life_past / half_life_future
                     closeness weights with separate decay before and after the block (`closeness_weights`): the
                     tracking-taught runs found the 5-season half-life too flat (the season next door counted 4x
                     gained -0.57 / -0.81 squared points of gap)

Model layer: frames in, arrays out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BLOCKS = ("spot", "start", "putback", "clock", "context", "fatigue", "prev")
# ablation-only blocks: never in the default list
EXTRA_BLOCKS = ("clockfine", "tracking", "start_type", "start_time", "start_early")
SUBMODELS = ("rim", "mid", "three")
RIM_FT = 4.0
# the tracking teacher's true-distance spline per sub-model (the rim sub-model is the 0-6 ft zone, true 0-9 ft or so)
TRUE_DIST_KNOTS = {"rim": (0.5, 1.5, 3.0, 4.5, 6.5), "mid": (5.0, 8.0, 12.0, 16.0, 20.0), "three": (23.0, 24.0, 25.5, 27.5)}
LEVEL_K = 150.0
START_TYPES = ("made_fg", "made_ft", "dreb", "dreb_team", "steal", "dead_tov", "jump_ball", "period_start", "forced")
RESET_KINDS = ("oreb", "dfoul", "jump")          # besides the start types themselves
KNOTS = {"rim": (0.5, 1.5, 2.5, 3.5), "mid": (5.0, 8.0, 12.0, 16.0, 20.0, 23.0),
         "three": (22.5, 24.0, 25.0, 26.5, 28.5, 32.0)}
TPOSS_KNOTS = (2.0, 5.0, 8.0, 12.0, 18.0)        # seconds into the possession (the `start` block)
ON_KNOTS = (1.0, 3.0, 6.0, 10.0, 16.0)           # minutes on court (the `fatigue` block)
# knots at training quantiles (`knots=`): Harrell's placements for 3-7 knots, evenly spaced from 5% to 95% otherwise
QUANTILE_PROBS = {3: (0.10, 0.50, 0.90), 4: (0.05, 0.35, 0.65, 0.95), 5: (0.05, 0.275, 0.50, 0.725, 0.95),
                  6: (0.05, 0.23, 0.41, 0.59, 0.77, 0.95),
                  7: (0.025, 0.1833, 0.3417, 0.50, 0.6583, 0.8167, 0.975)}
INTER_BLOCKS = ("spot2d", "start_x_time", "clock_x_dist", "putback_x_dist", "context_x")
# the start groups of `start_x_time`; after a make (made_fg, made_ft) is the reference
START_GROUPS = {"dreb": ("dreb", "dreb_team"), "steal": ("steal",),
                "dead": ("dead_tov", "jump_ball", "period_start", "forced")}
# The other-half level cells (`relevel`): one-foot bins where the shots are dense, as the shipped curve's are (a
# season's own bins padded toward its spline with 150 shots), so the season's distance shape is the other half's
# and the model supplies only what the play-by-play adds inside a bin.  Coarse bands lost on threes (2026-10-06).
BANDS = {"rim": (0.0, 1.0, 2.0, 3.0, 4.0),
         "mid": (4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 99.0),
         "three": (0.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 27.0, 28.0, 30.0, 33.0, 99.0)}


# ------------------------------------------------------------------------------------------ the shared numerics
def natural_spline(x: np.ndarray, knots) -> np.ndarray:
    """Natural cubic spline basis (ESL 5.4): [1, x, N_3 .. N_K]."""
    k = np.asarray(knots, dtype=float)
    K = len(k)

    def d(j):
        return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3) / (k[-1] - k[j])

    cols = [np.ones_like(x), x] + [d(j) - d(K - 2) for j in range(K - 2)]
    return np.column_stack(cols)


def logistic_irls(X, y, w=None, ridge=1e-3, iters=25, offset=None, beta0=None, penalise_first=False):
    """IRLS for a logistic regression with a ridge on every coefficient but the first (the intercept).

    `ridge` is one value for every column, or a vector with one per column (the search penalises main, interaction
    and era columns apart); the first entry is ignored unless `penalise_first`."""
    n, p = X.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=float).copy()
    if np.ndim(ridge) == 0:
        P = ridge * np.eye(p)
    else:
        r = np.asarray(ridge, dtype=float)
        if r.shape != (p,):
            raise ValueError(f"ridge has {r.shape} entries for {p} columns")
        P = np.diag(r)
    if not penalise_first:
        P[0, 0] = 0.0
    for _ in range(iters):
        xb = X @ beta
        eta = xb + off
        mu = 1.0 / (1.0 + np.exp(-eta))
        s = np.maximum(mu * (1 - mu), 1e-6) * w
        z = xb + (y - mu) / np.maximum(mu * (1 - mu), 1e-6)
        Xs = X * s[:, None]                                      # once per step: it is the n x p temporary
        new = np.linalg.solve(Xs.T @ X + P, Xs.T @ z)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------------------------------------ the design
def submodel_of(frame: pd.DataFrame) -> np.ndarray:
    two = frame["value"].to_numpy() == 2
    rim = two & ((frame["dist_xy"].to_numpy(float) < RIM_FT) | frame["noloc"].to_numpy(bool))
    return np.where(~two, "three", np.where(rim, "rim", "mid"))


def _onehot(values, levels) -> np.ndarray:
    v = np.asarray(values, dtype=object)
    return np.column_stack([(v == lv).astype(float) for lv in levels])


def _bands(x, edges) -> np.ndarray:
    """One column per band [e_i, e_{i+1}); NaN falls in no band."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([((x >= a) & (x < b)).astype(float) for a, b in zip(edges[:-1], edges[1:])])


def _knots(knot_pos, key, default):
    return default if not knot_pos or knot_pos.get(key) is None else knot_pos[key]


def block_columns(f: pd.DataFrame, sub: str, block: str, knot_pos: dict | None = None) -> tuple[np.ndarray, list]:
    """The columns of one block for one sub-model, and their names.  No intercept (the design adds it).

    `knot_pos` ({"dist": .., "tposs": .., "on": ..}, any subset) replaces the fixed knot POSITIONS of the distance,
    possession-time and time-on-court splines (`knot_positions` places them at training quantiles).
    Blocks named in shotfeatures.BLOCKS (the search's new features) are built there."""
    from . import shotfeatures
    if block in shotfeatures.BLOCKS:
        return shotfeatures.BLOCKS[block](f, sub)
    if block == "spot":
        d = np.clip(f["dist_xy"].to_numpy(float), 0.0, 40.0)
        S = natural_spline(d, _knots(knot_pos, "dist", KNOTS[sub]))[:, 1:]
        cols, names = [S], [f"dist{i}" for i in range(S.shape[1])]
        ang = np.clip(f["angle"].to_numpy(float), 0.0, 90.0) / 90.0
        if sub != "rim":
            cols += [ang[:, None], (ang ** 2)[:, None]]
            names += ["angle", "angle2"]
        if sub == "three":
            cols.append(f["corner3"].to_numpy(float)[:, None])
            names.append("corner3")
        if sub == "rim":
            cols.append(f["noloc"].to_numpy(float)[:, None])
            names.append("noloc")
        return np.column_stack(cols), names
    if block in ("start_type", "start_time", "start_early"):           # ablation-only parts of `start`
        X, nm = block_columns(f, sub, "start", knot_pos)
        keep = [i for i, n in enumerate(nm) if (block == "start_type" and n.startswith("start_"))
                or (block == "start_time" and n.startswith("tposs")) or (block == "start_early" and n.startswith("early_"))]
        return X[:, keep], [nm[i] for i in keep]
    if block == "start":
        st = f["poss_start"].to_numpy(object)
        t = np.clip(f["secs_into_poss"].to_numpy(float), 0.0, 30.0)
        O = _onehot(st, START_TYPES[1:])                             # made_fg is the reference
        S = natural_spline(t, _knots(knot_pos, "tposs", TPOSS_KNOTS))[:, 1:]
        early = (t <= 6.0).astype(float)
        E = np.column_stack([early * (st == k) for k in ("dreb", "steal", "made_fg", "made_ft", "dead_tov")])
        return np.column_stack([O, S, E]), ([f"start_{k}" for k in START_TYPES[1:]] + [f"tposs{i}" for i in range(S.shape[1])]
                                           + [f"early_{k}" for k in ("dreb", "steal", "made_fg", "made_ft", "dead_tov")])
    if block == "putback":
        s = f["secs_since_oreb"].to_numpy(float)
        B = _bands(s, (0.0, 1.0, 2.0, 4.0, 8.0, 1e9))
        return B, ["oreb_0_1", "oreb_1_2", "oreb_2_4", "oreb_4_8", "oreb_8p"]
    if block in ("clock", "clockfine"):
        sc = np.clip(f["sc_eff"].to_numpy(float), 0.0, 24.0)
        if block == "clock":
            S = _bands(sc, (0.0, 4.0, 10.0, 18.0, 24.01))[:, [0, 1, 3]]      # 10-18 s is the reference
            sn = ["sc_0_4", "sc_4_10", "sc_18_24"]
        else:
            S = natural_spline(sc, (1.5, 4.0, 8.0, 13.0, 19.0, 23.0))[:, 1:]
            sn = [f"sc{i}" for i in range(S.shape[1])]
        off = f["clock_off"].to_numpy(float)[:, None]
        R = _onehot(f["reset_kind"].to_numpy(object), RESET_KINDS)
        return np.column_stack([S, off, R]), sn + ["clock_off"] + [f"reset_{k}" for k in RESET_KINDS]
    if block == "context":
        per = np.clip(f["period"].to_numpy(int), 1, 5)
        P = _onehot(per, (2, 3, 4, 5))
        left = f["clock"].to_numpy(float)
        end = np.column_stack([(left <= 3.0), (left > 3.0) & (left <= 8.0)]).astype(float)
        m = f["margin"].to_numpy(float)
        M = _bands(m, (-1e9, -15.0, -6.0, 6.0, 15.0, 1e9))[:, [0, 1, 3, 4]]     # the close band is the reference
        gt = ((per == 4) & (np.abs(m) >= 15.0) & (left < 360.0)).astype(float)[:, None]
        return np.column_stack([P, end, M, gt]), ["q2", "q3", "q4", "ot", "end3", "end8", "m_le15", "m_le6", "m_ge6", "m_ge15", "garbage"]
    if block == "fatigue":
        s = np.clip(f["shooter_secs_on"].to_numpy(float), 0.0, 1500.0)
        s = np.where(np.isfinite(s), s, 300.0)
        S = natural_spline(s / 60.0, _knots(knot_pos, "on", ON_KNOTS))[:, 1:]
        return S, [f"on{i}" for i in range(S.shape[1])]
    if block == "tracking":
        # the teacher's own inputs (tracked seasons only): how open, how much clock, off the catch or the bounce
        dd = np.log1p(np.clip(f["def_dist"].to_numpy(float), 0.0, 10.0))
        D = natural_spline(dd, (np.log1p(1.0), np.log1p(2.5), np.log1p(4.0), np.log1p(6.0), np.log1p(9.0)))[:, 1:]
        # the TRUE (tracking) distance where the teacher has it: the logged spot inside 10 ft is coded to two zones
        # because it knows the result (DECISIONS.md, the location leak); the teacher must not lean on it
        has_true = "true_dist" in f.columns
        dist = np.clip((f["true_dist"] if has_true else f["dist_xy"]).to_numpy(float), 0.0, 40.0)
        DX = D[:, :1] * (dist[:, None] / 20.0)                        # the defender matters more further out
        tsc = f["true_sc"].to_numpy(float)
        off = np.isnan(tsc)
        S = natural_spline(np.where(off, 12.0, np.clip(tsc, 0.0, 24.0)), (2.0, 5.0, 9.0, 15.0, 21.0))[:, 1:]
        S[off] = 0.0
        full = (tsc >= 23.95).astype(float)                           # the full reset off the rim (tips)
        dr = f["dribbles"].to_numpy(float)
        DR = _bands(dr, (0, 1, 2, 3, 7, 1e9))[:, 1:]                 # 0 dribbles is the reference
        tt = np.clip(f["touch_time"].to_numpy(float), 0.0, 24.0)
        TT = natural_spline(tt, (0.5, 1.5, 3.0, 6.0))[:, 1:]
        cs = ((dr == 0) & (tt <= 2.0)).astype(float)
        cols = [D, DX, S, off[:, None].astype(float), full[:, None], DR, TT, cs[:, None]]
        names = ([f"def{i}" for i in range(D.shape[1])] + ["def_x_dist"] + [f"tsc{i}" for i in range(S.shape[1])]
                 + ["tsc_off", "tsc_full"] + ["drib1", "drib2", "drib3_6", "drib7p"]
                 + [f"touch{i}" for i in range(TT.shape[1])] + ["catch_shoot"])
        if has_true:
            T = natural_spline(dist, TRUE_DIST_KNOTS[sub])[:, 1:]
            cols.append(T)
            names += [f"true_dist{i}" for i in range(T.shape[1])]
        return np.column_stack(cols), names
    if block == "prev":
        pv = f["start_prev_value"].to_numpy(float)
        pd_ = f["start_prev_dist"].to_numpy(float)
        made = f["start_prev_made"].to_numpy(float)
        miss = (made == 0)
        cols = np.column_stack([miss & (pv == 3), miss & (pv == 2) & (pd_ >= 16), (f["start_prev_blocked"].to_numpy(float) == 1)])
        return cols.astype(float), ["after_miss3", "after_miss_long2", "after_block"]
    raise ValueError(block)


def inter_columns(f: pd.DataFrame, sub: str, name: str, knot_pos: dict | None = None) -> tuple[np.ndarray, list]:
    """The columns of one interaction block for one sub-model (INTER_BLOCKS), and their names.

    Each is a plain-words effect a tree would find, kept inside the penalised regression (and its shooter and arena
    terms) so that it stays shooter-neutral and readable.  The splines use the same knots as their main blocks."""
    n = len(f)
    d = np.clip(f["dist_xy"].to_numpy(float), 0.0, 40.0)
    if name == "spot2d":                         # the angle's effect changes with distance; the corner three's too
        S = natural_spline(d, _knots(knot_pos, "dist", KNOTS[sub]))[:, 1:]
        ang = np.clip(f["angle"].to_numpy(float), 0.0, 90.0) / 90.0
        cols, names = [S * ang[:, None]], [f"dist{i}_x_angle" for i in range(S.shape[1])]
        if sub == "rim":                         # `spot` carries no angle at the rim
            cols.insert(0, ang[:, None])
            names.insert(0, "rim_angle")
        if sub == "three":
            cols.append((f["corner3"].to_numpy(float) * (d - 22.0))[:, None])
            names.append("corner3_x_dist")
        return np.column_stack(cols), names
    if name == "start_x_time":                   # a possession-time curve per start group; after a make the reference
        st = f["poss_start"].to_numpy(object)
        t = np.clip(f["secs_into_poss"].to_numpy(float), 0.0, 30.0)
        S = natural_spline(t, _knots(knot_pos, "tposs", TPOSS_KNOTS))[:, 1:]
        cols, names = [], []
        for g, kinds in START_GROUPS.items():
            cols.append(S * np.isin(st, kinds)[:, None])
            names += [f"tposs{i}_x_{g}" for i in range(S.shape[1])]
        return np.column_stack(cols), names
    if name == "clock_x_dist":                   # late in the clock a long shot is a forced one
        S = natural_spline(d, _knots(knot_pos, "dist", KNOTS[sub]))[:, 1:]
        sc = np.clip(f["sc_eff"].to_numpy(float), 0.0, 24.0)
        B = _bands(sc, (0.0, 4.0, 10.0, 18.0, 24.01))[:, [0, 1, 3]]          # 10-18 s is the reference
        cols, names = [], []
        for j, bn in enumerate(("sc_0_4", "sc_4_10", "sc_18_24")):
            cols.append(S * B[:, [j]])
            names += [f"dist{i}_x_{bn}" for i in range(S.shape[1])]
        return np.column_stack(cols), names
    if name == "putback_x_dist":                 # a tip-in a foot out is not a putback from four feet (rim only)
        if sub != "rim":
            return np.empty((n, 0)), []
        B = _bands(f["secs_since_oreb"].to_numpy(float), (0.0, 1.0, 2.0, 4.0, 8.0, 1e9))
        names = [f"{b}_x_dist" for b in ("oreb_0_1", "oreb_1_2", "oreb_2_4", "oreb_4_8", "oreb_8p")]
        return B * np.clip(d, 0.0, RIM_FT)[:, None], names
    if name == "context_x":                      # the last two minutes of a close fourth; end-of-period shots by distance
        per = np.clip(f["period"].to_numpy(int), 1, 5)
        left = f["clock"].to_numpy(float)
        late = ((per >= 4) & (left <= 120.0)).astype(float)
        M = _bands(f["margin"].to_numpy(float), (-1e9, -15.0, -6.0, 6.0, 15.0, 1e9))[:, [0, 1, 3, 4]]
        end = np.column_stack([(left <= 3.0), (left > 3.0) & (left <= 8.0)]).astype(float)
        names = ["late2", "m_le15_x_late2", "m_le6_x_late2", "m_ge6_x_late2", "m_ge15_x_late2", "end3_x_dist",
                 "end8_x_dist"]
        return np.column_stack([late[:, None], M * late[:, None], end * (d / 10.0)[:, None]]), names
    raise ValueError(f"unknown interaction block {name!r}; known: {INTER_BLOCKS}")


def era_of(season, era: dict) -> np.ndarray:
    """The era variable: (season - the rated block's middle season) / 10, clipped to the training range on request."""
    e = (np.asarray(season, dtype=float) - era["mid"]) / 10.0
    return np.clip(e, era["lo"], era["hi"]) if era.get("clip") else e


def design_full(f: pd.DataFrame, sub: str, blocks, knot_pos: dict | None = None, inter=(),
                era: dict | None = None) -> tuple[np.ndarray, list, np.ndarray]:
    """The design with the search's options, its names, and each column's kind ("main", "inter" or "era"; the
    intercept is "main").  With the defaults it is `design` exactly."""
    parts, names, kinds = [np.ones((len(f), 1))], ["intercept"], ["main"]
    e = None if era is None else era_of(f["season"].to_numpy(), era)[:, None]
    era_parts, era_names = ([e], ["era:intercept"]) if e is not None else ([], [])
    for b in blocks:
        X, nm = block_columns(f, sub, b, knot_pos)
        parts.append(X)
        names += nm
        kinds += ["main"] * len(nm)
        if e is not None and b != "spot":            # the era block: intercept and the non-spot main columns
            era_parts.append(X * e)
            era_names += [f"era:{x}" for x in nm]
    for b in inter:
        X, nm = inter_columns(f, sub, b, knot_pos)
        parts.append(X)
        names += [f"{b}:{x}" for x in nm]
        kinds += ["inter"] * len(nm)
    parts += era_parts
    names += era_names
    kinds += ["era"] * len(era_names)
    return np.column_stack(parts), names, np.asarray(kinds)


def design(f: pd.DataFrame, sub: str, blocks, knot_pos: dict | None = None, inter=(),
           era: dict | None = None) -> tuple[np.ndarray, list]:
    X, names, _ = design_full(f, sub, blocks, knot_pos, inter, era)
    return X, names


def knot_positions(f: pd.DataFrame, sub: str, counts: dict) -> dict:
    """Knot positions at this sub-model's training-row quantiles: {"dist": n, "tposs": n, "on": n} -> positions.

    Repeated quantiles (whole-second clocks) are merged, so a spline can get fewer knots than asked; fewer than three
    distinct positions, or no rows, keeps the fixed knots.  At the rim only located attempts place the distance
    knots."""
    def values(key):                                   # what each spline reads, clipped as block_columns clips it
        if key == "dist":
            d = f["dist_xy"].to_numpy(float)
            return np.clip(d[~f["noloc"].to_numpy(bool)] if sub == "rim" else d, 0.0, 40.0)
        if key == "tposs":
            return np.clip(f["secs_into_poss"].to_numpy(float), 0.0, 30.0)
        s = np.clip(f["shooter_secs_on"].to_numpy(float), 0.0, 1500.0)
        return np.where(np.isfinite(s), s, 300.0) / 60.0

    out = {}
    for key, k in counts.items():
        if key not in ("dist", "tposs", "on"):
            raise ValueError(f"knots: unknown spline {key!r}; known: dist, tposs, on")
        if k is None:
            continue
        x = values(key)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        probs = QUANTILE_PROBS.get(int(k), tuple(np.linspace(0.05, 0.95, int(k))))
        q = np.unique(np.round(np.quantile(x, probs), 6))
        if len(q) >= 3:
            out[key] = tuple(float(v) for v in q)
    return out


# ------------------------------------------------------------------------------------------ fitting
@dataclass
class ShotModel:
    blocks: tuple
    coef: dict = field(default_factory=dict)          # sub-model -> (names, beta), beta in natural units
    offsets: dict = field(default_factory=dict)       # sub-model -> {"shooter": Series, "arena": Series}
    seasons: tuple = ()                                # the seasons it was trained on
    # what rebuilds the design at prediction (all empty for the shipped model)
    inter: tuple = ()                                  # interaction blocks (INTER_BLOCKS)
    knots: dict = field(default_factory=dict)         # sub-model -> {"dist"/"tposs"/"on": positions}; absent = fixed
    scale: dict = field(default_factory=dict)         # sub-model -> the column sds a standardised fit divided by
    era: dict | None = None                            # {"mid", "lo", "hi", "clip"}: the era block's centre and range
    params: dict = field(default_factory=dict)        # the fit's arguments, for the record

    def design(self, f: pd.DataFrame, sub: str) -> tuple[np.ndarray, list]:
        """This model's design for one sub-model's rows (its knots, interactions and era), in natural units:
        X @ coef[sub][1] is the offset-free logit."""
        return design(f, sub, self.blocks, self.knots.get(sub), self.inter, self.era)

    def check_may_price(self, seasons) -> None:
        """A model never prices a season it trained on, nor a season next to one (the leakage rule)."""
        seen = set(self.seasons)
        bad = sorted(int(s) for s in set(seasons) if {s - 1, s, s + 1} & seen)
        if bad:
            raise ValueError(f"this model trained on {sorted(seen)} and may not price {bad}")

    def margin_train(self, f: pd.DataFrame, with_offsets: bool = True) -> np.ndarray:
        """The logit on rows of the seasons it TRAINED on (a booster's base margin), offsets kept by default.
        predict_raw refuses those seasons by design; this is the one door, and it checks the other way round."""
        bad = set(int(s) for s in np.unique(f["season"])) - set(self.seasons)
        if bad:
            raise ValueError(f"margin_train is for training seasons only; {sorted(bad)} are not among {sorted(self.seasons)}")
        saved, self.seasons = self.seasons, ()
        try:
            return self.predict_raw(f, with_offsets=with_offsets)
        finally:
            self.seasons = saved

    def predict_raw(self, f: pd.DataFrame, with_offsets: bool = False) -> np.ndarray:
        """The logit of a league-average shooter in a league-average arena (offsets zeroed), before relevel.

        `with_offsets` keeps the fitted shooter-season and arena-season offsets where the model has them: THIS
        shooter's make probability, which is what a teacher must hand a student that fits its own offsets."""
        self.check_may_price(np.unique(f["season"]))
        sub = submodel_of(f)
        eta = np.zeros(len(f))
        for s in SUBMODELS:
            m = sub == s
            if not m.any():
                continue
            X, _ = self.design(f[m], s)
            eta[m] = X @ self.coef[s][1]
            if with_offsets and s in self.offsets:
                g = f[m]
                ks = g["season"].astype(str) + ":" + g["shooter"].astype(str)
                ka = g["season"].astype(str) + ":" + g["arena"].astype(str)
                eta[m] += ks.map(self.offsets[s]["shooter"]).fillna(0.0).to_numpy()                     + ka.map(self.offsets[s]["arena"]).fillna(0.0).to_numpy()
        return eta


def _group_offsets(eta, y, w, codes, n_groups, lam, iters=2):
    """Ridge-penalised logit offsets per group, a few Newton steps from zero."""
    delta = np.zeros(n_groups)
    for _ in range(iters):
        mu = _sigmoid(eta + delta[codes])
        g = np.bincount(codes, weights=w * (y - mu), minlength=n_groups) - lam * delta
        h = np.bincount(codes, weights=w * mu * (1 - mu), minlength=n_groups) + lam
        delta += g / h
    return delta


def fit(train: pd.DataFrame, blocks, weights=None, ridge: float = 1.0, lam_shooter: float = 50.0,
        lam_arena: float = 50.0, rounds: int = 4, offsets: bool = True, standardise: bool = False,
        ridge_int: float | None = None, era_lambda: float | None = None, knots: dict | None = None, inter=(),
        half_life_past: float | None = None, half_life_future: float | None = None, block=None,
        era_clip: bool = False) -> ShotModel:
    """The three sub-models by backfitting: inputs given the offsets, then each offset given the rest.

    The search's options (all off by default, which is the shipped fit bit for bit; see the module docstring):
    `standardise` (ridges act on sd-scaled columns), `ridge_int` (the interactions' ridge; default `ridge`),
    `era_lambda` (the era block's ridge; None = no era block), `knots` ({"dist", "tposs", "on"} -> a number of knots
    at training quantiles), `inter` (INTER_BLOCKS), `half_life_past` / `half_life_future` (closeness weights for
    `block`, MULTIPLIED into `weights` -- so `weights` must then hold only the other factors, never the 5-season
    closeness too; float("inf") is no decay on that side).  `block` is the rated seasons the model is for (any
    iterable; its min and max are used): required by the era block and the half-lives.  `era_clip` holds the era
    variable inside the training seasons' range at prediction (by default a rated season is read at its own era)."""
    block = None if block is None else tuple(block)
    inter = tuple(dict.fromkeys(inter))
    bad = set(inter) - set(INTER_BLOCKS)
    if bad:
        raise ValueError(f"unknown interaction blocks {sorted(bad)}; known: {INTER_BLOCKS}")
    decay = half_life_past is not None or half_life_future is not None
    if (decay or era_lambda is not None) and block is None:
        raise ValueError("the era block and the past / future half-lives need `block`, the rated seasons")
    seasons = train["season"].to_numpy()
    model = ShotModel(blocks=tuple(blocks), seasons=tuple(sorted(int(s) for s in np.unique(seasons))), inter=inter)
    model.params = dict(ridge=ridge, lam_shooter=lam_shooter, lam_arena=lam_arena, rounds=rounds, offsets=offsets,
                        standardise=standardise, ridge_int=ridge_int, era_lambda=era_lambda,
                        knots=dict(knots) if knots else None, inter=inter, half_life_past=half_life_past,
                        half_life_future=half_life_future, era_clip=era_clip,
                        block=None if block is None else (int(min(block)), int(max(block))))
    sub = submodel_of(train)
    w_all = np.ones(len(train)) if weights is None else np.asarray(weights, dtype=float)
    if decay:
        w_all = w_all * closeness_weights(seasons, block, half_life_past=half_life_past,
                                          half_life_future=half_life_future)
    if era_lambda is not None:
        mid = (min(block) + max(block)) / 2.0
        e = (np.unique(seasons).astype(float) - mid) / 10.0
        model.era = dict(mid=float(mid), lo=float(e.min()), hi=float(e.max()), clip=bool(era_clip))
    for s in SUBMODELS:
        m = sub == s
        f = train[m]
        y = f["made"].to_numpy(float)
        w = w_all[m]
        if knots:
            kp = knot_positions(f, s, knots)
            if kp:
                model.knots[s] = kp
        X, names, kinds = design_full(f, s, blocks, model.knots.get(s), inter, model.era)
        pen = ridge
        if (kinds != "main").any():
            pen = np.where(kinds == "inter", ridge if ridge_int is None else ridge_int,
                           np.where(kinds == "era", ridge if era_lambda is None else era_lambda, ridge))
        scale = None
        if standardise:
            sd = X[:, 1:].std(axis=0) if len(X) else np.ones(X.shape[1] - 1)
            # a yes/no column (and its era copy) keeps scale 1: dividing a column of share p by sqrt(p(1-p)) would cut
            # its ridge to ridge x p(1-p), leaving rare indicators almost unpenalised (the search's review, 2026-10-06)
            Xc = X[:, 1:]
            binary = ((Xc == 0) | (Xc == 1)).all(axis=0) if len(X) else np.zeros(X.shape[1] - 1, bool)
            era = np.array([n.startswith("era:") for n in names[1:]]) if len(names) == X.shape[1] else np.zeros(X.shape[1] - 1, bool)
            base_binary = binary.copy()
            if era.any():
                plain = {n: b for n, b in zip(names[1:], binary)}
                base_binary = np.array([b or plain.get(n[4:], False) if e else b for n, b, e in zip(names[1:], binary, era)])
            scale = np.concatenate([[1.0], np.where(base_binary | (sd <= 1e-12), 1.0, sd)])
            X /= scale
            model.scale[s] = scale
        sh = pd.factorize(f["season"].astype(str) + ":" + f["shooter"].astype(str))
        ar = pd.factorize(f["season"].astype(str) + ":" + f["arena"].astype(str))
        d_sh = np.zeros(len(sh[1]))
        d_ar = np.zeros(len(ar[1]))
        beta = None
        for _ in range(rounds if offsets else 1):
            off = d_sh[sh[0]] + d_ar[ar[0]]
            beta = logistic_irls(X, y, w, ridge=pen, offset=off, beta0=beta)
            if not offsets:
                break
            eta = X @ beta
            d_sh = _group_offsets(eta + d_ar[ar[0]], y, w, sh[0], len(sh[1]), lam_shooter)
            d_ar = _group_offsets(eta + d_sh[sh[0]], y, w, ar[0], len(ar[1]), lam_arena)
        model.coef[s] = (names, beta if scale is None else beta / scale)
        model.offsets[s] = {"shooter": pd.Series(d_sh, index=sh[1]), "arena": pd.Series(d_ar, index=ar[1])}
    return model


def train_seasons(block, all_seasons, current: int = 2026) -> list:
    """The seasons a model for this block of rated seasons may train on: never the block, never the season on
    either side of it, never the current season."""
    lo, hi = min(block), max(block)
    return [s for s in all_seasons if (s < lo - 1 or s > hi + 1) and s != current]


def closeness_weights(seasons, block, half_life: float = 5.0, half_life_past: float | None = None,
                      half_life_future: float | None = None) -> np.ndarray:
    """0.5 ** (seasons from the block / half-life).  With `half_life_past` or `half_life_future` given, the seasons
    before and after the block decay apart and `half_life` is not used; a side left None does not decay.

    Apart because the two sides are not alike: 1997-99 trains only on later seasons and 2024-26 only on earlier ones,
    and the league drifts one way (more threes, better twos)."""
    lo, hi = min(block), max(block)
    s = np.asarray(seasons, dtype=float)
    if half_life_past is None and half_life_future is None:
        dist = np.where(s < lo, lo - s, np.where(s > hi, s - hi, 0.0))
        return 0.5 ** (dist / half_life)
    past = np.where(s < lo, lo - s, 0.0) / (np.inf if half_life_past is None else float(half_life_past))
    future = np.where(s > hi, s - hi, 0.0) / (np.inf if half_life_future is None else float(half_life_future))
    return 0.5 ** (past + future)


# ------------------------------------------------------------------------------------------ the level
def band_of(f: pd.DataFrame) -> np.ndarray:
    sub = submodel_of(f)
    d = f["dist_xy"].to_numpy(float)
    out = np.empty(len(f), dtype=object)
    for s in SUBMODELS:
        m = sub == s
        e = BANDS[s]
        idx = np.clip(np.searchsorted(np.asarray(e[1:-1]), d[m], side="right"), 0, len(e) - 2)
        out[m] = [f"{s}{i}" for i in idx]
    if "corner3" in f.columns:
        out[(sub == "three") & f["corner3"].to_numpy(bool)] = "three_corner"
    return out


UPDATE_BANDS = {"rim": (0.0, 2.0, 4.0), "mid": (4.0, 10.0, 16.0, 24.0), "three": (0.0, 25.0, 28.0, 99.0)}


def update_band_of(f: pd.DataFrame) -> np.ndarray:
    """The coarse shot classes the owner's update is sized on (scripts/119: v by class and clock band)."""
    sub = submodel_of(f)
    d = f["dist_xy"].to_numpy(float)
    out = np.empty(len(f), dtype=object)
    for s in SUBMODELS:
        m = sub == s
        e = UPDATE_BANDS[s]
        idx = np.clip(np.searchsorted(np.asarray(e[1:-1]), d[m], side="right"), 0, len(e) - 2)
        out[m] = [f"{s}{i}" for i in idx]
    if "corner3" in f.columns:
        out[(sub == "three") & f["corner3"].to_numpy(bool)] = "three_corner"
    return out


def relevel(eta: np.ndarray, f: pd.DataFrame, k: float = LEVEL_K, mode: str = "season") -> tuple[np.ndarray, pd.DataFrame]:
    """Shift each (season, band) cell's logits by the shift the season's regular-season makes want, padded toward 0.

    `mode="season"` (the default since 2026-10-06): every row of a season takes the shift fitted on all of that
    season's regular season, as the shipped curve does (one team's own shots are about 1/30 of a cell).
    `mode="half"`: rows in half A take the shift fitted on half B and the reverse -- REJECTED: the league's make rate
    moves between the two halves of a season by 0.40 points on threes (0.16 on twos), as much as the whole spread in
    the quality of threes defences allow (0.42), and that drift landed on every team (DECISIONS.md, "The level")."""
    if mode == "season":
        f = f.assign(half="ALL")
    band = band_of(f)
    half = f["half"].astype(object).to_numpy()
    season = f["season"].to_numpy()
    phase = f["phase"].to_numpy() if "phase" in f.columns else np.full(len(f), "RS")
    y = f["made"].to_numpy(float)
    out = eta.copy()
    rows = []
    rs = phase == "RS"
    for s in np.unique(season):
        for b in np.unique(band[season == s]):
            cell = (season == s) & (band == b)
            shifts = {}
            for src in ("A", "B", "ALL"):
                m = cell & rs & ((half == src) if src != "ALL" else True)
                n = int(m.sum())
                if n == 0:
                    shifts[src] = 0.0
                    continue
                delta = 0.0
                for _ in range(6):
                    mu = _sigmoid(eta[m] + delta)
                    delta += float((y[m] - mu).sum() / max((mu * (1 - mu)).sum(), 1e-9))
                shifts[src] = delta * n / (n + k)
            for dst, src in (("A", "B"), ("B", "A")):
                m = cell & rs & (half == dst)
                out[m] = eta[m] + shifts[src]
            other = cell & ~(rs & np.isin(half, ["A", "B"]))
            out[other] = eta[other] + shifts["ALL"]
            rows.append(dict(season=int(s), band=b, shift_A=shifts["A"], shift_B=shifts["B"], shift_all=shifts["ALL"]))
    return out, pd.DataFrame(rows)


def probability(eta) -> np.ndarray:
    return _sigmoid(np.asarray(eta, dtype=float))


def update_after(q: np.ndarray, made: np.ndarray, v: np.ndarray, cap: float = 0.15) -> np.ndarray:
    """The owner's update: quality given the result, m + (y - m) * w with w = v / (m (1 - m)), capped.

    `v` is the variance of tracking quality among shots the play-by-play cannot tell apart (estimated from the
    tracked seasons, by shot class).  A made shot was probably more open than it looked, a miss less."""
    q = np.asarray(q, dtype=float)
    w = np.clip(np.asarray(v, dtype=float) / np.maximum(q * (1 - q), 1e-6), 0.0, cap)
    return q + (np.asarray(made, dtype=float) - q) * w


def after_of(f: pd.DataFrame, q: np.ndarray, vtab: pd.DataFrame) -> np.ndarray:
    """The owner's update, quality given the result, with v looked up by shot class (sub-model, distance band,
    clock band) as scripts/119 measured it; classes it did not measure, and heaves, keep q."""
    sc = f["sc_eff"].to_numpy(float)
    key = pd.DataFrame(dict(sub=submodel_of(f), band=update_band_of(f),
                            clock=np.where(sc < 4, "0-4", np.where(sc < 10, "4-10", "10+"))))
    v = key.merge(vtab[["sub", "band", "clock", "v"]], on=["sub", "band", "clock"], how="left")["v"]
    v = v.fillna(0.0).clip(lower=0.0).to_numpy().copy()
    v[f["heave"].to_numpy()] = 0.0
    return update_after(q, f["made"].to_numpy(float), v)
