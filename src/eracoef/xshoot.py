"""Shooter-level expected points: the shooter's own padded rate, from the other half of the block.

The stage 4 shot term (xpts.py) priced a shot with the LINEUP's fitted eFG% and lost the
out-of-season test, because a shrunk lineup expectation replaced real shot-making signal
(FINDINGS.md section 17).  This module prices the shot with the SHOOTER's own rate instead --
the thing the free-throw term already does, and the free-throw term is the one luck adjustment that
paid.  A field-goal attempt is priced as

    x = curve(distance, season) * ratio(shooter)

where `curve` is the league's make probability at that distance this season (shotcurve.py) and
`ratio` is how much the shooter beats a league-average shooter FROM HIS OWN SPOTS, measured on the
other half of the block's games and padded toward 1:

    p_mix  = sum over his other-half attempts of curve(distance) / attempts
    p_pad  = pad.shrink(makes / attempts, attempts, k, target = p_mix)
    ratio  = p_pad / p_mix

The rate comes from the OTHER half of the block: a possession in a half-A game is priced with the
shooter's half-B totals, so his own makes never price the possessions he is scored on.  Playoff
rows use the whole regular season.  Everything is padded through pad.py, per the owner's rule.

The stints carry the counts by lineup slot (stints.SLOT_COUNTERS); this module turns them into
expectations at window-build time, because the block (and therefore the other half) is a property
of the training block, not of a season.  Each target is a callable `(seasons, cfg, wd_pts) ->
(WindowData, report)` that holdout.Context.design accepts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve
from .pad import mom_k, shrink

OTHER = {"A": "B", "B": "A"}                     # a row in half A is priced with half-B totals
CALIB_BAND = (0.995, 1.005)
SLOTS = ("1", "2", "3", "4", "5")


# ---------------------------------------------------------------------------------------- the tables
_SHOTS_CACHE: dict = {}


def load_shots(seasons, cfg) -> pd.DataFrame:
    """Per game x shooter field-goal totals with the league expectation, regular season, with halves
    (one parquet per season, read once per process)."""
    d = resolve(cfg, "stints")
    parts = []
    for s in seasons:
        p = d / f"{int(s)}_RS_shots.parquet"
        if p not in _SHOTS_CACHE:
            if not p.exists():
                raise FileNotFoundError(f"{p} is missing; rebuild the stints (scripts/02_stints.py {s} {s} RS --force)")
            _SHOTS_CACHE[p] = pd.read_parquet(p)
        parts.append(_SHOTS_CACHE[p])
    return pd.concat(parts, ignore_index=True)


def load_ft(seasons, cfg, halves: pd.Series) -> pd.DataFrame:
    """Per game x shooter free-throw totals from the box score, with the game's half."""
    from .boxtable import season_box
    b = season_box([int(s) for s in seasons], ["RS"], cfg)[["game_id", "player_id", "ftm", "ft_miss"]].copy()
    b["fta"] = b.ftm + b.ft_miss
    b["half"] = b.game_id.map(halves)
    return b[b.half.notna()]


@dataclass
class ShooterRates:
    """Per shooter, per totals-half ("A", "B", "RS"): the padded rates that price his attempts."""
    ratio2: dict            # half -> {player_id: p_pad / p_mix}   (location-adjusted twos)
    ratio3: dict
    p2: dict                # half -> {player_id: padded flat 2P%} (no location)
    p3: dict
    pft: dict               # half -> {player_id: padded FT%}
    league: dict            # kind -> league rate over the block
    k: dict                 # kind -> padding constant, in attempts

    def for_rows(self, kind: str, row_half) -> np.ndarray:
        """The half whose totals price each row: the other regular-season half, or all of it."""
        h = np.asarray(row_half, dtype=object)
        return np.where(h == "A", "B", np.where(h == "B", "A", "RS"))

    def _sorted(self, name: str, h: str) -> tuple[np.ndarray, np.ndarray]:
        """One rate table as (sorted player ids, values), built once per table and kept on the object."""
        cache = self.__dict__.setdefault("_sorted_cache", {})
        if (name, h) not in cache:
            d = getattr(self, name)[h]
            keys = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
            vals = np.fromiter(d.values(), dtype=float, count=len(d))
            o = np.argsort(keys, kind="stable")
            cache[(name, h)] = (keys[o], vals[o])
        return cache[(name, h)]

    def take(self, name: str, h: str, pid, fill: float) -> np.ndarray:
        """`pd.Series(pid).map(table).fillna(fill)` by binary search: the same numbers, ~10x faster, and the
        pricing loops call it once per lineup slot per half on every row of the design."""
        keys, vals = self._sorted(name, h)
        pid = np.asarray(pid, dtype=np.int64)
        if not len(keys):
            return np.full(len(pid), float(fill))
        i = np.searchsorted(keys, pid)
        np.clip(i, 0, len(keys) - 1, out=i)
        return np.where(keys[i] == pid, vals[i], float(fill))


def rates_from_tables(shots: pd.DataFrame, ft: pd.DataFrame, extra: pd.DataFrame | None = None,
                      k_fixed: dict | None = None) -> ShooterRates:
    """The padded per-shooter rates from the two per-game tables (the testable core of shooter_rates).

    `extra` is a shots table from seasons OUTSIDE the block (earlier ones), added whole to every
    half's totals: more attempts per shooter, and never a game that is being priced.  `k_fixed`
    overrides the method-of-moments padding constant per kind, e.g. {"fg3": 450.0} -- the value the
    split-half reliability of high-volume shooters gives (FINDINGS.md section 18).
    """
    def totals(df, cols):
        out = {}
        for h in ("A", "B"):
            out[h] = df[df.half == h].groupby("player_id")[cols].sum()
        out["RS"] = df.groupby("player_id")[cols].sum()
        if extra is not None and len(extra) and all(c in extra.columns for c in cols):
            e = extra.groupby("player_id")[cols].sum()
            out = {h: t.add(e, fill_value=0.0) for h, t in out.items()}
        return out

    T = totals(shots, SHOT_COLS) if isinstance(shots, pd.DataFrame) else shots     # or the totals themselves
    F = totals(ft, FT_COLS) if isinstance(ft, pd.DataFrame) else ft
    return rates_from_totals(T, F, k_fixed)


SHOT_COLS = ["fg2a", "fg2m", "xl2", "fg3a", "fg3m", "xl3"]
FT_COLS = ["ftm", "fta"]
_TOTALS_CACHE: dict = {}


def season_totals(season: int, cfg) -> tuple[dict, dict]:
    """Per shooter per half ("A", "B", "RS") totals of one regular season, shots and free throws, read once per
    process.  A block's totals are the sum over its seasons (`block_totals`)."""
    key = int(season)
    if key not in _TOTALS_CACHE:
        shots = load_shots([key], cfg)
        halves = shots.drop_duplicates("game_id").set_index("game_id")["half"]
        ft = load_ft([key], cfg, halves)
        T = {h: shots[shots.half == h].groupby("player_id")[SHOT_COLS].sum() for h in ("A", "B")}
        T["RS"] = shots.groupby("player_id")[SHOT_COLS].sum()
        F = {h: ft[ft.half == h].groupby("player_id")[FT_COLS].sum() for h in ("A", "B")}
        F["RS"] = ft.groupby("player_id")[FT_COLS].sum()
        _TOTALS_CACHE[key] = (T, F)
    return _TOTALS_CACHE[key]


def block_totals(seasons, cfg, extra_seasons=()) -> tuple[dict, dict]:
    """The per-half totals of a block of seasons: each half summed over the seasons; `extra_seasons` (earlier
    ones, `prev`) added whole to every half."""
    parts = [season_totals(s, cfg) for s in seasons]
    T, F = {}, {}
    for h in ("A", "B", "RS"):
        T[h] = pd.concat([t[h] for t, _ in parts]).groupby(level=0).sum()
        F[h] = pd.concat([f[h] for _, f in parts]).groupby(level=0).sum()
    if extra_seasons:
        ex = [season_totals(s, cfg) for s in extra_seasons]
        eT = pd.concat([t["RS"] for t, _ in ex]).groupby(level=0).sum()
        eF = pd.concat([f["RS"] for _, f in ex]).groupby(level=0).sum()
        T = {h: t.add(eT, fill_value=0.0) for h, t in T.items()}
        F = {h: f.add(eF, fill_value=0.0) for h, f in F.items()}
    return T, F


SHOT_TOTAL_COLS = [f"shot_{c}" for c in SHOT_COLS]        # shot_fg2a, shot_fg2m, shot_xl2, shot_fg3a, ...
SHOT_LEAGUE_COLS = ["shot_lg2", "shot_lg3", "shot_lgpps"]


def player_shot_frame(seasons, cfg, player_ids=None) -> pd.DataFrame:
    """Per-shooter regular-season shot totals over a block, with the block's own league levels beside them.

    `shot_fg2a ... shot_xl3` are the sums of `SHOT_COLS` over the block's games; `shot_lg2` / `shot_lg3` are
    the league's make rate on twos and threes over the same games and `shot_lgpps` its points per field-goal
    attempt, constant down the frame.  Those three are the pad targets of the shot-quality features
    (`gbdt_prior.add_shotq`): padding toward the BLOCK's own level rather than a constant leaves a 40-attempt
    player at his era's average instead of 1997's, and the league make rate on twos moved 0.468 -> 0.550
    across the 28 seasons.

    `player_ids` returns one row per id, in that order, zeros for a player who took no shot in the block;
    without it the frame carries a `player_id` column and only the shooters the block saw.
    """
    T, _ = block_totals([int(s) for s in seasons], cfg)
    t = T["RS"]
    m2, a2 = float(t.fg2m.sum()), float(t.fg2a.sum())
    m3, a3 = float(t.fg3m.sum()), float(t.fg3a.sum())
    lg2 = m2 / a2 if a2 > 0 else 0.48
    lg3 = m3 / a3 if a3 > 0 else 0.35
    lgpps = (2.0 * m2 + 3.0 * m3) / (a2 + a3) if a2 + a3 > 0 else 1.05
    tot = t[SHOT_COLS].astype(float)
    tot.index = tot.index.astype(np.int64)
    tot.columns = SHOT_TOTAL_COLS
    if player_ids is None:
        out = tot.rename_axis("player_id").reset_index()
    else:
        out = tot.reindex(np.asarray(player_ids, dtype=np.int64)).fillna(0.0).reset_index(drop=True)
    out["shot_lg2"], out["shot_lg3"], out["shot_lgpps"] = lg2, lg3, lgpps
    return out


def rates_from_totals(T: dict, F: dict, k_fixed: dict | None = None) -> ShooterRates:
    """`rates_from_tables` from the per-half totals."""
    league, k = {}, {}
    league["fg2"], k["fg2"] = mom_k(T["RS"].fg2m.to_numpy(), T["RS"].fg2a.to_numpy())
    league["fg3"], k["fg3"] = mom_k(T["RS"].fg3m.to_numpy(), T["RS"].fg3a.to_numpy())
    league["ft"], k["ft"] = mom_k(F["RS"].ftm.to_numpy(), F["RS"].fta.to_numpy(), fallback_rate=0.75)
    for kind, v in (k_fixed or {}).items():
        k[kind] = float(v)
    ratio2, ratio3, p2, p3, pft = {}, {}, {}, {}, {}
    for h in ("A", "B", "RS"):
        t = T[h]
        for kind, ratio, flat in (("fg2", ratio2, p2), ("fg3", ratio3, p3)):
            a, m, xl = t[f"{kind}a"].to_numpy(), t[f"{kind}m"].to_numpy(), t[f"xl{kind[-1]}"].to_numpy()
            p_mix = np.where(a > 0, xl / np.where(a > 0, a, 1.0), league[kind])
            raw = np.where(a > 0, m / np.where(a > 0, a, 1.0), 0.0)
            p_pad = shrink(raw, a, k[kind], p_mix)
            ratio[h] = dict(zip(t.index.astype(int), p_pad / np.maximum(p_mix, 1e-6)))
            flat[h] = dict(zip(t.index.astype(int), shrink(raw, a, k[kind], league[kind])))
        f = F[h]
        raw = np.where(f.fta > 0, f.ftm / np.where(f.fta > 0, f.fta, 1.0), 0.0)
        pft[h] = dict(zip(f.index.astype(int), shrink(raw, f.fta.to_numpy(), k["ft"], league["ft"])))
    return ShooterRates(ratio2, ratio3, p2, p3, pft, league, k)


def shooter_rates(seasons, cfg, prev: int = 0, k_fixed: dict | None = None) -> ShooterRates:
    """The padded per-shooter rates for a block of seasons, from the shots tables and the box scores.

    `prev` adds the P seasons BEFORE the block's first season, whole.  Anchored to the block's start
    rather than to each row's season so the held-out season (which sits inside the block's gap, or
    just after it) can never be one of them.  Seasons with no stints built are skipped.
    """
    seasons = sorted(int(s) for s in seasons)
    have = []
    if prev:
        d = resolve(cfg, "stints")
        have = [s for s in range(seasons[0] - prev, seasons[0]) if (d / f"{s}_RS_shots.parquet").exists()]
    T, F = block_totals(seasons, cfg, extra_seasons=have)
    return rates_from_totals(T, F, k_fixed=k_fixed)


# ---------------------------------------------------------------------------------------- pricing the rows
def expected_makes(cnt: pd.DataFrame, rates: ShooterRates, location: bool = True) -> pd.DataFrame:
    """Per row: x2pm, x3pm, xftm, and the first attempt's xpts1 and xchance1, from the slot counters.

    `cnt` is WindowData.counters (the offense's slot counters, the five player ids, the row's half).
    A shooter absent from the pricing half is the average shooter from his spots (ratio 1) or the
    league (flat); slot `x` (subbed out before the possession closed) is priced at the league.
    """
    n = len(cnt)
    which = rates.for_rows("fg2", cnt["half"].to_numpy())
    masks = [(h, m) for h in ("A", "B", "RS") if (m := which == h).any()]     # once, not once per slot
    x2, x3, xft = np.zeros(n), np.zeros(n), np.zeros(n)
    xp1, xc1 = np.zeros(n), np.zeros(n)
    for s in SLOTS:
        pid = cnt[f"pid_s{s}"].to_numpy()
        r2 = np.ones(n); r3 = np.ones(n); f2 = np.full(n, rates.league["fg2"]); f3 = np.full(n, rates.league["fg3"])
        q = np.full(n, rates.league["ft"])
        for h, m in masks:
            p = pid[m]
            r2[m] = rates.take("ratio2", h, p, 1.0)
            r3[m] = rates.take("ratio3", h, p, 1.0)
            f2[m] = rates.take("p2", h, p, rates.league["fg2"])
            f3[m] = rates.take("p3", h, p, rates.league["fg3"])
            q[m] = rates.take("pft", h, p, rates.league["ft"])
        c = {k: cnt[f"{k}_s{s}"].to_numpy(dtype=float) for k in
             ("fg2a", "xl2", "fg3a", "xl3", "fta", "fg2a1", "xl2_1", "fg3a1", "xl3_1", "fta1", "ftlast1")}
        if location:
            e2, e3, e2_1, e3_1 = c["xl2"] * r2, c["xl3"] * r3, c["xl2_1"] * r2, c["xl3_1"] * r3
        else:
            e2, e3, e2_1, e3_1 = c["fg2a"] * f2, c["fg3a"] * f3, c["fg2a1"] * f2, c["fg3a1"] * f3
        x2 += e2
        x3 += e3
        xft += c["fta"] * q
        xp1 += 2.0 * e2_1 + 3.0 * e3_1 + c["fta1"] * q
        xc1 += (c["fg2a1"] - e2_1) + (c["fg3a1"] - e3_1) + c["ftlast1"] * (1.0 - q)
    # slot x: the league's view, no shooter
    cx = {k: cnt[f"{k}_sx"].to_numpy(dtype=float) for k in
          ("fg2a", "xl2", "fg3a", "xl3", "fta", "fg2a1", "xl2_1", "fg3a1", "xl3_1", "fta1", "ftlast1")}
    q = rates.league["ft"]
    if location:
        e2, e3, e2_1, e3_1 = cx["xl2"], cx["xl3"], cx["xl2_1"], cx["xl3_1"]
    else:
        e2, e3 = cx["fg2a"] * rates.league["fg2"], cx["fg3a"] * rates.league["fg3"]
        e2_1, e3_1 = cx["fg2a1"] * rates.league["fg2"], cx["fg3a1"] * rates.league["fg3"]
    x2 += e2
    x3 += e3
    xft += cx["fta"] * q
    xp1 += 2.0 * e2_1 + 3.0 * e3_1 + cx["fta1"] * q
    xc1 += (cx["fg2a1"] - e2_1) + (cx["fg3a1"] - e3_1) + cx["ftlast1"] * (1.0 - q)
    return pd.DataFrame({"x2pm": x2, "x3pm": x3, "xftm": xft, "xpts1": xp1, "xchance1": np.clip(xc1, 0.0, None)})


def align(y: np.ndarray, y_pts: np.ndarray, w: np.ndarray, season: np.ndarray, poss: np.ndarray,
          band=CALIB_BAND) -> tuple[np.ndarray, pd.DataFrame]:
    """One scalar per season so the target's total equals actual points: the LEVEL, nothing else.

    Not an affine map.  A row-level regression of points on the target has a slope of about 0.6
    (the target carries shot-mix variation that does not all predict), so an affine "alignment" is a
    partial shrink of the target in disguise, and one that fires in some seasons and not others.  The
    scalar is applied in every season (it is continuous, so applying it inside the band changes
    nothing that matters); `applied` records whether the aggregate was outside the band.  Never per
    lineup.  On a training block this is legal; the held-out season is never in one.
    """
    y = y.copy()
    rows = []
    for s in np.unique(season):
        m = season == s
        ratio = float((y[m] * poss[m]).sum() / max((y_pts[m] * poss[m]).sum(), 1e-9))
        y[m] = y[m] / ratio
        rows.append(dict(season=int(s), ratio=ratio, applied=not (band[0] <= ratio <= band[1]), a=0.0, b=1.0 / ratio))
    return y, pd.DataFrame(rows)


def gates(cnt: pd.DataFrame, xm: pd.DataFrame, y: np.ndarray, season: np.ndarray) -> pd.DataFrame:
    """Per season: expected against realised makes by kind, points, and the first-attempt pieces."""
    rows = []
    poss = cnt["poss"].to_numpy(dtype=float)
    for s in np.unique(season):
        m = season == s
        c, x = cnt[m], xm[m]
        rows.append(dict(season=int(s),
                         r2=float(x.x2pm.sum() / max((c.fgm - c.fg3m).sum(), 1.0)),
                         r3=float(x.x3pm.sum() / max(c.fg3m.sum(), 1.0)),
                         rft=float(x.xftm.sum() / max(c.ftm.sum(), 1.0)),
                         pts=float((y[m] * poss[m] / 100.0).sum() / max(c.pts.sum(), 1.0)),
                         pts1=float(x.xpts1.sum() / max(c.pts1.sum(), 1.0)),
                         chance1=float(x.xchance1.sum() / max(c.chance1.sum(), 1.0))))
    g = pd.DataFrame(rows)
    for col in ("r2", "r3", "rft", "pts1", "chance1"):
        g[f"{col}_ok"] = g[col].between(0.98, 1.02)
    g["pts_ok"] = g.pts.between(*CALIB_BAND)
    return g


def _check(wd, col: str = "fg2a_s1"):
    """The slot counters this target needs are on the design (a design built with a `counter_cols`
    whitelist carries only the ones its own targets asked for)."""
    if wd.counters is None or col not in wd.counters.columns:
        raise RuntimeError(f"the design carries no {col}; rebuild the stints (STINT_SCHEMA 4) or widen counter_cols")


def _rates_for(seasons, cfg, x):
    """Block rates (x None) or each season's own (x == 1), keyed by the row's season."""
    seasons = [int(s) for s in seasons]
    if x is None or x >= len(seasons):
        r = shooter_rates(seasons, cfg)
        return {s: r for s in seasons}
    if x == 1:
        return {s: shooter_rates([s], cfg) for s in seasons}
    raise ValueError("x must be None (the block) or 1 (each season's own); other pools are not built")


def _priced(wd, seasons, cfg, x, location):
    cnt, season = wd.counters, wd.rows["season"].to_numpy()
    rates = _rates_for(seasons, cfg, x)
    parts = []
    for s in np.unique(season):
        m = season == s
        parts.append(expected_makes(cnt[m], rates[int(s)], location=location).set_index(np.flatnonzero(m)))
    return pd.concat(parts).sort_index().reset_index(drop=True), cnt, season


def shooter_design(seasons, cfg, wd_pts, x=None, location=True, calibrate=True):
    """y = expected points with every make replaced by the shooter's expectation; realised rebounds kept."""
    _check(wd_pts)
    xm, cnt, season = _priced(wd_pts, seasons, cfg, x, location)
    poss = cnt["poss"].to_numpy(dtype=float)
    y = 100.0 * (2.0 * xm.x2pm + 3.0 * xm.x3pm + xm.xftm + cnt["xftm_tech"].to_numpy(dtype=float)) / poss
    y = y.to_numpy()
    g = gates(cnt, xm, y, season)
    cal = pd.DataFrame()
    if calibrate:
        y, cal = align(y, wd_pts.y, wd_pts.w, season, poss)
    return wd_pts.with_target(y), dict(target="xshoot" + ("" if location else "_flat"), x=x, gates=g, calibration=cal)


def continuation_design(seasons, cfg, wd_pts, x=None, location=True, r="league", calibrate=True):
    """y = the first attempt's expectation plus its expected continuation: shot luck AND rebound luck out.

    The continuation uses the validated geometric closure (xpts.league_constants) with the season's
    league offensive-rebound rate, or the lineup's fitted OREB% (r="lineup", cached in data/xpts/).
    """
    from .xpts import league_constants, lineup_rates
    from .windows import window_label
    _check(wd_pts)
    xm, cnt, season = _priced(wd_pts, seasons, cfg, x, location)
    poss = cnt["poss"].to_numpy(dtype=float)
    consts = league_constants(cnt, season)
    V = np.zeros(len(cnt))
    rr = np.zeros(len(cnt))
    for s, k in consts.items():
        m = season == s
        rr[m] = k.oreb / 100.0
    if r == "lineup":
        lab = window_label(sorted(int(s) for s in seasons))
        rr = lineup_rates(wd_pts, seasons, cfg, cache_dir=Path(cfg["_root"]) / "data" / "xpts", label=lab)["oreb"].to_numpy() / 100.0
    for s, k in consts.items():
        m = season == s
        V[m] = (1.0 - k.t2) * (k.x_fg + k.x_ft) / (1.0 - (1.0 - k.t2) * k.m_att * rr[m])
    y = 100.0 * (xm.xpts1.to_numpy() + xm.xchance1.to_numpy() * rr * V + cnt["xftm_tech"].to_numpy(dtype=float)) / poss
    g = gates(cnt, xm, y, season)
    cal = pd.DataFrame()
    if calibrate:
        y, cal = align(y, wd_pts.y, wd_pts.w, season, poss)
    return wd_pts.with_target(y), dict(target=f"xcont_{r}", x=x, gates=g, calibration=cal)


def expected_threes(seasons, cfg, wd_pts, prev: int = 0, k3: float = 450.0):
    """Each row's EXPECTED opponent three-point makes: every attempt at the shooter's padded other-half 3P%
    (x3def's repricing, the one piece of it the four-factor eFG% needs too).  Returns (x3, rates)."""
    cnt = wd_pts.counters
    seasons = sorted(int(s) for s in seasons)
    rates = shooter_rates(seasons, cfg, prev=prev, k_fixed={"fg3": k3})
    which = rates.for_rows("fg3", cnt["half"].to_numpy())
    n = len(cnt)
    x3 = cnt["fg3a_sx"].to_numpy(dtype=float) * rates.league["fg3"]
    masks = [(h, m) for h in ("A", "B", "RS") if (m := which == h).any()]     # once, not once per slot
    for s in SLOTS:
        pid = cnt[f"pid_s{s}"].to_numpy()
        p = np.full(n, rates.league["fg3"])
        for h, m in masks:
            p[m] = rates.take("p3", h, pid[m], rates.league["fg3"])
        x3 += cnt[f"fg3a_s{s}"].to_numpy(dtype=float) * p
    return x3, rates


def def_three_design(seasons, cfg, wd_pts, prev: int = 0, k3: float = 450.0, calibrate: bool = True):
    """The DEFENSIVE target: actual points with every opponent three-point make replaced by
    3 x the shooter's padded 3P%, free throws adjusted as shipped, everything else as it happened.

    Team opponent 3P% needs 4,000-7,000 attempts to stabilise (Nylon Calculus, 2018), so the
    defenders' effect on whether a three drops is almost entirely noise; this removes it from the
    variable their coefficients are fit to.  The shooter's rate is his other-half-of-block 3P%, plus
    the `prev` seasons before the block, padded toward the league with k = `k3` attempts (450: the
    split-half reliability of high-volume shooters, FINDINGS.md section 18).  No location curve.
    Meant to supply the DEFENSIVE coefficients only; offense comes from the free-throw target.
    """
    _check(wd_pts, "fg3a_s1")
    cnt, season = wd_pts.counters, wd_pts.rows["season"].to_numpy()
    x3, rates = expected_threes(seasons, cfg, wd_pts, prev=prev, k3=k3)
    poss = cnt["poss"].to_numpy(dtype=float)
    c = cnt
    pts_adj = (c["pts"] - 3.0 * c["fg3m"] + 3.0 * x3
               - c["ftm"] - c["ftm_tech"] + c["xftm"] + c["xftm_tech"]).to_numpy(dtype=float)
    y = 100.0 * pts_adj / poss
    rows = []
    for s in np.unique(season):
        m = season == s
        rows.append(dict(season=int(s), r2=1.0, r3=float(x3[m].sum() / max(c.fg3m[m].sum(), 1.0)),
                         rft=float(c.xftm[m].sum() / max(c.ftm[m].sum(), 1.0)),
                         pts=float((y[m] * poss[m] / 100.0).sum() / max(c.pts[m].sum(), 1.0)), pts1=1.0, chance1=1.0))
    g = pd.DataFrame(rows)
    for col in ("r2", "r3", "rft", "pts1", "chance1"):
        g[f"{col}_ok"] = g[col].between(0.98, 1.02)
    g["pts_ok"] = g.pts.between(*CALIB_BAND)
    cal = pd.DataFrame()
    if calibrate:
        y, cal = align(y, wd_pts.y, wd_pts.w, season, poss)
    return wd_pts.with_target(y), dict(target=f"x3def_p{prev}", x=None, gates=g, calibration=cal, k3=rates.k["fg3"])


def _named(fn, name, **kw):
    def target(seasons, cfg, wd_pts):
        return fn(seasons, cfg, wd_pts, **kw)
    target.__name__ = name
    return target


# the targets the holdout CLI registers as systems (scripts/45_holdout.py: hybrid_<name>, split_<name>)
TARGET_REGISTRY = {
    "xshoot": _named(shooter_design, "xshoot"),
    "xshoot_flat": _named(shooter_design, "xshoot_flat", location=False),
    "xshoot_x1": _named(shooter_design, "xshoot_x1", x=1),
    "xcont": _named(continuation_design, "xcont"),
    "xcont_lineup": _named(continuation_design, "xcont_lineup", r="lineup"),
}

# the counters each defensive target reads (designcache's `counter_cols`; the slot ids and `half` are
# always kept).  x3def: actual points with the opponents' threes repriced, free throws as shipped.
DEFENSE_TARGET_COLUMNS = {
    "x3def": {"pts", "fg3m", "ftm", "ftm_tech", "xftm", "xftm_tech", "poss", "fg3a_sx",
              *(f"fg3a_s{s}" for s in SLOTS)},
}
DEFENSE_TARGET_COLUMNS["x3def_p1"] = DEFENSE_TARGET_COLUMNS["x3def"]
DEFENSE_TARGET_COLUMNS["x3def_p2"] = DEFENSE_TARGET_COLUMNS["x3def"]

# targets meant for the DEFENSIVE coefficients only (scripts/45_holdout.py: def3_<name> = offense from
# hybrid_xft, defense from a fit on this target); pN = N seasons before the block added to the rate
DEFENSE_TARGETS = {
    "x3def": _named(def_three_design, "x3def"),
    "x3def_p1": _named(def_three_design, "x3def_p1", prev=1),
    "x3def_p2": _named(def_three_design, "x3def_p2", prev=2),
}
