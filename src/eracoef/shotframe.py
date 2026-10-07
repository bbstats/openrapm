"""One row per field-goal attempt, 1997-2026: what the play-by-play saw, and the possession around it.

The shot-quality build (the owner's request, 2026-10-06) needs every attempt as its own row, which the stints
do not keep: they sum attempts by possession and by lineup slot.  This module runs the stint parser with its
shot logger on (stints.GameParser, `log_shots=True`) and writes data/shotframe/<season>_<phase>.parquet.
The logger only appends, so the stints it parses alongside are the cached ones exactly; `season_frame(...,
check=True)` asserts that, and scripts/114_shot_frame.py runs it.

What a row carries (stints.GameParser._log_shot):
    the shot      x, y (xLegacy / yLegacy, tenths of a foot, the rim at the origin), dist (the feed's integer
                  feet), value, made, at_heave (the feed's own Heave rows, 2026 on), lp (the shipped curve)
    who           shooter, team, opp, home, the ten players on the floor (off1-5, def1-5), lineups_ok
    the possession  poss_no, poss_start (how it began: made_fg, made_ft, dreb, dreb_team, steal, dead_tov,
                  jump_ball, period_start, forced), poss_start_clock, att_no / fga_no (this attempt's place in
                  it), first, margin (the shooting team's lead before the shot), prev_event_clock, events
                  (the clock events since it began: offensive rebounds, fouls, violations, timeouts, jumps)
    start_prev_*  the shot that handed the possession over (value, distance, x, y, made, blocked, clock), for
                  possessions that began with a defensive rebound or after a make; shooter_secs_on, game
                  seconds since the shooter last came on
    post_*        assisted / blocked / and-one.  They exist only because of the result, so they are kept
                  for checks and are NEVER a model input (the owner's ruling, 2026-10-06).

Beside it, <season>_<phase>_aux.parquet: every turnover (`kind` tov, `sub` its type -- the shot-clock check
needs the shot-clock violations) and every free-throw trip (`kind` trip, `sub` and1 / fouled_fga / shooting /
bonus / retained and its length), with the same possession context.

No scorer shot-type tag (`subType`) is stored: the owner ruled none may ever be a model input, not even dunk
or putback.  A putback is read from the event order (`secs_since_oreb`), the rim from x/y.

`derive` adds the fields the models read; `rim_map` maps each arena-season's near-rim shot depth onto the
league's, from the visiting teams' coordinates alone (scorers place close shots 0.8-2.6 ft deep by arena).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import game_table, load_gamelog
from .stints import build_game, build_season

FRAME_VERSION = 5           # bumped whenever a column changes meaning; written into every file
META = ("season", "phase", "game_id", "game_date", "arena", "neutral", "half")


def frame_dir(cfg) -> Path:
    return Path(cfg["_root"]) / "data" / "shotframe"


def frame_path(season: int, phase: str, cfg, aux: bool = False) -> Path:
    return frame_dir(cfg) / f"{int(season)}_{phase}{'_aux' if aux else ''}.parquet"


def season_frame(season: int, phase: str, cfg, check: bool = False, verbose: bool = True):
    """Parse every game of a season with the logger on.  Returns (frame, report).

    With `check`, the stints parsed alongside are compared with the cached stints file, column by column
    over every column the parser writes, and a mismatch raises: the logger must not change a stint.
    """
    from .boxtable import box_from_gamelog, ft_padding, ft_totals
    from .shotcurve import curve_for
    gl = load_gamelog(season, phase, cfg)
    games = game_table(gl)
    sb = box_from_gamelog(gl)
    ft_rates = (ft_totals(sb), *ft_padding(sb))
    curve = curve_for(season, cfg)
    gt_rule = cfg.get("garbage_time", {})
    cached, _ = build_season(season, phase, cfg)          # the cache (raises if stale); never rebuilt here
    halves = cached.drop_duplicates("game_id").set_index("game_id")["half"] if len(cached) else pd.Series(dtype=object)
    neutral_seasons = set(cfg.get("neutral_site_seasons_po", []))
    parts, aux_parts, st_parts, errors = [], [], [], []
    t0 = time.time()
    for k, g in enumerate(games.itertuples(index=False)):
        try:
            st, _d, _nm, _shots, log, aux = build_game(g.game_id, g.home_team_id, g.away_team_id, cfg, gt_rule,
                                                       ft_rates, curve, log_shots=True)
        except Exception as e:  # noqa: BLE001  the same games fail here as in build_season
            errors.append((g.game_id, repr(e)[:200]))
            continue
        if check and len(st):
            st_parts.append(st.assign(game_id=g.game_id))
        for frame_, sink in ((log, parts), (aux, aux_parts)):
            if len(frame_):
                frame_["season"] = int(season)
                frame_["phase"] = phase
                frame_["game_id"] = g.game_id
                frame_["game_date"] = g.game_date
                frame_["arena"] = int(g.home_team_id)
                frame_["neutral"] = bool((phase == "PO" and season in neutral_seasons) or bool(getattr(g, "neutral", False)))
                frame_["half"] = halves.get(g.game_id, None)
                sink.append(frame_)
        if verbose and (k + 1) % 200 == 0:
            print(f"  {season} {phase}: {k + 1}/{len(games)} games  {time.time() - t0:.0f}s", flush=True)
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    aux = pd.concat(aux_parts, ignore_index=True) if aux_parts else pd.DataFrame()
    report = dict(season=int(season), phase=phase, games=len(games), games_failed=len(errors),
                  shots=len(frame), aux_rows=len(aux), seconds=round(time.time() - t0, 1),
                  first_error=errors[0] if errors else None)
    if check:
        report.update(_compare_stints(cached, st_parts))
    return frame, aux, report


def _compare_stints(cached: pd.DataFrame, st_parts: list) -> dict:
    """The parser's stints with the logger on against the cached file: every parser column, exactly."""
    mine = pd.concat(st_parts, ignore_index=True) if st_parts else pd.DataFrame()
    if len(mine) != len(cached):
        raise AssertionError(f"logger changed the stints: {len(mine)} rows against {len(cached)} cached")
    cols = [c for c in mine.columns if c in cached.columns]
    missing = [c for c in mine.columns if c not in cached.columns]
    if missing:
        raise AssertionError(f"parser columns absent from the cache: {missing[:5]}")
    worst, worst_col = 0.0, None
    for c in cols:
        a, b = mine[c].to_numpy(), cached[c].to_numpy()
        if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
            d = float(np.nanmax(np.abs(a.astype(float) - b.astype(float)))) if len(a) else 0.0
            same_nan = np.array_equal(np.isnan(a.astype(float)), np.isnan(b.astype(float)))
            if d > 0 or not same_nan:
                raise AssertionError(f"logger changed stint column {c}: max diff {d}")
            if d > worst:
                worst, worst_col = d, c
        elif not np.array_equal(a.astype(str), b.astype(str)):
            raise AssertionError(f"logger changed stint column {c}")
    return dict(stints_checked=len(mine), stint_columns_checked=len(cols), stint_max_diff=worst)


def write_season(season: int, phase: str, cfg, check: bool = False, verbose: bool = True) -> dict:
    frame, aux, report = season_frame(season, phase, cfg, check=check, verbose=verbose)
    path = frame_path(season, phase, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.assign(frame_version=FRAME_VERSION).to_parquet(path, index=False)
    aux.assign(frame_version=FRAME_VERSION).to_parquet(frame_path(season, phase, cfg, aux=True), index=False)
    report["path"] = str(path)
    return report


def load_frame(seasons, cfg, phases=("RS", "PO"), aux: bool = False) -> pd.DataFrame:
    """The shot frames (or with `aux` the turnover / trip logs) of these seasons and phases, as written."""
    parts = []
    for s in seasons:
        for ph in phases:
            p = frame_path(int(s), ph, cfg, aux=aux)
            if not p.exists():
                raise FileNotFoundError(f"{p} is missing; run scripts/114_shot_frame.py {s} {s} {ph}")
            f = pd.read_parquet(p)
            if "frame_version" not in f.columns or int(f["frame_version"].iloc[0]) != FRAME_VERSION:
                raise RuntimeError(f"{p} predates frame version {FRAME_VERSION}; rebuild it")
            parts.append(f)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------------------------------ derived fields
def _last_event(events: pd.Series, kinds: tuple) -> np.ndarray:
    """The clock of the latest event of these kinds in each row's possession (NaN if none).  Events are
    written in order, so the last match is the latest."""
    out = np.full(len(events), np.nan)
    for i, e in enumerate(events.to_numpy()):
        if not e:
            continue
        for tok in reversed(e.split(";")):
            k, _, c = tok.rpartition("@")
            if k in kinds:
                out[i] = float(c)
                break
    return out


NEAR_FT = 10.0              # twos inside this are distance-mapped per arena-season (`rim_map`)
N_QUANT = 101


def recentre(frame: pd.DataFrame, radius: float = 60.0, min_shots: int = 200) -> pd.DataFrame:
    """REJECTED (stage-1 gate, 2026-10-06): a translation of each arena-season's coordinates to its densest
    near-rim point.  The three-point arc sits at the same distance in every arena (5th percentile of arc threes
    24.75 ft, sd 0.23 across arenas, 2024-26), so the origin is not what differs; what differs is how deep the
    scorer places shots near the rim (median depth of twos inside 6 ft 0.8-2.6 ft by arena).  Kept so the record
    can be rerun; `rim_map` replaces it."""
    f = frame[(frame["value"] == 2) & frame["x"].notna() & ~((frame["x"] == 0) & (frame["y"] == 0))]
    f = f[(f["x"].abs() <= radius) & (f["y"].abs() <= radius)]
    rows = []
    for (arena, season), g in f.groupby(["arena", "season"], sort=True):
        if len(g) < min_shots:
            rows.append(dict(arena=arena, season=season, dx=0.0, dy=0.0, n=len(g)))
            continue
        x, y = g["x"].to_numpy(float), g["y"].to_numpy(float)
        edges = np.arange(-radius, radius + 5.0, 5.0)
        h, ex, ey = np.histogram2d(x, y, bins=[edges, edges])
        hs = h[:-1, :-1] + h[1:, :-1] + h[:-1, 1:] + h[1:, 1:]
        ix, iy = np.unravel_index(int(np.argmax(hs)), hs.shape)
        cx, cy = ex[ix + 1], ey[iy + 1]
        near = (np.abs(x - cx) <= 10.0) & (np.abs(y - cy) <= 10.0)
        rows.append(dict(arena=arena, season=season, dx=float(x[near].mean()), dy=float(y[near].mean()),
                         n=int(near.sum())))
    return pd.DataFrame(rows)


def rim_map(frame: pd.DataFrame, near_ft: float = NEAR_FT, min_shots: int = 300) -> pd.DataFrame:
    """Per arena-season, the distance quantiles of the VISITING teams' twos inside `near_ft` (those at (0, 0) at
    depth 0), beside the league's for that season: the map that `derive` applies to every located two inside `near_ft` in that arena.

    Visitors are 29 different offences, so their shot depth in any arena should look like the league's; what is
    left is the scorer's habit (the research's followup_1: the 0-3 ft share's arena sd is 14 points before 2011 and
    8 points from 2021, against 4-6 for the shooting teams).  Tag-free: coordinates only.  Returns arena, season,
    q (0..1), arena_ft, league_ft; an arena-season with fewer than `min_shots` such attempts is left unmapped."""
    # The (0, 0) twos are in, at depth 0: before 2011 an arena that logged its rim shots there leaves only the
    # non-rim ones among its located twos, and a map of located twos alone pulled those toward the rim (and pushed
    # the rim shots of fully located arenas out): +0.84 between an arena's (0, 0) share and its overpricing, 2005.
    d = np.hypot(frame["x"].to_numpy(float), frame["y"].to_numpy(float)) / 10.0
    loc = frame["x"].notna().to_numpy()
    keep = (frame["value"] == 2).to_numpy() & loc & (d < near_ft) & (frame["team"] != frame["arena"]).to_numpy()
    f = frame.loc[keep, ["arena", "season"]].assign(d=d[keep])
    qs = np.linspace(0.0, 1.0, N_QUANT)
    rows = []
    for season, g in f.groupby("season"):
        lg = np.quantile(g["d"].to_numpy(), qs)
        for arena, h in g.groupby("arena"):
            if len(h) < min_shots:
                continue
            rows.append(pd.DataFrame(dict(arena=arena, season=season, q=qs,
                                          arena_ft=np.quantile(h["d"].to_numpy(), qs), league_ft=lg)))
    if not rows:
        return pd.DataFrame(columns=["arena", "season", "q", "arena_ft", "league_ft"])
    return pd.concat(rows, ignore_index=True)


def _apply_rim_map(f: pd.DataFrame, x: np.ndarray, y: np.ndarray, rmap: pd.DataFrame, near_ft: float = NEAR_FT):
    """Scale each mapped attempt's (x, y) along its ray from the rim so its distance becomes the league quantile."""
    d = np.hypot(x, y) / 10.0
    two = (f["value"].to_numpy() == 2) & ~f["noloc"].to_numpy() & (d < near_ft)
    xs, ys = x.copy(), y.copy()
    arena, season = f["arena"].to_numpy(), f["season"].to_numpy()
    for (a_, s_), m in rmap.groupby(["arena", "season"]):
        sel = two & (arena == a_) & (season == s_)
        if not sel.any():
            continue
        q = m["q"].to_numpy()
        a = np.maximum.accumulate(m["arena_ft"].to_numpy())
        u = np.interp(d[sel], a, q)
        new = np.interp(u, q, m["league_ft"].to_numpy())
        dd = d[sel]
        scale = np.where(dd > 0, new / np.maximum(dd, 1e-9), 1.0)
        xs[sel] = x[sel] * scale
        ys[sel] = np.where(dd > 0, y[sel] * scale, new * 10.0)
    return xs, ys


def derive(frame: pd.DataFrame, offsets: pd.DataFrame | None = None, rmap: pd.DataFrame | None = None) -> pd.DataFrame:
    """The fields the shot models read, from the logged ones.  Pure: no disk, no network.  `rmap` (rim_map) maps
    near-rim depth per arena-season; `offsets` (recentre, rejected) is kept only to rerun the record."""
    f = frame.copy()
    f["secs_into_poss"] = np.clip(f["poss_start_clock"] - f["clock"], 0.0, None)
    f["secs_since_prev"] = np.clip(f["prev_event_clock"] - f["clock"], 0.0, None)
    oreb = _last_event(f["events"], ("oreb", "oreb_team"))
    f["secs_since_oreb"] = np.clip(oreb - f["clock"].to_numpy(), 0.0, None)
    to = _last_event(f["events"], ("timeout",))
    f["secs_since_timeout"] = np.clip(to - f["clock"].to_numpy(), 0.0, None)
    f["n_oreb"] = f["events"].fillna("").str.count(r"(?:^|;)oreb")
    # coordinates: exact (0, 0) is its own cell (before 2011 mostly a shot the arena did not locate)
    f["noloc"] = (f["x"] == 0) & (f["y"] == 0)
    x, y = f["x"].to_numpy(float), f["y"].to_numpy(float)
    if offsets is not None and len(offsets):
        o = f[["arena", "season"]].merge(offsets[["arena", "season", "dx", "dy"]], on=["arena", "season"], how="left")
        dx, dy = o["dx"].fillna(0.0).to_numpy(), o["dy"].fillna(0.0).to_numpy()
        keep = ~f["noloc"].to_numpy()
        x = np.where(keep, x - dx, x)
        y = np.where(keep, y - dy, y)
    if rmap is not None and len(rmap):
        x, y = _apply_rim_map(f, x, y, rmap)
    f["xc"], f["yc"] = x, y
    f["dist_xy"] = np.hypot(x, y) / 10.0
    f["angle"] = np.degrees(np.arctan2(np.abs(x), y))          # 0 straight on, 90 along the baseline
    f["corner3"] = (f["value"] == 3) & (np.abs(x) >= 220) & (y <= 92.5)
    f["heave"] = (f["dist_xy"] >= 36.0) & (f["clock"] <= 2.0)
    return f


ZONE_FT = 10.0              # inside this the logged spot knows the result: coded as two zones (DECISIONS.md, the location leak)
ZONE_EDGE = 6.0
LOGGED = ("dist_xy", "xc", "yc", "angle")


def code_location(f: pd.DataFrame) -> pd.DataFrame:
    """The spot as the shot models may read it.  At the same true (tracking) distance the scorer logs a made close
    shot 0.7-1.3 ft nearer the rim than a missed one, so the exact logged spot inside ZONE_FT carries the result
    (DECISIONS.md, "The location leak").  A located attempt inside ZONE_FT gets its zone's distance -- 3 ft (0-6) or
    8 ft (6-10) -- straight on (x 0, y the zone, angle 0); unlocated attempts (0, 0) and everything beyond are
    unchanged.  The exact logged values are kept as `<col>_logged`, which shotfeatures.FORBIDDEN bans as inputs."""
    g = f.copy()
    for c in LOGGED:
        g[f"{c}_logged"] = g[c].to_numpy(float)
    d = g["dist_xy"].to_numpy(float)
    near = (d < ZONE_FT) & ~g["noloc"].to_numpy(bool)
    zone = np.where(d < ZONE_EDGE, ZONE_EDGE / 2.0, (ZONE_EDGE + ZONE_FT) / 2.0)
    g["dist_xy"] = np.where(near, zone, d)
    g["xc"] = np.where(near, 0.0, g["xc"].to_numpy(float))
    g["yc"] = np.where(near, 10.0 * zone, g["yc"].to_numpy(float))
    g["angle"] = np.where(near, 0.0, g["angle"].to_numpy(float))
    return g


def anchored(f: pd.DataFrame) -> pd.DataFrame:
    """The frame with every timing field measured from the shot's ANCHOR (the last row before it that cannot belong
    to it: stints.GameParser._log_event), never from its own timestamp.

    The timing leak (DECISIONS.md): the feed stamps makes and misses with different delays, so on the 2015-16
    tracked attempts timing measured from the stamp beat the same timing measured from the true release by 1.59 log
    loss per 1000 attempts (z -6.6): the result, leaking.  Anchored timing never reads the stamp.  The game clock of
    the anchor stands in for the shot's; the shot clock is rebuilt at the anchor (shotclock.rebuild with t = the
    anchor, so later events never count); seconds since the offensive rebound / timeout run to the anchor; time on
    court is the shooter's at the anchor."""
    g = f.copy()
    a = g["anchor_clock"].to_numpy(float)
    shot = g["clock"].to_numpy(float)
    g["clock"] = a
    g["secs_into_poss"] = np.clip(g["poss_start_clock"].to_numpy(float) - a, 0.0, None)
    g["secs_since_oreb"] = np.clip(g["secs_since_oreb"].to_numpy(float) - (a - shot), 0.0, None)
    g["secs_since_timeout"] = np.clip(g["secs_since_timeout"].to_numpy(float) - (a - shot), 0.0, None)
    g["shooter_secs_on"] = np.clip(g["shooter_secs_on"].to_numpy(float) - (a - shot), 0.0, None)
    return g


def player_game_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """FGA, FGM, 3PA, 3PM per shooter-game: the reconciliation against data/stints/*_shots.parquet."""
    f = frame.assign(fg3a=(frame["value"] == 3).astype(int), fg3m=((frame["value"] == 3) & (frame["made"] == 1)).astype(int))
    return f.groupby(["game_id", "shooter"]).agg(fga=("made", "size"), fgm=("made", "sum"), fg3a=("fg3a", "sum"),
                                                 fg3m=("fg3m", "sum")).reset_index()


__all__ = ["FRAME_VERSION", "META", "derive", "frame_path", "load_frame", "player_game_totals",
           "recentre", "rim_map", "season_frame", "write_season"]
