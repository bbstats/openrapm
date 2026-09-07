"""Teammate turnover: how much of a player's floor time is spent with people he has not played beside before.

A trade is the extreme case -- nearly every possession is with strangers -- but the same quantity is a
continuous number for everyone: a stayer on a rebuilt roster can be at 0.7, a stayer on a settled core at
0.2.  Everything in the trade-calibration work is expressed in it, so that "rating if traded" means "rating
at turnover 1.0" and never depends on a binary rule about main teams.

Definitions (regular season only, both ends of the floor):

  shared(p, t, s)     possessions player p and teammate t were on the floor together for in season s
  familiar(p, a -> b) the share of p's teammate-possessions in season b that were with a teammate he had
                      shared at least `min_shared` possessions with in season a (or in ANY season of a block
                      when a is a block):   sum_{t in T_a} shared(p, t, b) / sum_t shared(p, t, b)
  turnover(p, a -> b) 1 - familiar(p, a -> b)

`season_pairs` builds and caches the per-season teammate table (data/cache/teammates.parquet); the two
public functions compute turnover between any two sets of seasons from it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve
from .design import AWAY_SLOTS, HOME_SLOTS

MIN_SHARED = 100.0        # a teammate counts as familiar after this many shared possessions in the reference span


def teammates_from_stints(stints: pd.DataFrame) -> pd.DataFrame:
    """Per ordered (player_id, teammate_id): shared possessions (both ends) over these stints.

    `stints`: h1..h5, a1..a5, poss_h, poss_a.  Every stint contributes 20 ordered home pairs and 20 away
    pairs, weighted by the stint's possessions at both ends.
    """
    both = stints["poss_h"].to_numpy(dtype=float) + stints["poss_a"].to_numpy(dtype=float)
    keep = both > 0
    both = both[keep]
    parts = []
    for slots in (HOME_SLOTS, AWAY_SLOTS):
        S = stints.loc[keep, list(slots)].to_numpy()
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                parts.append(pd.DataFrame({"player_id": S[:, i], "teammate_id": S[:, j], "shared": both}))
    if not parts:
        return pd.DataFrame(columns=["player_id", "teammate_id", "shared"])
    d = pd.concat(parts, ignore_index=True)
    return (d.groupby(["player_id", "teammate_id"], as_index=False)["shared"].sum()
             .astype({"player_id": np.int64, "teammate_id": np.int64}))


def teammates_path(cfg) -> Path:
    root = Path(cfg["_root"]) if "_root" in cfg else Path(".")
    return root / "data" / "cache" / "teammates.parquet"


def build_teammates(cfg, seasons=None, force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Every season's teammate table -> data/cache/teammates.parquet: season, player_id, teammate_id, shared."""
    path = teammates_path(cfg)
    if path.exists() and not force:
        return pd.read_parquet(path)
    seasons = list(range(int(cfg["first_season"]), int(cfg["last_season"]) + 1)) if seasons is None else list(seasons)
    parts = []
    for s in seasons:
        st = pd.read_parquet(Path(resolve(cfg, "stints")) / f"{s}_RS.parquet",
                             columns=[*HOME_SLOTS, *AWAY_SLOTS, "poss_h", "poss_a"])
        t = teammates_from_stints(st)
        t.insert(0, "season", int(s))
        parts.append(t)
        if verbose:
            print(f"  teammates {s}: {t.player_id.nunique()} players, {len(t)} pairs", flush=True)
    out = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def cached_table(ctx) -> pd.DataFrame | None:
    """The teammate table read once per holdout Context (attribute `_teammates_tbl`); None if not built."""
    tbl = getattr(ctx, "_teammates_tbl", None)
    if tbl is None:
        p = teammates_path(ctx.cfg)
        tbl = pd.read_parquet(p) if p.exists() else False
        ctx._teammates_tbl = tbl
    return None if tbl is False else tbl


def familiar_share(tm: pd.DataFrame, ref_seasons, new_seasons, min_shared: float = MIN_SHARED,
                   player_ids=None) -> pd.DataFrame:
    """Per player: the share of his teammate-possessions over `new_seasons` spent with teammates he had shared
    at least `min_shared` possessions with over `ref_seasons`.

    Returns player_id, poss_new (his teammate-possessions in the new span, i.e. 4 x his on-floor possessions),
    familiar (the share), turnover (1 - familiar).  A player with no possessions in the new span is absent
    unless `player_ids` is given, in which case he gets NaN.
    """
    ref = tm[tm.season.isin(list(ref_seasons))].groupby(["player_id", "teammate_id"], as_index=False)["shared"].sum()
    ref = ref[ref.shared >= float(min_shared)][["player_id", "teammate_id"]].assign(known=1.0)
    new = tm[tm.season.isin(list(new_seasons))].groupby(["player_id", "teammate_id"], as_index=False)["shared"].sum()
    m = new.merge(ref, on=["player_id", "teammate_id"], how="left")
    m["known_shared"] = m.shared * m.known.fillna(0.0)
    g = m.groupby("player_id", as_index=False).agg(poss_new=("shared", "sum"), known_shared=("known_shared", "sum"))
    if len(g) == 0:
        g = pd.DataFrame(columns=["player_id", "poss_new", "known_shared"])
    g["familiar"] = np.where(g.poss_new > 0, g.known_shared / np.where(g.poss_new > 0, g.poss_new, 1.0), np.nan)
    g = g.drop(columns="known_shared")
    g["turnover"] = 1.0 - g["familiar"]
    if player_ids is not None:
        g = pd.DataFrame({"player_id": np.asarray(player_ids, dtype=np.int64)}).merge(g, on="player_id", how="left")
    return g


def season_turnover(tm: pd.DataFrame, min_shared: float = MIN_SHARED) -> pd.DataFrame:
    """Every consecutive-season pair: player_id, season (the NEW one), poss_new, familiar, turnover, for players
    on the floor in both `season - 1` and `season`."""
    seasons = sorted(tm.season.unique())
    parts = []
    for a, b in zip(seasons[:-1], seasons[1:]):
        if b != a + 1:
            continue
        f = familiar_share(tm, [a], [b], min_shared)
        seen = set(tm.loc[tm.season == a, "player_id"].unique())
        f = f[f.player_id.isin(seen)]
        parts.append(f.assign(season=int(b)))
    return pd.concat(parts, ignore_index=True)[["player_id", "season", "poss_new", "familiar", "turnover"]]


def window_pair_turnover(tm: pd.DataFrame, windows: list, min_shared: float = MIN_SHARED) -> pd.DataFrame:
    """Every ORDERED pair of windows (w, w'), w != w': per player on the floor in both, the turnover of w' with
    respect to w.  `windows`: list of (label, [seasons]).  Columns: player_id, window (w), window_to (w'),
    poss_new, familiar, turnover."""
    parts = []
    for lab_a, sa in windows:
        seen = set(tm.loc[tm.season.isin(list(sa)), "player_id"].unique())
        for lab_b, sb in windows:
            if lab_a == lab_b:
                continue
            f = familiar_share(tm, sa, sb, min_shared)
            f = f[f.player_id.isin(seen)]
            parts.append(f.assign(window=lab_a, window_to=lab_b))
    return pd.concat(parts, ignore_index=True)[["player_id", "window", "window_to", "poss_new", "familiar", "turnover"]]
