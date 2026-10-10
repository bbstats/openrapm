"""Player-level inputs for the prior that are not box-score rates: who he is and where he has been.

Two blocks, both aligned to a window's players the way `roles.career_inputs` is:

  BIO      height (inches), weight (pounds), draft_pick (1..60; undrafted and unknown = 61) -- one number per
           player for his whole career, the median over the seasons the bio feed (data/raw/bio) lists him.
           None of it is a rate, so none of it needs padding, and the tree can learn that a block from a
           7-footer and a block from a 6-5 wing are different events without a play-by-play counter.
  TENURE   tenure (seasons with his current main team, counting this one, possession-weighted over the
           window's seasons) and n_teams (distinct teams he played for in the window).  What the turnover
           work (FINDINGS 24) measured for the target window, measured here for the FEATURE window: a box
           line beside people he knows against one beside strangers, without a one-hot of the team.

A held-out season is never a training season, but the roles table knows its rosters; `exclude_seasons`
skips those rows so the tenure chain steps over H as if H were unknown, and nothing measured on H reaches
a covariate (the standing rule from HANDOFF Part 2's traps).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BIO_INPUTS = ["height", "weight", "draft_pick"]
TENURE_INPUTS = ["tenure", "n_teams"]
PLAYER_INPUTS = [*BIO_INPUTS, *TENURE_INPUTS]
UNDRAFTED = 61.0

_CACHE: dict = {}


def player_bio(cfg, force: bool = False) -> pd.DataFrame:
    """One row per player: height, weight, draft_pick.  Cached at data/cache/bio.parquet."""
    root = Path(cfg["_root"])
    path = root / "data" / "cache" / "bio.parquet"
    if "bio" in _CACHE and not force:
        return _CACHE["bio"]
    if path.exists() and not force:
        _CACHE["bio"] = pd.read_parquet(path)
        return _CACHE["bio"]
    files = sorted((root / "data" / "raw" / "bio").glob("*_RS.parquet"))
    if not files:
        raise FileNotFoundError("data/raw/bio has no *_RS.parquet; run scripts/01_ingest.py")
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=["PLAYER_ID", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "DRAFT_NUMBER"])
        parts.append(pd.DataFrame({
            "player_id": d.PLAYER_ID.astype(np.int64),
            "height": pd.to_numeric(d.PLAYER_HEIGHT_INCHES, errors="coerce"),
            "weight": pd.to_numeric(d.PLAYER_WEIGHT, errors="coerce"),
            "draft_pick": pd.to_numeric(d.DRAFT_NUMBER, errors="coerce"),
        }))
    d = pd.concat(parts, ignore_index=True)
    d.loc[~d.draft_pick.between(1, 60), "draft_pick"] = np.nan
    g = d.groupby("player_id").agg(height=("height", "median"), weight=("weight", "median"),
                                   draft_pick=("draft_pick", "median")).reset_index()
    g["draft_pick"] = g.draft_pick.fillna(UNDRAFTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(path, index=False)
    _CACHE["bio"] = g
    return g


def bio_inputs(bio: pd.DataFrame, player_ids) -> pd.DataFrame:
    """BIO_INPUTS aligned to `player_ids`; a player the feed never listed takes the medians (and 61)."""
    out = bio.set_index("player_id").reindex(np.asarray(player_ids)).reset_index(drop=True)
    out["height"] = out.height.fillna(float(bio.height.median())).astype(float)
    out["weight"] = out.weight.fillna(float(bio.weight.median())).astype(float)
    out["draft_pick"] = out.draft_pick.fillna(UNDRAFTED).astype(float)
    return out[BIO_INPUTS]


def season_tenure(roles: pd.DataFrame, exclude_seasons=()) -> pd.DataFrame:
    """Per (player, season): his main team (most on-court possessions) and his tenure with it, counting
    this season and every consecutive earlier season with the same main team; seasons in `exclude_seasons`
    are dropped first, so the chain steps over them.  Only seasons he actually played (poss_on > 0)."""
    r = roles[(roles.poss_on > 0) & ~roles.season.isin(list(exclude_seasons))]
    main = (r.sort_values(["player_id", "season", "poss_on"], ascending=[True, True, False])
             .drop_duplicates(["player_id", "season"])[["player_id", "season", "team_id"]]
             .sort_values(["player_id", "season"]).reset_index(drop=True))
    same = (main.team_id == main.groupby("player_id").team_id.shift()).to_numpy()
    # a run of consecutive rows with the same team: cumulative count that resets when the team changes
    run = np.zeros(len(main), dtype=float)
    for i in range(len(main)):
        run[i] = run[i - 1] + 1.0 if i and same[i] else 1.0
    main["tenure"] = run
    return main


def tenure_inputs(roles: pd.DataFrame, seasons, player_ids, exclude_seasons=()) -> pd.DataFrame:
    """TENURE_INPUTS for the block `seasons`, aligned to `player_ids`: `tenure` possession-weighted over his
    seasons in the block, `n_teams` the distinct teams he played for in it.  A player the block never saw
    gets tenure 1 and one team."""
    seasons = [int(s) for s in seasons]
    ex = set(int(s) for s in exclude_seasons)
    t = season_tenure(roles, ex)
    r = roles[(roles.poss_on > 0) & roles.season.isin(seasons) & ~roles.season.isin(list(ex))]
    w = r.groupby(["player_id", "season"]).poss_on.sum().rename("w").reset_index()
    tw = t[t.season.isin(seasons)].merge(w, on=["player_id", "season"], how="inner")
    tw["_tw"] = tw.tenure * tw.w
    g = tw.groupby("player_id").agg(_tw=("_tw", "sum"), w=("w", "sum"))
    ten = (g._tw / g.w).rename("tenure")
    nt = r.groupby("player_id").team_id.nunique().rename("n_teams")
    out = pd.DataFrame({"player_id": np.asarray(player_ids)})
    out["tenure"] = out.player_id.map(ten).fillna(1.0).astype(float)
    out["n_teams"] = out.player_id.map(nt).fillna(1.0).astype(float)
    return out[TENURE_INPUTS]


def player_inputs(cfg, roles: pd.DataFrame, seasons, player_ids, exclude_seasons=()) -> pd.DataFrame:
    """The whole block, PLAYER_INPUTS, aligned to `player_ids`."""
    b = bio_inputs(player_bio(cfg), player_ids)
    t = tenure_inputs(roles, seasons, player_ids, exclude_seasons)
    return pd.concat([b, t], axis=1)
