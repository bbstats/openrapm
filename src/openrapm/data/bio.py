"""Player-level inputs for the prior that are not box-score rates: who he is and where he has been.

Two blocks, both one row per player in the order of the player ids asked for, as `roles.career_inputs` gives
them; the season panel (data/panel.py) asks for one season's players.

  BIO      height (inches), weight (pounds), draft_pick (1..60; undrafted and unknown = 61) -- one number per
           player for his whole career, the median over the seasons the bio feed (data/raw/bio) lists him.
           None of it is a rate, so none of it needs padding, and the tree can learn that a block from a
           7-footer and a block from a 6-5 wing are different events without a play-by-play counter.
  TENURE   tenure (seasons with his current main team, counting this one, possession-weighted over the
           seasons asked for) and n_teams (distinct teams he played for in those seasons): a box line beside
           teammates he has played with for years against one beside new teammates, without a one-hot of the
           team.  Why: a box line is worth less beside new teammates, by about 0.85 points per 100 on offence
           and 0.55 on defence when every teammate is new (FINDINGS.md section 24, `git show
           archive/research-2026-09:FINDINGS.md`).

`exclude_seasons` drops those seasons from the roles table first, so the tenure chain steps over them as if
they were unknown: a season a fit must not see reaches no input through a roster (notes/DECISIONS.md, "The
measurement traps": an input measured on the held-out season needs a control from data the outcome could not
have touched).  The season panel excludes none.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve

# As this module (and the bio.parquet cache) names them; the season panel calls them height_in, weight_lb,
# draft_pick, seasons_with_team and teams_this_season (features.BIO; data/legacy_names.py maps one onto the other).
BIO_INPUTS = ["height", "weight", "draft_pick"]
TENURE_INPUTS = ["tenure", "n_teams"]
PLAYER_INPUTS = [*BIO_INPUTS, *TENURE_INPUTS]
UNDRAFTED = 61.0
# config body_inputs: "career_median", one height and weight per player over every season the feed lists him (as
# shipped), or "season_listing", each season's own listing, so a player's other seasons never reach his body inputs
# (the single-season ruling; notes/DECISIONS.md, the bug-fix pass of 2026-10-22, experiment 96)
BODY_INPUTS = ("career_median", "season_listing")
# The NBA has listed barefoot heights since the 2019-20 season (season 2020).  The same 400 players' listings drop
# 0.575 inches from 2019 to 2020 and move under 0.03 in every other year, so the listings before it are put on the
# barefoot ruler by taking this off
BAREFOOT_FROM, SHOE_INCHES = 2020, 0.575


def player_bio(cfg, force: bool = False) -> pd.DataFrame:
    """One row per player: height, weight, draft_pick.  Stored at bio.parquet under paths.cache (data/cache), built
    there from the bio feed when it is missing or `force` is set, and read once per process."""
    path = resolve(cfg, "cache") / "bio.parquet"
    if force or not path.exists():
        table = bio_from_feed(resolve(cfg, "raw") / "bio")
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
        _stored_bio.cache_clear()           # the file just changed, so a copy read before it is out of date
    return _stored_bio(path)


@functools.lru_cache(maxsize=None)
def _stored_bio(path: Path) -> pd.DataFrame:
    """bio.parquet as stored at `path`, read once per process; keyed on the path, so a test that points paths.cache
    at a temporary tree never gets the real tree's table."""
    return pd.read_parquet(path)


def bio_from_feed(raw_bio: Path) -> pd.DataFrame:
    """The bio table from the bio feed's files (data/raw/bio/*_RS.parquet): each player's median height, weight and
    draft pick over the seasons the feed lists him, the undrafted and unknown at pick 61."""
    files = sorted(raw_bio.glob("*_RS.parquet"))
    if not files:
        raise FileNotFoundError(f"{raw_bio} has no *_RS.parquet; the roles table fetches them "
                                f"(data/roles.py, season_ages), so run `openrapm data panel`")
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
    return g


@functools.lru_cache(maxsize=None)
def _season_listings(raw_bio: Path) -> pd.DataFrame:
    """player_id, season, height, weight: every season's own listing in the bio feed, heights on the barefoot ruler
    (BAREFOOT_FROM, SHOE_INCHES); read once per process."""
    parts = []
    for path in sorted(Path(raw_bio).glob("*_RS.parquet")):
        d = pd.read_parquet(path, columns=["PLAYER_ID", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT"])
        season = int(path.name[:4])
        height = pd.to_numeric(d.PLAYER_HEIGHT_INCHES, errors="coerce")
        parts.append(pd.DataFrame({"player_id": d.PLAYER_ID.astype(np.int64), "season": season,
                                   "height": height - (SHOE_INCHES if season < BAREFOOT_FROM else 0.0),
                                   "weight": pd.to_numeric(d.PLAYER_WEIGHT, errors="coerce")}))
    return pd.concat(parts, ignore_index=True).drop_duplicates(["player_id", "season"])


def season_body(cfg, season: int, player_ids) -> pd.DataFrame:
    """height and weight aligned to `player_ids` from `season`'s own listing in the bio feed, heights on the barefoot
    ruler; a player the season's feed does not list takes that season's medians.

    Example: Giannis Antetokounmpo's 2014 row reads 81 - 0.575 = 80.4 in and 205 lb, his listing that season, where
    the career median reads 83 in and 242 lb."""
    listings = _season_listings(resolve(cfg, "raw") / "bio")
    rows = listings[listings.season == int(season)].set_index("player_id")
    out = rows.reindex(np.asarray(player_ids))[["height", "weight"]].reset_index(drop=True)
    for column in ("height", "weight"):
        out[column] = out[column].fillna(float(rows[column].median())).astype(float)
    return out


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
    """The whole block, PLAYER_INPUTS, aligned to `player_ids`.  With config body_inputs "season_listing" and one
    season asked for, height and weight are that season's own listing (`season_body`)."""
    b = bio_inputs(player_bio(cfg), player_ids)
    mode = cfg.get("body_inputs", "career_median")
    if mode not in BODY_INPUTS:
        raise ValueError(f"unknown body_inputs {mode!r}; use one of {BODY_INPUTS}")
    if mode == "season_listing":
        if len(list(seasons)) != 1:
            raise ValueError("body_inputs season_listing reads one season's listing; ask for one season")
        b[["height", "weight"]] = season_body(cfg, list(seasons)[0], player_ids).to_numpy()
    t = tenure_inputs(roles, seasons, player_ids, exclude_seasons)
    return pd.concat([b, t], axis=1)
