"""Careers before the data starts: each veteran's seasons before config.yaml's first season, from stats.nba.com.

The roles table (data/roles.py) starts in 1997, so a veteran's seasons played, career possessions and seasons with his
team were counted from 1997 (notes/DECISIONS.md, experiment 57): A.C. Green, in the league since 1985-86 with eleven
seasons and about 26,000 minutes behind him, read 0 seasons played before 1996-97.  This module pulls such players'
season totals (nba_api's PlayerCareerStats, regular season and playoffs, one row per season and team) and turns the
seasons before the first one into rows shaped like the roles table's, which the season panel adds to the history its
`_from_debut` career columns are counted from (features.CAREER_FROM_DEBUT, data/panel.py).

    who is pulled   every player the roles table first sees by `LAST_FIRST_SEEN` (2002) who was drafted at least two
                    years before the first season (1995 or earlier) -- a player drafted in 1996 or later started in
                    1996-97 or after, so he has nothing before -- and every undrafted player (or one the feed gives no
                    draft year) first seen in the first season, or later at `OLDER_AT_FIRST_SEEN` (24) or older
    games, minutes  regular season and playoffs added together, as the roles table pools them; a traded player's
                    season keeps one row per team (the endpoint's TOT rows, team 0, are dropped)
    possessions     minutes x the first season's on-court possessions per minute in the roles table (both ends: about
                    3.7 in 1997), one rate for every earlier season.  The league played faster in the late 1980s and
                    early 1990s, so an earlier season's count reads somewhat low; a count in thousands of possessions
                    barely notices
    age             the endpoint's PLAYER_AGE for that season

Example: A.C. Green's 1993-94 row is season 1994, Phoenix: his 82 regular-season games and 2,825 minutes plus his
playoff games and minutes, and possessions = those minutes x the 1997 rate.

Cached: one file per player at paths.raw/careers/<player_id>.parquet (both phases, the endpoint's columns), and the
rows built from them at paths.cache/careers_before.parquet.  With OPENRAPM_OFFLINE=1 a missing file stops the pull.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve
from .ingest import TIMEOUT, _retry, raw_dir, refuse_if_offline

# a player first seen this season or earlier may have played before the first season (he may have sat out 1996-97)
LAST_FIRST_SEEN = 2002
# an undrafted player first seen after the first season is pulled only from this age: a younger one is a rookie
OLDER_AT_FIRST_SEEN = 24.0
# the roles table's columns, in its order (data/roles.py, build_roles)
ROLE_COLUMNS = ["player_id", "season", "team_id", "games", "starts", "minutes", "poss_on", "team_poss", "age"]
CAREER_COLUMNS = ["PLAYER_ID", "SEASON_ID", "LEAGUE_ID", "TEAM_ID", "PLAYER_AGE", "GP", "GS", "MIN"]
PHASE_FRAMES = {"RS": 0, "PO": 2}           # PlayerCareerStats' SeasonTotalsRegularSeason and SeasonTotalsPostSeason


def career_path(cfg: dict, player_id: int) -> Path:
    return raw_dir(cfg) / "careers" / f"{int(player_id)}.parquet"


def cache_path(cfg: dict) -> Path:
    return resolve(cfg, "cache") / "careers_before.parquet"


def draft_years(raw_bio: Path) -> pd.Series:
    """{player_id: the draft year the bio feed gives him}, NaN for the undrafted ("Undrafted") and the unknown."""
    files = sorted(raw_bio.glob("*_RS.parquet"))
    feed = pd.concat([pd.read_parquet(path, columns=["PLAYER_ID", "DRAFT_YEAR"]) for path in files], ignore_index=True)
    year = pd.to_numeric(feed.DRAFT_YEAR, errors="coerce")
    return year.groupby(feed.PLAYER_ID.astype(np.int64)).min()


def players_to_pull(roles: pd.DataFrame, drafted: pd.Series, first_season: int) -> list:
    """The players who may have seasons before `first_season` (see the module docstring): first seen by
    `LAST_FIRST_SEEN`, and drafted by `first_season` - 2, or undrafted and first seen in `first_season` or at
    `OLDER_AT_FIRST_SEEN` or older.

    Example: A.C. Green (first seen 1997, drafted 1985) is pulled; Kobe Bryant (first seen 1997, drafted 1996) is not;
    an undrafted 22-year-old first seen in 1999 is not.
    """
    played = roles[roles.games > 0]               # a roster row without a game (placeholder ids among them)
    first = played.sort_values("season").groupby("player_id")[["season", "age"]].first()
    first = first[first.season <= LAST_FIRST_SEEN]
    year = drafted.reindex(first.index)
    undrafted = year.isna() & ((first.season == first_season) | (first.age >= OLDER_AT_FIRST_SEEN))
    return sorted(int(player) for player in first.index[((year <= first_season - 2) | undrafted).to_numpy()])


def fetch_career(player_id: int, cfg: dict, force: bool = False) -> pd.DataFrame:
    """One player's season totals, regular season and playoffs (a `phase` column), cached at `career_path`."""
    path = career_path(cfg, player_id)
    if path.exists() and not force:
        return pd.read_parquet(path)
    refuse_if_offline(path, f"the career of player {player_id}")
    from nba_api.stats.endpoints import playercareerstats

    def pull():
        try:
            frames = playercareerstats.PlayerCareerStats(player_id=int(player_id), per_mode36="Totals",
                                                         timeout=TIMEOUT).get_data_frames()
        except KeyError:                          # the endpoint answers an id it does not know without a result set
            return pd.DataFrame(columns=[*CAREER_COLUMNS, "phase"])
        return pd.concat([frames[index][CAREER_COLUMNS].assign(phase=phase)
                          for phase, index in PHASE_FRAMES.items()], ignore_index=True)
    career = _retry(pull, what=f"career {player_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    career.to_parquet(path, index=False)
    return career


def pull_careers(cfg: dict, player_ids, delay: float = 0.6, force: bool = False) -> int:
    """Every career in `player_ids` not cached yet, one request at a time, `delay` seconds apart: how many were
    fetched."""
    missing = [player for player in player_ids if force or not career_path(cfg, player).exists()]
    started = time.time()
    for done, player in enumerate(missing, start=1):
        fetch_career(player, cfg, force=force)
        time.sleep(delay)
        if done % 50 == 0 or done == len(missing):
            print(f"  careers: {done} of {len(missing)} fetched ({time.time() - started:.0f}s)", flush=True)
    return len(missing)


def possessions_per_minute(roles: pd.DataFrame, season: int) -> float:
    """On-court possessions (both ends) per minute played in `season`, over every row of the roles table."""
    rows = roles[roles.season == season]
    return float(rows.poss_on.sum() / rows.minutes.sum())


def rows_before(careers: pd.DataFrame, first_season: int, per_minute: float) -> pd.DataFrame:
    """The seasons before `first_season` as roles-table rows (`ROLE_COLUMNS`): one per player, season and team, the
    regular season and playoffs added together, possessions = minutes x `per_minute`.

    Seasons are named by the year they end ("1993-94" is 1994); only NBA rows (league "00") with a team count (the
    endpoint's TOT rows, team 0, sum a traded player's teams and are dropped).
    """
    rows = careers[(careers.LEAGUE_ID.astype(str) == "00") & (careers.TEAM_ID.astype(np.int64) != 0)].copy()
    rows["season"] = rows.SEASON_ID.astype(str).str[:4].astype(int) + 1
    rows = rows[rows.season < first_season]
    if rows.empty:
        return pd.DataFrame(columns=ROLE_COLUMNS)
    for column in ("GP", "GS", "MIN"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
    age = (rows[rows.phase == "RS"].groupby(["PLAYER_ID", "season"]).PLAYER_AGE.first()
           .rename("age").reset_index().rename(columns={"PLAYER_ID": "player_id"}))
    out = (rows.groupby(["PLAYER_ID", "season", "TEAM_ID"], as_index=False)
           .agg(games=("GP", "sum"), starts=("GS", "sum"), minutes=("MIN", "sum"))
           .rename(columns={"PLAYER_ID": "player_id", "TEAM_ID": "team_id"}))
    out = out.astype({"player_id": np.int64, "team_id": np.int64}).merge(age, on=["player_id", "season"], how="left")
    out["poss_on"] = out.minutes * per_minute
    out["team_poss"] = np.nan
    out["age"] = pd.to_numeric(out.age, errors="coerce")
    return out[ROLE_COLUMNS].sort_values(["player_id", "season", "team_id"]).reset_index(drop=True)


def build_careers_before(cfg: dict, roles: pd.DataFrame) -> pd.DataFrame:
    """The rows before the first season of every player `players_to_pull` names, from the cached careers; raises,
    naming the command, when one is missing."""
    first = int(cfg["first_season"])
    players = players_to_pull(roles, draft_years(raw_dir(cfg) / "bio"), first)
    missing = [player for player in players if not career_path(cfg, player).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(players)} careers are not cached under "
                                f"{raw_dir(cfg) / 'careers'}; run `openrapm data careers` first")
    careers = pd.concat([pd.read_parquet(career_path(cfg, player)) for player in players], ignore_index=True)
    return rows_before(careers, first, possessions_per_minute(roles, first))


def careers_before(cfg: dict, roles: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """`build_careers_before`, cached at `cache_path` (built there when missing or `force`)."""
    path = cache_path(cfg)
    if path.exists() and not force:
        return pd.read_parquet(path)
    rows = build_careers_before(cfg, roles)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)
    return rows
