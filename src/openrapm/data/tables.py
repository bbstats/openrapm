"""Tracked and on-disk tables the ratings are joined with.

`player_names` reads params/player_names.csv: a frozen copy of the names the published ratings were built
with (the retired scripts/62's `_names()` at the tag archive/research-2026-10, from
artifacts/season_ratings.parquet), because no data source reproduces them
(gamelog, bio and stint names disagree on 16-22 players).  Two players carry no name there and one rated
player is missing from it, which leaves five rows of the published table without a name until the owner
fills them.

`team_of_rows` names the two teams of every team-game row from the per-game stint diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import resolve, resolve_path

PLAYER_NAMES = "params/player_names.csv"


def player_names(cfg: dict) -> pd.DataFrame:
    """(player_id, player_name), one row per player; an empty name reads back as missing, nothing else does."""
    return pd.read_csv(resolve_path(cfg, PLAYER_NAMES), dtype={"player_id": "int64", "player_name": "str"},
                       keep_default_na=False, na_values=[""], encoding="utf-8")


def team_of_rows(keys: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """(team_off, team_def) per row of `keys`: the team whose points the row counts, and the team defending.

    `keys` has season, phase, game_id and is_home_off per row.  The stints carry no team id (home and away are
    implied by the slot columns), so the ids come from the diagnostic frame written beside each season's
    stints, one row per game.
    """
    stints = resolve(cfg, "stints")
    frames = []
    for (season, phase), _ in keys.groupby(["season", "phase"], sort=True):
        path = stints / f"{season}_{phase}_diag.parquet"
        frames.append(pd.read_parquet(path, columns=["game_id", "home_team_id", "away_team_id"]))
    diag = pd.concat(frames, ignore_index=True).drop_duplicates("game_id")
    diag["game_id"] = diag.game_id.astype(str)
    merged = keys.assign(game_id=keys.game_id.astype(str)).merge(diag, on="game_id", how="left")
    if merged.home_team_id.isna().any():
        raise SystemExit("a team-game has no row in the stint diagnostics; rebuild the stints")
    home, away = merged.home_team_id.to_numpy(np.int64), merged.away_team_id.to_numpy(np.int64)
    is_home_off = merged.is_home_off.to_numpy(bool)
    return pd.DataFrame({"team_off": np.where(is_home_off, home, away),
                         "team_def": np.where(is_home_off, away, home)})
