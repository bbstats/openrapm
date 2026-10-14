"""Tracked and on-disk tables the ratings are joined with.

`player_names` reads params/player_names.csv: a frozen copy of the names the published ratings used, made on
2026-10-11 from the older system's ratings table (artifacts/season_ratings.parquet, removed then and recoverable
with `git show archive/research-2026-10:artifacts/season_ratings.parquet`), because no data source reproduces
them (gamelog, bio and stint names disagree on 16-22 players).  Two players carry no name there and one rated
player is missing from it, which leaves five rows of the published table without a name until the owner
fills them.

`team_of_rows` names the two teams of every team-game row from the per-game stint diagnostics.

`load_ratings` reads a ratings table as player_id, season, o, d in raw sign, the form the swap test, step 5 and
the year-over-year test compare lineups in.

`load_consensus` reads data/external/consensus.csv, the consensus of modern all-in-one metrics the ratings are
checked against (openrapm.consensus) and never fitted to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..swap_test import ratings_in_raw_sign
from .config import resolve, resolve_path

PLAYER_NAMES = "params/player_names.csv"
CONSENSUS_FILE = "data/external/consensus.csv"


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


def design_row_teams(design, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(team_off, team_def): each row of a design, the team on offence and the team on defence (`team_of_rows`, the
    row's game id read off the design's game table)."""
    game_ids = design.games.set_index("game_idx").game_id.reindex(design.rows.game_idx.to_numpy()).to_numpy()
    keys = pd.DataFrame({"season": design.rows.season.to_numpy(), "phase": design.rows.phase.to_numpy(),
                         "game_id": game_ids, "is_home_off": design.rows.is_home_off.to_numpy()})
    teams = team_of_rows(keys, cfg)
    return teams.team_off.to_numpy(), teams.team_def.to_numpy()


def load_ratings(path, columns: str = "rating", possessions: bool = False) -> pd.DataFrame:
    """player_id, season, o, d in raw sign (d = points allowed) from a ratings table stored positive-good.

    `columns` "rating" reads rating_off / rating_def, "prior" reads prior_off / prior_def
    (`swap_test.ratings_in_raw_sign`).  `possessions` adds `poss`, the table's offensive possessions
    (`poss_off`), which a fill (scoring.replacement_level) and the year-over-year test's coverage read.

    Example: a row with rating_off +2.0, rating_def +1.5 and poss_off 1,800 reads o = +2.0, d = -1.5 and, with
    `possessions`, poss = 1,800.
    """
    table = pd.read_parquet(path)
    ratings = ratings_in_raw_sign(table, columns)
    if not possessions:
        return ratings
    if "poss_off" not in table.columns:
        raise ValueError(f"{path}: the ratings table lacks poss_off")
    return ratings.assign(poss=table.poss_off.astype(float))


def load_consensus(cfg: dict) -> pd.DataFrame:
    """data/external/consensus.csv as it is: one row per player of 2024-26, the consensus's ratings by name."""
    return pd.read_csv(resolve_path(cfg, CONSENSUS_FILE))
