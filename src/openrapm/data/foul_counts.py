"""Five more counts read straight from the play-by-play, per game and player (experiment 119, the owner, 2026-10-28):
the fouls and goaltends called on him, his shots that were blocked, and the three-shot trips he earns.

    goaltend           a Violation row "Defensive Goaltending", the defender named: 1.1 a game in 1997, 0.5 in 2025.
                       Before 2002 most of those rows name only the team, so the season panel reads the goaltends
                       from 2002 only (features.GOALTENDS_FROM); the cache keeps the named ones of every season
    shooting foul      a Foul row "Shooting", or "Shooting Block", the blocking foul on a shooter some seasons log
                       apart (0.7 a game in 2014)
    loose-ball foul    a Foul row "Loose Ball"
    shot blocked       a Missed Shot row with a BLOCK row right after it, credited to the shooter; the blocker's own
                       count is the box score's blocks (data/event_counts.py reads the same pairs)
    three-shot trip    a Free Throw row "Free Throw 1 of 3": fouled shooting a three.  Flagrant and clear-path trips
                       carry subTypes of their own and are not counted

These are the referee's calls and the block, not the scorer's shot descriptions, which stay out of every input (the
owner, 2026-10-06).  The feed names the fouler and, in recent seasons, the referee, never the player fouled, so the
three-shot trips are the only fouls drawn read here.  `openrapm data events` caches each season beside the event
counts, as {season}_fouls.parquet; the season panel pads them per 100 possessions (data/panel.py,
features.FOUL_RATES).

Example: the row "Adebayo Violation:Defensive Goaltending (R.Hussey)", from a 2025 game, gives Adebayo one goaltend.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..features import FOUL_RATES
from .config import resolve
from .event_counts import KEYS, PBP_COLUMNS, _flagged, read_game
from .ingest import GAME_PREFIX, raw_dir

__all__ = ["FOUL_COUNTS", "game_foul_counts", "season_foul_counts", "foul_counts_path", "build_foul_counts",
           "load_foul_counts"]

# the per-game counts, in features.FOUL_RATES order: "goaltends" is padded into "goaltends_per100"
FOUL_COUNTS = [rate.removesuffix("_per100") for rate in FOUL_RATES]
GOALTENDS, SHOOTING_FOULS, LOOSE_BALL_FOULS, SHOTS_BLOCKED, THREE_SHOT_TRIPS = FOUL_COUNTS
# {count: (actionType, the subTypes that count)}, every one on a row that names a player
CALLS = {GOALTENDS: ("Violation", ("Defensive Goaltending",)),
         SHOOTING_FOULS: ("Foul", ("Shooting", "Shooting Block")),
         LOOSE_BALL_FOULS: ("Foul", ("Loose Ball",)),
         THREE_SHOT_TRIPS: ("Free Throw", ("Free Throw 1 of 3",))}


def game_foul_counts(pbp: pd.DataFrame) -> pd.DataFrame:
    """One game's play-by-play, rows in the file's order -> one row per player with any of the counts: player_id,
    then `FOUL_COUNTS`."""
    game = read_game(pbp)
    rows = {name: np.flatnonzero((game.action == action) & np.isin(game.kind, kinds) & game.player)
            for name, (action, kinds) in CALLS.items()}
    blocks = np.flatnonzero(_flagged(game, "BLOCK"))
    shot = np.maximum(blocks - 1, 0)
    rows[SHOTS_BLOCKED] = shot[(blocks > 0) & (game.action[shot] == "Missed Shot") & game.player[shot]]
    out = pd.DataFrame({name: pd.Series(np.ones(len(at)), index=game.person[at]).groupby(level=0).sum()
                        for name, at in rows.items()})
    out = out.reindex(columns=FOUL_COUNTS).fillna(0.0)
    return out.rename_axis("player_id").reset_index()


def season_foul_counts(season: int, cfg: dict) -> pd.DataFrame:
    """Every game of the season, regular season then playoffs, from data/raw/pbp/{season}: `KEYS` then
    `FOUL_COUNTS`, one row per game, phase and player with any count."""
    folder = raw_dir(cfg) / "pbp" / str(season)
    parts = []
    for phase, prefix in GAME_PREFIX.items():
        for path in sorted(folder.glob(f"{prefix}*.parquet")):
            counts = game_foul_counts(pd.read_parquet(path, columns=PBP_COLUMNS))
            parts.append(counts.assign(game_id=path.stem, season=int(season), phase=phase))
    if not parts:
        raise FileNotFoundError(f"no play-by-play for {season} in {folder}; run `openrapm data ingest {season}`")
    return pd.concat(parts, ignore_index=True)[[*KEYS, *FOUL_COUNTS]]


def foul_counts_path(season: int, cfg: dict):
    return resolve(cfg, "event_counts") / f"{int(season)}_fouls.parquet"


def build_foul_counts(cfg: dict, seasons, force: bool = False) -> None:
    """Count and cache every season asked for (a cached season is read, unless `force`); one line per season with
    each count per game."""
    for season in seasons:
        started, path = time.time(), foul_counts_path(season, cfg)
        if force or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            season_foul_counts(season, cfg).to_parquet(path, index=False)
        counts = pd.read_parquet(path)
        per_game = counts[FOUL_COUNTS].sum() / max(counts.game_id.nunique(), 1)
        read = ", ".join(f"{name} {value:.2f}" for name, value in per_game.items())
        print(f"  {season}: per game, {read} ({time.time() - started:.0f}s)", flush=True)


def load_foul_counts(season: int, cfg: dict) -> pd.DataFrame:
    """The season's cached counts (`KEYS` + `FOUL_COUNTS`); refused with the command to run when not built."""
    path = foul_counts_path(season, cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run `openrapm data events {int(season)}` (the fouls, "
                                f"goaltends and blocked shots, data/foul_counts.py)")
    counts = pd.read_parquet(path)
    return counts.assign(game_id=counts.game_id.astype(str))
