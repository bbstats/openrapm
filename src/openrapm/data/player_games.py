"""Each player's games, one row per player and game he was on the court for: the table the multi-stat network reads
(game_rates.py), and that a within-season build cuts to its fit games.

Words used below
    game row     one player in one game he played possessions in (a box-score line without a possession, 25 in the
                 2026 regular season with 13 counts between them, is left out: there is nothing to divide by)
    piece        a season and phase of stints made ready for the design (data/designs.py): its per-game possession
                 and box-count tables are this table's source, so its rows are the rows every rating reads

Columns, and where each comes from:
    season, phase, game_id, game_date   the pieces' game table
    game_idx                            the game's number in the season's design: the regular season first, then
                                        the playoffs, each in date order (design.order_games)
    player_id, poss_off, poss_def       the pieces' per-game possessions, from the stints: his team's possessions with
                                        the ball and without it while he was on the court
    the 13 box counts                   the pieces' per-game box counts (config.yaml box_counts), from the game log
    home                                +1 when he stood in the home five of the game's stints, -1 in the away five,
                                        0 at a neutral site
    team_poss_off, team_poss_def        his team's possessions with the ball and without it in the whole game, summed
                                        over the game's stints
    team_id, opponent_id                the game log
    started                             the game's box score, each team's first five rows (data/roles.py)
    the 9 hustle counts (`HUSTLE`)      the per-game hustle box scores (data/dashboards.py), 2017 on; missing before,
                                        and for a player a game's hustle box score does not list

`write_player_games` writes one file a season to paths.cache/player_games/<season>.parquet (`openrapm data
player-games`).  `trainable_player_games` reads the seasons a fit may train on, `season_games` any one season, the one
in progress included, since a season's rating reads its own games.

Example: Nikola Jokić's row for Denver's 2026 opener, 2025-10-23 at Golden State: home -1, started 1, poss_off 82 and
poss_def 81, his team's 106 and 105, 2 of 13 threes, 6 of 10 twos, 12 defensive rebounds, 10 assists, 8 deflections.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..design import PIECE_PLAYER_SEASON, SEASON_KEY
from .box_scores import season_box
from .config import resolve
from .dashboards import HUSTLE_GAME_COLUMNS, hustle_game_rows
from .designs import stored_piece
from .ingest import game_table, load_gamelog, played_phases
from .roles import season_boxes, starts_by_game
from .seasons import drop_untrainable

HUSTLE = [name.lower() for name in HUSTLE_GAME_COLUMNS]
KEYS = ["season", "phase", "game_id", "game_date", "game_idx", "player_id", "team_id", "opponent_id", "home", "started"]
POSSESSIONS = ["poss_off", "poss_def", "team_poss_off", "team_poss_def"]


def columns(cfg: dict) -> list:
    """The table's columns, in order."""
    return [*KEYS, *POSSESSIONS, *cfg["box_counts"], *HUSTLE]


def _phase_rows(piece: dict, offset: int, box_counts: list) -> pd.DataFrame:
    """One piece's game rows: possessions, box counts, home or away, his team's possessions; `offset` added to the
    piece's game numbers so a playoff piece's games follow the regular season's, as in the season's design."""
    rows = piece["game_poss"].merge(piece["game_box"].drop(columns="phase"), on=["game_idx", PIECE_PLAYER_SEASON],
                                    how="left")
    rows[box_counts] = rows[box_counts].fillna(0.0)
    stint_game = np.asarray(piece["game_idx"], dtype=np.int64)
    n_games = int(piece["n_games"])
    home_five = np.tile(np.r_[np.ones(5, bool), np.zeros(5, bool)], len(stint_game))
    slots = np.asarray(piece["slot_loc"]).ravel()
    fives = pd.DataFrame({"game_idx": np.repeat(stint_game, 10), PIECE_PLAYER_SEASON: slots,
                          "home_five": home_five}).drop_duplicates(["game_idx", PIECE_PLAYER_SEASON])
    rows = rows.merge(fives, on=["game_idx", PIECE_PLAYER_SEASON], how="left")
    poss_home = np.bincount(stint_game, weights=np.asarray(piece["tcols"]["poss_h"], float), minlength=n_games)
    poss_away = np.bincount(stint_game, weights=np.asarray(piece["tcols"]["poss_a"], float), minlength=n_games)
    neutral = np.bincount(stint_game, weights=np.asarray(piece["neutral"], float), minlength=n_games) > 0
    game = rows["game_idx"].to_numpy(dtype=np.int64)
    in_home = rows["home_five"].to_numpy(dtype=bool)
    rows["home"] = np.where(neutral[game], 0.0, np.where(in_home, 1.0, -1.0))
    rows["team_poss_off"] = np.where(in_home, poss_home[game], poss_away[game])
    rows["team_poss_def"] = np.where(in_home, poss_away[game], poss_home[game])
    rows["player_id"] = np.asarray(piece["keys_u"], dtype=np.int64)[rows[PIECE_PLAYER_SEASON].to_numpy()] % SEASON_KEY
    rows = rows.merge(piece["games"][["game_idx", "game_id", "game_date", "phase"]], on="game_idx", how="left")
    rows["game_idx"] = game + int(offset)
    return rows


def _teams(rows: pd.DataFrame, season: int, cfg: dict) -> pd.DataFrame:
    """`rows` with team_id and opponent_id: each player's team in each game from the game log, and the other team.  A
    row the game log has no line for (one in 1997: possessions in the stints, no box line, so counts of 0 as in the
    season's design) takes the team of the five he stood in, from the game table's home and away teams."""
    phases = played_phases(season, cfg)
    box = season_box([season], phases, cfg)[["game_id", "player_id", "team_id"]]
    games = pd.concat([game_table(load_gamelog(season, phase, cfg)) for phase in phases], ignore_index=True)
    games["game_id"] = games["game_id"].astype(str)
    rows = rows.merge(box.drop_duplicates(["game_id", "player_id"]), on=["game_id", "player_id"], how="left")
    sides = rows[["game_id"]].merge(games[["game_id", "home_team_id", "away_team_id"]], on="game_id", how="left")
    by_side = np.where(rows["home_five"].to_numpy(dtype=bool), sides["home_team_id"], sides["away_team_id"])
    rows["team_id"] = rows["team_id"].fillna(pd.Series(by_side, index=rows.index))
    other = np.where(rows["team_id"].to_numpy() == sides["home_team_id"].to_numpy(), sides["away_team_id"],
                     sides["home_team_id"])
    rows["opponent_id"] = other
    return rows


def season_player_games(season: int, cfg: dict) -> pd.DataFrame:
    """One season's game rows (the module docstring), sorted by player, then game."""
    season = int(season)
    box_counts = list(cfg["box_counts"])
    parts, offset = [], 0
    for phase in played_phases(season, cfg):
        piece = stored_piece(season, phase, cfg)
        parts.append(_phase_rows(piece, offset, box_counts))
        offset += int(piece["n_games"])
    rows = pd.concat(parts, ignore_index=True)
    rows = rows[rows.poss_off + rows.poss_def > 0].copy()
    rows["season"] = season
    rows["game_id"] = rows["game_id"].astype(str)
    rows["game_date"] = pd.to_datetime(rows["game_date"])
    rows = _teams(rows.reset_index(drop=True), season, cfg)
    if rows[["team_id", "opponent_id"]].isna().any().any():
        missing = rows[rows[["team_id", "opponent_id"]].isna().any(axis=1)].iloc[0]
        raise ValueError(f"{season}: game rows with no team, the first player {missing.player_id} in "
                         f"{missing.game_id}")
    rows = rows.merge(starts_by_game(season_boxes(season, cfg)), on=["game_id", "player_id"], how="left")
    rows["started"] = rows["started"].fillna(0.0)
    hustle = hustle_game_rows(cfg, season)
    if hustle is None:
        rows[HUSTLE] = np.nan
    else:
        rows = rows.merge(hustle, on=["game_id", "player_id"], how="left")
    rows = rows.astype({"opponent_id": np.int64, "team_id": np.int64, "game_idx": np.int64})
    return rows[columns(cfg)].sort_values(["player_id", "game_idx"], kind="stable").reset_index(drop=True)


def _path(cfg: dict, season: int) -> Path:
    return resolve(cfg, "cache") / "player_games" / f"{int(season)}.parquet"


def write_player_games(cfg: dict, seasons, force: bool = False, verbose: bool = True) -> list:
    """`season_player_games` for each season, written to paths.cache/player_games/<season>.parquet; a season already
    written is kept unless `force`.  Returns the paths written."""
    written = []
    for season in seasons:
        path = _path(cfg, season)
        if path.exists() and not force:
            continue
        rows = season_player_games(season, cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(path, index=False)
        written.append(path)
        if verbose:
            print(f"  game rows {season}: {len(rows):,} rows, {rows.player_id.nunique()} players, "
                  f"{rows.game_id.nunique()} games, hustle on {rows[HUSTLE[0]].notna().mean():.0%}", flush=True)
    return written


def season_games(cfg: dict, season: int) -> pd.DataFrame:
    """One season's game rows as written: what that season's rating reads, the season in progress included."""
    path = _path(cfg, season)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; build it with `openrapm data player-games`")
    return pd.read_parquet(path)


def trainable_player_games(cfg: dict, seasons) -> pd.DataFrame:
    """The game rows of `seasons` a fit may train on: a season whose Finals are not over is dropped and said
    (data/seasons.py), so the network never trains on the season in progress."""
    frames = [season_games(cfg, season) for season in seasons if _path(cfg, season).exists()]
    missing = [int(season) for season in seasons if not _path(cfg, season).exists()]
    if missing:
        raise FileNotFoundError(f"no game rows for {missing}; build them with `openrapm data player-games`")
    rows, _ = drop_untrainable(pd.concat(frames, ignore_index=True), cfg, what="the game rows")
    return rows
