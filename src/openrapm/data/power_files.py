"""The two files the Power page's one rating reads (power_forecast.py says what the rating is).

    team_game_counts   a season's regular-season stints and game log as one row per team per game: the counts the
                       last-10 stats are made of
    load_knots         params/power_forecast.json: the multipliers and stats weights at each fitted point of a season

Both were in power_forecast.py until 2026-11-09, which made a model module read the disk, against the rule
tests/test_layer_boundary.py keeps: the model is handed data, the data layer loads it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..power_forecast import COUNTS, team_rows
from .config import resolve


def team_game_counts(season: int, cfg: dict) -> pd.DataFrame:
    """One row per team per regular-season game: game_id, game_date, tm, the team's counts `*_o` and its opponents'
    `*_d`, three-pointers in `thr` without the long ones in `lthr`."""
    columns = ["game_id", "game_date", *(f"{count}_{end}" for count in COUNTS for end in "ha")]
    stints = pd.read_parquet(resolve(cfg, "stints") / f"{season}_RS.parquet", columns=columns)
    stints["game_id"] = stints.game_id.astype(str).str.zfill(10)
    games = stints.groupby("game_id", as_index=False).agg(
        game_date=("game_date", "first"), **{c: (c, "sum") for c in columns[2:]}
    )
    for end in "ha":  # the stints' three-pointers hold the long ones
        games[f"fga_thr_{end}"] -= games[f"fga_lthr_{end}"]
        games[f"fgm_thr_{end}"] -= games[f"fgm_lthr_{end}"]
    log = pd.read_parquet(
        resolve(cfg, "raw") / "gamelog" / f"{season}_RS.parquet",
        columns=["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP"],
    ).drop_duplicates(["GAME_ID", "TEAM_ID"])
    log["GAME_ID"] = log.GAME_ID.astype(str).str.zfill(10)
    # a neutral-site game lists both teams as "@": the stints' home side is then the lower team id (data/ingest.py)
    log["is_home"] = log.MATCHUP.str.contains(" vs. ", regex=False)
    log = log.sort_values(["GAME_ID", "is_home", "TEAM_ID"], ascending=[True, False, True], kind="stable")
    games["home_tm"] = games.game_id.map(log.groupby("GAME_ID").TEAM_ABBREVIATION.first())
    games["away_tm"] = games.game_id.map(log.groupby("GAME_ID").TEAM_ABBREVIATION.last())
    return team_rows(games)


def load_knots(path: Path) -> list:
    """The knots of `params/power_forecast.json`, each {"share", "offense_multiplier", "defense_multiplier", "stats"},
    earliest first."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return sorted(document["knots"], key=lambda knot: knot["share"])
