"""Per-game player box counts (from LeagueGameLog) in the 13-count layout the padded rates read (box_rates.py).

game_box columns: game_id, player_id, season, phase, fg3m, fg3_miss, fg2m, fg2_miss, ftm, ft_miss,
                  orb, drb, ast, tov, stl, blk, pf
Per-game player possessions come from the stints (design.piece_from_frames), not from here.
"""
from __future__ import annotations

import functools

import pandas as pd

from ..design import SEASON_PHASES
from .config import resolve
from .ingest import load_gamelog


def box_from_gamelog(gl: pd.DataFrame) -> pd.DataFrame:
    g = gl
    out = pd.DataFrame({
        "game_id": g["GAME_ID"].astype(str),
        "player_id": g["PLAYER_ID"].astype(int),
        "team_id": g["TEAM_ID"].astype(int),
        "player_name": g["PLAYER_NAME"].astype(str),
        "season": g["season"].astype(int),
        "phase": g["phase"].astype(str),
        "minutes": g["MIN"].fillna(0).astype(float),
        "fg3m": g["FG3M"].fillna(0).astype(float),
        "fg3_miss": (g["FG3A"].fillna(0) - g["FG3M"].fillna(0)).astype(float),
        "fg2m": (g["FGM"].fillna(0) - g["FG3M"].fillna(0)).astype(float),
        "fg2_miss": ((g["FGA"].fillna(0) - g["FG3A"].fillna(0))
                     - (g["FGM"].fillna(0) - g["FG3M"].fillna(0))).astype(float),
        "ftm": g["FTM"].fillna(0).astype(float),
        "ft_miss": (g["FTA"].fillna(0) - g["FTM"].fillna(0)).astype(float),
        "orb": g["OREB"].fillna(0).astype(float),
        "drb": g["DREB"].fillna(0).astype(float),
        "ast": g["AST"].fillna(0).astype(float),
        "tov": g["TOV"].fillna(0).astype(float),
        "stl": g["STL"].fillna(0).astype(float),
        "blk": g["BLK"].fillna(0).astype(float),
        "pf": g["PF"].fillna(0).astype(float),
        "pts": g["PTS"].fillna(0).astype(float),
    })
    return out


@functools.lru_cache(maxsize=None)
def _season_phase_box(season: int, phase: str, raw: str) -> pd.DataFrame:
    """One season and phase's box rows, from the game log under `raw` (the resolved paths.raw folder).

    Read once per process.  The cache is keyed on the folder as well as the season, so a test that points paths.raw
    at a temporary tree never gets the real tree's rows.  No size limit: a 30-season build asks for all 60 season
    and phase tables, and a regular season's is about 5 MB in memory (4.6 for 1997, 5.2 for 2026)."""
    return box_from_gamelog(load_gamelog(season, phase, {"paths": {"raw": raw}}))


def season_box(seasons, phases, cfg) -> pd.DataFrame:
    """Per game x player box rows for the seasons and phases; each (season, phase) is read once per process."""
    raw = str(resolve(cfg, "raw").absolute())
    return pd.concat([_season_phase_box(int(season), str(phase), raw) for season in seasons for phase in phases],
                     ignore_index=True)


def main_team_by_minutes(season: int, cfg) -> dict:
    """player_id -> the team he played the most minutes for in `season`, regular season and playoffs together.

    The rule the year-over-year test's `movers` split and the checks' mover groups use: box-score minutes, where
    `roles.main_teams` uses on-court possessions.  The two disagree on a handful of traded players, so each
    check keeps the rule its numbers were made with.
    """
    box = season_box([season], list(SEASON_PHASES), cfg)
    minutes = box.groupby(["player_id", "team_id"], as_index=False)["minutes"].sum()
    most = minutes.sort_values("minutes").drop_duplicates("player_id", keep="last")
    return dict(zip(most.player_id.astype(int), most.team_id.astype(int)))


def ft_padding(box: pd.DataFrame, min_fta: float = 20.0) -> tuple[float, float]:
    """League free-throw percentage and the empirical-Bayes padding constant k, in ATTEMPTS.

    Method of moments on player-season totals.  A made/attempt proportion has within-player variance
    p(1-p)/n, so the between-player variance is what is left after subtracting it:

        tau2 = Var(p_hat) - E[p_hat (1 - p_hat) / n]
        k    = E[p_hat (1 - p_hat)] / tau2

    k is in ATTEMPTS, which is the unit free-throw percentage is measured in.  The estimator lives in
    `box_rates.mom_k`, the one padding helper; this is the free-throw call to it.
    """
    from ..box_rates import mom_k
    g = box.groupby(["player_id", "season"], as_index=False)[["ftm", "ft_miss"]].sum()
    return mom_k(g.ftm.to_numpy(), (g.ftm + g.ft_miss).to_numpy(), min_att=min_fta, fallback_rate=0.75)


def ft_totals(box: pd.DataFrame) -> dict:
    """player_id -> (ftm, fta) season totals, for the leave-one-game-out shooter rate."""
    g = box.groupby("player_id", as_index=False)[["ftm", "ft_miss"]].sum()
    return {int(r.player_id): (float(r.ftm), float(r.ftm + r.ft_miss)) for r in g.itertuples(index=False)}
