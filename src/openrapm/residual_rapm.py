"""Residual RAPM, for the Bias page: what a season's own games say about each player beyond his OpenRAPM rating.

The published rating is taken as the prediction for every team-game of the season, and a ridge is fitted to what is
left, on the same players' columns (`trade_set`: check 70's machinery on one season alone, as the Bias page of
2026-09-18 had it).  A player's coefficient is his residual RAPM: positive, the games rate him higher than OpenRAPM.

Words used below
    team-game     one team's offense in one game: its points per 100 possessions on the build's target for that side
                  (settings.Targets), weighted by its possessions
    share         a player's share of his team's possessions in that game; the five on the court sum to 5
    multiplier    one free number per side on the published rating, unpenalized, so a rating spread too narrow or too
                  wide for the games is read once, as a multiplier, and not written into every player's residual
    penalty       the ridge's lambda on the residuals: the build's own, 13,037 (settings.RidgeSettings)
    fill          the rating of a player the ratings table has no row for: a quarter of the possession-weighted mean
                  rating of its players under 500 possessions (trade_set.replacement_fill)

The fit, per side: the offense's on the offense target, the defense's on the defense target, each keeping its own half
of the players' columns.

    minimize   sum over team-games  w (y - Z r - s (Z r) - F c - Z a)^2  +  13,037 |a|^2

    y   the team-game's points per 100;  w its possessions;  Z the shares;  r the published rating, raw sign (defense
        is points allowed);  s the multiplier minus one;  F the season's level, home court and the playoff columns;
        a the residual RAPM, raw sign

No free level per team, unlike 2026-09-18's.  On the 514 team-games of 2027 through 2026-11-23, five game folds held
out, team levels raised the held-out mean square from 104.5 to 127.1 on offense and from 54.9 to 67.5 on defense.  The
games so far support no residual at all: each larger penalty did better held out, and none, the residual switched off,
did best (101.6 and 53.2).  So the penalty is the build's own, and a residual is what OpenRAPM's own ridge would add if
it were run once more on the season's team-games.  It is small, and the page's standard errors say how small.

Example: a rating of +2.0 on offense, a multiplier of 1.4 and a residual of +0.1: the games read him at
1.4 x 2.0 + 0.1 = +2.9, and his residual RAPM is +0.1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import trade_set
from .settings import RidgeSettings

PENALTY = RidgeSettings().penalty
SIDES = ("offense", "defense")
COLUMNS = ["player_id", "residual_offense", "residual_defense"]


def raw_ratings(table: pd.DataFrame, season: int) -> pd.DataFrame:
    """player_id, o, d, poss of `season` from a ratings table stored positive-good (rating_off, rating_def, poss_off):
    raw sign, as the design's columns carry it.  Example: rating_def +1.5 reads d = -1.5, 1.5 fewer points allowed."""
    rows = table[table.season == season]
    return pd.DataFrame({"player_id": rows.player_id.astype("int64").to_numpy(),
                         "o": rows.rating_off.astype(float).to_numpy(),
                         "d": -rows.rating_def.astype(float).to_numpy(),
                         "poss": rows.poss_off.astype(float).to_numpy()})


def season_residual(designs: dict, table: pd.DataFrame, season: int,
                    penalty: float = PENALTY) -> tuple[pd.DataFrame, dict]:
    """(`COLUMNS`, one row per player the season's games hold; {"offense": multiplier, "defense": multiplier}).

    `designs` holds the season's design on each side's target, {"offense": ..., "defense": ...}, the build's targets
    (settings.Targets; the command loads them).  The residuals are positive-good on both sides: a defensive residual
    of +0.1 is 0.1 fewer points allowed per 100.
    """
    ratings = raw_ratings(table, season)
    fill = trade_set.replacement_fill(ratings)
    residual, multipliers, players = {}, {}, None
    for position, side in enumerate(SIDES):
        games = trade_set.team_game_design(designs[side])
        if players is None:
            players = games.player_ids
        elif not np.array_equal(players, games.player_ids):
            raise ValueError("the two targets built different player columns")
        alpha, free = trade_set.fit_alpha(games, trade_set.offset_vector(players, ratings, *fill), penalty)
        n = games.n_players
        residual[side] = alpha[:n] if side == "offense" else -alpha[n:]
        multipliers[side] = 1.0 + float(free[position])
    frame = pd.DataFrame({"player_id": np.asarray(players, dtype=np.int64),
                          "residual_offense": residual["offense"], "residual_defense": residual["defense"]})
    return frame, multipliers
