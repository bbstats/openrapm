"""The Bias page: where OpenRAPM rates players higher or lower than a consensus of outside systems, stat by stat.

The question: among a season's players, do those with more of a stat (blocks, made threes, a high on-court
plus-minus) get more from OpenRAPM than from the consensus, or less?  One number per stat and side answers it, in
points per 100 possessions per one standard deviation of the stat.  Positive: OpenRAPM rates those players higher.

Words used below
    consensus     the weighted mean of five outside systems' ratings, per side, at the owner's weights
                  (`CONSENSUS_WEIGHTS`); a player counts only when all five list him.  This page's alone: consensus.py
                  and data/external/consensus.csv are the 2024-26 snapshot the adoption check reads
    weight        a player's offensive possessions: whose disagreement counts for how much
    width match   OpenRAPM's ratings are spread wider or narrower than the consensus's, which would show as a lean on
                  every stat good players have.  So each side of OpenRAPM is multiplied once, league-wide, to the
                  consensus's standard deviation before anything is subtracted
    gap           width-matched OpenRAPM minus the consensus, each taken from its own mean, so the gaps average 0
    stat          one of `STATS`, from the season panel's padded per-100 values: the 13 box rates, each the mean of a
                  player's offense row and defense row, and his team's plus-minus with him on the court and off it
    slope         how far the gap moves per one standard deviation of the stat: the page's number

Equations, over the compared players i with weights w_i (m is the weighted mean, s the weighted standard deviation)

    consensus    c_i = 0.30 xRAPM_i + 0.20 RAPM3y_i + 0.20 LOKI_i + 0.15 DARKO_i + 0.15 Pulse_i        each side
    width match  k = s(c) / s(o)          gap_i = k (o_i - m(o)) - (c_i - m(c))                         each side
    total        gap_i(total) = gap_i(offense) + gap_i(defense)
    stat         z_i = (x_i - m(x)) / s(x)
    slope        b = m(z x gap)
    its error    e_i = gap_i - b z_i      se = sqrt( n / (n - 2) x sum (w_i z_i e_i)^2 ) / sum w_i

b is the weighted least squares slope of the gap on z, and it is linear in the gap, so a stat's total is its offense
plus its defense.  The standard error is the heteroskedasticity-robust one (Huber-White, HC1): the possessions say
whose gap counts, they are not the gaps' precisions, so the classical formula does not apply.  It takes the multiplier
and the stat's mean and spread as given, and teammates' gaps as independent.

Example with numbers.  Four players, possessions 100, 300, 300, 100, blocks per 100 of 0.5, 0.5, 1.5, 1.5: m = 1.0,
s = 0.5, z = -1, -1, +1, +1.
    offense   OpenRAPM -3, 1, 1, 5 (m 1, s 2), consensus 2, 0, -1, 1 (m 0, s 1): k = 0.5, gap -4, 0, 1, 1,
              b = (400 + 0 + 300 + 100) / 800 = 1.0, e = -3, 1, 0, 0, se = sqrt(2 x 180,000) / 800 = 0.75
    defense   OpenRAPM 1, -1, 0, 2 (m 0, s 1), consensus -2, 2, -2, 2 (m 0, s 2): k = 2, gap 4, -4, 2, 2,
              b = (-400 + 1,200 + 600 + 200) / 800 = 2.0, se = 1.5
    total     gap 0, -4, 3, 3: b = 3.0 = 1.0 + 2.0, se = 0.75
So players one standard deviation higher in blocks, 0.5 per 100, are rated 3.0 higher by OpenRAPM than by the consensus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import BOX_RATES

# The owner, 2026-11-05: "Five, with Pulse", then xrapm.com's 3-year RAPM in the place of nbarapm.com's time-decay
# RAPM, whose robots.txt disallows the pull (notes/DECISIONS.md, "The Bias page's consensus").  Keys as in
# data/public_ratings.SYSTEMS.
CONSENSUS_WEIGHTS = {"xrapm": 0.30, "rapm_3y": 0.20, "loki": 0.20, "darko": 0.15, "pulse": 0.15}
ON_COURT_PLUS_MINUS = "on_court_plus_minus_per100"
OFF_COURT_PLUS_MINUS = "off_court_plus_minus_per100"
STATS = [*BOX_RATES, ON_COURT_PLUS_MINUS, OFF_COURT_PLUS_MINUS]
SIDES = ("offense", "defense")
OURS = {"offense": "rating_off", "defense": "rating_def"}      # the ratings table's column of each side


def consensus(tables: dict, weights: dict = CONSENSUS_WEIGHTS) -> pd.DataFrame:
    """player_id, offense, defense: the weighted mean of the systems' ratings, for the players every system lists.

    `tables` maps a system's key to its table (player_id, offense, defense; data/public_ratings.py).
    Example: a player at 3.0, 1.0, 3.0, 2.0 and 0.0 on a side reads 0.9 + 0.2 + 0.6 + 0.3 + 0.0 = 2.0.
    """
    missing = sorted(set(weights) - set(tables))
    if missing:
        raise ValueError(f"the consensus needs every system; missing {missing}")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"the weights add up to {sum(weights.values())}, not 1")
    listed = [tables[key].set_index("player_id")[list(SIDES)] for key in weights]
    everyone = listed[0].index
    for table in listed[1:]:
        everyone = everyone.intersection(table.index)
    blend = sum(weight * table.loc[everyone] for weight, table in zip(weights.values(), listed))
    return blend.sort_index().rename_axis("player_id").reset_index()


def player_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """player_id and `STATS`, one row per player, from one season's panel rows (two a player, offense and defense).

    A box rate is the mean of his two rows, which count it per 100 of his offensive and of his defensive possessions.
    The plus-minus columns are the same on both rows: on court, his team's points scored minus allowed per 100 with
    him playing; off court, the same in the stints he sat out of the games he played (on_court.py).
    """
    court = ["on_court_scored_per100", "on_court_allowed_per100", "off_court_scored_per100", "off_court_allowed_per100"]
    stats = panel.groupby("player_id")[[*BOX_RATES, *court]].mean()
    stats[ON_COURT_PLUS_MINUS] = stats.on_court_scored_per100 - stats.on_court_allowed_per100
    stats[OFF_COURT_PLUS_MINUS] = stats.off_court_scored_per100 - stats.off_court_allowed_per100
    return stats[STATS].reset_index()


def compared_players(ratings: pd.DataFrame, blend: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """The players the page compares, by player id: those of `ratings` (one season's shown players: player_id,
    poss_season, rating_off, rating_def) whom the consensus and the panel both hold.  Columns: player_id, weight,
    rating_off, rating_def, consensus_offense, consensus_defense and `STATS`."""
    ours = ratings[["player_id", "poss_season", *OURS.values()]].rename(columns={"poss_season": "weight"})
    theirs = blend.rename(columns={side: f"consensus_{side}" for side in SIDES})
    return ours.merge(theirs, on="player_id").merge(stats, on="player_id").sort_values("player_id") \
               .reset_index(drop=True)


def weighted_mean(values, weights) -> float:
    return float(np.sum(weights * values) / np.sum(weights))


def weighted_sd(values, weights) -> float:
    """The weighted standard deviation around the weighted mean (weights as frequencies, no n - 1)."""
    return float(np.sqrt(weighted_mean((values - weighted_mean(values, weights)) ** 2, weights)))


def width_match(ours, theirs, weights) -> tuple[np.ndarray, float]:
    """(gap, multiplier) for one side: OpenRAPM's ratings from their mean, multiplied to the consensus's standard
    deviation, minus the consensus from its mean.  Example: the module's offense, multiplier 0.5, gap -4, 0, 1, 1."""
    multiplier = weighted_sd(theirs, weights) / weighted_sd(ours, weights)
    gap = multiplier * (ours - weighted_mean(ours, weights)) - (theirs - weighted_mean(theirs, weights))
    return gap, multiplier


def slope_per_sd(stat, gap, weights) -> tuple[float, float]:
    """(slope, standard error): how far `gap` moves per one standard deviation of `stat`, and the robust standard
    error of that (the module's equations).  A stat with no spread among the players has no slope: (nan, nan)."""
    spread = weighted_sd(stat, weights)
    if not spread > 0:
        return float("nan"), float("nan")
    z = (stat - weighted_mean(stat, weights)) / spread
    slope = weighted_mean(z * gap, weights)
    leftover = gap - weighted_mean(gap, weights) - slope * z
    n = len(gap)
    error = np.sqrt(n / (n - 2) * np.sum((weights * z * leftover) ** 2)) / np.sum(weights)
    return slope, float(error)


def bias_table(players: pd.DataFrame, stats=STATS) -> tuple[pd.DataFrame, dict]:
    """(one row per stat, {"offense": multiplier, "defense": multiplier}) from `compared_players`.

    The rows: stat, sd (the stat's standard deviation among the players, in its own units), then offense, defense
    and total, the slopes, and offense_se, defense_se and total_se, their standard errors.
    """
    weights = players.weight.to_numpy(float)
    gaps, multipliers = {}, {}
    for side in SIDES:
        gaps[side], multipliers[side] = width_match(players[OURS[side]].to_numpy(float),
                                                    players[f"consensus_{side}"].to_numpy(float), weights)
    gaps["total"] = gaps["offense"] + gaps["defense"]
    rows = []
    for stat in stats:
        values = players[stat].to_numpy(float)
        row = {"stat": stat, "sd": weighted_sd(values, weights)}
        for side, gap in gaps.items():
            row[side], row[f"{side}_se"] = slope_per_sd(values, gap, weights)
        rows.append(row)
    columns = ["stat", "sd", "offense", "defense", "total", "offense_se", "defense_se", "total_se"]
    return pd.DataFrame(rows)[columns], multipliers
