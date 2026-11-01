"""The padded box-score rates: a player's counts per 100 possessions, pulled toward players with as many possessions.

A box score counts what a player did (made threes, steals, fouls, ...); divided by the possessions he was on the court
for, a count becomes a rate per 100 possessions.  Over a few hundred possessions most of a rate is luck, so every rate
the box-score prior reads is padded: blended with a typical value, in proportion to how little the possessions can be
trusted.  The season panel's thirteen rates and their `_uncentred` twins come from here (data/panel.py).

Words used below
    count          one of the thirteen per-game box counts of config.yaml's box_counts (fg3m, fg3_miss, ..., pf)
    possessions    n, the possessions he was on the court for: his team's on offence, the opponent's on defence
    rate           100 x count / n
    pad k          how many possessions' worth of the typical value is blended in, one per count and season,
                   measured from every player's odd- against even-numbered games (`split_half_k`)
    bin            one of six ranges of possessions, equally wide on a log scale
    target         the typical value a rate is padded toward: that season's rate among the players of his bin (the
                   possession-conditional target, config.yaml's pad_target), or the season's league rate, or the rate
                   his height, weight and bin predict (`body_targets`, pad_target body_and_possessions)
    group pad      k_group, how many possessions' worth of the league rate a bin's own rate is blended with before it
                   serves as a target, one per count and season (`group_k`; config.yaml's pad_bin_target)
    centred        the possession-weighted average player subtracted, so 0 is an average player

The padding (`shrink`):

    padded = (n x rate + k x target) / (n + k)

    n        his possessions
    k        the pad k = sigma^2_within / tau^2_between: the noise variance of one possession's worth of the rate,
             over the variance of the true rates between players
    target   that season's rate among the players of his possession bin

Example: a 200-possession bench player with 2.1 steals per 100, k = 450 and a bin rate of 1.4 is padded to
(200 x 2.1 + 450 x 1.4) / (200 + 450) = 1.6 steals per 100; a 3,000-possession starter with the same 2.1 keeps almost
all of it: (3000 x 2.1 + 450 x 1.4) / 3450 = 2.0.

The bin's own rate is a guess from a sample too (pad_bin_target league; notes/DECISIONS.md, experiment 127).  From
the first 88 games of a season a bin's steal rate rests on about 50 steals, and over the 30 finished seasons a bin
1.00 per 100 above the league at that point was followed by only 0.14 more over the rest of the season.  So the bin
rate is padded toward the season's league rate by the same rule, one level up, before it serves as a target:

    target_b = (P_b x rate_b + k_group x league) / (P_b + k_group)

    P_b        the bin's offensive possessions in the design, its players' summed
    rate_b     its rate, 100 x its players' counts / P_b (`bin_targets`)
    k_group    sigma^2_within / tau^2_groups: the same per-possession noise over the variance of the true bin rates
               around the league rate, which is the possession-weighted spread of the bin rates less what chance
               alone would produce, sigma^2_within / P_b for each bin (`group_k`, the method of moments `mom_k`
               uses); K_MAX when nothing is left, so the target is the league rate

Example: at 88 games a bin with 9,000 possessions and 1.9 steals per 100, k_group 12,000 and a league rate of 1.6 is
padded to (9,000 x 1.9 + 12,000 x 1.6) / 21,000 = 1.73; over a whole season the same bin holds 130,000 possessions
and keeps (130,000 x 1.9 + 12,000 x 1.6) / 142,000 = 1.87.  The players' pad k is then measured around the padded
target (pad_k_around target).  With pad_bin_target none, as shipped, the bin rate is the target as counted.

The body target (pad_target body_and_possessions; notes/DECISIONS.md, experiment 86; the owner, 2026-10-20: "out of
fold ... priors for the padded metrics (eg use height and weight and, the buckets we were using earlier like for
possessions)").  Per season and count, a weighted least-squares line of the players' rates on height, weight and one
column per possession bin, weighted by offensive possessions, fitted without the player's fifth of the players
(player_id % 5) and read at his own height, weight and bin; never below 0.  His own games never set his target.  The
pad k is then measured around it: the halves' covariance of (rate - his target), not of (rate - the mean), so k says
how much of the spread the target leaves.  A better target leaves less, so k rises: the rate leans more on a better
guess.

Example: a 7-foot rookie with 300 possessions and 4.0 blocks per 100 is padded toward what players of his height,
weight and bin block, say 3.0, where the bin alone would say 1.2.

`mom_k` is the same idea for a made / attempted proportion (a shooter's free-throw percentage, data/shots.py), its k in
attempts.

Each step below keeps the arithmetic, and the order of every sum, that the published season panel was built with: the
box-score prior reads these numbers, and a rate that moved in its last digit would move the published ratings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["K_MAX", "K_FALLBACK", "PAD_TARGETS", "PAD_BIN_TARGETS", "shrink", "mom_k", "split_half_k",
           "split_half_noise", "GameTotals", "game_totals", "player_possessions", "league_rates", "pad_k",
           "pad_noise", "possession_bins", "bin_of", "bin_targets", "bin_possessions", "group_k",
           "padded_bin_targets", "pad_rates", "lineups", "SideRates", "BoxRates", "box_rates", "rate_table"]

K_MAX = 1e5                     # the pad k of a rate that shows no spread between players: the target, all but whole
K_FALLBACK = 40.0               # mom_k's k when too few players carry a proportion to estimate it
MIN_HALF_POSSESSIONS = 20.0     # a player enters split_half_k with at least this many possessions in each half
POSSESSION_BINS = 6
MIN_BIN_PLAYERS = 5             # a season's bin needs this many players to have a rate of its own
PAD_TARGETS = ("league", "poss_conditional", "body_and_possessions")
# where the pad k measures the spread between players (config pad_k_around): around the league rate, as shipped, or
# around the target the rates are padded toward.  The body target always measures it around itself; the
# possession-conditional one measured it around the league until 2026-10-22, which counts the spread the bins
# already explain as signal and pads too little: fouls k 174 / 234 / 346 in 1998 / 2010 / 2024 against
# 217 / 307 / 387 around the bins (notes/DECISIONS.md, the bug-fix pass, experiment 96)
PAD_K_AROUND = ("league", "target")
# whether each possession bin's own rate is padded toward the season's league rate by the bin's possessions before it
# serves as a target (config pad_bin_target): "none", as shipped, the bin rate as counted; "league", the group pad of
# the module docstring.  From the first 7% of the 30 finished seasons, one extra padded steal per 100 was followed by
# 0.70 later in the season with the bin rate as counted and 0.89 with the league rate as the target; the bin rate
# itself, 1.00 higher, by 0.14 (notes/DECISIONS.md, experiment 127)
PAD_BIN_TARGETS = ("none", "league")
BODY = ("height_in", "weight_lb")      # the body target's inputs (data/bio.py), one row per player
BODY_FOLDS = 5                          # its player folds: player_id % 5
MIN_BODY_PLAYERS = 20                   # a season's fold needs this many players with possessions to fit a line
LINEUP = 5                      # players on the court per side


# ------------------------------------------------------------------------------------------ the padding itself
def shrink(rate, n, k, target):
    """(n x rate + k x target) / (n + k), elementwise with broadcasting; the target where n + k is 0.

    Example: shrink(0.9, 10, 30, 0.5) = (9 + 15) / 40 = 0.6.
    """
    rate = np.asarray(rate, dtype=float)
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    target = np.asarray(target, dtype=float)
    denom = n + k
    safe = np.where(denom > 0, denom, 1.0)
    return np.where(denom > 0, (n * rate + k * target) / safe, target)


def mom_k(made, att, min_att: float = 20.0, fallback_rate: float | None = None) -> tuple[float, float]:
    """The league rate and the pad k, in ATTEMPTS, of a made / attempted proportion, by the method of moments.

    A proportion over n attempts has sampling variance p(1 - p) / n, so the spread between players is what is left of
    the observed spread once that is taken out:

        tau^2 = Var_w(p) - E_w[p (1 - p) / n]        k = E_w[p (1 - p)] / tau^2

    with attempt weights w.  Players under `min_att` attempts are left out of the estimate (their p is too coarse) but
    are still padded by the caller.  With fewer than 20 players left the estimate is not trusted and (league,
    K_FALLBACK) comes back, the league rate from whatever attempts exist, or `fallback_rate` when there are none.

    Example: rates drawn from a Beta(30, 10) distribution (mean 0.75) give k close to 30 + 10 = 40.
    """
    made = np.asarray(made, dtype=float)
    att = np.asarray(att, dtype=float)
    tot_m, tot_a = made.sum(), att.sum()
    league = float(tot_m / tot_a) if tot_a > 0 else (0.5 if fallback_rate is None else float(fallback_rate))
    keep = att >= min_att
    if keep.sum() < 20:
        return league, K_FALLBACK
    p, w = made[keep] / att[keep], att[keep]
    mu = np.average(p, weights=w)
    between = np.average((p - mu) ** 2, weights=w)
    within = np.average(p * (1.0 - p) / w, weights=w)
    tau2 = max(between - within, 1e-6)
    return league, float(np.average(p * (1.0 - p), weights=w) / tau2)


# ------------------------------------------------------------------------------------------- the pad k
def _odd_even_totals(player: np.ndarray, game: np.ndarray, counts: np.ndarray, possessions: np.ndarray,
                     n_players: int) -> tuple:
    """Each player's counts and possessions summed over his 1st, 3rd, 5th, ... games (half A) and over his 2nd, 4th,
    ... games (half B), his games taken in the order they were played.  Returns (counts A, possessions A, counts B,
    possessions B)."""
    order = np.lexsort((game, player))
    player_sorted, counts_sorted, possessions_sorted = player[order], counts[order], possessions[order]
    first = np.r_[True, player_sorted[1:] != player_sorted[:-1]]
    first_row = np.maximum.accumulate(np.where(first, np.arange(len(player_sorted)), 0))
    half_a = (np.arange(len(player_sorted)) - first_row) % 2 == 0
    counts_a, counts_b = np.zeros((n_players, counts.shape[1])), np.zeros((n_players, counts.shape[1]))
    possessions_a, possessions_b = np.zeros(n_players), np.zeros(n_players)
    np.add.at(counts_a, player_sorted[half_a], counts_sorted[half_a])
    np.add.at(possessions_a, player_sorted[half_a], possessions_sorted[half_a])
    np.add.at(counts_b, player_sorted[~half_a], counts_sorted[~half_a])
    np.add.at(possessions_b, player_sorted[~half_a], possessions_sorted[~half_a])
    return counts_a, possessions_a, counts_b, possessions_b


def _half_rates(counts_a: np.ndarray, possessions_a: np.ndarray, counts_b: np.ndarray,
                possessions_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r_A, r_B, w): the two half rates per 100, one row per player, and the weight w = P_A + P_B as a column."""
    rate_a = 100.0 * counts_a / possessions_a[:, None]
    rate_b = 100.0 * counts_b / possessions_b[:, None]
    return rate_a, rate_b, (possessions_a + possessions_b)[:, None]


def _noise(rate_a: np.ndarray, rate_b: np.ndarray, possessions_a: np.ndarray, possessions_b: np.ndarray) -> np.ndarray:
    """sigma^2_within per count: the mean over players of (r_A - r_B)^2 / (1/P_A + 1/P_B), the noise variance of one
    possession's worth of the rate, since the squared gap between a player's two halves is pure noise."""
    inverse = (1.0 / possessions_a + 1.0 / possessions_b)[:, None]
    return np.mean((rate_a - rate_b) ** 2 / inverse, axis=0)


def _spread(rate_a: np.ndarray, rate_b: np.ndarray, weight: np.ndarray,
            target: np.ndarray | None = None) -> np.ndarray:
    """tau^2_between per count: sum w (r_A - mean_A)(r_B - mean_B) / sum w, the halves' weighted covariance, around the
    weighted means or, with `target` (one row per player), around it: the spread the target leaves."""
    if target is None:
        mean_a = (weight * rate_a).sum(0) / weight.sum()
        mean_b = (weight * rate_b).sum(0) / weight.sum()
    else:
        mean_a = mean_b = target
    return ((weight * (rate_a - mean_a) * (rate_b - mean_b)).sum(0) / weight.sum())


def _k_of(within: np.ndarray, between: np.ndarray) -> np.ndarray:
    """within / between per count, K_MAX where between is 0 or less (or NaN), never above K_MAX."""
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(between > 0, within / np.maximum(between, 1e-12), K_MAX)
    return np.minimum(k, K_MAX)


def _season_k(counts_a: np.ndarray, possessions_a: np.ndarray, counts_b: np.ndarray,
              possessions_b: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """One season's pad k per count from its players' two half rates r_A and r_B (per 100, over P_A and P_B
    possessions):

        sigma^2_within = mean over players of (r_A - r_B)^2 / (1/P_A + 1/P_B)
        tau^2_between  = sum w (r_A - mean_A)(r_B - mean_B) / sum w,   w = P_A + P_B
        k              = sigma^2_within / tau^2_between,  K_MAX when tau^2 <= 0

    The two halves share a player's true rate and nothing else, so their weighted covariance is the variance of the
    true rates between players; the squared gap between them is pure noise.  With `target` (one row per player, the
    body target), tau^2 is the weighted mean of (r_A - target)(r_B - target): the spread the target leaves.
    """
    rate_a, rate_b, weight = _half_rates(counts_a, possessions_a, counts_b, possessions_b)
    within = _noise(rate_a, rate_b, possessions_a, possessions_b)
    return _k_of(within, _spread(rate_a, rate_b, weight, target))


def _season_halves(player: np.ndarray, game: np.ndarray, counts: np.ndarray, possessions: np.ndarray,
                   season_of_player: np.ndarray, n_seasons: int, min_half_possessions: float) -> dict:
    """{season: the mask of its players carrying `min_half_possessions` in both halves} for the seasons with at least
    three of them, with the half totals of every player-season (`_odd_even_totals`): (masks, counts_a, possessions_a,
    counts_b, possessions_b)."""
    counts_a, possessions_a, counts_b, possessions_b = _odd_even_totals(player, game, counts, possessions,
                                                                        len(season_of_player))
    masks = {}
    for season in range(n_seasons):
        both = ((season_of_player == season) & (possessions_a >= min_half_possessions)
                & (possessions_b >= min_half_possessions))
        if both.sum() >= 3:
            masks[season] = both
    return masks, counts_a, possessions_a, counts_b, possessions_b


def split_half_k(player: np.ndarray, game: np.ndarray, counts: np.ndarray, possessions: np.ndarray,
                 season_of_player: np.ndarray, n_seasons: int,
                 min_half_possessions: float = MIN_HALF_POSSESSIONS, target: np.ndarray | None = None) -> np.ndarray:
    """The pad k of every count in every season, (seasons x counts), in possessions; NaN for a season with fewer than
    three players carrying `min_half_possessions` in both halves.

    `player`, `game`, `counts` and `possessions` are one row per (game, player-season); `season_of_player` gives each
    player-season's season.  `_season_k` has the estimator.

    Example: a count whose true rates spread with variance 0.36 between players and whose per-possession noise variance
    is 162 (a rate of about 1.6 per 100) gets k = 162 / 0.36 = 450 possessions.
    """
    masks, counts_a, possessions_a, counts_b, possessions_b = _season_halves(
        player, game, counts, possessions, season_of_player, n_seasons, min_half_possessions)
    k = np.full((n_seasons, counts.shape[1]), np.nan)
    for season, both in masks.items():
        k[season] = _season_k(counts_a[both], possessions_a[both], counts_b[both], possessions_b[both],
                              None if target is None else target[both])
    return k


def split_half_noise(player: np.ndarray, game: np.ndarray, counts: np.ndarray, possessions: np.ndarray,
                     season_of_player: np.ndarray, n_seasons: int,
                     min_half_possessions: float = MIN_HALF_POSSESSIONS) -> np.ndarray:
    """sigma^2_within of every count in every season, (seasons x counts): the noise variance of one possession's worth
    of the rate, from the same players and halves as `split_half_k`; NaN for a season with fewer than three of them.

    Example: Poisson counts at 1.6 per 100 give about 160: a rate over P possessions has noise variance 160 / P.
    """
    masks, counts_a, possessions_a, counts_b, possessions_b = _season_halves(
        player, game, counts, possessions, season_of_player, n_seasons, min_half_possessions)
    noise = np.full((n_seasons, counts.shape[1]), np.nan)
    for season, both in masks.items():
        rate_a, rate_b, _ = _half_rates(counts_a[both], possessions_a[both], counts_b[both], possessions_b[both])
        noise[season] = _noise(rate_a, rate_b, possessions_a[both], possessions_b[both])
    return noise


# ---------------------------------------------------------------------- the counts and possessions behind a rate
@dataclass(frozen=True)
class GameTotals:
    """One row per game and player-season: his box counts in the game and his possessions on each side."""
    game: np.ndarray
    player_season: np.ndarray
    counts: np.ndarray                   # (rows, counts)
    offensive_possessions: np.ndarray
    defensive_possessions: np.ndarray


def game_totals(design) -> GameTotals:
    """Every (game, player-season) of the design's games: his box counts and his possessions.

    The design's possession table (from the stints) and its box table are joined: a player with possessions and no box
    line counts zeros, one with a box line and no possessions has zero possessions.  A game counts when the design has
    rows from it and a box score for it, so a design cut to part of a season counts that part only.
    """
    counts = list(design.spec.box_counts)
    possessions = design.game_poss[["game_idx", "player_season", "poss_off", "poss_def"]]
    box = design.game_box[["game_idx", "player_season", "phase", *counts]]
    box_games = np.unique(box["game_idx"].to_numpy())
    table = possessions.merge(box.drop(columns="phase"), on=["game_idx", "player_season"], how="outer")
    table[counts] = table[counts].fillna(0.0)
    table[["poss_off", "poss_def"]] = table[["poss_off", "poss_def"]].fillna(0.0)
    played = np.intersect1d(np.unique(design.rows["game_idx"].to_numpy()), box_games)
    table = table[np.isin(table["game_idx"].to_numpy().astype(np.int64), played)]
    return GameTotals(game=table["game_idx"].to_numpy().astype(np.int64),
                      player_season=table["player_season"].to_numpy().astype(np.int64),
                      counts=table[counts].to_numpy(dtype=float),
                      offensive_possessions=table["poss_off"].to_numpy(dtype=float),
                      defensive_possessions=table["poss_def"].to_numpy(dtype=float))


def player_possessions(design, game_ids=None) -> dict:
    """{"O", "D"}: each player's possessions over the design's counted games (`game_totals`), in the design's player
    order, as `box_rates` sums them; with `game_ids`, over those games alone.  The per-game hustle counts of the season
    in progress divide by it, since the league has not processed last night's games by morning
    (data/dashboard_rates.py).

    Example: a player on the floor for 300 offensive possessions over 4 games, 80 of them last night, has 220 over the
    other 3."""
    spec = design.spec
    player_of = (spec.player_of_player_season if spec.player_of_player_season is not None
                 else np.arange(spec.n_player_seasons))
    totals = game_totals(design)
    keep = np.ones(len(totals.game), dtype=bool)
    if game_ids is not None:
        wanted = {str(game) for game in game_ids}
        chosen = design.games.loc[design.games["game_id"].astype(str).isin(wanted), "game_idx"].to_numpy(np.int64)
        keep = np.isin(totals.game, chosen)
    per_season = _sum_matrix(totals.player_season[keep], spec.n_player_seasons)
    per_player = _sum_matrix(player_of, spec.n_players)
    return {side: np.asarray(per_player @ (per_season @ values[keep])).ravel()
            for side, values in (("O", totals.offensive_possessions), ("D", totals.defensive_possessions))}


def _sum_matrix(index: np.ndarray, n_out: int) -> sp.csr_matrix:
    """The (n_out x len(index)) matrix whose product with v sums v's entries into `index`: S @ v adds v[i] into row
    index[i], in the order of i."""
    index = np.asarray(index, dtype=np.int64)
    return sp.csr_matrix((np.ones(len(index)), (index, np.arange(len(index)))), shape=(n_out, len(index)))


def league_rates(counts: np.ndarray, offensive_possessions: np.ndarray, season_of: np.ndarray,
                 n_seasons: int) -> np.ndarray:
    """(seasons x counts): each season's league rate, 100 x every player's counts / every offensive possession.

    Example: 59,000 assists over 1,230,000 player-possessions on offence (246,000 team possessions, five players
    each) is 4.8 per 100."""
    rates = np.zeros((n_seasons, counts.shape[1]))
    for season in range(n_seasons):
        mask = season_of == season
        total = offensive_possessions[mask].sum()
        rates[season] = 100.0 * counts[mask].sum(0) / total if total > 0 else 0.0
    return rates


def pad_k(totals: GameTotals, season_of: np.ndarray, n_seasons: int, target: np.ndarray | None = None) -> np.ndarray:
    """(seasons x counts): `split_half_k` on the offensive possessions, a season without enough players taking the
    mean k of the seasons that have them (all zeros when none has); around `target` (one row per player-season) when
    given."""
    k = split_half_k(totals.player_season, totals.game, totals.counts, totals.offensive_possessions, season_of,
                     n_seasons, target=target)
    if not np.isfinite(k).any():
        return np.zeros((n_seasons, totals.counts.shape[1]))
    return np.where(np.isnan(k), np.nanmean(k, axis=0, keepdims=True), k)


def pad_noise(totals: GameTotals, season_of: np.ndarray, n_seasons: int) -> np.ndarray:
    """(seasons x counts): `split_half_noise` on the offensive possessions, as `pad_k` measures its k; NaN for a season
    without enough players, which `group_k` reads as no evidence (K_MAX, the league rate)."""
    return split_half_noise(totals.player_season, totals.game, totals.counts, totals.offensive_possessions, season_of,
                            n_seasons)


# ------------------------------------------------------------------------------ the possession-conditional target
def possession_bins(offensive_possessions: np.ndarray, player_possessions: np.ndarray) -> np.ndarray | None:
    """The edges of the six possession bins: 0, then five edges equally spaced in log possessions from the 1st
    percentile (at least 1) to the most any player had, then infinity.  None with fewer than 30 players.

    `offensive_possessions` (one per player-season) says who played; `player_possessions` (one per player-season, his
    whole possessions in the design) is what is binned.

    Example: possessions from 12 to 3,600 give edges 0, 31, 80, 208, 538, 1,391, infinity.
    """
    played = offensive_possessions > 0
    if played.sum() < MIN_BIN_PLAYERS * POSSESSION_BINS:
        return None
    low = np.log(np.maximum(np.percentile(player_possessions[played], 1), 1.0))
    high = np.log(player_possessions[played].max())
    return np.r_[0.0, np.exp(np.linspace(low, high, POSSESSION_BINS + 1))[1:-1], np.inf]


def bin_of(edges: np.ndarray, possessions: np.ndarray) -> np.ndarray:
    """The bin (0 to 5) each number of possessions falls in."""
    return np.clip(np.searchsorted(edges, possessions, side="right") - 1, 0, len(edges) - 2)


def _group_rate(counts: np.ndarray, possessions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """100 x the group's counts / its possessions; NaN for an empty group or one without possessions."""
    if mask.sum() and possessions[mask].sum() > 0:
        return 100.0 * counts[mask].sum(0) / possessions[mask].sum()
    return np.full(counts.shape[1], np.nan)


def bin_targets(counts: np.ndarray, offensive_possessions: np.ndarray, season_of: np.ndarray, n_seasons: int,
                bins: np.ndarray, league: np.ndarray) -> np.ndarray:
    """(seasons x bins x counts): the target of a player in each season and bin.

    The season's rate among the players of the bin when it has at least five; else the bin's rate over every season;
    else the season's league rate.  A deep-bench player's 2.1 steals per 100 is padded toward what other deep-bench
    players did, not toward the starters who dominate the league rate.
    """
    played = offensive_possessions > 0
    pooled = np.array([_group_rate(counts, offensive_possessions, played & (bins == b))
                       for b in range(POSSESSION_BINS)])
    targets = np.zeros((n_seasons, POSSESSION_BINS, counts.shape[1]))
    for season in range(n_seasons):
        for b in range(POSSESSION_BINS):
            mask = played & (bins == b) & (season_of == season)
            if mask.sum() >= MIN_BIN_PLAYERS and offensive_possessions[mask].sum() > 0:
                targets[season, b] = 100.0 * counts[mask].sum(0) / offensive_possessions[mask].sum()
            elif np.isfinite(pooled[b]).all():
                targets[season, b] = pooled[b]
            else:
                targets[season, b] = league[season]
    return targets


def bin_possessions(offensive_possessions: np.ndarray, season_of: np.ndarray, n_seasons: int,
                    bins: np.ndarray) -> np.ndarray:
    """(seasons x bins): P_b, each bin's offensive possessions in each season, its player-seasons' summed; the
    possessions `bin_targets` divides the bin's counts by."""
    held = np.zeros((n_seasons, POSSESSION_BINS))
    np.add.at(held, (np.asarray(season_of, dtype=np.int64), np.asarray(bins, dtype=np.int64)),
              np.asarray(offensive_possessions, dtype=float))
    return held


def group_k(noise: np.ndarray, targets: np.ndarray, held: np.ndarray, league: np.ndarray) -> np.ndarray:
    """(seasons x counts): k_group, the pad k of the bin targets themselves, sigma^2_within / tau^2_groups.

    `noise` is sigma^2_within per season and count (`pad_noise`); `targets` the bin targets (seasons x bins x counts,
    `bin_targets`); `held` each bin's possessions (seasons x bins, `bin_possessions`); `league` the league rates
    (seasons x counts).  tau^2_groups is the variance of the true bin rates around the league rate: the
    possession-weighted spread of the bin targets around it, less the noise variance each carries, sigma^2_within / P_b,
    weighted the same way (the method of moments, as `mom_k`).  K_MAX where nothing is left, or where the season has no
    noise reading, so the target is the league rate.

    Example: steals, sigma^2_within 160 and bin rates whose true spread is 0.013 give 160 / 0.013 = 12,000
    possessions.
    """
    weight = held[:, :, None]
    total = np.maximum(weight.sum(1), 1e-12)
    spread = (weight * (targets - league[:, None, :]) ** 2).sum(1) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chance = noise * (held > 0).sum(1)[:, None] / total
    return _k_of(noise, spread - chance)


def padded_bin_targets(targets: np.ndarray, held: np.ndarray, league: np.ndarray, k_group: np.ndarray) -> np.ndarray:
    """(seasons x bins x counts): the bin targets padded toward the season's league rate by the bin's own possessions,
    (P_b x target_b + k_group x league) / (P_b + k_group) (`shrink`); a bin without possessions takes the league rate.

    Example: 9,000 possessions, 1.9 steals per 100, k_group 12,000 and a league rate of 1.6 give 1.73; with 130,000
    possessions, 1.87.
    """
    return shrink(targets, held[:, :, None], k_group[:, None, :], league[:, None, :])


def body_targets(counts: np.ndarray, offensive_possessions: np.ndarray, season_of: np.ndarray, n_seasons: int,
                 train_bins: np.ndarray, apply_bins: np.ndarray, body: np.ndarray, player_ids: np.ndarray,
                 fallback: np.ndarray) -> np.ndarray:
    """(player-seasons x counts): each player-season's body target (module docstring).

    `body` is (player-seasons x 2), height and weight; `train_bins` the bins the lines are fitted on (offensive
    possessions), `apply_bins` the bins each player-season is read at (this side's possessions); `fallback` the
    target where a season's fold has too few players to fit a line.  A small ridge, 1e-8 of the mean diagonal, keeps a
    bin without players from making the fit singular.
    """
    played = offensive_possessions > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(played[:, None], 100.0 * counts / np.where(played, offensive_possessions, 1.0)[:, None], 0.0)

    def design(bins):
        dummies = (np.asarray(bins)[:, None] == np.arange(1, POSSESSION_BINS)[None, :]).astype(float)
        return np.column_stack([np.ones(len(bins)), body, dummies])

    fit_rows, read_rows = design(train_bins), design(apply_bins)
    fold = np.asarray(player_ids, dtype=np.int64) % BODY_FOLDS
    out = np.asarray(fallback, dtype=float).copy()
    for season in range(n_seasons):
        in_season = season_of == season
        for left_out in range(BODY_FOLDS):
            train = in_season & played & (fold != left_out)
            read = in_season & (fold == left_out)
            if train.sum() < MIN_BODY_PLAYERS or not read.any():
                continue
            weighted = fit_rows[train] * offensive_possessions[train][:, None]
            gram = weighted.T @ fit_rows[train]
            gram += 1e-8 * np.mean(np.diag(gram)) * np.eye(len(gram))
            beta = np.linalg.solve(gram, weighted.T @ rate[train])
            out[read] = np.maximum(read_rows[read] @ beta, 0.0)
    return out


# ------------------------------------------------------------------------------------------- one player, one rate
def _blend(per_season: np.ndarray, weight: np.ndarray, player_of: np.ndarray, n_players: int) -> np.ndarray:
    """A per-season quantity (one row per player-season) brought down to one row per player: the weighted mean over
    his seasons in the design, the plain mean when he has no weight.  In a one-season design every player has one
    season, so this is that season's value."""
    summing = _sum_matrix(player_of, n_players)
    numerator = np.asarray(summing @ (per_season * weight[:, None]))
    denominator = np.asarray(summing @ weight).ravel()
    total = np.asarray(summing @ per_season)
    seasons = np.asarray(summing @ np.ones(len(player_of))).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator[:, None] > 0, numerator / np.where(denominator > 0, denominator, 1.0)[:, None],
                        total / np.maximum(seasons, 1.0)[:, None])


def pad_rates(counts: np.ndarray, possessions: np.ndarray, k: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(players x counts): every rate padded, (n x rate + k x target) / (n + k) with rate = 100 x count / n and n his
    possessions; a player without possessions gets the target.

    The published panel wrote n as P^2 / P, a form that allowed weighting games (a bootstrap, retired); possessions
    are whole numbers, so P^2 / P is P to the last bit and n is the possessions themselves."""
    played = possessions > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(played[:, None], 100.0 * counts / np.where(played, possessions, 1.0)[:, None], 0.0)
    return shrink(rate, possessions[:, None], k, target)


def lineups(design) -> tuple[np.ndarray, np.ndarray]:
    """(offence, defence): each row's five player columns, in increasing order, as two (rows x 5) arrays."""
    if design.parts is not None:
        return design.parts["lineup_o"], design.parts["lineup_d"]
    out = []
    for columns in (design.spec.offense_columns, design.spec.defense_columns):
        block = design.X[:, columns].tocsr()
        block.sort_indices()
        if not np.all(np.diff(block.indptr) == LINEUP):
            raise ValueError("every row must have exactly five players on each side")
        out.append(block.indices.reshape(-1, LINEUP))
    return out[0], out[1]


def _lineup_mean(rates: np.ndarray, lineup: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The weighted mean over the design's rows of the five players' rates summed: what an average lineup carries."""
    summed = np.zeros((len(lineup), rates.shape[1]))
    for slot in range(LINEUP):
        summed += rates[lineup[:, slot]]
    return (weights[:, None] * summed).sum(0) / weights.sum()


# ------------------------------------------------------------------------------------------------- the whole step
@dataclass(frozen=True)
class SideRates:
    """One side's rates: offence rates are per 100 of his team's offensive possessions, defence per 100 defensive."""
    rates: np.ndarray            # (players x counts) padded, level kept (the panel's `_uncentred` columns)
    centred: np.ndarray          # the same minus the average player (the panel's rate columns)
    possessions: np.ndarray      # his possessions on this side over the counted games


@dataclass(frozen=True)
class BoxRates:
    """The padded rates of every player of a design (one row per player column, `spec.players` order)."""
    offence: SideRates
    defence: SideRates
    counts: np.ndarray                 # (players x counts): his box counts over the counted games
    pad_k: np.ndarray                  # (seasons x counts), possessions
    league: np.ndarray                 # (seasons x counts): league rates per 100 offensive possessions
    bin_edges: np.ndarray | None       # the six possession bins, None under the league target
    bin_targets: np.ndarray | None     # (seasons x bins x counts), the targets the rates are padded toward
    group_k: np.ndarray | None = None  # (seasons x counts): the bin targets' own pad k under pad_bin_target league


def _targets(league, edges, targets, player_possessions, season_of, player_of) -> np.ndarray:
    """(player-seasons x counts): each player-season's target, from the bin his possessions in the design put him in,
    or the season's league rate."""
    if edges is None:
        return league[season_of]
    return targets[season_of, bin_of(edges, player_possessions[player_of])]


def box_rates(design, pad_target: str = "poss_conditional", body: pd.DataFrame | None = None,
              pad_k_around: str = "league", pad_bin_target: str = "none") -> BoxRates:
    """The padded rates of every player of `design` (data/designs.py), both sides, from the design's games.

    Per player-season: his counts and possessions summed over the counted games (`game_totals`), each season's league
    rates, its pad k (`pad_k`) and, under the possession-conditional target, each bin's rate (`bin_targets`).  Per
    player: k and the target blended over his seasons by possessions, then `pad_rates`.  Centred: minus a fifth of the
    possession-weighted (row-weighted) mean over the design's rows of the five players' rates summed, so the rates of
    an average lineup add to 0.

    The same padding gives both sides: on offence n is his team's offensive possessions, on defence the defensive ones.
    Under the body target, `body` (indexed by player_id, `BODY`) gives every player's height and weight.
    `pad_k_around` "target" measures the possession-conditional target's pad k around the target (`PAD_K_AROUND`).
    `pad_bin_target` "league" pads each bin's rate toward the league rate by the bin's possessions before it serves as
    the target (`PAD_BIN_TARGETS`, the group pad of the module docstring; the possession-conditional target only).
    """
    if pad_target not in PAD_TARGETS:
        raise ValueError(f"unknown pad_target {pad_target!r}; use one of {PAD_TARGETS}")
    if pad_k_around not in PAD_K_AROUND:
        raise ValueError(f"unknown pad_k_around {pad_k_around!r}; use one of {PAD_K_AROUND}")
    if pad_bin_target not in PAD_BIN_TARGETS:
        raise ValueError(f"unknown pad_bin_target {pad_bin_target!r}; use one of {PAD_BIN_TARGETS}")
    if pad_bin_target == "league" and pad_target != "poss_conditional":
        raise ValueError("pad_bin_target pads the possession-conditional target's bins; the league target has none "
                         "and the body target is not shipped")
    if pad_target == "body_and_possessions" and body is None:
        raise ValueError("the body target needs every player's height and weight")
    spec = design.spec
    player_of = (spec.player_of_player_season if spec.player_of_player_season is not None
                 else np.arange(spec.n_player_seasons))
    season_of = (spec.season_of_player_season if spec.season_of_player_season is not None
                 else spec.season_of_player)
    totals = game_totals(design)
    per_season = _sum_matrix(totals.player_season, spec.n_player_seasons)
    counts = np.asarray(per_season @ totals.counts)
    possessions = {"O": np.asarray(per_season @ totals.offensive_possessions).ravel(),
                   "D": np.asarray(per_season @ totals.defensive_possessions).ravel()}
    per_player = _sum_matrix(player_of, spec.n_players)
    player_counts = np.asarray(per_player @ counts)
    player_possessions = {side: np.asarray(per_player @ values).ravel() for side, values in possessions.items()}
    league = league_rates(counts, possessions["O"], season_of, spec.n_seasons)
    edges = (possession_bins(possessions["O"], player_possessions["O"][player_of])
             if pad_target in ("poss_conditional", "body_and_possessions") else None)
    train_bins = None if edges is None else bin_of(edges, player_possessions["O"][player_of])
    targets = None if edges is None else bin_targets(counts, possessions["O"], season_of, spec.n_seasons, train_bins,
                                                     league)
    group = None
    if pad_bin_target == "league" and edges is not None:
        held = bin_possessions(possessions["O"], season_of, spec.n_seasons, train_bins)
        group = group_k(pad_noise(totals, season_of, spec.n_seasons), targets, held, league)
        targets = padded_bin_targets(targets, held, league, group)
    side_targets = {side: _targets(league, edges, targets, player_possessions[side], season_of, player_of)
                    for side in ("O", "D")}
    k_target = side_targets["O"] if pad_k_around == "target" and edges is not None else None
    if pad_target == "body_and_possessions" and edges is not None:
        player_ids = spec.players.set_index("player_column").player_id.reindex(player_of).to_numpy()
        # a player the body table lacks reads its medians, as the panel's own body columns do (data/bio.py); one
        # missing height would otherwise leave his season's whole line, and every target read off it, missing
        heights = body.reindex(player_ids)[list(BODY)].fillna(body[list(BODY)].median()).to_numpy(float)
        side_targets = {side: body_targets(counts, possessions["O"], season_of, spec.n_seasons, train_bins,
                                           bin_of(edges, player_possessions[side][player_of]), heights, player_ids,
                                           side_targets[side])
                        for side in ("O", "D")}
        k_target = side_targets["O"]
    k = pad_k(totals, season_of, spec.n_seasons, k_target)
    sides = {}
    for side, lineup in zip(("O", "D"), lineups(design)):
        player_k = _blend(k[season_of], possessions[side], player_of, spec.n_players)
        target = _blend(side_targets[side], possessions[side], player_of, spec.n_players)
        rates = pad_rates(player_counts, player_possessions[side], player_k, target)
        centred = rates - _lineup_mean(rates, lineup, np.asarray(design.w, dtype=float)) / LINEUP
        sides[side] = SideRates(rates=rates, centred=centred, possessions=player_possessions[side])
    return BoxRates(offence=sides["O"], defence=sides["D"], counts=player_counts, pad_k=k, league=league,
                    bin_edges=edges, bin_targets=targets, group_k=group)


def rate_table(rates: BoxRates, side: str, names) -> pd.DataFrame:
    """One side's rates as a table: one column per count, `names` (the panel's), then the same `_uncentred`."""
    chosen = rates.offence if side == "O" else rates.defence
    frame = pd.DataFrame(chosen.centred, columns=list(names))
    frame[[f"{name}_uncentred" for name in names]] = chosen.rates
    return frame
