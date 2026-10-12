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
                   measured from his odd- against his even-numbered games (`split_half_k`)
    bin            one of six ranges of possessions, equally wide on a log scale
    target         the typical value a rate is padded toward: that season's rate among the players of his bin (the
                   possession-conditional target, config.yaml's pad_target), or the season's league rate
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

__all__ = ["K_MAX", "K_FALLBACK", "PAD_TARGETS", "shrink", "mom_k", "split_half_k", "GameTotals", "game_totals",
           "league_rates", "pad_k", "possession_bins", "bin_of", "bin_targets", "pad_rates", "lineups", "SideRates",
           "BoxRates", "box_rates", "rate_table"]

K_MAX = 1e5                     # the pad k of a rate that shows no spread between players: the target, all but whole
K_FALLBACK = 40.0               # mom_k's k when too few players carry a proportion to estimate it
MIN_HALF_POSSESSIONS = 20.0     # a player enters split_half_k with at least this many possessions in each half
POSSESSION_BINS = 6
MIN_BIN_PLAYERS = 5             # a season's bin needs this many players to have a rate of its own
PAD_TARGETS = ("league", "poss_conditional")
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


def _season_k(counts_a: np.ndarray, possessions_a: np.ndarray, counts_b: np.ndarray,
              possessions_b: np.ndarray) -> np.ndarray:
    """One season's pad k per count from its players' two half rates r_A and r_B (per 100, over P_A and P_B
    possessions):

        sigma^2_within = mean over players of (r_A - r_B)^2 / (1/P_A + 1/P_B)
        tau^2_between  = sum w (r_A - mean_A)(r_B - mean_B) / sum w,   w = P_A + P_B
        k              = sigma^2_within / tau^2_between,  K_MAX when tau^2 <= 0

    The two halves share a player's true rate and nothing else, so their weighted covariance is the variance of the
    true rates between players; the squared gap between them is pure noise.
    """
    rate_a = 100.0 * counts_a / possessions_a[:, None]
    rate_b = 100.0 * counts_b / possessions_b[:, None]
    inverse = (1.0 / possessions_a + 1.0 / possessions_b)[:, None]
    within = np.mean((rate_a - rate_b) ** 2 / inverse, axis=0)
    weight = (possessions_a + possessions_b)[:, None]
    mean_a = (weight * rate_a).sum(0) / weight.sum()
    mean_b = (weight * rate_b).sum(0) / weight.sum()
    between = ((weight * (rate_a - mean_a) * (rate_b - mean_b)).sum(0) / weight.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(between > 0, within / np.maximum(between, 1e-12), K_MAX)
    return np.minimum(k, K_MAX)


def split_half_k(player: np.ndarray, game: np.ndarray, counts: np.ndarray, possessions: np.ndarray,
                 season_of_player: np.ndarray, n_seasons: int,
                 min_half_possessions: float = MIN_HALF_POSSESSIONS) -> np.ndarray:
    """The pad k of every count in every season, (seasons x counts), in possessions; NaN for a season with fewer than
    three players carrying `min_half_possessions` in both halves.

    `player`, `game`, `counts` and `possessions` are one row per (game, player-season); `season_of_player` gives each
    player-season's season.  `_season_k` has the estimator.

    Example: a count whose true rates spread with variance 0.36 between players and whose per-possession noise variance
    is 162 (a rate of about 1.6 per 100) gets k = 162 / 0.36 = 450 possessions.
    """
    counts_a, possessions_a, counts_b, possessions_b = _odd_even_totals(player, game, counts, possessions,
                                                                        len(season_of_player))
    k = np.full((n_seasons, counts.shape[1]), np.nan)
    for season in range(n_seasons):
        both = ((season_of_player == season) & (possessions_a >= min_half_possessions)
                & (possessions_b >= min_half_possessions))
        if both.sum() >= 3:
            k[season] = _season_k(counts_a[both], possessions_a[both], counts_b[both], possessions_b[both])
    return k


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
    counts = list(design.spec.features)
    possessions = design.game_poss[["game_idx", "psx_idx", "poss_off", "poss_def"]]
    box = design.game_box[["game_idx", "psx_idx", "phase", *counts]]
    box_games = np.unique(box["game_idx"].to_numpy())
    table = possessions.merge(box.drop(columns="phase"), on=["game_idx", "psx_idx"], how="outer")
    table[counts] = table[counts].fillna(0.0)
    table[["poss_off", "poss_def"]] = table[["poss_off", "poss_def"]].fillna(0.0)
    played = np.intersect1d(np.unique(design.rows["game_idx"].to_numpy()), box_games)
    table = table[np.isin(table["game_idx"].to_numpy().astype(np.int64), played)]
    return GameTotals(game=table["game_idx"].to_numpy().astype(np.int64),
                      player_season=table["psx_idx"].to_numpy().astype(np.int64),
                      counts=table[counts].to_numpy(dtype=float),
                      offensive_possessions=table["poss_off"].to_numpy(dtype=float),
                      defensive_possessions=table["poss_def"].to_numpy(dtype=float))


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


def pad_k(totals: GameTotals, season_of: np.ndarray, n_seasons: int) -> np.ndarray:
    """(seasons x counts): `split_half_k` on the offensive possessions, a season without enough players taking the
    mean k of the seasons that have them (all zeros when none has)."""
    k = split_half_k(totals.player_season, totals.game, totals.counts, totals.offensive_possessions, season_of,
                     n_seasons)
    if not np.isfinite(k).any():
        return np.zeros((n_seasons, totals.counts.shape[1]))
    return np.where(np.isnan(k), np.nanmean(k, axis=0, keepdims=True), k)


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
    for columns in (design.spec.zo, design.spec.zd):
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
    """The padded rates of every player of a design (one row per design column, `spec.ps_table` order)."""
    offence: SideRates
    defence: SideRates
    counts: np.ndarray                 # (players x counts): his box counts over the counted games
    pad_k: np.ndarray                  # (seasons x counts), possessions
    league: np.ndarray                 # (seasons x counts): league rates per 100 offensive possessions
    bin_edges: np.ndarray | None       # the six possession bins, None under the league target
    bin_targets: np.ndarray | None     # (seasons x bins x counts)


def _targets(league, edges, targets, player_possessions, season_of, player_of) -> np.ndarray:
    """(player-seasons x counts): each player-season's target, from the bin his possessions in the design put him in,
    or the season's league rate."""
    if edges is None:
        return league[season_of]
    return targets[season_of, bin_of(edges, player_possessions[player_of])]


def box_rates(design, pad_target: str = "poss_conditional") -> BoxRates:
    """The padded rates of every player of `design` (data/designs.py), both sides, from the design's games.

    Per player-season: his counts and possessions summed over the counted games (`game_totals`), each season's league
    rates, its pad k (`pad_k`) and, under the possession-conditional target, each bin's rate (`bin_targets`).  Per
    player: k and the target blended over his seasons by possessions, then `pad_rates`.  Centred: minus a fifth of the
    possession-weighted (row-weighted) mean over the design's rows of the five players' rates summed, so the rates of
    an average lineup add to 0.

    The same padding gives both sides: on offence n is his team's offensive possessions, on defence the defensive ones.
    """
    if pad_target not in PAD_TARGETS:
        raise ValueError(f"unknown pad_target {pad_target!r}; use one of {PAD_TARGETS}")
    spec = design.spec
    player_of = spec.ps_of_psx if spec.ps_of_psx is not None else np.arange(spec.n_psx)
    season_of = spec.season_of_psx if spec.season_of_psx is not None else spec.season_of_ps
    totals = game_totals(design)
    per_season = _sum_matrix(totals.player_season, spec.n_psx)
    counts = np.asarray(per_season @ totals.counts)
    possessions = {"O": np.asarray(per_season @ totals.offensive_possessions).ravel(),
                   "D": np.asarray(per_season @ totals.defensive_possessions).ravel()}
    per_player = _sum_matrix(player_of, spec.n_ps)
    player_counts = np.asarray(per_player @ counts)
    player_possessions = {side: np.asarray(per_player @ values).ravel() for side, values in possessions.items()}
    league = league_rates(counts, possessions["O"], season_of, spec.n_seasons)
    k = pad_k(totals, season_of, spec.n_seasons)
    edges = (possession_bins(possessions["O"], player_possessions["O"][player_of])
             if pad_target == "poss_conditional" else None)
    targets = None if edges is None else bin_targets(counts, possessions["O"], season_of, spec.n_seasons,
                                                     bin_of(edges, player_possessions["O"][player_of]), league)
    sides = {}
    for side, lineup in zip(("O", "D"), lineups(design)):
        player_k = _blend(k[season_of], possessions[side], player_of, spec.n_ps)
        target = _blend(_targets(league, edges, targets, player_possessions[side], season_of, player_of),
                        possessions[side], player_of, spec.n_ps)
        rates = pad_rates(player_counts, player_possessions[side], player_k, target)
        centred = rates - _lineup_mean(rates, lineup, np.asarray(design.w, dtype=float)) / LINEUP
        sides[side] = SideRates(rates=rates, centred=centred, possessions=player_possessions[side])
    return BoxRates(offence=sides["O"], defence=sides["D"], counts=player_counts, pad_k=k, league=league,
                    bin_edges=edges, bin_targets=targets)


def rate_table(rates: BoxRates, side: str, names) -> pd.DataFrame:
    """One side's rates as a table: one column per count, `names` (the panel's), then the same `_uncentred`."""
    chosen = rates.offence if side == "O" else rates.defence
    frame = pd.DataFrame(chosen.centred, columns=list(names))
    frame[[f"{name}_uncentred" for name in names]] = chosen.rates
    return frame
