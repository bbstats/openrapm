"""The similarity page: which other player-seasons a player-season is most like.

The owner's design (2026-10-12): every player-season in the pool is described by the same inputs, each input is put
on the same footing by standardising it, and two player-seasons are as similar as their standardised inputs are near.
Each player-season's list holds the nearest seasons of OTHER players: every season of his own is left out, because a
player's own seasons are the nearest to each other.  A classifier with one class per player-season, read for a
Shepard-Luce similarity, was tried first and found a player's own seasons less well (notes/DECISIONS.md, "The
similarity page: nearest neighbours, after a classifier").

Words used below
    player-season   one player in one season (NBA id x season; regular season and playoffs together)
    pool            the player-seasons with at least a minimum of offensive possessions (1,000, the ratings page's
                    minimum; commands/site.py), numbered in the pool's order (season, then player id)
    inputs          what describes a player-season (`similarity_inputs`): every input the box-score prior reads for a
                    rated season (both sides' lists, features.FEATURE_SETS), his possessions on each side, the
                    season's games' part on each side (u_off, u_def: how far step 3 moved him from the prior) and the
                    final rating (offence, defence, total)
    standardised    an input minus its pool mean, over its pool standard deviation, so every input counts the same:
                    a hundred possessions and a point of true shooting are each measured in their spread
    distance        the straight-line distance between two player-seasons' standardised inputs, below
    the two lists   for each row, the nearest rows of other players ("seasons"), and the nearest other players, each
                    at his nearest row ("players")

    z[r, i]  = (x[r, i] - mean of input i) / (standard deviation of input i)
    d(r, t)  = sqrt( sum over inputs i of (z[r, i] - z[t, i])^2 )

Example: with two inputs, standardised, season A at (1.0, 0.0), season B at (0.4, 0.8) and season C at (-1.0, 0.5),
d(A, B) = sqrt(0.6^2 + 0.8^2) = 1.0 and d(A, C) = sqrt(2.0^2 + 0.5^2) = 2.06: season B is the nearer one to A.

Model layer: everything here takes frames and arrays.  commands/site.py loads the tables and writes the files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .box_prior import rated_rows

__all__ = ["KEYS", "TABLE_INPUTS", "TABLE_COLUMNS", "player_seasons", "side_inputs", "similarity_inputs",
           "standardise", "distances", "closest", "own_season_ranks"]

KEYS = ["player_id", "season"]
# what the inputs take from the ratings table: the season's games' part and the final rating, each side
TABLE_INPUTS = ["u_off", "u_def", "rating_off", "rating_def", "rating_total"]
TABLE_COLUMNS = [*KEYS, "player_name", "poss_off", "poss_def", *TABLE_INPUTS]


def player_seasons(table: pd.DataFrame, min_possessions: float, seasons) -> pd.DataFrame:
    """The pool: the ratings table's rows with at least `min_possessions` offensive possessions in `seasons`,
    sorted by season and then player id, so that a row's position is how the lists name it.  Columns
    `TABLE_COLUMNS`."""
    missing = [column for column in TABLE_COLUMNS if column not in table.columns]
    if missing:
        raise KeyError(f"the ratings table lacks {missing}")
    keep = (table.poss_off >= min_possessions) & table.season.isin(list(seasons))
    rows = table.loc[keep, TABLE_COLUMNS].sort_values(["season", "player_id"], kind="stable").reset_index(drop=True)
    if rows.duplicated(KEYS).any():
        raise ValueError("the ratings table holds a player-season twice")
    return rows


def side_inputs(panel: pd.DataFrame, side: str, names) -> pd.DataFrame:
    """One side's inputs exactly as the prior reads a rated season (`box_prior.rated_rows`), for every season at once:
    player_id, season, the side's list `names` and row_possessions (his possessions on that side).

    Every derived column is computed within its own row, so all seasons at once equal one season at a time.
    """
    rows = panel[(panel.side == side) & (panel.possessions > 0)]
    return rated_rows(rows, names)[[*KEYS, *names, "row_possessions"]]


def _side_rows(rows: pd.DataFrame, panel: pd.DataFrame, side: str, names) -> pd.DataFrame:
    """The side's inputs for the pool's rows, in their order; raises if a pool row has no panel row on that side."""
    out = rows[KEYS].merge(side_inputs(panel, side, names), on=KEYS, how="left", validate="1:1")
    lacking = out.row_possessions.isna()
    if lacking.any():
        first = out.loc[lacking, KEYS].iloc[0].tolist()
        raise ValueError(f"{int(lacking.sum())} pool rows have no {side} panel row with possessions, e.g. {first}")
    return out


def _paired_columns(offence: pd.DataFrame, defence: pd.DataFrame, offence_names, defence_names) -> dict:
    """The box-score inputs, named: the offence list, then the defence list's other names.  An input on both lists
    that is equal on every row is read once under its own name; one that differs is read from each side, as
    `<name>_off` in the offence list's place and `<name>_def` in the defence list's."""
    differs = {name for name in offence_names if name in defence_names
               and not np.array_equal(offence[name].to_numpy(float), defence[name].to_numpy(float))}
    columns = {}
    for name in offence_names:
        columns[f"{name}_off" if name in differs else name] = offence[name].to_numpy(float)
    for name in defence_names:
        if name in differs:
            columns[f"{name}_def"] = defence[name].to_numpy(float)
        elif name not in offence_names:
            columns[name] = defence[name].to_numpy(float)
    return columns


def similarity_inputs(rows: pd.DataFrame, panel: pd.DataFrame, feature_set: dict) -> pd.DataFrame:
    """The inputs, one row per pool row in the pool's order.

    In this order: the box-score inputs (`_paired_columns` on the two lists of `feature_set`, the prior's
    {"O": [...], "D": [...]}), his possessions on each side (poss_off and poss_def: the rated row's row_possessions;
    its row_seasons is always 1 and is left out), then `TABLE_INPUTS` from the pool's rows.  With the shipped lists
    that is 27 + 2 + 5 = 34 columns.  Raises if any value is missing.
    """
    offence = _side_rows(rows, panel, "O", feature_set["O"])
    defence = _side_rows(rows, panel, "D", feature_set["D"])
    columns = _paired_columns(offence, defence, feature_set["O"], feature_set["D"])
    columns["poss_off"] = offence.row_possessions.to_numpy(float)
    columns["poss_def"] = defence.row_possessions.to_numpy(float)
    for name in TABLE_INPUTS:
        columns[name] = rows[name].to_numpy(float)
    out = pd.DataFrame(columns)
    if out.isna().any().any():
        raise ValueError(f"missing inputs in {out.columns[out.isna().any()].tolist()}")
    return out


def standardise(inputs: pd.DataFrame) -> np.ndarray:
    """z: each input minus its pool mean, over its pool standard deviation (an input with no spread is left at 0)."""
    values = inputs.to_numpy(float)
    spread = values.std(axis=0)
    return (values - values.mean(axis=0)) / np.where(spread > 0, spread, 1.0)


def distances(z: np.ndarray, rows: slice) -> np.ndarray:
    """d of the rows `rows` to every row, one row of the result per row asked for.

    The squared differences are summed input by input, in the same order for every pair, so d(r, t) and d(t, r) come
    out identical to the last bit, whichever block each is computed in.
    """
    total = np.zeros((rows.stop - rows.start, len(z)))
    for column in range(z.shape[1]):
        total += (z[rows, column][:, None] - z[None, :, column]) ** 2
    return np.sqrt(total)


def _blocks(n: int, block_rows: int):
    """slice(0, block_rows), slice(block_rows, 2 x block_rows), ... up to n."""
    return (slice(start, min(start + block_rows, n)) for start in range(0, n, block_rows))


def _without_own(block: np.ndarray, players: np.ndarray, rows: slice) -> np.ndarray:
    """The block with every row of each row's own player, itself included, set to infinity."""
    return np.where(players[rows][:, None] == players[None, :], np.inf, block)


def _nearest_per_player(block: np.ndarray, order: np.ndarray, starts: np.ndarray) -> tuple:
    """Per row of the block: each player's smallest distance, and the first row (in the pool's order) that has it.

    `order` sorts the pool's rows by player, keeping the pool's order within a player; `starts` is where each
    player's run of rows begins in it."""
    by_player = block[:, order]
    best = np.minimum.reduceat(by_player, starts, axis=1)
    reaches = by_player == np.repeat(best, np.diff(np.r_[starts, len(order)]), axis=1)
    first = np.minimum.reduceat(np.where(reaches, order[None, :], len(order)), starts, axis=1)
    return best, first


def _rank_players(best: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Each row's players, nearest first; on a tie, the player whose nearest row comes first in the pool."""
    by_row = np.argsort(first, axis=1, kind="stable")
    by_best = np.argsort(np.take_along_axis(best, by_row, axis=1), axis=1, kind="stable")
    return np.take_along_axis(by_row, by_best, axis=1)


def closest(distance, players, matches: int, block_rows: int = 1024) -> dict:
    """The two lists of every row, as row positions and distances, each one row per pool row by `matches`:

        seasons, season_distances   the nearest rows of other players, nearest first
        players, player_distances   the nearest other players, each at his nearest row (positions are rows)

    `distance(rows)` gives the rows `rows` against every row (for the page, `partial(distances, z)`); `players` is
    each row's player id.  Every row of a row's own player, itself included, is left out.  Ties go to the row that
    comes first in the pool (an earlier season, then a lower id), and among players to the one whose nearest row
    comes first.
    """
    players = np.asarray(players)
    order = np.argsort(players, kind="stable")
    starts = np.flatnonzero(np.r_[True, players[order][1:] != players[order][:-1]])
    parts = {key: [] for key in ("seasons", "season_distances", "players", "player_distances")}
    for rows in _blocks(len(players), block_rows):
        block = _without_own(distance(rows), players, rows)
        seasons = np.argsort(block, axis=1, kind="stable")[:, :matches]
        parts["seasons"].append(seasons)
        parts["season_distances"].append(np.take_along_axis(block, seasons, axis=1))
        best, first = _nearest_per_player(block, order, starts)
        ranked = _rank_players(best, first)[:, :matches]
        parts["players"].append(np.take_along_axis(first, ranked, axis=1))
        parts["player_distances"].append(np.take_along_axis(best, ranked, axis=1))
    return {key: np.vstack(blocks) for key, blocks in parts.items()}


def own_season_ranks(distance, players, block_rows: int = 1024) -> np.ndarray:
    """For each row, where his nearest other season ranks among every other row:

        rank = 1 + the number of other players' rows nearer to him than his nearest other season

    so 1 means no other player's season comes before one of his own; ties count in his favour.  NaN for a row whose
    player has no other season in the pool.  `distance` and `players` as in `closest`.
    """
    players = np.asarray(players)
    n = len(players)
    ranks = np.full(n, np.nan)
    for rows in _blocks(n, block_rows):
        block = distance(rows)
        own = players[rows][:, None] == players[None, :]
        itself = np.arange(rows.start, rows.stop)[:, None] == np.arange(n)[None, :]
        best_own = np.where(own & ~itself, block, np.inf).min(axis=1)
        nearer = (np.where(own, np.inf, block) < best_own[:, None]).sum(axis=1)
        ranks[rows] = np.where(np.isfinite(best_own), 1.0 + nearer, np.nan)
    return ranks
