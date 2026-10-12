"""Step 1 of 5: plain RAPM, and the label the box-score prior learns from.

RAPM (regularised adjusted plus-minus) asks of every stint: given the ten players on the court, how many points
per 100 possessions did the offence score?  Five players share every possession, so one stint cannot say who
was good; thousands of stints with different lineups can, once a ridge penalty keeps the answer from chasing
noise.

Words used below
    stint       a stretch of play with the same ten players on the court
    row         one stint seen from one side: five players on offence, five on defence
    target      what the ridge explains for each row, y = 100 x adjusted points / possessions; the row's weight
                w is its possessions.  One adjustment per side ships (settings.Targets):
                  offence, "points_with_expected_free_throws": points - made free throws + expected free
                      throws, each free throw counted at the shooter's padded free-throw %, that game's own
                      attempts left out
                  defence, "points_with_expected_opponent_threes": the same free-throw adjustment, and three
                      quarters of the shooters' three-point luck taken off: points - 3 x 0.75 x (threes made -
                      threes expected), threes expected = attempts x the shooter's three-point % from the other
                      half of the season's games, padded toward the league by 450 attempts; then one number per
                      season rescales the target so the season's total equals the points actually scored
                      (data/shots.align)
    design      one column per player per side (offence, defence) plus the context columns
    context     per-season intercepts, home court, playoffs, playoff home court, garbage time, score margin
                (clipped at +-25), margin x share of the game left
    raw impact  the coefficient an unpenalised fit would give a player
    penalty     the ridge's lambda.  Seen alone, a player with n possessions keeps about n / (n + lambda) of
                his raw impact
    rated season    the season whose ratings are being built
    unseen seasons  the seasons a label may not use: the rated one, and in the test setting its two
                neighbours as well
    label       a player's RAPM from one ridge fit over every season except the unseen ones: ONE coefficient
                per player per side for all those seasons together, his career level, assumed constant from
                season to season.  Step 2's prior learns to predict it

The fit, for one set of seasons:

    minimise   sum over rows  w (y - B theta)^2  +  lambda_O |theta_O|^2  +  lambda_D |theta_D|^2
    solution   (G + Lambda) theta = b,     G = B' W B,   b = B' W y

    B          the design matrix, one line per row: [Z_O | Z_D | C]
    Z_O, Z_D   the player columns: Z_O[row, j] = 1 when player j is one of the five on offence, Z_D the same
               for the five on defence; Z = [Z_O | Z_D]
    C          the context columns
    y, w       each row's target and weight (its possessions); W the diagonal matrix of the weights
    theta      the coefficients: theta_O on Z_O, theta_D on Z_D, then gamma on C
    lambda_O, lambda_D   the offence and defence penalties (40,000 each); the context is not penalised
    Lambda     the diagonal matrix of per-column penalties: lambda_O, lambda_D, 0 (`block_penalty`)
    G, b       the Gram matrix and right-hand side of the weighted least-squares normal equations

G and b are sums over rows, so they add up over seasons.  `SeasonSums` stores each season's share once; the fit
that leaves the rated season out is then one solve on the sums of the other seasons, not thirty refits.  Each
season keeps its own context columns, so leaving a season out drops its rows and its context together.

Example: with lambda = 40,000, a player with 54,000 career possessions keeps 57% of his raw impact
(54,000 / 94,000) and a player with 2,600 keeps 6%.

The defensive label is "un-shrunk" (`unshrink`) before the prior learns from it, so long and short careers sit on
one scale; the offensive label is not.  Un-shrinking both sides helped the year-over-year test (paired difference
in team-game mean squared error -0.84, z -3.77) but made the order within each side worse (each side rescaled:
+1.79, z +7.75, better in 8 of 56) and cut the consensus agreement from 0.771 to 0.680.  Un-shrinking defence alone
passed every check (-0.42, z -5.24, better in 43 of 56; notes/DECISIONS.md, experiment 22).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

__all__ = ["design_blocks", "NormalEquations", "normal_equations", "player_possessions", "block_penalty",
           "solve_ridge", "SeasonSums", "unshrink", "career_labels"]


def design_blocks(design):
    """Split a season's design matrix into (Z as CSR sparse, C as a dense array).

    Z = [Z_O | Z_D] has 2 x n columns, n = the season's players (`design.spec.n_players`); C is the next
    `len(design.spec.context_names)` columns.  The matrix's last column holds each row's game number, used to group
    rows by game, and is not a regressor.
    """
    n_players = design.spec.n_players
    n_context = len(design.spec.context_names)
    players = design.X[:, :2 * n_players].tocsr()
    context = np.asarray(design.X[:, 2 * n_players:2 * n_players + n_context].todense())
    return players, context


class NormalEquations(NamedTuple):
    """G and b of one design, kept in blocks so seasons can be added up:

        G = [[Z'WZ, Z'WC], [C'WZ, C'WC]],   b = [Z'Wy, C'Wy]
    """
    gram_players: np.ndarray           # Z'WZ
    gram_players_context: np.ndarray   # Z'WC
    gram_context: np.ndarray           # C'WC
    rhs_players: np.ndarray            # Z'Wy
    rhs_context: np.ndarray            # C'Wy


def normal_equations(players, context, target, weights) -> NormalEquations:
    """The weighted least-squares sums of one design: G = B'WB, b = B'Wy, block by block.

    The arithmetic, and the order of every sum, is the one every published rating was built with, so the bits
    match: change the order and the published ratings move in their last digits.
    """
    weighted = players.T.multiply(weights)
    weighted_context = (context * weights[:, None]).T
    return NormalEquations(gram_players=np.asarray((weighted @ players).todense()),
                           gram_players_context=np.asarray(weighted @ context),
                           gram_context=weighted_context @ context,
                           rhs_players=np.asarray(weighted @ target).ravel(),
                           rhs_context=weighted_context @ target)


def player_possessions(players, possessions, n_players: int) -> np.ndarray:
    """Each player's possessions in the design: the possessions of the rows he is on offence in.

    Only one side is counted.  His defensive possessions are within a few of the offensive ones, and counting
    both would credit every player with about twice the trips he was on the court for.
    """
    return np.asarray(players[:, :n_players].multiply(possessions[:, None]).sum(axis=0)).ravel()


def block_penalty(sizes, penalties) -> np.ndarray:
    """One penalty per column from one penalty per block.

    Example: block sizes (3, 3, 2) with penalties (40000, 40000, 0) give three columns at 40,000 for offence,
    three at 40,000 for defence and two context columns at 0.
    """
    return np.concatenate([np.full(size, penalty) for size, penalty in zip(sizes, penalties)])


def solve_ridge(gram, rhs, penalty) -> np.ndarray:
    """Solve (G + diag(penalty)) theta = b, one penalty per column.

    An unpenalised context column can be all zeros (the 2020 bubble's playoff home court) and then the plain
    solve raises.  The retry adds a jitter of 1e-9 x the mean diagonal to the UNPENALISED diagonal only, which
    leaves the player block untouched; least squares is the last resort.
    """
    penalty = np.asarray(penalty, dtype=float)
    matrix = gram + np.diag(penalty)
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        jitter = 1e-9 * max(float(np.mean(np.abs(np.diag(matrix)))), 1.0)
        matrix[np.diag_indices_from(matrix)] += np.where(penalty == 0.0, jitter, 0.0)
        try:
            return np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


class SeasonSums:
    """Each season's normal equations, stored once; any set of seasons can then be left out and solved.

    Columns: [offence of every player seen | defence of every player seen | each season's context].  A player's
    column is fixed the first time a season brings him, so seasons must be added in ascending order for the
    columns (and so the bits of every solve) to match the published ratings.
    """

    def __init__(self, min_possessions: float = 100.0):
        self.min_possessions = float(min_possessions)
        self.player_ids = np.empty(0, dtype=np.int64)
        self._column: dict = {}
        self._seasons: dict = {}

    @property
    def seasons(self) -> list:
        """The seasons added so far, ascending."""
        return sorted(self._seasons)

    def _columns_for(self, ids) -> np.ndarray:
        """Each player's column, giving a new column to a player seen for the first time."""
        new = [int(pid) for pid in ids if int(pid) not in self._column]
        if new:
            start = len(self._column)
            for offset, pid in enumerate(new):
                self._column[pid] = start + offset
            self.player_ids = np.concatenate([self.player_ids, np.asarray(new, dtype=np.int64)])
        return np.array([self._column[int(pid)] for pid in ids], dtype=np.int64)

    def add(self, season: int, design) -> "SeasonSums":
        """Store one season's normal equations and each player's possessions in it."""
        players, context = design_blocks(design)
        self._seasons[int(season)] = dict(
            columns=self._columns_for(design.spec.players["player_id"].to_numpy()),
            n_context=context.shape[1],
            sums=normal_equations(players, context, design.y, design.w),
            possessions=player_possessions(players, design.rows["poss"].to_numpy(), design.spec.n_players))
        return self

    def player_columns(self, season: int) -> np.ndarray:
        """The column of each of a season's players, in the order of that season's design.

        `openrapm site --decomposition` reads this and `context_size` to place each season inside `assemble`'s G.
        """
        return self._seasons[int(season)]["columns"]

    def context_size(self, season: int) -> int:
        """How many context columns a season brought."""
        return self._seasons[int(season)]["n_context"]

    def _context_starts(self, kept: list) -> tuple[dict, int]:
        """({season: its first context column}, total columns): the context blocks follow the 2 x players."""
        starts, total = {}, 2 * len(self._column)
        for season in kept:
            starts[season] = total
            total += self._seasons[season]["n_context"]
        return starts, total

    def assemble(self, unseen=()):
        """G, b, each player's possessions and the number of context columns, over the seasons not in `unseen`.

        The context columns follow the 2 x players in ascending season order.
        """
        dropped = {int(season) for season in unseen}
        kept = [season for season in self.seasons if season not in dropped]
        n_players = len(self._column)
        starts, total = self._context_starts(kept)
        gram = np.zeros((total, total))
        rhs = np.zeros(total)
        possessions = np.zeros(n_players)
        for season in kept:
            block = self._seasons[season]
            sums, columns = block["sums"], block["columns"]
            both_sides = np.concatenate([columns, columns + n_players])
            first = starts[season]
            stop = first + block["n_context"]
            gram[np.ix_(both_sides, both_sides)] += sums.gram_players
            gram[np.ix_(both_sides, np.arange(first, stop))] += sums.gram_players_context
            gram[np.ix_(np.arange(first, stop), both_sides)] += sums.gram_players_context.T
            gram[first:stop, first:stop] += sums.gram_context
            rhs[both_sides] += sums.rhs_players
            rhs[first:stop] += sums.rhs_context
            possessions[columns] += block["possessions"]
        return gram, rhs, possessions, total - 2 * n_players

    def ratings(self, unseen=(), penalties=(40000.0, 40000.0, 0.0)) -> pd.DataFrame:
        """One row per player with at least `min_possessions` in the seasons used (every season not in `unseen`).

        penalties = (offence, defence, context).  Columns: player_id, offense, defense (raw sign: defence is
        points allowed, lower is better), possessions.
        """
        gram, rhs, possessions, n_context = self.assemble(unseen)
        n_players = possessions.size
        sizes = (n_players, n_players, n_context)
        coefficients = solve_ridge(gram, rhs, block_penalty(sizes, [float(value) for value in penalties]))
        keep = possessions >= self.min_possessions
        return pd.DataFrame({"player_id": self.player_ids[keep],
                             "offense": coefficients[:n_players][keep],
                             "defense": coefficients[n_players:2 * n_players][keep],
                             "possessions": possessions[keep]})


def unshrink(ratings: pd.DataFrame, column: str, penalty: float, full_trust_possessions: float = 4444.0,
             tier_edges=(0.0, 500.0, 1500.0, 4444.0)) -> pd.DataFrame:
    """Put every player's label on one scale, and pull a player with few possessions toward his tier's level.

        label = beta / max(s, s_cap)  +  (1 - s / max(s, s_cap)) x m(tier)

        beta    the player's ridge coefficient in `column`
        n       his possessions
        lambda  the ridge penalty the coefficient was fitted with
        s       = n / (n + lambda), the share of his raw impact the ridge kept
        s_cap   = N / (N + lambda), N = full_trust_possessions
        m(tier) = mean beta / mean s over the players in his possession tier: the tier's level, un-shrunk

    Above N possessions the label is the raw-scale coefficient beta / s.  Below it, dividing by s would
    multiply noise by hundreds, so the division stops at s_cap and the rest comes from the player's tier:
    possessions in [0, 500), [500, 1500), [1500, 4444) or 4444 and more.

    Example: lambda = 40,000, so s_cap = 4,444 / 44,444 = 0.10.  A 1,000-possession player (s = 0.024) with
    beta = 0.05 in a tier whose level is -3.0 gets 0.05 / 0.10 + (1 - 0.24) x -3.0 = -1.77.
    """
    out = ratings.copy()
    possessions = out.possessions.to_numpy(float)
    tier = np.digitize(possessions, list(tier_edges[1:]))
    beta = out[column].to_numpy(float)
    kept = possessions / (possessions + penalty)
    kept_capped = np.maximum(kept, full_trust_possessions / (full_trust_possessions + penalty))
    tier_level = np.zeros(len(tier_edges))
    for tier_index in range(len(tier_level)):
        members = tier == tier_index
        if members.any():
            tier_level[tier_index] = beta[members].mean() / kept[members].mean()
    out[column] = beta / kept_capped + (1.0 - kept / kept_capped) * tier_level[tier]
    return out


def career_labels(sums: dict, unseen, label_settings, targets) -> dict:
    """{"O": offence labels, "D": defence labels}: each side's RAPM over every season not in `unseen`.

    `sums` maps a target name to its `SeasonSums`; `label_settings` is `settings.LabelSettings` and `targets`
    `settings.Targets`.  The offence label is the `offense` column of the fit on the offence target, the defence
    label the `defense` column of the fit on the defence target.  A column named in
    `label_settings.unshrink_columns` (shipped: defence only; notes/DECISIONS.md, experiment 22) is un-shrunk.

    Example: rating 2026 in the product setting, unseen = [2026]; each table holds player_id, offense, defense
    and possessions for every player with 100+ possessions in 1997-2025.
    """
    penalties = (label_settings.offense_penalty, label_settings.defense_penalty, label_settings.context_penalty)
    out = {}
    for side, column, target, penalty in (("O", "offense", targets.offence, label_settings.offense_penalty),
                                          ("D", "defense", targets.defence, label_settings.defense_penalty)):
        table = sums[target].ratings(unseen, penalties)
        if column in label_settings.unshrink_columns:
            table = unshrink(table, column, penalty, label_settings.full_trust_possessions,
                             label_settings.tier_edges)
        out[side] = table
    return out
