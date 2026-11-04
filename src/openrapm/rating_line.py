"""Between steps 2 and 3: the second prior's weights fitted on the rating itself, instead of on lineup swaps.

The line of second_prior.py is fitted to forecast held-back swaps, with no regard for what step 3 then does: the ridge
reads the same fit games, so an input that restates their lineup results earns weight on the swaps and counts those
games twice in the rating (notes/DECISIONS.md, experiment 128: the team's rates with him on the court, z -5.80 on the
swaps, z +7.27 on the within-season test).  This module fits the weights on the test the ratings are judged by.

Step 3 is a ridge, so a rating is a straight line in its prior, and the test is a least-squares problem in the weights.

Words used below
    fit games, held-back games   one within-season fold's two sets of games (data/within_folds.py)
    Z, C, w                      the fit games' player columns [offence | defence], context columns and possessions
    Z_h                          the same player columns on the held-back stints
    fitted part                  r(t): the players' part of the ridge on [Z | C] fitted to a target t, the penalty on
                                 the players alone
    explained part               X p = r(Z p): what that ridge fits to the lineups' summed priors p
    input                        one number per player, on one end: z_j
    weight                       beta_j, the line's weight on z_j: the prior is p = sum_j beta_j z_j
    within layer                 step 4's multipliers inside a team (prior_shrink.WITHIN_COLUMNS): the prior part times
                                 m, the games' part times g
    team levels                  a free level for each attacking and defending team of the fold, home and playoffs,
                                 taken out of every held-back stint, as the within-season test takes them out

The deviation step 3 adds to a prior p is r(y - Z p) = r(y) - X p, so inside a team after step 4 a rating is

    m p + g (r(y) - X p)

and the prediction of a held-back stint is

    g Z_h r(y) + sum_j beta_j Z_h (m z_j - g X z_j)

Each input becomes one column of the held-back stints, what it still says once the ridge has read the fit games, and
each end's fitted part another.  With M the team levels taken out and W the stints' possessions, the fold's share of the
normal equations over those columns B is

    Q = (M B)' W (M B),   c = (M B)' W M (y - stand-ins),   yy = (M (y - stand-ins))' W M (y - stand-ins),   sw = sum w

so the within-season test's error at any weights beta is (yy - 2 beta' c + beta' Q beta) / sw.  A season's sums add its
folds'.  The fitted parts' coefficients are fitted with the rest and thrown away: they stand where g stands, and step 4
fits its own multipliers afterwards.

Both ends sit in one regression, since one stint holds both, so the line for a rated season solves both ends together on
the seasons it may train on, each column scaled to one standard deviation over those stints before the ridge penalty is
set (`coefficients`).  An input with no spread gets weight 0.  There are no player folds: a stint holds ten players, so
none is left without a given fold's players, and 45 weights over 28 seasons cannot learn one player.

As in second_prior.py, the box-score prior keeps each team's total and every other input only splits a team's credit:
each is moved so a team's possession-weighted total is 0 (`team_free`), so the columns are the ones the line is applied
with (second_prior.hold_team_totals).

Example: one input, m = 0.7, g = 1.0, a player whose z is +1.0 and whose X z is +0.2: the ridge has already read a fifth
of his input in the fit games, so his entry behind the column is 0.7 x 1.0 - 1.0 x 0.2 = +0.5.  If z is a result of the
fit games, r(y) holds it too, and the regression gives z's column whatever is left beyond r(y)'s, which may be nothing.

The fit reproduces the pipeline: on the folds of 30 seasons the test's error from these sums, with the line's own prior,
misses check 151's by a mean +0.008, standard deviation 0.042 (notes/DECISIONS.md, "The direct screen").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .rapm import normal_equations, solve_ridge

__all__ = ["RatingSums", "HeldBack", "FITTED_PARTS", "ridge_parts", "team_free", "prior_columns", "team_levels",
           "held_back_sums", "coefficients"]

FITTED_PARTS = 2                 # the columns before the inputs: each end's fitted part
NO_SPREAD = 1e-14                # a column's possession-weighted mean square at or below this has no spread


@dataclass
class RatingSums:
    """One fold's or season's share of the normal equations (the module docstring), over the columns [offence's fitted
    part, defence's fitted part, offence's inputs, defence's inputs]."""
    Q: np.ndarray
    c: np.ndarray
    yy: float
    sw: float

    def __add__(self, other: "RatingSums") -> "RatingSums":
        return RatingSums(self.Q + other.Q, self.c + other.c, self.yy + other.yy, self.sw + other.sw)

    def error(self, weights: np.ndarray) -> float:
        """The within-season test's error at `weights`, one per column."""
        return float((self.yy - 2.0 * weights @ self.c + weights @ self.Q @ weights) / self.sw)


@dataclass
class HeldBack:
    """One fold's held-back stints, rated by its fit games' ratings."""
    players: sp.csr_matrix       # Z_h: stints x [offence | defence] player columns, the fit games' column order
    y: np.ndarray                # points per 100 as scored
    w: np.ndarray                # possessions
    levels: sp.csr_matrix        # stints x the fold's team levels (`team_levels`)
    rated: np.ndarray            # one per player column: the fit games rated him
    stand_in: tuple              # (offence, defence): the rating an unrated player stands in at, as check 151 reads it


def ridge_parts(players, context: np.ndarray, weights: np.ndarray, offence_target: np.ndarray,
                defence_target: np.ndarray, priors: np.ndarray, penalty: float) -> tuple:
    """(r(offence target)'s offence half, r(defence target)'s defence half, X priors): one fold's ridge on [Z | C] for
    two targets and any number of priors at once.  `priors` is (2 n x columns), [offence | defence] in Z's column order.
    Step 3 fits each target on its own and keeps that end, so each fitted part reads its own target, and X is the same
    for both since only the target differs."""
    n_columns = players.shape[1]
    sums = normal_equations(players, context, offence_target, weights)
    weighted, weighted_context = players.T.multiply(weights), (context * weights[:, None]).T
    offence = np.concatenate([sums.rhs_players, sums.rhs_context])
    defence = np.concatenate([np.asarray(weighted @ defence_target).ravel(), weighted_context @ defence_target])
    gram = np.block([[sums.gram_players, sums.gram_players_context],
                     [sums.gram_players_context.T, sums.gram_context]])
    right = np.vstack([sums.gram_players @ priors, sums.gram_players_context.T @ priors])
    penalties = np.concatenate([np.full(n_columns, float(penalty)), np.zeros(context.shape[1])])
    solved = solve_ridge(gram, np.column_stack([offence, defence, right]), penalties)
    half = n_columns // 2
    return solved[:half, 0], solved[half:n_columns, 1], solved[:n_columns, 2:]


def team_free(values: np.ndarray, player_ids: np.ndarray, team_poss: pd.DataFrame) -> tuple:
    """(the ids of `team_poss`, `values` moved so every team's possession-weighted total is 0, one row per id):
    second_prior.hold_team_totals on many columns at once, with an anchor of 0.  `values` is (players x columns), a row
    per id of `player_ids`; a player of `team_poss` without a row reads 0 before he is moved.  A player of two teams
    moves by his mix of their amounts.

    Example: one team, two players of 600 and 400 possessions at +1.0 and 0.0: the mean is 0.6, so +0.4 and -0.6.
    """
    ids, player = np.unique(team_poss.player_id.to_numpy(dtype=np.int64), return_inverse=True)
    teams, team = np.unique(team_poss.team.to_numpy(dtype=np.int64), return_inverse=True)
    possessions = np.zeros((len(teams), len(ids)))
    np.add.at(possessions, (np.asarray(team).ravel(), np.asarray(player).ravel()), team_poss.poss.to_numpy(float))
    total = possessions.sum(axis=0)
    mix = (possessions / np.where(total > 0, total, 1.0)).T
    at = pd.Series(np.arange(len(player_ids)), index=player_ids).reindex(ids)
    own = np.zeros((len(ids), values.shape[1]))
    found = at.notna().to_numpy()
    own[found] = values[at[found].to_numpy(dtype=int)]
    return ids, own - mix @ np.linalg.solve(possessions @ mix, possessions @ own)


def prior_columns(inputs: dict, player_ids: np.ndarray, team_poss: dict, box_column: int = 0) -> np.ndarray:
    """(2 n x 2 K): every input as a prior, offence's K then defence's, each on its own half of [offence | defence] in
    the order of `player_ids`, the design's.  The input at `box_column`, the box-score prior, is centred on the end's
    possessions and keeps each team's total; the others are `team_free` (the module docstring).  A player without inputs
    reads 0.

    `inputs` is {"O", "D"}: player_id -> K inputs; `team_poss` {"O", "D"}: second_prior.team_possessions."""
    n_players, n_inputs = len(player_ids), inputs["O"].shape[1]
    out = np.zeros((2 * n_players, 2 * n_inputs))
    position = pd.Series(np.arange(n_players), index=player_ids)
    for half, side in enumerate(("O", "D")):
        ids = inputs[side].index.to_numpy(dtype=np.int64)
        values = inputs[side].to_numpy(dtype=float)
        moved_ids, moved = team_free(values, ids, team_poss[side])
        rows = position.reindex(moved_ids).to_numpy()
        found = ~np.isnan(rows)
        out[n_players * half + rows[found].astype(int), n_inputs * half:n_inputs * (half + 1)] = moved[found]
        weight = team_poss[side].groupby("player_id").poss.sum().reindex(ids).fillna(0.0).to_numpy(float)
        box = values[:, box_column] - np.average(values[:, box_column], weights=weight)
        rows = position.reindex(ids).to_numpy()
        found = ~np.isnan(rows)
        out[:, n_inputs * half + box_column] = 0.0
        out[n_players * half + rows[found].astype(int), n_inputs * half + box_column] = box[found]
    return out


def team_levels(team_off, team_def, is_home_off, is_playoffs) -> sp.csr_matrix:
    """Stints x a free level for every attacking team and every defending team, home and playoffs, as the within-season
    test takes them out (scripts/checks/151_wowy.py)."""
    off_codes, _ = pd.factorize(np.asarray(team_off))
    def_codes, _ = pd.factorize(np.asarray(team_def))
    n_rows = len(off_codes)
    rows = np.arange(n_rows)
    return sp.hstack([
        sp.csr_matrix((np.ones(n_rows), (rows, off_codes)), shape=(n_rows, off_codes.max() + 1)),
        sp.csr_matrix((np.ones(n_rows), (rows, def_codes)), shape=(n_rows, def_codes.max() + 1))[:, 1:],
        sp.csr_matrix(np.asarray(is_home_off, dtype=float)[:, None]),
        sp.csr_matrix(np.asarray(is_playoffs, dtype=float)[:, None])]).tocsr()


def held_back_sums(held: HeldBack, fitted: tuple, priors: np.ndarray, explained: np.ndarray,
                   within: tuple) -> RatingSums:
    """One fold's `RatingSums`.  `fitted` is (offence's, defence's) fitted part and `priors`, `explained` `ridge_parts`'
    priors and X priors; `within` is (m_O, g_O, m_D, g_D), prior_shrink.WITHIN_COLUMNS' order.  An unrated player
    counts at his stand-in, as in the test, and enters no column.  The within layer's factors are held at the season's:
    only a line's weights are free."""
    keep = held.w > 0
    players, y, w, levels = held.players[keep], held.y[keep], held.w[keep], held.levels[keep]
    half = players.shape[1] // 2
    rated = sp.diags(held.rated.astype(float))
    unrated = players @ sp.diags((~held.rated).astype(float))
    stand_in = (np.asarray(unrated[:, :half].sum(axis=1)).ravel() * held.stand_in[0]
                + np.asarray(unrated[:, half:].sum(axis=1)).ravel() * held.stand_in[1])
    on_court = players @ rated
    offence, defence = on_court[:, :half], on_court[:, half:]
    m_off, g_off, m_def, g_def = within
    columns = np.column_stack([offence @ fitted[0], defence @ fitted[1],
                               offence @ (m_off * priors[:half] - g_off * explained[:half])
                               + defence @ (m_def * priors[half:] - g_def * explained[half:])])
    root = np.sqrt(w)
    basis, spread, _ = np.linalg.svd(levels.multiply(root[:, None]).toarray(), full_matrices=False)
    basis = basis[:, spread > 1e-10 * spread.max()]
    matrix = columns * root[:, None]
    matrix -= basis @ (basis.T @ matrix)
    aim = (y - stand_in) * root
    aim -= basis @ (basis.T @ aim)
    return RatingSums(Q=matrix.T @ matrix, c=matrix.T @ aim, yy=float(aim @ aim), sw=float(w.sum()))


def coefficients(parts: list, n_folds: int, penalty: float) -> dict:
    """{"O", "D"}: the rated season's line, an array (n_folds + 1, K) like second_prior.season_coefficients', every row
    the same (the module docstring: no player folds), from both ends solved together on `parts`, the seasons it may
    train on.  Each column is scaled to one standard deviation over those stints; lambda = `penalty` x the mean
    diagonal, so the penalty means the same at any number of seasons and on any input's unit.

    Example: a column with no spread gets weight 0, and leaves the others as they would be without it."""
    total = sum(parts[1:], parts[0])
    mean_square = np.diag(total.Q) / total.sw
    has_spread = mean_square > NO_SPREAD
    scale = np.where(has_spread, 1.0 / np.sqrt(np.where(has_spread, mean_square, 1.0)), 0.0)
    gram = total.Q * np.outer(scale, scale)
    ridge = float(penalty) * float(np.mean(np.diag(gram)))
    weights = np.linalg.solve(gram + ridge * np.eye(len(scale)), total.c * scale) * scale
    n_inputs = (len(scale) - FITTED_PARTS) // 2
    offence, defence = weights[FITTED_PARTS:FITTED_PARTS + n_inputs], weights[FITTED_PARTS + n_inputs:]
    return {"O": np.tile(offence, (int(n_folds) + 1, 1)), "D": np.tile(defence, (int(n_folds) + 1, 1))}
