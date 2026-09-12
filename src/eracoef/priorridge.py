"""PriorRidgeCV: the ratings ridge, with the box prior as its centre and three penalties chosen by CV.

Four things make this different from `sklearn.linear_model.RidgeCV`, and each one was a magic constant:

1. **It shrinks toward the prior, not toward zero.**  A player with no minutes comes out at what the box
   score says he is, not at the league average.
2. **Three penalties, swept independently** -- offense, defense, and the context block (season intercept,
   home, playoff, garbage time, margin, margin-by-time).  The project used to run defense at a fixed
   0.6245 of offense and the context at exactly zero; both are dials with grids around them now.  A
   context penalty of 0 reproduces the old Frisch-Waugh fit, so the previous behaviour is a point on the
   grid rather than an assumption.
3. **The penalty is chosen on whole GAMES.**  The ten players on the floor repeat across a game's stints,
   so splitting inside a game leaks the answer across the fold.
4. **The objective is a team-game error, weighted toward close games.**  A stint MSE is mostly binomial
   noise, so a fold is scored the way a board is used: sum each team's points over its rows in a game,
   compare with the prediction, and weight that team-game by its possessions divided by the game's
   possession-weighted average |margin|.  A 30-point blowout says much less about who is good than a game
   decided by two.  `closeness_floor` keeps 1/x finite.

The number reported is **ARMSE**: the root of the weighted mean square, scaled by sqrt(2 / pi) so it reads
on the mean-absolute scale -- what a typical team-game actually misses by, in points per 100.

    from eracoef.priorridge import PriorRidgeCV

    model = PriorRidgeCV().fit(design, prior_offense, prior_defense)
    print(model.offense_lambda_, model.defense_lambda_, model.context_lambda_)
    print(model.cv_armse_.head())          # every triple tried, best first
    ratings = model.ratings_               # player_id, offense, defense, prior_*, possessions
    Ratings(model.as_ratings_frame())      # the same under holdout's names, for the scorers
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["PriorRidgeCV", "armse", "penalty_grid", "team_game_weights", "team_game_mse",
           "DEFAULT_PLAYER_LAMBDAS", "DEFAULT_CONTEXT_LAMBDAS", "MAE_SCALE"]

# The grids must BRACKET the answer on both sides -- an argmax on a boundary has chosen nothing.  They run
# this high because with a good box prior and three quarters of one season, the team-game objective is
# nearly flat above ~1e5: the residual a lighter penalty would buy is worth less than the noise in it.
DEFAULT_PLAYER_LAMBDAS = np.round(np.logspace(np.log10(500.0), np.log10(2.0e6), 9), 1)
DEFAULT_CONTEXT_LAMBDAS = np.array([0.0, 1.0e3, 1.0e5, 1.0e7])

# A root mean square reads bigger than the typical miss, because squaring pays extra attention to the tail.
# For a normal error the mean ABSOLUTE deviation is sqrt(2 / pi) = 0.7979 of the standard deviation, so
# scaling by it puts the number on the scale people reason about.
MAE_SCALE = 0.7978845608028654


def armse(mean_squared_error):
    """A mean squared error on the mean-absolute scale: sqrt(mse) * sqrt(2 / pi)."""
    return np.sqrt(np.asarray(mean_squared_error, dtype=float)) * MAE_SCALE


def penalty_grid(offense_lambdas=None, defense_lambdas=None, context_lambdas=None) -> list:
    """Every (offense, defense, context) triple.  `defense_lambdas=None` reuses the offense grid."""
    offense = DEFAULT_PLAYER_LAMBDAS if offense_lambdas is None else np.atleast_1d(offense_lambdas)
    defense = offense if defense_lambdas is None else np.atleast_1d(defense_lambdas)
    context = DEFAULT_CONTEXT_LAMBDAS if context_lambdas is None else np.atleast_1d(context_lambdas)
    return [(float(o), float(d), float(c)) for o in offense for d in defense for c in context]


def team_game_weights(design, weight_by_closeness: bool = True, closeness_floor: float = 1.0):
    """(team-game index per row, possessions per row, weight per team-game, average |margin| per game).

    A team-game is one team's rows inside one game -- the unit a board is actually used on.  Its weight is
    its possessions, divided by the game's possession-weighted average |margin| when `weight_by_closeness`
    is on.
    """
    game = design.rows["game_idx"].to_numpy()
    home = design.rows["is_home_off"].to_numpy().astype(np.int64)
    possessions = design.rows["poss"].to_numpy(dtype=float)
    _, key = np.unique(np.stack([game, home]), axis=1, return_inverse=True)

    margin = np.abs(np.asarray(design.X[:, design.spec.f_col("margin")].todense()).ravel())
    game_ids, game_index = np.unique(game, return_inverse=True)
    game_possessions = np.bincount(game_index, weights=possessions, minlength=game_ids.size)
    average_margin = (np.bincount(game_index, weights=possessions * margin, minlength=game_ids.size)
                      / np.maximum(game_possessions, 1e-9))

    n_team_games = int(key.max()) + 1
    weight = np.bincount(key, weights=possessions, minlength=n_team_games)
    if weight_by_closeness:
        closeness = np.zeros(n_team_games)
        closeness[key] = 1.0 / np.maximum(average_margin, float(closeness_floor))[game_index]
        weight = weight * closeness
    return key, possessions, weight, average_margin


def team_game_mse(key, possessions, weight, row_error, rows=None):
    """Row errors in points per 100, pooled into team-games, then a weighted mean square of those."""
    rows = np.arange(row_error.size) if rows is None else rows
    n = weight.size
    points = np.bincount(key[rows], weights=row_error * possessions[rows], minlength=n)
    counted = np.bincount(key[rows], weights=possessions[rows], minlength=n)
    seen = counted > 0
    error = points[seen] / counted[seen]
    w = weight[seen]
    return float((w * error ** 2).sum() / w.sum()), float(w.sum())


def solve_penalised(gram, rhs, sizes, triple):
    """(gram + diag(penalty)) x = rhs, with one penalty per block.  `sizes` is (offense, defense, context)."""
    penalty = np.concatenate([np.full(n, lam) for n, lam in zip(sizes, triple)])
    matrix = gram + np.diag(penalty)
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:              # a zero context penalty can leave that block rank-deficient
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


class PriorRidgeCV:
    """Ridge on a stint design, centred on a per-player prior, three penalties chosen by game-grouped CV.

    offense_lambdas / defense_lambdas / context_lambdas   the grids; None uses the module defaults
    n_folds               CV folds over whole games; 1 skips CV and takes the first triple
    weight_by_closeness   weight each team-game by 1 / average |margin|; False = possessions alone
    closeness_floor       the smallest average margin the weighting will believe, in points
    seed                  fold assignment
    """

    def __init__(self, offense_lambdas=None, defense_lambdas=None, context_lambdas=None, n_folds: int = 5,
                 weight_by_closeness: bool = True, closeness_floor: float = 1.0, seed: int = 0):
        self.grid = penalty_grid(offense_lambdas, defense_lambdas, context_lambdas)
        self.n_folds = int(n_folds)
        self.weight_by_closeness = bool(weight_by_closeness)
        self.closeness_floor = float(closeness_floor)
        self.seed = int(seed)

    # ------------------------------------------------------------------ the pieces of one design
    def _parts(self, design, prior_offense, prior_defense):
        n_players = design.spec.n_ps
        n_fixed = len(design.spec.f_names)
        player_ids = design.spec.ps_table["player_id"].to_numpy()

        def align(prior):
            if prior is None:
                return np.zeros(n_players)
            return pd.Series(prior).reindex(player_ids).fillna(0.0).to_numpy(dtype=float)

        prior = np.concatenate([align(prior_offense), align(prior_defense)])
        players = design.X[:, :2 * n_players].tocsr()
        # the design's own fixed block already carries a season intercept, so no extra ones column
        context = np.asarray(design.X[:, 2 * n_players:2 * n_players + n_fixed].todense())
        return player_ids, players, context, prior

    @staticmethod
    def _blocks(players, context, prior, target, weights, rows):
        """The normal equations of [players | context] on `rows`, with the prior taken out of the target."""
        block = players[rows]
        ctx = context[rows]
        w = weights[rows]
        centred = (target - players @ prior)[rows]

        weighted = block.T.multiply(w)
        gram_pp = np.asarray((weighted @ block).todense())
        gram_pc = np.asarray(weighted @ ctx)
        gram_cc = (ctx * w[:, None]).T @ ctx
        rhs = np.concatenate([np.asarray(weighted @ centred).ravel(), (ctx * w[:, None]).T @ centred])
        return np.block([[gram_pp, gram_pc], [gram_pc.T, gram_cc]]), rhs

    # ------------------------------------------------------------------ fit
    def fit(self, design, prior_offense=None, prior_defense=None):
        player_ids, players, context, prior = self._parts(design, prior_offense, prior_defense)
        n_players, n_context = design.spec.n_ps, context.shape[1]
        sizes = (n_players, n_players, n_context)
        target, weights = design.y, design.w
        games = design.rows["game_idx"].to_numpy()
        key, possessions, team_game_weight, average_margin = team_game_weights(
            design, self.weight_by_closeness, self.closeness_floor)
        self.average_margin_ = average_margin

        if self.n_folds > 1 and np.unique(games).size >= self.n_folds and len(self.grid) > 1:
            unique_games = np.unique(games)
            fold_of_game = np.random.default_rng(self.seed).permutation(unique_games.size) % self.n_folds
            fold = pd.Series(fold_of_game, index=unique_games).reindex(games).to_numpy()
            error = {t: 0.0 for t in self.grid}
            seen = {t: 0.0 for t in self.grid}
            centred = target - players @ prior
            for f in range(self.n_folds):
                train = np.flatnonzero(fold != f)
                valid = np.flatnonzero(fold == f)
                if train.size == 0 or valid.size == 0:
                    continue
                gram, rhs = self._blocks(players, context, prior, target, weights, train)
                for triple in self.grid:
                    coef = solve_penalised(gram, rhs, sizes, triple)
                    gap = (centred[valid] - players[valid] @ coef[:2 * n_players]
                           - context[valid] @ coef[2 * n_players:])
                    mse, w = team_game_mse(key, possessions, team_game_weight, gap, valid)
                    error[triple] += mse * w
                    seen[triple] += w
            table = pd.DataFrame([{"offense_lambda": t[0], "defense_lambda": t[1], "context_lambda": t[2],
                                   "mse": error[t] / seen[t]} for t in self.grid])
            table["armse"] = armse(table.mse.to_numpy())
            self.cv_armse_ = table.sort_values("armse").reset_index(drop=True)
            best = self.grid[int(np.argmin(table.mse.to_numpy()))]
        else:
            self.cv_armse_ = None
            best = self.grid[0]

        self.offense_lambda_, self.defense_lambda_, self.context_lambda_ = best
        gram, rhs = self._blocks(players, context, prior, target, weights, np.arange(target.size))
        coef = solve_penalised(gram, rhs, sizes, best)
        self.residual_ = coef[:2 * n_players]
        self.level_ = coef[2 * n_players:]
        possessions_of = np.asarray(players[:, :n_players]
                                    .multiply(design.rows["poss"].to_numpy()[:, None]).sum(axis=0)).ravel()
        self.ratings_ = pd.DataFrame({
            "player_id": player_ids,
            "offense": prior[:n_players] + self.residual_[:n_players],
            "defense": prior[n_players:] + self.residual_[n_players:],
            "prior_offense": prior[:n_players],
            "prior_defense": prior[n_players:],
            "possessions": possessions_of,
        })
        return self

    # ------------------------------------------------------------------ what the rest of the code wants
    def as_ratings_frame(self) -> pd.DataFrame:
        """The same table under the column names `holdout.Ratings` expects (`o`, `d`, `poss`, raw sign).

        A frame rather than a `Ratings`, because `holdout` is the data layer and this module may not
        import it (`tests/test_layer_boundary.py`).  The caller wraps: `Ratings(model.as_ratings_frame())`.
        """
        table = self.ratings_
        return pd.DataFrame({"player_id": table.player_id, "o": table.offense, "d": table.defense,
                             "poss": table.possessions, "prior_o": table.prior_offense,
                             "prior_d": table.prior_defense})
