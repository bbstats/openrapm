"""PriorRidgeCV: the ratings ridge, with the box prior as its centre and the penalty chosen by CV.

Three things make this different from `sklearn.linear_model.RidgeCV`, and each one is a thing that was
a magic constant before:

1. **It shrinks toward the prior, not toward zero.**  A player with no minutes comes out at what the box
   score says he is, not at the league average.
2. **The fixed effects are not penalised.**  Intercept, home, playoff, garbage time and the margin terms
   are context, not attribution; sklearn would shrink them along with the players.  They are projected
   out first (Frisch-Waugh), so the penalty lands only on the 2 x n_players block.
3. **Offense and defense get different penalties.**  One `alpha`, times `defense_penalty_ratio` on the
   defensive block -- the project has always run defense at ~0.62 of offense.

The penalty itself is picked by cross-validation over whole GAMES, not over stints: the ten players on
the floor repeat across a game's stints, so splitting inside a game leaks the answer across the fold.

**The objective is a team-game error, weighted toward close games.**  A stint-level MSE is mostly binomial
noise, so a fold is scored the way the board is used: sum each team's points over its rows in a game,
compare that with what the ratings predicted, and weight the team-game by its possessions TIMES
`1 / average |margin| over the game`.  A 30-point blowout says much less about who is good than a game
decided by two, and garbage time is where the ratings are least identified anyway.  The average margin is
possession-weighted across the game's stints and clipped the way the design clips it (+/- 25 points), so a
40-point win and a 30-point win are both simply blowouts.  `closeness_floor` keeps 1/x finite.

The number reported is **ARMSE**, not MSE: the root of the weighted mean square, scaled by sqrt(2 / pi) so
it reads on the mean-absolute scale -- what a typical team-game actually misses by.

Fitting is by normal equations, not by calling a solver once per alpha.  Z is sparse with exactly ten
non-zeros a row, so `Z' W Z` is cheap to form once per fold and every alpha after that is one Cholesky
solve of a (2 n_players) system.  Nine alphas by five folds costs about what two naive fits would.

    from eracoef.priorridge import PriorRidgeCV

    model = PriorRidgeCV().fit(design, prior_offense, prior_defense)
    print(model.alpha_)          # what the season's own games asked for
    print(model.cv_armse_)       # the fold error per alpha, on the mean-absolute scale
    ratings = model.ratings_     # player_id, offense, defense, prior_offense, prior_defense, possessions
    Ratings(model.as_ratings_frame())   # the same thing under holdout's names, for the scorers
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["PriorRidgeCV"]

# The grid must BRACKET the answer on both sides -- an argmax on a boundary has chosen nothing.  It runs
# this high because with a good box prior and three quarters of one season of games, the team-game
# objective is nearly flat above ~1e5: the residual it would buy is worth less than the noise in it.
DEFAULT_ALPHAS = np.round(np.logspace(np.log10(300.0), np.log10(2.0e7), 25), 1)

# A root mean square reads bigger than the typical miss, because squaring pays extra attention to the tail.
# For a normal error the mean ABSOLUTE deviation is sqrt(2 / pi) = 0.7979 of the standard deviation, so
# scaling by it puts the number on the scale people reason about: "a typical team-game misses by this much".
MAE_SCALE = 0.7978845608028654


def armse(mean_squared_error):
    """A mean squared error on the mean-absolute scale: sqrt(mse) * sqrt(2 / pi)."""
    return np.sqrt(np.asarray(mean_squared_error, dtype=float)) * MAE_SCALE


def team_game_weights(design, weight_by_closeness: bool = True, closeness_floor: float = 1.0):
    """(team-game index per row, possessions per row, weight per team-game, average |margin| per game).

    A team-game is one team's rows inside one game -- the unit a board is actually used on.  Its weight is
    its possessions, divided by the game's possession-weighted average |margin| when `weight_by_closeness`
    is on, because a 30-point blowout says much less about who is good than a game decided by two.
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


def _weighted_least_squares(columns, target, weights):
    left = (columns * weights[:, None]).T
    return np.linalg.lstsq(left @ columns, left @ target, rcond=None)[0]


class PriorRidgeCV:
    """Ridge on a stint design, centred on a per-player prior, penalty chosen by game-grouped CV.

    alphas                  penalties to try; None uses `DEFAULT_ALPHAS`
    defense_penalty_ratio   the defensive block's penalty as a fraction of the offensive one
    n_folds                 CV folds over whole games; 1 skips CV and uses the first alpha
    weight_by_closeness     weight each team-game by 1 / average |margin|; False = possessions alone
    closeness_floor         the smallest average margin the weighting will believe, in points
    seed                    fold assignment
    """

    def __init__(self, alphas=None, defense_penalty_ratio: float = 0.624519, n_folds: int = 5,
                 weight_by_closeness: bool = True, closeness_floor: float = 1.0, seed: int = 0):
        self.alphas = DEFAULT_ALPHAS if alphas is None else np.asarray(alphas, dtype=float)
        self.defense_penalty_ratio = float(defense_penalty_ratio)
        self.n_folds = int(n_folds)
        self.weight_by_closeness = bool(weight_by_closeness)
        self.closeness_floor = float(closeness_floor)
        self.seed = int(seed)

    # ------------------------------------------------------------------ the team-game objective
    def _team_games(self, design):
        return team_game_weights(design, self.weight_by_closeness, self.closeness_floor)

    _weighted_mse = staticmethod(team_game_mse)

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
        fixed = np.asarray(design.X[:, 2 * n_players:2 * n_players + n_fixed].todense())
        context = np.column_stack([np.ones(design.X.shape[0]), fixed])
        penalty = np.concatenate([np.ones(n_players), np.full(n_players, self.defense_penalty_ratio)])
        return player_ids, players, context, prior, penalty

    @staticmethod
    def _solve(players, context, prior, penalty, target, weights, rows, alphas):
        """Coefficients per alpha on `rows`, with the context refit on those rows first."""
        centred = target - players @ prior
        level = _weighted_least_squares(context[rows], centred[rows], weights[rows])
        residual = centred[rows] - context[rows] @ level
        block = players[rows]
        weighted = block.T.multiply(weights[rows])
        gram = np.asarray((weighted @ block).todense())
        rhs = np.asarray(weighted @ residual).ravel()
        out = {}
        for alpha in alphas:
            out[float(alpha)] = np.linalg.solve(gram + np.diag(alpha * penalty), rhs)
        return level, out

    # ------------------------------------------------------------------ fit
    def fit(self, design, prior_offense=None, prior_defense=None):
        player_ids, players, context, prior, penalty = self._parts(design, prior_offense, prior_defense)
        target, weights = design.y, design.w
        n_rows = target.size
        games = design.rows["game_idx"].to_numpy()
        key, possessions, team_game_weight, average_margin = self._team_games(design)
        self.average_margin_ = average_margin

        if self.n_folds > 1 and np.unique(games).size >= self.n_folds:
            unique_games = np.unique(games)
            fold_of_game = np.random.default_rng(self.seed).permutation(unique_games.size) % self.n_folds
            fold = pd.Series(fold_of_game, index=unique_games).reindex(games).to_numpy()
            error = {float(a): 0.0 for a in self.alphas}
            seen = {float(a): 0.0 for a in self.alphas}
            for f in range(self.n_folds):
                train = np.flatnonzero(fold != f)
                valid = np.flatnonzero(fold == f)
                if train.size == 0 or valid.size == 0:
                    continue
                level, coefs = self._solve(players, context, prior, penalty, target, weights,
                                           train, self.alphas)
                centred_valid = (target - players @ prior)[valid] - context[valid] @ level
                block_valid = players[valid]
                for alpha, coef in coefs.items():
                    gap = centred_valid - block_valid @ coef
                    fold_mse, fold_weight = self._weighted_mse(key, possessions, team_game_weight,
                                                               gap, valid)
                    error[alpha] += fold_mse * fold_weight
                    seen[alpha] += fold_weight
            self.cv_error_ = pd.Series({a: error[a] / seen[a] for a in error}).sort_index()
            self.cv_armse_ = pd.Series(armse(self.cv_error_.to_numpy()), index=self.cv_error_.index)
            self.alpha_ = float(self.cv_error_.idxmin())
        else:
            self.cv_error_ = self.cv_armse_ = None
            self.alpha_ = float(self.alphas[0])

        level, coefs = self._solve(players, context, prior, penalty, target, weights,
                                   np.arange(n_rows), [self.alpha_])
        residual = coefs[self.alpha_]
        n_players = design.spec.n_ps
        possessions = np.asarray(players[:, :n_players]
                                 .multiply(design.rows["poss"].to_numpy()[:, None]).sum(axis=0)).ravel()
        self.level_ = level
        self.residual_ = residual
        self.ratings_ = pd.DataFrame({
            "player_id": player_ids,
            "offense": prior[:n_players] + residual[:n_players],
            "defense": prior[n_players:] + residual[n_players:],
            "prior_offense": prior[:n_players],
            "prior_defense": prior[n_players:],
            "possessions": possessions,
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
