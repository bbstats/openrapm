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

Fitting is by normal equations, not by calling a solver once per alpha.  Z is sparse with exactly ten
non-zeros a row, so `Z' W Z` is cheap to form once per fold and every alpha after that is one Cholesky
solve of a (2 n_players) system.  Nine alphas by five folds costs about what two naive fits would.

    from eracoef.priorridge import PriorRidgeCV

    model = PriorRidgeCV().fit(design, prior_offense, prior_defense)
    print(model.alpha_)          # what the season's own games asked for
    ratings = model.ratings_     # player_id, offense, defense, prior_offense, prior_defense, possessions
    Ratings(model.as_ratings_frame())   # the same thing under holdout's names, for the scorers
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["PriorRidgeCV"]

DEFAULT_ALPHAS = np.round(np.logspace(np.log10(300.0), np.log10(60000.0), 11), 1)


def _weighted_least_squares(columns, target, weights):
    left = (columns * weights[:, None]).T
    return np.linalg.lstsq(left @ columns, left @ target, rcond=None)[0]


class PriorRidgeCV:
    """Ridge on a stint design, centred on a per-player prior, penalty chosen by game-grouped CV.

    alphas                  penalties to try; None uses `DEFAULT_ALPHAS`
    defense_penalty_ratio   the defensive block's penalty as a fraction of the offensive one
    n_folds                 CV folds over whole games; 1 skips CV and uses the first alpha
    seed                    fold assignment
    """

    def __init__(self, alphas=None, defense_penalty_ratio: float = 0.624519, n_folds: int = 5, seed: int = 0):
        self.alphas = DEFAULT_ALPHAS if alphas is None else np.asarray(alphas, dtype=float)
        self.defense_penalty_ratio = float(defense_penalty_ratio)
        self.n_folds = int(n_folds)
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

        if self.n_folds > 1 and np.unique(games).size >= self.n_folds:
            unique_games = np.unique(games)
            fold_of_game = np.random.default_rng(self.seed).permutation(unique_games.size) % self.n_folds
            fold = pd.Series(fold_of_game, index=unique_games).reindex(games).to_numpy()
            error = {float(a): 0.0 for a in self.alphas}
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
                    error[alpha] += float(weights[valid] @ (gap ** 2))
            self.cv_error_ = pd.Series(error).sort_index()
            self.alpha_ = float(self.cv_error_.idxmin())
        else:
            self.cv_error_ = None
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
