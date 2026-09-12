"""LeaveSeasonOutRAPM: one rating per PLAYER, from every season except one.

The prior's target used to be `rapm1` for a single player-season -- one season of evidence, heavily shrunk
toward a linear box-score fit, and 97.6% correlated with that fit.  This is the replacement the owner asked
for: for each held-out season H, a genuine RAPM over **all the other seasons**, one rating per player rather
than one per player-season, kept only for players with at least `min_possessions` of them.

Why it is not thirty separate fits.  With player units the design is the same 2 x n_players columns in every
season, so the normal equations are additive over seasons:

    G = sum_s Z_s' W_s Z_s          b = sum_s Z_s' W_s y_s

and the fit that leaves season H out is `G - G_H`, `b - b_H`.  Accumulate each season once, then every
held-out season is one Cholesky solve.  Thirty refits become thirty solves.

The per-season context (intercept, home, playoff, garbage time, margin) is projected out of each season
BEFORE it is accumulated, so every season keeps its own level -- scoring drifts across eras and that is not
what the ratings are for.  By Frisch-Waugh that is a rank-8 correction to each season's gram:

    G_s = Z_s' W_s Z_s - (Z_s' W_s A_s)(A_s' W_s A_s)^-1 (A_s' W_s Z_s)

which costs one 8 x 8 inverse and never densifies Z.

**A possession is one offensive AND one defensive possession**, counted once -- a player on the floor for a
hundred trips has a hundred possessions, not two hundred.  `min_possessions` is applied to that count.

**The lambda is not a detail.**  This target is what the prior learns, so a penalty that is too heavy
teaches the prior to predict a shrunken thing and a penalty that is too light teaches it to predict noise.
`sweep` scores each candidate the way the board is used -- held-out GAMES, team-game error, weighted toward
close ones (`priorridge`) -- rather than by any in-sample number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["LeaveSeasonOutRAPM"]


class LeaveSeasonOutRAPM:
    """Accumulate seasons, then solve for any one of them held out.

    defense_penalty_ratio   the defensive block's penalty as a fraction of the offensive one
    min_possessions         a player needs this many to get a rating (offensive = defensive, counted once)
    """

    def __init__(self, defense_penalty_ratio: float = 0.624519, min_possessions: float = 100.0):
        self.defense_penalty_ratio = float(defense_penalty_ratio)
        self.min_possessions = float(min_possessions)
        self.player_ids = np.empty(0, dtype=np.int64)
        self._index: dict = {}
        self._gram: dict = {}
        self._rhs: dict = {}
        self._possessions: dict = {}

    # ------------------------------------------------------------------ accumulation
    def _slot(self, ids):
        """Global column positions for `ids`, extending the table when a season brings new players."""
        new = [int(i) for i in ids if int(i) not in self._index]
        if new:
            start = len(self._index)
            for offset, pid in enumerate(new):
                self._index[pid] = start + offset
            self.player_ids = np.concatenate([self.player_ids, np.asarray(new, dtype=np.int64)])
        return np.array([self._index[int(i)] for i in ids], dtype=np.int64)

    def add_season(self, season: int, design) -> "LeaveSeasonOutRAPM":
        """Fold one season's normal equations in, with that season's own context projected out first."""
        n_players = design.spec.n_ps
        n_fixed = len(design.spec.f_names)
        slot = self._slot(design.spec.ps_table["player_id"].to_numpy())

        players = design.X[:, :2 * n_players].tocsr()
        context = np.column_stack([np.ones(design.X.shape[0]),
                                   np.asarray(design.X[:, 2 * n_players:2 * n_players + n_fixed].todense())])
        target, weights = design.y, design.w

        weighted = players.T.multiply(weights)
        gram = np.asarray((weighted @ players).todense())
        rhs = np.asarray(weighted @ target).ravel()
        cross = np.asarray((weighted @ context))                       # (2 n_players) x n_context
        # lstsq, not solve: a single-season design already carries its own intercept dummy, so the ones
        # column is redundant and A'WA is singular.  The projection onto col(A) is well defined anyway.
        left = (context * weights[:, None]).T
        inner = left @ context
        gram -= cross @ np.linalg.lstsq(inner, cross.T, rcond=None)[0]
        rhs -= cross @ np.linalg.lstsq(inner, left @ target, rcond=None)[0]

        possessions = np.asarray(players[:, :n_players]
                                 .multiply(design.rows["poss"].to_numpy()[:, None]).sum(axis=0)).ravel()

        self._gram[int(season)] = (slot, gram)
        self._rhs[int(season)] = (slot, rhs)
        self._possessions[int(season)] = (slot, possessions)
        return self

    @property
    def seasons(self) -> list:
        return sorted(self._gram)

    # ------------------------------------------------------------------ solving
    def _totals(self, exclude=None):
        n = len(self._index)
        gram = np.zeros((2 * n, 2 * n))
        rhs = np.zeros(2 * n)
        possessions = np.zeros(n)
        for season in self._gram:
            if exclude is not None and season == int(exclude):
                continue
            slot, block = self._gram[season]
            wide = np.concatenate([slot, slot + n])
            gram[np.ix_(wide, wide)] += block
            rhs[wide] += self._rhs[season][1]
            possessions[slot] += self._possessions[season][1]
        return gram, rhs, possessions

    def ratings(self, held_out_season=None, alpha: float = 5000.0) -> pd.DataFrame:
        """One row per player with at least `min_possessions` over the seasons used.

        `held_out_season=None` uses every accumulated season -- the all-seasons fit, which no prior that
        will be applied to season H may ever see.
        """
        gram, rhs, possessions = self._totals(exclude=held_out_season)
        n = possessions.size
        penalty = np.concatenate([np.ones(n), np.full(n, self.defense_penalty_ratio)]) * float(alpha)
        coefficients = np.linalg.solve(gram + np.diag(penalty), rhs)
        keep = possessions >= self.min_possessions
        return pd.DataFrame({"player_id": self.player_ids[keep],
                             "offense": coefficients[:n][keep],
                             "defense": coefficients[n:][keep],
                             "possessions": possessions[keep]})

    # ------------------------------------------------------------------ choosing the penalty
    def predict_error(self, design, ratings) -> np.ndarray:
        """Actual minus predicted points per 100, per row of `design`, with the level refit on it.

        A player the fit never rated scores 0 -- the average player -- exactly as the criterion does.
        """
        n_players = design.spec.n_ps
        n_fixed = len(design.spec.f_names)
        ids = design.spec.ps_table["player_id"].to_numpy()
        lookup = ratings.set_index("player_id")
        offense = lookup["offense"].reindex(ids).fillna(0.0).to_numpy()
        defense = lookup["defense"].reindex(ids).fillna(0.0).to_numpy()

        players = design.X[:, :2 * n_players].tocsr()
        context = np.column_stack([np.ones(design.X.shape[0]),
                                   np.asarray(design.X[:, 2 * n_players:2 * n_players + n_fixed].todense())])
        target, weights = design.y, design.w
        centred = target - players @ np.concatenate([offense, defense])
        left = (context * weights[:, None]).T
        return centred - context @ np.linalg.lstsq(left @ context, left @ centred, rcond=None)[0]

    def sweep(self, alphas, design_for, held_out_season=None, scoring_seasons=None,
              weight_by_closeness: bool = True, closeness_floor: float = 1.0, verbose: bool = True):
        """Score each penalty on SEASONS THIS RATING HAS NOT SEEN, not on any in-sample number.

        For each scoring season S, fit on every accumulated season except S (and except `held_out_season`,
        which nothing that touches it may ever see), then predict S's games and take the team-game error
        weighted toward close ones.  That is the same objective `PriorRidgeCV` uses, so the penalty that
        builds the target and the penalty that builds the board are chosen by the same question.

        `design_for(season)` returns that season's design; the caller decides whether to cache them.
        Returns a frame indexed by alpha with the pooled ARMSE.
        """
        from .priorridge import armse, team_game_mse, team_game_weights

        scoring = [int(s) for s in (scoring_seasons if scoring_seasons is not None else self.seasons)
                   if held_out_season is None or int(s) != int(held_out_season)]
        total = {float(a): 0.0 for a in alphas}
        seen = {float(a): 0.0 for a in alphas}
        for season in scoring:
            design = design_for(season)
            key, possessions, weight, _ = team_game_weights(design, weight_by_closeness, closeness_floor)
            for alpha in alphas:
                drop = [season] if held_out_season is None else [season, int(held_out_season)]
                rating = self.ratings_excluding(drop, alpha=float(alpha))
                mse, w = team_game_mse(key, possessions, weight, self.predict_error(design, rating))
                total[float(alpha)] += mse * w
                seen[float(alpha)] += w
            if verbose:
                print(f"  swept {season}", flush=True)
        pooled = pd.Series({a: total[a] / seen[a] for a in total}).sort_index()
        return pd.DataFrame({"mse": pooled, "armse": armse(pooled.to_numpy())})

    def ratings_excluding(self, seasons, alpha: float = 5000.0) -> pd.DataFrame:
        """`ratings` with several seasons held out at once -- what the sweep needs."""
        drop = {int(s) for s in seasons}
        n = len(self._index)
        gram = np.zeros((2 * n, 2 * n))
        rhs = np.zeros(2 * n)
        possessions = np.zeros(n)
        for season in self._gram:
            if season in drop:
                continue
            slot, block = self._gram[season]
            wide = np.concatenate([slot, slot + n])
            gram[np.ix_(wide, wide)] += block
            rhs[wide] += self._rhs[season][1]
            possessions[slot] += self._possessions[season][1]
        penalty = np.concatenate([np.ones(n), np.full(n, self.defense_penalty_ratio)]) * float(alpha)
        coefficients = np.linalg.solve(gram + np.diag(penalty), rhs)
        keep = possessions >= self.min_possessions
        return pd.DataFrame({"player_id": self.player_ids[keep], "offense": coefficients[:n][keep],
                             "defense": coefficients[n:][keep], "possessions": possessions[keep]})
