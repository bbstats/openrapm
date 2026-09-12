"""LeaveSeasonOutRAPM: one rating per PLAYER, from every season except one.

The prior's target used to be `rapm1` for a single player-season -- one season of evidence, heavily shrunk
toward a linear box-score fit, and 97.6% correlated with that fit.  This is the replacement: for each
held-out season H, a genuine RAPM over **all the other seasons**, one rating per player rather than one per
player-season, kept only for players with at least `min_possessions` of them.

Why it is not thirty separate fits.  With player units the design is the same 2 x n_players columns in every
season, so the normal equations are additive over seasons:

    G = sum_s B_s' W_s B_s          b = sum_s B_s' W_s y_s

and the fit that leaves season H out is `G - G_H`, `b - b_H`.  Accumulate each season once, then every
held-out season is one solve.  Thirty refits become thirty solves, about 0.7s each at 5,960 columns.

**Each season keeps its own context columns.**  Intercept, home, playoff, garbage time, margin and the
margin-by-time term are per season -- scoring drifts across eras and that is not what the ratings are for --
so the global design is `[2 x n_players | season 1997's context | ... | season 2026's context]`.  They are
carried rather than projected out because the context penalty is a DIAL now, swept like the other two; a
context penalty of zero reproduces the Frisch-Waugh fit exactly.  Leaving season H out drops H's context
columns along with its rows, so nothing is left rank-deficient by a column of zeros.

**Three penalties, all swept**: offense, defense, context.  The project used to run defense at a fixed
0.6245 of offense and the context at zero; neither is assumed here.

**A possession is one offensive AND one defensive possession**, counted once -- a player on the floor for a
hundred trips has a hundred possessions, not two hundred.  `min_possessions` is applied to that count.

**The penalty is not a detail.**  This target is what the prior learns, so too heavy and the prior learns to
predict a shrunken thing, too light and it learns noise.  `sweep` scores each triple the way the board is
scored -- fit without season S, predict S's games, team-game error weighted toward close games
(`priorridge`) -- never by an in-sample number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .priorridge import armse, penalty_grid, solve_penalised, team_game_mse, team_game_weights

__all__ = ["LeaveSeasonOutRAPM"]


class LeaveSeasonOutRAPM:
    """Accumulate seasons, then solve for any subset of them held out.

    min_possessions   a player needs this many to get a rating (offensive = defensive, counted once)
    """

    def __init__(self, min_possessions: float = 100.0):
        self.min_possessions = float(min_possessions)
        self.player_ids = np.empty(0, dtype=np.int64)
        self._index: dict = {}
        self._season: dict = {}

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
        """Fold one season's normal equations in, keeping its context columns as its own."""
        n_players = design.spec.n_ps
        n_fixed = len(design.spec.f_names)
        slot = self._slot(design.spec.ps_table["player_id"].to_numpy())

        players = design.X[:, :2 * n_players].tocsr()
        context = np.asarray(design.X[:, 2 * n_players:2 * n_players + n_fixed].todense())
        target, weights = design.y, design.w

        weighted = players.T.multiply(weights)
        gram_pp = np.asarray((weighted @ players).todense())
        gram_pc = np.asarray(weighted @ context)
        gram_cc = (context * weights[:, None]).T @ context
        rhs_p = np.asarray(weighted @ target).ravel()
        rhs_c = (context * weights[:, None]).T @ target
        possessions = np.asarray(players[:, :n_players]
                                 .multiply(design.rows["poss"].to_numpy()[:, None]).sum(axis=0)).ravel()

        self._season[int(season)] = dict(slot=slot, n_context=n_fixed, gram_pp=gram_pp, gram_pc=gram_pc,
                                         gram_cc=gram_cc, rhs_p=rhs_p, rhs_c=rhs_c, possessions=possessions)
        return self

    @property
    def seasons(self) -> list:
        return sorted(self._season)

    # ------------------------------------------------------------------ solving
    def _assemble(self, exclude=()):
        """The global normal equations over the seasons not in `exclude`, plus their context columns."""
        drop = {int(s) for s in np.atleast_1d(exclude)} if len(np.atleast_1d(exclude)) else set()
        keep = [s for s in self.seasons if s not in drop]
        n = len(self._index)
        offsets, total = {}, 2 * n
        for s in keep:
            offsets[s] = total
            total += self._season[s]["n_context"]

        gram = np.zeros((total, total))
        rhs = np.zeros(total)
        possessions = np.zeros(n)
        for s in keep:
            block = self._season[s]
            slot = block["slot"]
            wide = np.concatenate([slot, slot + n])
            c0 = offsets[s]
            c1 = c0 + block["n_context"]
            gram[np.ix_(wide, wide)] += block["gram_pp"]
            gram[np.ix_(wide, np.arange(c0, c1))] += block["gram_pc"]
            gram[np.ix_(np.arange(c0, c1), wide)] += block["gram_pc"].T
            gram[c0:c1, c0:c1] += block["gram_cc"]
            rhs[wide] += block["rhs_p"]
            rhs[c0:c1] += block["rhs_c"]
            possessions[slot] += block["possessions"]
        return gram, rhs, possessions, total - 2 * n

    def ratings(self, held_out_season=None, offense_lambda: float = 43089.0,
                defense_lambda: float | None = None, context_lambda: float = 0.0) -> pd.DataFrame:
        """One row per player with at least `min_possessions` over the seasons used.

        `held_out_season=None` uses every accumulated season.  `defense_lambda=None` matches offense.
        """
        exclude = () if held_out_season is None else np.atleast_1d(held_out_season)
        gram, rhs, possessions, n_context = self._assemble(exclude)
        n = possessions.size
        triple = (float(offense_lambda),
                  float(offense_lambda if defense_lambda is None else defense_lambda),
                  float(context_lambda))
        coefficients = solve_penalised(gram, rhs, (n, n, n_context), triple)
        keep = possessions >= self.min_possessions
        return pd.DataFrame({"player_id": self.player_ids[keep],
                             "offense": coefficients[:n][keep],
                             "defense": coefficients[n:2 * n][keep],
                             "possessions": possessions[keep]})

    # ------------------------------------------------------------------ choosing the penalties
    def predict_error(self, design, ratings) -> np.ndarray:
        """Actual minus predicted points per 100, per row of `design`, with the context refit on it.

        A player the fit never rated scores 0 -- the average player -- exactly as the criterion does.
        """
        n_players = design.spec.n_ps
        n_fixed = len(design.spec.f_names)
        ids = design.spec.ps_table["player_id"].to_numpy()
        lookup = ratings.set_index("player_id")
        offense = lookup["offense"].reindex(ids).fillna(0.0).to_numpy()
        defense = lookup["defense"].reindex(ids).fillna(0.0).to_numpy()

        players = design.X[:, :2 * n_players].tocsr()
        context = np.asarray(design.X[:, 2 * n_players:2 * n_players + n_fixed].todense())
        target, weights = design.y, design.w
        centred = target - players @ np.concatenate([offense, defense])
        left = (context * weights[:, None]).T
        return centred - context @ np.linalg.lstsq(left @ context, left @ centred, rcond=None)[0]

    def sweep(self, design_for, offense_lambdas=None, defense_lambdas=None, context_lambdas=None,
              held_out_season=None, scoring_seasons=None, weight_by_closeness: bool = True,
              closeness_floor: float = 1.0, verbose: bool = True) -> pd.DataFrame:
        """Score each (offense, defense, context) triple on SEASONS THE FIT HAS NOT SEEN.

        For each scoring season S, fit on every accumulated season except S -- and except
        `held_out_season`, which nothing that touches it may ever see -- then predict S's games and take
        the team-game error weighted toward close ones.  Same objective as `PriorRidgeCV`, so the penalties
        that build the target and the penalties that build the board answer the same question.

        `design_for(season)` returns that season's design; the caller decides whether to cache them.
        Returns a frame sorted best first.
        """
        grid = penalty_grid(offense_lambdas, defense_lambdas, context_lambdas)
        scoring = [int(s) for s in (scoring_seasons if scoring_seasons is not None else self.seasons)
                   if held_out_season is None or int(s) != int(held_out_season)]
        total = {t: 0.0 for t in grid}
        seen = {t: 0.0 for t in grid}
        for season in scoring:
            design = design_for(season)
            key, possessions, weight, _ = team_game_weights(design, weight_by_closeness, closeness_floor)
            drop = [season] if held_out_season is None else [season, int(held_out_season)]
            gram, rhs, player_possessions, n_context = self._assemble(drop)
            n = player_possessions.size
            keep = player_possessions >= self.min_possessions
            for triple in grid:
                coefficients = solve_penalised(gram, rhs, (n, n, n_context), triple)
                rating = pd.DataFrame({"player_id": self.player_ids[keep],
                                       "offense": coefficients[:n][keep],
                                       "defense": coefficients[n:2 * n][keep]})
                mse, w = team_game_mse(key, possessions, weight, self.predict_error(design, rating))
                total[triple] += mse * w
                seen[triple] += w
            if verbose:
                print(f"  swept {season} ({len(grid)} triples)", flush=True)
        table = pd.DataFrame([{"offense_lambda": t[0], "defense_lambda": t[1], "context_lambda": t[2],
                               "mse": total[t] / seen[t]} for t in grid])
        table["armse"] = armse(table.mse.to_numpy())
        return table.sort_values("armse").reset_index(drop=True)
