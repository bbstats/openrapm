"""The swap adjustment: re-split each team's credit by what its lineup swaps say PI-RAPM missed.

The owner's design (2026-10-01): rank players among their own team, as an adjustment on top of PI-RAPM, from
"player vs replacement in lineups" -- the same four teammates with him and with someone else.  It is judged
on the swap test (swaptest.py), on the swaps of the seasons BEFORE and AFTER the rated one.

Every swap in the rated season leaves a RESIDUAL: the difference between the two lineups' results that
PI-RAPM's own prediction did not account for (the ten players' ratings plus the season's level, home,
playoffs, garbage time, score margin, time in the period and closing minutes).  If PI-RAPM had the two
swapped players' gap right, the residual would be noise.  Two parts read those residuals, in this order:

1. **By player type.**  Across many seasons, what kinds of players does PI-RAPM give too much or too little
   credit compared with the teammates they swap with?  A linear model of each swap's residual on the
   DIFFERENCE between the two swapped players' features (box-score rates, share of team possessions, share
   of games started, age, size, PI-RAPM's own prior and residual).  Learned from other players -- five
   player folds, a player's correction from the fold that never saw his swaps -- and from seasons at least
   two away from the rated one, so no player's own seasons reach his correction (ruling 2) and neither
   scored season reaches the test.
2. **By player.**  His own swaps in his own season, which ruling 1 allows (that season's games), pulled
   toward the type prediction by a penalty: a short record moves him little.

**Each team's total stays exactly where PI-RAPM put it**: the corrections of a team's players sum to zero,
weighted by the possessions each played for that team.  Only the split inside a team moves.

Everything here is handed its data (the model layer, tests/test_layer_boundary.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .swaptest import Season, _side_rates, aligned, predict

__all__ = ["SIDES", "residual_pairs", "team_possessions", "PairGrams", "pair_grams", "type_coefficients",
           "SwapSystem", "swap_system", "solve", "corrections", "hold_spread"]

SIDES = ("offense", "defense")


def residual_pairs(season: Season, ratings: pd.DataFrame, context: str = "nofatigue") -> dict:
    """side -> DataFrame(team, fifth_a, fifth_b, h, r): every swap pair of the season, its information `h`
    in possessions, and the swap difference the ratings did not predict, `r` = observed minus predicted, in
    points per 100 and raw sign (on defense, points ALLOWED: a positive `r` says fifth_a's lineup allowed
    more than predicted, so fifth_a is the worse defender of the two than the ratings say)."""
    o, d = aligned(season.ids, ratings)
    pred, _ = predict(season, o, d, context)
    rates = _side_rates(season, pred, o, d)
    out = {}
    for side in SIDES:
        obs, prd, P, _ = rates[side]
        pr = season.pairs[side]
        a, b = pr.lineup_a.to_numpy(), pr.lineup_b.to_numpy()
        out[side] = pd.DataFrame({"team": pr.team.to_numpy(), "fifth_a": pr.fifth_a.to_numpy(),
                                  "fifth_b": pr.fifth_b.to_numpy(), "h": P[a] * P[b] / (P[a] + P[b]),
                                  "r": (obs[a] - obs[b]) - (prd[a] - prd[b])})
    return out


def team_possessions(season: Season) -> dict:
    """side -> DataFrame(player_id, team, poss): the possessions each player played for each team on that end."""
    out = {}
    for side in SIDES:
        t = season.tables[side]
        long = pd.DataFrame({"player_id": t[["p1", "p2", "p3", "p4", "p5"]].to_numpy().ravel(),
                             "team": np.repeat(t.team.to_numpy(), 5), "poss": np.repeat(t.poss.to_numpy(), 5)})
        out[side] = long.groupby(["player_id", "team"], as_index=False).poss.sum()
    return out


@dataclass
class PairGrams:
    """One season-side's contribution to the type model's normal equations.

    G, b         sum over its pairs of h * dx dx' and h * r * dx, where dx is the swapped players' feature
                 difference (fifth_a minus fifth_b)
    G_fold, b_fold  the same over the pairs that involve a player of each fold -- what that fold's model must
                 leave out
    n            pairs used (both players had features)
    """
    G: np.ndarray
    b: np.ndarray
    G_fold: np.ndarray
    b_fold: np.ndarray
    n: int


def pair_grams(pairs: pd.DataFrame, features: pd.DataFrame, fold_of: pd.Series, folds: int) -> PairGrams:
    """`features`: one row per player_id of this season, the type model's columns, standardised.
    `fold_of`: player_id -> fold.  A pair is used only if both players have features."""
    xa = features.reindex(pairs.fifth_a.to_numpy()).to_numpy(dtype=float)
    xb = features.reindex(pairs.fifth_b.to_numpy()).to_numpy(dtype=float)
    ok = ~(np.isnan(xa).any(axis=1) | np.isnan(xb).any(axis=1))
    dx, h, r = (xa - xb)[ok], pairs.h.to_numpy(dtype=float)[ok], pairs.r.to_numpy(dtype=float)[ok]
    fa = fold_of.reindex(pairs.fifth_a.to_numpy()).to_numpy()[ok]
    fb = fold_of.reindex(pairs.fifth_b.to_numpy()).to_numpy()[ok]
    p = dx.shape[1]
    G_fold, b_fold = np.zeros((folds, p, p)), np.zeros((folds, p))
    for k in range(folds):
        m = (fa == k) | (fb == k)
        G_fold[k] = (dx[m] * h[m, None]).T @ dx[m]
        b_fold[k] = dx[m].T @ (h[m] * r[m])
    return PairGrams(G=(dx * h[:, None]).T @ dx, b=dx.T @ (h * r), G_fold=G_fold, b_fold=b_fold, n=int(ok.sum()))


def type_coefficients(grams: list, fold: int | None = None, ridge: float = 1e-4) -> np.ndarray:
    """The type model's coefficients from a set of seasons' grams, leaving out `fold`'s pairs (None: all).

    `ridge` is relative: it is multiplied by the mean diagonal of the summed gram, so it means the same at
    any number of seasons."""
    G = sum(g.G - (g.G_fold[fold] if fold is not None else 0.0) for g in grams)
    b = sum(g.b - (g.b_fold[fold] if fold is not None else 0.0) for g in grams)
    lam = float(ridge) * float(np.mean(np.diag(G)))
    return np.linalg.solve(G + lam * np.eye(len(b)), b)


@dataclass
class SwapSystem:
    """One season-side's swap equations, built once and solved at any penalty and any type prediction.

    ids  every player with possessions or a swap; A  (teams x players) each team's possession shares;
    G, g  sum over pairs of h * e e' and h * r * e, where e is +1 at fifth_a and -1 at fifth_b."""
    ids: np.ndarray
    A: np.ndarray
    G: np.ndarray
    g: np.ndarray


def swap_system(pairs: pd.DataFrame, team_poss: pd.DataFrame) -> SwapSystem:
    ids = np.unique(np.concatenate([team_poss.player_id.to_numpy(dtype=np.int64),
                                    pairs.fifth_a.to_numpy(dtype=np.int64), pairs.fifth_b.to_numpy(dtype=np.int64)]))
    n = len(ids)
    teams, team_idx = np.unique(team_poss.team.to_numpy(), return_inverse=True)
    A = np.zeros((len(teams), n))
    np.add.at(A, (np.asarray(team_idx).ravel(), np.searchsorted(ids, team_poss.player_id.to_numpy(dtype=np.int64))),
              team_poss.poss.to_numpy(dtype=float))
    A /= A.sum(axis=1, keepdims=True)                # possession shares: the same constraint, better scaled
    a = np.searchsorted(ids, pairs.fifth_a.to_numpy(dtype=np.int64))
    b = np.searchsorted(ids, pairs.fifth_b.to_numpy(dtype=np.int64))
    k = len(pairs)
    D = sp.csr_matrix((np.r_[np.ones(k), -np.ones(k)], (np.r_[np.arange(k), np.arange(k)], np.r_[a, b])), shape=(k, n))
    h = pairs.h.to_numpy(dtype=float)
    G = np.asarray((D.T @ sp.diags(h) @ D).todense())
    g = np.asarray(D.T @ (h * pairs.r.to_numpy(dtype=float))).ravel()
    return SwapSystem(ids=ids, A=A, G=G, g=g)


def solve(system: SwapSystem, mean: pd.Series, tau: float) -> pd.Series:
    """`corrections` on equations already built."""
    ids, A = system.ids, system.A
    m = mean.reindex(ids).fillna(0.0).to_numpy(dtype=float)
    if np.isinf(tau):
        return pd.Series(m - A.T @ np.linalg.solve(A @ A.T, A @ m), index=ids)
    n, t = len(ids), A.shape[0]
    kkt = np.block([[system.G + float(tau) * np.eye(n), A.T], [A, np.zeros((t, t))]])
    sol = np.linalg.solve(kkt, np.r_[system.g + float(tau) * m, np.zeros(t)])
    return pd.Series(sol[:n], index=ids)


def corrections(pairs: pd.DataFrame, team_poss: pd.DataFrame, mean: pd.Series, tau: float) -> pd.Series:
    """player_id -> the correction to his rating on this side, raw sign.

        argmin  sum over pairs  h * (r - (c_a - c_b))^2  +  tau * sum over players (c - mean)^2
        subject to, for every team,  sum over its players of (possessions for that team) * c = 0

    `mean` is the type prediction (missing = 0).  `tau` = inf gives the nearest point to the type prediction
    with every team's total at zero -- each player gives back in proportion to his share of the team's
    possessions; `tau` near 0 lets his own swaps speak almost unshrunk.  Players with possessions in
    `team_poss` all get a row, with or without a swap.
    """
    return solve(swap_system(pairs, team_poss), mean, tau)


def hold_spread(adjusted: np.ndarray, base: np.ndarray, weight: np.ndarray, team: np.ndarray,
                how: str = "within") -> tuple[np.ndarray, float]:
    """One season and side: the adjusted ratings with their spread held to the base's, and the factor used.

    The adjustment moves credit inside teams, so the list comes out wider than the base (on 1997-2026, about 12%
    on offense and 27% on defense), and the year-over-year test charges for width whatever the order.  Two ways
    to give the width back, each one factor per season and side, weighted by possessions:

    `within`  shrink each player's distance from his team's mean -- the adjusted ratings' own team mean, so the
              team means, and with them the team totals, do not move -- until the spread INSIDE teams equals
              the base's.  The order inside every team is the adjustment's, untouched.  A player with no team
              (NaN) keeps his adjusted number.
    `whole`   scale the whole list about its mean until its spread equals the base's.  The order on each side is
              the adjustment's, untouched; team totals shrink with everything else.
    """
    adjusted, base, weight = (np.asarray(x, dtype=float) for x in (adjusted, base, weight))
    w = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)

    def spread(x, centre):
        return float(np.sum(w * (x - centre) ** 2) / np.sum(w))

    if how == "whole":
        mean_a = float(np.sum(w * adjusted) / np.sum(w))
        mean_b = float(np.sum(w * base) / np.sum(w))
        k = np.sqrt(spread(base, mean_b) / spread(adjusted, mean_a))
        return mean_a + k * (adjusted - mean_a), float(k)
    if how != "within":
        raise ValueError(f"how must be 'within' or 'whole', got {how!r}")
    team = pd.Series(team)
    known = team.notna().to_numpy()
    key = team.fillna(-1).to_numpy()

    def team_mean(x):
        s = pd.DataFrame({"t": key, "wx": w * x, "w": w}).groupby("t")[["wx", "w"]].transform("sum")
        return np.where(s.w.to_numpy() > 0, s.wx.to_numpy() / np.where(s.w.to_numpy() > 0, s.w.to_numpy(), 1.0), x)

    m_a, m_b = team_mean(adjusted), team_mean(base)
    m_a, m_b = np.where(known, m_a, adjusted), np.where(known, m_b, base)
    k = np.sqrt(spread(base, m_b) / spread(adjusted, m_a))
    return m_a + k * (adjusted - m_a), float(k)
