"""B2 and B3, experiment 35's ridge baselines: vanilla RAPM, and vanilla RAPM centered on a prior (B1, the linear
box-score SPM) with a multiplier per side.

One season's (or one fold's rating games') stint design, actual points, possession weights; the context block
(intercept, home, playoffs, garbage time, margin terms) unpenalized; one penalty per side on the players:

    minimize  sum w (y - X b)^2 + lam_o |o - m_o p_o|^2 + lam_d |d - m_d p_d|^2

For a penalty pair the solution is linear in the multipliers:

    b(m) = b0 + m_o q_o + m_d q_d,   b0 = (G + L)^-1 X'Wy,   q_side = (G + L)^-1 L p_side

so one factorization per penalty pair gives B2 (m = 0) and every B3.  Centering, the stand-in for unrated players
and the held-out contributions are all linear in the rating, so the held-out error is an exact quadratic in
(m_o, m_d): storing six numbers per penalty pair per scored block lets the multipliers and the penalties be chosen
afterwards, nested, without refitting anything.

Ratings are raw sign (`o` adds points scored, `d` adds points allowed).  A player with no possessions in the rating
games is unrated (the stand-in), as in the scoring code.  Model layer: nothing here reads a file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp

from .scorecard import Block, level_matrix, _wls_residual


@dataclass
class Components:
    """Three rating tables (raw sign, rated players only): the m = 0 part and one per unit multiplier."""
    player_id: np.ndarray
    poss: np.ndarray
    base: np.ndarray     # (n, 2): o, d
    unit_o: np.ndarray   # (n, 2)
    unit_d: np.ndarray   # (n, 2)

    def table(self, m_o: float = 0.0, m_d: float = 0.0) -> pd.DataFrame:
        v = self.base + m_o * self.unit_o + m_d * self.unit_d
        return pd.DataFrame({"player_id": self.player_id, "o": v[:, 0], "d": v[:, 1], "poss": self.poss})


class Ridge:
    """The normal equations of one stint design, factorized once per penalty pair."""

    def __init__(self, X: sp.spmatrix, y: np.ndarray, w: np.ndarray, n_players: int, player_ids: np.ndarray,
                 poss: np.ndarray, n_fixed: int | None = None):
        """`X` is [offense players | defense players | fixed effects | anything else].  `n_fixed` keeps only the
        design's named fixed effects: a cached design (`designcache.make_X`) carries the game index as one more column,
        and left in, an unpenalized column it becomes a within-season time trend the shipped ridge does not have
        (experiment 37, stage 0).  None keeps every column (the callers before that fix)."""
        X = sp.csr_matrix(X)
        if n_fixed is not None:
            X = X[:, :2 * int(n_players) + int(n_fixed)]
        self.m = int(n_players)
        self.c = X.shape[1] - 2 * self.m
        WX = X.multiply(np.asarray(w, dtype=float)[:, None]).tocsr()
        self.G = np.asarray((X.T @ WX).todense())
        self.b = np.asarray(WX.T @ np.asarray(y, dtype=float)).ravel()
        self.player_ids = np.asarray(player_ids, dtype=np.int64)
        self.poss = np.asarray(poss, dtype=float)
        self.rated = self.poss > 0

    def components(self, lam_o: float, lam_d: float, prior_o: np.ndarray, prior_d: np.ndarray) -> Components:
        m, c = self.m, self.c
        pen = np.r_[np.full(m, float(lam_o)), np.full(m, float(lam_d)), np.zeros(c)]
        A = self.G + np.diag(pen)
        empty = np.flatnonzero((np.diagonal(self.G) <= 0) & (pen == 0))
        A[empty, empty] += 1.0                       # an unpenalized context column no row touches: coefficient 0
        po = np.r_[np.nan_to_num(prior_o) * lam_o, np.zeros(m + c)]
        pd_ = np.r_[np.zeros(m), np.nan_to_num(prior_d) * lam_d, np.zeros(c)]
        sol = sla.cho_solve(sla.cho_factor(A, check_finite=False), np.column_stack([self.b, po, pd_]),
                            check_finite=False)
        keep = self.rated

        def side(col):
            return np.column_stack([sol[:m, col][keep], sol[m:2 * m, col][keep]])
        comp = Components(self.player_ids[keep], self.poss[keep], side(0), side(1), side(2))
        # centered on the rating games' possessions, part by part (the mean of a sum is the sum of the means)
        w = comp.poss
        for part in (comp.base, comp.unit_o, comp.unit_d):
            part -= np.average(part, axis=0, weights=w)
        return comp


def stand_in(values: np.ndarray, poss: np.ndarray, max_poss: float = 500.0, shrink: float = 0.25) -> np.ndarray:
    """`tradeset.replacement_fill` on one part: a quarter of the possession-weighted mean of players under 500
    possessions (linear in the rating, so it splits over the parts)."""
    low = (poss > 0) & (poss < max_poss)
    if not low.any():
        return np.zeros(values.shape[1])
    return shrink * np.average(values[low], axis=0, weights=poss[low])


def contributions(block: Block, ids: np.ndarray, values: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Held-out team-game contribution of one part: rated players at their values, everyone else at `fill`."""
    n = block.n_players
    idx = pd.Series(np.arange(len(ids)), index=ids)
    pos = pd.Series(block.player_ids).map(idx).to_numpy(dtype=float)
    rated = ~np.isnan(pos)
    o = np.where(rated, values[np.nan_to_num(pos).astype(int), 0], fill[0])
    d = np.where(rated, values[np.nan_to_num(pos).astype(int), 1], fill[1])
    return np.asarray(block.Z[:, :n] @ o).ravel() + np.asarray(block.Z[:, n:] @ d).ravel()


def quadratic(block: Block, comp: Components, fill: bool = True, level: str = "home") -> dict:
    """The six numbers that give the block's error for any (m_o, m_d): weighted cross-products of the profiled
    base miss and the two unit contributions, and the weight total."""
    parts = []
    for v in (comp.base, comp.unit_o, comp.unit_d):
        f = stand_in(v, comp.poss) if fill else np.zeros(2)
        parts.append(contributions(block, comp.player_id, v, f))
    A = level_matrix(block, level)
    U = _wls_residual(A, np.column_stack([block.y - parts[0], parts[1], parts[2]]), block.w)
    W = (U * block.w[:, None]).T @ U
    return dict(sw=float(block.w.sum()), g00=W[0, 0], g0o=W[0, 1], g0d=W[0, 2], goo=W[1, 1], god=W[1, 2], gdd=W[2, 2])


def error_at(q: pd.DataFrame, m_o=0.0, m_d=0.0) -> np.ndarray:
    """Per row of stored quadratics, the error at multipliers (m_o, m_d)."""
    return (q.g00 - 2 * (m_o * q.g0o + m_d * q.g0d) + m_o ** 2 * q.goo + 2 * m_o * m_d * q.god + m_d ** 2 * q.gdd) / q.sw


def best_multipliers(q: pd.DataFrame) -> tuple:
    """The (m_o, m_d) that minimize the pooled error of the stored quadratics (summed over rows)."""
    s = q[["g0o", "g0d", "goo", "god", "gdd"]].sum()
    H = np.array([[s.goo, s.god], [s.god, s.gdd]])
    m = np.linalg.solve(H, np.array([s.g0o, s.g0d]))
    return float(m[0]), float(m[1])
