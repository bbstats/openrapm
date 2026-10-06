"""Experiment 37: a box-score prior fit directly on held-out games.

The prior-informed RAPM of B3 (vanilla.py) is centered on a prior `p` with a ridge penalty.  When the prior is LINEAR
in a set of inputs per side -- p_o = F_o beta_o, p_d = F_d beta_d -- the rating is linear in the input weights:

    b(beta) = b0 + sum_j beta_j r_j,     b0 = (G + L)^-1 X'Wy,   r_j = (G + L)^-1 L f_j

so one factorization per penalty pair gives every input's response.  Centering, the stand-in for unrated players and
the held-out contributions are linear too, and the level refit is a projection, so the held-out team-game error is an
EXACT quadratic in beta:

    error(beta) = (G00 - 2 g'beta + beta' A beta) / sw,     [G00 g'; g A] = U'WU

with U the profiled columns [y - base contribution, each input's contribution].  Stored once per scored block (pooled
per season), every choice of inputs, anchor and penalty is then algebra on small matrices -- feature and model
selection with the held-out test itself as the target.  B3 is the case of one input per side (B1's prediction); B2 the
case of no inputs.

`level="team"` also profiles out each offensive and defensive team's level on the scored games: what is left sees only
how credit is split inside teams (a guardrail, never the fitting target).

Model layer: nothing here reads a file.  The scripts build the input tables and pass them in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.linalg as sla

from .scorecard import Block, _wls_residual
from .vanilla import Ridge

# The lockbox (the owner, 2026-10-05: interleaved, so every era is in both halves): every third season.  No selection
# may touch these seasons' games; the frozen winner is scored on them once.
LOCKBOX = tuple(range(1999, 2027, 3))

# The input groups.  Each column is read from the side's own row of the fold's rebuilt panel (offense inputs from the
# offensive row, defense inputs from the defensive row); `b1` and `b1_inputs` come from the linear box-score SPM.
GROUPS = {
    "b1": ["b1"],                                               # B1's prediction (the anchor)
    "b1_inputs": ["bx_" + c for c in ("fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast",
                                      "tov", "stl", "blk", "pf")] + ["bx_min_share"],
    "rates": ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl", "blk", "pf"],
    "efficiency": ["efg", "ts", "p3r", "ftr", "fg3p", "fg2p", "ftp", "astr", "tovr", "orbsh"],
    "shot_location": ["q2", "q3", "m2", "m3", "xps", "mpts"],
    "role": ["poss_pct", "gs_pct"],
    "body_career": ["age", "height", "weight", "exp_yrs", "exp_poss", "entry_age", "draft_pick", "tenure", "n_teams"],
    "score_context": ["gt_share", "closeness", "abs_margin", "po_share"],
    "on_court": ["onc_o", "onc_d"],
    "off_court": ["offc_o", "offc_d"],
    "same_games_rapm": ["apm"],
    "booster": ["booster"],                                     # OpenRAPM's boosted prior, raw (reads on-court inputs)
}

# Columns that must never be an input, and why.
BANNED = {
    "test_poss_off": "held-out possessions: a direct leak of the scored games",
    "test_poss_def": "held-out possessions: a direct leak of the scored games",
    "rating_off": "the fold's own OpenRAPM rating (benchmarking the model against itself)",
    "rating_def": "the fold's own OpenRAPM rating",
    "rating_total": "the fold's own OpenRAPM rating",
    "prior_off": "the fold's own OpenRAPM prior after its fitted scale",
    "prior_def": "the fold's own OpenRAPM prior after its fitted scale",
    "u_off": "the fold's own OpenRAPM games part",
    "u_def": "the fold's own OpenRAPM games part",
    "prior_scale_off": "fitted on the fold's games by OpenRAPM",
    "prior_scale_def": "fitted on the fold's games by OpenRAPM",
    "poss": "scales with the fold size (use shares)",
    "net_o": "onc minus offc, implied by the two",
    "net_d": "onc minus offc, implied by the two",
    **{f"raw_{c}": "an exact per-season shift of the padded rate" for c in GROUPS["rates"]},
    **{c: "an exact linear combination of the rates" for c in ("pts", "fga", "fta", "fg3a", "usage", "bigness", "reb",
                                                               "stocks", "creation", "shotmix")},
}


def columns_of(groups) -> list:
    """The input columns of a list of group names, refusing any banned column."""
    cols = [c for g in groups for c in GROUPS[g]]
    bad = [c for c in cols if c in BANNED]
    if bad:
        raise ValueError(f"banned inputs {bad}: {[BANNED[c] for c in bad]}")
    return cols


def impute(F: np.ndarray, weight: np.ndarray, rated: np.ndarray) -> np.ndarray:
    """Missing values at the rated players' weighted mean (a constant input moves nothing, so the level is free)."""
    F = np.array(F, dtype=float)
    w = np.where(rated, weight, 0.0)
    for j in range(F.shape[1]):
        bad = np.isnan(F[:, j])
        if bad.any():
            ok = ~bad & (w > 0)
            F[bad, j] = np.average(F[ok, j], weights=w[ok]) if ok.any() else 0.0
    return F


# ------------------------------------------------------------------------------------------ responses
def responses(ridge: Ridge, lam_o: float, lam_d: float, F_o: np.ndarray, F_d: np.ndarray) -> tuple:
    """Every input's response on the rated players, centered on their possessions.

    F_o (m x Ko), F_d (m x Kd): the inputs on the design's player columns.  Returns (player_id, poss, V_o, V_d) with
    V_side (n_rated x (1 + Ko + Kd)): column 0 the rating with no prior, column 1 + j the rating's change per unit of
    offensive input j, column 1 + Ko + j per unit of defensive input j."""
    m, c = ridge.m, ridge.c
    ko, kd = F_o.shape[1], F_d.shape[1]
    pen = np.r_[np.full(m, float(lam_o)), np.full(m, float(lam_d)), np.zeros(c)]
    A = ridge.G + np.diag(pen)
    empty = np.flatnonzero((np.diagonal(ridge.G) <= 0) & (pen == 0))
    A[empty, empty] += 1.0
    rhs = np.zeros((2 * m + c, 1 + ko + kd))
    rhs[:, 0] = ridge.b
    rhs[:m, 1:1 + ko] = float(lam_o) * np.nan_to_num(F_o)
    rhs[m:2 * m, 1 + ko:] = float(lam_d) * np.nan_to_num(F_d)
    sol = sla.cho_solve(sla.cho_factor(A, check_finite=False), rhs, check_finite=False)
    keep = ridge.rated
    w = ridge.poss[keep]
    V_o, V_d = sol[:m][keep], sol[m:2 * m][keep]
    V_o = V_o - np.average(V_o, axis=0, weights=w)
    V_d = V_d - np.average(V_d, axis=0, weights=w)
    return ridge.player_ids[keep], w, V_o, V_d


def contributions(block: Block, ids: np.ndarray, w: np.ndarray, V_o: np.ndarray, V_d: np.ndarray,
                  fill: bool = True, max_poss: float = 500.0, shrink: float = 0.25) -> np.ndarray:
    """Held-out contributions (rows x columns) of every response column; unrated players at the stand-in
    (`tradeset.replacement_fill`'s rule, linear in the rating, so it applies column by column)."""
    n = block.n_players
    pos = pd.Series(np.arange(len(ids)), index=ids).reindex(block.player_ids).to_numpy()
    rated = ~np.isnan(pos)
    idx = np.nan_to_num(pos).astype(int)
    low = (w > 0) & (w < max_poss)
    out = 0.0
    for V, Z in ((V_o, block.Z[:, :n]), (V_d, block.Z[:, n:])):
        f = shrink * np.average(V[low], axis=0, weights=w[low]) if (fill and low.any()) else np.zeros(V.shape[1])
        full = np.where(rated[:, None], V[idx], f[None, :])
        out = out + Z @ full
    return np.asarray(out)


def level_matrix(block: Block, level: str, teams: tuple | None = None) -> np.ndarray:
    """"home": intercept and home edge; "team": also every offensive and every defensive team's level."""
    base = np.column_stack([np.ones(len(block.y)), block.home])
    if level == "home":
        return base
    if level != "team":
        raise ValueError(f"level must be 'home' or 'team', got {level!r}")
    t_off, t_def = teams
    return np.column_stack([base, pd.get_dummies(pd.Series(t_off)).to_numpy(float),
                            pd.get_dummies(pd.Series(t_def)).to_numpy(float)])


def gram(block: Block, C: np.ndarray, level: str = "home", teams: tuple | None = None,
         score_weights: np.ndarray | None = None) -> np.ndarray:
    """U'WU for U = the profiled [y - C0, C1, ...]: (1 + K) x (1 + K).  `score_weights` re-weights the rows after
    the level refit (which uses the block's own weights), so complementary weightings split the error exactly."""
    U = _wls_residual(level_matrix(block, level, teams), np.column_stack([block.y - C[:, 0], C[:, 1:]]), block.w)
    sw_rows = block.w if score_weights is None else np.asarray(score_weights, dtype=float)
    return (U * sw_rows[:, None]).T @ U


# ------------------------------------------------------------------------------------------ the algebra
def error_at(G: np.ndarray, sw: float, beta: np.ndarray) -> float:
    """The error of a stored (or pooled) gram at input weights beta."""
    beta = np.asarray(beta, dtype=float)
    return float((G[0, 0] - 2 * G[0, 1:] @ beta + beta @ G[1:, 1:] @ beta) / sw)


def fit(G: np.ndarray, sw: float, penalty: np.ndarray) -> np.ndarray:
    """The weights minimizing the pooled error plus sum(penalty_j beta_j^2) (penalty 0 = a free, anchored weight)."""
    A = G[1:, 1:] / sw + np.diag(np.asarray(penalty, dtype=float))
    return np.linalg.lstsq(A, G[0, 1:] / sw, rcond=None)[0]


def subset(G: np.ndarray, keep) -> np.ndarray:
    """The gram restricted to inputs `keep` (indices into the K inputs; the base row/column 0 always stays)."""
    idx = np.r_[0, 1 + np.asarray(keep, dtype=int)]
    return G[np.ix_(idx, idx)]


def split_gram(block: Block, C: np.ndarray, C_traded: np.ndarray, level: str = "home",
               teams: tuple | None = None) -> np.ndarray:
    """U'WU for U = the profiled [y - C0, the inputs' contributions from players who stayed, from players on a new
    team]: (1 + 2K) x (1 + 2K) (experiment 39).  `C_traded` is `contributions` on the traded entries alone
    (scorecard.split_traded); `collapse_split` of the result is `gram` exactly."""
    Cs = C[:, 1:] - C_traded[:, 1:]
    U = _wls_residual(level_matrix(block, level, teams), np.column_stack([block.y - C[:, 0], Cs, C_traded[:, 1:]]),
                      block.w)
    return (U * block.w[:, None]).T @ U


def split_subset(G: np.ndarray, keep, K: int) -> np.ndarray:
    """A split gram restricted to inputs `keep`: rows and columns [base, stayed `keep`, traded `keep`]."""
    keep = np.asarray(keep, dtype=int)
    idx = np.r_[0, 1 + keep, 1 + K + keep]
    return G[np.ix_(idx, idx)]


def collapse_split(G: np.ndarray, K: int) -> np.ndarray:
    """The plain gram of a split one: each input's stayed and traded halves added back together."""
    T = np.zeros((1 + 2 * K, 1 + K))
    T[0, 0] = 1.0
    T[1:1 + K, 1:] = np.eye(K)
    T[1 + K:, 1:] = np.eye(K)
    return T.T @ G @ T


def increment_gram(Gs: np.ndarray, beta_ref: np.ndarray, beta_new: np.ndarray) -> np.ndarray:
    """3 x 3 gram of [the reference weights' miss, the new weights' change on the players who stayed, on the traded
    players], from a split gram restricted to the same k inputs (`split_subset`).

    With Q this matrix and sw the possessions, the error with the change applied at weights (a_s, a_t) is
    (Q00 - 2 (a_s Q01 + a_t Q02) + a'Q[1:,1:]a) / sw: (1, 0) gives the change to the players who stayed only, (0, 1) to
    the traded players only.  solve(Q[1:,1:], Q[1:,0]) are the blend weights: 1 = the change is all signal, 0 = none."""
    beta_ref, beta_new = np.asarray(beta_ref, dtype=float), np.asarray(beta_new, dtype=float)
    k = len(beta_ref)
    d = beta_new - beta_ref
    A = np.zeros((1 + 2 * k, 3))
    A[0, 0] = 1.0
    A[1:1 + k, 0] = -beta_ref
    A[1 + k:, 0] = -beta_ref
    A[1:1 + k, 1] = d
    A[1 + k:, 2] = d
    return A.T @ Gs @ A


def drift(G: np.ndarray, t: float) -> np.ndarray:
    """The gram of the inputs plus each input times `t` (constant within the block): weights that drift linearly with
    t, beta_j + gamma_j t, without another solve."""
    K = G.shape[0] - 1
    T = np.zeros((1 + 2 * K, 1 + K))
    T[0, 0] = 1.0
    T[1:1 + K, 1:] = np.eye(K)
    T[1 + K:, 1:] = t * np.eye(K)
    return T @ G @ T.T
