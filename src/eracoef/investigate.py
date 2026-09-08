"""The investigator: where the board is wrong out of season, and who is standing there.

For a held-out season H the shipped system's MAPPED ratings -- the tracker dump for H (fitted without H) under
the shipping map (fitted without H) -- predict every stint row of H the way the criterion does, level refit
and all, and the residual r = actual - predicted, in points per 100 of the row's offense, is what the board
did not know.  Three questions, in order of how much of the miss they put on one player:

  lineups    five-man units with the largest possession-weighted mean residual over their rows (`lineups`).
             A unit that scores 8 per 100 more than its five ratings say, over 400 possessions, is a fact
             about the unit; which of the five it is about is the next two questions.
  on-court   per player and side, the weighted mean residual of the rows he was on the floor for
             (`on_court`): every lineup with him against everything else.  Blames him for his teammates.
  ridge      r regressed on [Z_O | Z_D] with a ridge (`residual_ridge`): the residual RAPM, the on-court
             number with the teammates' share taken out.  This is the "same four, plus and minus him"
             question asked of every lineup at once, and the column to sort by.

Signs are the BOARD's: `miss_o` and `miss_d` are points per 100 the board should add to his offense and to
his defense (positive = under-rated on that side).  The row residual is in the offense's terms, so a
defender's coefficient is flipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def residual_ridge(Zo, Zd, r, w, lam: float = 2000.0):
    """r ~ Zo a + Zd b, weighted by w, ridge `lam` on every coefficient (in the weights' units: a player with
    `lam` possessions is shrunk halfway to zero).  Returns (a, b, se_a, se_b): b in the row's (offense's)
    sign, so flip it for the board's.  The standard errors are the ridge's own, sigma^2 diag(S^-1 G S^-1)."""
    X = sp.hstack([sp.csr_matrix(Zo), sp.csr_matrix(Zd)]).tocsr()
    w = np.asarray(w, dtype=float)
    r = np.asarray(r, dtype=float)
    Xw = X.T.multiply(w).tocsr()
    G = (Xw @ X).toarray()
    g = np.asarray(Xw @ r).ravel()
    S = G + float(lam) * np.eye(G.shape[0])
    Sinv = np.linalg.inv(S)
    theta = Sinv @ g
    fit = X @ theta
    n_eff = max(float(w.sum()) / max(float(np.average(w, weights=w)), 1.0), 1.0)   # rows, weighted
    sigma2 = float(np.average((r - fit) ** 2, weights=w)) * float(np.average(w, weights=w))
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ki->i", Sinv, G, Sinv) * sigma2, 0.0))
    m = Zo.shape[1]
    return theta[:m], theta[m:], se[:m], se[m:]


def attributable(Zo, Zd, r, w, lam: float = 2000.0) -> dict:
    """The investigator's score: how much of the out-of-season residual's weighted variance the player ridge
    can put on players -- `total` (the weighted mean square of r), `player` (total minus the ridge's residual
    mean square, i.e. what named players explain), and the ridge's own weighted sum of squared coefficients
    per side (`ss_o`, `ss_d`, possession-weighted).  A board with better attribution leaves less for the
    ridge to find; the team-game criterion is nearly blind to this (FINDINGS 26.5), this is not."""
    X = sp.hstack([sp.csr_matrix(Zo), sp.csr_matrix(Zd)]).tocsr()
    w = np.asarray(w, dtype=float)
    r = np.asarray(r, dtype=float)
    Xw = X.T.multiply(w).tocsr()
    G = (Xw @ X).toarray()
    g = np.asarray(Xw @ r).ravel()
    theta = np.linalg.solve(G + float(lam) * np.eye(G.shape[0]), g)
    fit = X @ theta
    total = float(np.average(r ** 2, weights=w))
    left = float(np.average((r - fit) ** 2, weights=w))
    m = Zo.shape[1]
    po, pd_ = np.asarray(Xw[:m].sum(axis=1)).ravel(), np.asarray(Xw[m:].sum(axis=1)).ravel()
    return dict(total=total, player=total - left,
                ss_o=float(np.average(theta[:m] ** 2, weights=np.maximum(po, 1e-9))),
                ss_d=float(np.average(theta[m:] ** 2, weights=np.maximum(pd_, 1e-9))))


def on_court(Zo, Zd, r, w):
    """Per player: the weighted mean residual of his offensive rows and of his defensive rows, and the
    possessions behind each.  Returns (mean_o, poss_o, mean_d, poss_d), mean_d in the offense's sign."""
    w = np.asarray(w, dtype=float)
    r = np.asarray(r, dtype=float)
    out = []
    for Z in (Zo, Zd):
        Z = sp.csr_matrix(Z)
        p = np.asarray(Z.T @ w).ravel()
        s = np.asarray(Z.T @ (w * r)).ravel()
        out += [np.where(p > 0, s / np.where(p > 0, p, 1.0), 0.0), p]
    return tuple(out)


def _lineup_keys(Z) -> np.ndarray:
    """The five column indices of each row, sorted, as an (n, 5) array; a row without exactly five gets -1s."""
    Z = sp.csr_matrix(Z)
    Z.sort_indices()
    n = Z.shape[0]
    counts = np.diff(Z.indptr)
    keys = np.full((n, 5), -1, dtype=np.int64)
    ok = counts == 5
    keys[ok] = Z.indices[np.repeat(ok, counts)].reshape(-1, 5)
    return keys


def lineups(Zo, Zd, ids, r, w, min_poss: float = 200.0) -> pd.DataFrame:
    """Five-man units on either side: weighted mean residual over their rows, possessions, rows.  `ids` maps a
    column to a player id.  The residual is in the OFFENSE's terms; for a defensive unit it is flipped so
    that positive = the unit is better than its five ratings say, on both sides."""
    w = np.asarray(w, dtype=float)
    r = np.asarray(r, dtype=float)
    rows = []
    for side, Z, sign in (("O", Zo, 1.0), ("D", Zd, -1.0)):
        k = _lineup_keys(Z)
        ok = (k >= 0).all(axis=1)
        d = pd.DataFrame({"key": [tuple(int(ids[j]) for j in row) for row in k[ok]], "w": w[ok], "wr": w[ok] * r[ok] * sign})
        g = d.groupby("key").agg(poss=("w", "sum"), wr=("wr", "sum"), rows=("w", "size")).reset_index()
        g["miss"] = g.wr / g.poss
        g["side"] = side
        rows.append(g[g.poss >= float(min_poss)][["side", "key", "miss", "poss", "rows"]])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values("miss", key=np.abs, ascending=False).reset_index(drop=True)


def season_table(h: int, ids, Zo, Zd, r, w, rat: pd.DataFrame | None = None, lam: float = 2000.0) -> pd.DataFrame:
    """One held-out season, per player: the ridge miss per side (board's sign), its se, the on-court miss,
    the possessions, and the mapped rating and prior he was scored with (board's sign) if `rat` is given
    (columns player_id, o, d, prior_o, prior_d in RAW sign)."""
    a, b, se_a, se_b = residual_ridge(Zo, Zd, r, w, lam)
    mo, po, md, pd_ = on_court(Zo, Zd, r, w)
    t = pd.DataFrame({"held_out": int(h), "player_id": np.asarray(ids), "miss_o": a, "se_o": se_a, "miss_d": -b, "se_d": se_b,
                      "oncourt_o": mo, "oncourt_d": -md, "poss_o": po, "poss_d": pd_})
    if rat is not None:
        m = t.merge(rat[["player_id", "o", "d", "prior_o", "prior_d"]], on="player_id", how="left")
        t["rating_o"], t["rating_d"] = m.o.to_numpy(), -m.d.to_numpy()
        t["prior_o"], t["prior_d"] = m.prior_o.to_numpy(), -m.prior_d.to_numpy()
    return t


def pooled(players: pd.DataFrame, min_poss: float = 1000.0) -> pd.DataFrame:
    """Across seasons, per player and side: the inverse-variance-weighted mean miss, its se, z, seasons, the
    possessions.  Only player-seasons with `min_poss` possessions on that side enter."""
    out = []
    for side in ("o", "d"):
        d = players[players[f"poss_{side}"] >= float(min_poss)].copy()
        v = d[f"se_{side}"].to_numpy() ** 2
        d["_iw"] = 1.0 / np.maximum(v, 1e-9)
        d["_iwm"] = d._iw * d[f"miss_{side}"]
        g = d.groupby("player_id").agg(iw=("_iw", "sum"), iwm=("_iwm", "sum"), seasons=("held_out", "nunique"),
                                       poss=(f"poss_{side}", "sum"))
        g["miss"] = g.iwm / g.iw
        g["se"] = 1.0 / np.sqrt(g.iw)
        g["z"] = g.miss / g.se
        g["side"] = side.upper()
        out.append(g.reset_index()[["player_id", "side", "miss", "se", "z", "seasons", "poss"]])
    return pd.concat(out, ignore_index=True)
