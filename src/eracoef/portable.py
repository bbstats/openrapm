"""Experiment 40: a portable rating -- what a player's rating says about his impact on a NEW team.

A rating splits into its box-score prior part and the part beyond the box score (the games part plus the swap
adjustment).  On held-out team-games each part's contribution is split by whether the player stayed with his team or
was traded (`scorecard.split_traded`) and by his playing-time tier.  Per side s and part c:

    y = level + sum_k beta[s,c,k] * (X_stayed[s,c,k] + ratio[s,c] * X_traded[s,c,k]) + tier levels + traded level

Players who stayed set each tier's slope; traded players get the SAME tier's slope times one ratio per side and part,
common to every tier.  Traded players are therefore compared with players of the same playing time who stayed, so a
role player is not marked down for being a role player (the owner, 2026-10-05: "provided we adjust for the fact that
traded players are worse often").  The tier levels and the traded level keep an average miss by tier, or an average
new-team effect, out of the slopes.

The portable rating is the rating with each part times its ratio; the team rating is the rating as shipped.

Model layer: nothing here reads a file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import scorecard as sc

SIDES = ("O", "D")
PARTS = ("prior", "beyond")
TIERS = (0, 1, 2)
TIER_EDGES = (1000.0, 2500.0)          # full-season-equivalent possessions: <1,000 / 1,000-2,500 / 2,500+ (scorecard)


def slope_names() -> list:
    return [f"{s} {c} t{k}" for s in SIDES for c in PARTS for k in TIERS]


def level_names() -> list:
    return [f"{s} traded level" for s in SIDES] + [f"{s} tier{k} level" for s in SIDES for k in TIERS[:-1]]


def column_names() -> list:
    return ([f"{n} stayed" for n in slope_names()] + [f"{n} traded" for n in slope_names()] + level_names())


def tier_of(poss: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """0 / 1 / 2 by full-season-equivalent possessions (`scale` = 1 / the rating games' share of the season)."""
    return np.digitize(np.asarray(poss, dtype=float) * scale, TIER_EDGES)


def columns(block: sc.Block, Zt, parts: pd.DataFrame) -> dict:
    """The held-out columns of `column_names()`.  `parts`: player_id, o_prior, o_beyond, d_prior, d_beyond (raw sign:
    d adds points allowed), tier; players without a row contribute nothing (they are the stand-ins' offset)."""
    n = block.n_players
    p = parts.drop_duplicates("player_id").set_index("player_id").reindex(block.player_ids)
    rated = p.tier.notna().to_numpy()
    tier = np.where(rated, p.tier.fillna(-1).to_numpy(), -1).astype(int)
    stay, traded = sc.split_traded(block, Zt)
    out = {}
    for s, lo in (("O", 0), ("D", n)):
        Zs, Zt_ = stay.Z[:, lo:lo + n], traded.Z[:, lo:lo + n]
        Zall = block.Z[:, lo:lo + n]
        for c in PARTS:
            v = np.nan_to_num(p[f"{s.lower()}_{c}"].to_numpy(dtype=float))
            for k in TIERS:
                vk = np.where(tier == k, v, 0.0)
                out[f"{s} {c} t{k} stayed"] = np.asarray(Zs @ vk).ravel()
                out[f"{s} {c} t{k} traded"] = np.asarray(Zt_ @ vk).ravel()
        out[f"{s} traded level"] = np.asarray(Zt_ @ rated.astype(float)).ravel()
        for k in TIERS[:-1]:
            out[f"{s} tier{k} level"] = np.asarray(Zall @ (tier == k).astype(float)).ravel()
    return {k: out[k] for k in column_names()}


# ------------------------------------------------------------------------------------------ the fit
def _solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(A, b, rcond=None)[0]


def groups_for(variant: str) -> dict:
    """slope name -> its ratio's key (None = held at 1) for each portable variant:
    "box+beyond"       one ratio per side and part (4)
    "beyond"           the box part held at 1, one ratio per side for the part beyond it (2)
    "beyond-one"       the box part held at 1, one ratio for the part beyond it on both sides (1)
    "box+beyond-tier"  one ratio per side, part and playing-time tier (12)"""
    out = {}
    for nm in slope_names():
        s, c, t = nm.split(" ")
        k = int(t[1:])
        if variant == "box+beyond":
            out[nm] = (s, c)
        elif variant == "beyond":
            out[nm] = (s, c) if c == "beyond" else None
        elif variant == "beyond-one":
            out[nm] = ("both", c) if c == "beyond" else None
        elif variant == "box+beyond-tier":
            out[nm] = (s, c, k)
        else:
            raise ValueError(f"unknown variant {variant!r}")
    return out


def fit(xx: np.ndarray, xy: np.ndarray, variant: str = "box+beyond", iters: int = 500, tol: float = 1e-13) -> dict:
    """The constrained model on pooled normal equations (columns in `column_names()` order): alternating least squares
    between (tier slopes, levels) given the ratios and (ratios, levels) given the tier slopes; each step lowers the
    residual sum of squares, which is returned up to its constant (-2 b'xy + b'xx b).  `ratio` maps each slope name
    to its ratio (1 where the variant holds it)."""
    names = column_names()
    j = {nm: i for i, nm in enumerate(names)}
    slopes, levels = slope_names(), level_names()
    group = groups_for(variant)
    keys = sorted({g for g in group.values() if g is not None}, key=str)
    rk = {k: 1.0 for k in keys}
    beta = np.ones(len(slopes))
    lev = np.zeros(len(levels))

    def r_of(nm):
        return 1.0 if group[nm] is None else rk[group[nm]]

    def full(beta, lev):
        b = np.zeros(len(names))
        for i, nm in enumerate(slopes):
            b[j[f"{nm} stayed"]] = beta[i]
            b[j[f"{nm} traded"]] = r_of(nm) * beta[i]
        for i, nm in enumerate(levels):
            b[j[nm]] = lev[i]
        return b

    def ssr(b):
        return float(-2 * b @ xy + b @ xx @ b)

    last = ssr(full(beta, lev))
    for _ in range(iters):
        # (1) tier slopes and levels, the ratios held
        T = np.zeros((len(names), len(slopes) + len(levels)))
        for i, nm in enumerate(slopes):
            T[j[f"{nm} stayed"], i] = 1.0
            T[j[f"{nm} traded"], i] = r_of(nm)
        for i, nm in enumerate(levels):
            T[j[nm], len(slopes) + i] = 1.0
        phi = _solve(T.T @ xx @ T, T.T @ xy)
        beta, lev = phi[:len(slopes)], phi[len(slopes):]
        # (2) ratios and levels, the tier slopes held (a held ratio's traded column stays in the offset)
        b0 = np.zeros(len(names))
        for i, nm in enumerate(slopes):
            b0[j[f"{nm} stayed"]] = beta[i]
            if group[nm] is None:
                b0[j[f"{nm} traded"]] = beta[i]
        T = np.zeros((len(names), len(keys) + len(levels)))
        for i, nm in enumerate(slopes):
            if group[nm] is not None:
                T[j[f"{nm} traded"], keys.index(group[nm])] = beta[i]
        for i, nm in enumerate(levels):
            T[j[nm], len(keys) + i] = 1.0
        phi = _solve(T.T @ xx @ T, T.T @ (xy - xx @ b0))
        rk = {k: float(phi[i]) for i, k in enumerate(keys)}
        lev = phi[len(keys):]
        now = ssr(full(beta, lev))
        if last - now < tol * max(1.0, abs(last)):
            last = now
            break
        last = now
    ratio = {nm: r_of(nm) for nm in slopes}
    return dict(ratio=ratio, keys=rk, beta=dict(zip(slopes, beta)), levels=dict(zip(levels, lev)), ssr=last,
                coef=full(beta, lev))


def free_fit(xx: np.ndarray, xy: np.ndarray) -> pd.Series:
    """Every column's own coefficient (no constraint): the per-tier stayed and traded slopes, for the role check."""
    return pd.Series(_solve(xx, xy), index=column_names())


def portable(parts: pd.DataFrame, ratio: dict) -> pd.DataFrame:
    """player_id, o, d, poss (raw sign) of the portable rating: each part times its ratio (`fit(...)["ratio"]`,
    keyed by slope name, so a tier's ratio applies to that tier's players)."""
    tier = parts.tier.to_numpy().astype(int)
    out = {"player_id": parts.player_id.to_numpy(), "poss": parts.poss.to_numpy()}
    for s, col in (("O", "o"), ("D", "d")):
        v = np.zeros(len(parts))
        for c in PARTS:
            r = np.array([ratio[f"{s} {c} t{k}"] for k in TIERS])[tier]
            v += r * parts[f"{col}_{c}"].to_numpy(dtype=float)
        out[col] = v
    return pd.DataFrame(out)


def portable_table(table: pd.DataFrame, ratio_beyond: float, weight: str = "poss_season") -> pd.DataFrame:
    """The portable rating of a rankings table (columns prior_*, u_*, c_* and rating_*, defense as points prevented):
    the box-prior part kept, the part beyond it (u + c) times `ratio_beyond`, re-centered so each season's
    possession-weighted mean equals the rating's.  Returns player_id, season, portable_off, portable_def,
    portable_total in the table's row order."""
    out = table[["player_id", "season"]].copy()
    for side in ("off", "def"):
        v = table[f"prior_{side}"] + ratio_beyond * (table[f"u_{side}"] + table[f"c_{side}"])
        w = table[weight].astype(float)
        shift = ((table[f"rating_{side}"] - v) * w).groupby(table.season).sum() / w.groupby(table.season).sum()
        out[f"portable_{side}"] = (v + table.season.map(shift).fillna(0.0)).to_numpy()
    out["portable_total"] = out.portable_off + out.portable_def
    return out
