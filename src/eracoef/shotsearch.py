"""The shot-quality search harness: the registered split, fixed row samples, a fast relevel, and SALL.

The owner, 2026-10-06: "let's go bananas on making the model as good as possible! random search, different models,
stacking, you name it."  The protocol (DECISIONS.md, "The shot-quality search") is registered before any trial:

    SEARCH blocks   1997-99, 2003-05, 2009-11, 2018-20, 2021-23 -- every trial is scored here, and only here
    CONFIRM blocks  2000-02, 2006-08, 2012-14, 2015-17, 2024-26 -- read once, after the shortlist is registered;
                    they hold every tracked season (2014-17) and the live block

Each block's model trains on shotmodel.train_seasons (never the block, the season either side, or 2026).

SALL, the objective: shooter-adjusted held-out log loss.  Each scored regular-season attempt (heaves out) gets the
arm's logit relevelled from the whole season (shotmodel.relevel, mode "season"), PLUS an offset for its shooter-season
and shot value fitted on the OTHER half of that season's games under the same arm (ridge `SALL_LAM` = 25, fixed in
advance, never tuned).  So an arm is judged on what the ratings target uses -- quality times the shooter's skill
measured elsewhere -- and cannot win by knowing who takes which shots: the shooter's own offset already says it.

Model layer: frames in, numbers out.  The driver that reads tables and runs trials is scripts/127_shot_search.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import shotmodel as sm

SEARCH_BLOCKS = ((1997, 1999), (2003, 2005), (2009, 2011), (2018, 2020), (2021, 2023))
CONFIRM_BLOCKS = ((2000, 2002), (2006, 2008), (2012, 2014), (2015, 2017), (2024, 2026))
SEARCH_SEASONS = frozenset(s for a, b in SEARCH_BLOCKS for s in range(a, b + 1))
CONFIRM_SEASONS = frozenset(s for a, b in CONFIRM_BLOCKS for s in range(a, b + 1))
SALL_LAM = 25.0
SEARCH_ROWS = 750_000
FULL_ROWS = 1_500_000


def assert_phase(seasons, phase: str) -> None:
    """A search-phase scorer must never be handed a confirm season (and the reverse)."""
    s = set(int(x) for x in np.unique(np.asarray(list(seasons))))
    bad = s & (CONFIRM_SEASONS if phase == "search" else SEARCH_SEASONS)
    if bad:
        raise ValueError(f"{phase} phase handed {sorted(bad)}: the registered split forbids it")


def row_perm(n: int, season: int, seed: int = 0) -> np.ndarray:
    """One fixed permutation of a season's training rows, so the 750k search sample is nested in the 1.5M one."""
    return np.random.default_rng(1_000_003 * (seed + 1) + int(season)).permutation(n)


def block_rows(n_train_seasons: int, max_rows: int) -> int:
    return max_rows // max(n_train_seasons, 1)


# ------------------------------------------------------------------------------------------ fast band and relevel
_SUB_CODE = {"rim": 0, "mid": 1, "three": 2}


def band_codes(f: pd.DataFrame) -> np.ndarray:
    """shotmodel.band_of as integers (same cells), vectorised: sub-model x 100 + bin; the corner three is 299."""
    sub = sm.submodel_of(f)
    d = f["dist_xy"].to_numpy(float)
    out = np.empty(len(f), dtype=np.int64)
    for s, code in _SUB_CODE.items():
        m = sub == s
        e = np.asarray(sm.BANDS[s][1:-1])
        idx = np.clip(np.searchsorted(e, d[m], side="right"), 0, len(sm.BANDS[s]) - 2)
        out[m] = code * 100 + idx
    if "corner3" in f.columns:
        out[(sub == "three") & f["corner3"].to_numpy(bool)] = 299
    return out


def relevel_fast(eta: np.ndarray, season: np.ndarray, band: np.ndarray, made: np.ndarray, rs: np.ndarray,
                 k: float = sm.LEVEL_K, iters: int = 6) -> np.ndarray:
    """shotmodel.relevel(mode="season") vectorised: one shift per (season, band) cell from its regular-season rows,
    the same six Newton steps from zero, padded n / (n + k), applied to every row of the cell."""
    key = pd.factorize(pd.Series(season.astype(np.int64) * 1000 + band))[0]
    G = key.max() + 1
    rk, ry, re = key[rs], made[rs].astype(float), eta[rs]
    n = np.bincount(rk, minlength=G).astype(float)
    delta = np.zeros(G)
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-(re + delta[rk])))
        g = np.bincount(rk, weights=ry - mu, minlength=G)
        h = np.maximum(np.bincount(rk, weights=mu * (1 - mu), minlength=G), 1e-9)
        delta = delta + g / h
    shift = delta * n / (n + k)
    return eta + shift[key]


# ------------------------------------------------------------------------------------------ SALL
def _logloss(y, eta):
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ridge_offsets(eta, y, codes, n_groups, lam, iters=4):
    d = np.zeros(n_groups)
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-(eta + d[codes])))
        g = np.bincount(codes, weights=y - mu, minlength=n_groups) - lam * d
        h = np.bincount(codes, weights=mu * (1 - mu), minlength=n_groups) + lam
        d = d + g / h
    return d


def sall_table(eta_lev: np.ndarray, f: pd.DataFrame, lam: float = SALL_LAM) -> pd.DataFrame:
    """Per season and sub-model: n, SALL (shooter-adjusted log loss) and raw log loss of the relevelled logit.

    `f` holds the scored rows: season, half (A/B), shooter, value, made, and what submodel_of needs.  Rows outside
    the halves are ignored."""
    sub = sm.submodel_of(f)
    half = f["half"].to_numpy(object)
    keep = np.isin(half, ["A", "B"])
    y = f["made"].to_numpy(float)
    season = f["season"].to_numpy()
    grp = pd.factorize(pd.Series(season.astype(np.int64) * 10_000_000_000 + f["shooter"].to_numpy(np.int64) * 10
                                 + f["value"].to_numpy(np.int64)))[0]
    G = grp.max() + 1
    adj = np.full(len(f), np.nan)
    for src, dst in (("A", "B"), ("B", "A")):
        ms, md = keep & (half == src), keep & (half == dst)
        d = _ridge_offsets(eta_lev[ms], y[ms], grp[ms], G, lam)
        adj[md] = eta_lev[md] + d[grp[md]]
    ok = keep & np.isfinite(adj)
    out = pd.DataFrame(dict(season=season[ok], sub=sub[ok], sall=_logloss(y[ok], adj[ok]), raw=_logloss(y[ok], eta_lev[ok])))
    return out.groupby(["season", "sub"]).agg(n=("sall", "size"), sall=("sall", "mean"), raw=("raw", "mean")).reset_index()


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str = "sall", sub: str | None = None) -> dict:
    """Arm a minus arm b, per season (attempt-weighted over the sub-models unless `sub`), paired across seasons."""
    def per(t):
        t = t if sub is None else t[t["sub"] == sub]
        return t.groupby("season").apply(lambda g: np.average(g[col], weights=g["n"]), include_groups=False)
    d = (per(a) - per(b)).dropna()
    if len(d) < 2:
        return dict(diff=float(d.mean()) if len(d) else np.nan, z=np.nan, won=int((d < 0).sum()), n=len(d))
    return dict(diff=float(d.mean()), z=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), won=int((d < 0).sum()),
                n=len(d))
