"""A smooth per-side calibration map for the finished ratings, fitted on the out-of-season criterion.

The role and rank diagnostics (FINDINGS 19) say the multi-stage board's offense is too timid at the top
(starters want x1.20, the best decile x1.24) and its defense is about right.  The owner's question: is
there a smooth function f_o, f_d of the rating alone that, applied to the finished ratings, predicts
held-out games better?  This module answers it in three steps.

1. `dump_ratings`: fit every named system once per held-out season and K, and keep the RATINGS (not the
   scores) -- outputs/ratings_<tag>.parquet.  This is the expensive part, ~2 min per system per K.
2. `SeasonFrame`: the held-out season's design reduced to what scoring needs: the two lineup matrices,
   the target, weights, the level columns and the team-game aggregation.
3. `evaluate`: for a map (per side: a family in the rating, plus optionally a level term in the player's
   training exposure), fit the parameters on every held-out season EXCEPT H (the same rule the rank map
   used), apply them to H, and score H exactly as the criterion does (`predict_season` + `score`).  The
   fitting objective is the TEAM-GAME error, the owner's north star, not the stint error the rank map was
   fitted on.

Every map is linear in its parameters (a basis in the rating and in the exposure), so every fit is one
weighted least squares on the pooled team-game residuals with the per-season level (intercept, home)
profiled out.  What it found (FINDINGS 20): a map of the rating alone is the identity at game level for
the multi-stage board; the exposure term is worth -0.8 to -1.0 per 100 on every system.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .holdout import (RESULT_COLUMNS, THREAD_VARS, Context, Holdout, Ratings, _fit, level_columns, predict_season,
                      score)


# ---------------------------------------------------------------------------------------- 1. the dump
def dump_systems(ho: Holdout, systems: list, ctx: Context, held=None, verbose: bool = True) -> pd.DataFrame:
    """Fit every system on every held-out season and K (this process) and return the ratings, one row per
    (system, k, held_out, player): the columns of Ratings.df plus the fill values for unseen players."""
    rows = []
    t0 = time.time()
    held = ho.seasons() if held is None else [int(h) for h in held]
    for h in held:
        ctx.current_h = h
        for k in ho.ks:
            ctx.current_k = k
            train = ctx.neighbourhood(h, k)
            for s in systems:
                t1 = time.time()
                rat = _fit(s, train, ctx, ho.lams[0])
                secs = time.time() - t1
                d = rat.df.copy()
                for c in ("prior_o", "prior_d"):
                    if c not in d.columns:
                        d[c] = np.nan
                rows.append(d.assign(held_out=int(h), k=int(k), system=s.name, fill_o=rat.fill_o, fill_d=rat.fill_d,
                                     seconds=secs))
        if verbose:
            print(f"  {h} dumped ({time.time() - t0:.0f}s)", flush=True)
    ctx.current_h = None
    return pd.concat(rows, ignore_index=True)


def _dump_worker(job: dict) -> pd.DataFrame:
    from .systems import registry
    cfg = job["cfg"]
    ctx = Context.load(cfg)
    reg = registry(cfg, rankmap=job.get("rankmap"))
    return dump_systems(job["holdout"], [reg[n] for n in job["names"]], ctx, held=job["held"], verbose=job.get("verbose", True))


def dump_ratings(ho: Holdout, names: list, out: Path, workers: int = 4, rankmap=None, verbose: bool = True) -> pd.DataFrame:
    """Every system's ratings for every held-out season and K, one parquet.  Same process layout as
    holdout.run_parallel (spawned workers, BLAS threads pinned)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    held = ho.seasons()
    workers = max(1, min(int(workers), len(held)))
    threads = max(1, (os.cpu_count() or workers) // workers)
    chunks = [c.tolist() for c in np.array_split(np.asarray(held), workers)]
    old = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update({k: str(threads) for k in THREAD_VARS})
    t0 = time.time()
    try:
        jobs = [dict(cfg=ho.cfg, holdout=ho, names=list(names), held=c, rankmap=rankmap, verbose=verbose) for c in chunks if c]
        if workers == 1:
            parts = [_dump_worker(jobs[0])]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
                parts = [f.result() for f in [ex.submit(_dump_worker, j) for j in jobs]]
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    R = pd.concat(parts, ignore_index=True).sort_values(["system", "k", "held_out", "player_id"]).reset_index(drop=True)
    R.to_parquet(out, index=False)
    if verbose:
        print(f"dumped {len(R)} rating rows to {out} ({time.time() - t0:.0f}s)", flush=True)
    return R


def ratings_for(dump: pd.DataFrame, system: str, k: int, h: int) -> Ratings:
    d = dump[(dump.system == system) & (dump.k == k) & (dump.held_out == h)]
    if len(d) == 0:
        raise KeyError(f"no dumped ratings for {system} K={k} H={h}")
    cols = ["player_id", "o", "d", "poss", "prior_o", "prior_d"]
    return Ratings(d[cols].reset_index(drop=True), fill_o=float(d.fill_o.iloc[0]), fill_d=float(d.fill_d.iloc[0]))


# ---------------------------------------------------------------------------------------- 2. the season frame
@dataclass
class SeasonFrame:
    """One held-out season, reduced to what a map's fit and score need.

    `Zo`, `Zd`  the offensive / defensive lineup matrices (rows x players), `ids` the players in column order
    `y`, `w`    the target (points per 100) and the fit weights; `poss` the row's possessions
    `A`         the level columns refit on the season (holdout.level_columns)
    `L`         (A'WA)^-1 A'W, so that the level profiled out of any column c is c - A (L c)
    `G`         team-game aggregation to points-per-100 units: G @ (per-100 rates) = per-100 team-game rates
    `wg`        team-game possessions, the team-game weights
    """
    h: int
    wd: object
    Zo: sp.csr_matrix
    Zd: sp.csr_matrix
    ids: np.ndarray
    y: np.ndarray
    w: np.ndarray
    poss: np.ndarray
    A: np.ndarray
    L: np.ndarray
    G: sp.csr_matrix
    wg: np.ndarray
    extra: pd.DataFrame | None = None     # per player in `ids` order: covariates AT H (age), for the Age term
    ctx: object = None                    # for covariates that depend on the training block (moved, per K)

    def covariates(self, k: int) -> pd.DataFrame | None:
        """`extra` plus the K-dependent ones: `moved` = main team in H differs from the main team in the nearest
        training season he appears in (holdout.by_movers)."""
        if self.extra is None or self.ctx is None:
            return self.extra
        ex = self.extra.copy()
        train = self.ctx.neighbourhood(self.h, k)
        now = self.ctx.main_team(self.h)
        before: dict = {}
        for s in sorted(train, key=lambda s: abs(s - self.h)):
            for pid, t in self.ctx.main_team(s).items():
                before.setdefault(pid, t)
        ex["moved"] = [1.0 if (q in now and q in before and now[q] != before[q]) else 0.0 for q in ex.player_id]
        return ex

    @classmethod
    def build(cls, ctx: Context, h: int, level: str = "home") -> "SeasonFrame":
        wd = ctx.design([h], "pts")
        extra = None
        if ctx.role_inputs is not None:
            ri = ctx.role_inputs[ctx.role_inputs.season == h][["player_id", "age"]]
            extra = pd.DataFrame({"player_id": wd.spec.ps_table["player_id"].to_numpy()}).merge(ri, on="player_id", how="left")
            extra["age"] = extra.age.fillna(float(ri.age.median()) if len(ri) else 27.0)
        m = wd.spec.n_ps
        A = level_columns(wd, level)
        w = np.asarray(wd.w, dtype=float)
        AtW = (A * w[:, None]).T
        L = np.linalg.pinv(AtW @ A) @ AtW          # pinv: the "full" level block can carry a zero or duplicate column
        poss = wd.rows["poss"].to_numpy(dtype=float)
        key = pd.factorize(pd.MultiIndex.from_arrays([wd.rows["game_idx"].to_numpy(), wd.rows["is_home_off"].to_numpy()]))[0]
        n_tg = int(key.max()) + 1
        wg = np.bincount(key, weights=poss, minlength=n_tg)
        G = sp.csr_matrix((poss / wg[key], (key, np.arange(len(key)))), shape=(n_tg, len(key)))
        return cls(h=h, wd=wd, Zo=wd.X[:, :m].tocsr(), Zd=wd.X[:, m:2 * m].tocsr(),
                   ids=wd.spec.ps_table["player_id"].to_numpy(), y=np.asarray(wd.y, dtype=float), w=w, poss=poss,
                   A=A, L=L, G=G, wg=wg, extra=extra, ctx=ctx)

    def profiled(self, C: np.ndarray) -> np.ndarray:
        """Columns with the season's level profiled out (what the criterion's refit does to any contribution)."""
        return C - self.A @ (self.L @ C)

    def game(self, C: np.ndarray) -> np.ndarray:
        return np.asarray(self.G @ C)


def load_frames(ctx: Context, seasons, level: str = "home", verbose: bool = True) -> dict:
    t0 = time.time()
    out = {int(h): SeasonFrame.build(ctx, int(h), level) for h in seasons}
    if verbose:
        print(f"built {len(out)} season frames ({time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------------------- 3. the map families
class Family:
    """A per-side map of the RATING, f(x) = sum_j theta_j phi_j(x; scale), phi_1 = x.  `scale` standardises x,
    fixed per (system, K, side) from the ratings themselves so the parameters mean the same thing in every season."""
    name: str = "identity"
    n_params: int = 1

    def basis(self, x: np.ndarray, scale: float) -> np.ndarray:
        return np.asarray(x, dtype=float)[:, None]

    def identity(self) -> np.ndarray:
        th = np.zeros(self.n_params)
        th[0] = 1.0
        return th


class Linear(Family):
    """f(x) = a x: one scalar per side, the `scale_*` diagnostic made into a map."""
    name, n_params = "linear", 1


class Poly2(Family):
    """f(x) = a x + b x^2 / s: a quadratic through zero (b > 0 stretches the top and compresses the bottom of a
    raw-sign rating)."""
    name, n_params = "poly2", 2

    def basis(self, x, scale):
        x = np.asarray(x, dtype=float)
        return np.column_stack([x, x * x / scale])


class Poly3(Family):
    """f(x) = a x + b x^2 / s + c x^3 / s^2: the cubic through zero (the rank diagnostic's own curve)."""
    name, n_params = "poly3", 3

    def basis(self, x, scale):
        x = np.asarray(x, dtype=float)
        return np.column_stack([x, x * x / scale, x ** 3 / scale ** 2])


class Sinh(Family):
    """f(x) = a x + b (sinh(x / s) - x / s) s: the 'opposite of a sigmoid', both tails stretched alike."""
    name, n_params = "sinh", 2

    def basis(self, x, scale):
        x = np.asarray(x, dtype=float)
        u = x / scale
        return np.column_stack([x, (np.sinh(u) - u) * scale])


class Expo(Family):
    """f(x) = a x + b (exp(x / s) - 1 - x / s) s: only the top stretched (or compressed), the bottom left linear."""
    name, n_params = "expo", 2

    def basis(self, x, scale):
        x = np.asarray(x, dtype=float)
        u = x / scale
        return np.column_stack([x, (np.exp(u) - 1.0 - u) * scale])


class Hinge(Family):
    """f(x) = a softplus(x / s) s - b softplus(-x / s) s: a smooth broken line, one slope per tail (a = b = 1 is
    the identity, since softplus(u) - softplus(-u) = u)."""
    name, n_params = "hinge", 2
    sharp = 2.0

    def basis(self, x, scale):
        x = np.asarray(x, dtype=float)
        u = self.sharp * x / scale
        return np.column_stack([np.logaddexp(0.0, u), -np.logaddexp(0.0, -u)]) / self.sharp * scale

    def identity(self):
        return np.ones(2)


class Exposure:
    """A per-side LEVEL term in the player's training possessions, g(poss) = sum_j eta_j psi_j(poss), added to the
    mapped rating.  Every psi is 0 at poss = 0, so a player the block never saw keeps the criterion's 0 and the
    term is the replacement gap between him and the rated players, as a smooth function of exposure.  A constant
    across the five on the floor is absorbed by the season's intercept, so only the shape is identified."""
    name: str = "none"
    n_params: int = 0

    def basis(self, poss: np.ndarray, extra: pd.DataFrame | None = None, x=None) -> np.ndarray:
        return np.zeros((len(poss), 0))


class Sat(Exposure):
    """g = c poss / (poss + s): rises fast, saturates by ~3 s; the bins say that is the shape."""
    def __init__(self, s: float = 1000.0):
        self.s = float(s)
        self.name = "sat" if s == 1000.0 else f"sat{s:g}"
        self.n_params = 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        return (p / (p + self.s))[:, None]


class LogExp(Exposure):
    """g = c log(1 + poss / s) [+ e log(1 + poss / s)^2 with `quad`]."""
    def __init__(self, s: float = 1000.0, quad: bool = False):
        self.s, self.quad = float(s), bool(quad)
        self.name = ("log2" if quad else "log") + ("" if s == 1000.0 else f"{s:g}")
        self.n_params = 2 if quad else 1

    def basis(self, poss, extra=None, x=None):
        u = np.log1p(np.asarray(poss, dtype=float) / self.s)
        return np.column_stack([u, u * u]) if self.quad else u[:, None]


class Bins(Exposure):
    """One level per training-exposure bin (cfg holdout.exposure_edges: none, 1-499, 500-1499, 1500-3999; 4000+ is
    the reference): the unsmoothed shape, for reading."""
    name, n_params = "bins", 4
    edges = (0.0, 1.0, 500.0, 1500.0, 4000.0, 1e18)

    def basis(self, poss, extra=None, x=None):
        b = np.digitize(np.asarray(poss, dtype=float), self.edges) - 1
        return np.column_stack([(b == j).astype(float) for j in range(4)])


class Unseen(Exposure):
    """One level for the players the block never saw: the replacement level, fitted on the criterion."""
    name, n_params = "unseen", 1

    def basis(self, poss, extra=None, x=None):
        return (np.asarray(poss, dtype=float) <= 0).astype(float)[:, None]


class Age(Exposure):
    """A level in the player's AGE in the held-out season: c1 (age - 27) / 5 [+ c2 ((age - 27) / 5)^2 with `quad`],
    times an indicator that the block saw him (so the unseen player keeps 0).  The block's ratings are his level
    over seasons before and after H; a player who is young in H got better from the earlier seasons to H and a
    player who is old got worse, and the criterion sees that as a term in his age at H.  Not shippable as a
    rating of the window (the window has no single 'age at H'); a prediction-time term only.  `extra` is
    the frame's per-player covariates; with none (the shipped ratings) the term is 0."""
    def __init__(self, quad: bool = True, centre: float = 27.0, scale: float = 5.0):
        self.quad, self.centre, self.scale = bool(quad), float(centre), float(scale)
        self.name = "age2" if quad else "age"
        self.n_params = 2 if quad else 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "age" not in extra.columns:
            return np.zeros((len(p), self.n_params))
        u = (np.asarray(extra["age"], dtype=float) - self.centre) / self.scale * (p > 0)
        return np.column_stack([u, u * u]) if self.quad else u[:, None]


class Moved(Exposure):
    """The player changed team between the nearest training season he appears in and H (holdout.by_movers'
    rule; `extra["moved"]`, set per K by build_design).  `slope`: a term in moved x the standardised rating
    (does a mover's rating carry less?); otherwise a level."""
    def __init__(self, slope: bool = False):
        self.slope = bool(slope)
        self.name = "movedx" if slope else "moved"
        self.n_params = 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "moved" not in extra.columns:
            return np.zeros((len(p), 1))
        m = np.asarray(extra["moved"], dtype=float) * (p > 0)
        if self.slope:
            return (m * (np.zeros(len(p)) if x is None else np.asarray(x, dtype=float)))[:, None]
        return m[:, None]


class UnseenAge(Exposure):
    """The unseen player's age (a rookie against a veteran the block never saw): (age - 27) / 5 on poss = 0."""
    name, n_params = "uage", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "age" not in extra.columns:
            return np.zeros((len(p), 1))
        return ((np.asarray(extra["age"], dtype=float) - 27.0) / 5.0 * (p <= 0))[:, None]


class AgeSat(Exposure):
    """(age - 27) / 5 times the exposure saturation poss / (poss + 1000): does the age term depend on how much
    the block saw of him?"""
    name, n_params = "agesat", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "age" not in extra.columns:
            return np.zeros((len(p), 1))
        return ((np.asarray(extra["age"], dtype=float) - 27.0) / 5.0 * p / (p + 1000.0))[:, None]


class XSat(Exposure):
    """The rating's scalar varies with exposure: b x sat(poss) (x standardised by the side's scale), added to the
    family's a x.  What a weaker or stronger ridge does, as a map term."""
    def __init__(self, s: float = 1000.0):
        self.s = float(s)
        self.name = "xsat" if s == 1000.0 else f"xsat{s:g}"
        self.n_params = 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        xx = np.zeros(len(p)) if x is None else np.asarray(x, dtype=float)
        return (xx * p / (p + self.s))[:, None]


class XLog(Exposure):
    """b x log(1 + poss / 1000)."""
    name, n_params = "xlog", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        xx = np.zeros(len(p)) if x is None else np.asarray(x, dtype=float)
        return (xx * np.log1p(p / 1000.0))[:, None]


class XAge(Exposure):
    """b x (age - 27) / 5: does the rating carry differently by age?"""
    name, n_params = "xage", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "age" not in extra.columns or x is None:
            return np.zeros((len(p), 1))
        return (np.asarray(x, dtype=float) * (np.asarray(extra["age"], dtype=float) - 27.0) / 5.0)[:, None]


class Prior(Exposure):
    """c x prior / scale: the prior part of the rating as its own column, so the map can re-weight the prior against
    the residual (f = a x + c prior = a resid + (a + c) prior): the ridge's prior-vs-data blend, re-chosen on the
    criterion.  `extra["prior"]` is set per side by build_design / mapped_ratings from the dump's prior_o / prior_d."""
    name, n_params = "prior", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "prior" not in extra.columns:
            return np.zeros((len(p), 1))
        return np.nan_to_num(np.asarray(extra["prior"], dtype=float))[:, None]


class PriorSat(Exposure):
    """c x prior x sat(poss): the prior's re-weighting allowed to depend on exposure."""
    name, n_params = "priorsat", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "prior" not in extra.columns:
            return np.zeros((len(p), 1))
        return (np.nan_to_num(np.asarray(extra["prior"], dtype=float)) * p / (p + 1000.0))[:, None]


class Combo(Exposure):
    """Several exposure terms side by side: "sat&age2"."""
    def __init__(self, parts):
        self.parts = list(parts)
        self.name = "&".join(p.name for p in self.parts)
        self.n_params = sum(p.n_params for p in self.parts)

    def basis(self, poss, extra=None, x=None):
        return np.column_stack([p.basis(poss, extra, x) for p in self.parts])


FAMILIES = {f.name: f for f in (Linear(), Poly2(), Poly3(), Sinh(), Expo(), Hinge())}
EXPOSURES = {e.name: e for e in (Exposure(), Sat(), Sat(250), Sat(500), Sat(2000), Sat(4000), LogExp(), LogExp(quad=True),
                                 Bins(), Unseen(), Age(), Age(quad=False), Moved(), Moved(slope=True), UnseenAge(),
                                 AgeSat(), XSat(), XSat(300), XSat(3000), XLog(), XAge(), Prior(), PriorSat())}


def parse_exposure(name: str) -> Exposure:
    name = name or "none"
    if "&" in name:
        return Combo([EXPOSURES[p] for p in name.split("&")])
    return EXPOSURES[name]


@dataclass(frozen=True)
class SideMap:
    """One side's map: a rating family plus an exposure term, named "<family>+<exposure>" ("linear+sat")."""
    fam: Family
    expo: Exposure = Exposure()

    @property
    def name(self) -> str:
        return self.fam.name if self.expo.n_params == 0 else f"{self.fam.name}+{self.expo.name}"

    @property
    def n_params(self) -> int:
        return self.fam.n_params + self.expo.n_params

    @classmethod
    def parse(cls, name: str) -> "SideMap":
        fam, _, expo = name.partition("+")
        return cls(FAMILIES[fam], parse_exposure(expo))

    def basis(self, x, poss, scale: float, extra=None) -> np.ndarray:
        return np.column_stack([self.fam.basis(x, scale), self.expo.basis(poss, extra, np.asarray(x, dtype=float) / scale)])

    def apply(self, x, poss, theta, scale: float, extra=None) -> np.ndarray:
        return self.basis(x, poss, scale, extra) @ np.asarray(theta, dtype=float)

    def identity(self) -> np.ndarray:
        return np.concatenate([self.fam.identity(), np.zeros(self.expo.n_params)])


# ---------------------------------------------------------------------------------------- 4. fitting and scoring
def _side_scale(dump: pd.DataFrame, system: str, k: int, side: str, min_poss: float = 1000.0) -> float:
    d = dump[(dump.system == system) & (dump.k == k) & (dump.poss >= min_poss)]
    return float(np.sqrt(np.average(d[side].to_numpy() ** 2, weights=d.poss.to_numpy()))) or 1.0


@dataclass
class Design:
    """The pooled team-game regression for a (system, K, map) choice: one block per held-out season."""
    h: np.ndarray            # season of each team-game row
    X: np.ndarray            # team-game columns, per-season level profiled out
    y: np.ndarray            # team-game target, level profiled out
    w: np.ndarray            # team-game possessions
    scale_o: float
    scale_d: float


def build_design(dump: pd.DataFrame, frames: dict, system: str, k: int, map_o: SideMap, map_d: SideMap,
                 min_poss: float = 1000.0) -> Design:
    so, sd = _side_scale(dump, system, k, "o", min_poss), _side_scale(dump, system, k, "d", min_poss)
    hs, Xs, ys, ws = [], [], [], []
    for h, f in frames.items():
        r = ratings_for(dump, system, k, h).aligned(f.ids)
        poss = r.poss.to_numpy(dtype=float)
        ex = f.covariates(k)
        ex_o = ex_d = ex
        if ex is not None and "prior_o" in r.columns:
            ex_o = ex.assign(prior=r.prior_o.to_numpy(dtype=float) / so)
            ex_d = ex.assign(prior=r.prior_d.to_numpy(dtype=float) / sd)
        Bo = map_o.basis(r.o.to_numpy(), poss, so, ex_o)
        Bd = map_d.basis(r.d.to_numpy(), poss, sd, ex_d)
        C = np.column_stack([np.asarray(f.Zo @ Bo), np.asarray(f.Zd @ Bd)])
        Xs.append(f.game(f.profiled(C)))
        ys.append(f.game(f.profiled(f.y[:, None])).ravel())
        ws.append(f.wg)
        hs.append(np.full(len(f.wg), h))
    return Design(h=np.concatenate(hs), X=np.vstack(Xs), y=np.concatenate(ys), w=np.concatenate(ws), scale_o=so, scale_d=sd)


def fit_theta(D: Design, exclude_h=None, ridge: float = 0.0, map_o: SideMap | None = None, map_d: SideMap | None = None):
    """WLS of the team-game target on the mapped columns over every season but `exclude_h`.  `ridge` pulls
    the parameters toward the identity map, in units of the pooled row weight."""
    sel = np.ones(len(D.h), dtype=bool) if exclude_h is None else D.h != exclude_h
    X, y, w = D.X[sel], D.y[sel], D.w[sel]
    XtW = (X * w[:, None]).T
    G, b = XtW @ X, XtW @ y
    if ridge > 0:
        ident = np.concatenate([map_o.identity(), map_d.identity()])
        pen = ridge * w.sum() * np.eye(len(ident))
        G, b = G + pen, b + pen @ ident
    return np.linalg.solve(G, b)


def mapped_ratings(rat: Ratings, theta, map_o: SideMap, map_d: SideMap, scale_o: float, scale_d: float,
                   extra: pd.DataFrame | None = None) -> Ratings:
    """`extra`: per-player covariates at the held-out season keyed by player_id (SeasonFrame.extra); players of
    the ratings table not in it get NaN -> the Age term treats them through the frame's own alignment."""
    p = map_o.n_params
    d = rat.df.copy()
    poss = d.poss.to_numpy(dtype=float)
    ex = None
    if extra is not None:
        ex = d[["player_id"]].merge(extra, on="player_id", how="left")
        ex["age"] = ex.age.fillna(float(extra.age.median()))
        if "moved" in ex.columns:
            ex["moved"] = ex.moved.fillna(0.0)
    ex_o = ex_d = ex
    if ex is not None and "prior_o" in d.columns:
        ex_o = ex.assign(prior=d.prior_o.to_numpy(dtype=float) / scale_o)
        ex_d = ex.assign(prior=d.prior_d.to_numpy(dtype=float) / scale_d)
    d["o"] = map_o.apply(d.o.to_numpy(), poss, theta[:p], scale_o, ex_o)
    d["d"] = map_d.apply(d.d.to_numpy(), poss, theta[p:], scale_d, ex_d)
    zero = np.zeros(1)
    fo = float(map_o.apply(np.array([rat.fill_o]), zero, theta[:p], scale_o)[0])
    fd = float(map_d.apply(np.array([rat.fill_d]), zero, theta[p:], scale_d)[0])
    return Ratings(d, fill_o=fo, fill_d=fd)


def _param_row(name, system, k, h, map_o, map_d, D, th) -> dict:
    p = map_o.n_params
    return dict(system=name, base=system, k=k, held_out=h, map_o=map_o.name, map_d=map_d.name,
                scale_o=D.scale_o, scale_d=D.scale_d,
                **{f"o{j}": float(v) for j, v in enumerate(th[:p])}, **{f"d{j}": float(v) for j, v in enumerate(th[p:])})


def evaluate(dump: pd.DataFrame, frames: dict, system: str, k: int, map_o: SideMap, map_d: SideMap, name: str,
             ridge: float = 0.0, lam: float = 0.0, level: str = "home", min_poss: float = 1000.0):
    """Leave-one-season-out: fit the map on the other seasons' team-game residuals, apply it to H, score H
    with the criterion's own scorer.  Returns (result rows in RESULT_COLUMNS, the per-season parameters,
    held_out = -1 for the all-seasons fit)."""
    D = build_design(dump, frames, system, k, map_o, map_d, min_poss)
    rows, params = [], []
    for h, f in frames.items():
        th = fit_theta(D, exclude_h=h, ridge=ridge, map_o=map_o, map_d=map_d)
        rat = mapped_ratings(ratings_for(dump, system, k, h), th, map_o, map_d, D.scale_o, D.scale_d, extra=f.covariates(k))
        p = predict_season(rat, f.wd, level=level)
        rows.append(dict(held_out=h, k=k, train="", system=name, lam=lam, split="all", group="all", **score(p), seconds=0.0))
        params.append(_param_row(name, system, k, h, map_o, map_d, D, th))
    th_all = fit_theta(D, exclude_h=None, ridge=ridge, map_o=map_o, map_d=map_d)
    params.append(_param_row(name, system, k, -1, map_o, map_d, D, th_all))
    return pd.DataFrame(rows)[RESULT_COLUMNS], pd.DataFrame(params)


def unmapped_rows(dump: pd.DataFrame, frames: dict, system: str, k: int, lam: float = 0.0, level: str = "home") -> pd.DataFrame:
    """The base system scored from the dump, so the paired test is against exactly the same fits."""
    rows = []
    for h, f in frames.items():
        p = predict_season(ratings_for(dump, system, k, h), f.wd, level=level)
        rows.append(dict(held_out=h, k=k, train="", system=system, lam=lam, split="all", group="all", **score(p), seconds=0.0))
    return pd.DataFrame(rows)[RESULT_COLUMNS]


# ---------------------------------------------------------------------------------------- 5. the system
def apply_params(row: pd.Series, o: np.ndarray, d: np.ndarray, poss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The map of one parameter row applied to raw-sign offense and defense arrays with the training possessions."""
    map_o, map_d = SideMap.parse(str(row.map_o)), SideMap.parse(str(row.map_d))
    th_o = np.array([float(row[f"o{j}"]) for j in range(map_o.n_params)])
    th_d = np.array([float(row[f"d{j}"]) for j in range(map_d.n_params)])
    poss = np.asarray(poss, dtype=float)
    return (map_o.apply(np.asarray(o, dtype=float), poss, th_o, float(row.scale_o)),
            map_d.apply(np.asarray(d, dtype=float), poss, th_d, float(row.scale_d)))


def params_row(params: pd.DataFrame, system: str, base: str, k: int, held_out: int | None) -> pd.Series:
    """The parameter row for (system, k, held_out); held_out None or absent falls back to the all-seasons fit (-1)."""
    t = params[(params.system == system) & (params.base == base) & (params.k == int(k))]
    if len(t) == 0:
        raise KeyError(f"no calmap parameters for {system} (base {base}) at K={k}")
    if held_out is not None and (t.held_out == int(held_out)).any():
        return t[t.held_out == int(held_out)].iloc[0]
    return t[t.held_out == -1].iloc[0]


@dataclass
class CalMappedSystem:
    """`inner` with the calibration map from a parameter table (scripts/53_calmap.py fit), scored honestly:
    the row for held-out season H was fitted with H left out.  With no current held-out season (the shipped
    ratings) the all-seasons row is used.  K defaults to the training block's length."""
    name: str
    inner: object
    params: pd.DataFrame
    mapped: str            # the map's name in the table (e.g. "mspi_linear+sat")

    def fit(self, train, ctx: Context) -> Ratings:
        r = self.inner.fit(train, ctx)
        k = ctx.current_k if ctx.current_k is not None else len(train)
        row = params_row(self.params, self.mapped, self.inner.name, k, ctx.current_h)
        d = r.df.copy()
        d["o"], d["d"] = apply_params(row, d.o.to_numpy(), d.d.to_numpy(), d.poss.to_numpy())
        fo, fd = apply_params(row, np.array([r.fill_o]), np.array([r.fill_d]), np.zeros(1))
        return Ratings(d, fill_o=float(fo[0]), fill_d=float(fd[0]))
