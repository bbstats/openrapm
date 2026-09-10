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
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .design import SCORE_PHASES
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
            for s in systems:
                # an in-season system names its own training seasons and the part of H it did not see
                train = s.train_for(h, ctx) if hasattr(s, "train_for") else ctx.neighbourhood(h, k)
                if train is None:
                    continue
                t1 = time.time()
                rat = _fit(s, train, ctx, ho.lams[0])
                secs = time.time() - t1
                d = rat.df.copy()
                for c in ("prior_o", "prior_d"):
                    if c not in d.columns:
                        d[c] = np.nan
                q = getattr(s, "cut", None)
                rows.append(d.assign(held_out=int(h), k=int(k), system=s.name, fill_o=rat.fill_o, fill_d=rat.fill_d,
                                     seconds=secs, train=",".join(map(str, train)),
                                     cut=np.nan if q is None else float(q)))
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


def dump_ratings(ho: Holdout, names: list, out: Path, workers: int = 4, rankmap=None, verbose: bool = True,
                 held: list | None = None) -> pd.DataFrame:
    """Every system's ratings for every held-out season and K, one parquet.  Same process layout as
    holdout.run_parallel (spawned workers, BLAS threads pinned).

    `held` restricts the held-out seasons, the way `Holdout.run` already allows; None = all of `ho.seasons()`.
    A hyperparameter search uses it to score on half the seasons and keep the other half untouched."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    held = list(held) if held is not None else ho.seasons()
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


def cut_of(dump: pd.DataFrame, system: str) -> float | None:
    """The cut a dumped in-season system was fit at, or None: the frames it is scored on must match it."""
    if "cut" not in dump.columns:
        return None
    v = dump.loc[dump.system == system, "cut"].dropna()
    return float(v.iloc[0]) if len(v) else None


def train_of(dump: pd.DataFrame, system: str, k: int, h: int) -> list | None:
    """The seasons a dumped system actually trained on for (K, H) -- a kernel system's are not
    `Context.neighbourhood`'s, and the map's covariates are measured on the training seasons."""
    if "train" not in dump.columns:
        return None
    d = dump[(dump.system == system) & (dump.k == k) & (dump.held_out == h)]
    if len(d) == 0 or pd.isna(d.train.iloc[0]):
        return None
    return [int(x) for x in str(d.train.iloc[0]).split(",") if x]


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
    key: np.ndarray | None = None         # team-game index of each row (the rows G aggregates)
    extra: pd.DataFrame | None = None     # per player in `ids` order: covariates AT H (age), for the Age term
    ctx: object = None                    # for covariates that depend on the training block (moved, per K)
    cut: float | None = None              # the share of H the fit had seen; the frame holds the rows AFTER it
    wd_full: object = None                # the whole season's design, for the covariates that need its games
    _turn: dict = field(default_factory=dict, repr=False)   # the turnover columns per K (they cost a second each)

    def turnover(self, k: int, train: list) -> dict | None:
        """Teammate turnover of H with respect to the training block, per player in `extra` order
        (turnover.familiar_share; None without data/cache/teammates.parquet):
          turn   the share of his H teammate-possessions spent with people he never shared 100 possessions
                 with anywhere in the block -- 1.0 for a player whose whole context is new
          turnp  the same against the block's PAST seasons only (the K = 3 block brackets H, so a mover's
                 H+1 season is with his NEW teammates and `turn` reads him as half familiar)
          turna  `turn` measured on HALF of H's games (the `A` half): the within-season control, as HShareA
                 is for HShare -- an exogenous covariate keeps its value on half the games, a leak does not."""
        if k in self._turn:
            return self._turn[k]
        tm = _teammates(self.ctx)
        if tm is None:
            self._turn[k] = None
            return None
        from .turnover import familiar_share, teammates_from_stints
        ids = self.extra.player_id.to_numpy()
        out = {"turn": familiar_share(tm, train, [self.h], player_ids=ids).turnover.fillna(0.0).to_numpy()}
        past = [s for s in train if s < self.h]
        out["turnp"] = (familiar_share(tm, past, [self.h], player_ids=ids).turnover.fillna(0.0).to_numpy()
                        if past else out["turn"].copy())
        g = self.wd.games
        ga = set(g.loc[(g["half"] == "A") & (g["phase"] == "RS"), "game_id"].astype(str))
        from .config import resolve
        from .design import AWAY_SLOTS, HOME_SLOTS
        st = pd.read_parquet(Path(resolve(self.ctx.cfg, "stints")) / f"{self.h}_RS.parquet",
                             columns=[*HOME_SLOTS, *AWAY_SLOTS, "poss_h", "poss_a", "game_id"])
        ta = teammates_from_stints(st[st["game_id"].astype(str).isin(ga)]).assign(season=int(self.h))
        tm_a = pd.concat([tm[tm.season.isin(list(train))], ta], ignore_index=True)
        out["turna"] = familiar_share(tm_a, train, [self.h], player_ids=ids).turnover.fillna(0.0).to_numpy()
        self._turn[k] = out
        return out

    def covariates(self, k: int, train: list | None = None) -> pd.DataFrame | None:
        """`extra` plus the K-dependent ones: `moved` = main team in H differs from the main team in the nearest
        training season he appears in (holdout.by_movers); `turn` / `turnp` / `turna` = teammate turnover of H
        with respect to the block (`turnover`).

        `train`: the seasons the system being mapped actually trained on (`train_of`).  A kernel system's are
        not `Context.neighbourhood`'s, and `tshare` -- the role measured on the training block -- would
        otherwise be read off seasons the fit never saw.
        """
        if self.extra is None or self.ctx is None:
            return self.extra
        ex = self.extra.copy()
        train = list(train) if train else self.ctx.neighbourhood(self.h, k)
        t = self.turnover(k, train)
        if t is not None:
            for key, val in t.items():
                ex[key] = val
        now = self.ctx.main_team(self.h)
        before: dict = {}
        for s in sorted(train, key=lambda s: abs(s - self.h)):
            for pid, t in self.ctx.main_team(s).items():
                before.setdefault(pid, t)
        ex["moved"] = [1.0 if (q in now and q in before and now[q] != before[q]) else 0.0 for q in ex.player_id]
        ex["blk_team"] = [int(before.get(q, -1)) for q in ex.player_id]
        ri = self.ctx.role_inputs
        if ri is not None and self.cut is not None and int(self.h) in train:
            # the training block includes the season being scored, up to the cut: the share SO FAR, not the
            # share it ends at (roles.cut_role_inputs, the table the prior was built from)
            from .inseason import keep_games
            from .spm import cut_inputs_cached
            ri = cut_inputs_cached(self.ctx, keep_games(self.wd_full, int(self.h), float(self.cut)))
        if ri is not None:                       # the same role measured on the TRAINING block instead of H
            t = ri[ri.season.isin(list(train)) & (ri.games > 0)].groupby("player_id")[["poss_on", "team_poss"]].sum()
            sh = (t.poss_on / t.team_poss.replace(0.0, np.nan)).to_dict()
            ex["tshare"] = [float(sh.get(q, 0.0) or 0.0) for q in ex.player_id]
        return ex

    @classmethod
    def build(cls, ctx: Context, h: int, level: str = "home", cut: float | None = None) -> "SeasonFrame":
        """`cut`: score the season from that share of it onward -- the rows an in-season fit at that cut has
        not seen (holdout.cut_season).  The level, the team-game grouping and every covariate are then built
        on those rows alone, so the map agrees with the runner row for row."""
        wd_full = ctx.design([h], "pts", SCORE_PHASES)   # the map is fitted on what the criterion scores
        wd = wd_full
        if cut is not None and float(cut) < 1.0:
            from .holdout import cut_season
            wd = cut_season(wd_full, int(h), float(cut))
        extra = None
        if ctx.role_inputs is not None:
            ri = ctx.role_inputs[ctx.role_inputs.season == h][["player_id", "age", "poss_pct", "gs_pct", "poss_on", "team_poss"]]
            extra = pd.DataFrame({"player_id": wd.spec.ps_table["player_id"].to_numpy()}).merge(ri, on="player_id", how="left")
            extra["age"] = extra.age.fillna(float(ri.age.median()) if len(ri) else 27.0)
            for c in ("poss_pct", "gs_pct", "poss_on"):        # the held-out season's ROLE: known at prediction
                extra[c] = extra[c].fillna(0.0)             # time, like the lineups themselves and the age
            extra["team_poss"] = extra.pop("team_poss").fillna(0.0) if "team_poss" in extra.columns else 0.0
            # the same share measured on HALF of H only: within-season feedback (play badly, sit down) can
            # reach the whole-season share but not the other half's
            gp = wd.game_poss.merge(wd.games[["game_idx", "half"]], on="game_idx", how="left")
            pa = gp[gp.half == "A"].groupby("psx_idx")["poss_off"].sum()
            per_psx = np.zeros(wd.spec.n_psx)
            per_psx[pa.index.to_numpy()] = pa.to_numpy(dtype=float)
            ps_poss_a = np.bincount(wd.spec.ps_of_psx, weights=per_psx, minlength=wd.spec.n_ps)
            tp = extra["team_poss"].to_numpy(dtype=float)
            extra["share_a"] = np.where(tp > 0, 2.0 * ps_poss_a / np.where(tp > 0, tp, 1.0), 0.0)
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
                   A=A, L=L, G=G, wg=wg, key=key, extra=extra, ctx=ctx,
                   cut=None if cut is None else float(cut), wd_full=wd_full)

    def profiled(self, C: np.ndarray) -> np.ndarray:
        """Columns with the season's level profiled out (what the criterion's refit does to any contribution)."""
        return C - self.A @ (self.L @ C)

    def game(self, C: np.ndarray) -> np.ndarray:
        return np.asarray(self.G @ C)


def _teammates(ctx) -> pd.DataFrame | None:
    """data/cache/teammates.parquet (turnover.build_teammates), read once per Context; None if not built."""
    from .turnover import cached_table
    return cached_table(ctx)


def load_frames(ctx: Context, seasons, level: str = "home", verbose: bool = True, cut: float | None = None) -> dict:
    t0 = time.time()
    out = {int(h): SeasonFrame.build(ctx, int(h), level, cut=cut) for h in seasons}
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


class Turn(Exposure):
    """Teammate turnover of H with respect to the training block (SeasonFrame.turnover; `extra[key]`).
    `key`: "turn" = against the whole block, "turnp" = against its past seasons only, "turna" = the half-season
    control.  `on`: None = a level in the turnover; "x" = a slope on the standardised rating (does a rating carry
    less into a new context?); "prior" = a slope on the prior part alone (is it the box line or the possession
    evidence that fails to travel?).  A player the block never saw has no rating to carry, so the term is 0."""
    def __init__(self, key: str = "turn", on: str | None = None):
        self.key, self.on = key, on
        self.name = key + ("" if on is None else ("x" if on == "x" else "prior"))
        self.n_params = 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or self.key not in extra.columns:
            return np.zeros((len(p), 1))
        t = np.nan_to_num(np.asarray(extra[self.key], dtype=float)) * (p > 0)
        if self.on == "x":
            return (t * (np.zeros(len(p)) if x is None else np.asarray(x, dtype=float)))[:, None]
        if self.on == "prior":
            if "prior" not in extra.columns:
                return np.zeros((len(p), 1))
            return (t * np.nan_to_num(np.asarray(extra["prior"], dtype=float)))[:, None]
        return t[:, None]


class MovedPrior(Exposure):
    """moved x the prior part of the rating: the binary-rule twin of Turn("turn", "prior")."""
    name, n_params = "mprior", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "moved" not in extra.columns or "prior" not in extra.columns:
            return np.zeros((len(p), 1))
        m = np.nan_to_num(np.asarray(extra["moved"], dtype=float)) * (p > 0)
        return (m * np.nan_to_num(np.asarray(extra["prior"], dtype=float)))[:, None]


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


class HShare(Exposure):
    """A level in the player's role in the HELD-OUT season: c x poss_pct / 0.1 (share of his team's possessions
    while he is on the floor, `roles.window_inputs`).

    The held-out season's lineups are an input of the criterion, so how much a player plays in H is known at
    prediction time -- as much as his age is.  A rating fitted on a bench role and carried into a starter's
    role is the case this asks about.  Not a rating of the window (the window has no role at H): a
    prediction-time term, like the Age one.
    """
    name, n_params = "hshare", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "poss_pct" not in extra.columns:
            return np.zeros((len(p), 1))
        return (np.nan_to_num(np.asarray(extra["poss_pct"], dtype=float)) / 0.1)[:, None]


class XHShare(Exposure):
    """b x (share at H) / 0.1: does a rating carry further for a player who plays a big role in H?"""
    name, n_params = "xhshare", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "poss_pct" not in extra.columns or x is None:
            return np.zeros((len(p), 1))
        return (np.asarray(x, dtype=float) * np.nan_to_num(np.asarray(extra["poss_pct"], dtype=float)) / 0.1)[:, None]


class HShareA(Exposure):
    """`HShare` from HALF of the held-out season's games (the `A` half), doubled: the control for within-season
    feedback -- a player benched for playing badly loses whole-season minutes, but the other half's minutes
    were spent before anyone knew."""
    name, n_params = "hsharea", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "share_a" not in extra.columns:
            return np.zeros((len(p), 1))
        return (np.nan_to_num(np.asarray(extra["share_a"], dtype=float)) / 0.1)[:, None]


class TShare(Exposure):
    """`HShare` measured on the TRAINING block instead of the held-out season: the same role variable with
    nothing of H in it, the control that says whether the H-season term is a leak or a real signal."""
    name, n_params = "tshare", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "tshare" not in extra.columns:
            return np.zeros((len(p), 1))
        return (np.nan_to_num(np.asarray(extra["tshare"], dtype=float)) / 0.1)[:, None]


class XTShare(Exposure):
    """b x the training-block role: does a rating carry further for a player who was a starter when it was fit?"""
    name, n_params = "xtshare", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "tshare" not in extra.columns or x is None:
            return np.zeros((len(p), 1))
        return (np.asarray(x, dtype=float) * np.nan_to_num(np.asarray(extra["tshare"], dtype=float)) / 0.1)[:, None]


class HStarts(Exposure):
    """A level in the player's games-started share at H."""
    name, n_params = "hgs", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "gs_pct" not in extra.columns:
            return np.zeros((len(p), 1))
        return np.nan_to_num(np.asarray(extra["gs_pct"], dtype=float))[:, None]


class HGrow(Exposure):
    """A level in how much the player's role GREW into the held-out season: log((poss at H + 200) /
    (his possessions per training season + 200)).  `extra["grow"]` is set by `build_design`."""
    name, n_params = "hgrow", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "grow" not in extra.columns:
            return np.zeros((len(p), 1))
        return np.nan_to_num(np.asarray(extra["grow"], dtype=float))[:, None]


class XHGrow(Exposure):
    """b x the role growth: a rating carried into a bigger role, re-weighted."""
    name, n_params = "xhgrow", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "grow" not in extra.columns or x is None:
            return np.zeros((len(p), 1))
        return (np.asarray(x, dtype=float) * np.nan_to_num(np.asarray(extra["grow"], dtype=float)))[:, None]


class TeamMean(Exposure):
    """c x (the mean rating of the player's TRAINING-BLOCK teammates, him left out), standardised.

    The five ratings of a team-game come out of one training block, where a team that outscored its true
    strength lifts everyone who played for it: those errors are correlated, and the sum of five of them
    carries the team's share five times.  This is that share as a per-player column, so the map can take
    some of it back off.  `extra["team"]` is set by `build_design` / `mapped_ratings` from the block's teams
    (`SeasonFrame.covariates`'s `blk_team`) and the dump's own ratings; without it the term is 0.
    """
    name, n_params = "team", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "team" not in extra.columns:
            return np.zeros((len(p), 1))
        return np.nan_to_num(np.asarray(extra["team"], dtype=float))[:, None]


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


class Prior2(Exposure):
    """c x prior^2 (standardised): a bend in the prior's re-weighting."""
    name, n_params = "prior2", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "prior" not in extra.columns:
            return np.zeros((len(p), 1))
        v = np.nan_to_num(np.asarray(extra["prior"], dtype=float))
        return (v * v)[:, None]


class PriorAge(Exposure):
    """c x prior x (age - 27) / 5: the prior's weight by age (a young player's box line means more or less)."""
    name, n_params = "priorage", 1

    def basis(self, poss, extra=None, x=None):
        p = np.asarray(poss, dtype=float)
        if extra is None or "prior" not in extra.columns or "age" not in extra.columns:
            return np.zeros((len(p), 1))
        v = np.nan_to_num(np.asarray(extra["prior"], dtype=float))
        return (v * (np.asarray(extra["age"], dtype=float) - 27.0) / 5.0)[:, None]


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
                                 AgeSat(), XSat(), XSat(300), XSat(3000), XLog(), XAge(), Prior(), PriorSat(), Prior2(),
                                 PriorAge(), TeamMean(), HShare(), XHShare(), HStarts(), HGrow(), XHGrow(), TShare(), HShareA(), XTShare(),
                                 MovedPrior(), *(Turn(key, on) for key in ("turn", "turnp", "turna") for on in (None, "x", "prior")))}


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


class TeamBend:
    """A bend in the TEAM's summed rating, fitted after the per-player map.

    The map is a function of one player's rating, so the criterion's prediction for a team-game is the
    possession-weighted SUM of the five on the floor: whatever the map does, the total is linear in it.  A
    bend in that total -- extreme team-games pulled in, the middle stretched -- is not reachable per player,
    and out of season it is worth 0.07-0.08 per 100 on every board tried (FINDINGS 21.20).  `basis` is in the
    mapped total u, standardised by its own weighted RMS, and gamma = (1, 0, ...) is the map left alone.

    It is a PREDICTION-TIME term, like the age term: a rating carries no team, so it cannot ship in the board.
    """
    name, n_params = "none", 0
    row_powers: tuple = ()          # odd powers of the STINT-level contribution to add as extra columns

    def basis(self, u: np.ndarray, s: float) -> np.ndarray:
        return np.asarray(u, dtype=float)[:, None]

    def columns(self, u: np.ndarray, s: float, rows: np.ndarray | None = None) -> np.ndarray:
        """The team-level basis with the row-level columns (`row_columns`) beside it."""
        B = self.basis(u, s)
        return B if rows is None or not self.row_powers else np.column_stack([B, rows])


class CubicBend(TeamBend):
    """g(u) = a u + b u^3 / s^2."""
    name, n_params = "cubic", 2

    def basis(self, u, s):
        u = np.asarray(u, dtype=float)
        return np.column_stack([u, u ** 3 / s ** 2])


class RowCubicBend(CubicBend):
    """The team-game cubic plus the same cubic taken at STINT level and then aggregated: g = a u + b u^3/s^2
    + c mean(c_row^3)/s_row^2.

    (mean c)^3 and mean(c^3) differ by the spread of the lineups inside the team-game, so the pair separates
    "this team-game's total is extreme" from "the lineups inside it were extreme" -- and the criterion wants
    both, with opposite signs (FINDINGS 21.22).  The row column alone is worse than the team one; together
    they are worth 0.21 per 100 more than the team cubic.
    """
    name, n_params, row_powers = "rowcubic", 3, (3,)


class QuadBend(TeamBend):
    """g(u) = a u + b u|u| / s: the same compression with a sign asymmetry allowed."""
    name, n_params = "quad", 2

    def basis(self, u, s):
        u = np.asarray(u, dtype=float)
        return np.column_stack([u, u * np.abs(u) / s])


class TanhBend(TeamBend):
    """g(u) = a u + b (tanh(u / s) - u / s) s: a saturating compression, no runaway cubic tail."""
    name, n_params = "tanh", 2

    def basis(self, u, s):
        u = np.asarray(u, dtype=float)
        return np.column_stack([u, (np.tanh(u / s) - u / s) * s])


BENDS = {b.name: b for b in (TeamBend(), CubicBend(), RowCubicBend(), QuadBend(), TanhBend())}


def parse_maps(fam: str):
    """"<family_o>[:<family_d>][|<bend>]" -> (map_o, map_d, bend).  The name a run logs is the same string
    with ":" and "|" replaced by "_"."""
    rest, _, bend = fam.partition("|")
    fo, fd = (rest.split(":") + [None])[:2]
    return SideMap.parse(fo), SideMap.parse(fd or fo), BENDS[bend or "none"]


def fit_bend(D: "Design", th: np.ndarray, bend: TeamBend, exclude_h=None, rows: np.ndarray | None = None
             ) -> tuple[np.ndarray, float]:
    """The bend's parameters on every season but `exclude_h`, given the map's own parameters `th`.  `rows`:
    the stacked row-level columns of `row_columns`, in D's own row order."""
    sel = np.ones(len(D.h), dtype=bool) if exclude_h is None else D.h != exclude_h
    u, y, w = D.X[sel] @ th, D.y[sel], D.w[sel]
    s = float(np.sqrt(np.average(u ** 2, weights=w))) or 1.0
    B = bend.columns(u, s, None if rows is None else rows[sel])
    BtW = (B * w[:, None]).T
    return np.linalg.solve(BtW @ B, BtW @ y), s


def row_columns(dump: pd.DataFrame, frames: dict, system: str, k: int, map_o: SideMap, map_d: SideMap,
                D: "Design", th: np.ndarray, powers) -> tuple[np.ndarray, dict, float]:
    """Per team-game, the aggregated odd powers of the STINT-level mapped contribution, under the map
    parameters `th`.  Returns (the columns stacked in D's row order, the same per season, the standardising
    RMS of the stint contribution).  Rebuilt per held-out season, because it is a NONLINEAR function of the
    map's parameters and those are refit each time."""
    prof, ws = {}, []
    for h, f in frames.items():
        rat = mapped_ratings(ratings_for(dump, system, k, h), th, map_o, map_d, D.scale_o, D.scale_d,
                             extra=f.covariates(k, train_of(dump, system, k, h)))
        r = rat.aligned(f.ids)
        c = np.asarray(f.Zo @ np.nan_to_num(r.o.to_numpy(dtype=float))) +             np.asarray(f.Zd @ np.nan_to_num(r.d.to_numpy(dtype=float)))
        prof[h] = f.profiled(c)
        ws.append(f.w)
    s = float(np.sqrt(np.average(np.concatenate([prof[h] for h in frames]) ** 2,
                                 weights=np.concatenate(ws)))) or 1.0
    per = {h: np.column_stack([frames[h].game(prof[h] ** q / s ** (q - 1)) for q in powers]) for h in frames}
    return np.vstack([per[h] for h in frames]), per, s


# ---------------------------------------------------------------------------------------- 4. fitting and scoring
def _side_scale(dump: pd.DataFrame, system: str, k: int, side: str, min_poss: float = 1000.0) -> float:
    d = dump[(dump.system == system) & (dump.k == k) & (dump.poss >= min_poss)]
    return float(np.sqrt(np.average(d[side].to_numpy() ** 2, weights=d.poss.to_numpy()))) or 1.0


def team_mean(x: np.ndarray, poss: np.ndarray, team: np.ndarray) -> np.ndarray:
    """Per player, the possession-weighted mean rating of the OTHERS on his training-block team (0 with no
    team, or when he is the only one).  Leaving him out matters: with him in, the column is partly his own
    rating and the map cannot tell the two apart."""
    x = np.nan_to_num(np.asarray(x, dtype=float))
    w = np.nan_to_num(np.asarray(poss, dtype=float))
    t = np.asarray(team)
    idx, uniq = pd.factorize(t)
    ok = idx >= 0
    sw = np.bincount(idx[ok], weights=w[ok], minlength=len(uniq))
    sx = np.bincount(idx[ok], weights=(w * x)[ok], minlength=len(uniq))
    den = sw[idx] - w
    out = np.where(den > 0, (sx[idx] - w * x) / np.where(den > 0, den, 1.0), 0.0)
    return np.where(ok & (t != -1), out, 0.0)


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
        ex = f.covariates(k, train_of(dump, system, k, h))
        ex_o = ex_d = ex
        if ex is not None and "prior_o" in r.columns:
            ex_o = ex.assign(prior=r.prior_o.to_numpy(dtype=float) / so)
            ex_d = ex.assign(prior=r.prior_d.to_numpy(dtype=float) / sd)
        if ex is not None and "poss_on" in ex.columns:
            hp = np.nan_to_num(np.asarray(ex["poss_on"], dtype=float))
            ex_o = ex_o.assign(grow=np.log((hp + 200.0) / (poss / max(k, 1) + 200.0)))
            ex_d = ex_d.assign(grow=ex_o["grow"].to_numpy())
        if ex is not None and "blk_team" in ex.columns:
            tm = ex["blk_team"].to_numpy()
            ex_o = ex_o.assign(team=team_mean(r.o.to_numpy(dtype=float), poss, tm) / so)
            ex_d = ex_d.assign(team=team_mean(r.d.to_numpy(dtype=float), poss, tm) / sd)
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
        for c in ("moved", "turn", "turnp", "turna"):
            if c in ex.columns:
                ex[c] = ex[c].fillna(0.0)
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


def _param_row(name, system, k, h, map_o, map_d, D, th, bend=None, gamma=None, s_u=np.nan) -> dict:
    p = map_o.n_params
    row = dict(system=name, base=system, k=k, held_out=h, map_o=map_o.name, map_d=map_d.name,
               scale_o=D.scale_o, scale_d=D.scale_d,
               **{f"o{j}": float(v) for j, v in enumerate(th[:p])}, **{f"d{j}": float(v) for j, v in enumerate(th[p:])})
    if bend is not None and bend.n_params:
        row.update(bend=bend.name, scale_u=float(s_u), **{f"g{j}": float(v) for j, v in enumerate(np.atleast_1d(gamma))})
    return row


def bent_prediction(p, f: SeasonFrame, bend: TeamBend, gamma, s_u: float, level: str = "home",
                    rows: np.ndarray | None = None):
    """`predict_season`'s prediction with the team-game total bent: each row's contribution takes its own
    team-game's g(u) - u, so the team-game total is exactly g(u) and the season's level is refit around it."""
    from .holdout import _wls
    c = p.c_off + p.c_def
    u = f.game(f.profiled(c))
    delta = bend.columns(u, s_u, rows) @ np.asarray(gamma, dtype=float) - u
    c2 = c + delta[f.key]
    pred = f.A @ _wls(f.A, f.y - c2, f.w) + c2
    return replace(p, pred=pred, c_off=p.c_off + delta[f.key], c_def=p.c_def)


def evaluate(dump: pd.DataFrame, frames: dict, system: str, k: int, map_o: SideMap, map_d: SideMap, name: str,
             ridge: float = 0.0, lam: float = 0.0, level: str = "home", min_poss: float = 1000.0,
             bend: TeamBend | None = None):
    """Leave-one-season-out: fit the map on the other seasons' team-game residuals, apply it to H, score H
    with the criterion's own scorer.  Returns (result rows in RESULT_COLUMNS, the per-season parameters,
    held_out = -1 for the all-seasons fit).  `bend`: a TeamBend fitted on the same other seasons, after the
    map, on the team-game total."""
    D = build_design(dump, frames, system, k, map_o, map_d, min_poss)
    rows, params = [], []
    for h, f in frames.items():
        th = fit_theta(D, exclude_h=h, ridge=ridge, map_o=map_o, map_d=map_d)
        rat = mapped_ratings(ratings_for(dump, system, k, h), th, map_o, map_d, D.scale_o, D.scale_d,
                             extra=f.covariates(k, train_of(dump, system, k, h)))
        p = predict_season(rat, f.wd, level=level)
        gamma, s_u = (None, np.nan)
        if bend is not None and bend.n_params:
            rcols = per_h = None
            if bend.row_powers:
                rcols, per_h, _ = row_columns(dump, frames, system, k, map_o, map_d, D, th, bend.row_powers)
            gamma, s_u = fit_bend(D, th, bend, exclude_h=h, rows=rcols)
            p = bent_prediction(p, f, bend, gamma, s_u, level=level, rows=None if per_h is None else per_h[h])
        rows.append(dict(held_out=h, k=k, train="", system=name, lam=lam, cut=f.cut if f.cut is not None else np.nan,
                         split="all", group="all", **score(p), seconds=0.0))
        params.append(_param_row(name, system, k, h, map_o, map_d, D, th, bend, gamma, s_u))
    th_all = fit_theta(D, exclude_h=None, ridge=ridge, map_o=map_o, map_d=map_d)
    g_all, s_all = (None, np.nan)
    if bend is not None and bend.n_params:
        rows_all = None
        if bend.row_powers:
            rows_all, _, _ = row_columns(dump, frames, system, k, map_o, map_d, D, th_all, bend.row_powers)
        g_all, s_all = fit_bend(D, th_all, bend, rows=rows_all)
    params.append(_param_row(name, system, k, -1, map_o, map_d, D, th_all, bend, g_all, s_all))
    return pd.DataFrame(rows)[RESULT_COLUMNS], pd.DataFrame(params)


def unmapped_rows(dump: pd.DataFrame, frames: dict, system: str, k: int, lam: float = 0.0, level: str = "home") -> pd.DataFrame:
    """The base system scored from the dump, so the paired test is against exactly the same fits."""
    rows = []
    for h, f in frames.items():
        p = predict_season(ratings_for(dump, system, k, h), f.wd, level=level)
        rows.append(dict(held_out=h, k=k, train="", system=system, lam=lam, cut=f.cut if f.cut is not None else np.nan,
                         split="all", group="all", **score(p), seconds=0.0))
    return pd.DataFrame(rows)[RESULT_COLUMNS]


# ---------------------------------------------------------------------------------------- 5. the system
def apply_params(row: pd.Series, o: np.ndarray, d: np.ndarray, poss: np.ndarray, prior_o=None, prior_d=None,
                 extra: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The map of one parameter row applied to raw-sign offense and defense arrays with the training possessions.
    `prior_o` / `prior_d` (raw sign) feed the map's prior terms; `extra` any other per-player covariates (age)."""
    map_o, map_d = SideMap.parse(str(row.map_o)), SideMap.parse(str(row.map_d))
    th_o = np.array([float(row[f"o{j}"]) for j in range(map_o.n_params)])
    th_d = np.array([float(row[f"d{j}"]) for j in range(map_d.n_params)])
    poss = np.asarray(poss, dtype=float)
    ex_o = ex_d = extra
    if prior_o is not None:
        base = extra if extra is not None else pd.DataFrame(index=range(len(poss)))
        ex_o = base.assign(prior=np.asarray(prior_o, dtype=float) / float(row.scale_o))
        ex_d = base.assign(prior=np.asarray(prior_d, dtype=float) / float(row.scale_d))
    return (map_o.apply(np.asarray(o, dtype=float), poss, th_o, float(row.scale_o), ex_o),
            map_d.apply(np.asarray(d, dtype=float), poss, th_d, float(row.scale_d), ex_d))


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
        po = d.prior_o.to_numpy() if "prior_o" in d.columns else None
        pdd = d.prior_d.to_numpy() if "prior_d" in d.columns else None
        d["o"], d["d"] = apply_params(row, d.o.to_numpy(), d.d.to_numpy(), d.poss.to_numpy(), po, pdd)
        fo, fd = apply_params(row, np.array([r.fill_o]), np.array([r.fill_d]), np.zeros(1))
        return Ratings(d, fill_o=float(fo[0]), fill_d=float(fd[0]))
