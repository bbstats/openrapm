"""The out-of-season criterion, once.

Hold out season H.  Train on the K seasons nearest to H, balanced either side of it and never
including it.  Predict every stint of H from the ten players on the floor, with only an intercept
and a home term refit on H.  Score possession-weighted squared error in points per 100 -- at stint
level, and again after summing each team's points within a game so that noise inside a team's ~100
possessions cancels before it is squared.

This is the one test in the project that is both EXTERNAL to the model (it scores actual points in a
season the ratings never saw) and LEGAL to select on (it uses no outside data, unlike the consensus,
which is validation only).  Symmetric-around-H rather than train-on-a-block-predict-the-flanks
because the held-out season is then identical across every method and every K, so every comparison
is exactly paired and aging cancels.  FINDINGS.md sections 16-17 are the record of what it has
decided so far.

How to use it:

    ctx = Context.load(cfg)
    ho = Holdout.from_config(cfg)
    res = ho.run([TableSystem("mine", table), TableSystem("theirs", other)], ctx)
    print(report(res, ref="theirs"))

A System is anything with a name and `fit(train_seasons, ctx) -> Ratings`.  `TableSystem` scores a
table of ratings you already have, which is how the tests check the runner against a known truth;
`ReplacementSystem` gives a player the training block never saw a replacement level.

Conventions, applied identically to every system so none is favoured:
  * Ratings are in the model's RAW sign: `o` adds points scored, `d` adds points ALLOWED.
  * Players absent from the training block score 0 (the average player).
  * The level (intercept, home) is refit on the held-out season.  Scoring drifts across eras and
    that is not what is under test.
  * Team-game aggregation converts rates back to points with `rows["poss"]`, never with `w`, because
    `w` is `poss * gt_weight` and the project runs gt_weight at 0.5 and 0 for robustness.
  * Lambda is FIXED across systems.  Tuning it on this criterion and then reporting this criterion
    would be circular.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .data.designs import build_window
from .design import SEASON_PHASES, WindowData
from .windows import window_label, window_seasons

RESULT_COLUMNS = ["held_out", "k", "train", "system", "lam", "cut", "split", "group", "n", "mse", "base", "calib",
                  "calib_side", "scale", "scale_off", "scale_def", "covered", "tg", "tg_base", "tg_n", "seconds"]


# ---------------------------------------------------------------------------------------- ratings
@dataclass(frozen=True)
class Ratings:
    """One row per player: `o`, `d` in raw sign, `poss` = training regular-season possessions.
    `fill_o` / `fill_d` are what a player ABSENT from the training block scores: 0 (the average player,
    the criterion's default) unless a system sets a replacement level (ReplacementSystem)."""
    df: pd.DataFrame
    fill_o: float = 0.0
    fill_d: float = 0.0

    def aligned(self, player_ids) -> pd.DataFrame:
        """The ratings in the order of `player_ids`; absent players score the fill (0 = the average player)."""
        r = pd.DataFrame({"player_id": np.asarray(player_ids)}).merge(self.df, on="player_id", how="left")
        return r.fillna({"o": float(self.fill_o), "d": float(self.fill_d), "poss": 0.0})


class System(Protocol):
    name: str

    def fit(self, train: list[int], ctx: "Context") -> Ratings: ...


# ---------------------------------------------------------------------------------------- context
def default_loader(seasons, cfg, target, phases=SEASON_PHASES, counter_cols=None, min_den=0.0):
    return build_window(list(seasons), cfg, phases=tuple(phases), target=target, counter_cols=counter_cols,
                        min_den=min_den)


@dataclass
class Context:
    """Everything a System needs that is not the training seasons.  Built once per run.

    `loader(seasons, cfg, target)` builds a WindowData; the default reads the real stints, the tests
    inject one that slices simulated frames.  `roles` / `role_inputs` feed the bench and rookie splits.
    `current_h` is the one piece of state: the runner sets it before each fit so a system can be kept
    off every window the held-out season touches.
    """
    cfg: dict
    loader: Callable = default_loader
    current_h: int | None = None
    current_k: int | None = None
    cache_size: int = 4
    roles: pd.DataFrame | None = None          # data/cache/roles.parquet (roles.build_roles)
    role_inputs: pd.DataFrame | None = None    # roles.player_season_inputs(roles)
    _cache: dict = field(default_factory=dict, repr=False)
    _teams: dict = field(default_factory=dict, repr=False)
    _bigness: dict = field(default_factory=dict, repr=False)
    _bench: dict = field(default_factory=dict, repr=False)
    reports: list = field(default_factory=list, repr=False)

    @classmethod
    def load(cls, cfg, loader=None) -> "Context":
        ctx = cls(cfg=cfg, loader=loader or default_loader)
        from .data.roles import player_season_inputs, roles_path
        rp = roles_path(cfg)
        if rp.exists():
            ctx.roles = pd.read_parquet(rp)
            ctx.role_inputs = player_season_inputs(ctx.roles, cap=float(cfg.get("roles", {}).get("share_cap", 0.9)))
        return ctx

    def bigness(self, season: int, min_minutes: float = 500.0) -> dict:
        """player_id -> True if in the top tercile of the big-man score (orb + blk + 0.3 drb - 0.5 ast - 0.4 fg3m
        per 36, the definition of the retired scripts/22_vs_consensus.py) among players with `min_minutes`."""
        if season not in self._bigness:
            from .data.box_scores import season_box
            b = season_box([season], list(SEASON_PHASES), self.cfg)
            g = b.groupby("player_id")[["orb", "blk", "drb", "ast", "fg3m", "minutes"]].sum()
            g = g[g.minutes >= min_minutes]
            per36 = g[["orb", "blk", "drb", "ast", "fg3m"]].div(g.minutes.clip(lower=1), axis=0) * 36
            score = per36.orb + per36.blk + 0.3 * per36.drb - 0.5 * per36.ast - 0.4 * per36.fg3m
            cut = float(score.quantile(2 / 3)) if len(score) else np.inf
            self._bigness[season] = {int(k): bool(v >= cut) for k, v in score.items()}
        return self._bigness[season]

    def bench(self, season: int) -> dict:
        """player_id -> True if his games-started share that season is below cfg holdout.bench_gs_pct."""
        if season not in self._bench:
            thr = float(self.cfg.get("holdout", {}).get("bench_gs_pct", 0.5))
            if self.role_inputs is None:
                self._bench[season] = {}
            else:
                r = self.role_inputs[(self.role_inputs.season == season) & (self.role_inputs.games > 0)]
                self._bench[season] = {int(k): bool(v < thr) for k, v in zip(r.player_id, r.gs_pct)}
        return self._bench[season]

    @property
    def features(self) -> list:
        return list(self.cfg["box_counts"])

    @property
    def win_of(self) -> dict:
        return {s: window_label(list(range(w[0], w[1] + 1)))
                for w in window_seasons(self.cfg) for s in range(w[0], w[1] + 1)}

    def neighbourhood(self, h: int, k: int) -> list[int]:
        """The K seasons nearest H, excluding H, balanced either side where the era allows."""
        s0, s1 = int(self.cfg["first_season"]), int(self.cfg["last_season"])
        out, d = [], 1
        while len(out) < k and d <= (s1 - s0):
            for s in (h - d, h + d):
                if s0 <= s <= s1 and len(out) < k:
                    out.append(s)
            d += 1
        return sorted(out)

    def labels(self, train, panel: pd.DataFrame | None = None) -> set:
        """Window labels a prior must stay off: the training block's and the held-out season's.

        `panel`: a role panel whose own labels are used instead of the configured windows, so a per-season
        panel (scripts/49_role_panel.py --season) excludes exactly the seasons and not their 3-season
        neighbours.  On a block panel the two agree by construction."""
        seasons = list(train) + ([self.current_h] if self.current_h is not None else [])
        if panel is not None:
            from .windows import labels_covering
            return labels_covering(panel.window.unique(), seasons)
        return {self.win_of[s] for s in seasons}

    def design(self, seasons, target="pts", phases=SEASON_PHASES, counter_cols=None, min_den=0.0) -> WindowData:
        """A cached design.  `target` is a design.TARGETS key, or a callable
        (seasons, cfg, wd_pts) -> WindowData | (WindowData, report) for a derived target.  `phases` names the
        phases whose rows enter the design and defaults to `design.SEASON_PHASES`, both of them -- for the
        fit AND for the rows the criterion scores, because a season is one entity.
        `counter_cols` keeps only those per-possession counters (data.designs.build_window_cached); it is
        honoured by the default loader only, and it is part of the cache key."""
        phases = tuple(phases)
        own = self.loader is default_loader
        cc = None if counter_cols is None or not own else tuple(sorted(counter_cols))
        md = float(min_den) if own else 0.0
        key = (tuple(int(s) for s in seasons), target if isinstance(target, str) else getattr(target, "__name__", repr(target)),
               *(() if phases == SEASON_PHASES else (phases,)), *(() if cc is None else (cc,)),
               *(() if not md else (md,)))
        if key not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.clear()
            if isinstance(target, str):
                kw = {}
                if cc is not None:
                    kw['counter_cols'] = cc
                if md:
                    kw['min_den'] = md
                wd = self.loader(list(seasons), self.cfg, target) if phases == SEASON_PHASES and not kw else \
                    self.loader(list(seasons), self.cfg, target, phases, **kw)
            else:
                wd = target(list(seasons), self.cfg, self.design(seasons, "pts", phases))
                if isinstance(wd, tuple):
                    wd, rep = wd
                    self.reports.append(dict(seasons=key[0], target=key[1], report=rep))
            self._cache[key] = wd
        return self._cache[key]

    def main_team(self, season: int) -> dict:
        """player_id -> the team he played the most minutes for that season (box scores)."""
        if season not in self._teams:
            from .data.box_scores import season_box
            b = season_box([season], list(SEASON_PHASES), self.cfg).groupby(
                ["player_id", "team_id"], as_index=False)["minutes"].sum()
            b = b.sort_values("minutes").drop_duplicates("player_id", keep="last")
            self._teams[season] = dict(zip(b.player_id.astype(int), b.team_id.astype(int)))
        return self._teams[season]


# ---------------------------------------------------------------------------------------- systems
@dataclass
class ReplacementSystem:
    """`inner` with a replacement level for players the training block never saw: the possession-weighted
    mean rating of the block's low-exposure players (under `max_poss` training possessions), per side.
    Uses no held-out data; the criterion's default gives an unseen rookie the average player's 0."""
    name: str
    inner: System
    max_poss: float = 500.0
    shrink: float = 1.0        # a fraction of that level.  The year-over-year test settled on 0.25
                               # (DECISIONS.md: the only depth better in BOTH directions -- deeper
                               # keeps paying on departures and stops paying on rookies).  The
                               # default stays 1.0 because the registered systems were built with it.

    def train_for(self, h, ctx: Context):
        return self.inner.train_for(h, ctx) if hasattr(self.inner, "train_for") else ctx.neighbourhood(h, ctx.current_k)

    def fit(self, train, ctx: Context) -> Ratings:
        r = self.inner.fit(train, ctx)
        low = r.df[(r.df.poss > 0) & (r.df.poss < self.max_poss)]
        if len(low) == 0:
            return r
        w = low.poss.to_numpy(dtype=float)
        return Ratings(r.df, fill_o=self.shrink * float(np.average(low.o, weights=w)),
                       fill_d=self.shrink * float(np.average(low.d, weights=w)))


@dataclass
class TableSystem:
    """Ratings from a table you already have (player_id, season, o, d[, poss]): the mean over the training seasons."""
    name: str
    table: pd.DataFrame

    def fit(self, train, ctx: Context) -> Ratings:
        t = self.table[self.table.season.isin(list(train))]
        if "poss" not in t.columns:
            t = t.assign(poss=1.0)
        g = t.groupby("player_id", as_index=False).agg(o=("o", "mean"), d=("d", "mean"), poss=("poss", "sum"))
        return Ratings(g)


# ---------------------------------------------------------------------------------------- scoring
@dataclass
class Prediction:
    """Everything about one system's prediction of one held-out season that the scores and splits need."""
    y: np.ndarray
    w: np.ndarray
    poss: np.ndarray
    home: np.ndarray
    c_off: np.ndarray          # the offensive five's ratings summed, per row
    c_def: np.ndarray
    pred: np.ndarray           # intercept + home refit on the season, plus c_off + c_def
    base_pred: np.ndarray      # intercept + home only
    game_idx: np.ndarray
    is_home_off: np.ndarray
    rat: pd.DataFrame          # ratings aligned to the season's Z columns
    A: np.ndarray              # the level columns refit on the season (level_columns)


def _wls(A, y, w):
    """Weighted least squares; a zero column (a system with no ratings) gets coefficient 0, not an error."""
    AtW = (A * w[:, None]).T
    return np.linalg.lstsq(AtW @ A, AtW @ y, rcond=None)[0]


def level_columns(wd_h: WindowData, level: str = "home") -> np.ndarray:
    """The nuisance columns refit on the held-out season.

    "home"  an intercept and the home term -- what every earlier script used
    "full"  the design's whole fixed block: intercept, home, playoff, garbage time, margin and the
            margin-by-time term.  Those are context, not attribution, and leaving them out attenuates
            the ratings wherever a strong lineup builds a lead and then scores less on it.
    """
    m = wd_h.spec.n_ps
    if level == "home":
        home = np.asarray(wd_h.X[:, wd_h.spec.f_col("home")].todense()).ravel()
        return np.column_stack([np.ones(wd_h.X.shape[0]), home])
    if level == "full":
        nfix = len(wd_h.spec.f_names)
        F = np.asarray(wd_h.X[:, 2 * m:2 * m + nfix].todense())
        return np.column_stack([np.ones(wd_h.X.shape[0]), F])
    raise ValueError(f"level must be 'home' or 'full', got {level!r}")


def predict_season(rat: Ratings, wd_h: WindowData, level: str = "home") -> Prediction:
    """Predict every row of the held-out season from the ratings; only the level (`level_columns`) is refit."""
    m = wd_h.spec.n_ps
    r = rat.aligned(wd_h.spec.ps_table["player_id"].to_numpy())
    c_off = wd_h.X[:, :m] @ r["o"].to_numpy()
    c_def = wd_h.X[:, m:2 * m] @ r["d"].to_numpy()
    y, w = wd_h.y, wd_h.w
    home = np.asarray(wd_h.X[:, wd_h.spec.f_col("home")].todense()).ravel()
    A = level_columns(wd_h, level)
    contrib = c_off + c_def
    pred = A @ _wls(A, y - contrib, w) + contrib
    base_pred = A @ _wls(A, y, w)
    return Prediction(y=y, w=w, poss=wd_h.rows["poss"].to_numpy(dtype=float), home=home, c_off=c_off, c_def=c_def,
                      pred=pred, base_pred=base_pred, game_idx=wd_h.rows["game_idx"].to_numpy(),
                      is_home_off=wd_h.rows["is_home_off"].to_numpy(), rat=r, A=A)


def team_game_mse(y, pred, poss, game_idx, is_home_off) -> tuple[float, float]:
    """Sum actual and predicted POINTS over each team's rows in a game, then score; returns (mse, possessions)."""
    g = pd.DataFrame({"g": game_idx, "h": is_home_off, "poss": poss,
                      "act": y * poss / 100.0, "prd": pred * poss / 100.0}).groupby(["g", "h"]).sum()
    return float(np.average(((g.act - g.prd) / g.poss * 100.0) ** 2, weights=g.poss)), float(g.poss.sum())


def _keeps_whole_team_games(p: Prediction, idx: np.ndarray) -> bool:
    """True if `idx` takes every row of each team-game it takes any row of."""
    key = np.stack([np.asarray(p.game_idx), np.asarray(p.is_home_off, dtype=np.int64)])
    _, inv, cnt = np.unique(key, axis=1, return_inverse=True, return_counts=True)
    kept = np.bincount(inv[idx], minlength=len(cnt))
    return bool(np.all((kept == 0) | (kept == cnt)))


def score(p: Prediction, mask=None) -> dict:
    """The scores of a prediction, on all rows or on a subset (the level stays the full-season refit).

      mse, base        weighted stint MSE with the ratings and with the level only
      calib, calib_side  MSE after a scalar (per side: two scalars) fitted ON the held-out season --
                       an amplitude diagnostic and an upper bound, never a score
      scale_*          those scalars; 1.0 = calibrated, below 1 = the ratings are too wide
      covered          share of the season's players the training block had a rating for
      tg, tg_base      the same two errors after team-game aggregation -- NaN when `mask` cuts a team-game

    **`tg` is only defined on a mask that keeps whole team-games.**  It sums a team's points over its rows in
    a game and scores that total; on a mask that takes some of a team-game's stints and not others, the sum is
    a partial point total scored against a level fitted on complete ones, and the number is not the criterion
    restricted to those rows -- it is not the criterion at all.  Measured on the shipped board, the
    stint-level `exposure` split's groups recombine to 337.6 where the pooled score is 113.6, a factor of
    three, and the sign of a map comparison flips inside it.  So masks that cut team-games (`exposure`,
    `bench`, `bigs`, `movers`, `rookie` -- anything labelling a STINT) return NaN here and must be read on
    `mse`, at stint level.  `game_bench_share` labels a team-game and recombines to the pooled score exactly.
    """
    idx = np.arange(len(p.y)) if mask is None else np.flatnonzero(mask)
    y, w = p.y[idx], p.w[idx]

    def wmse(pred):
        return float(np.average((y - pred) ** 2, weights=w))

    Ac = np.column_stack([p.A[idx], p.c_off[idx] + p.c_def[idx]])
    cc = _wls(Ac, y, w)
    A2 = np.column_stack([p.A[idx], p.c_off[idx], p.c_def[idx]])
    c2 = _wls(A2, y, w)
    if mask is None or _keeps_whole_team_games(p, idx):
        tg, tg_n = team_game_mse(y, p.pred[idx], p.poss[idx], p.game_idx[idx], p.is_home_off[idx])
        tg_base, _ = team_game_mse(y, p.base_pred[idx], p.poss[idx], p.game_idx[idx], p.is_home_off[idx])
    else:
        tg = tg_base = tg_n = np.nan
    return dict(n=float(w.sum()), mse=wmse(p.pred[idx]), base=wmse(p.base_pred[idx]), calib=wmse(Ac @ cc),
                calib_side=wmse(A2 @ c2), scale=float(cc[-1]), scale_off=float(c2[-2]), scale_def=float(c2[-1]),
                covered=float((p.rat.poss.to_numpy() > 0.0).mean()), tg=tg, tg_base=tg_base, tg_n=tg_n)


# ---------------------------------------------------------------------------------------- splits
SplitFn = Callable[[Prediction, WindowData, Context, int, list], np.ndarray]   # -> group label per row


def by_movers(p: Prediction, wd_h: WindowData, ctx: Context, h: int, train: list) -> np.ndarray:
    """How many of the ten on the floor changed team between the training block and the held-out season.

    A player MOVED if his main team in H differs from his main team in the nearest training season he
    appears in.  Players absent from either are not movers, they are unknown.  Mid-season trades count
    only if the bulk of his minutes shifted.  This is the traded-to-an-average-team test: a rating
    that is really his old teammates' credit stops predicting once he plays with new ones.
    """
    m = wd_h.spec.n_ps
    ids = wd_h.spec.ps_table["player_id"].to_numpy()
    now = ctx.main_team(h)
    before: dict = {}
    for s in sorted(train, key=lambda s: abs(s - h)):
        for pid, t in ctx.main_team(s).items():
            before.setdefault(pid, t)
    moved = np.array([1.0 if (q in now and q in before and now[q] != before[q]) else 0.0 for q in ids])
    n_movers = wd_h.X[:, :2 * m] @ np.tile(moved, 2)
    return np.where(n_movers == 0, "no movers", np.where(n_movers <= 2, "1-2 movers", "3+ movers"))


def by_exposure(p: Prediction, wd_h: WindowData, ctx: Context, h: int, train: list) -> np.ndarray:
    """The SMALLEST training exposure among the ten players on the row, binned (cfg holdout.exposure_edges)."""
    edges = [float(e) for e in ctx.cfg.get("holdout", {}).get("exposure_edges", [0, 1, 500, 1500, 4000, 1e9])]
    labels = [f"{int(a)}-{int(b) - 1}" if b < 1e8 else f"{int(a)}+" for a, b in zip(edges[:-1], edges[1:])]
    labels[0] = "none (0)" if edges[1] == 1 else labels[0]
    m = wd_h.spec.n_ps
    big = 1e9
    poss = p.rat.poss.to_numpy(dtype=float)
    # Z has exactly five ones per block, so a min over the on-court players is a max over (big - poss)
    minposs = big - (wd_h.X[:, :2 * m].multiply(np.tile(big - poss, 2))).max(axis=1).toarray().ravel()
    b = np.clip(np.digitize(minposs, edges) - 1, 0, len(labels) - 1)
    return np.asarray(labels, dtype=object)[b]


def _count_split(flag: np.ndarray, wd_h: WindowData, edges, what: str) -> np.ndarray:
    """Rows binned by how many of the ten on the floor carry `flag`, with edges like [0, 3, 4, 5, 11]."""
    m = wd_h.spec.n_ps
    edges = [float(e) for e in edges]
    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        a, b = int(a), int(b)
        labels.append(f"{a} {what}" if b == a + 1 else (f"{a}+ {what}" if b > 10 else f"{a}-{b - 1} {what}"))
    n = wd_h.X[:, :2 * m] @ np.tile(np.asarray(flag, dtype=float), 2)
    b = np.clip(np.digitize(n, edges) - 1, 0, len(labels) - 1)
    return np.asarray(labels, dtype=object)[b]


def by_bigs(p: Prediction, wd_h: WindowData, ctx: Context, h: int, train: list) -> np.ndarray:
    """How many of the ten on the floor are big men (top tercile of the big-man score in season H).
    The loss-wise archetype check: is a system worse where the bigs are, not just tilted toward them."""
    big = ctx.bigness(h)
    ids = wd_h.spec.ps_table["player_id"].to_numpy()
    flag = np.array([1.0 if big.get(int(q), False) else 0.0 for q in ids])
    return _count_split(flag, wd_h, ctx.cfg.get("holdout", {}).get("bigs_edges", [0, 3, 4, 5, 11]), "bigs")


def by_bench(p: Prediction, wd_h: WindowData, ctx: Context, h: int, train: list) -> np.ndarray:
    """How many of the ten on the floor are bench players in season H (games-started share below
    cfg holdout.bench_gs_pct): where the role prior is meant to help, and where garbage time lives."""
    bench = ctx.bench(h)
    ids = wd_h.spec.ps_table["player_id"].to_numpy()
    flag = np.array([1.0 if bench.get(int(q), False) else 0.0 for q in ids])
    return _count_split(flag, wd_h, ctx.cfg.get("holdout", {}).get("bench_edges", [0, 3, 5, 7, 11]), "bench")


def by_rookie(p: Prediction, wd_h: WindowData, ctx: Context, h: int, train: list) -> np.ndarray:
    """How many of the ten on the floor had never played a game before season H (roles.parquet knows every
    season from 1997, so a 1997 or 1998 row calls veterans rookies -- read the split from 1999 on).  The
    in-season question the block board answers worst: a player the fit has no past for at all."""
    ri = ctx.role_inputs
    seen = set() if ri is None else set(ri[(ri.season < int(h)) & (ri.games > 0)].player_id.astype(int))
    ids = wd_h.spec.ps_table["player_id"].to_numpy()
    flag = np.array([0.0 if int(q) in seen else 1.0 for q in ids])
    return _count_split(flag, wd_h, [0, 1, 2, 11], "rookies")


def by_gt(p: Prediction, wd_h: WindowData, ctx: Context, h: int, train: list) -> np.ndarray:
    """Garbage-time rows (the stint parser's rule: a 4th-quarter margin of 15 late or 20 at any point) against the rest."""
    gt = wd_h.rows["is_gt"].to_numpy().astype(bool)
    return np.where(gt, "garbage time", "competitive")


def by_game_bench_share(p: Prediction, wd_h: WindowData, ctx: Context, h: int, train: list) -> np.ndarray:
    """Each TEAM-GAME binned by the share of its possessions played by players the training block barely saw.

    `by_exposure` labels a STINT, so its groups cut a team-game in half and the criterion's own unit -- points
    summed over a team's rows in a game -- is never complete inside one of them.  Their numbers are still the
    right question ("how well do we predict the stints a bench player was on the floor for"), but they are NOT
    a decomposition of the pooled score and cannot be read as one telling you where the pooled gain came from.
    This one is constant within a team-game, so the groups partition the team-games exactly and their scores
    do add up to the pooled one.

    The share counts each of the ten on the floor as a tenth of the row's possessions, low = fewer than
    `holdout.low_exposure` training possessions (500, `low_poss_threshold` at season scale).
    """
    m = wd_h.spec.n_ps
    thr = float(ctx.cfg.get("holdout", {}).get("low_exposure", 500.0))
    edges = [float(e) for e in ctx.cfg.get("holdout", {}).get("game_bench_share_edges", [0.0, 0.05, 0.15, 0.30, 1.01])]
    low = (p.rat.poss.to_numpy(dtype=float) < thr).astype(float)
    n_low = wd_h.X[:, :2 * m] @ np.tile(low, 2)
    poss = p.poss
    d = pd.DataFrame({"g": p.game_idx, "h": p.is_home_off, "poss": poss, "lp": poss * n_low / 10.0})
    tg = d.groupby(["g", "h"]).sum()
    share = (tg.lp / tg.poss.clip(lower=1e-9)).rename("share")
    labels = [f"{a:.0%}-{b:.0%}" if b <= 1.0 else f"{a:.0%}+" for a, b in zip(edges[:-1], edges[1:])]
    b = np.clip(np.digitize(share.to_numpy(), edges) - 1, 0, len(labels) - 1)
    lab = pd.Series(np.asarray(labels, dtype=object)[b], index=share.index)
    return lab.reindex(pd.MultiIndex.from_arrays([d.g, d.h])).to_numpy()


SPLITS = {"movers": by_movers, "exposure": by_exposure, "bigs": by_bigs, "bench": by_bench, "gt": by_gt,
          "rookie": by_rookie, "game_bench_share": by_game_bench_share}


# ---------------------------------------------------------------------------------------- the runner
@dataclass
class Holdout:
    cfg: dict
    first: int
    last: int
    ks: list
    lams: list
    rank_bins: int = 10
    rank_min_poss: float = 1000.0
    level: str = "home"
    rank_: pd.DataFrame | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, cfg, **override) -> "Holdout":
        h = dict(cfg.get("holdout", {}))
        h.update({k: v for k, v in override.items() if v is not None})
        lam = h.get("lam")
        lams = [float(cfg["lam_plugin"])] if lam is None else [float(x) for x in (lam if isinstance(lam, list) else [lam])]
        return cls(cfg=cfg, first=int(h.get("first", cfg["first_season"] + 1)), last=int(h.get("last", cfg["last_season"] - 1)),
                   ks=[int(k) for k in h.get("ks", [2, 4])], lams=lams, rank_bins=int(h.get("rank_bins", 10)),
                   rank_min_poss=float(h.get("rank_min_poss", 1000)), level=str(h.get("level", "home")))

    def seasons(self) -> list[int]:
        s0, s1 = int(self.cfg["first_season"]), int(self.cfg["last_season"])
        return [h for h in range(self.first, self.last + 1) if s0 <= h <= s1]

    def run(self, systems: list, ctx: Context, splits: dict | None = None, rank: bool = False,
            out: Path | None = None, verbose: bool = True, held: list | None = None) -> pd.DataFrame:
        """One row per held-out season x K x lambda x system x split x group, columns RESULT_COLUMNS.
        `held` restricts the held-out seasons; default = all of `seasons()`."""
        splits = splits or {}
        rows, rank_rows = [], []
        t0 = time.time()
        held = self.seasons() if held is None else [int(h) for h in held]
        if verbose:
            print(f"held-out {held[0]}-{held[-1]} ({len(held)}), K {self.ks}, lambdas {self.lams}, "
                  f"systems {[s.name for s in systems]}", flush=True)
        for h in held:
            ctx.current_h = h
            # ALWAYS scored against actual points, and over the whole season -- playoffs included, because
            # a season is one entity.  This MOVED the criterion's estimand on 2026-09-10; every number
            # measured before that date was on regular-season rows alone.
            wd_h = ctx.design([h], "pts")
            for k in self.ks:
                ctx.current_k = k
                for system in systems:
                    # a system may say which seasons it trains on (63's neighbouring-season tables do)
                    train = system.train_for(h, ctx) if hasattr(system, "train_for") else ctx.neighbourhood(h, k)
                    if train is None:
                        continue
                    for lam in self.lams:
                        t1 = time.time()
                        rat = system.fit(train, ctx)
                        p = predict_season(rat, wd_h, level=self.level)
                        base = dict(held_out=h, k=k, train=",".join(map(str, train)), system=system.name, lam=lam,
                                    cut=np.nan)
                        rows.append(dict(**base, split="all", group="all", **score(p), seconds=time.time() - t1))
                        for sname, fn in splits.items():
                            grp = fn(p, wd_h, ctx, h, train)
                            for g in pd.unique(grp):
                                rows.append(dict(**base, split=sname, group=str(g), **score(p, grp == g), seconds=0.0))
                        if rank:
                            rc = rank_calibration(p, wd_h, n_bins=self.rank_bins, min_poss=self.rank_min_poss)
                            rank_rows.append(rc.assign(**base))
            if verbose:
                print(f"  {h} done ({time.time() - t0:.0f}s)", flush=True)
            if out is not None:
                pd.DataFrame(rows)[RESULT_COLUMNS].to_parquet(out, index=False)       # checkpoint
        res = pd.DataFrame(rows)[RESULT_COLUMNS]
        if rank_rows:
            self.rank_ = pd.concat(rank_rows, ignore_index=True)
        if out is not None:
            res.to_parquet(out, index=False)
        ctx.current_h = None
        return res


# ---------------------------------------------------------------------------------------- summaries
# Every error in this module is stored as a MEAN SQUARE, because that is what pools and what pairs.  It is
# REPORTED as ARMSE: the root of it, scaled by sqrt(2 / pi), which is the mean absolute deviation of a
# normal with that standard deviation.  A root mean square reads bigger than the typical miss because
# squaring pays extra attention to the tail; ARMSE is the number to say out loud -- "a typical team-game
# misses by this much, in points per 100".
from .scoring import MAE_SCALE, armse  # noqa: E402,F401

POOL_BY = ("k", "lam", "system", "split", "group")


def pooled(res: pd.DataFrame, by=POOL_BY) -> pd.DataFrame:
    """Possession-weighted means over the held-out seasons.  `vs_no_ratings` = share of error the ratings remove."""
    def agg(d):
        n, tn = d.n.sum(), d.tg_n.sum()
        return pd.Series({
            "seasons": int(d.held_out.nunique()),
            "mse": float((d.mse * d.n).sum() / n),
            "base": float((d.base * d.n).sum() / n),
            "vs_no_ratings": 1.0 - float((d.mse * d.n).sum() / (d.base * d.n).sum()),
            "calib_side": float((d.calib_side * d.n).sum() / n),
            "scale_off": float((d.scale_off * d.n).sum() / n),
            "scale_def": float((d.scale_def * d.n).sum() / n),
            "covered": float(d.covered.mean()),
            # NaN, not a number, when the group's mask cut team-games: `score` refuses to aggregate a
            # partial team-game and this is the one place that would quietly turn its NaN into a divide
            "game": float((d.tg * d.tg_n).sum() / tn) if tn > 0 else np.nan,
            "game_base": float((d.tg_base * d.tg_n).sum() / tn) if tn > 0 else np.nan,
            "game_vs_no_ratings": (1.0 - float((d.tg * d.tg_n).sum() / (d.tg_base * d.tg_n).sum())
                                   if tn > 0 else np.nan),
            "share": float(n),          # this group's share of the pooled rows, not the role input
            "armse": armse((d.mse * d.n).sum() / n),
            "game_armse": armse((d.tg * d.tg_n).sum() / tn) if tn > 0 else np.nan,
        })
    out = res.groupby(list(by), sort=True).apply(agg, include_groups=False).reset_index()
    tot = out.groupby([b for b in by if b != "group"] if "group" in by else list(by))["share"].transform("sum")
    out["share"] = out["share"] / tot
    return out


def paired(res: pd.DataFrame, ref: str, value: str = "mse", by=("k", "lam", "split", "group")) -> pd.DataFrame:
    """Every system against `ref`, paired by held-out season: mean difference, its SE, z, seasons won.

    `value` is "mse" (stint) or "tg" (team-game).  Negative = better than the reference.
    """
    rows = []
    for key, d in res.groupby(list(by), sort=True):
        p = d.pivot_table(index="held_out", columns="system", values=value)
        if ref not in p.columns:
            continue
        for s in p.columns:
            if s == ref:
                continue
            diff = (p[s] - p[ref]).dropna()
            n = len(diff)
            sd = diff.std(ddof=1) if n > 1 else np.nan
            rows.append(dict(**dict(zip(by, key)), system=s, ref=ref, value=value, mean_diff=float(diff.mean()),
                             se=float(sd / np.sqrt(n)) if n > 1 else np.nan,
                             z=float(diff.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
                             wins=int((diff < 0).sum()), n_seasons=n))
    return pd.DataFrame(rows)


def report(res: pd.DataFrame, ref: str | None = None, digits: int = 4) -> str:
    """The printed summary: pooled scores per K, then the paired tests against `ref` at stint and game level."""
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", digits)
    lines = ["=== out-of-season reliability, pooled over held-out seasons (LOWER IS BETTER)",
             "    game_armse: what a typical TEAM-GAME misses by, points per 100 -- sqrt of the mean square,",
             "    scaled by sqrt(2/pi) onto the mean-absolute scale.  armse is the same at stint level.",
             "    vs_no_ratings: share of held-out error the ratings remove.  scale_*: what the held-out season",
             "    wants the ratings multiplied by (1 = calibrated; below 1 = too wide) -- a diagnostic, not a score."]
    P = pooled(res)
    cols = ["system", "seasons", "armse", "vs_no_ratings", "scale_off", "scale_def", "covered",
            "game_armse", "game_vs_no_ratings"]
    for (k, lam, split, group), d in P.groupby(["k", "lam", "split", "group"], sort=True):
        head = f"\n-- K = {k}, lambda {lam:.0f}" + ("" if split == "all" else f", {split} = {group} (share {d.share.iloc[0]:.2f})")
        lines.append(head)
        lines.append(d[cols].to_string(index=False))
    if ref is not None and ref in set(res.system):
        lines.append(f"\n=== paired by held-out season against {ref}; negative = better")
        for value, label in (("mse", "stint"), ("tg", "team-game")):
            t = paired(res, ref, value)
            if len(t):
                lines.append(f"\n-- {label} level")
                lines.append(t.drop(columns=["ref", "value"]).to_string(index=False))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------- rank calibration
def rank_calibration(p: Prediction, wd_h: WindowData, n_bins: int = 10, min_poss: float = 1000.0) -> pd.DataFrame:
    """What the held-out season wants each decile of the ratings multiplied by, out of sample.

    Player rank (side "o" / "d"): players with `min_poss` training possessions are ranked into deciles
    by their rating; everyone else is the "unranked" group.  Regress the held-out stints on the ten
    per-decile lineup contributions (plus the other side's whole contribution, the intercept and home):
    slope_j is the multiplier the season wants on decile j, 1.0 = calibrated.  A slope that rises with
    rank says the top is under-rated; one that falls says the top is over-rated.

    Lineup rank (side "lineup"): rows binned by their predicted margin; within each, the WLS slope of
    actual on predicted and the mean residual -- the calibration of "players with these coefficients,
    when they get together".
    """
    m = wd_h.spec.n_ps
    y, w = p.y, p.w
    one = np.ones(len(y))
    rows = []
    for side, col, Z, other in (("o", "o", wd_h.X[:, :m], p.c_def), ("d", "d", wd_h.X[:, m:2 * m], p.c_off)):
        v = p.rat[col].to_numpy(dtype=float)
        poss = p.rat.poss.to_numpy(dtype=float)
        ranked = (poss >= min_poss) & (v != 0.0)
        dec = np.full(len(v), -1)
        if ranked.sum() >= n_bins * 2:
            dec[ranked] = pd.qcut(v[ranked], n_bins, labels=False, duplicates="drop")
        groups = sorted(set(dec.tolist()))
        # The slope as a smooth function of the rating, s(v) = c1 + c2 u + c3 u^2 with u the
        # standardised rating, reported at each decile's mean.  NOT one slope per decile: every row
        # has exactly five players a side, so the decile headcounts sum to a constant and a per-decile
        # fit is collinear with the intercept (it came out 0.3 to 0.9 on ratings that were exactly
        # right).  The curve is identified, and it answers the question -- does the top of the board
        # want a bigger or a smaller multiplier than the bottom -- directly.
        u = np.zeros(len(v))
        if ranked.sum() >= 3:
            u[ranked] = (v[ranked] - v[ranked].mean()) / max(v[ranked].std(), 1e-9)
        A = np.column_stack([p.A, other, Z @ v, Z @ (v * u), Z @ (v * u ** 2)])
        AtW = (A * w[:, None]).T
        G = AtW @ A
        coef = np.linalg.lstsq(G, AtW @ y, rcond=None)[0]
        e = y - A @ coef
        sigma2 = float(np.average(e ** 2, weights=w))
        cov = sigma2 * np.linalg.pinv(G) * (w.sum() / len(w))    # weights are possessions: rescale to a per-row variance
        c, V = coef[-3:], cov[-3:, -3:]
        for g in groups:
            sel = dec == g
            ug = float(u[sel].mean()) if sel.any() else 0.0
            b = np.array([1.0, ug, ug ** 2])
            rows.append(dict(side=side, decile=int(g), n_players=int(sel.sum()),
                             mean_rating=float(v[sel].mean()) if sel.any() else np.nan,
                             slope=float(b @ c), se=float(np.sqrt(max(b @ V @ b, 0.0)))))
    # lineup rank
    margin = p.pred - p.base_pred
    q = (np.asarray(pd.qcut(margin, n_bins, labels=False, duplicates="drop"), dtype=float)
         if np.ptp(margin) > 0 else np.full(len(margin), np.nan))      # a system with no ratings has no lineup rank
    for g in sorted(pd.unique(q[np.isfinite(q)])):
        sel = q == g
        yy, ww, pp = y[sel], w[sel], p.pred[sel]
        A = np.column_stack([np.ones(sel.sum()), pp])
        c = _wls(A, yy, ww)
        rows.append(dict(side="lineup", decile=int(g), n_players=int(sel.sum()),
                         mean_rating=float(np.average(pp - p.base_pred[sel], weights=ww)),
                         slope=float(c[1]), se=np.nan,
                         mean_resid=float(np.average(yy - pp, weights=ww))))
    return pd.DataFrame(rows)


def pooled_rank(rank: pd.DataFrame, exclude_h=None) -> pd.DataFrame:
    """Inverse-variance pooled decile slopes over held-out seasons; z = (slope - 1) / se."""
    r = rank if exclude_h is None else rank[rank.held_out != exclude_h]
    out = []
    for (k, lam, system, side, dec), d in r.groupby(["k", "lam", "system", "side", "decile"]):
        if side == "lineup" or d.se.isna().all():
            out.append(dict(k=k, lam=lam, system=system, side=side, decile=dec, slope=float(d.slope.mean()),
                            se=float(d.slope.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan,
                            mean_rating=float(d.mean_rating.mean()), seasons=len(d)))
        else:
            iv = 1.0 / np.maximum(d.se.to_numpy() ** 2, 1e-12)
            slope = float((d.slope.to_numpy() * iv).sum() / iv.sum())
            out.append(dict(k=k, lam=lam, system=system, side=side, decile=dec, slope=slope,
                            se=float(np.sqrt(1.0 / iv.sum())), mean_rating=float(d.mean_rating.mean()), seasons=len(d)))
    o = pd.DataFrame(out)
    o["z"] = (o.slope - 1.0) / o.se
    return o


def fit_rank_map(rank: pd.DataFrame, system: str, side: str, k: int, exclude_h=None) -> Callable:
    """A monotone piecewise-linear map rating -> slope(rating) * rating from the pooled decile slopes,
    leaving `exclude_h` out so the correction is scored on a season that did not choose it."""
    pr = pooled_rank(rank, exclude_h)
    pr = pr[(pr.system == system) & (pr.side == side) & (pr.k == k) & (pr.decile >= 0)].sort_values("mean_rating")
    if len(pr) < 2:
        return lambda x: x
    xs = pr.mean_rating.to_numpy()
    ys = np.maximum.accumulate(xs * pr.slope.to_numpy())          # enforce monotone

    def f(x):
        x = np.asarray(x, dtype=float)
        lo, hi = pr.slope.iloc[0], pr.slope.iloc[-1]
        out = np.interp(x, xs, ys)
        out = np.where(x < xs[0], ys[0] + lo * (x - xs[0]), out)
        return np.where(x > xs[-1], ys[-1] + hi * (x - xs[-1]), out)
    return f


# ---------------------------------------------------------------------------------------- validation and diagnostics
def consensus_table_missing(out) -> str:
    """Why `outputs/vs_consensus.parquet` is not there, and what to use instead.

    The script that built it, `scripts/22_vs_consensus.py`, was deleted in 73f0ab9 and nothing
    replaced it, so this path has been dead on every machine since.  The live external check is
    `tests/test_vs_consensus.py`, which joins `data/external/consensus.csv` (tracked) to a board
    by name; point `OPENRAPM_BOARD` at a candidate to score it without touching the shipped one.
    """
    return (f"{out} is not built, and nothing in the tree builds it any more: "
            "scripts/22_vs_consensus.py was deleted in 73f0ab9.  The external check that survived it "
            "is the test suite -- `OPENRAPM_BOARD=<your board>.parquet pytest "
            "tests/test_vs_consensus.py -q` -- which reads data/external/consensus.csv directly.  "
            "This mode needs a per-player table with a `bigness` column and no longer has one.")


def vs_consensus(rat: Ratings, cfg, min_poss: float | None = None) -> dict:
    """The external check, read once: rank agreement with the consensus of modern metrics on 2024-26.

    Nothing is ever selected on this.  Ratings arrive in raw sign; defense is flipped here so that
    positive = good, as on the board.
    """
    out = Path(cfg["_root"]) / "outputs" / "vs_consensus.parquet"
    if not out.exists():
        raise RuntimeError(consensus_table_missing(out))
    from scipy.stats import spearmanr
    min_poss = float(cfg.get("holdout", {}).get("consensus", {}).get("min_poss", 1000) if min_poss is None else min_poss)
    con = pd.read_parquet(out)[["player_id", "adj_offense", "adj_defense", "adj_overall", "bigness"]]
    g = rat.df.merge(con, on="player_id")
    g = g[g.poss >= min_poss].copy()
    g["dd"] = -g.d
    g["t"] = g.o + g.dd

    def z(s):
        return (s - s.mean()) / s.std()

    return dict(n=int(len(g)), con_total=float(spearmanr(g.t, g.adj_overall).statistic),
                con_off=float(spearmanr(g.o, g.adj_offense).statistic),
                con_def=float(spearmanr(g.dd, g.adj_defense).statistic),
                spread_off=float(g.o.std() / g.adj_offense.std()), spread_def=float(g.dd.std() / g.adj_defense.std()),
                bias_total=float((z(g.t) - z(g.adj_overall)).corr(g.bigness)),
                bias_off=float((z(g.o) - z(g.adj_offense)).corr(g.bigness)),
                bias_def=float((z(g.dd) - z(g.adj_defense)).corr(g.bigness)))


def _team_of_rows(p: Prediction, wd_h: WindowData, main_team: dict) -> tuple[np.ndarray, np.ndarray]:
    """The offensive and defensive team of every row: the modal main team of the five on each side."""
    m = wd_h.spec.n_ps
    ids = wd_h.spec.ps_table["player_id"].to_numpy()
    team = np.array([main_team.get(int(q), -1) for q in ids])
    Zo, Zd = wd_h.X[:, :m].tocsr(), wd_h.X[:, m:2 * m].tocsr()

    def modal(Z):
        t = team[Z.indices].reshape(Z.shape[0], 5)
        out = np.empty(Z.shape[0], dtype=int)
        for i in range(Z.shape[0]):
            vals, cnt = np.unique(t[i], return_counts=True)
            out[i] = vals[np.argmax(cnt)]
        return out
    return modal(Zo), modal(Zd)


def team_residual(p: Prediction, wd_h: WindowData, main_team: dict) -> pd.DataFrame:
    """How much each team out- or under-performed its players' ratings, on offense and on defense.

    The residual mean per team with its SE (delta method on the possession weights).  A rating that
    captures "how much he helps his current team" leaves nothing here; a systematic residual is the
    part of team quality the additive model does not carry -- fit, scheme, or chemistry.
    """
    to, td = _team_of_rows(p, wd_h, main_team)
    e = p.y - p.pred
    rows = []
    for t in sorted(set(to.tolist()) - {-1}):
        for side, sel in (("offense", to == t), ("defense", td == t)):
            w = p.w[sel]
            mean = float(np.average(e[sel], weights=w))
            se = float(np.sqrt((w ** 2 * (e[sel] - mean) ** 2).sum()) / w.sum())
            rows.append(dict(team_id=int(t), side=side, poss=float(p.poss[sel].sum()), resid=mean, se=se,
                             z=mean / se if se > 0 else np.nan))
    return pd.DataFrame(rows)


def replacement_quality(p: Prediction, wd_h: WindowData, main_team: dict, min_poss: float = 1000.0) -> pd.DataFrame:
    """Is a player's rating tied to how bad the players who replace him are?

    For each player on a team: the mean rating-sum of his team's OFFENSIVE lineups when he sits, minus the
    mean rating-sum of the OTHER FOUR when he plays (same for defense).  Positive = good replacements.
    A rating that says "he is good because his replacement is bad" correlates negatively with this;
    the additive model's coefficient should not, and `corr` at the bottom of the frame is the test.
    """
    m = wd_h.spec.n_ps
    ids = wd_h.spec.ps_table["player_id"].to_numpy()
    to, td = _team_of_rows(p, wd_h, main_team)
    o, d, poss = p.rat.o.to_numpy(), p.rat.d.to_numpy(), p.rat.poss.to_numpy()
    Zo, Zd = wd_h.X[:, :m].tocsr(), wd_h.X[:, m:2 * m].tocsr()
    rows = []
    for j in np.flatnonzero(poss >= min_poss):
        t = main_team.get(int(ids[j]), -1)
        if t < 0:
            continue
        for side, Z, team_rows, v in (("o", Zo, to == t, o), ("d", Zd, td == t, d)):
            on = np.asarray(Z[:, j].todense()).ravel() > 0
            sel_on, sel_off = team_rows & on, team_rows & ~on
            if sel_on.sum() < 20 or sel_off.sum() < 20:
                continue
            lineup = Z @ v
            with_him = np.average(lineup[sel_on] - v[j], weights=p.w[sel_on])
            without = np.average(lineup[sel_off], weights=p.w[sel_off])
            rows.append(dict(player_id=int(ids[j]), team_id=int(t), side=side, rating=float(v[j]),
                             replacement=float(without - with_him), poss_on=float(p.poss[sel_on].sum())))
    out = pd.DataFrame(rows)
    if len(out):
        out.attrs["corr"] = {s: float(g.rating.corr(g.replacement)) for s, g in out.groupby("side")}
    return out
