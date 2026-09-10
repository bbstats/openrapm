"""The multi-stage board fitted in one pass per block.

`systems.registry`'s `mspi` is a SplitSystem of two PluginSystems (offense on the free-throw target, defense on
the opponent-3PM-replaced one), and each of the two builds its own design, fits its own BoxExposure, rebuilds
the role prior and solves the mixed model for both sides.  Everything but the target and the solve is the same
work twice.  `MspiFast` does it once: one design (the counters give the free-throw target as a linear
combination, `design.TARGETS`; the defensive target is derived from the same design), one exposure fit, one
offset, and one ridge solve per side on the shared transformed design.  The ratings are the same numbers.

The knobs are the ones the criterion may still want moved: the ridge (`lam`, `lam_ratio`), the GBDT prior's
training parameters (`gbdt_params`, chimeraboost overrides, cached per parameter set on the Context), the
per-side targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cv import make_exposure
from .design import FIT_PHASES, TARGETS
from .estimator import MixedModelRAPM, Moments, _Layout
from .holdout import Context, Ratings


# a per-process section clock, on only with FASTFIT_TIMER set: where a run's fit seconds go
SECTIONS: dict = {}


def _timer():
    import os
    import time
    if not os.environ.get("FASTFIT_TIMER"):
        return lambda name: None
    if not SECTIONS:
        import atexit
        atexit.register(lambda: print("  fit sections (pid, seconds): " + "  ".join(
            f"{k} {v:.2f}" for k, v in SECTIONS.items()) + f"  total {sum(SECTIONS.values()):.2f}", flush=True))
    last = [time.time()]

    def mark(name):
        now = time.time()
        SECTIONS[name] = SECTIONS.get(name, 0.0) + now - last[0]
        last[0] = now
    return mark


def derived_target(wd, name: str) -> np.ndarray:
    """A `design.TARGETS` response recomputed from the design's own counters (the rows are the same, the
    weight is the same: only y changes)."""
    tg = TARGETS[name]
    c = wd.counters
    num = sum(k * c[col].to_numpy(dtype=float) for col, k in tg["num"].items())
    den = c[tg["den"]].to_numpy(dtype=float)
    return tg["scale"] * num / den


# ---------------------------------------------------------------- the four-factor defence (HANDOFF 3.2)
FACTORS = ("efg", "tov", "oreb", "ftr")
# per factor (lam, lam_D / lam_O): selected once by REML-in-band on the factor's own design and its own
# scale (FINDINGS 15, 2024-2026, zero prior, `factors.select_lambda`), never on the criterion.  The asymmetry
# is the point: forcing turnovers is a defensive skill (0.75), holding the opponents' eFG% down much less of
# one (1.5), keeping them off the offensive glass barely one at all (3.0).
FACTOR_LAMS = {"efg": (3495.0, 1.5), "tov": (2176.0, 0.75), "oreb": (414.0, 3.0), "ftr": (1355.0, 1.0)}
FACTOR_COUNTERS = {"fgm", "fg3m", "fga", "tov", "poss", "reb_cont", "reb_chance", "fta", "att"}


def factor_rows(wd, name: str, x3=None) -> tuple[np.ndarray, np.ndarray]:
    """A factor's response and weight on the POINTS design's rows: the rate per 100 of its own denominator,
    weighted by that denominator (design.TARGETS: an eFG% row carries information in proportion to its
    attempts, not its possessions).  A row with no denominator gets weight 0 and response 0, so it drops out
    of every cross-product without leaving the design.  `x3` (expected opponent three-point makes per row,
    xshoot.expected_threes) reprices the eFG% numerator the way x3def reprices the points target: the makes
    from three replaced by the shooters' expectation, so the defenders are not fit to whether threes dropped."""
    tg = TARGETS[name]
    c = wd.counters
    num = sum(k * c[col].to_numpy(dtype=float) for col, k in tg["num"].items())
    if name == "efg" and x3 is not None:
        num = num - 1.5 * c["fg3m"].to_numpy(dtype=float) + 1.5 * np.asarray(x3, dtype=float)
    den = c[tg["den"]].to_numpy(dtype=float)
    ok = den > 0
    y = np.zeros(len(den))
    y[ok] = tg["scale"] * num[ok] / den[ok]
    return y, np.where(ok, den, 0.0)


def points_per_factor(y_pts, ys: dict, ws: dict, w_poss, season) -> tuple[dict, float]:
    """d(points per 100) / d(factor), one number per factor: a possession-weighted least squares of the row's
    points on its four rates with a level per season, over the rows where every rate is defined.  It is the
    linearisation of the identity pts = 2 eFG FGA + FT% FTA around the block's own play (the rates are the
    realised ones, so nothing is attenuated); what it leaves is the curvature, and the R^2 says how much."""
    ok = np.ones(len(y_pts), bool)
    for f in FACTORS:
        ok &= ws[f] > 0
    s = np.asarray(season)[ok]
    levels = np.unique(s)
    D = (s[:, None] == levels[None, :]).astype(float)
    X = np.column_stack([D] + [np.asarray(ys[f])[ok] for f in FACTORS])
    sw = np.sqrt(np.asarray(w_poss, dtype=float)[ok])
    yk = np.asarray(y_pts, dtype=float)[ok]
    coef = np.linalg.lstsq(X * sw[:, None], yk * sw, rcond=None)[0]
    res = yk - X @ coef
    mu = np.average(yk, weights=sw ** 2)
    r2 = 1.0 - float(np.sum((sw * res) ** 2) / np.sum((sw * (yk - mu)) ** 2))
    return {f: float(coef[len(levels) + i]) for i, f in enumerate(FACTORS)}, r2


def _wslope(x, y, w):
    """The weighted least-squares slope of y on x through the weighted means."""
    w = np.asarray(w, dtype=float)
    if w.sum() <= 0:
        return 0.0
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / vx) if vx > 0 else 0.0


REML_GRID = (1.0 / 16.0, 64.0, 21)      # the REML search for a factor's ridge: FACTOR_LAMS times this log grid
RATIO_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)   # ... and, for reml="2d", over lam_D / lam_O as well


def factor_defense(wd, layout: _Layout, off, y_o, y_d, y_pts, w, lam, ratio, lam_buckets, factor_lams=None,
                   lam_scale: float = 1.0, min_poss: float = 1000.0, reml=False, x3=None) -> dict:
    """The defensive effects from four factor fits instead of one points fit (HANDOFF 3.2).

    Each factor is solved on the SAME layout as the points fit -- same players, same fixed block, same
    exposure -- with its own response and denominator weight (`factor_rows`) and its own ridge and
    offense/defense ratio (`FACTOR_LAMS`), so the shrinkage differs by what kind of defending it is.  The
    points prior `off` (the GBDT, in points per 100) is shared out across the factors the way the block's own
    zero-prior effects split it -- the slope of each factor's zero-prior effect on the zero-prior points
    effect, per side, normalised so the four shares recombine to exactly one prior -- and each factor fit
    shrinks toward its share.  The four defensive effects are then recombined into points allowed with the
    gradients of `points_per_factor`, so the rating is prior_d + sum_f g_f u_f: the same object as the points
    fit's prior_d + u_d, with the residual shrunk factor by factor.

    Returns the raw-sign defensive rating (`d`), its per-player residual (`u_d`) and the diagnostics."""
    from dataclasses import replace as _rep
    m = wd.spec.n_ps
    Z = layout.Z
    poss_row = wd.rows["poss"].to_numpy(dtype=float)
    sw = w / np.where(poss_row > 0, poss_row, 1.0)          # the season weights the fit put on top of poss
    season = wd.rows["season"].to_numpy()
    lams = {**FACTOR_LAMS, **(factor_lams or {})}
    ys, ws = {}, {}
    for f in FACTORS:
        ys[f], den = factor_rows(wd, f, x3=x3)
        ws[f] = den * sw
    g, r2 = points_per_factor(y_pts, ys, ws, w, season)

    # the zero-prior solves, points and factors, for the split of the prior
    lay0 = _rep(layout, offset=None)
    mm_p = MixedModelRAPM(lam=lam, lam_ratio=ratio, spec=wd.spec, lam_buckets=lam_buckets)
    mom0 = Moments(lay0, np.asarray(y_o, dtype=float), w, mm_p._season_cols(), mm_p._scale())
    mom0.want_edf = False
    u_o0 = np.asarray(mom0.solve_chol(lam).u, dtype=float)
    u_d0 = np.asarray(mom0.with_y(lay0, np.asarray(y_d, dtype=float), w).solve_chol(lam).u, dtype=float)
    poss_o = np.asarray(Z[:, :m].T @ w).ravel()          # each player's weighted possessions on offense
    poss_d = np.asarray(Z[:, m:].T @ w).ravel()
    keep_o, keep_d = poss_o >= min_poss, poss_d >= min_poss

    moms, u0, h_o, h_d = {}, {}, {}, {}
    for f in FACTORS:
        lam_f, ratio_f = lams[f]
        lam_f = float(lam_f) * (1.0 if reml else float(lam_scale))
        mm_f = MixedModelRAPM(lam=lam_f, lam_ratio=float(ratio_f), spec=wd.spec, lam_buckets=lam_buckets)
        mom = Moments(lay0, ys[f], ws[f], mm_f._season_cols(), mm_f._scale())
        mom.want_edf = False
        moms[f] = (mom, lam_f)
        u0[f] = np.asarray(mom.solve_chol(lam_f).u, dtype=float)
        h_o[f] = _wslope(u_o0[:m], u0[f][:m], poss_o * keep_o)
        h_d[f] = _wslope(u_d0[m:], u0[f][m:], poss_d * keep_d)
    # normalise the shares so the factor priors recombine to exactly the points prior
    s_o = sum(g[f] * h_o[f] for f in FACTORS)
    s_d = sum(g[f] * h_d[f] for f in FACTORS)
    for f in FACTORS:
        h_o[f] = h_o[f] / s_o if abs(s_o) > 1e-9 else 0.0
        h_d[f] = h_d[f] / s_d if abs(s_d) > 1e-9 else 0.0

    off = np.asarray(off, dtype=float)
    u_d = np.zeros(m)
    sd_u, lam_used, ratio_used, at_edge = {}, {}, {}, {}
    for f in FACTORS:
        mom, lam_f = moms[f]
        ratio_f = float(lams[f][1])
        pi = np.concatenate([h_o[f] * off[:m], h_d[f] * off[m:]])
        lay_f = _rep(layout, offset=np.asarray(Z @ pi).ravel())
        mom_f = mom.with_y(lay_f, ys[f], ws[f])
        if reml:
            # the factor's ridge re-selected on the residual AROUND ITS PRIOR SHARE: FINDINGS 15's value was
            # chosen with no prior, and a residual around a prior is smaller, so its ridge is tighter.  REML
            # profile over a log grid around that value.  "2d" searches the offense/defense ratio as well:
            # the defensive half is the one this fit reads, and a ratio fixed by the joint fit of section 15
            # (1.5 on eFG%) leaves it shrunk far too little against the offensive half's real skill.
            grid = float(lam_f) * np.geomspace(REML_GRID[0], REML_GRID[1], REML_GRID[2])
            ratios = list(RATIO_GRID) if reml == "2d" else [ratio_f]
            best = None
            for r in ratios:
                mom_r = mom_f if r == ratio_f else Moments(
                    lay_f, ys[f], ws[f], mom.season_cols,
                    MixedModelRAPM(lam=lam_f, lam_ratio=r, spec=wd.spec, lam_buckets=lam_buckets)._scale())
                mom_r.want_edf = False
                prof = [float(mom_r.solve_eig(float(l), reml=True).reml) for l in grid]
                j = int(np.argmin(prof))
                if best is None or prof[j] < best[0]:
                    best = (prof[j], j, r, mom_r)
            _, j, r, mom_f = best
            at_edge[f] = j in (0, len(grid) - 1) or (len(ratios) > 1 and r in (ratios[0], ratios[-1]))
            lam_f, ratio_f = float(grid[j]) * float(lam_scale), float(r)
        u_f = np.asarray(mom_f.solve_chol(lam_f).u, dtype=float)
        lam_used[f], ratio_used[f] = lam_f, ratio_f
        u_d += g[f] * u_f[m:]
        sd_u[f] = float(np.sqrt(np.average(u_f[m:][keep_d] ** 2, weights=poss_d[keep_d]))) if keep_d.any() else 0.0
    diag = dict(g=g, r2=r2, h_o=h_o, h_d=h_d, share_sum_o=float(s_o), share_sum_d=float(s_d), sd_u=sd_u,
                lam=lam_used, ratio=ratio_used, at_edge=at_edge,
                sd_u_pts=float(np.sqrt(np.average(u_d0[m:][keep_d] ** 2, weights=poss_d[keep_d]))) if keep_d.any() else 0.0)
    return dict(d=off[m:] + u_d, u_d=u_d, u_d_pts0=u_d0[m:], diag=diag)


def direct_layout(wd, exp, prior_offset: np.ndarray) -> _Layout:
    """The estimator's layout for a plug-in fit with beta = 0, built from the design's own parts: the exposure
    columns straight from the padded rates and the lineups (what BoxExposure.transform computes, without the
    dense -> sparse -> dense round trip), Z and F as the design built them."""
    spec, parts = wd.spec, wd.parts
    n_feat = len(exp.feature_names_)
    cached = getattr(exp, "_exposures_cache", None)
    Xo, Xd = cached if cached is not None else exp._exposures_parts(parts, parts["game_idx"])
    if exp.center and n_feat:
        Xo = Xo - exp.means_o_
        Xd = Xd - exp.means_d_
    box = np.hstack([Xo, Xd])
    Z, F = parts["Z"], parts["F"]
    is_po = F[:, spec.f_names.index("is_po")]
    offset = Z @ np.asarray(prior_offset, dtype=float)
    p1 = len(spec.f_names)
    return _Layout(Z, F, list(spec.f_names), np.zeros(p1), n_feat, 0, slice(0, 0), slice(0, p1), slice(p1, p1),
                   offset, box, is_po, 0)


@dataclass
class MspiFast:
    name: str
    sides: tuple = ("O", "D")
    mode: str = "full"
    target: str = "rapm1"
    scale: float = 1.0
    gbdt_params: dict | None = None
    lam: float | None = None
    lam_ratio: float | None = None
    off_target: str = "xpts_ft"
    def_target: str = "x3def"
    lam_buckets: dict | None = None      # extra ridge multipliers per spec.col_groups name (low_poss, high_poss, ...)
    phases: tuple = FIT_PHASES           # the stints this trains on; the held-out scoring stays RS
    decay: float | None = None           # rows of training season s weighted decay^(|s - H| - 1), H = ctx.current_h
    decay_exposure: bool = False         # the same weights on the games behind the padded rates (BoxExposure.game_mult)
    season_weights: dict | None = None   # explicit weight per season offset (s - H), e.g. {-2: 0.5, -1: 0.8, 1: 1.0};
                                         # overrides `decay`; offsets not listed get 1
    pad_scale: float = 1.0               # BoxExposure pad_scale: the padding constants of the rates times this
    pad_target: str | None = None        # BoxExposure pad_target ("league" | "poss_conditional"); None = config
    panel: str | None = None             # a role panel other than the configured one for the GBDT prior (path)
    target_d: str | None = None          # the GBDT's training target on DEFENSE when it differs from `target`
    gbdt_features: dict | None = None    # {"O": [...], "D": [...]} for the GBDT instead of the configured lists
    gbdt_params_d: dict | None = None    # chimeraboost overrides for the DEFENSIVE prior (None = the same)
    win_decay_d: float | None = None     # the window discount on defense (None = the same as offense)
    win_decay: float = 1.0               # the prior's target pooled over the player's windows with this decay
    min_den: float = 0.0                 # drop design rows under this many possessions (designcache)
    turn: bool | str = False             # the turnover-aware prior: ridge toward its settled-context value, then add
                                         # the trade delta to the held-out season's context (spm.chain_offset `turn`);
                                         # "ref" = the settled-context prior alone, no delta; "pairs" = the pair-row
                                         # prior without the turnover feature (the un-pooling control)
    turn_ref: float = 0.35               # the settled-context turnover the ridge shrinks toward (spm.TURN_REF)
    turn_sides: tuple = ("O", "D")       # the sides that take the turnover-aware prior (the other keeps the pooled one)
    def_factors: float | None = None     # HANDOFF 3.2: the defensive residual from four factor fits (efg, tov, oreb,
                                         # ftr), each with its own ridge and ratio, recombined into points allowed
                                         # (`factor_defense`).  1.0 replaces the points fit's residual, a fraction
                                         # blends the two, None = the points fit as shipped
    factor_lams: dict | None = None      # per-factor (lam, ratio) overrides of FACTOR_LAMS
    factor_lam_scale: float = 1.0        # every factor's ridge times this (after the REML choice, if on)
    factor_reml: bool | str = False      # each factor's ridge re-selected by REML on the residual around its prior
                                         # share; "2d" selects the offense/defense ratio as well
    factor_x3: bool = False              # the eFG% factor with opponent threes repriced at the shooter's expectation
    no_def_prior: bool = False           # DIAGNOSTIC: the defensive prior zeroed before the solves
    gbdt_folds: int = 0                  # player-grouped cross-fitting of the GBDT prior (GBDTPrior folds): a player
                                         # is scored by a model that never saw a row of his, so the prior cannot
                                         # read his own other windows off his rate fingerprint (scratch/foldtest.py)
    kernel: dict | None = None           # in-season mode (inseason.py): the weight of each training season by its
                                         # offset from the ANCHOR (= max(train)), e.g. {0: 1, -1: 0.5, -2: 0.25}; an
                                         # offset the kernel does not name gets 0.  Unlike `decay` / `season_weights`
                                         # it needs no held-out season, so it applies on the board as well.
    cut: float | None = None             # the share of the ANCHOR season's games the fit may see (0..1); the rest of
                                         # that season is weighted 0 in the ridge, in the exposure, and in every input
                                         # built from season tables (inseason.keep_games).  None = the whole season
    half: str | None = None              # "A" / "B": fit on that half of the GAMES only, the other weighted 0
                                         # everywhere (rows, exposure, padded rates), so the untouched half can
                                         # score it.  Playoff games alternate A/B WITHIN a series, so this is a
                                         # within-series split and both halves see the same teams and lineups.
    prior_from: object | None = None     # a callable (train, ctx, wd) -> the 2*n_ps prior offset, INSTEAD of the
                                         # box-score chain.  It is what makes a rating the prior for another
                                         # rating: `playoffs.RegularSeasonPrior` fits this same estimator on the
                                         # regular season and hands its ratings to a fit that sees only playoff
                                         # possessions, so the playoffs move a player off his season number by as
                                         # much as 6.5% of the data can justify and no more.

    def counter_columns(self) -> set | None:
        """The per-possession counters this system's two targets read, so the design need not assemble the
        other hundred (`designcache.build_window_cached`).  None if a target's needs are not declared."""
        from . import teamloo, xshoot
        from .design import target_counter_columns
        cols = set()
        for t in (self.off_target, self.def_target):
            if t in TARGETS:
                cols |= target_counter_columns(t)
            elif t in xshoot.DEFENSE_TARGET_COLUMNS:
                cols |= xshoot.DEFENSE_TARGET_COLUMNS[t]
            elif t in teamloo.TARGET_COLUMNS:
                cols |= teamloo.TARGET_COLUMNS[t]
            else:
                return None
        if self.def_factors is not None:
            cols |= FACTOR_COUNTERS
        return cols | {"poss"}

    def fit(self, train, ctx: Context) -> Ratings:
        from . import teamloo, xshoot
        from .spm import chain_offset
        cfg = ctx.cfg
        T = _timer()
        wd = ctx.design(train, "pts", tuple(self.phases), counter_cols=self.counter_columns(), min_den=self.min_den)
        T("design")
        ys = {}
        # the in-season kernel (inseason.py): one per-game weight array feeds the ridge rows, the games behind
        # the padded rates and the possessions, so they cannot disagree; `keep` names the same games for the
        # inputs built from season tables rather than from the design
        kern_mult, keep = None, None
        if self.kernel is not None or self.cut is not None:
            from .inseason import anchor_of, kernel_game_mult, keep_games
            anchor = anchor_of(train)
            kern_mult = kernel_game_mult(wd, anchor, self.kernel, self.cut)
            keep = keep_games(wd, anchor, self.cut)

        def target_y(name):
            if name not in ys:
                if name == "pts":
                    ys[name] = wd.y
                elif name in TARGETS:
                    ys[name] = derived_target(wd, name)
                else:
                    # the callable targets: a defensive repricing (xshoot) or a team-game luck
                    # adjustment (teamloo).  Both take (train, cfg, wd, keep=) and return a WindowData
                    # (or (WindowData, report)); `keep` carries the in-season cut down either path.
                    fn = xshoot.DEFENSE_TARGETS.get(name) or teamloo.TARGETS.get(name)
                    if fn is None:
                        raise KeyError(f"unknown target {name!r}: not in design.TARGETS, "
                                       f"xshoot.DEFENSE_TARGETS or teamloo.TARGETS")
                    wd_t = fn(train, cfg, wd, keep=keep)
                    ys[name] = (wd_t[0] if isinstance(wd_t, tuple) else wd_t).y
            return ys[name]

        y_o, y_d = target_y(self.off_target), target_y(self.def_target)
        T("targets")
        game_mult = None

        def season_weight(season):
            """The weight of each training season's rows for the held-out season H (1 without an H)."""
            season = np.asarray(season, dtype=float)
            if ctx.current_h is None:
                return np.ones(len(season))
            if self.season_weights:
                off = np.rint(season - float(ctx.current_h)).astype(int)
                return np.array([float(self.season_weights.get(int(o), 1.0)) for o in off])
            if self.decay is None:
                return np.ones(len(season))
            dist = np.abs(season - float(ctx.current_h))
            return float(self.decay) ** np.maximum(dist - 1.0, 0.0)

        if self.decay_exposure and (self.decay is not None or self.season_weights):
            g = wd.games
            gm = np.ones(int(g["game_idx"].max()) + 1)
            gm[g["game_idx"].to_numpy()] = season_weight(g["season"].to_numpy())
            game_mult = gm
        if self.half is not None:
            hm = (wd.game_half == str(self.half)).astype(float)
            kern_mult = hm if kern_mult is None else kern_mult * hm
        if kern_mult is not None:
            game_mult = kern_mult
        w_rows = np.asarray(wd.w, dtype=float)
        if kern_mult is not None:            # never mutate wd.w: ctx.design caches the design across systems
            w_rows = w_rows * kern_mult[wd.rows["game_idx"].to_numpy()]
        exp = make_exposure(wd, mode="full", pad_target=self.pad_target or cfg["pad_target"], game_mult=game_mult,
                            pad_scale=float(self.pad_scale), phases=tuple(self.phases))
        if wd.parts is not None:
            exp.parts = wd.parts
            exp.fit(None, sample_weight=w_rows)
        else:
            exp.fit(wd.X, sample_weight=w_rows)
        T("exposure")
        chain_kw = dict(scale=self.scale, target=self.target, params=self.gbdt_params, panel=self.panel,
                        target_d=self.target_d, features=self.gbdt_features, win_decay=self.win_decay,
                        params_d=self.gbdt_params_d, win_decay_d=self.win_decay_d)
        chain_kw["turn_ref"] = float(self.turn_ref)
        chain_kw["turn_sides"] = tuple(self.turn_sides)
        chain_kw["folds"] = int(self.gbdt_folds or 0)
        tmode = None if not self.turn else ("pairs" if self.turn == "pairs" else "ref")
        off = (np.asarray(self.prior_from(train, ctx, wd), dtype=float) if self.prior_from is not None
               else chain_offset(self.sides, self.mode, turn=tmode, **chain_kw)(train, ctx, wd, exp=exp, keep=keep))
        delta = 0.0
        if self.turn is True and self.prior_from is None:   # the trade delta: the prior at H's turnover minus at the settled value
            delta = chain_offset(self.sides, self.mode, turn="h", **chain_kw)(train, ctx, wd, exp=exp, keep=keep) - off
        T("prior")
        if self.no_def_prior:
            off = np.array(off, dtype=float)
            off[wd.spec.n_ps:] = 0.0
        nf = len(wd.spec.features)
        beta = np.zeros(2 * nf)
        lam = float(cfg["lam_plugin"] if self.lam is None else self.lam)
        ratio = float(cfg["lam_ratio_plugin"] if self.lam_ratio is None else self.lam_ratio)
        m = wd.spec.n_ps
        # one layout and one set of cross-products; the second side changes only the response
        mm = MixedModelRAPM(lam=lam, lam_ratio=ratio, beta_fixed=beta, prior_offset=off, spec=wd.spec,
                            lam_buckets=self.lam_buckets)
        w = w_rows
        if self.decay is not None or self.season_weights:
            w = w * season_weight(wd.rows["season"].to_numpy())
        if wd.parts is not None:
            layout = direct_layout(wd, exp, off)
        else:
            X, _, _ = mm._validate(exp.transform(wd.X), np.asarray(y_o, dtype=float), wd.w)
            layout = mm._layout(X)
        T("layout")
        mom = Moments(layout, np.asarray(y_o, dtype=float), w, mm._season_cols(), mm._scale())
        mom.want_edf = False
        T("moments")
        u = {"o": np.asarray(mom.solve_chol(lam).u, dtype=float)}
        u["d"] = np.asarray(mom.with_y(layout, np.asarray(y_d, dtype=float), w).solve_chol(lam).u, dtype=float)
        T("solves")
        d_resid = u["d"][m:]
        self.factor_diag = None
        if self.def_factors is not None:
            x3 = xshoot.expected_threes(train, cfg, wd, keep=keep)[0] if self.factor_x3 else None
            fd = factor_defense(wd, layout, off, y_o, y_d, target_y("pts"), w, lam, ratio, self.lam_buckets,
                                factor_lams=self.factor_lams, lam_scale=float(self.factor_lam_scale),
                                reml=self.factor_reml, x3=x3)
            a = float(self.def_factors)
            d_resid = a * fd["u_d"] + (1.0 - a) * d_resid
            self.factor_diag = fd["diag"]
            T("factors")
        off = off + delta            # the ridge shrank toward the settled-context prior; the rating carries the delta
        df = pd.DataFrame({"player_id": wd.spec.ps_table["player_id"].to_numpy(),
                           "o": off[:m] + u["o"][:m], "d": off[m:] + d_resid,
                           "poss": np.asarray(exp.season_poss_off_, dtype=float),
                           "poss_d": np.asarray(exp.season_poss_def_, dtype=float),
                           "prior_o": off[:m], "prior_d": off[m:]})
        if kern_mult is not None:
            # a player the kernel gave no possessions is not rated: his prior was built from inputs the cut
            # kept off, and scoring him would put a number on the games the fit is being tested against
            df = df[df.poss > 0].reset_index(drop=True)
        return Ratings(df)
