"""The role prior: APM -> Simple SPM -> the ridge pulled toward it.

Three stages, per side (O and D), per window:

  APM        the plugin ridge with no box prior and a TINY penalty (config spm.apm_lam, 100 against the
             shipped 18,352), so nobody is pulled toward zero by more than a few percent.  The rating a
             player "would have" if we trusted his own possessions.  Noisy for the bench, unbiased for
             everyone, and the only target here that is not itself a shrunk number.
  Simple SPM a possession-weighted ridge of APM on seven role-and-age inputs (roles.INPUTS: poss_pct,
             its square, the games-started share gs_pct, its square, age, age^2, age^3), fit
             LEAVE-WINDOW-OUT so the prior used for a window never saw that window's APM.
  RAPM_1     the shipped ridge (`lam_plugin`, `lam_ratio_plugin`) with `prior_offset` = Simple SPM.
             A player with no possessions gets exactly his role level; a 10th man sits almost on it.

Sign convention: everything here is in the model's RAW sign on both sides (defense positive = points
allowed), like `xrapm_panel.u`.  Offsets built here go straight into `plugin_fit(prior_offset=...)`
without negation; the flip to positive-is-good happens only in `windows.player_ratings_table` and
scripts/08_ratings.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .cv import make_exposure, plugin_fit
from .roles import CAREER_INPUTS, INPUTS, RAW_INPUTS, career_inputs, design7, window_inputs

SIDES = ("O", "D")


def centred_rates(exp) -> tuple[np.ndarray, np.ndarray]:
    """The per-player padded full-window rates centred on the average player (the exposure columns are
    centred on the sum of five, so divide the lineup mean by five) -- what the panel stores as the 13 rates."""
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    return ro, rd


def apm_fit(wd, cfg, lam: float | None = None, lam_ratio: float | None = None) -> dict:
    """Adjusted plus-minus: beta = 0, penalty `lam` (config spm.apm_lam).  Returns u per side, possessions,
    the centred rates and the fitted pipeline."""
    s = cfg.get("spm", {})
    lam = float(s.get("apm_lam", 100.0) if lam is None else lam)
    lam_ratio = float(s.get("apm_lam_ratio", 1.0) if lam_ratio is None else lam_ratio)
    nf = len(wd.spec.features)
    m = wd.spec.n_ps
    pipe = plugin_fit(wd, np.zeros(2 * nf), lam=lam, lam_ratio=lam_ratio, pad_target=cfg["pad_target"])
    mm, exp = pipe["mm"], pipe["exposure"]
    ro, rd = centred_rates(exp)
    return dict(pipe=pipe, lam=lam, u_o=np.asarray(mm.u_[:m], dtype=float), u_d=np.asarray(mm.u_[m:], dtype=float),
                poss_o=np.asarray(exp.season_poss_off_, dtype=float), poss_d=np.asarray(exp.season_poss_def_, dtype=float),
                ro=ro, rd=rd)


@dataclass(frozen=True)
class SPMFit:
    """A fitted Simple SPM for one side: standardised-input ridge coefficients and the standardisation."""
    side: str
    coef: np.ndarray          # 7, on standardised inputs
    intercept: float
    mean: np.ndarray          # 7
    sd: np.ndarray            # 7
    n: int
    pen: float

    def table(self) -> pd.DataFrame:
        """Coefficients per unit of the RAW input (coef / sd), for reading."""
        return pd.DataFrame({"side": self.side, "input": INPUTS, "coef_std": self.coef, "coef_raw": self.coef / self.sd})


def _weighted_std(X, w):
    mu = np.average(X, axis=0, weights=w)
    var = np.average((X - mu) ** 2, axis=0, weights=w)
    sd = np.sqrt(var)
    return mu, np.where(sd > 0, sd, 1.0)


def fit_spm(panel: pd.DataFrame, side: str, exclude=(), pen: float = 1.0, min_poss: float = 500.0,
            target: str = "apm") -> SPMFit:
    """Possession-weighted ridge of `target` (APM, raw sign) on the seven role inputs, one side,
    on every window NOT in `exclude`.  Inputs are standardised (weighted), the intercept is free."""
    ex = set(exclude)
    d = panel[(panel.side == side) & ~panel.window.isin(ex) & (panel.poss >= float(min_poss))]
    if len(d) < 50:
        raise RuntimeError(f"fit_spm {side}: only {len(d)} rows after excluding {sorted(ex)}")
    X = design7(d["poss_pct"].to_numpy(), d["gs_pct"].to_numpy(), d["age"].to_numpy())
    w = d["poss"].to_numpy(dtype=float)
    y = d[target].to_numpy(dtype=float)
    mu, sd = _weighted_std(X, w)
    Xs = (X - mu) / sd
    A = np.column_stack([np.ones(len(d)), Xs])
    P = float(pen) * np.eye(A.shape[1])
    P[0, 0] = 0.0
    AtW = (A * w[:, None]).T
    b = np.linalg.solve(AtW @ A + P, AtW @ y)
    return SPMFit(side=side, coef=b[1:], intercept=float(b[0]), mean=mu, sd=sd, n=int(len(d)), pen=float(pen))


def spm_predict(fit: SPMFit, poss_pct, gs_pct, age) -> np.ndarray:
    X = design7(poss_pct, gs_pct, age)
    return fit.intercept + ((X - fit.mean) / fit.sd) @ fit.coef


def spm_offset(fit_o: SPMFit, fit_d: SPMFit, inputs: pd.DataFrame, poss_o, poss_d, centre: bool = True) -> np.ndarray:
    """The (2m,) raw-sign offset [O | D] from the window's role inputs (aligned to ps_idx), possession-centred
    per side so the level of the board does not move against the fixed block."""
    parts = []
    for fit, poss in ((fit_o, poss_o), (fit_d, poss_d)):
        g = spm_predict(fit, inputs["poss_pct"].to_numpy(), inputs["gs_pct"].to_numpy(), inputs["age"].to_numpy())
        w = np.maximum(np.asarray(poss, dtype=float), 0.0)
        if centre and w.sum() > 0:
            g = g - np.average(g, weights=w)
        parts.append(g)
    return np.concatenate(parts)


def apm_lambda_check(panels: dict, side: str, **kw) -> pd.DataFrame:
    """SPM coefficients (per unit of the raw input) at each APM penalty side by side, with the largest
    difference relative to the coefficient's own scale.  `panels`: lam -> panel with an `apm` column."""
    cols = {}
    for lam, p in panels.items():
        f = fit_spm(p, side, **kw)
        cols[f"lam_{lam:g}"] = f.coef / f.sd
    t = pd.DataFrame(cols, index=INPUTS)
    scale = t.abs().max(axis=1).replace(0.0, np.nan)
    t["max_rel_diff"] = (t.max(axis=1) - t.min(axis=1)) / scale
    return t


def season_of_units(wd) -> np.ndarray:
    """The season (year) each Z unit played most, from the design."""
    return np.asarray(wd.spec.seasons, dtype=np.int64)[np.asarray(wd.spec.season_of_ps, dtype=np.int64)]


TURN_REF = 0.35     # a typical stayer's teammate turnover, season to season (median 0.36; FINDINGS 24)


def chain_offset(gbdt_sides=(), mode: str = "residual", scale: float = 1.0, target: str = "rapm1",
                 params: dict | None = None, panel: str | None = None, target_d: str | None = None,
                 features: dict | None = None, win_decay: float = 1.0,
                 params_d: dict | None = None, win_decay_d: float | None = None,
                 turn: str | None = None, turn_ref: float = TURN_REF, turn_sides=("O", "D")) -> Callable:
    """The per-player offset builder for a PluginSystem.  Signature `offset(train, ctx, wd) -> (2 * n_ps,)`,
    raw sign, possession-centred per side.

      mode "residual"  Simple SPM on both sides, plus the GBDT trained on the residual beyond it (`u`) on
                       `gbdt_sides`.
      mode "full"      the GBDT trained on RAPM_1 itself with the role inputs among its features, alone, on
                       `gbdt_sides`; the Simple SPM on any side not in `gbdt_sides`.

    Everything is kept off the window labels in `ctx.labels(train)`: the SPM is refit without them, and the
    GBDT model for that exclusion set is fetched from `ctx.gbdt` / `ctx.mspi` (cached per set).

    `turn`: the TURNOVER-AWARE prior (GBDTPrior `turn`, trained on window pairs with the teammate turnover as a
    feature).  "ref" evaluates it at `turn_ref` for everyone -- the box line's worth in a settled context, the
    offset the ridge should shrink toward, since the block's possessions were played in the block's context;
    "h" evaluates it at each player's turnover of the held-out season with respect to the block
    (turnover.familiar_share; `turn_ref` for a player H never sees, and for the shipped board, which has no H).
    The difference of the two is the trade delta, applied AFTER the ridge (fastfit.MspiFast)."""
    sides = tuple(gbdt_sides)
    if mode not in ("residual", "full"):
        raise ValueError(f"mode must be 'residual' or 'full', got {mode!r}")
    if turn not in (None, "ref", "h", "pairs"):
        raise ValueError(f"turn must be None, 'ref', 'h' or 'pairs', got {turn!r}")
    # `scale` multiplies the GBDT's prediction before it becomes the offset (the criterion said the prior
    # trained on the shrunk RAPM_1 is too timid: starters want 1.2, deep bench 2.0); `target` = "apm" uses
    # the chain trained on unshrunk APM instead (ctx.mspi_apm)

    def offset(train, ctx, wd, exp=None) -> np.ndarray:
        if ctx.rpanel is None or ctx.role_inputs is None:
            raise RuntimeError("outputs/role_panel.parquet or data/cache/roles.parquet is missing; "
                               "run scripts/49_role_panel.py")
        cfg = ctx.cfg
        s = cfg.get("spm", {})
        exclude = ctx.labels(train)
        inputs = window_inputs(wd, ctx.role_inputs, cap=float(cfg.get("roles", {}).get("share_cap", 0.9)))
        if exp is None:
            exp = make_exposure(wd, mode="full", pad_target=cfg["pad_target"]).fit(wd.X, sample_weight=wd.w)
        poss_o = np.asarray(exp.season_poss_off_, dtype=float)
        poss_d = np.asarray(exp.season_poss_def_, dtype=float)
        m = wd.spec.n_ps
        if mode == "full" and set(sides) >= {"O", "D"}:
            off = np.zeros(2 * m)              # the GBDT replaces the SPM on both sides: no SPM fit needed
        else:
            fo = fit_spm(ctx.rpanel, "O", exclude, pen=float(s.get("pen", 1.0)), min_poss=float(s.get("min_poss", 500)))
            fd = fit_spm(ctx.rpanel, "D", exclude, pen=float(s.get("pen", 1.0)), min_poss=float(s.get("min_poss", 500)))
            off = spm_offset(fo, fd, inputs, poss_o, poss_d)
        if sides:
            t_d = target_d or target
            def col(t):
                return t if (t == "apm" or t.startswith("blend")) else None

            # the two sides may want different priors: the criterion likes a rich feature set and a bagged,
            # searched booster on OFFENSE, and the consensus floors do not tolerate either on DEFENSE
            # (FINDINGS 21.26).  `params_d` / `win_decay_d` default to the offensive settings.
            p_d = params if params_d is None else params_d
            wd_d = win_decay if win_decay_d is None else float(win_decay_d)
            if (params or p_d or panel or target_d or features or target.startswith("blend")
                    or win_decay != 1.0 or wd_d != 1.0 or turn):
                tkey = "pairs" if turn == "pairs" else bool(turn)
                prior_o = ctx.prior(mode, col(target), params, panel, features, win_decay,
                                    turn=tkey if "O" in turn_sides else False)
                prior_d = ctx.prior(mode, col(t_d), p_d, panel, features, wd_d,
                                    turn=tkey if "D" in turn_sides else False)
            else:
                prior_o = ctx.gbdt if mode == "residual" else getattr(ctx, "mspi_apm" if target == "apm" else "mspi", None)
                prior_d = prior_o
            if prior_o is None or prior_d is None:
                raise RuntimeError(f"Context has no GBDT prior for mode {mode!r} (outputs/role_panel.parquet)")
            from .gbdt_prior import DREDGE_ANY, SHOTQ, gbdt_offset
            ro, rd = centred_rates(exp)
            wants = set(prior_o.features["O"]) | set(prior_d.features["D"])
            shots = None
            if wants & set(SHOTQ):
                # the shot totals of the TRAINING block only, the way the panel row's came from its window's
                # three seasons -- the held-out season is not among `train`, so nothing here has seen it
                from .xshoot import player_shot_frame
                shots = player_shot_frame(train, cfg, wd.spec.ps_table["player_id"].to_numpy())
            dredge = None
            if wants & set(DREDGE_ANY):
                # and the same for the play-by-play event counts: the training block's own totals and its own
                # league levels, so the padding target is the era's and never the held-out season's
                from .dredge import player_dredge_frame
                dredge = player_dredge_frame(train, cfg, wd.spec.ps_table["player_id"].to_numpy())
            extra = inputs[list(RAW_INPUTS)].reset_index(drop=True)
            if wants & set(CAREER_INPUTS):
                # seasons played, career possessions and entry age BEFORE the block's first season, so the
                # count stops two seasons short of H and cannot have seen it
                ci = career_inputs(ctx.role_inputs, min(int(s_) for s_ in train),
                                   wd.spec.ps_table["player_id"].to_numpy(), age=inputs["age"].to_numpy())
                extra = pd.concat([extra, ci[list(CAREER_INPUTS)]], axis=1)
            from .bio import PLAYER_INPUTS
            from .gbdt_prior import BIO_BINS
            want_bio = (wants & set(PLAYER_INPUTS)) | {BIO_BINS[f][0] for f in wants if f in BIO_BINS}
            if want_bio:
                # height, weight, draft slot (a constant per player) and tenure with his team over the training
                # block; the held-out season's rosters are skipped so the tenure chain never reads H.  A binned
                # name (height2, weight15) needs its base column, which add_derived bins in both paths alike
                from .bio import player_inputs
                from .roles import build_roles
                pi = player_inputs(cfg, build_roles(cfg, verbose=False), [int(s_) for s_ in train],
                                   wd.spec.ps_table["player_id"].to_numpy(),
                                   exclude_seasons=() if ctx.current_h is None else (int(ctx.current_h),))
                extra = pd.concat([extra, pi[[c for c in PLAYER_INPUTS if c in want_bio]]], axis=1)
            if turn in ("ref", "h"):
                extra = extra.assign(turn=float(turn_ref))
                if turn == "h" and ctx.current_h is not None:
                    from .turnover import cached_table, familiar_share
                    tm = cached_table(ctx)
                    if tm is None:
                        raise RuntimeError("data/cache/teammates.parquet is missing (turnover.build_teammates)")
                    f = familiar_share(tm, list(train), [int(ctx.current_h)], player_ids=wd.spec.ps_table["player_id"].to_numpy())
                    extra["turn"] = f.turnover.fillna(float(turn_ref)).to_numpy()
            from .gbdt_prior import PAST, past_inputs
            if wants & set(PAST):
                # his on-court record before the block, per side: every panel window before the excluded ones
                ids = wd.spec.ps_table["player_id"].to_numpy()
                extra = {s_: pd.concat([extra, past_inputs(ctx.rpanel, s_, exclude, ids)], axis=1) for s_ in ("O", "D")}
            common = dict(features=list(wd.spec.features), extra=extra,
                          raw=(exp.season_rates_, exp.season_rates_d_), shots=shots, dredge=dredge)
            g = np.zeros(2 * m)
            if "O" in sides:
                g += gbdt_offset(prior_o, ro, rd, season_of_units(wd), poss_o, poss_d, exclude, sides=("O",), **common)
            if "D" in sides:
                g += gbdt_offset(prior_d, ro, rd, season_of_units(wd), poss_o, poss_d, exclude, sides=("D",), **common)
            g = g * float(scale)
            if mode == "residual":
                off = off + g
            else:                                   # the GBDT replaces the SPM on the sides it covers
                for side, sl in (("O", slice(0, m)), ("D", slice(m, 2 * m))):
                    if side in sides:
                        off[sl] = g[sl]
        return off

    offset.__name__ = (f"chain_offset_{mode}_{target}{'_' + target_d if target_d else ''}_x{scale:g}_" + ("".join(sides) or "spm")
                       + (f"_turn{turn}" if turn else ""))
    return offset


def panel_inputs_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Possession-weighted mean and sd of the raw inputs and of apm/spm per window and side, for the log."""
    rows = []
    for (w, s), d in panel.groupby(["window", "side"]):
        wt = d["poss"].to_numpy(dtype=float)
        r = dict(window=w, side=s, n=len(d))
        for c in [*RAW_INPUTS, "apm", "spm", "u", "rapm1"]:
            if c in d.columns:
                v = d[c].to_numpy(dtype=float)
                mu = np.average(v, weights=wt) if wt.sum() > 0 else float("nan")
                r[f"{c}_mean"] = mu
                r[f"{c}_sd"] = float(np.sqrt(np.average((v - mu) ** 2, weights=wt))) if wt.sum() > 0 else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)
