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
from .design import TARGETS
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
    phases: tuple = ("RS",)              # ("RS", "PO"): train on the playoff stints too (the held-out scoring stays RS)
    decay: float | None = None           # rows of training season s weighted decay^(|s - H| - 1), H = ctx.current_h
    decay_exposure: bool = False         # the same weights on the games behind the padded rates (BoxExposure.game_mult)
    season_weights: dict | None = None   # explicit weight per season offset (s - H), e.g. {-2: 0.5, -1: 0.8, 1: 1.0};
                                         # overrides `decay`; offsets not listed get 1
    pad_scale: float = 1.0               # BoxExposure pad_scale: the padding constants of the rates times this
    pad_target: str | None = None        # BoxExposure pad_target ("league" | "poss_conditional"); None = config
    panel: str | None = None             # a role panel other than the configured one for the GBDT prior (path)
    target_d: str | None = None          # the GBDT's training target on DEFENSE when it differs from `target`
    gbdt_features: dict | None = None    # {"O": [...], "D": [...]} for the GBDT instead of the configured lists
    min_den: float = 0.0                 # drop design rows under this many possessions (designcache)

    def counter_columns(self) -> set | None:
        """The per-possession counters this system's two targets read, so the design need not assemble the
        other hundred (`designcache.build_window_cached`).  None if a target's needs are not declared."""
        from . import xshoot
        from .design import target_counter_columns
        cols = set()
        for t in (self.off_target, self.def_target):
            if t in TARGETS:
                cols |= target_counter_columns(t)
            elif t in xshoot.DEFENSE_TARGET_COLUMNS:
                cols |= xshoot.DEFENSE_TARGET_COLUMNS[t]
            else:
                return None
        return cols | {"poss"}

    def fit(self, train, ctx: Context) -> Ratings:
        from . import xshoot
        from .spm import chain_offset
        cfg = ctx.cfg
        T = _timer()
        wd = ctx.design(train, "pts", tuple(self.phases), counter_cols=self.counter_columns(), min_den=self.min_den)
        T("design")
        ys = {}

        def target_y(name):
            if name not in ys:
                if name == "pts":
                    ys[name] = wd.y
                elif name in TARGETS:
                    ys[name] = derived_target(wd, name)
                else:
                    wd_t = xshoot.DEFENSE_TARGETS[name](train, cfg, wd)
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
        exp = make_exposure(wd, mode="full", pad_target=self.pad_target or cfg["pad_target"], game_mult=game_mult,
                            pad_scale=float(self.pad_scale))
        if wd.parts is not None:
            exp.parts = wd.parts
            exp.fit(None, sample_weight=wd.w)
        else:
            exp.fit(wd.X, sample_weight=wd.w)
        T("exposure")
        off = chain_offset(self.sides, self.mode, scale=self.scale, target=self.target,
                           params=self.gbdt_params, panel=self.panel, target_d=self.target_d,
                           features=self.gbdt_features)(train, ctx, wd, exp=exp)
        T("prior")
        nf = len(wd.spec.features)
        beta = np.zeros(2 * nf)
        lam = float(cfg["lam_plugin"] if self.lam is None else self.lam)
        ratio = float(cfg["lam_ratio_plugin"] if self.lam_ratio is None else self.lam_ratio)
        m = wd.spec.n_ps
        # one layout and one set of cross-products; the second side changes only the response
        mm = MixedModelRAPM(lam=lam, lam_ratio=ratio, beta_fixed=beta, prior_offset=off, spec=wd.spec,
                            lam_buckets=self.lam_buckets)
        w = np.asarray(wd.w, dtype=float)
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
        df = pd.DataFrame({"player_id": wd.spec.ps_table["player_id"].to_numpy(),
                           "o": off[:m] + u["o"][:m], "d": off[m:] + u["d"][m:],
                           "poss": np.asarray(exp.season_poss_off_, dtype=float),
                           "prior_o": off[:m], "prior_d": off[m:]})
        return Ratings(df)
