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

    def fit(self, train, ctx: Context) -> Ratings:
        from . import xshoot
        from .spm import chain_offset
        cfg = ctx.cfg
        wd = ctx.design(train, "pts", tuple(self.phases))
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
        game_mult = None
        if self.decay is not None and self.decay_exposure and ctx.current_h is not None:
            g = wd.games
            gm = np.ones(int(g["game_idx"].max()) + 1)
            dist = np.abs(g["season"].to_numpy(dtype=float) - float(ctx.current_h))
            gm[g["game_idx"].to_numpy()] = float(self.decay) ** np.maximum(dist - 1.0, 0.0)
            game_mult = gm
        exp = make_exposure(wd, mode="full", pad_target=cfg["pad_target"], game_mult=game_mult)
        if wd.parts is not None:
            exp.parts = wd.parts
            exp.fit(None, sample_weight=wd.w)
        else:
            exp.fit(wd.X, sample_weight=wd.w)
        off = chain_offset(self.sides, self.mode, scale=self.scale, target=self.target,
                           params=self.gbdt_params)(train, ctx, wd, exp=exp)
        nf = len(wd.spec.features)
        beta = np.zeros(2 * nf)
        lam = float(cfg["lam_plugin"] if self.lam is None else self.lam)
        ratio = float(cfg["lam_ratio_plugin"] if self.lam_ratio is None else self.lam_ratio)
        m = wd.spec.n_ps
        # one layout and one set of cross-products; the second side changes only the response
        mm = MixedModelRAPM(lam=lam, lam_ratio=ratio, beta_fixed=beta, prior_offset=off, spec=wd.spec,
                            lam_buckets=self.lam_buckets)
        w = np.asarray(wd.w, dtype=float)
        if self.decay is not None and ctx.current_h is not None:
            dist = np.abs(wd.rows["season"].to_numpy(dtype=float) - float(ctx.current_h))
            w = w * float(self.decay) ** np.maximum(dist - 1.0, 0.0)
        if wd.parts is not None:
            layout = direct_layout(wd, exp, off)
        else:
            X, _, _ = mm._validate(exp.transform(wd.X), np.asarray(y_o, dtype=float), wd.w)
            layout = mm._layout(X)
        mom = Moments(layout, np.asarray(y_o, dtype=float), w, mm._season_cols(), mm._scale())
        mom.want_edf = False
        u = {"o": np.asarray(mom.solve_chol(lam).u, dtype=float)}
        u["d"] = np.asarray(mom.with_y(layout, np.asarray(y_d, dtype=float), w).solve_chol(lam).u, dtype=float)
        df = pd.DataFrame({"player_id": wd.spec.ps_table["player_id"].to_numpy(),
                           "o": off[:m] + u["o"][:m], "d": off[m:] + u["d"][m:],
                           "poss": np.asarray(exp.season_poss_off_, dtype=float),
                           "prior_o": off[:m], "prior_d": off[m:]})
        return Ratings(df)
