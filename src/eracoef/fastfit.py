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
from .estimator import MixedModelRAPM
from .holdout import Context, Ratings


def derived_target(wd, name: str) -> np.ndarray:
    """A `design.TARGETS` response recomputed from the design's own counters (the rows are the same, the
    weight is the same: only y changes)."""
    tg = TARGETS[name]
    c = wd.counters
    num = sum(k * c[col].to_numpy(dtype=float) for col, k in tg["num"].items())
    den = c[tg["den"]].to_numpy(dtype=float)
    return tg["scale"] * num / den


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

    def fit(self, train, ctx: Context) -> Ratings:
        from . import xshoot
        from .spm import chain_offset
        cfg = ctx.cfg
        wd = ctx.design(train, "pts")
        y_o = derived_target(wd, self.off_target) if self.off_target != "pts" else wd.y
        if self.def_target == "pts":
            y_d = wd.y
        elif self.def_target in TARGETS:
            y_d = derived_target(wd, self.def_target)
        else:
            wd_d = xshoot.DEFENSE_TARGETS[self.def_target](train, cfg, wd)
            y_d = (wd_d[0] if isinstance(wd_d, tuple) else wd_d).y
        exp = make_exposure(wd, mode="full", pad_target=cfg["pad_target"]).fit(wd.X, sample_weight=wd.w)
        Xt = exp.transform(wd.X)
        off = chain_offset(self.sides, self.mode, scale=self.scale, target=self.target,
                           params=self.gbdt_params)(train, ctx, wd, exp=exp)
        nf = len(wd.spec.features)
        beta = np.zeros(2 * nf)
        lam = float(cfg["lam_plugin"] if self.lam is None else self.lam)
        ratio = float(cfg["lam_ratio_plugin"] if self.lam_ratio is None else self.lam_ratio)
        m = wd.spec.n_ps
        u = {}
        for side, y in (("o", y_o), ("d", y_d)):
            mm = MixedModelRAPM(lam=lam, lam_ratio=ratio, beta_fixed=beta, prior_offset=off, spec=wd.spec)
            mm.fit(Xt, np.asarray(y, dtype=float), sample_weight=wd.w)
            u[side] = np.asarray(mm.u_, dtype=float)
        df = pd.DataFrame({"player_id": wd.spec.ps_table["player_id"].to_numpy(),
                           "o": off[:m] + u["o"][:m], "d": off[m:] + u["d"][m:],
                           "poss": np.asarray(exp.season_poss_off_, dtype=float),
                           "prior_o": off[:m], "prior_d": off[m:]})
        return Ratings(df)
