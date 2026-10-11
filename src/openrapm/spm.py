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
allowed).  scripts/49_role_panel.py builds the role panel from these pieces.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cv import plugin_fit
from .data.roles import INPUTS, RAW_INPUTS, design7

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


def season_of_units(wd, weights=None) -> np.ndarray:
    """The season (year) each Z unit played most, from the design.

    `weights`: per player-season possessions to decide "most" with, instead of the design's own unweighted
    ones -- the exposure's `psx_poss_off_`, which carries the in-season kernel and cut (retired; tag
    archive/research-2026-10).  Without it a player whose current season is cut away would still be labelled
    with it.
    """
    years = np.asarray(wd.spec.seasons, dtype=np.int64)
    if weights is None:
        return years[np.asarray(wd.spec.season_of_ps, dtype=np.int64)]
    spec = wd.spec
    w = np.asarray(weights, dtype=float)
    ps_of_psx = spec.ps_of_psx if spec.ps_of_psx is not None else np.arange(spec.n_psx)
    season_of_psx = spec.season_of_psx if spec.season_of_psx is not None else spec.season_of_ps
    best = np.full(spec.n_ps, -1.0)
    out = np.asarray(spec.season_of_ps, dtype=np.int64).copy()
    for k in np.argsort(w):
        i = int(ps_of_psx[k])
        if w[k] >= best[i]:
            best[i], out[i] = w[k], int(season_of_psx[k])
    return years[out]


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
