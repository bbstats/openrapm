"""Shot-quality models: the probability a LEAGUE-AVERAGE shooter makes this exact shot (the shot-quality build).

The owner's rulings (2026-10-06): the quality describes the shot alone -- not who took it, not who defended it --
and no play-by-play scorer tag (`subType`) is ever an input.  Everything here is read from the shot frame
(shotframe.derive) and the rebuilt shot clock (shotclock.rebuild).

The model.  Three penalised logistic regressions -- shots at the rim (two-pointers within 4 ft, and the
pre-2011 twos with no coordinates), other twos, threes -- on BLOCKS of inputs, so the ablation can add one block
at a time:

    spot      distance (natural spline), angle off the rim's axis, corner three, the no-coordinates cell
    start     how the possession began, seconds into it (spline), and the early seconds of each start type
    putback   seconds since the offence's own offensive rebound, in bands
    clock     the rebuilt shot clock (what the offence really had: the game clock when shorter) in FOUR BANDS
              (0-4, 4-10, 10-18, 18-24 s), whether the clock was off, and the kind of its last reset.  Bands, not a
              spline: the rebuild is 0.8 s off at the median against the 2014-15 truth, but the shot-clock
              violations read only 68-74% within 2 s of zero in every season (stage-3 gate failed as written; the
              plan's fallback).  `clockfine` (the spline) is an ablation row only
    context   period, the last seconds of a period, score margin, garbage time
    fatigue   the shooter's game seconds on court since he last came on
    prev      for a possession that began with a defensive rebound or after a make, the shot that handed it
              over: a three, a long two, blocked

Two OFFSETS are fitted alongside and set to ZERO when pricing: one per shooter-season and one per arena-season
(ridge-penalised, by backfitting).  Fitted, they stop the inputs from absorbing who takes which shots (good
shooters take harder ones) and the arena's coordinate habits; zeroed, they leave a league-average shooter in a
league-average arena.

Training is on OTHER seasons only (`train_seasons`): a block of three rated seasons never sees itself or the
season on either side, nor the current season (ruling 3).  The level is then set per season, game half, shot
value and distance band from the OTHER half's makes, padded toward zero with `LEVEL_K` attempts (`relevel`):
the league's shooting moves year to year (2P% .465 in 2000-02, .547 in 2024-26) and the half being priced
never sets its own level.

Model layer: frames in, arrays out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BLOCKS = ("spot", "start", "putback", "clock", "context", "fatigue", "prev")
# ablation-only blocks: never in the default list
EXTRA_BLOCKS = ("clockfine", "tracking", "start_type", "start_time", "start_early")
SUBMODELS = ("rim", "mid", "three")
RIM_FT = 4.0
LEVEL_K = 150.0
START_TYPES = ("made_fg", "made_ft", "dreb", "dreb_team", "steal", "dead_tov", "jump_ball", "period_start", "forced")
RESET_KINDS = ("oreb", "dfoul", "jump")          # besides the start types themselves
KNOTS = {"rim": (0.5, 1.5, 2.5, 3.5), "mid": (5.0, 8.0, 12.0, 16.0, 20.0, 23.0),
         "three": (22.5, 24.0, 25.0, 26.5, 28.5, 32.0)}
# The other-half level cells (`relevel`): one-foot bins where the shots are dense, as the shipped curve's are (a
# season's own bins padded toward its spline with 150 shots), so the season's distance shape is the other half's
# and the model supplies only what the play-by-play adds inside a bin.  Coarse bands lost on threes (2026-10-06).
BANDS = {"rim": (0.0, 1.0, 2.0, 3.0, 4.0),
         "mid": (4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 99.0),
         "three": (0.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 27.0, 28.0, 30.0, 33.0, 99.0)}


# ------------------------------------------------------------------------------------------ the shared numerics
def natural_spline(x: np.ndarray, knots) -> np.ndarray:
    """Natural cubic spline basis (ESL 5.4): [1, x, N_3 .. N_K]."""
    k = np.asarray(knots, dtype=float)
    K = len(k)

    def d(j):
        return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3) / (k[-1] - k[j])

    cols = [np.ones_like(x), x] + [d(j) - d(K - 2) for j in range(K - 2)]
    return np.column_stack(cols)


def logistic_irls(X, y, w=None, ridge=1e-3, iters=25, offset=None, beta0=None, penalise_first=False):
    """IRLS for a logistic regression with a ridge on every coefficient but the first (the intercept)."""
    n, p = X.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=float).copy()
    P = ridge * np.eye(p)
    if not penalise_first:
        P[0, 0] = 0.0
    for _ in range(iters):
        eta = X @ beta + off
        mu = 1.0 / (1.0 + np.exp(-eta))
        s = np.maximum(mu * (1 - mu), 1e-6) * w
        z = X @ beta + (y - mu) / np.maximum(mu * (1 - mu), 1e-6)
        new = np.linalg.solve((X * s[:, None]).T @ X + P, (X * s[:, None]).T @ z)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------------------------------------ the design
def submodel_of(frame: pd.DataFrame) -> np.ndarray:
    two = frame["value"].to_numpy() == 2
    rim = two & ((frame["dist_xy"].to_numpy(float) < RIM_FT) | frame["noloc"].to_numpy(bool))
    return np.where(~two, "three", np.where(rim, "rim", "mid"))


def _onehot(values, levels) -> np.ndarray:
    v = np.asarray(values, dtype=object)
    return np.column_stack([(v == lv).astype(float) for lv in levels])


def _bands(x, edges) -> np.ndarray:
    """One column per band [e_i, e_{i+1}); NaN falls in no band."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([((x >= a) & (x < b)).astype(float) for a, b in zip(edges[:-1], edges[1:])])


def block_columns(f: pd.DataFrame, sub: str, block: str) -> tuple[np.ndarray, list]:
    """The columns of one block for one sub-model, and their names.  No intercept (the design adds it)."""
    if block == "spot":
        d = np.clip(f["dist_xy"].to_numpy(float), 0.0, 40.0)
        S = natural_spline(d, KNOTS[sub])[:, 1:]
        cols, names = [S], [f"dist{i}" for i in range(S.shape[1])]
        ang = np.clip(f["angle"].to_numpy(float), 0.0, 90.0) / 90.0
        if sub != "rim":
            cols += [ang[:, None], (ang ** 2)[:, None]]
            names += ["angle", "angle2"]
        if sub == "three":
            cols.append(f["corner3"].to_numpy(float)[:, None])
            names.append("corner3")
        if sub == "rim":
            cols.append(f["noloc"].to_numpy(float)[:, None])
            names.append("noloc")
        return np.column_stack(cols), names
    if block in ("start_type", "start_time", "start_early"):           # ablation-only parts of `start`
        X, nm = block_columns(f, sub, "start")
        keep = [i for i, n in enumerate(nm) if (block == "start_type" and n.startswith("start_"))
                or (block == "start_time" and n.startswith("tposs")) or (block == "start_early" and n.startswith("early_"))]
        return X[:, keep], [nm[i] for i in keep]
    if block == "start":
        st = f["poss_start"].to_numpy(object)
        t = np.clip(f["secs_into_poss"].to_numpy(float), 0.0, 30.0)
        O = _onehot(st, START_TYPES[1:])                             # made_fg is the reference
        S = natural_spline(t, (2.0, 5.0, 8.0, 12.0, 18.0))[:, 1:]
        early = (t <= 6.0).astype(float)
        E = np.column_stack([early * (st == k) for k in ("dreb", "steal", "made_fg", "made_ft", "dead_tov")])
        return np.column_stack([O, S, E]), ([f"start_{k}" for k in START_TYPES[1:]] + [f"tposs{i}" for i in range(S.shape[1])]
                                           + [f"early_{k}" for k in ("dreb", "steal", "made_fg", "made_ft", "dead_tov")])
    if block == "putback":
        s = f["secs_since_oreb"].to_numpy(float)
        B = _bands(s, (0.0, 1.0, 2.0, 4.0, 8.0, 1e9))
        return B, ["oreb_0_1", "oreb_1_2", "oreb_2_4", "oreb_4_8", "oreb_8p"]
    if block in ("clock", "clockfine"):
        sc = np.clip(f["sc_eff"].to_numpy(float), 0.0, 24.0)
        if block == "clock":
            S = _bands(sc, (0.0, 4.0, 10.0, 18.0, 24.01))[:, [0, 1, 3]]      # 10-18 s is the reference
            sn = ["sc_0_4", "sc_4_10", "sc_18_24"]
        else:
            S = natural_spline(sc, (1.5, 4.0, 8.0, 13.0, 19.0, 23.0))[:, 1:]
            sn = [f"sc{i}" for i in range(S.shape[1])]
        off = f["clock_off"].to_numpy(float)[:, None]
        R = _onehot(f["reset_kind"].to_numpy(object), RESET_KINDS)
        return np.column_stack([S, off, R]), sn + ["clock_off"] + [f"reset_{k}" for k in RESET_KINDS]
    if block == "context":
        per = np.clip(f["period"].to_numpy(int), 1, 5)
        P = _onehot(per, (2, 3, 4, 5))
        left = f["clock"].to_numpy(float)
        end = np.column_stack([(left <= 3.0), (left > 3.0) & (left <= 8.0)]).astype(float)
        m = f["margin"].to_numpy(float)
        M = _bands(m, (-1e9, -15.0, -6.0, 6.0, 15.0, 1e9))[:, [0, 1, 3, 4]]     # the close band is the reference
        gt = ((per == 4) & (np.abs(m) >= 15.0) & (left < 360.0)).astype(float)[:, None]
        return np.column_stack([P, end, M, gt]), ["q2", "q3", "q4", "ot", "end3", "end8", "m_le15", "m_le6", "m_ge6", "m_ge15", "garbage"]
    if block == "fatigue":
        s = np.clip(f["shooter_secs_on"].to_numpy(float), 0.0, 1500.0)
        s = np.where(np.isfinite(s), s, 300.0)
        S = natural_spline(s / 60.0, (1.0, 3.0, 6.0, 10.0, 16.0))[:, 1:]
        return S, [f"on{i}" for i in range(S.shape[1])]
    if block == "tracking":
        # the teacher's own inputs (tracked seasons only): how open, how much clock, off the catch or the bounce
        dd = np.log1p(np.clip(f["def_dist"].to_numpy(float), 0.0, 10.0))
        D = natural_spline(dd, (np.log1p(1.0), np.log1p(2.5), np.log1p(4.0), np.log1p(6.0), np.log1p(9.0)))[:, 1:]
        dist = np.clip(f["dist_xy"].to_numpy(float), 0.0, 40.0)
        DX = D[:, :1] * (dist[:, None] / 20.0)                        # the defender matters more further out
        tsc = f["true_sc"].to_numpy(float)
        off = np.isnan(tsc)
        S = natural_spline(np.where(off, 12.0, np.clip(tsc, 0.0, 24.0)), (2.0, 5.0, 9.0, 15.0, 21.0))[:, 1:]
        S[off] = 0.0
        full = (tsc >= 23.95).astype(float)                           # the full reset off the rim (tips)
        dr = f["dribbles"].to_numpy(float)
        DR = _bands(dr, (0, 1, 2, 3, 7, 1e9))[:, 1:]                 # 0 dribbles is the reference
        tt = np.clip(f["touch_time"].to_numpy(float), 0.0, 24.0)
        TT = natural_spline(tt, (0.5, 1.5, 3.0, 6.0))[:, 1:]
        cs = ((dr == 0) & (tt <= 2.0)).astype(float)
        cols = [D, DX, S, off[:, None].astype(float), full[:, None], DR, TT, cs[:, None]]
        names = ([f"def{i}" for i in range(D.shape[1])] + ["def_x_dist"] + [f"tsc{i}" for i in range(S.shape[1])]
                 + ["tsc_off", "tsc_full"] + ["drib1", "drib2", "drib3_6", "drib7p"]
                 + [f"touch{i}" for i in range(TT.shape[1])] + ["catch_shoot"])
        return np.column_stack(cols), names
    if block == "prev":
        pv = f["start_prev_value"].to_numpy(float)
        pd_ = f["start_prev_dist"].to_numpy(float)
        made = f["start_prev_made"].to_numpy(float)
        miss = (made == 0)
        cols = np.column_stack([miss & (pv == 3), miss & (pv == 2) & (pd_ >= 16), (f["start_prev_blocked"].to_numpy(float) == 1)])
        return cols.astype(float), ["after_miss3", "after_miss_long2", "after_block"]
    raise ValueError(block)


def design(f: pd.DataFrame, sub: str, blocks) -> tuple[np.ndarray, list]:
    parts, names = [np.ones((len(f), 1))], ["intercept"]
    for b in blocks:
        X, nm = block_columns(f, sub, b)
        parts.append(X)
        names += nm
    return np.column_stack(parts), names


# ------------------------------------------------------------------------------------------ fitting
@dataclass
class ShotModel:
    blocks: tuple
    coef: dict = field(default_factory=dict)          # sub-model -> (names, beta)
    offsets: dict = field(default_factory=dict)       # sub-model -> {"shooter": Series, "arena": Series}
    seasons: tuple = ()                                # the seasons it was trained on

    def check_may_price(self, seasons) -> None:
        """A model never prices a season it trained on, nor a season next to one (the leakage rule)."""
        seen = set(self.seasons)
        bad = sorted(int(s) for s in set(seasons) if {s - 1, s, s + 1} & seen)
        if bad:
            raise ValueError(f"this model trained on {sorted(seen)} and may not price {bad}")

    def predict_raw(self, f: pd.DataFrame, with_offsets: bool = False) -> np.ndarray:
        """The logit of a league-average shooter in a league-average arena (offsets zeroed), before relevel.

        `with_offsets` keeps the fitted shooter-season and arena-season offsets where the model has them: THIS
        shooter's make probability, which is what a teacher must hand a student that fits its own offsets."""
        self.check_may_price(np.unique(f["season"]))
        sub = submodel_of(f)
        eta = np.zeros(len(f))
        for s in SUBMODELS:
            m = sub == s
            if not m.any():
                continue
            X, _ = design(f[m], s, self.blocks)
            eta[m] = X @ self.coef[s][1]
            if with_offsets and s in self.offsets:
                g = f[m]
                ks = g["season"].astype(str) + ":" + g["shooter"].astype(str)
                ka = g["season"].astype(str) + ":" + g["arena"].astype(str)
                eta[m] += ks.map(self.offsets[s]["shooter"]).fillna(0.0).to_numpy()                     + ka.map(self.offsets[s]["arena"]).fillna(0.0).to_numpy()
        return eta


def _group_offsets(eta, y, w, codes, n_groups, lam, iters=2):
    """Ridge-penalised logit offsets per group, a few Newton steps from zero."""
    delta = np.zeros(n_groups)
    for _ in range(iters):
        mu = _sigmoid(eta + delta[codes])
        g = np.bincount(codes, weights=w * (y - mu), minlength=n_groups) - lam * delta
        h = np.bincount(codes, weights=w * mu * (1 - mu), minlength=n_groups) + lam
        delta += g / h
    return delta


def fit(train: pd.DataFrame, blocks, weights=None, ridge: float = 1.0, lam_shooter: float = 50.0,
        lam_arena: float = 50.0, rounds: int = 4, offsets: bool = True) -> ShotModel:
    """The three sub-models by backfitting: inputs given the offsets, then each offset given the rest."""
    model = ShotModel(blocks=tuple(blocks), seasons=tuple(sorted(int(s) for s in np.unique(train["season"]))))
    sub = submodel_of(train)
    w_all = np.ones(len(train)) if weights is None else np.asarray(weights, dtype=float)
    for s in SUBMODELS:
        m = sub == s
        f = train[m]
        y = f["made"].to_numpy(float)
        w = w_all[m]
        X, names = design(f, s, blocks)
        sh = pd.factorize(f["season"].astype(str) + ":" + f["shooter"].astype(str))
        ar = pd.factorize(f["season"].astype(str) + ":" + f["arena"].astype(str))
        d_sh = np.zeros(len(sh[1]))
        d_ar = np.zeros(len(ar[1]))
        beta = None
        for _ in range(rounds if offsets else 1):
            off = d_sh[sh[0]] + d_ar[ar[0]]
            beta = logistic_irls(X, y, w, ridge=ridge, offset=off, beta0=beta)
            if not offsets:
                break
            eta = X @ beta
            d_sh = _group_offsets(eta + d_ar[ar[0]], y, w, sh[0], len(sh[1]), lam_shooter)
            d_ar = _group_offsets(eta + d_sh[sh[0]], y, w, ar[0], len(ar[1]), lam_arena)
        model.coef[s] = (names, beta)
        model.offsets[s] = {"shooter": pd.Series(d_sh, index=sh[1]), "arena": pd.Series(d_ar, index=ar[1])}
    return model


def train_seasons(block, all_seasons, current: int = 2026) -> list:
    """The seasons a model for this block of rated seasons may train on: never the block, never the season on
    either side of it, never the current season."""
    lo, hi = min(block), max(block)
    return [s for s in all_seasons if (s < lo - 1 or s > hi + 1) and s != current]


def closeness_weights(seasons, block, half_life: float = 5.0) -> np.ndarray:
    lo, hi = min(block), max(block)
    s = np.asarray(seasons, dtype=float)
    dist = np.where(s < lo, lo - s, np.where(s > hi, s - hi, 0.0))
    return 0.5 ** (dist / half_life)


# ------------------------------------------------------------------------------------------ the level
def band_of(f: pd.DataFrame) -> np.ndarray:
    sub = submodel_of(f)
    d = f["dist_xy"].to_numpy(float)
    out = np.empty(len(f), dtype=object)
    for s in SUBMODELS:
        m = sub == s
        e = BANDS[s]
        idx = np.clip(np.searchsorted(np.asarray(e[1:-1]), d[m], side="right"), 0, len(e) - 2)
        out[m] = [f"{s}{i}" for i in idx]
    if "corner3" in f.columns:
        out[(sub == "three") & f["corner3"].to_numpy(bool)] = "three_corner"
    return out


UPDATE_BANDS = {"rim": (0.0, 2.0, 4.0), "mid": (4.0, 10.0, 16.0, 24.0), "three": (0.0, 25.0, 28.0, 99.0)}


def update_band_of(f: pd.DataFrame) -> np.ndarray:
    """The coarse shot classes the owner's update is sized on (scripts/119: v by class and clock band)."""
    sub = submodel_of(f)
    d = f["dist_xy"].to_numpy(float)
    out = np.empty(len(f), dtype=object)
    for s in SUBMODELS:
        m = sub == s
        e = UPDATE_BANDS[s]
        idx = np.clip(np.searchsorted(np.asarray(e[1:-1]), d[m], side="right"), 0, len(e) - 2)
        out[m] = [f"{s}{i}" for i in idx]
    if "corner3" in f.columns:
        out[(sub == "three") & f["corner3"].to_numpy(bool)] = "three_corner"
    return out


def relevel(eta: np.ndarray, f: pd.DataFrame, k: float = LEVEL_K) -> tuple[np.ndarray, pd.DataFrame]:
    """Shift each (season, half, band) cell's logits by the shift the OTHER half's makes want, padded toward 0.

    Regular-season rows in half A take the shift fitted on half B and the reverse; rows outside the halves
    (playoffs, or a missing half) take the shift fitted on the whole regular season of that season."""
    band = band_of(f)
    half = f["half"].astype(object).to_numpy()
    season = f["season"].to_numpy()
    phase = f["phase"].to_numpy() if "phase" in f.columns else np.full(len(f), "RS")
    y = f["made"].to_numpy(float)
    out = eta.copy()
    rows = []
    rs = phase == "RS"
    for s in np.unique(season):
        for b in np.unique(band[season == s]):
            cell = (season == s) & (band == b)
            shifts = {}
            for src in ("A", "B", "ALL"):
                m = cell & rs & ((half == src) if src != "ALL" else True)
                n = int(m.sum())
                if n == 0:
                    shifts[src] = 0.0
                    continue
                delta = 0.0
                for _ in range(6):
                    mu = _sigmoid(eta[m] + delta)
                    delta += float((y[m] - mu).sum() / max((mu * (1 - mu)).sum(), 1e-9))
                shifts[src] = delta * n / (n + k)
            for dst, src in (("A", "B"), ("B", "A")):
                m = cell & rs & (half == dst)
                out[m] = eta[m] + shifts[src]
            other = cell & ~(rs & np.isin(half, ["A", "B"]))
            out[other] = eta[other] + shifts["ALL"]
            rows.append(dict(season=int(s), band=b, shift_A=shifts["A"], shift_B=shifts["B"], shift_all=shifts["ALL"]))
    return out, pd.DataFrame(rows)


def probability(eta) -> np.ndarray:
    return _sigmoid(np.asarray(eta, dtype=float))


def update_after(q: np.ndarray, made: np.ndarray, v: np.ndarray, cap: float = 0.15) -> np.ndarray:
    """The owner's update: quality given the result, m + (y - m) * w with w = v / (m (1 - m)), capped.

    `v` is the variance of tracking quality among shots the play-by-play cannot tell apart (estimated from the
    tracked seasons, by shot class).  A made shot was probably more open than it looked, a miss less."""
    q = np.asarray(q, dtype=float)
    w = np.clip(np.asarray(v, dtype=float) / np.maximum(q * (1 - q), 1e-6), 0.0, cap)
    return q + (np.asarray(made, dtype=float) - q) * w
