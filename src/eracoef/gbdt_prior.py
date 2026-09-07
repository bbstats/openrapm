"""The GBDT box prior: what a player's box line adds to his role level.

One chimeraboost model per side.  A training row is a player in one window W with W's 13 padded,
centred rates and the season he played most in W; its target is his RAPM_1 (the ridge pulled toward
the Simple SPM, spm.py) POOLED over his OTHER windows, possession-weighted, and its weight is that
pooled possession count.  So the model learns "a player whose box line looks like this in this era
is worth this much on court in the rest of his career", and a player seen in only one window is
scored but never trained on.

Cross-fitting is leave-window-out: the model used for a window (or for a holdout block) is fit
without every window in the exclusion set, and an excluded window's RAPM_1 never enters a pooled
target either.  The owner's point from the distributional-bias paper (RLOOCV): holding a block out
drags the training mean away from the full mean, in the direction opposite to the held-out block.
`drag` measures it per exclusion set and `counterbalance` rescales the weights of the training rows
on the drag's side so the weighted training mean equals the full-panel mean.  With the target centred
inside every window the drag is expected to be near zero; it is reported either way.

Feature selection: BorutaShap once per side on the pooled panel, with chimeraboost's own exact SHAP
values (BorutaShap's shap path calls shap.TreeExplainer, which does not know chimeraboost).
`ChimeraBorutaShap` overrides just that; the shadow-feature test and the binomial decision are the
library's.  The library also needs two compatibility shims (`np.NaN`, `scipy.stats.binom_test`).

Everything is in the model's raw sign; the offset is possession-centred per side before it is
handed to `plugin_fit(prior_offset=...)`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .design import FEATURES
from .roles import CAREER_INPUTS

SIDES = ("O", "D")
ROLE_INPUTS = ["share", "gs_pct", "age"]
DEFAULT_FEATURES = [*FEATURES, "season"]                 # mode "residual": what the box line adds to the role level
FULL_FEATURES = [*FEATURES, "season", *ROLE_INPUTS]       # mode "full": the GBDT is the whole prior

# Aggregations of the 13 rates, in the rates' own (centred, per-100) units.  An oblivious tree splits one
# column at a time, so a quantity that lives on a DIAGONAL of the rate space -- points, shot volume, the
# big-man axis -- costs it a staircase of splits to approximate and costs us nothing to hand over.  Every
# one is LINEAR in the rates, so it means the same thing centred as uncentred and no panel column is needed.
DERIVED = {
    "pts":      {"fg2m": 2.0, "fg3m": 3.0, "ftm": 1.0},                       # points per 100
    "fga":      {"fg2m": 1.0, "fg2_miss": 1.0, "fg3m": 1.0, "fg3_miss": 1.0},
    "fta":      {"ftm": 1.0, "ft_miss": 1.0},
    "fg3a":     {"fg3m": 1.0, "fg3_miss": 1.0},
    "usage":    {"fg2m": 1.0, "fg2_miss": 1.0, "fg3m": 1.0, "fg3_miss": 1.0,   # possessions he finishes
                 "ftm": 0.44, "ft_miss": 0.44, "tov": 1.0},
    "bigness":  {"orb": 1.0, "blk": 1.0, "drb": 0.3, "ast": -0.5, "fg3m": -0.4},   # 22_vs_consensus's definition
    "reb":      {"orb": 1.0, "drb": 1.0},
    "stocks":   {"stl": 1.0, "blk": 1.0},
    "creation": {"ast": 1.0, "tov": -1.0},
    "shotmix":  {"fg3m": 1.0, "fg3_miss": 1.0, "fg2m": -1.0, "fg2_miss": -1.0},    # threes minus twos taken
}
# Efficiency: a ratio of two rates, which an axis-aligned tree cannot make at all.  These need the rates'
# LEVEL back (a ratio of centred rates is meaningless), so they are built from the `raw_` columns -- the
# uncentred padded rates, stored in the panel by scratch/add_raw_rates.py and passed through by
# `chain_offset` at prediction time.  Each denominator is padded so a 200-possession player is not a ratio
# of two roundings.  (num, den, pad in attempts per 100.)
RATIOS = {
    "efg":   ({"fg2m": 1.0, "fg3m": 1.5}, {"fg2m": 1, "fg2_miss": 1, "fg3m": 1, "fg3_miss": 1}, 3.0, 0.50),
    "ts":    ({"fg2m": 2.0, "fg3m": 3.0, "ftm": 1.0},
              {"fg2m": 2, "fg2_miss": 2, "fg3m": 2, "fg3_miss": 2, "ftm": 0.88, "ft_miss": 0.88}, 6.0, 0.54),
    "p3r":   ({"fg3m": 1, "fg3_miss": 1}, {"fg2m": 1, "fg2_miss": 1, "fg3m": 1, "fg3_miss": 1}, 3.0, 0.25),
    "ftr":   ({"ftm": 1, "ft_miss": 1}, {"fg2m": 1, "fg2_miss": 1, "fg3m": 1, "fg3_miss": 1}, 3.0, 0.25),
    "fg3p":  ({"fg3m": 1}, {"fg3m": 1, "fg3_miss": 1}, 3.0, 0.34),
    "fg2p":  ({"fg2m": 1}, {"fg2m": 1, "fg2_miss": 1}, 3.0, 0.48),
    "ftp":   ({"ftm": 1}, {"ftm": 1, "ft_miss": 1}, 2.0, 0.75),
    "astr":  ({"ast": 1}, {"fg2m": 1, "fg2_miss": 1, "fg3m": 1, "fg3_miss": 1, "ftm": 0.44, "ft_miss": 0.44,
                           "tov": 1}, 3.0, 0.18),
    "tovr":  ({"tov": 1}, {"fg2m": 1, "fg2_miss": 1, "fg3m": 1, "fg3_miss": 1, "ftm": 0.44, "ft_miss": 0.44,
                           "tov": 1}, 3.0, 0.13),
    "orbsh": ({"orb": 1}, {"orb": 1, "drb": 1}, 3.0, 0.22),
}
# Shot quality: WHERE his attempts came from, which no box rate can say.  `data/stints/{season}_RS_shots.parquet`
# carries, per shooter per game, fg2a / fg2m / xl2 and fg3a / fg3m / xl3, where xl2 and xl3 are the LEAGUE's
# expected makes from HIS locations (shotcurve.py, and calibrated per season: sum(xl2) == sum(fg2m) to four
# figures).  Summed over a block (xshoot.player_shot_frame) they split what the rates give as one number:
#
#   difficulty    xl / a         the league make probability of his average attempt -- a rim-runner and a
#                                mid-range shooter at the same FG% sit at opposite ends of it
#   shot-making   (m - xl) / a   how far he beats a league shooter FROM HIS OWN SPOTS
#
# `fg2p` (the RATIOS entry) is the sum of the two; the pair is the decomposition, and the tree cannot make it
# from the rates because neither half is a function of makes and attempts alone.  `xps` and `mpts` are the same
# pair priced in points across both shot types, which is where the three-versus-rim trade-off lives.
#
# Every one is padded in ATTEMPTS toward the BLOCK's own league level (`shot_lg2` / `shot_lg3` / `shot_lgpps`,
# per-window columns in the panel, recomputed from the training block at prediction time), so a 40-attempt
# player is his era's average and not 1997's -- the league make rate on twos went 0.468 -> 0.550 over the 28
# seasons.  The shot-making constants are the reliability ones (FINDINGS 18: threes need ~450 attempts);
# difficulty is a near-deterministic property of a player's shot chart and barely needs padding at all.
SHOT_TOTALS = ["shot_fg2a", "shot_fg2m", "shot_xl2", "shot_fg3a", "shot_fg3m", "shot_xl3"]
SHOT_LEAGUE = ["shot_lg2", "shot_lg3", "shot_lgpps"]
SHOTQ_K = {"q2": 50.0, "q3": 50.0, "m2": 250.0, "m3": 450.0, "xps": 100.0, "mpts": 300.0}
SHOTQ = ["q2", "q3", "m2", "m3", "xps", "mpts"]

# Dredge: what the events behind the box line say, which the nightly summary threw away.  `dredge.py`
# counts them per (player, season) out of the play-by-play and `dredge.player_dredge_frame` sums a block's
# seasons into `dr_*` totals with the block's own league totals (`dr_lg_*`) beside them, exactly as
# `xshoot.player_shot_frame` does for shot quality.  Each feature is one counter over one denominator,
# padded toward the BLOCK's league level so a 300-possession player lands on his era's rate and not on the
# thirty-season average -- blocks per 100 possessions have fallen by a third since 1997 and `season` is a
# feature, so an unpadded rate would hand the tree the era twice.
#
# The pairs are the point.  Dredge's finding is that `blk` is the wrong SHAPE, not the wrong size: a block
# the defence recovers is worth twice a raw one, a rim block more again, and a blocked three "did not test
# well".  So each block counter appears both as a rate (how many) and as a SHARE of his blocks (what kind),
# and the share is the half the 13 rates cannot express at all.  Same for assisted versus unassisted makes
# and for stolen versus dropped turnovers.
#
#   name -> (numerator counter, denominator, padding constant)
# a denominator of poss_* is a possession count and the feature is per 100; anything else is a counter and
# the feature is a share.  `poss_all` (offense + defense) and `fgm_all` are made from the stored columns.
DREDGE_RATES = {
    "unast":   ("fgm_unast", "poss_off", 600.0),     # unassisted makes per 100:  Dredge's UnAstShot%
    "russ":    ("blk_rus", "poss_def", 900.0),       # Russells per 100:          his 0.445 against 0.236 for BLK
    "blkrim":  ("blk_rim", "poss_def", 900.0),       # rim blocks per 100:        0.523
    "loose":   ("foul_loose", "poss_all", 900.0),    # loose-ball fouls per 100:  ~0.33, a hustle proxy
    "techflg": ("foul_tech", "poss_all", 2500.0),    # technicals and flagrants:  +1.25, and the sign is his
    "stolen":  ("tov_stolen", "poss_off", 600.0),    # turnovers he had stolen:   ~2x as costly as the rest
    "offoul":  ("foul_off", "poss_off", 900.0),      # offensive fouls COMMITTED (the drawn ones need v2 pbp)
    "goalt":   ("goaltend", "poss_def", 4000.0),     # defensive goaltends: NOT in DREDGE by default (era trend)
}
DREDGE_SHARES = {
    "unastsh":  ("fgm_unast", "fgm_all", 90.0),
    "russsh":   ("blk_rus", "blk", 40.0),
    "blkrimsh": ("blk_rim", "blk", 40.0),
    "blk3sh":   ("blk_3", "blk", 40.0),
    "stolensh": ("tov_stolen", "tov_all", 60.0),
}
DREDGE_TOTALS: list = []          # filled from dredge.py below (import kept local: it reads the stints)
DREDGE_LEAGUE: list = []
DREDGE_ALL = [*DREDGE_RATES, *DREDGE_SHARES]
DREDGE_REL = [f"{n}_r" for n in DREDGE_ALL]        # the era-relative twin of each, built alongside it
# what a feature set gets by default: `goalt` is held out until its era trend is reconciled against a
# published source (dredge.py's module docstring, HANDOFF 3.1)
DREDGE = [f for f in DREDGE_ALL if f != "goalt"]
DREDGE_R = [f"{f}_r" for f in DREDGE]
DREDGE_ANY = [*DREDGE_ALL, *DREDGE_REL]            # every name add_dredge builds, for the "is it wanted" tests


def _dredge_cols():
    """The panel column names the block needs, imported lazily so `gbdt_prior` stays cheap to import."""
    global DREDGE_TOTALS, DREDGE_LEAGUE
    if not DREDGE_TOTALS:
        from .dredge import DREDGE_LEAGUE_COLS, DREDGE_TOTAL_COLS
        DREDGE_TOTALS, DREDGE_LEAGUE = list(DREDGE_TOTAL_COLS), list(DREDGE_LEAGUE_COLS)
    return DREDGE_TOTALS, DREDGE_LEAGUE


def add_dredge(df: pd.DataFrame) -> pd.DataFrame:
    """Add the `DREDGE_ALL` columns if the frame carries the counter totals and the block's league totals.

    Every one is `(num + k * league_ratio) / (den + k)`, times 100 when the denominator is possessions:
    the player's own rate shrunk toward the block's league rate with `k` denominator units of padding, the
    same shape `add_shotq` uses and the same shape `pad.shrink` uses for the 13 rates.
    """
    tot, lgc = _dredge_cols()
    if "russ" in df.columns or not all(c in df.columns for c in (*tot, *lgc)):     # already built, or cannot be
        return df
    col = {c[3:]: df[c].to_numpy(dtype=float) for c in tot}      # dr_blk    -> blk
    # per ROW, not per frame: the league columns are constant down a WINDOW and `training_rows` hands this
    # every window at once, so a scalar here pads 2026 toward 1997's league and hides the era in the feature
    lg = {c[6:]: df[c].to_numpy(dtype=float) for c in lgc}       # dr_lg_blk -> blk
    for d in (col, lg):
        d["poss_all"] = d["poss_off"] + d["poss_def"]
        d["fgm_all"] = d["fgm_unast"] + d["fgm_ast"]
    for name, (num, den, k) in ((*DREDGE_RATES.items(), *DREDGE_SHARES.items())):
        scale = 100.0 if den.startswith("poss") else 1.0
        level = lg[num] / np.maximum(lg[den], 1e-9)
        v = (col[num] + k * level) / (col[den] + k)
        df[name] = scale * v
        # and the same number as a multiple of the block's own league level: dimensionless, so a change in
        # how the feed RECORDS an event divides out and only the change in who does it survives
        df[f"{name}_r"] = v / np.maximum(level, 1e-12)
    return df


DERIVED_FEATURES = [*FULL_FEATURES, *DERIVED]
RATIO_FEATURES = [*DERIVED_FEATURES, *RATIOS]
SHOT_FEATURES = [*RATIO_FEATURES, *SHOTQ]
DREDGE_FEATURES = [*SHOT_FEATURES, *DREDGE]
# Experience (roles.career_inputs, stored in the panel and rebuilt from the training block at prediction
# time): seasons played, career possessions in thousands and the age he entered at, all counted BEFORE the
# block's first season.  Age is in the prior already and is not the same thing -- a 25-year-old rookie and a
# 25-year-old in year seven are different players, and the panel had no way to say which was which.
CAREER = list(CAREER_INPUTS)                          # defined in roles.py, where they are built
PRIOR_FEATURES = [*SHOT_FEATURES, *CAREER]           # everything: the accuracy-first prior of FINDINGS 22


def add_shotq(df: pd.DataFrame) -> pd.DataFrame:
    """Add the `SHOTQ` columns if the frame carries the shot totals and the block's league levels."""
    if "q2" in df.columns or not all(c in df.columns for c in (*SHOT_TOTALS, *SHOT_LEAGUE)):
        return df
    col = {c: df[c].to_numpy(dtype=float) for c in (*SHOT_TOTALS, *SHOT_LEAGUE)}
    a2, m2, x2 = col["shot_fg2a"], col["shot_fg2m"], col["shot_xl2"]
    a3, m3, x3 = col["shot_fg3a"], col["shot_fg3m"], col["shot_xl3"]
    lg2, lg3, lgp = col["shot_lg2"], col["shot_lg3"], col["shot_lgpps"]
    k, a = SHOTQ_K, a2 + a3
    df["q2"] = (x2 + k["q2"] * lg2) / (a2 + k["q2"])
    df["q3"] = (x3 + k["q3"] * lg3) / (a3 + k["q3"])
    df["m2"] = (m2 - x2) / (a2 + k["m2"])                           # padded toward 0: the league beats nobody
    df["m3"] = (m3 - x3) / (a3 + k["m3"])
    df["xps"] = (2.0 * x2 + 3.0 * x3 + k["xps"] * lgp) / (a + k["xps"])
    df["mpts"] = (2.0 * (m2 - x2) + 3.0 * (m3 - x3)) / (a + k["mpts"])
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add every `DERIVED`, `RATIOS` and `SHOTQ` column the frame can make (in place; the rest are skipped)."""
    for name, wts in DERIVED.items():
        if name in df.columns or not all(c in df.columns for c in wts):
            continue
        df[name] = sum(k * df[c].to_numpy(dtype=float) for c, k in wts.items())
    for name, (num, den, pad, target) in RATIOS.items():
        cols = set(num) | set(den)
        if name in df.columns or not all(f"raw_{c}" in df.columns for c in cols):
            continue
        n = sum(k * df[f"raw_{c}"].to_numpy(dtype=float) for c, k in num.items())
        d = sum(k * df[f"raw_{c}"].to_numpy(dtype=float) for c, k in den.items())
        df[name] = (n + pad * target) / (d + pad)
    return add_dredge(add_shotq(df))


# ------------------------------------------------------------------------------------ training rows
def training_rows(panel: pd.DataFrame, side: str, exclude=(), features=None, target_col: str = "rapm1",
                  poss_col: str = "poss", win_decay: float = 1.0, win_past: float = 1.0,
                  sat_poss: float | None = None) -> pd.DataFrame:
    """Rows of one side with every window in `exclude` removed from BOTH the rows and the pooled targets.

    target = possession-weighted mean of `target_col` over the player's other (non-excluded) windows,
    weight = the possessions behind that mean.  Rows without another window are dropped.

    `win_decay` < 1 weights a window by `win_decay ** |i - j|` in the pool, so "what he is worth in the rest
    of his career" becomes "what he is worth in the windows either side of this one".  1.0 (the default)
    weights every other window alike and is the original pooling exactly.  `win_past` multiplies the windows
    BEFORE this one on top of that, since aging is directional and the distance kernel is not.

    `sat_poss` (n0) turns the row weight from raw pooled possessions n into the inverse-variance weight
    n / (1 + n / n0): the pooled target's variance is sigma^2 / n + tau^2, so beyond n0 = sigma^2 / tau^2
    possessions more of them buy almost no precision and should not buy more say.  None = raw possessions.
    """
    feats = list(DEFAULT_FEATURES if features is None else features)
    ex = set(exclude)
    p = panel[(panel.side == side) & ~panel.window.isin(ex)].copy()
    if any(f in DERIVED or f in RATIOS or f in SHOTQ or f in DREDGE_ANY for f in feats):
        add_derived(p)
    w = p[poss_col].to_numpy(dtype=float)
    v = p[target_col].to_numpy(dtype=float)
    if float(win_decay) != 1.0 or float(win_past) != 1.0:
        other_w, other_wv = _pooled_by_distance(p, w, v, float(win_decay), float(win_past))
    else:
        p["_wv"] = w * v
        p["_w"] = w
        g = p.groupby("player_id")
        tot_wv = g["_wv"].transform("sum").to_numpy()
        tot_w = g["_w"].transform("sum").to_numpy()
        other_w = tot_w - w
        other_wv = tot_wv - w * v
    keep = other_w > 0
    out = p.loc[keep, ["player_id", "window", *feats]].copy()
    out["target"] = other_wv[keep] / other_w[keep]
    n = other_w[keep]
    out["weight"] = n if sat_poss is None else n / (1.0 + n / float(sat_poss))
    return out.reset_index(drop=True)


def _pooled_by_distance(p: pd.DataFrame, w: np.ndarray, v: np.ndarray, decay: float, past: float = 1.0):
    """The pooled weight and weighted sum over the player's OTHER windows, each discounted by `decay` to the
    power of how many windows away it is and, if `past` != 1, by `past` again when it is an EARLIER window.
    One (player, window) row per player per window, so the whole thing is two small matrix products against
    a 10 x 10 distance kernel."""
    wins = sorted(p.window.unique())
    wi = p.window.map({lab: i for i, lab in enumerate(wins)}).to_numpy()
    pid, pi = np.unique(p.player_id.to_numpy(), return_inverse=True)
    n_w = len(wins)
    W = np.zeros((len(pid), n_w))
    WV = np.zeros((len(pid), n_w))
    np.add.at(W, (pi, wi), w)
    np.add.at(WV, (pi, wi), w * v)
    d = np.arange(n_w)
    K = float(decay) ** np.abs(d[:, None] - d[None, :])
    if float(past) != 1.0:
        K = K * np.where(d[None, :] < d[:, None], float(past), 1.0)     # row i = the window being scored
    np.fill_diagonal(K, 0.0)                       # his own window never enters his own target
    return (W @ K)[pi, wi], (WV @ K)[pi, wi]


def reference_mean(panel: pd.DataFrame, side: str, **kw) -> float:
    """The full-panel weighted target mean the leave-window-out training sets are compared against."""
    r = training_rows(panel, side, exclude=(), **kw)
    return float(np.average(r["target"], weights=r["weight"])) if len(r) else 0.0


def drag(rows: pd.DataFrame, full_mean: float) -> float:
    """Weighted training-target mean minus the full-panel mean (the distributional bias of this exclusion set)."""
    if len(rows) == 0:
        return 0.0
    return float(np.average(rows["target"], weights=rows["weight"]) - full_mean)


def counterbalance(rows: pd.DataFrame, full_mean: float, tol: float = 0.02) -> tuple[pd.DataFrame, dict]:
    """If the drag exceeds `tol` (points per 100), rescale the weights of the rows on the drag's side of the
    full mean by the one factor that brings the weighted mean back to `full_mean`.  Returns (rows, report)."""
    d = drag(rows, full_mean)
    rep = dict(drag_before=d, drag_after=d, factor=1.0, n_side=0, applied=False)
    if len(rows) == 0 or abs(d) <= tol:
        return rows, rep
    t = rows["target"].to_numpy(dtype=float)
    w = rows["weight"].to_numpy(dtype=float)
    side = t > full_mean if d > 0 else t < full_mean
    S_side, W_side = float((w[side] * t[side]).sum()), float(w[side].sum())
    S_other, W_other = float((w[~side] * t[~side]).sum()), float(w[~side].sum())
    denom = S_side - full_mean * W_side
    f = (full_mean * W_other - S_other) / denom if denom != 0 else 1.0
    f = float(np.clip(f, 0.0, 1.0))
    out = rows.copy()
    out.loc[side, "weight"] = w[side] * f
    rep.update(drag_after=drag(out, full_mean), factor=f, n_side=int(side.sum()), applied=True)
    return out, rep


# ------------------------------------------------------------------------------------ the model
def fit_gbdt(rows: pd.DataFrame, features, seed: int = 0, thread_count=None, **params):
    """chimeraboost at its defaults (early stopping on), weighted by `rows.weight`, with the player as the
    group so his rows never straddle the early-stopping split."""
    from chimeraboost import ChimeraBoostRegressor

    kw = dict(random_state=int(seed))
    if thread_count:
        kw["thread_count"] = int(thread_count)
    kw.update(params)
    m = ChimeraBoostRegressor(**kw)
    X = rows[list(features)].to_numpy(dtype=float)
    y = rows["target"].to_numpy(dtype=float)
    w = rows["weight"].to_numpy(dtype=float)
    m.fit(X, y, sample_weight=w, groups=rows["player_id"].to_numpy())
    return m


class GBDTPrior:
    """Leave-window-out chimeraboost priors over the role panel, one per side, cached per exclusion set.

    Two modes, both tested on the criterion (the first build counted the role level twice by training on
    RAPM_1 and then adding the GBDT on top of the Simple SPM):
      mode "residual"  target `u` (RAPM_1 minus the role prior, pooled over the player's other windows),
                       features = rates + season; the offset is Simple SPM + this.
      mode "full"      target `rapm1` itself, features = rates + season + the role inputs; the offset is
                       this alone, the GBDT being the whole prior.
    """

    def __init__(self, panel: pd.DataFrame, cfg: dict, seed: int | None = None, thread_count=None, features=None,
                 mode: str | None = None, target_col: str | None = None, win_decay: float = 1.0,
                 win_past: float = 1.0, sat_poss: float | None = None):
        g = cfg.get("gbdt", {})
        self.panel = panel
        self.cfg = cfg
        self.mode = str(g.get("mode", "residual") if mode is None else mode)
        if self.mode not in ("residual", "full"):
            raise ValueError(f"gbdt mode must be 'residual' or 'full', got {self.mode!r}")
        self.target_col = target_col or ("u" if self.mode == "residual" else "rapm1")   # "apm": the unshrunk target
        self.seed = int(g.get("seed", 0) if seed is None else seed)
        self.thread_count = g.get("thread_count") if thread_count is None else thread_count
        self.tol = float(g.get("drag_tol", 0.02))
        self.features = {}
        default = DEFAULT_FEATURES if self.mode == "residual" else FULL_FEATURES
        key = "features_{}" if self.mode == "residual" else "features_full_{}"
        for side in SIDES:
            f = (features or {}).get(side) if isinstance(features, dict) else features
            f = f or g.get(key.format(side)) or default
            self.features[side] = list(f)
        self.params = dict(g.get("params", {}) or {})
        self.win_decay = float(win_decay)
        self.win_past = float(win_past)
        self.sat_poss = None if sat_poss is None else float(sat_poss)
        self._pool = dict(win_decay=self.win_decay, win_past=self.win_past, sat_poss=self.sat_poss)
        self._models: dict = {}
        self._ref = {side: reference_mean(panel, side, target_col=self.target_col, **self._pool)
                     for side in SIDES}
        self.reports: list = []

    def model(self, side: str, exclude=()):
        key = (side, frozenset(exclude))
        if key not in self._models:
            rows = training_rows(self.panel, side, exclude, self.features[side], target_col=self.target_col,
                                 **self._pool)
            rows, rep = counterbalance(rows, self._ref[side], self.tol)
            m = fit_gbdt(rows, self.features[side], seed=self.seed, thread_count=self.thread_count, **self.params)
            rep.update(mode=self.mode, side=side, exclude=",".join(sorted(exclude)), n_rows=int(len(rows)),
                       best_iteration=int(getattr(m, "best_iteration_", -1) or -1))
            self.reports.append(rep)
            self._models[key] = (m, rep)
        return self._models[key]

    def predict(self, side: str, X: pd.DataFrame, exclude=()) -> np.ndarray:
        m, _ = self.model(side, exclude)
        return np.asarray(m.predict(X[self.features[side]].to_numpy(dtype=float)), dtype=float)


def gbdt_offset(prior: GBDTPrior, ro, rd, season, poss_o, poss_d, exclude=(), sides=SIDES, features=None,
                extra: pd.DataFrame | None = None, raw=None, shots: pd.DataFrame | None = None,
                dredge: pd.DataFrame | None = None) -> np.ndarray:
    """The (2m,) raw-sign GBDT offset from a window's centred rates and seasons (and, for mode "full", the role
    inputs in `extra`, aligned to ps_idx), possession-centred per side; zeros on a side not in `sides`.

    `shots` is `xshoot.player_shot_frame(train, cfg, ps_table.player_id)`: the training block's own shot totals
    and league levels, in ps_idx order, from which `add_shotq` rebuilds the SHOTQ features exactly as the panel
    build did for a training row.  `dredge` is `dredge.player_dredge_frame(train, cfg, ...)` and does
    the same for the Dredge counters."""
    feats = list(FEATURES if features is None else features)
    m = len(season)
    out = np.zeros(2 * m)
    for j, (side, R, poss, off) in enumerate((("O", ro, poss_o, 0), ("D", rd, poss_d, m))):
        if side not in sides:
            continue
        X = pd.DataFrame(np.asarray(R, dtype=float), columns=feats)
        X["season"] = np.asarray(season, dtype=float)
        if raw is not None:                              # the uncentred rates, for the efficiency ratios
            for c, col in zip(feats, np.asarray(raw[j], dtype=float).T):
                X[f"raw_{c}"] = col
        if extra is not None:
            for c in extra.columns:
                if c in prior.features[side] and c not in X.columns:
                    X[c] = np.asarray(extra[c], dtype=float)
        if shots is not None:
            for c in (*SHOT_TOTALS, *SHOT_LEAGUE):
                X[c] = np.asarray(shots[c], dtype=float)
        if dredge is not None:
            for c in (*_dredge_cols()[0], *_dredge_cols()[1]):
                X[c] = np.asarray(dredge[c], dtype=float)
        if any(f in DERIVED or f in RATIOS or f in SHOTQ or f in DREDGE_ANY for f in prior.features[side]):
            add_derived(X)
        g = prior.predict(side, X, exclude)
        w = np.maximum(np.asarray(poss, dtype=float), 0.0)
        if w.sum() > 0:
            g = g - np.average(g, weights=w)
        out[off:off + m] = g
    return out


def partial_dependence(model, X: pd.DataFrame, features, feature: str, grid, seasons=None) -> pd.DataFrame:
    """Mean prediction with `feature` set to each grid value (and `season` to each of `seasons`), the rest of
    the rows as they are.  chimeraboost has no PD helper; this is the plain definition."""
    feats = list(features)
    rows = []
    base = X[feats].to_numpy(dtype=float)
    j = feats.index(feature)
    js = feats.index("season") if "season" in feats else None
    for s in ([None] if (seasons is None or js is None) else seasons):
        Xs = base.copy()
        if s is not None:
            Xs[:, js] = float(s)
        for v in grid:
            Xv = Xs.copy()
            Xv[:, j] = float(v)
            rows.append(dict(feature=feature, season=s, value=float(v), pd=float(np.mean(model.predict(Xv)))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ Boruta on chimeraboost
def _import_borutashap():
    """BorutaShap 1.0.17 against numpy 2 / scipy 1.16: two names it imports were removed."""
    import scipy.stats as st
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    if not hasattr(st, "binom_test"):
        def binom_test(x, n=None, p=0.5, alternative="two-sided"):
            return st.binomtest(int(x), int(n), p, alternative=alternative).pvalue
        st.binom_test = binom_test
    import BorutaShap
    return BorutaShap


def make_boruta(model, importance_measure: str = "shap"):
    """A BorutaShap whose SHAP importances come from chimeraboost's exact `shap_values`."""
    BS = _import_borutashap()

    class ChimeraBorutaShap(BS.BorutaShap):
        def check_model(self):          # chimeraboost exposes feature_importances_ only once fitted
            pass

        def explain(self):
            X = self.X_boruta
            sv = np.asarray(self.model.shap_values(X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else X))
            if sv.ndim == 3:
                sv = np.abs(sv).sum(axis=0)
            self.shap_values = np.abs(sv).mean(0)

    return ChimeraBorutaShap(model=model, importance_measure=importance_measure, classification=False)


def run_boruta(rows: pd.DataFrame, features, n_trials: int = 50, seed: int = 0, thread_count=None,
               verbose: bool = False, **params) -> dict:
    """BorutaShap on the pooled training rows of one side; returns accepted / tentative / rejected and the history."""
    from chimeraboost import ChimeraBoostRegressor

    kw = dict(random_state=int(seed))
    if thread_count:
        kw["thread_count"] = int(thread_count)
    kw.update(params)
    fs = make_boruta(ChimeraBoostRegressor(**kw))
    X = rows[list(features)].reset_index(drop=True)
    y = pd.Series(rows["target"].to_numpy(dtype=float))
    w = pd.Series(rows["weight"].to_numpy(dtype=float))
    fs.fit(X=X, y=y, sample_weight=w, n_trials=int(n_trials), random_state=int(seed), sample=False,
           train_or_test="test", normalize=True, verbose=verbose)
    hist = getattr(fs, "history_x", None)
    return dict(accepted=sorted(fs.accepted), tentative=sorted(fs.tentative), rejected=sorted(fs.rejected),
                history=hist.copy() if hist is not None else None)
