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
ROLE_INPUTS = ["poss_pct", "gs_pct", "age"]
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

DERIVED_FEATURES = [*FULL_FEATURES, *DERIVED]
RATIO_FEATURES = [*DERIVED_FEATURES, *RATIOS]
SHOT_FEATURES = [*RATIO_FEATURES, *SHOTQ]
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


# Who he is (bio.py): height and weight BINNED, 2 inches and 15 pounds.  The fine pair names the player -- on
# the offensive line it buys 0.27 of the prior's own fit and the bins keep 0.03 of it (FINDINGS 26) -- so the
# binned form is the one that carries physiology and not identity.   name -> (base column, bin width)
BIO_BINS = {"height2": ("height", 2.0), "weight15": ("weight", 15.0)}


# Plus-minus as an input (FINDINGS 28; the owner: "like DRIP and DARKO do, but extremely smart about how").
# His own on-court record BEFORE the window: the possession-weighted APM (raw sign, per side) over the panel
# windows before this one, each discounted by PAST_DECAY per window of distance, the discounted possessions
# behind it in thousands (so the booster can weigh a 40,000-possession record against a 900-possession one),
# and the same on the ridge-shrunk RAPM_1.  Leak-free by construction: a pooled training row's target CONTAINS
# the past windows, so these are only allowed on PAIR rows (pair_rows), where the pair's target window is
# excluded from the past as well as the exclusion set; at prediction time the past is every panel window
# before the block, the block's own windows excluded (past_inputs).  A player with no past reads 0 / 0 / 0.
# the teammate-turnover feature of the pair rows (24.4): the share of the TARGET window's teammate-possessions spent
# with people he never shared 100 possessions with in the feature window; 1 = every teammate new, a stayer ~0.35
TURN_FEATURE = "target_pct_new_teammates"
_TM_TABLE: list = [None]          # the teammates table for context.DEST on pair rows (set by GBDTPrior from the Context)
PAST_OWN = ["past_apm", "past_poss", "past_rapm"]                 # his record on THIS prior's side
PAST_CROSS = ["past_apm_o", "past_poss_o", "past_apm_d", "past_poss_d"]   # both sides, named, for either prior
PAST = [*PAST_OWN, *PAST_CROSS]
# His LUCK-ADJUSTED ON-COURT ratings over the same past windows (the owner, 2026-09-09).  Not an APM: it
# does not separate him from his teammates, so it is biased toward whoever he played with and carries far
# less variance than `past_apm` -- which is exactly why it may be worth having beside it.  Needs a panel
# with `onc_o` / `onc_d` (scripts/49_role_panel.py, or scratch/add_onc_cols.py on an older one).
PAST_ONC = ["past_onc_o", "past_onc_d"]
PAST_ONC_CROSS = ["past_onc_o_o", "past_onc_d_o", "past_onc_o_d", "past_onc_d_d"]
PAST_DECAY = 0.5           # per 3-season window; `past_decay_for` rescales it to a panel's own window length


def past_decay_for(wins, decay: float | None = None) -> float:
    """The discount per window of distance for a panel whose windows are `wins`.  PAST_DECAY is one
    3-SEASON window, so a per-season panel discounts by its cube root and a player's record still reaches
    the same number of YEARS back; the configured windows return PAST_DECAY exactly."""
    if decay is not None:
        return float(decay)
    from .windows import label_step
    step = label_step(wins)
    return PAST_DECAY if step == 3.0 else float(PAST_DECAY ** (step / 3.0))


def past_features(p: pd.DataFrame, wins: list, keys: pd.DataFrame, exclude=(), decay: float = PAST_DECAY,
                  suffix: str = "") -> pd.DataFrame:
    """PAST for each row of `keys` (player_id, window[, window_to]) from one side's panel rows `p` (player_id,
    window, poss, apm, rapm1): the windows before `window` in the order `wins`, not in `exclude`, not
    `window_to`, discounted by decay ** distance.  `window` may be a label beyond the panel (the block) given as
    an index in `wins` via a `_wi` column instead.  `suffix` names the columns for a side ("_o" / "_d")."""
    idx = {lab: i for i, lab in enumerate(wins)}
    # `onc_o` / `onc_d` are his luck-adjusted on-court ratings (investigate.oncourt_rates), carried here
    # when the panel has them so PAST can offer the booster a BIASED, low-variance record of the same
    # player beside the unbiased noisy one (`apm`).  A panel built before they existed simply omits them.
    onc = [c for c in ("onc_o", "onc_d") if c in p.columns]
    q = p[~p.window.isin(set(exclude))][["player_id", "window", "poss", "apm", "rapm1"] + onc].copy()
    q["wi"] = q.window.map(idx).astype(float)
    n = len(keys)
    pid = keys.player_id.to_numpy()
    wi = keys["_wi"].to_numpy(dtype=float) if "_wi" in keys.columns else keys.window.map(idx).to_numpy(dtype=float)
    to = keys.window_to.map(idx).to_numpy(dtype=float) if "window_to" in keys.columns else np.full(n, -1.0)
    # every (key row, past window) pair at once: a merge on the player, then the window tests and the discount
    k = pd.DataFrame({"_i": np.arange(n), "player_id": pid, "_w": wi, "_to": to})
    m = k.merge(q, on="player_id", how="inner")
    m = m[(m.wi < m._w) & (m.wi != m._to)]
    wt = m.poss.to_numpy(dtype=float) * float(decay) ** (m._w.to_numpy() - m.wi.to_numpy())
    s = np.bincount(m._i.to_numpy(), weights=wt, minlength=n)
    sa = np.bincount(m._i.to_numpy(), weights=wt * m.apm.to_numpy(dtype=float), minlength=n)
    sr = np.bincount(m._i.to_numpy(), weights=wt * m.rapm1.to_numpy(dtype=float), minlength=n)
    ok = s > 0
    a, r = np.where(ok, sa / np.where(ok, s, 1.0), 0.0), np.where(ok, sr / np.where(ok, s, 1.0), 0.0)
    out = {f"past_apm{suffix}": a, f"past_poss{suffix}": s / 1000.0, f"past_rapm{suffix}": r}
    for c in onc:
        sc = np.bincount(m._i.to_numpy(), weights=wt * m[c].to_numpy(dtype=float), minlength=n)
        out[f"past_{c}{suffix}"] = np.where(ok, sc / np.where(ok, s, 1.0), 0.0)
    return pd.DataFrame(out, index=keys.index)


def past_all(panel: pd.DataFrame, side: str, wins: list, keys: pd.DataFrame, exclude=(),
             decay: float | None = None) -> pd.DataFrame:
    """Every PAST column for one prior's side: his own side's record unsuffixed, and both sides named.
    `decay` None = `past_decay_for(wins)`, the panel's own window length."""
    decay = past_decay_for(wins, decay)
    own = past_features(panel[panel.side == side], wins, keys, exclude=exclude, decay=decay)
    o = past_features(panel[panel.side == "O"], wins, keys, exclude=exclude, decay=decay, suffix="_o")
    d = past_features(panel[panel.side == "D"], wins, keys, exclude=exclude, decay=decay, suffix="_d")
    cols_o = [c for c in ("past_apm_o", "past_poss_o", "past_onc_o_o", "past_onc_d_o") if c in o.columns]
    cols_d = [c for c in ("past_apm_d", "past_poss_d", "past_onc_o_d", "past_onc_d_d") if c in d.columns]
    return pd.concat([own, o[cols_o], d[cols_d]], axis=1)


def past_inputs(panel: pd.DataFrame, side: str, exclude, player_ids, decay: float | None = None) -> pd.DataFrame:
    """PAST at prediction time, aligned to `player_ids`: every panel window before the excluded ones (the
    block's own), the excluded ones left out.  With nothing excluded every window is past."""
    wins = sorted(panel.window.unique())
    ex = set(exclude)
    first = min((i for i, w in enumerate(wins) if w in ex), default=len(wins))
    keys = pd.DataFrame({"player_id": np.asarray(player_ids), "_wi": float(first)})
    return past_all(panel, side, wins, keys, exclude=ex, decay=decay)


# How much he played, times what he did (the owner, 2026-09-10: "gs% * feature and poss played % x feature,
# for all available features").  Playing time is not one more box column: it says how much of what the box
# score shows is real.  Two blocks per 100 possessions in 200 possessions and two per 100 in 5,000 are the
# same NUMBER and nothing like the same evidence, and the only way a tree can say so is to split on the rate
# and then again on the exposure inside every leaf -- which costs depth the defensive booster does not have
# (depth 4, and `gs_pct` is one of only eleven names it carries).  The product hands it that directly.
# `gsx_<f>` = gs_pct * f, `ppx_<f>` = poss_pct * f.  Built on demand from the requested feature list, since
# the cross of two multipliers with everything a panel can make is a few hundred columns.
ROLE_MULTS = {"gsx": "gs_pct", "ppx": "poss_pct"}


def role_x(name: str):
    """"gsx_blk" -> ("gs_pct", "blk"); anything else -> None."""
    pre, _, base = str(name).partition("_")
    return (ROLE_MULTS[pre], base) if pre in ROLE_MULTS and base else None


def _past_based(name: str) -> bool:
    """True for an interaction whose BASE is a leak-carrying column (`ppx_past_apm`, `gsx_onc_o`, ...)."""
    from .context import DEST_ALL
    x = role_x(name)
    return bool(x) and (x[1] in PAST or x[1] in PAST_ONC or x[1] in PAST_ONC_CROSS or x[1] in DEST_ALL)


def role_x_needs(names) -> set:
    """The columns any `gsx_` / `ppx_` name in `names` needs on the frame before it can be built: its base
    and its multiplier.  A prediction frame is assembled from a want-list, so an interaction that does not
    put its own ingredients on that list is a column the training rows have and the prediction rows do not."""
    out: set = set()
    for f in names:
        x = role_x(f)
        if x:
            out |= {x[0], x[1]}
    return out


def role_interactions(names) -> list:
    """Every `gsx_` / `ppx_` name that can be formed from `names`, both multipliers, in a stable order."""
    base = [f for f in dict.fromkeys(names) if f not in ROLE_MULTS.values() and role_x(f) is None]
    return [f"{pre}_{f}" for pre in ROLE_MULTS for f in base]


def _wants_derived(feats) -> bool:
    return any(f in DERIVED or f in RATIOS or f in SHOTQ or f in BIO_BINS or role_x(f) for f in feats)


def add_derived(df: pd.DataFrame, feats=None) -> pd.DataFrame:
    """Add every `DERIVED`, `RATIOS`, `SHOTQ`, Dredge and `BIO_BINS` column the frame can make (in place; the
    rest are skipped), then the `gsx_` / `ppx_` interactions named in `feats`."""
    for name, (base, width) in BIO_BINS.items():
        if name not in df.columns and base in df.columns:
            df[name] = np.round(df[base].to_numpy(dtype=float) / width) * width
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
    add_shotq(df)
    # last, so an interaction may multiply a DERIVED, RATIOS or SHOTQ column as well as a raw one.  Built in
    # one assignment: there are 118 of them on the full sink and inserting those one at a time fragments the
    # frame badly enough that pandas warns about it.
    new = {}
    for name in (feats or ()):
        x = role_x(name)
        if x is None or name in df.columns or name in new:
            continue
        mult, base = x
        if mult in df.columns and base in df.columns:
            new[name] = df[mult].to_numpy(dtype=float) * df[base].to_numpy(dtype=float)
    return df if not new else pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1, copy=False)


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
    if any(f in PAST or _past_based(f) for f in feats):
        raise ValueError("past_* features leak into a POOLED target (it contains the past windows); train on pair "
                         "rows (GBDTPrior pairs=True, which any PAST feature switches on)")
    ex = set(exclude)
    p = panel[(panel.side == side) & ~panel.window.isin(ex)].copy()
    if _wants_derived(feats):
        p = add_derived(p, feats)
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


def pair_rows(panel: pd.DataFrame, side: str, exclude=(), features=None, target_col: str = "rapm1",
              poss_col: str = "poss", win_decay: float = 1.0, win_past: float = 1.0,
              turn: pd.DataFrame | None = None) -> pd.DataFrame:
    """`training_rows` un-pooled: one row per ORDERED pair of the player's windows (w -> w'), the feature line
    of w, the target of w', weight = the possessions behind that target times `win_decay` to the power of the
    distance (and `win_past` again when w' is earlier), and -- the reason to un-pool -- the teammate TURNOVER of
    w' with respect to w as the column `turn` (turnover.window_pair_turnover: player_id, window, window_to,
    turnover).  For a model without `turn` the weighted least-squares optimum is the pooled fit's exactly, since
    the within-player spread of the targets is a constant; with it the booster can learn what a box line is
    worth in a context that has changed.  Pairs the turnover table does not cover are dropped."""
    feats = list(DEFAULT_FEATURES if features is None else features)
    feats = [f for f in feats if f != TURN_FEATURE]
    from .context import DEST_ALL
    # the PAST family is BUILT below from other windows, not read off this row -- reading `onc_o` off
    # the row itself would be the target's own window and a leak.  A `gsx_` / `ppx_` interaction ON one of
    # them is built below too, in the second `add_derived` pass, and belongs in the same exemption: without
    # this the pair frame is asked for `gsx_past_apm` before `past_apm` exists.
    _built = set(PAST) | set(PAST_ONC) | set(PAST_ONC_CROSS) | set(DEST_ALL)
    _later = _built | {TURN_FEATURE}
    _deferred = [f for f in feats if (role_x(f) or ("", ""))[1] in _later]
    # ...and `ppx_past_apm` makes `past_apm` wanted even when the list does not name it on its own
    _need = dict.fromkeys([*feats, *role_x_needs(_deferred)])
    past = [f for f in _need if f in PAST or f in PAST_ONC or f in PAST_ONC_CROSS]
    dest = [f for f in _need if f in DEST_ALL]
    feats = [f for f in feats if f not in _built and f not in _deferred]
    # the pair frame carries only the selected columns, so a deferred interaction's MULTIPLIER has to be
    # carried along even when it is not a feature -- `ppx_past_apm` on a defensive list with no `poss_pct`
    # in it built fine on the panel (which has the column) and then not at all on the pairs, which did not
    helpers = [c for c in sorted(role_x_needs(_deferred) - set(feats) - _later) if c in panel.columns]
    ex = set(exclude)
    p = panel[(panel.side == side) & ~panel.window.isin(ex)].copy()
    if _wants_derived(feats):
        p = add_derived(p, feats)
    wins = sorted(panel.window.unique())
    idx = {lab: i for i, lab in enumerate(wins)}
    left = p[["player_id", "window", *feats, *helpers]]
    right = p[["player_id", "window", poss_col, target_col]].rename(
        columns={"window": "window_to", poss_col: "_poss_to", target_col: "target"})
    out = left.merge(right, on="player_id")
    out = out[out.window != out.window_to]
    d = out.window_to.map(idx).to_numpy() - out.window.map(idx).to_numpy()
    w = out["_poss_to"].to_numpy(dtype=float) * float(win_decay) ** np.abs(d)
    if float(win_past) != 1.0:
        w = w * np.where(d < 0, float(win_past), 1.0)
    out["weight"] = w
    if turn is not None:
        out = out.merge(turn[["player_id", "window", "window_to", "turnover"]].rename(columns={"turnover": TURN_FEATURE}),
                        on=["player_id", "window", "window_to"], how="inner")
        out = out[out[TURN_FEATURE].notna()]
    out = out[out.weight > 0].drop(columns="_poss_to").reset_index(drop=True)
    if past:
        # his record before w, the pair's target window w' left out of it as well as the exclusion set
        pf = past_all(panel, side, wins, out[["player_id", "window", "window_to"]], exclude=ex)
        for f in past:
            out[f] = pf[f].to_numpy()
    if dest:
        # the destination (context.py): the target window's teammates, each measured in the feature window
        if _TM_TABLE[0] is None:
            raise ValueError("DEST features need the teammates table (data/cache/teammates.parquet; GBDTPrior teammates=)")
        from .context import destination_features, usage_minutes_panel
        windows = {lab: list(range(int(lab[:4]), int(lab[5:9]) + 1)) for lab in wins}
        df = destination_features(usage_minutes_panel(panel), _TM_TABLE[0], windows, out[["player_id", "window", "window_to"]])
        for f in dest:
            out[f] = df[f].to_numpy()
    # again, now that the PAST / DEST / turnover columns exist: an interaction whose base is one of those
    # could not be built in the first pass, and a silently missing feature column is a KeyError at fit time
    # at best and a different model than the one asked for at worst
    out = add_derived(out, [f for f in (features or ()) if role_x(f)])
    return out.drop(columns=[c for c in helpers if c not in (features or ())], errors="ignore")


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


def _weighted_mean(rows: pd.DataFrame) -> float:
    return float(np.average(rows["target"], weights=rows["weight"])) if len(rows) else 0.0


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
                 win_past: float = 1.0, sat_poss: float | None = None, turn: pd.DataFrame | None = None,
                 pairs: bool = False, teammates: pd.DataFrame | None = None, folds: int = 0):
        g = cfg.get("gbdt", {})
        self.panel = panel
        self.cfg = cfg
        # `folds` > 0: player-grouped cross-fitting.  Players are split into that many groups by a stable rule
        # (their rank among the panel's ids), one model per (exclusion set, group) is trained WITHOUT that
        # group's rows, and a player is scored by the model that never saw a row of his.  So no feature can
        # identify him to a row of his own: the prior's fingerprint channel (scratch/foldtest.py: the shipped
        # defensive list loses 0.08 of 0.83 MSE with the player's rows out, a raw player id loses everything)
        # is closed, and his own history reaches the prior only through the explicit, time-ordered PAST block.
        self.folds = int(folds or 0)
        ids = np.sort(panel.player_id.unique())
        self._fold_of = pd.Series(np.arange(len(ids)) % max(self.folds, 1), index=ids) if self.folds else None
        # `turn`: the window-pair teammate turnover table.  With it the prior trains on PAIR rows (pair_rows)
        # with `turn` as a feature on both sides, so a prediction needs a `turn` column: 1.0 asks what the box
        # line is worth among strangers, a typical stayer's 0.35 what it is worth where he is.  `pairs` alone
        # trains on the pair rows WITHOUT the feature (the control for the un-pooling itself).
        self.turn = turn
        self.pairs = bool(pairs) or turn is not None
        if teammates is not None:
            _TM_TABLE[0] = teammates
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
            if self.turn is not None and TURN_FEATURE not in self.features[side]:
                self.features[side].append(TURN_FEATURE)
            from .context import DEST_ALL
            # a PAST or DEST feature is only leak-free on pair rows -- and so is `ppx_past_apm`, which is
            # one multiplied by a role input.  Without the second clause the interaction reads as an
            # ordinary name, the pooled path is taken, and `past_apm` is not on the panel to multiply.
            if any(x in PAST or x in DEST_ALL or _past_based(x) for x in self.features[side]):
                self.pairs = True
        self.params = dict(g.get("params", {}) or {})
        self.win_decay = float(win_decay)
        self.win_past = float(win_past)
        self.sat_poss = None if sat_poss is None else float(sat_poss)
        self._pool = dict(win_decay=self.win_decay, win_past=self.win_past, sat_poss=self.sat_poss)
        self._models: dict = {}
        self._ref = {side: (reference_mean(panel, side, target_col=self.target_col, **self._pool) if not self.pairs
                            else _weighted_mean(self.rows(side)))
                     for side in SIDES}
        self.reports: list = []

    def rows(self, side: str, exclude=()) -> pd.DataFrame:
        """This prior's training rows for one side and exclusion set (pooled, or pair rows, with `turn` if set)."""
        if not self.pairs:
            return training_rows(self.panel, side, exclude, self.features[side], target_col=self.target_col, **self._pool)
        return pair_rows(self.panel, side, exclude, self.features[side], target_col=self.target_col,
                         win_decay=self.win_decay, win_past=self.win_past, turn=self.turn)

    def model(self, side: str, exclude=(), fold: int | None = None):
        """The model for one side and exclusion set; with `fold` given, the one trained without that player
        group's rows (self.folds > 0).  Cached per (side, exclusion, fold)."""
        key = (side, frozenset(exclude), fold)
        if key not in self._models:
            rows = self.rows(side, exclude)
            if fold is not None:
                rows = rows[rows.player_id.map(self._fold_of).to_numpy() != int(fold)]
            rows, rep = counterbalance(rows, self._ref[side], self.tol)
            m = fit_gbdt(rows, self.features[side], seed=self.seed, thread_count=self.thread_count, **self.params)
            rep.update(mode=self.mode, side=side, exclude=",".join(sorted(exclude)), n_rows=int(len(rows)),
                       fold=-1 if fold is None else int(fold),
                       best_iteration=int(getattr(m, "best_iteration_", -1) or -1))
            self.reports.append(rep)
            self._models[key] = (m, rep)
        return self._models[key]

    def predict(self, side: str, X: pd.DataFrame, exclude=(), player_ids=None) -> np.ndarray:
        """Predictions for the rows of X.  With folds on, `player_ids` (one per row) routes each row to the
        model that never saw that player; a player the panel does not know goes to the all-rows model."""
        Xm = X[self.features[side]].to_numpy(dtype=float)
        if not self.folds or player_ids is None:
            m, _ = self.model(side, exclude)
            return np.asarray(m.predict(Xm), dtype=float)
        f = pd.Series(np.asarray(player_ids)).map(self._fold_of).to_numpy()
        out = np.zeros(len(Xm))
        unknown = pd.isna(f)
        if unknown.any():
            m, _ = self.model(side, exclude)
            out[unknown] = np.asarray(m.predict(Xm[unknown]), dtype=float)
        for k in np.unique(f[~unknown]).astype(int):
            sel = (~unknown) & (f == k)
            m, _ = self.model(side, exclude, fold=int(k))
            out[sel] = np.asarray(m.predict(Xm[sel]), dtype=float)
        return out


def gbdt_offset(prior: GBDTPrior, ro, rd, season, poss_o, poss_d, exclude=(), sides=SIDES, features=None,
                extra: pd.DataFrame | None = None, raw=None, shots: pd.DataFrame | None = None,
                player_ids=None) -> np.ndarray:
    """The (2m,) raw-sign GBDT offset from a window's centred rates and seasons (and, for mode "full", the role
    inputs in `extra`, aligned to ps_idx), possession-centred per side; zeros on a side not in `sides`.

    `shots` is `xshoot.player_shot_frame(train, cfg, ps_table.player_id)`: the training block's own shot totals
    and league levels, in ps_idx order, from which `add_shotq` rebuilds the SHOTQ features exactly as the panel
    build did for a training row."""
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
        ex = extra[side] if isinstance(extra, dict) else extra      # per-side inputs (PAST) or one table for both
        if ex is not None:
            need = set(prior.features[side]) | role_x_needs(prior.features[side])
            need |= {BIO_BINS[f][0] for f in need if f in BIO_BINS}
            for c in ex.columns:
                if c in need and c not in X.columns:
                    X[c] = np.asarray(ex[c], dtype=float)
        if shots is not None:
            for c in (*SHOT_TOTALS, *SHOT_LEAGUE):
                X[c] = np.asarray(shots[c], dtype=float)
        if _wants_derived(prior.features[side]):
            X = add_derived(X, prior.features[side])
        g = prior.predict(side, X, exclude, player_ids=player_ids)
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
