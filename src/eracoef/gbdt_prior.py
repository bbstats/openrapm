"""The box-score feature definitions the single-year prior is built from (singleyear.py).

`DERIVED` are linear aggregations of the 13 padded rates, `RATIOS` efficiency ratios of the uncentred rates,
`SHOTQ` shot difficulty and shot-making from each shooter's own locations, `BIO_BINS` binned height and
weight, and `add_derived` adds every one of them a frame can make.  The constants' names and term order are
load-bearing: the shipped feature lists and the booster read them by name and position.
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


# How much he played, times what he did (the owner, 2026-09-10: "gs% * feature and poss played % x feature,
# for all available features").  Playing time is not one more box column: it says how much of what the box
# score shows is real.  Two blocks per 100 possessions in 200 possessions and two per 100 in 5,000 are the
# same NUMBER and nothing like the same evidence, and the only way a tree can say so is to split on the rate
# and then again on the exposure inside every leaf -- which costs depth the defensive booster does not have
# (depth 4, and `gs_pct` is one of only eleven names it carries).  The product hands it that directly.
# `gs_pct_x_<f>` = gs_pct * f, `poss_pct_x_<f>` = poss_pct * f.  Built on demand from the requested feature list, since
# the cross of two multipliers with everything a panel can make is a few hundred columns.
PLAYING_TIME = ["gs_pct", "poss_pct"]        # the two the owner named: games started %, possessions played %


def interaction_parts(name: str):
    """"gs_pct_x_blk" -> ("gs_pct", "blk"); anything that is not an interaction feature -> None."""
    for pt in PLAYING_TIME:
        head = f"{pt}_x_"
        if str(name).startswith(head) and len(str(name)) > len(head):
            return (pt, str(name)[len(head):])
    return None


def add_derived(df: pd.DataFrame, feats=None) -> pd.DataFrame:
    """Add every `DERIVED`, `RATIOS`, `SHOTQ` and `BIO_BINS` column the frame can make (in place; the
    rest are skipped), then the `gs_pct_x_` / `poss_pct_x_` interactions named in `feats`."""
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
        x = interaction_parts(name)
        if x is None or name in df.columns or name in new:
            continue
        mult, base = x
        if mult in df.columns and base in df.columns:
            new[name] = df[mult].to_numpy(dtype=float) * df[base].to_numpy(dtype=float)
    return df if not new else pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1, copy=False)
