"""The single-year prior: which features it sees, and how a player's other seasons are folded into one row.

The pipeline is TARGET -> PRIOR -> RATING (scripts/62_single_year_board.py).  This module owns the middle
stage's inputs, because they were copied into three places -- the notebook builder, the board script and the
end-to-end sweep -- and drifted from what the rest of the project had already learned.

**The features.**  `gbdt_prior.SHOT_FEATURES` (the 13 padded rates, the role inputs, the ten linear
`DERIVED` combinations, the ten `RATIOS` and the six `SHOTQ` shot-quality columns) plus career, bio and the
four on-court columns.  The hand-picked list this replaces had the 13 rates and the raw shot totals and
none of the rest -- which threw away exactly the features an axis-aligned tree cannot rebuild for itself.
A tree splits one column at a time, so it can approximate `pts` with a staircase of splits but it can never
form `ts` or `efg` at all: a ratio of two columns is not a function of either one.  Boruta's accepted
offensive list is full of them.

`season` is dropped.  It is in `SHOT_FEATURES` because the shipped prior trains on player-WINDOWS, where it
names the era.  Here a training row is a player pooled over every season but the held-out one, so his
"season" is an average of twelve of them and means nothing.

**The aggregation order.**  Average the INPUTS over the player's seasons, then build the derived columns on
the average -- not the other way round.  A possession-weighted mean of `ts` is not the `ts` of the mean
(measured on this panel: correlation 0.9996, mean gap 0.0009).  Small, and free to get right.

Everything is averaged by possessions, the shot TOTALS included.  Summing them would be the natural reading
of "his career", but the `SHOTQ` columns pad in attempts (`SHOTQ_K`: 50 for difficulty, 250 and 450 for
shot-making), so a summed twelve-season row is barely padded while the single season it has to predict is
padded hard -- the same column would mean two different things on the two sides of the model.  A
possession-weighted mean makes a training row read as "his typical season", on the prediction row's scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gbdt_prior import CAREER, SHOT_FEATURES, SHOT_LEAGUE, SHOT_TOTALS, add_derived
from .design import FEATURES

__all__ = ["PRIOR_FEATURES", "INPUT_COLUMNS", "BIO", "ONC", "ROLE_INPUTS", "aggregate", "season_frame",
           "prior_rows", "season_rows", "FEATURE_SETS", "feature_set", "OFFENSE_TARGET", "DEFENSE_TARGET",
           "RAPM_OFFENSE_LAMBDA",
           "RAPM_DEFENSE_LAMBDA", "RAPM_CONTEXT_LAMBDA", "MIN_POSSESSIONS"]

# The pipeline's settings, here rather than in the board script so the Boruta run selects features
# against the same target the board will train on.  What each side EXPLAINS: `xpts_ft` replaces made
# free throws by the shooter's padded expectation (+0.26 per 100 against raw points), and
# `x3def_w0.25` additionally replaces three quarters of every opponent three-point make by that
# shooter's own padded 3P% (-0.39 to -0.47 per 100, z -4.4 to -5.1).  DECISIONS.md 37-52.
OFFENSE_TARGET = "xpts_ft"
DEFENSE_TARGET = "x3def_w0.25"

# The leave-one-season-out RAPM's three penalties, swept on the end-to-end score (DECISIONS.md), and the
# possessions a player needs to be in that target at all.
RAPM_OFFENSE_LAMBDA = 40000.0
RAPM_DEFENSE_LAMBDA = 40000.0
RAPM_CONTEXT_LAMBDA = 0.0
MIN_POSSESSIONS = 100.0

# `bio.PLAYER_INPUTS`, repeated rather than imported: `bio` is a data-layer module (it loads the bio
# table) and this one may not import it.  `tests/test_singleyear.py` asserts the two stay equal.
BIO = ["height", "weight", "draft_pick", "tenure", "n_teams"]
ONC = ["onc_o", "onc_d", "onc_poss_o", "onc_poss_d"]        # his own season's on-court record, per side
ROLE_INPUTS = ["poss_pct", "gs_pct", "age"]

# What the booster sees.  54 names: 42 of SHOT_FEATURES (all but `season`), 3 career, 5 bio, 4 on-court.
PRIOR_FEATURES = [f for f in SHOT_FEATURES if f != "season"] + CAREER + BIO + ONC

# What the panel must carry for `add_derived` to build the rest.  The raw (uncentred) rates are the
# RATIOS' inputs; the shot totals and the block league levels are the SHOTQ inputs.
INPUT_COLUMNS = (list(FEATURES) + ROLE_INPUTS + [f"raw_{c}" for c in FEATURES]
                 + list(SHOT_TOTALS) + list(SHOT_LEAGUE) + CAREER + BIO + ONC)


# ---------------------------------------------------------------------------------- named feature sets
# `scripts/50_boruta.py --modes=single_year` selects against this pipeline's own target.  BORUTA PRUNES,
# IT DOES NOT DECIDE: an acceptance means "not noise against the offline target", never "this belongs on
# the board" (HANDOFF 3.1).  Accepted + tentative is what is kept -- a tentative name is one the trials
# ran out of evidence on, not one they rejected.
BORUTA_O = ["ast", "creation", "efg", "exp_poss", "exp_yrs", "fg3p", "fga", "fta", "onc_d", "onc_o",
            "onc_poss_d", "onc_poss_o", "orb", "orbsh", "p3r", "poss_pct", "pts", "stl", "tenure", "ts",
            "weight"]
BORUTA_D = ["blk", "drb", "exp_poss", "exp_yrs", "fga", "gs_pct", "height", "onc_d", "onc_o",
            "onc_poss_d", "poss_pct", "pts", "stl", "stocks", "ts", "usage", "weight"]

# `onc_*` STAYS IN, and the reasoning is worth the paragraph because it went the other way first.
#
# The four columns are the player's own on-court points per 100 over the panel row.  Both Boruta runs rank
# them first by a factor of six -- `onc_o` 7.26 against 1.18 for the next name on offence, `onc_d` 6.61 on
# defence -- and dropping them costs 0.169 game ARMSE on the 75/25 diagnostic.
#
# That diagnostic is not trustworthy here.  `outputs/role_panel_season.parquet` is built from each season's
# FULL design, so season H's `onc_o` is averaged over every game of H, INCLUDING the quarter the diagnostic
# scores.  Rebuild it from the first 75% and change nothing else (`scratch/onc_leak.py`; the whole-season
# rebuild arm reproduces the panel to 0.0000) and the board goes 8.8503 -> 9.2249, +0.375 per 100 at z
# +7.47, 4 of 4 seasons.  **So the 75/25 number cannot compare two boards that differ in `onc_*`.**  It can
# still compare boards that hold them fixed, which is every other comparison in this module.
#
# It does NOT follow that the columns should go, and that was the wrong turn.  The leak is in the
# EVALUATION, not in the product.  The board rates a COMPLETED season, ruling 12 allows H's own games as
# the evidence, and at ship time H's on-court record is legitimately known -- as it is to every public
# metric in `data/external/consensus.csv`, all of which use the season's own plus-minus.  "Would `onc_*`
# help if we only had 75% of the season" is a question the product never asks.
#
# With the criterion silent, the external consensus decides, as a sanity check and not a fitting target:
# with `onc_*` the board reads rho 0.753 / 0.811 / 0.766 (offence / defence / total) and clears nine of
# ten floors; without, 0.721 / 0.667 / 0.663 and it fails all three agreement floors, defence by 11%.
# That is a gross miss, which the standing rule does treat as a veto.
#
# The cost is real and is recorded here rather than hidden: `onc_*` makes the PRIOR and the EVIDENCE the
# same games, so the ridge is no longer combining two independent sources.  It shows up as the plus-minus
# stage doing less -- the season's own games move the board by sd 0.19 on offence with them and 0.32
# without.  `boruta_noonc` is kept so the comparison can be re-run, and an honest 75/25 diagnostic would
# need a panel rebuilt from a 75% design, which is not built.

FEATURE_SETS = {
    "sy": {"O": PRIOR_FEATURES, "D": PRIOR_FEATURES},
    "sy_noonc": {side: [f for f in PRIOR_FEATURES if f not in ONC] for side in ("O", "D")},
    "boruta": {"O": BORUTA_O, "D": BORUTA_D or PRIOR_FEATURES},
    "boruta_noonc": {"O": [f for f in BORUTA_O if f not in ONC],
                     "D": [f for f in (BORUTA_D or PRIOR_FEATURES) if f not in ONC]},
}


def feature_set(name: str) -> dict:
    """{"O": [...], "D": [...]} for a named set; raises on an unknown name rather than guessing."""
    if name not in FEATURE_SETS:
        raise KeyError(f"unknown feature set {name!r}; have {sorted(FEATURE_SETS)}")
    return {side: list(v) for side, v in FEATURE_SETS[name].items()}


def _check(frame: pd.DataFrame, features) -> pd.DataFrame:
    """`add_derived` skips, silently, any family whose inputs are absent.  So ask whether it built them.

    The same guard `scripts/49_role_panel.py` puts on the on-court columns: a missing feature here would
    not raise, it would train a prior on a quietly shorter list.
    """
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise KeyError(f"add_derived did not build {missing} -- the panel is missing their inputs. "
                       f"Needed: {[c for c in INPUT_COLUMNS if c not in frame.columns]}")
    return frame


def aggregate(rows: pd.DataFrame, features=None) -> pd.DataFrame:
    """One row per player: his `INPUT_COLUMNS` averaged over `rows` by possessions, then derived.

    `rows` is already the side and the seasons the caller wants (typically every season but the held-out
    one, one side).  Indexed by `player_id`, so it joins straight onto a target frame.
    """
    features = list(PRIOR_FEATURES if features is None else features)
    cols = [c for c in INPUT_COLUMNS if c in rows.columns]
    weight = rows.groupby("player_id").poss.sum()
    mean = rows[cols].mul(rows.poss, axis=0).groupby(rows.player_id).sum().div(weight, axis=0)
    return _check(add_derived(mean, features), features)


def season_frame(rows: pd.DataFrame, features=None) -> pd.DataFrame:
    """The prediction side: one season's panel rows with the derived columns built on them as they are."""
    features = list(PRIOR_FEATURES if features is None else features)
    return _check(add_derived(rows.copy(), features), features)


def prior_rows(target: pd.DataFrame, rows: pd.DataFrame, column: str, features=None) -> pd.DataFrame:
    """Training rows for the single-year prior: aggregated features joined to an EXTERNAL target.

    `gbdt_prior.training_rows` manufactures its target by pooling the player's other windows.  That is
    exactly what `LeaveSeasonOutRAPM` has already done here, so pooling again would count it twice --
    the target IS the leave-one-out quantity.  This joins it instead, on `player_id`, and takes the
    weight from the possessions behind it.
    """
    features = list(PRIOR_FEATURES if features is None else features)
    out = (aggregate(rows, features)
           .join(target.set_index("player_id")[[column, "possessions"]], how="inner").dropna())
    return out.assign(target=out[column].to_numpy(float), weight=out.possessions.to_numpy(float))


def season_rows(labels: dict, rows: pd.DataFrame, column: str, features=None,
                cap_per_player: bool = False) -> pd.DataFrame:
    """Training rows, one per PLAYER-SEASON: a season's own panel row, labelled from his OTHER seasons.

    `labels[s]` is the target frame (`player_id`, `offense`, `defense`, `possessions`) fit WITHOUT season
    s -- and without the rated season, which the caller already left out of `rows` -- so a row's box score
    and its label never share a game.  The weight is the possessions behind the label, as in `prior_rows`.

    `cap_per_player=True` divides each row's weight by the number of rows the player has, so his rows
    together weigh what his one row weighed under `prior_rows`.  Without it a fifteen-season player has
    fifteen rows each carrying his whole career's possessions, and the top tenth of players hold 53% of
    the training weight against 43% under one row per player (measured, 2026-09-13, rated season 2024).
    The rows of one player are not independent, so that is not more evidence, just more weight.

    Why one row per player-season and not one per player (`prior_rows`, 2026-09-13): at inference the
    prior is handed ONE season's box score, and a booster trained on career averages has learned the map
    at the clean end and never seen how it degrades with noise -- on the year-over-year test the
    one-row-per-player prior was too wide for the neighbouring season in 28 of 28 seasons.  Here a
    training row is the same kind of row as the inference row.  The rows of one player are not
    independent, so the effective sample is still the number of players; what the extra rows carry is the
    noise level, not new players.
    """
    features = list(PRIOR_FEATURES if features is None else features)
    parts = []
    for s, target in labels.items():
        own = rows[rows.season == s]
        if own.empty:
            continue
        frame = season_frame(own, features).set_index("player_id")
        lab = target.set_index("player_id")[[column, "possessions"]]
        parts.append(frame.join(lab, how="inner").dropna(subset=features + [column]))
    out = pd.concat(parts)
    weight = out.possessions.to_numpy(float)
    if cap_per_player:
        weight = weight / out.groupby(level=0).possessions.transform("count").to_numpy(float)
    return out.assign(target=out[column].to_numpy(float), weight=weight)
