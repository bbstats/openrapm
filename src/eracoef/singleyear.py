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
from .rloocv import BalancedGroupKFold

__all__ = ["PRIOR_FEATURES", "INPUT_COLUMNS", "BIO", "ONC", "OFFC", "NET", "SHOT_MIX", "ROLE_INPUTS", "CLOSENESS", "LEVEL_COVARIATES", "aggregate", "season_frame",
           "prior_rows", "season_rows", "chunk_rows", "chunk_season_sets", "CHUNK_FEATURES", "stratified_player_folds",
           "fold_mean_shift", "FEATURE_SETS", "feature_set", "OFFENSE_TARGET", "DEFENSE_TARGET",
           "team_movement", "reweight_by_movement", "CAREER_BANDS", "PIECE_NAMES", "PIECES", "PO_SHARE", "SAME_TEAM",
           "FRANCHISE_MOVES", "franchise", "harmonic_overlap", "TEAM_SEASON", "AgingCurve", "adjacent_rows",
           "adjacent_season_sets", "RAPM_OFFENSE_LAMBDA",
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
# his team's record WITHOUT him, same games, same centring and padding (`investigate.offcourt_rates`),
# and the on/off net.  The owner, 2026-09-14: "usually we also include off-court-rating".
OFFC = ["offc_o", "offc_d", "offc_poss_o", "offc_poss_d"]
NET = ["net_o", "net_d"]                                    # onc minus offc, per side
# his teammates' shot locations with him on the floor against without him, padded (scripts/100_shot_mix_panel.py)
SHOT_MIX = ["mix_lift"]
ROLE_INPUTS = ["poss_pct", "gs_pct", "age"]

# What score state a player's statistics were compiled in (scripts/69_closeness_panel.py).  `closeness` is
# the possession-weighted mean of 1 / max(|margin|, 1) -- the same form `priorridge.team_game_weights`
# uses on the ratings objective, so "close" means one thing in both halves of the pipeline.
CLOSENESS = ["gt_share", "closeness", "abs_margin"]

# The pieces of his season's vanilla RAPM, per side (experiment 26, the owner 2026-09-28): the Decomposition
# page's two splits -- by player (on_rtg, teammates, opponents, context, ridge) and by possession (on_signal,
# off_adj_gp, off_adj_dnp, team_sos) -- plus the actual off-court rating, `_o` from a fit on the offensive
# design and `_d` from one on the defensive design, each padded toward 0 over its possessions.  Positive =
# good on both sides.  scripts/86_context_panel.py writes them; `pieces.COLUMNS` is the same list, repeated
# here because `pieces` imports this module (tests/test_singleyear.py asserts the two stay equal).
PIECE_NAMES = ["on_rtg", "teammates", "opponents", "context", "ridge",
               "on_signal", "off_adj_gp", "off_adj_dnp", "team_sos", "off_rtg"]
PIECES = [f"pc_{k}_{tag}" for tag in ("o", "d") for k in PIECE_NAMES]
# the share of his possessions that came in the playoffs (86_context_panel.py): a share, not padded
PO_SHARE = ["po_share"]
# How much of a training row's team context its label shares (`chunk_rows(shares=...)`): 1 on every career
# row, 0 for a chunk on a team he never played for otherwise, and 0 for everyone at rating time -- "rate him
# as if he changed teams".  Not a panel column: it belongs to a (row, label) pair, so the caller appends it
# to the model's inputs the way `CHUNK_FEATURES` are.
SAME_TEAM = "same_team"
# Each training row's main team-season (experiment 27): the (franchise, season) holding most of the row's
# possessions, as season * 10**10 + team id, or -1 where the team table has none.  The GROUP of the
# team-season random intercept; not an input.  The rated season is never a training season, so its
# team-seasons carry no intercept and the rating is made as if on an average team.
TEAM_SEASON = "team_season"

# One franchise, two team ids: the Charlotte Hornets moved to New Orleans after 2001-02 and took a new id
# with them (1610612740 from 2003); from 2005 1610612766 is the Bobcats, now the Hornets, and is itself.
# Unmapped, a player who stayed through the move looks like he changed teams -- 12 of Charlotte's 15 in
# 2002.  Every other team id in 1997-2026 covers all thirty seasons.  (old id, first, last season) -> id.
FRANCHISE_MOVES = {(1610612766, 1997, 2002): 1610612740}


def franchise(team_id, season) -> np.ndarray:
    """Team ids as franchises: `FRANCHISE_MOVES` applied, everything else unchanged."""
    team = np.array(team_id, dtype=np.int64)
    season = np.asarray(season)
    for (old, first, last), new in FRANCHISE_MOVES.items():
        team[(team == old) & (season >= first) & (season <= last)] = new
    return team

# What a REPLACEMENT LEVEL may be modelled on (experiment 13, the owner 2026-09-15).  Every one of these
# is measured EXACTLY however few minutes a man played -- his age and height do not get noisy at 40
# possessions, his box-score RATES do -- which is the whole point: the blend hands a player with few possessions over to
# the covariates that still work instead of to a floating constant.  `scripts/67_blend_apm.py` fits the
# coefficients against APM, and they are pinned by players who DO have minutes, so a 155-possession man
# borrows strength from players with heavy minutes who look like him.
#
# `age` is imputed to the season median for about 10.5% of player-seasons: `roles.player_season_inputs`
# sets an `age_imputed` flag and `scripts/49_role_panel.py` does not carry it, so the flag cannot be used
# as a column here.  Anyone adding it must widen the panel first.
#
# `poss_pct` and `gs_pct` are measured on the RATED season and are the covariate DECISIONS.md already
# sized as almost entirely hindsight: possession share at H was the largest map gain ever measured
# (-0.19, z -3.7) and worth -0.006 on HALF of H's games, keeping 3% of its value.  For a replacement
# level the coach's revealed opinion of a man nobody has plus-minus on is legitimate information, but it
# carries that control or it is not measured.
LEVEL_COVARIATES = ["age", "exp_yrs", "exp_poss", "entry_age", "height", "weight", "draft_pick",
                    "poss_pct", "gs_pct", "tenure", "n_teams"]

# What the booster sees.  54 names: 42 of SHOT_FEATURES (all but `season`), 3 career, 5 bio, 4 on-court.
PRIOR_FEATURES = [f for f in SHOT_FEATURES if f != "season"] + CAREER + BIO + ONC

# What the panel must carry for `add_derived` to build the rest.  The raw (uncentred) rates are the
# RATIOS' inputs; the shot totals and the block league levels are the SHOTQ inputs.
INPUT_COLUMNS = (list(FEATURES) + ROLE_INPUTS + CLOSENESS + [f"raw_{c}" for c in FEATURES]
                 + list(SHOT_TOTALS) + list(SHOT_LEAGUE) + CAREER + BIO + ONC + OFFC + NET + PIECES + PO_SHARE + SHOT_MIX)


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
# EVALUATION, not in the product.  The board rates a COMPLETED season, ruling 1 allows H's own games as
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

# The booster half of the stacked prior (`stackprior.StackedSPM`, the owner 2026-10-02): scripts/92_stack_boruta.py,
# 50 trials on 2026's training rows, candidates every `PRIOR_FEATURES` name but the plus-minus columns (`ONC`, which
# go to the elastic net).  It kept ALL 50 on both sides (outputs/csv/boruta_stack_table.csv): 35,000 rows, a
# career row and its chunks per player, give it the power to call almost any real signal "not noise".
STACK_BOOSTER_O = [f for f in PRIOR_FEATURES if f not in ONC]
STACK_BOOSTER_D = [f for f in PRIOR_FEATURES if f not in ONC]

FEATURE_SETS = {
    # the stacked prior: the plus-minus columns for the elastic net, the Boruta list for the booster
    # (on-court and off-court, the owner 2026-10-02; `stackprior.PLUS_MINUS`)
    "stack": {"O": ONC + OFFC + STACK_BOOSTER_O, "D": ONC + OFFC + STACK_BOOSTER_D},
    "sy": {"O": PRIOR_FEATURES, "D": PRIOR_FEATURES},
    "sy_noonc": {side: [f for f in PRIOR_FEATURES if f not in ONC] for side in ("O", "D")},
    "boruta": {"O": BORUTA_O, "D": BORUTA_D or PRIOR_FEATURES},
    "boruta_noonc": {"O": [f for f in BORUTA_O if f not in ONC],
                     "D": [f for f in (BORUTA_D or PRIOR_FEATURES) if f not in ONC]},
    # experiment 6 (2026-09-14): the on-court columns off the DEFENSIVE list only.  `onc_d` is points
    # allowed while he is on the floor, a lineup quantity; the shipped defensive prior has no on-court
    # column and beats the single-year one on the year-over-year test (DECISIONS.md, experiment 1).
    "boruta_noonc_d": {"O": BORUTA_O, "D": [f for f in BORUTA_D if f not in ONC]},
    # the owner's idea (2026-09-14): the off-court record and the on/off net beside the on-court record,
    # both sides.  `scripts/65_offcourt_panel.py` writes the columns into the season panel.
    "boruta_offc": {"O": BORUTA_O + OFFC + NET, "D": BORUTA_D + OFFC + NET},
    # the owner, 2026-10-01: "let's put them all in" -- the four ratings (on-court and off-court, offence and
    # defence) in BOTH priors.  The on-court pair was already in both; this adds the off-court pair and nothing
    # else (no possession counts, no net), on the current pipeline rather than the 2026-09-14 one.
    "boruta_onoff": {"O": BORUTA_O + ["offc_o", "offc_d"], "D": BORUTA_D + ["offc_o", "offc_d"]},
    # the owner, 2026-10-01 ("try it"): on/off ALONE -- `net_o` / `net_d` (on-court minus off-court) in place of the
    # raw on-court pair in both priors, so the prior never sees how good the team is, only the difference the player
    # makes.  The possession counts stay.
    "boruta_net": {side: [{"onc_o": "net_o", "onc_d": "net_d"}.get(f, f) for f in feats]
                   for side, feats in (("O", BORUTA_O), ("D", BORUTA_D))},
    # experiment 15 (the owner, 2026-09-15): tell the prior what SCORE STATE a player's statistics were
    # compiled in.  The ratings side already knows -- the design carries a garbage-time column and a
    # margin slope -- but nothing in the prior's inputs does, and that asymmetry is the leading
    # explanation for the prior being flat in exposure where APM is steep (DECISIONS.md, experiment 14).
    # A man with 0-50 possessions takes 61% of them in garbage time against 2.5% for a 4,000+ player, at
    # an average score gap of 18 points against 6.7, so his per-possession rates describe a different
    # game.  `scripts/69_closeness_panel.py` writes the three columns.  The owner's call was to hand the
    # prior the exposure and let the booster learn the discount, rather than reweighting the statistics.
    "boruta_close": {"O": BORUTA_O + CLOSENESS, "D": BORUTA_D + CLOSENESS},
    # experiment 26 (the owner, 2026-09-28): team and game context.  scripts/87_context_boruta.py, 50 trials,
    # on the rows 62 trained on for 2026 with outside labels (outputs/csv/boruta_context_table.csv), from
    # the shipped lists plus every piece, same_team, the closeness columns and po_share.  Accepted + tentative:
    # offence drops only `onc_d`; defence drops pc_on_rtg_d, pc_on_signal_o, pc_on_rtg_o, pc_teammates_o.
    # `same_team` is accepted on both sides.  Needs --rows=chunks --chunk_label=outside.
    # experiment 30 (the owner, 2026-09-30): the incumbent's lists plus the same-team measure alone, for
    # `adjacent_rows` -- rated as if every player had been traded.  Needs --chunk_label=adjacent (or outside).
    # experiment 34 (the owner, 2026-10-04): teammates' shot mix with him on the floor, on the OFFENSIVE list only.
    # `scripts/100_shot_mix_panel.py` writes the column into the season panel.
    "boruta_mix": {"O": BORUTA_O + SHOT_MIX, "D": BORUTA_D},
    "boruta_same_team": {"O": BORUTA_O + [SAME_TEAM], "D": BORUTA_D + [SAME_TEAM]},
    "boruta_context": {
        "O": [f for f in BORUTA_O if f != "onc_d"] + PIECES + [SAME_TEAM] + CLOSENESS + PO_SHARE,
        "D": BORUTA_D + [p for p in PIECES if p not in ("pc_on_rtg_d", "pc_on_signal_o", "pc_on_rtg_o",
                                                           "pc_teammates_o")] + [SAME_TEAM] + CLOSENESS + PO_SHARE,
    },
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


def aggregate(rows: pd.DataFrame, features=None, by=None) -> pd.DataFrame:
    """One row per player: his `INPUT_COLUMNS` averaged over `rows` by possessions, then derived.

    `rows` is already the side and the seasons the caller wants (typically every season but the held-out
    one, one side).  Indexed by `player_id`, so it joins straight onto a target frame.  `by` is an
    alternative grouping key aligned with `rows` (`chunk_rows` uses one per chunk of a player's seasons).
    """
    features = list(PRIOR_FEATURES if features is None else features)
    cols = [c for c in INPUT_COLUMNS if c in rows.columns]
    key = rows.player_id if by is None else by
    weight = rows.poss.groupby(key).sum()
    mean = rows[cols].mul(rows.poss, axis=0).groupby(key).sum().div(weight, axis=0)
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


# How much evidence a training row's box score rests on.  Two extra features the booster sees under
# `chunk_rows`, so it can learn that a one-season row is to be trusted less than a career row; the rated
# season's own row reads its possessions and 1.
CHUNK_FEATURES = ["chunk_poss", "chunk_seasons"]


def stratified_player_folds(train: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
    """A fold id per training row, all of a player's rows in one fold, folds balanced on the LABEL.

    The out-of-player prior (2026-09-14): every player's prior comes from a booster that never saw any of
    his rows, so it cannot learn "this fingerprint is LeBron" and return his career number (measured:
    Curry's 2026 prior fell 1.8 per 100 and LeBron's 1.4 when their own rows were left out).

    Why the folds are balanced on the label.  Austin, Pe'er and Korem (2025): hold a fold out and the
    training mean label moves away from the fold's own mean by -n_j (p_j - pbar) / (S - n_j), and a
    booster's baseline is that mean, so every player in the fold is predicted a constant too low or too
    high.  The shift is zero when every fold's weighted mean label equals the full mean.  So: players
    sorted by label, dealt in snake order into the folds, weights carried -- no partner fold dropped, no
    rows lost, and `fold_mean_shift` prints the residual shift so it can be seen to be ~0.

    `train` is `chunk_rows` / `prior_rows` output: indexed by player_id, with `target` and `weight`.
    The splitter itself is `rloocv.BalancedGroupKFold`, reusable wherever groups need balanced folds.
    """
    return BalancedGroupKFold(n_folds).fold_ids(train.target.to_numpy(float), train.index.to_numpy(),
                                                train.weight.to_numpy(float))


def fold_mean_shift(train: pd.DataFrame, fold: np.ndarray) -> np.ndarray:
    """Per fold: the training mean label with that fold held out, minus the full mean (weighted)."""
    label, w = train.target.to_numpy(float), train.weight.to_numpy(float)
    S, W = float((w * label).sum()), float(w.sum())
    out = []
    for f in np.unique(fold):
        m = fold == f
        out.append((S - (w[m] * label[m]).sum()) / max(W - w[m].sum(), 1e-9) - S / W)
    return np.asarray(out)


def _chunk_members(rows: pd.DataFrame, sizes=(1, 2, 3)) -> pd.DataFrame:
    """`rows` repeated once per chunk each row belongs to, with the chunk's key in a `chunk` column.

    A chunk is a run of `size` CONSECUTIVE seasons among the ones `rows` holds for the player, in the
    order he played them, for each size in `sizes`.  Consecutive in `rows`, not on the calendar: when the
    caller has already taken the rated season and its neighbours out, a chunk can straddle the gap (a
    2013 + 2017 chunk around a rated 2015), and a season he missed is simply skipped.
    """
    ordered = rows.sort_values(["player_id", "season"]).reset_index(drop=True)
    ordered["k"] = ordered.groupby("player_id").cumcount()
    n_seasons = ordered.groupby("player_id").k.transform("max") + 1
    parts = []
    for size in sizes:
        for offset in range(int(size)):
            start = ordered.k - offset
            fits = (start >= 0) & (start + size <= n_seasons)
            part = ordered[fits]
            key = part.player_id.astype(str) + ":" + str(size) + ":" + start[fits].astype(str)
            parts.append(part.assign(chunk=key.to_numpy()))
    return pd.concat(parts, ignore_index=True)


def _season_key(seasons) -> tuple:
    """A set of seasons as the one hashable form every label lookup uses: sorted ints, no repeats."""
    return tuple(sorted({int(s) for s in seasons}))


def harmonic_overlap(left: pd.DataFrame, right: pd.DataFrame, key: str = "key") -> pd.Series:
    """Per key: the sum over teams of HM(left's possession share with the team, right's), HM(a, b) = 2ab / (a + b).

    The owner's soft same-team measure (2026-09-28).  `left` and `right` are rows of (`key`, `team_id`,
    `poss_on`); shares are taken within each key.  1 when the two spreads over teams are identical, 0 when they
    have no team in common; a shared team counts by a softened smaller share -- 50/50 against 100/0 is
    HM(0.5, 1) = 0.67.  Indexed by `left`'s keys; a key with nothing in `right` is 0.
    """
    def shares(frame):
        f = frame.groupby([key, "team_id"], as_index=False).poss_on.sum()
        f = f[f.poss_on > 0]
        return f.assign(share=f.poss_on / f.groupby(key).poss_on.transform("sum"))

    a, b = shares(left), shares(right)
    both = a.merge(b, on=[key, "team_id"], suffixes=("_a", "_b"))
    hm = 2.0 * both.share_a * both.share_b / (both.share_a + both.share_b)
    return hm.groupby(both[key]).sum().reindex(pd.unique(left[key])).fillna(0.0)


def _chunk_same_team(members: pd.DataFrame, shares: pd.DataFrame, outside: bool) -> pd.Series:
    """Per chunk key: `harmonic_overlap` of the chunk's teams against the teams of the seasons its label is fit on.

    `shares` is one row per (player_id, season, team_id) with `poss_on`, teams as franchises, restricted by the
    caller to the seasons the career label is fit on.  Outside labels are fit on those seasons minus the chunk's,
    so the chunk's own possessions come off each team; a career label covers all of them.
    """
    per = shares.groupby(["player_id", "season", "team_id"], as_index=False).poss_on.sum()
    career = per.groupby(["player_id", "team_id"], as_index=False).poss_on.sum()
    inside = (members[["chunk", "player_id", "season"]].merge(per, on=["player_id", "season"])
              .groupby(["chunk", "player_id", "team_id"], as_index=False).poss_on.sum())
    label = members[["chunk", "player_id"]].drop_duplicates().merge(career, on="player_id")
    if outside:
        label = label.merge(inside[["chunk", "team_id", "poss_on"]], on=["chunk", "team_id"], how="left",
                            suffixes=("", "_in"))
        label["poss_on"] = (label.poss_on - label.poss_on_in.fillna(0.0)).clip(lower=0.0)
    got = harmonic_overlap(inside.rename(columns={"chunk": "key"}), label.rename(columns={"chunk": "key"}))
    return got.reindex(pd.unique(members.chunk)).fillna(0.0)


def _top_team_season(frame: pd.DataFrame, key: str) -> pd.Series:
    """Per `key`: season * 10**10 + team of the (season, team) with the most `poss_on`; ties to the earlier."""
    top = (frame.sort_values([key, "poss_on", "season", "team_id"], ascending=[True, False, True, True])
           .drop_duplicates(key))
    return pd.Series(top.season.to_numpy(np.int64) * 10**10 + top.team_id.to_numpy(np.int64),
                     index=top[key].to_numpy())


class AgingCurve:
    """A player's expected level by age, relative to `reference` years old, from season-to-season changes.

    The owner's call (2026-09-30), after experiments 26-27 turned out to carry an age tilt.  The delta method:
    every pair of CONSECUTIVE seasons a player has outside `exclude`, the change in his single-season rating,
    weighted by the harmonic mean of the two seasons' possessions, regressed on his age with a polynomial of
    `degree` (evaluated half a year on, since the change runs from age a to a + 1); the curve is that increment
    integrated from `reference`.  Ages are clamped to [19, 40].  Players who fall out of the league do not
    make a pair, so the decline at the old end is if anything understated (the usual survivor bias).

    `ratings` rows: player_id, season, age, the rating column `value` (in the sign the label uses) and `poss`.
    """

    def __init__(self, ratings: pd.DataFrame, value: str, poss: str = "poss", exclude=(), degree: int = 2,
                 reference: float = 27.0):
        r = ratings[~ratings.season.isin(list(exclude))][["player_id", "season", "age", value, poss]]
        pairs = r.merge(r, on="player_id", suffixes=("", "_next"))
        pairs = pairs[(pairs.season_next == pairs.season + 1) & (pairs[poss] > 0) & (pairs[f"{poss}_next"] > 0)]
        x = np.clip(pairs.age.to_numpy(float), 19, 40) + 0.5 - reference
        change = (pairs[f"{value}_next"] - pairs[value]).to_numpy(float)
        w = 2.0 / (1.0 / pairs[poss].to_numpy(float) + 1.0 / pairs[f"{poss}_next"].to_numpy(float))
        A = np.column_stack([x ** k for k in range(degree + 1)])
        self.reference, self.pairs = float(reference), int(len(pairs))
        # too few pairs to fit a curve: a flat one, which moves no label
        self.coef = (np.linalg.solve((A * w[:, None]).T @ A, (A * w[:, None]).T @ change)
                     if self.pairs > 5 * (degree + 1) else np.zeros(degree + 1))

    def __call__(self, age) -> np.ndarray:
        d = np.clip(np.asarray(age, dtype=float), 19, 40) - self.reference
        return sum(c * d ** (k + 1) / (k + 1) for k, c in enumerate(self.coef))


def _age_shift(members: pd.DataFrame, rows: pd.DataFrame, curve, outside: bool) -> pd.Series:
    """Per chunk key: the curve at the chunk's own ages minus the curve over the seasons its label is fit on,
    both possession-weighted.  Outside labels are fit on his seasons in `rows` minus the chunk's; a career label
    on all of them."""
    r = rows[["player_id", "season", "poss", "age"]].copy()
    r["pc"] = r.poss.to_numpy(float) * curve(r.age.to_numpy(float))
    career = r.groupby("player_id")[["poss", "pc"]].sum()
    inside = members[["chunk", "player_id", "season"]].merge(r, on=["player_id", "season"])
    per = inside.groupby("chunk").agg(player_id=("player_id", "first"), poss=("poss", "sum"), pc=("pc", "sum"))
    own = per.pc / per.poss.where(per.poss > 0)
    tot = career.reindex(per.player_id.to_numpy())
    rest_poss = tot.poss.to_numpy() - (per.poss.to_numpy() if outside else 0.0)
    rest_pc = tot.pc.to_numpy() - (per.pc.to_numpy() if outside else 0.0)
    label = np.where(rest_poss > 0, rest_pc / np.where(rest_poss > 0, rest_poss, 1.0), np.nan)
    return (own - label).fillna(0.0)


def _adjacent_windows(rows: pd.DataFrame, sizes=(1, 2, 3)) -> pd.DataFrame:
    """Every window of 2k CONSECUTIVE seasons a player has in `rows`, for k in `sizes`, split in the middle, both
    ways round: one row per (window, direction) with the feature half's and the label half's seasons.

    Consecutive in `rows`, as the chunks are: a window can straddle seasons the caller left out."""
    seasons = rows.groupby("player_id").season.apply(lambda s: sorted(set(int(x) for x in s)))
    out = []
    for pid, have in seasons.items():
        for k in sizes:
            for i in range(len(have) - 2 * int(k) + 1):
                first, second = tuple(have[i:i + k]), tuple(have[i + k:i + 2 * k])
                out.append((f"{pid}:{k}:{i}:f", pid, first, second))
                out.append((f"{pid}:{k}:{i}:b", pid, second, first))
    return pd.DataFrame(out, columns=["key", "player_id", "feature_seasons", "label_seasons"])


def adjacent_season_sets(rows: pd.DataFrame, sizes=(1, 2, 3)) -> list:
    """Every distinct set of seasons a window row's label is fit ON (`adjacent_rows`), as sorted tuples."""
    return sorted(set(_adjacent_windows(rows, sizes).label_seasons))


def adjacent_rows(target: pd.DataFrame, rows: pd.DataFrame, column: str, labels: dict, features=None,
                  sizes=(1, 2, 3), unseen=(), shares: pd.DataFrame | None = None) -> pd.DataFrame:
    """The career row plus rows labelled by the seasons right NEXT to them (the owner, 2026-09-30, experiment 30).

    For every window of 2k consecutive seasons a player has in `rows` (k in `sizes`: two, four or six seasons),
    split in the middle, each half predicts the other: its box score is the features, and the label is his RAPM
    fit on the other half's seasons alone, `labels[<those seasons>]`.  So a label always sits right beside its
    features in time -- at nearly the same age, where experiment 25's outside labels reached across his whole
    career -- and when he changed teams between the two halves the row is a traded example: `SAME_TEAM` (from
    `shares`, as in `chunk_rows`) is near 0 there and near 1 for a player who stayed.  One-team players count in
    full; they are the same-team examples, and the traded players teach the model what a trade does.

    The career row is `prior_rows`' with `target`, same team 1.  A player's window rows together weigh what his
    career row weighs, split by the feature half's possessions -- the `chunk_rows` rule, so no player's weight
    grows with career length or follows how much his labels rest on (experiment 25b).  A window whose label has
    no row for him (fewer than `MIN_POSSESSIONS` in that half) is dropped.  Columns beside the features:
    `row_kind` ("career" / "window"), `feature_half` and `label_half` (the seasons, as text),
    `label_possessions`, and `CHUNK_FEATURES` for the feature half.
    """
    features = list(PRIOR_FEATURES if features is None else features)
    per_player = rows.groupby("player_id")
    base = prior_rows(target, rows, column, features)
    base = base.assign(chunk_poss=per_player.poss.sum().reindex(base.index).to_numpy(float),
                       chunk_seasons=per_player.season.nunique().reindex(base.index).to_numpy(float),
                       label_possessions=base.possessions.to_numpy(float),
                       label_key=_key_text(_season_key(unseen)), row_kind="career", feature_half="", label_half="")
    windows = _adjacent_windows(rows, sizes)
    members = (windows[["key", "player_id", "feature_seasons"]].explode("feature_seasons")
               .rename(columns={"feature_seasons": "season"}))
    members["season"] = members.season.astype(int)
    members = members.merge(rows, on=["player_id", "season"], how="inner")
    chunks = aggregate(members, features, by=members.key)
    grouped = members.groupby("key")
    chunks["chunk_poss"] = grouped.poss.sum().reindex(chunks.index).to_numpy(float)
    chunks["chunk_seasons"] = grouped.season.nunique().reindex(chunks.index).to_numpy(float)
    win = windows.set_index("key").reindex(chunks.index)
    chunks["player_id"] = win.player_id.to_numpy()
    chunks["feature_half"] = win.feature_seasons.map(_key_text).to_numpy()
    chunks["label_half"] = win.label_seasons.map(_key_text).to_numpy()
    by_half = {_key_text(_season_key(k)): v for k, v in labels.items()}
    missing = sorted(set(chunks.label_half) - set(by_half))
    if missing:
        raise KeyError(f"no label fit on the seasons {missing[:3]}{'...' if len(missing) > 3 else ''}; "
                       f"build one per `adjacent_season_sets(rows)` set")
    long = pd.concat([by_half[h][["player_id", column, "possessions"]]
                      .rename(columns={"possessions": "label_possessions"}).assign(label_half=h)
                      for h in dict.fromkeys(chunks.label_half)], ignore_index=True)
    chunks = chunks.join(long.set_index(["player_id", "label_half"]), on=["player_id", "label_half"], how="inner")
    chunks = chunks.join(target.set_index("player_id")[["possessions"]], on="player_id", how="inner")
    if shares is not None:
        per = shares.groupby(["player_id", "season", "team_id"], as_index=False).poss_on.sum()

        def half_shares(col):
            h = (win.loc[chunks.index, ["player_id", col]].explode(col).rename(columns={col: "season"})
                 .reset_index().rename(columns={"index": "key"}))
            h["season"] = h.season.astype(int)
            return h.merge(per, on=["player_id", "season"])
        feat, lab = half_shares("feature_seasons"), half_shares("label_seasons")
        chunks[SAME_TEAM] = harmonic_overlap(feat, lab).reindex(chunks.index).fillna(0.0).to_numpy(float)
        chunks[TEAM_SEASON] = _top_team_season(feat, "key").reindex(chunks.index).fillna(-1).to_numpy(np.int64)
        base[SAME_TEAM] = 1.0
        base[TEAM_SEASON] = _top_team_season(per, "player_id").reindex(base.index).fillna(-1).to_numpy(np.int64)
    chunks["row_kind"], chunks["label_key"] = "window", ""
    chunks = chunks.dropna(subset=features + [column]).set_index("player_id")
    share = chunks.chunk_poss / chunks.groupby(level=0).chunk_poss.transform("sum")
    chunks = chunks.assign(target=chunks[column].to_numpy(float),
                           weight=(share * chunks.possessions).to_numpy(float))
    return pd.concat([base, chunks])


def _key_text(key) -> str:
    """A season key as plain text ("2014,2015,2016"): tuples make awkward index levels."""
    return ",".join(str(s) for s in key)


def chunk_season_sets(rows: pd.DataFrame, sizes=(1, 2, 3)) -> list:
    """Every distinct set of seasons a chunk of `rows` covers, as sorted tuples.

    What the caller needs to build `chunk_rows(labels=...)`: one label per set, each a RAPM with that
    set left out (on top of whatever the caller already leaves out).
    """
    members = _chunk_members(rows, sizes)
    return sorted(set(members.groupby("chunk").season.agg(_season_key)))


def chunk_rows(target: pd.DataFrame, rows: pd.DataFrame, column: str, features=None,
               sizes=(1, 2, 3), labels: dict | None = None, unseen=(), weight_by: str = "career",
               shares: pd.DataFrame | None = None, age_curve=None, label_scale=None) -> pd.DataFrame:
    """The owner's design (2026-09-13): the career row per player, PLUS rows built from chunks of his seasons.

    `prior_rows` is kept exactly -- one row per player, his box score averaged over every season in `rows`
    -- and to it are added, for the same player, one row per CONTIGUOUS run of `size` of his seasons (in
    the order he played them) for each size in `sizes`: his inputs averaged over that chunk, then derived.
    The point is an artificial increase of the sample that shows the booster the same player at several
    noise levels, with `CHUNK_FEATURES` saying which level.  Rows of one player are not independent, so
    this is not more players; it is information about how the map degrades with less evidence, which is
    the padding question learned from data instead of set per stat.

    **The label of a chunk row.**  With `labels=None` (the incumbent) every chunk row carries the SAME
    label as his career row -- his RAPM over every season in `rows`, the chunk's own seasons included.
    That label is constant across a player's rows, so nothing that varies BETWEEN his chunks can be
    learned from it: a feature that differs chunk to chunk can only teach how players differ from one
    another (DECISIONS.md, experiment 15).

    `labels` switches to OUTSIDE labels (the owner, 2026-09-28, experiment 25): a chunk row is labelled
    with his RAPM over the seasons OUTSIDE the chunk, so its box score and its label share no game --
    the same relationship `season_rows` has, and the one the rated season's row has to its unseen truth.
    `labels` maps an excluded set of seasons to a target frame fit without them; the chunk row looks up
    `unseen` plus the chunk's own seasons, where `unseen` is what the caller already left out of `rows`
    (the rated season and its neighbours) and what `target` was fit without.  `chunk_season_sets` lists
    the sets needed.  A 1-season chunk of season s therefore carries exactly `season_rows`'s label for s.
    The career row keeps `target`.  A chunk whose player has no evidence outside it -- no row for him in
    its label, which needs `MIN_POSSESSIONS` there -- is dropped.  Keys may be any iterable of seasons;
    they are compared as sets.

    Weights: the career row keeps the weight `prior_rows` gives it (the possessions behind the label); a
    player's chunk rows TOGETHER weigh the same, split among them by chunk possessions.  So every player's
    total weight is twice his career row's, whatever his career length, and half of it sits on the row that
    matches the inference row least and half on the ones that match it more.  Unchanged by the outside
    labels: `possessions` stays the career label's, `label_possessions` says what the row's own label
    rests on, and `label_key` names the seasons that label was fit without ("2014,2015,2016").

    `weight_by="label"` (the owner, 2026-09-29, after experiment 25) puts each chunk row's OWN label
    possessions where the career label's were: `share x label_possessions`.  An outside label rests on his
    career minus the chunk, so it is noisier than the career label, and a label's noise variance falls as
    1 / possessions; with `"career"` those noisier labels kept the career label's weight.  The career row is
    untouched, a player's chunk rows together now weigh less than it, and a chunk whose label rests on
    little weighs little.  With career labels the two settings are identical.  (Rejected, experiment 25b:
    weighting by a label's possessions is weighting by career length, which is quality.)

    `shares` (experiment 26, the owner 2026-09-28) adds `SAME_TEAM` to every row: `harmonic_overlap` of the
    row's possessions by franchise against those of the seasons its label is fit on.  The career row is
    exactly 1 (the same seasons); a chunk on a team he never played for otherwise is 0.  `shares` is one row
    per (player_id, season, team_id) with `poss_on`, teams as `franchise`s, and must cover the seasons the
    career label is fit on and no others.  Only outside labels make this vary within a player's label.

    `age_curve` (an `AgingCurve`, the owner 2026-09-30) moves each chunk's label to the chunk's own age: a label
    fit on his other seasons describes him at THEIR ages, so it gains `label_scale(label_possessions) x (curve at
    the chunk's ages - curve over the label's seasons)`, both possession-weighted over `rows` (which must carry
    `age`).  `label_scale` puts the curve on the label's own scale: a label shrunk by its penalty moves by the same
    fraction.  The career row's seasons are its label's seasons, so it never moves; `age_shift` records the move.

    Contiguous only: a chunk of 2004 and 2024 averaged together is nobody's season.
    """
    if weight_by not in ("career", "label"):
        raise ValueError(f"weight_by={weight_by!r}: write career or label")
    features = list(PRIOR_FEATURES if features is None else features)
    label = target.set_index("player_id")[[column, "possessions"]]
    per_player = rows.groupby("player_id")
    base = prior_rows(target, rows, column, features)
    base = base.assign(chunk_poss=per_player.poss.sum().reindex(base.index).to_numpy(float),
                       chunk_seasons=per_player.season.nunique().reindex(base.index).to_numpy(float),
                       label_possessions=base.possessions.to_numpy(float),
                       label_key=_key_text(_season_key(unseen)))

    dup = _chunk_members(rows, sizes)
    chunks = aggregate(dup, features, by=dup.chunk)
    grouped = dup.groupby("chunk")
    chunks["chunk_poss"] = grouped.poss.sum().reindex(chunks.index).to_numpy(float)
    chunks["chunk_seasons"] = grouped.season.nunique().reindex(chunks.index).to_numpy(float)
    chunks["player_id"] = grouped.player_id.first().reindex(chunks.index).to_numpy()
    if labels is None:
        chunks = chunks.join(label, on="player_id", how="inner")
        chunks["label_possessions"] = chunks.possessions.to_numpy(float)
        chunks["label_key"] = _key_text(_season_key(unseen))
    else:
        chunks = _join_outside_labels(chunks, grouped.season.agg(_season_key), labels, unseen, label, column)
    if shares is not None:
        chunks[SAME_TEAM] = (_chunk_same_team(dup, shares, outside=labels is not None)
                             .reindex(chunks.index).fillna(0.0).to_numpy(float))
        base[SAME_TEAM] = 1.0
        # the team-season intercept's group (experiment 27): the chunk's biggest team-season, and for the
        # career row the biggest of his whole label span
        per = shares.groupby(["player_id", "season", "team_id"], as_index=False).poss_on.sum()
        inside = dup[["chunk", "player_id", "season"]].merge(per, on=["player_id", "season"])
        chunks[TEAM_SEASON] = (_top_team_season(inside, "chunk").reindex(chunks.index).fillna(-1)
                               .to_numpy(np.int64))
        base[TEAM_SEASON] = (_top_team_season(per, "player_id").reindex(base.index).fillna(-1)
                             .to_numpy(np.int64))
    if age_curve is not None:
        # each chunk's label moved to the chunk's own age; the career row is its label's own seasons
        gap = _age_shift(dup, rows, age_curve, outside=labels is not None).reindex(chunks.index).fillna(0.0)
        scale = (np.ones(len(chunks)) if label_scale is None
                 else np.asarray(label_scale(chunks.label_possessions.to_numpy(float)), dtype=float))
        chunks["age_shift"] = scale * gap.to_numpy(float)
        chunks[column] = chunks[column].to_numpy(float) + chunks.age_shift.to_numpy(float)
        base["age_shift"] = 0.0
    chunks = chunks.dropna(subset=features + [column]).set_index("player_id")
    share = chunks.chunk_poss / chunks.groupby(level=0).chunk_poss.transform("sum")
    behind = chunks.possessions if weight_by == "career" else chunks.label_possessions
    chunks = chunks.assign(target=chunks[column].to_numpy(float),
                           weight=(share * behind).to_numpy(float))
    return pd.concat([base, chunks])


def _join_outside_labels(chunks: pd.DataFrame, chunk_seasons: pd.Series, labels: dict, unseen,
                         career: pd.DataFrame, column: str) -> pd.DataFrame:
    """Each chunk row's label from `labels[unseen + its own seasons]`; the weight's possessions stay the
    career label's.  Inner joins on both, so a chunk with no outside label, or a player with no career
    label, is dropped.  `label_key` names the excluded set each row's label was fit without."""
    by_key = {_key_text(_season_key(k)): v for k, v in labels.items()}
    left = _season_key(unseen)
    wanted = chunk_seasons.reindex(chunks.index).map(lambda s: _key_text(_season_key(left + tuple(s))))
    missing = sorted(set(wanted) - set(by_key))
    if missing:
        raise KeyError(f"no label for the excluded season sets {missing[:3]}{'...' if len(missing) > 3 else ''}; "
                       f"build one per `chunk_season_sets(rows)` set, plus `unseen`")
    frames = [by_key[k][["player_id", column, "possessions"]]
              .rename(columns={"possessions": "label_possessions"}).assign(_key=k)
              for k in dict.fromkeys(wanted)]
    long = pd.concat(frames, ignore_index=True).set_index(["player_id", "_key"])
    out = chunks.assign(_key=wanted.to_numpy())
    out = out.join(long, on=["player_id", "_key"], how="inner").rename(columns={"_key": "label_key"})
    return out.join(career[["possessions"]], on="player_id", how="inner")


def team_movement(per_team: pd.DataFrame, min_poss: float = 100.0) -> pd.Series:
    """Per player, the chance that two possessions of his career came from different teams.

    `1 - sum(share_i ** 2)` over his teams: the Gini-Simpson index.  The owner's measure (2026-09-16) of
    how much team variation sits behind a player's label, in the form he corrected it to the same day.

    **The first form was `1 - share on his most-played team`, and it ignores the shape of the tail.**  A
    player at 50/10/10/10/10/10 scored 0.50, identical to one at 50/50, though the first has five
    independent contrasts and the second has one.  This index is a strict generalisation, and never below
    the old form: the two agree exactly when every team he played for got the same share of him (one team,
    or 50/50, or three at a third each) and this one is higher otherwise -- 0.70 rather than 0.50 for the
    six-team case.  On the 2,584 players with at least 100 possessions before 2026 they correlate 0.985
    (rank 0.992) and 70% of players gain, the largest gap being 0.211, so this refines the measure rather
    than replacing the idea; the 767 one-team players sit at exactly zero under both.

    Not a mean of the shares: a geometric or harmonic mean is crushed by one tiny share, which is
    backwards -- a cameo on a seventh team is a little more contrast, not almost none.  Not entropy
    perplexity either, which is too generous to the tail (it reads the 50/10-times-five case as 4.47
    effective teams against this index's 3.33).  `1 / sum(share ** 2)` is that effective number of teams
    and is this index's own reciprocal.

    Why this and not "was he traded this season": the prior's label is ONE leave-season-out RAPM per
    player pooled over his whole career, so how well it is identified depends on the team variation
    across all of it, not on what happened in any one season.  A mid-season trade flag would be the
    right idea measured at the wrong granularity.

    What it is still blind to: teams, not teammate sets.  Two seasons on one team with a rebuilt roster
    give real contrast and score zero.

    `per_team` is one row per (player_id, team_id) with `poss_on`, restricted by the caller to the
    seasons the label was fitted on -- never the rated season, so the two agree about what evidence
    exists.  Returned indexed by player_id.
    """
    totals = per_team.groupby("player_id").poss_on.sum()
    share = per_team.poss_on.to_numpy(float) / totals.reindex(per_team.player_id).to_numpy(float)
    concentration = pd.Series(share ** 2, index=per_team.player_id.to_numpy()).groupby(level=0).sum()
    movement = 1.0 - concentration.reindex(totals.index)
    movement.index.name = totals.index.name
    return movement[totals >= float(min_poss)]


CAREER_BANDS = [0.0, 2000.0, 5000.0, 15000.0, 40000.0, np.inf]     # label possessions


def reweight_by_movement(train: pd.DataFrame, movement: pd.Series, floor: float = 0.0,
                         bands=None) -> pd.DataFrame:
    """Multiply every training row's weight by its player's `team_movement`, plus `floor`.

    The total weight is preserved, so the booster's own regularisation means the same thing before and
    after and only the DISTRIBUTION of weight across players changes.

    `floor=0` is the owner's rule exactly, and it drops the one-team players outright: 29.4% of the
    players behind a 2026 label, holding 10.1% of the weight.  A positive floor keeps them in at
    reduced weight, which is the knob to sweep if the pure version overshoots.

    Note what this does NOT do, because the guess went the other way first: it does not tilt the map
    toward journeymen.  Movement RISES with career length (mean 0.11 under 2,000 possessions against
    0.41 over 30,000), so the weight moves toward long careers -- the same rows the memorisation work
    already found are the easiest targets to predict.  The criterion is the only arbiter of that.

    `bands` (the owner, 2026-09-30) removes exactly that: edges on the label's possessions (`possessions`,
    his career over the training seasons; `CAREER_BANDS` by default), and each band keeps the total weight
    it had, so the weighting moves weight only between players of similar career length -- from one-team
    players to players who moved -- and never from short careers to long ones.  Experiment 25b showed where
    a weight that follows career length goes: long careers belong to good players.
    """
    factor = movement.reindex(train.index).fillna(0.0).to_numpy(float) + float(floor)
    before = train.weight.to_numpy(float)
    weight = before * factor
    if weight.sum() <= 0:
        raise ValueError("the movement weighting left no training weight at all")
    if bands is None:
        return train.assign(weight=weight * (before.sum() / weight.sum()))
    edges = CAREER_BANDS if bands is True else list(bands)
    band = np.digitize(train.possessions.to_numpy(float), edges[1:-1])
    out = weight.copy()
    for b in np.unique(band):
        k = band == b
        if weight[k].sum() > 0:
            out[k] = weight[k] * (before[k].sum() / weight[k].sum())
        else:                                  # a band of one-team players only, at floor 0: left as it was
            out[k] = before[k]
    return train.assign(weight=out)
