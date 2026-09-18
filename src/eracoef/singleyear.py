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

__all__ = ["PRIOR_FEATURES", "INPUT_COLUMNS", "BIO", "ONC", "OFFC", "NET", "ROLE_INPUTS", "CLOSENESS", "LEVEL_COVARIATES", "aggregate", "season_frame",
           "prior_rows", "season_rows", "chunk_rows", "CHUNK_FEATURES", "stratified_player_folds",
           "fold_mean_shift", "FEATURE_SETS", "feature_set", "OFFENSE_TARGET", "DEFENSE_TARGET",
           "team_movement", "reweight_by_movement",
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
# his team's record WITHOUT him, same games, same centring and padding (`investigate.offcourt_rates`),
# and the on/off net.  The owner, 2026-09-14: "usually we also include off-court-rating".
OFFC = ["offc_o", "offc_d", "offc_poss_o", "offc_poss_d"]
NET = ["net_o", "net_d"]                                    # onc minus offc, per side
ROLE_INPUTS = ["poss_pct", "gs_pct", "age"]

# What score state a player's statistics were compiled in (scripts/69_closeness_panel.py).  `closeness` is
# the possession-weighted mean of 1 / max(|margin|, 1) -- the same form `priorridge.team_game_weights`
# uses on the ratings objective, so "close" means one thing in both halves of the pipeline.
CLOSENESS = ["gt_share", "closeness", "abs_margin"]

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
                 + list(SHOT_TOTALS) + list(SHOT_LEAGUE) + CAREER + BIO + ONC + OFFC + NET)


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

FEATURE_SETS = {
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
    # experiment 15 (the owner, 2026-09-15): tell the prior what SCORE STATE a player's statistics were
    # compiled in.  The ratings side already knows -- the design carries a garbage-time column and a
    # margin slope -- but nothing in the prior's inputs does, and that asymmetry is the leading
    # explanation for the prior being flat in exposure where APM is steep (DECISIONS.md, experiment 14).
    # A man with 0-50 possessions takes 61% of them in garbage time against 2.5% for a 4,000+ player, at
    # an average score gap of 18 points against 6.7, so his per-possession rates describe a different
    # game.  `scripts/69_closeness_panel.py` writes the three columns.  The owner's call was to hand the
    # prior the exposure and let the booster learn the discount, rather than reweighting the statistics.
    "boruta_close": {"O": BORUTA_O + CLOSENESS, "D": BORUTA_D + CLOSENESS},
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


def chunk_rows(target: pd.DataFrame, rows: pd.DataFrame, column: str, features=None,
               sizes=(1, 2, 3)) -> pd.DataFrame:
    """The owner's design (2026-09-13): the career row per player, PLUS rows built from chunks of his seasons.

    `prior_rows` is kept exactly -- one row per player, his box score averaged over every season in `rows`
    -- and to it are added, for the same player, one row per CONTIGUOUS run of `size` of his seasons (in
    the order he played them) for each size in `sizes`: his inputs averaged over that chunk, then derived,
    carrying the SAME label as his career row.  The point is an artificial increase of the sample that shows
    the booster the same player at several noise levels, with `CHUNK_FEATURES` saying which level.  Rows of
    one player are not independent, so this is not more players; it is information about how the map
    degrades with less evidence, which is the padding question learned from data instead of set per stat.

    Weights: the career row keeps the weight `prior_rows` gives it (the possessions behind the label); a
    player's chunk rows TOGETHER weigh the same, split among them by chunk possessions.  So every player's
    total weight is twice his career row's, whatever his career length, and half of it sits on the row that
    matches the inference row least and half on the ones that match it more.

    Contiguous only: a chunk of 2004 and 2024 averaged together is nobody's season.
    """
    features = list(PRIOR_FEATURES if features is None else features)
    label = target.set_index("player_id")[[column, "possessions"]]
    per_player = rows.groupby("player_id")
    base = prior_rows(target, rows, column, features)
    base = base.assign(chunk_poss=per_player.poss.sum().reindex(base.index).to_numpy(float),
                       chunk_seasons=per_player.season.nunique().reindex(base.index).to_numpy(float))

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
    dup = pd.concat(parts, ignore_index=True)
    chunks = aggregate(dup, features, by=dup.chunk)
    grouped = dup.groupby("chunk")
    chunks["chunk_poss"] = grouped.poss.sum().reindex(chunks.index).to_numpy(float)
    chunks["chunk_seasons"] = grouped.season.nunique().reindex(chunks.index).to_numpy(float)
    chunks["player_id"] = grouped.player_id.first().reindex(chunks.index).to_numpy()
    chunks = (chunks.join(label, on="player_id", how="inner")
              .dropna(subset=features + [column]).set_index("player_id"))
    share = chunks.chunk_poss / chunks.groupby(level=0).chunk_poss.transform("sum")
    chunks = chunks.assign(target=chunks[column].to_numpy(float),
                           weight=(share * chunks.possessions).to_numpy(float))
    return pd.concat([base, chunks])


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


def reweight_by_movement(train: pd.DataFrame, movement: pd.Series, floor: float = 0.0) -> pd.DataFrame:
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
    """
    factor = movement.reindex(train.index).fillna(0.0).to_numpy(float) + float(floor)
    weight = train.weight.to_numpy(float) * factor
    total = weight.sum()
    if total <= 0:
        raise ValueError("the movement weighting left no training weight at all")
    return train.assign(weight=weight * (train.weight.to_numpy(float).sum() / total))
