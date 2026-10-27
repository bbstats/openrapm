"""Step 2 of 5: the box-score prior, a booster that predicts a player's RAPM label from his box score.

Step 1 gives every player with enough possessions a label: his RAPM over every season except the rated one
(the season whose ratings are being built).  This step teaches LightGBM (gradient-boosted trees) to predict
that label from the season panel's box-score columns, then asks it about the rated season's own box score.
Step 3 pulls the ratings toward what it says.

Words used below
    booster       one fitted LightGBM model
    panel row     one player, one season, one side ("O" offence, "D" defence): his padded box-score rates,
                  role, career and body (features.py says what each column is)
    training row  a player's box score averaged over some of his seasons, carrying his label
    unseen        the seasons no training row may come from: the rated one (and, in the test setting, the
                  seasons either side of it)
    career row    the average over every season he has outside the unseen ones
    chunk row     the average over a run of 1, 2 or 3 consecutive seasons of his; consecutive among the
                  seasons he has, so a run may straddle the rated season
    part-season row   one of his training seasons so far: its panel row rebuilt from the games through the day a
                  share of its regular season had been played (data.panel.season_to_date), a quarter, a half and three
                  quarters when settings.PriorSettings.part_season_shares asks; none as shipped
    row weight    how much a training row counts in the fit
    player fold   one of five groups of players.  Each player's prior comes from the booster fitted without
                  his fold, so it cannot have learned "this box score is LeBron" and returned his label

Every row carries the player's career label, so what differs between his rows is only how much evidence the
box score rests on; two inputs say how much (`row_possessions`, the possessions behind the row, and
`row_seasons`, the seasons it covers, a part-season row its share of the regular season).  Part-season rows
(experiment 107; the owner, 2026-10-23: "want to support 'at any point in the season' really") teach the booster what
a box score of a season so far says, so a season can be rated on any day of it with `row_seasons` that day's share; a
booster trained on whole seasons only has seen no row under one season, and a quarter-season box score of a starter
looks to it like a whole season of a bench player.

With `settings.PriorSettings.row_labels` "outside" (experiment 79) each chunk
row carries instead his RAPM over the seasons outside its own, and the career row, whose box score covers them all,
is dropped: no row's label shares a game with its box score.  With "nearest" (experiment 83) each chunk row carries
his RAPM over only the season before it and the season after it (rapm.nearest_labels), through the same path, so its
label is the player at about the row's own age.  The weights:

    career row      weight = L
    chunk row c     weight = L x poss_c / (sum of poss over his chunk rows), his part-season rows among them

    L        the possessions behind his label (step 1)
    poss_c   the possessions behind chunk row c

so every player's rows weigh 2L in total, half on the career row and half spread over the chunks.

Example: three seasons of 1,000, 2,000 and 3,000 possessions and a label resting on L = 6,000.  The chunk rows
are the three single seasons, the two two-season runs (3,000 and 5,000) and the whole run (6,000); their
possessions sum to 20,000, so they weigh 300, 600, 900, 900, 1,500 and 1,800, together 6,000, the career row's
weight.  All seven rows carry the same label.

With an era half-life (settings.PriorSettings.era_halflife; 0, every season alike, is shipped; notes/DECISIONS.md,
experiment 56) each row also counts its seasons' kernel, applied after the weights are rescaled to mean 1:

    kernel of season s    0.5 ^ (max(|s - rated| - start, 0) / half-life)
    kernel of a row       the possession-weighted mean of its seasons' kernels
    start                 the nearest season a fit may train on: 1 in the product setting, 2 in the test setting

so a row near the rated season counts fully and one half-life further away counts half, and a narrow kernel is
less data under the same leaf penalties.

Each booster (per side, one on every training row and one without each player fold; LightGBM regression with
the settings in params/booster_params_lgb1.json) solves:

    minimise over f   sum over training rows  v (label - f(x))^2  +  LightGBM's penalties on the leaves

    x      the row's inputs: the side's shipped list (features.SHIPPED), then row_possessions and row_seasons
    v      the row weight divided by the mean row weight (`fit_lightgbm`)
    f(x)   a sum of trees, each scaled by the learning rate; a tree splits on one input at a time (x_j < c)
           and fits, in each leaf, a linear model on the inputs its branch split on, ridge penalty
           linear_lambda (LightGBM's linear_tree)

Player folds: players are sorted by label and assigned to folds 0,1,2,3,4,4,3,2,1,0,0,1,..., so every fold gets the
same mix of good and bad players and leaving one out does not move the training mean (notes/DECISIONS.md, "The SPM
memorises long-career stars", 2026-09-14; SPM was the box-score prior's earlier name).

Bags (settings.PriorSettings.bags, 10 as shipped): a player's prior is the mean of K answers, each from boosters
fitted on their own resample of the players (`player_bootstrap`).  How each answer avoids the boosters that saw him
(settings.PriorSettings.prediction):

    "folds"        every resample split into the five player folds above: a bag answers for him with the booster
                   fitted without his fold, or with its full booster when its resample did not draw him (shipped)
    "out_of_bag"   one booster per resample, no folds: his prior is the mean of the boosters whose resample did not
                   draw him, about 37% of them, so 60 bags cost what the shipped 10 bags of 1 + 5 boosters do

With settings.PriorSettings.split_per_bag each resample splits its players into the folds its own random way
(`bag_split`), so a prior averages over K splits instead of one pattern (notes/DECISIONS.md, "The noise control read
closely": how the players are split moves a regular's total rating a median 0.05 per 100).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_SETS, MISSING_OK, PANEL_INPUTS, ROW_FEATURES, add_features
from .settings import PREDICTIONS

__all__ = ["average_over_seasons", "career_rows", "season_runs", "chunk_seasons", "chunk_season_sets",
           "training_rows", "balanced_player_folds",
           "LIGHTGBM_FIXED", "COLUMN_SHARES", "INPUT_CLIP", "ClippedBooster", "fit_lightgbm", "OutOfPlayerPrior",
           "fit_out_of_player", "BaggedPrior", "player_bootstrap", "PREDICTIONS", "BAG_DRAW_STEP", "bag_split",
           "OutOfBagPrior", "out_of_bag_weights", "fit_prior", "rated_rows", "predict", "game_rated_rows",
           "predict_chosen", "SidePrior", "era_kernel", "season_priors"]

# columns a training row adds for its own bookkeeping: the label it carries, its weight in the fit, the
# possessions behind the label and its era kernel.  None may share a name with an input (notes/DECISIONS.md,
# experiment 46: the row weight once overwrote the body-weight input, then called `weight`)
BOOKKEEPING = ("label", "row_weight", "label_possessions", "era_kernel")


def _refuse_clashes(feature_names) -> None:
    """Raise if an input shares a name with a bookkeeping column, which would silently overwrite it."""
    clash = sorted(set(BOOKKEEPING) & set(feature_names))
    if clash:
        raise ValueError(f"bookkeeping columns {clash} share a name with the prior's inputs; rename one")


def average_over_seasons(rows: pd.DataFrame, feature_names, by=None, panel_columns=PANEL_INPUTS) -> pd.DataFrame:
    """One row per player (or per group in `by`): the panel columns averaged by possessions, then derived.

        average of a column = sum over his rows of (possessions x value) / sum over his rows of possessions

    Average first, derive second: true shooting of the average season is not the average of each season's
    true shooting.

    Example: 1.0 steals per 100 above the league over 1,000 possessions and 2.5 over 3,000 average to
    (1,000 x 1.0 + 3,000 x 2.5) / 4,000 = 2.125.
    """
    columns = [name for name in panel_columns if name in rows.columns]
    key = rows.player_id if by is None else by
    weight = rows.possessions.groupby(key).sum()
    mean = rows[columns].mul(rows.possessions, axis=0).groupby(key).sum().div(weight, axis=0)
    optional = [name for name in columns if name in MISSING_OK]
    if optional:
        # a column a season may lack (features.MISSING_OK) is averaged over the seasons that have it, and is missing
        # for a row none of whose seasons has it; the sum above counted a missing season as an average one
        has = rows[optional].notna().mul(rows.possessions, axis=0).groupby(key).sum()
        total = rows[optional].mul(rows.possessions, axis=0).groupby(key).sum(min_count=1)
        mean[optional] = total.div(has.where(has > 0), axis=0)
    return add_features(mean, feature_names)


def _labels(label: pd.DataFrame, column: str) -> pd.DataFrame:
    """Step 1's table for one side, indexed by player_id: his label (`column`) and the possessions behind it
    (`label_possessions`)."""
    return label.set_index("player_id")[[column, "possessions"]].rename(columns={"possessions": "label_possessions"})


def career_rows(label: pd.DataFrame, rows: pd.DataFrame, column: str, feature_names,
                panel_columns=PANEL_INPUTS) -> pd.DataFrame:
    """One career row per labelled player, indexed by player_id; weight = the possessions behind his label."""
    out = average_over_seasons(rows, feature_names, panel_columns=panel_columns).join(_labels(label, column),
                                                                                      how="inner")
    out = out.dropna(subset=[name for name in out.columns if name not in MISSING_OK])
    return out.assign(label=out[column].to_numpy(float), row_weight=out.label_possessions.to_numpy(float))


def part_season_members(part_seasons: pd.DataFrame) -> pd.DataFrame:
    """The part-season rows as members of chunks of their own, one each, keyed "player:part<share>:season", their
    `season_share` (the share of the regular season the row's games hold) kept for `row_seasons`.

    Example: player 7's 2011 season so far at a quarter is the chunk "7:part0.25:2011".
    """
    key = (part_seasons.player_id.astype(str) + ":part" + part_seasons.share.map("{:g}".format) + ":"
           + part_seasons.season.astype(str))
    return part_seasons.assign(chunk=key.to_numpy())


def season_runs(rows: pd.DataFrame, chunk_sizes=(1, 2, 3)) -> pd.DataFrame:
    """`rows` repeated once per run each row belongs to, with the run's key ("player:size:start") in `chunk`.

    A run is `size` consecutive seasons among the ones `rows` holds for the player, in the order he played them.

    Example: seasons 2010, 2011 and 2013 give the pairs (2010, 2011) and (2011, 2013), keys "7:2:0" and "7:2:1"
    for player 7; the 2011 row appears in both.
    """
    ordered = rows.sort_values(["player_id", "season"]).reset_index(drop=True)
    ordered["season_number"] = ordered.groupby("player_id").cumcount()
    n_seasons = ordered.groupby("player_id").season_number.transform("max") + 1
    parts = []
    for size in chunk_sizes:
        for offset in range(int(size)):
            start = ordered.season_number - offset
            fits = (start >= 0) & (start + size <= n_seasons)
            part = ordered[fits]
            key = part.player_id.astype(str) + ":" + str(size) + ":" + start[fits].astype(str)
            parts.append(part.assign(chunk=key.to_numpy()))
    return pd.concat(parts, ignore_index=True)


def chunk_seasons(members: pd.DataFrame) -> pd.Series:
    """chunk key -> the seasons the chunk row averages, a frozenset; `members` is `season_runs`'s output."""
    return members.groupby("chunk").season.agg(lambda seasons: frozenset(int(season) for season in seasons))


def chunk_season_sets(rows: pd.DataFrame, chunk_sizes=(1, 2, 3)) -> set:
    """Every distinct set of seasons a chunk row of `rows` averages: the solves outside labels need
    (rapm.outside_labels)."""
    return set(chunk_seasons(season_runs(rows, chunk_sizes)))


def _outside_values(chunks: pd.DataFrame, members: pd.DataFrame, outside: dict, column: str) -> np.ndarray:
    """Each chunk row's label from his RAPM over the seasons outside its own: `outside` maps a set of seasons to the
    label table with those seasons left out; NaN where the player has no label there."""
    seasons = chunk_seasons(members).reindex(chunks.index)
    tables = {key: table.set_index("player_id")[column] for key, table in outside.items()}
    return np.array([tables[key].get(player, np.nan) for key, player in zip(seasons, chunks.player_id)], dtype=float)


def training_rows(label: pd.DataFrame, rows: pd.DataFrame, column: str, feature_names, chunk_sizes=(1, 2, 3),
                  panel_columns=PANEL_INPUTS, outside: dict | None = None,
                  part_seasons: pd.DataFrame | None = None) -> pd.DataFrame:
    """The career row and every chunk row of each labelled player, indexed by player_id.

    `label` is step 1's table for this side (player_id, `column`, possessions); `rows` the side's panel rows for
    the training seasons.  The row order (career rows by player, then chunk rows by run key) is what LightGBM
    is handed, so it is kept exactly.  Columns beside the averaged inputs: `column`, label_possessions, label,
    row_weight, row_possessions, row_seasons.

    `part_seasons`, the side's rows of the training seasons so far (data.panel.build_panel_so_far: each with `share`
    and `season_share`), adds a chunk row for each (`part_season_members`): its row_seasons is its season_share, its
    label the player's, and it shares the chunk rows' weight by its possessions.  None, as shipped, adds none, and the
    rows are those of before experiment 107 to the byte.

    With `outside` ({set of seasons: this side's label table with those seasons left out}, rapm.outside_labels),
    each chunk row is labelled with the player's RAPM over the seasons outside its own, so its box score and its
    label share no game; a chunk without such a label is dropped, and so is the career row, whose box score covers
    every season (notes/DECISIONS.md, experiment 79).  The weights stay the career label's: each player's chunk
    rows together weigh the possessions behind his career label.
    """
    _refuse_clashes(feature_names)
    per_player = rows.groupby("player_id")
    career = career_rows(label, rows, column, feature_names, panel_columns)
    career = career.assign(row_possessions=per_player.possessions.sum().reindex(career.index).to_numpy(float),
                           row_seasons=per_player.season.nunique().reindex(career.index).to_numpy(float))
    members = season_runs(rows, chunk_sizes)
    if part_seasons is not None:
        members = pd.concat([members.assign(season_share=1.0), part_season_members(part_seasons)], ignore_index=True)
    chunks = average_over_seasons(members, feature_names, by=members.chunk, panel_columns=panel_columns)
    grouped = members.groupby("chunk")
    chunks["row_possessions"] = grouped.possessions.sum().reindex(chunks.index).to_numpy(float)
    covered = grouped.season.nunique() if part_seasons is None else grouped.season_share.sum()
    chunks["row_seasons"] = covered.reindex(chunks.index).to_numpy(float)
    chunks["player_id"] = grouped.player_id.first().reindex(chunks.index).to_numpy()
    chunks = chunks.join(_labels(label, column), on="player_id", how="inner")
    if outside is not None:
        chunks[column] = _outside_values(chunks, members, outside, column)
    chunks = chunks.dropna(subset=[name for name in [*feature_names, column] if name not in MISSING_OK])
    chunks = chunks.set_index("player_id")
    share = chunks.row_possessions / chunks.groupby(level=0).row_possessions.transform("sum")
    chunks = chunks.assign(label=chunks[column].to_numpy(float),
                           row_weight=(share * chunks.label_possessions).to_numpy(float))
    return chunks if outside is not None else pd.concat([career, chunks])


def _player_means(label, players, weights):
    """(each row's player as a position 0, 1, 2, ... in sorted id order, each player's weighted mean label)."""
    keys, inverse = np.unique(players, return_inverse=True)
    total = np.bincount(inverse, weights=weights * label, minlength=keys.size)
    weight = np.bincount(inverse, weights=weights, minlength=keys.size)
    return inverse, total / np.maximum(weight, 1e-12)


def balanced_player_folds(train: pd.DataFrame, n_folds: int = 5, fold_assignment: int = 0) -> np.ndarray:
    """A fold per training row; all of a player's rows in one fold, every fold the same mix of labels.

    Players are sorted by their weighted mean label and assigned to folds 0,1,2,3,4,4,3,2,1,0,0,1,... in that
    order (`fold_assignment=0`, shipped).  `fold_assignment=k` with k >= 1 is the noise control: it measures
    how much the ratings move when nothing changes but which players share a fold.  Players are taken five at a
    time in label order and each group of five gets the folds 0-4 in a random order from numpy's
    default_rng(k), so every fold still gets one player from each group.  k = 1 reproduces notes/DECISIONS.md's run
    noise_regroup2.
    """
    inverse, mean = _player_means(train.label.to_numpy(float), train.index.to_numpy(),
                                  train.row_weight.to_numpy(float))
    order = np.argsort(mean, kind="stable")
    fold_of_player = np.empty(mean.size, dtype=int)
    if not fold_assignment:
        there_and_back = np.concatenate([np.arange(n_folds), np.arange(n_folds)[::-1]])
        fold_of_player[order] = there_and_back[np.arange(mean.size) % there_and_back.size]
    else:
        rng = np.random.default_rng(int(fold_assignment))
        for start in range(0, mean.size, n_folds):
            block = order[start:start + n_folds]
            fold_of_player[block] = rng.permutation(n_folds)[:len(block)]
    return fold_of_player[inverse]


# No row or column subsampling, one thread, deterministic: the same rows always give the same booster
# (notes/DECISIONS.md, experiment 44).  These win over anything in the settings file, except the share of the inputs
# each tree or each split is shown (`COLUMN_SHARES`), which a settings file may name (check 150); the draw is seeded by
# random_state, so the same rows still give the same booster.
LIGHTGBM_FIXED = dict(subsample=1.0, subsample_freq=0, colsample_bytree=1.0, n_jobs=1, deterministic=True,
                      force_row_wise=True, verbose=-1, random_state=0)
COLUMN_SHARES = ("colsample_bytree", "feature_fraction_bynode")
# A settings file may name "input_clip" q (2026-10-15, the Stockton look, notes/DECISIONS.md "Stockton above Jordan"):
# every input is clipped to the q and 1 - q quantiles of the fit's own training rows, in training and in prediction,
# so a leaf's straight line is never followed past the range its rows cover.  Not a LightGBM setting; none as shipped.
INPUT_CLIP = "input_clip"


@dataclass
class ClippedBooster:
    """A fitted booster that clips each input to the bounds of its training rows before it answers."""
    model: object
    lower: np.ndarray
    upper: np.ndarray

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.model.predict(np.clip(np.asarray(inputs, dtype=float), self.lower, self.upper))


def fit_lightgbm(params: dict, inputs: np.ndarray, label: np.ndarray, weights: np.ndarray,
                 kernel: np.ndarray | None = None, seed: int = 0):
    """One LightGBM fit with the weights rescaled to mean 1, so the leaf penalties mean the same at any scale.

    `kernel` (each row's era kernel, at most 1) multiplies the rescaled weights, so a row far from the rated season
    counts a share of a full row: a narrow kernel is less data, not the same data under lighter penalties.  With
    `params["input_clip"]` the fit is a `ClippedBooster` (`INPUT_CLIP`); missing inputs stay missing.  A nonzero
    `seed` replaces LightGBM's random_state, which draws the inputs each tree is shown (each bag's own draw,
    `BaggedPrior`).
    """
    import lightgbm
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.mean()
    if kernel is not None:
        weights = weights * np.asarray(kernel, dtype=float)
    clip = params.get(INPUT_CLIP)
    params = {name: value for name, value in params.items() if name != INPUT_CLIP}
    shares = {name: params[name] for name in COLUMN_SHARES if name in params}
    model = lightgbm.LGBMRegressor(**{**params, **LIGHTGBM_FIXED, **shares, **({"random_state": seed} if seed else {})})
    if not clip:
        return model.fit(inputs, label, sample_weight=weights)
    inputs = np.asarray(inputs, dtype=float)
    lower = np.nanquantile(inputs, float(clip), axis=0)
    upper = np.nanquantile(inputs, 1.0 - float(clip), axis=0)
    model.fit(np.clip(inputs, lower, upper), label, sample_weight=weights)
    return ClippedBooster(model, lower, upper)


@dataclass
class OutOfPlayerPrior:
    """The booster fitted on every training row, and one per fold fitted without that fold's players."""
    full: object
    fold_models: list
    held_out_players: list          # per fold model, the set of player ids it never saw
    inputs: list                    # its input columns, in the order it reads them


def fit_out_of_player(train: pd.DataFrame, inputs, params: dict, n_folds: int = 5,
                      fold_assignment: int = 0, kernel: np.ndarray | None = None, seed: int = 0) -> OutOfPlayerPrior:
    """1 + `n_folds` LightGBM fits on the training rows: the full fit, then one without each fold.

    `kernel` is each row's era kernel (`fit_lightgbm`); the folds are drawn from the row weights alone, so they do
    not depend on it.  `seed` as in `fit_lightgbm`.
    """
    matrix = train[inputs].to_numpy(float)
    label = train.label.to_numpy(float)
    weight = train.row_weight.to_numpy(float)
    kernel = None if kernel is None else np.asarray(kernel, dtype=float)
    prior = OutOfPlayerPrior(full=fit_lightgbm(params, matrix, label, weight, kernel, seed), fold_models=[],
                             held_out_players=[], inputs=list(inputs))
    if n_folds > 1:
        fold = balanced_player_folds(train, n_folds, fold_assignment)
        for left_out in range(n_folds):
            keep = fold != left_out
            prior.fold_models.append(fit_lightgbm(params, matrix[keep], label[keep], weight[keep],
                                                  None if kernel is None else kernel[keep], seed))
            prior.held_out_players.append(set(train.index[~keep].tolist()))
    return prior


# Bagging (notes/DECISIONS.md, experiment 78; the owner, 2026-10-17: "Better to average or to do bagging?"): K
# out-of-player priors, bag k fitted on a bootstrap of the players at LightGBM seed k, and a player's prior the
# mean of the K answers.  A bootstrap draws as many players as there are, with replacement; a player drawn n times
# keeps all his rows at n times their weight, a player not drawn is left out of that bag, so the bag's boosters never
# saw him and its full fit answers for him.  0 bags is the single booster at seed 0.
@dataclass
class BaggedPrior:
    """K out-of-player priors, each on its own bootstrap of the players; a prediction is their mean."""
    members: list
    inputs: list


def player_bootstrap(train: pd.DataFrame, bag: int, kernel: np.ndarray | None = None
                     ) -> tuple[pd.DataFrame, np.ndarray | None]:
    """(the rows of bag `bag`'s bootstrap of the players, each weight times the times its player was drawn; their era
    kernel).  Rows are indexed by player_id, as `training_rows` writes them."""
    players = train.index.unique().to_numpy()
    rng = np.random.default_rng([int(bag), 1093])
    drawn = pd.Series(rng.choice(players, size=players.size, replace=True)).value_counts()
    keep = train.index.isin(drawn.index)
    rows = train[keep].copy()
    rows["row_weight"] = rows.row_weight.to_numpy(float) * drawn.reindex(rows.index).to_numpy(float)
    return rows, None if kernel is None else np.asarray(kernel, dtype=float)[keep]


# bag k of draw d is drawn and fitted at k + BAG_DRAW_STEP x d (settings.PriorSettings.bag_draw), so draw 0 is the
# shipped bags and every draw's bags differ from every other's while a draw has fewer than 100 bags; check 155's
# random draws are the same numbers
BAG_DRAW_STEP = 100


def bag_split(fold_assignment: int, draw: int) -> int:
    """The fold assignment of the bag drawn at `draw` (bag + BAG_DRAW_STEP x bag_draw) when each bag splits its players
    its own random way: 1 + draw + 1000 x `fold_assignment`, a random split for every bag (`balanced_player_folds`).

    Example: the shipped draws with fold_assignment 0 give bags 0-9 the splits 1-10; fold_assignment 1 gives them
    1001-1010, another split for every bag.
    """
    return 1 + int(draw) + 1000 * int(fold_assignment)


@dataclass
class OutOfBagPrior:
    """One booster per bag, each on its own resample of the players; a player's prediction is the mean of the
    boosters whose resample did not draw him."""
    models: list
    drawn_players: list             # per booster, the set of player ids its resample drew
    inputs: list


def out_of_bag_weights(prior: OutOfBagPrior, player_ids) -> np.ndarray:
    """A row per player, a column per booster of `prior`: 1 / (the number of boosters that never saw him) where the
    booster's resample did not draw him, 0 where it did, so each row sums to 1 and a player with no training rows
    (a rookie) weighs every booster alike.

    Example: three boosters, a player drawn by the second only: 0.5, 0, 0.5.
    """
    unseen = np.array([[pid not in drawn for drawn in prior.drawn_players] for pid in player_ids], dtype=float)
    unseen = unseen.reshape(len(player_ids), len(prior.models))
    count = unseen.sum(axis=1)
    if (count == 0).any():
        raise ValueError(f"{int((count == 0).sum())} players were drawn by all {len(prior.models)} bags, so no "
                         "booster answers for them without having seen them; use more bags")
    return unseen / count[:, None]


def fit_prior(train: pd.DataFrame, inputs, params: dict, n_folds: int = 5, fold_assignment: int = 0,
              kernel: np.ndarray | None = None, bags: int = 0, prediction: str = "folds",
              split_per_bag: bool = False, bag_draw: int = 0) -> OutOfPlayerPrior | BaggedPrior | OutOfBagPrior:
    """`fit_out_of_player` with 0 bags; with K bags, bag k on `player_bootstrap(train, k')` at seed k', where k' = k +
    BAG_DRAW_STEP x `bag_draw`: a `BaggedPrior` of out-of-player priors, each split by `fold_assignment` or, with
    `split_per_bag`, by its own `bag_split`; with `prediction` "out_of_bag", an `OutOfBagPrior` of the bags' full
    boosters alone (the module docstring)."""
    if prediction not in PREDICTIONS:
        raise ValueError(f"prediction is one of {PREDICTIONS}, not {prediction!r}")
    if not bags:
        if prediction != "folds" or split_per_bag or bag_draw:
            raise ValueError("out-of-bag prediction, a split per bag and another draw of the bags all need bags")
        return fit_out_of_player(train, inputs, params, n_folds, fold_assignment, kernel)
    if int(bag_draw) and int(bags) > BAG_DRAW_STEP:
        raise ValueError(f"another draw of the bags reaches at most {BAG_DRAW_STEP} bags, not {bags}")
    members, drawn = [], []
    for bag in range(int(bags)):
        draw = bag + BAG_DRAW_STEP * int(bag_draw)
        rows, bag_kernel = player_bootstrap(train, draw, kernel)
        if prediction == "out_of_bag":
            members.append(fit_lightgbm(params, rows[inputs].to_numpy(float), rows.label.to_numpy(float),
                                        rows.row_weight.to_numpy(float), bag_kernel, draw))
            drawn.append(set(rows.index.unique().tolist()))
            continue
        split = bag_split(fold_assignment, draw) if split_per_bag else fold_assignment
        members.append(fit_out_of_player(rows, inputs, params, n_folds, split, bag_kernel, seed=draw))
    if prediction == "out_of_bag":
        return OutOfBagPrior(models=members, drawn_players=drawn, inputs=list(inputs))
    return BaggedPrior(members=members, inputs=list(inputs))


def rated_rows(season_rows: pd.DataFrame, feature_names, share: float = 1.0) -> pd.DataFrame:
    """The rated season's own panel rows, derived, with row_possessions = his possessions and row_seasons = `share`:
    1 for a whole season; the share of the regular season played so far for a rating of the season so far
    (`openrapm folds --cut`), which only boosters trained with part-season rows are told."""
    rows = add_features(season_rows.copy(), feature_names)
    return rows.assign(row_possessions=rows.possessions.to_numpy(float), row_seasons=float(share))


def predict(prior: OutOfPlayerPrior | BaggedPrior | OutOfBagPrior, rows: pd.DataFrame) -> np.ndarray:
    """Each player's prediction from the fold model that never saw him; for a `BaggedPrior`, the mean over its bags;
    for an `OutOfBagPrior`, the mean of the boosters whose resample did not draw him (`out_of_bag_weights`).

    A player with no training rows at all (a rookie, or a player whose other seasons earned no label) is in no
    fold, so the full fit answers for him.
    """
    if isinstance(prior, BaggedPrior):
        return np.mean([predict(member, rows) for member in prior.members], axis=0)
    if isinstance(prior, OutOfBagPrior):
        matrix = rows[prior.inputs].to_numpy(float)
        answers = np.column_stack([model.predict(matrix) for model in prior.models])
        return (answers * out_of_bag_weights(prior, rows.player_id.to_numpy())).sum(axis=1) + 0.0
    matrix = rows[prior.inputs].to_numpy(float)
    out = prior.full.predict(matrix)
    ids = rows.player_id.to_numpy()
    for model, held_out in zip(prior.fold_models, prior.held_out_players):
        mine = np.fromiter((pid in held_out for pid in ids), dtype=bool, count=len(ids))
        if mine.any():
            # + 0.0 turns a -0.0 prediction into 0.0.  The published build averaged each player's predictions
            # over its fold assignments, (0.0 + prediction) / 1 with the one it shipped, and that sum did the
            # same; every other value is unchanged
            out[mine] = model.predict(matrix[mine]) + 0.0
    return out


def game_rated_rows(rows: pd.DataFrame, feature_names, least_possessions: float = 1.0,
                    least_share: float = 0.0) -> pd.DataFrame:
    """The rated season's game rows (game_prior.game_rows: the state before each game), derived, with row_possessions
    = his possessions before the game and row_seasons = the share of the regular season played by that day (1 in
    the playoffs), each held at least at the smallest the training rows reach: a linear leaf follows its line past
    the range its rows cover, and a first game's row has no possessions at all."""
    out = add_features(rows.copy(), feature_names)
    return out.assign(row_possessions=np.maximum(out.possessions.to_numpy(float), float(least_possessions)),
                      row_seasons=np.maximum(out.season_progress.to_numpy(float), float(least_share)))


def predict_chosen(prior: OutOfPlayerPrior | BaggedPrior | OutOfBagPrior, rows: pd.DataFrame) -> np.ndarray:
    """`predict`'s answers, each booster asked only about the rows it answers for: in each bag a player's rows go to
    the fold booster that never saw him, or to the bag's full booster.  The same numbers at one pass over the rows a
    bag instead of two; a season's game rows are about thirty times its player rows."""
    if isinstance(prior, BaggedPrior):
        return np.mean([predict_chosen(member, rows) for member in prior.members], axis=0)
    if isinstance(prior, OutOfBagPrior):
        return predict(prior, rows)
    matrix = rows[prior.inputs].to_numpy(float)
    ids = rows.player_id.to_numpy()
    chosen = np.full(len(ids), -1)
    for fold, held_out in enumerate(prior.held_out_players):
        chosen[np.fromiter((pid in held_out for pid in ids), dtype=bool, count=len(ids))] = fold
    out = np.empty(len(ids))
    if (chosen < 0).any():
        out[chosen < 0] = prior.full.predict(matrix[chosen < 0])
    for fold, model in enumerate(prior.fold_models):
        if (chosen == fold).any():
            out[chosen == fold] = model.predict(matrix[chosen == fold]) + 0.0
    return out


@dataclass
class SidePrior:
    """One side's prior for the rated season: {player_id: prior} and the boosters that made it."""
    values: dict
    model: OutOfPlayerPrior | BaggedPrior | OutOfBagPrior
    feature_names: list


def era_kernel(seasons, rated: int, half_life: float, start: int = 1) -> np.ndarray:
    """How much each training season counts when `rated` is rated (experiment 56):

        kernel = 0.5 ^ (max(|season - rated| - start, 0) / half_life)

    `start` is the nearest season a fit may train on (1 in the product setting, 2 in the test setting, which also
    leaves out the seasons either side), so the nearest seasons count 1 in both.

    Example: half-life 8, start 2, rating 2015: 2013 and 2017 count 1, 2005 counts 0.5 ^ (8 / 8) = 0.5.
    """
    if half_life <= 0:
        raise ValueError(f"an era half-life must be positive, not {half_life}")
    distance = np.maximum(np.abs(np.asarray(seasons, dtype=float) - float(rated)) - float(start), 0.0)
    return 0.5 ** (distance / float(half_life))


def training_seasons(panel: pd.DataFrame | None, side: str, unseen, season: int, settings) -> pd.DataFrame | None:
    """A side's rows of `panel` outside the unseen seasons, each with its era kernel when the setting has a half-life
    (`era_kernel`; on this copy, never on the panel every rated season shares); None for no panel."""
    if panel is None:
        return None
    training = panel[(panel.side == side) & ~panel.season.isin(unseen)]
    half_life = float(settings.prior.era_halflife)
    if half_life > 0:
        training = training.assign(era_kernel=era_kernel(training.season, season, half_life,
                                                         settings.exclude_neighbours + 1))
    return training


def season_priors(labels: dict, panel: pd.DataFrame, season: int, unseen, settings, params: dict,
                  outside: dict | None = None, part_seasons: pd.DataFrame | None = None,
                  rated: pd.DataFrame | None = None, share: float = 1.0) -> dict:
    """{"O": SidePrior, "D": SidePrior} for one rated season.

    `labels` is step 1's {"O", "D"} tables; `panel` the season panel with the seasons a build may not train on
    removed (`data.seasons.drop_untrainable`) and only rows with possessions;
    `unseen` the seasons no training row may come from (the rated one, plus its neighbours in the test
    setting); `params` each side's LightGBM settings.  Inputs: the shipped list, then row_possessions, row_seasons.
    With an era half-life every training row also counts its era kernel (`era_kernel`), averaged over its seasons
    by possessions like an input.  With `outside` (rapm.outside_labels), every training row is labelled from the
    seasons outside its own (`training_rows`).  `part_seasons`, the panel of each season so far gated the same way
    (data.panel.trainable_panel_so_far, the shares asked for), adds the part-season rows of the training seasons.
    `rated`, the rated season's own panel rows when `panel` cannot hold them (the season in progress, which no fit
    may train on), and `share`, the share of its regular season played (`rated_rows`), rate a season so far.
    """
    half_life = float(settings.prior.era_halflife)
    columns = [*PANEL_INPUTS, "era_kernel"] if half_life > 0 else PANEL_INPUTS
    out = {}
    for side, column in (("O", "offense"), ("D", "defense")):
        names = list(FEATURE_SETS[settings.prior.feature_set][side])
        training = training_seasons(panel, side, unseen, season, settings)
        side_outside = None if outside is None else {key: tables[side] for key, tables in outside.items()}
        train = training_rows(labels[side], training, column, names, settings.prior.chunk_sizes, columns,
                              side_outside, training_seasons(part_seasons, side, unseen, season, settings))
        kernel = train.era_kernel.to_numpy(float) if half_life > 0 else None
        model = fit_prior(train, names + ROW_FEATURES, params[side], settings.prior.n_folds,
                          settings.prior.fold_assignment, kernel, settings.prior.bags, settings.prior.prediction,
                          settings.prior.split_per_bag, settings.prior.bag_draw)
        source = panel if rated is None else rated
        rows = rated_rows(source[(source.side == side) & (source.season == season)], names, share)
        out[side] = SidePrior(values=dict(zip(rows.player_id.to_numpy(), predict(model, rows))), model=model,
                              feature_names=names)
    return out
