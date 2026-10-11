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
    row weight    how much a training row counts in the fit
    player fold   one of five groups of players.  Each player's prior comes from the booster fitted without
                  his fold, so it cannot have learned "this box score is LeBron" and returned his label

Every row carries the player's career label, so what differs between his rows is only how much evidence the
box score rests on; two inputs say how much (`chunk_poss`, the possessions behind the row, and
`chunk_seasons`, the seasons it covers).  The weights:

    career row      weight = L
    chunk row c     weight = L x poss_c / (sum of poss over his chunk rows)

    L        the possessions behind his label (step 1)
    poss_c   the possessions behind chunk row c

so every player's rows weigh 2L in total, half on the career row and half spread over the chunks.

Example: three seasons of 1,000, 2,000 and 3,000 possessions and a label resting on L = 6,000.  The chunk rows
are the three single seasons, the two two-season runs (3,000 and 5,000) and the whole run (6,000); their
possessions sum to 20,000, so they weigh 300, 600, 900, 900, 1,500 and 1,800, together 6,000, the career row's
weight.  All seven rows carry the same label.

Each booster (per side, one on every training row and one without each player fold; LightGBM regression with
the settings in params/booster_params_lgb1.json) solves:

    minimise over f   sum over training rows  v (label - f(x))^2  +  LightGBM's penalties on the leaves

    x      the row's inputs: the side's shipped list (features.SHIPPED), then chunk_poss and chunk_seasons
    v      the row weight divided by the mean row weight (`fit_lightgbm`)
    f(x)   a sum of trees, each scaled by the learning rate; a tree splits on one input at a time (x_j < c)
           and fits, in each leaf, a linear model on the inputs its branch split on, ridge penalty
           linear_lambda (LightGBM's linear_tree)

Player folds: players are sorted by label and assigned to folds 0,1,2,3,4,4,3,2,1,0,0,1,..., so every fold gets the
same mix of good and bad players and leaving one out does not move the training mean (DECISIONS.md, "The SPM
memorises long-career stars", 2026-09-14; SPM was the box-score prior's earlier name).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_SETS, PANEL_INPUTS, ROW_FEATURES, add_features

__all__ = ["average_over_seasons", "career_rows", "season_runs", "training_rows", "balanced_player_folds",
           "LIGHTGBM_FIXED", "fit_lightgbm", "OutOfPlayerPrior", "fit_out_of_player", "rated_rows", "predict",
           "SidePrior", "season_priors"]

# columns a training row adds for its own bookkeeping; none may share a name with an input (DECISIONS.md,
# experiment 46: the row weight once overwrote the body-weight input `weight`)
BOOKKEEPING = ("target", "row_weight", "possessions")


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
    weight = rows.poss.groupby(key).sum()
    mean = rows[columns].mul(rows.poss, axis=0).groupby(key).sum().div(weight, axis=0)
    return add_features(mean, feature_names)


def career_rows(label: pd.DataFrame, rows: pd.DataFrame, column: str, feature_names,
                panel_columns=PANEL_INPUTS) -> pd.DataFrame:
    """One career row per labelled player, indexed by player_id; weight = the possessions behind his label."""
    out = (average_over_seasons(rows, feature_names, panel_columns=panel_columns)
           .join(label.set_index("player_id")[[column, "possessions"]], how="inner").dropna())
    return out.assign(target=out[column].to_numpy(float), row_weight=out.possessions.to_numpy(float))


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


def training_rows(label: pd.DataFrame, rows: pd.DataFrame, column: str, feature_names, chunk_sizes=(1, 2, 3),
                  panel_columns=PANEL_INPUTS) -> pd.DataFrame:
    """The career row and every chunk row of each labelled player, indexed by player_id.

    `label` is step 1's table for this side (player_id, `column`, possessions); `rows` the side's panel rows for
    the training seasons.  The row order (career rows by player, then chunk rows by run key) is what LightGBM
    is handed, so it is kept exactly.  Columns beside the averaged inputs: `column`, possessions (behind the
    label), target, row_weight, chunk_poss, chunk_seasons.
    """
    _refuse_clashes(feature_names)
    labels = label.set_index("player_id")[[column, "possessions"]]
    per_player = rows.groupby("player_id")
    career = career_rows(label, rows, column, feature_names, panel_columns)
    career = career.assign(chunk_poss=per_player.poss.sum().reindex(career.index).to_numpy(float),
                           chunk_seasons=per_player.season.nunique().reindex(career.index).to_numpy(float))
    members = season_runs(rows, chunk_sizes)
    chunks = average_over_seasons(members, feature_names, by=members.chunk, panel_columns=panel_columns)
    grouped = members.groupby("chunk")
    chunks["chunk_poss"] = grouped.poss.sum().reindex(chunks.index).to_numpy(float)
    chunks["chunk_seasons"] = grouped.season.nunique().reindex(chunks.index).to_numpy(float)
    chunks["player_id"] = grouped.player_id.first().reindex(chunks.index).to_numpy()
    chunks = chunks.join(labels, on="player_id", how="inner")
    chunks = chunks.dropna(subset=list(feature_names) + [column]).set_index("player_id")
    share = chunks.chunk_poss / chunks.groupby(level=0).chunk_poss.transform("sum")
    chunks = chunks.assign(target=chunks[column].to_numpy(float),
                           row_weight=(share * chunks.possessions).to_numpy(float))
    return pd.concat([career, chunks])


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
    default_rng(k), so every fold still gets one player from each group.  k = 1 reproduces DECISIONS.md's run
    noise_regroup2.
    """
    inverse, mean = _player_means(train.target.to_numpy(float), train.index.to_numpy(),
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
# (DECISIONS.md, experiment 44).  These win over anything in the settings file.
LIGHTGBM_FIXED = dict(subsample=1.0, subsample_freq=0, colsample_bytree=1.0, n_jobs=1, deterministic=True,
                      force_row_wise=True, verbose=-1, random_state=0)


def fit_lightgbm(params: dict, inputs: np.ndarray, label: np.ndarray, weights: np.ndarray):
    """One LightGBM fit with the weights rescaled to mean 1, so the leaf penalties mean the same at any scale."""
    import lightgbm
    weights = np.asarray(weights, dtype=float)
    model = lightgbm.LGBMRegressor(**{**params, **LIGHTGBM_FIXED})
    return model.fit(inputs, label, sample_weight=weights / weights.mean())


@dataclass
class OutOfPlayerPrior:
    """The booster fitted on every training row, and one per fold fitted without that fold's players."""
    full: object
    fold_models: list
    held_out_players: list          # per fold model, the set of player ids it never saw
    inputs: list                    # its input columns, in the order it reads them


def fit_out_of_player(train: pd.DataFrame, inputs, params: dict, n_folds: int = 5,
                      fold_assignment: int = 0) -> OutOfPlayerPrior:
    """1 + `n_folds` LightGBM fits on the training rows: the full fit, then one without each fold."""
    matrix = train[inputs].to_numpy(float)
    label = train.target.to_numpy(float)
    weight = train.row_weight.to_numpy(float)
    prior = OutOfPlayerPrior(full=fit_lightgbm(params, matrix, label, weight), fold_models=[],
                             held_out_players=[], inputs=list(inputs))
    if n_folds > 1:
        fold = balanced_player_folds(train, n_folds, fold_assignment)
        for left_out in range(n_folds):
            keep = fold != left_out
            prior.fold_models.append(fit_lightgbm(params, matrix[keep], label[keep], weight[keep]))
            prior.held_out_players.append(set(train.index[~keep].tolist()))
    return prior


def rated_rows(season_rows: pd.DataFrame, feature_names) -> pd.DataFrame:
    """The rated season's own panel rows, derived, with chunk_poss = his possessions and chunk_seasons = 1."""
    rows = add_features(season_rows.copy(), feature_names)
    return rows.assign(chunk_poss=rows.poss.to_numpy(float), chunk_seasons=1.0)


def predict(prior: OutOfPlayerPrior, rows: pd.DataFrame) -> np.ndarray:
    """Each player's prediction from the fold model that never saw him.

    A player with no training rows at all (a rookie, or a player whose other seasons earned no label) is in no
    fold, so the full fit answers for him.
    """
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


@dataclass
class SidePrior:
    """One side's prior for the rated season: {player_id: prior} and the boosters that made it."""
    values: dict
    model: OutOfPlayerPrior
    feature_names: list


def season_priors(labels: dict, panel: pd.DataFrame, season: int, unseen, settings, params: dict) -> dict:
    """{"O": SidePrior, "D": SidePrior} for one rated season.

    `labels` is step 1's {"O", "D"} tables; `panel` the season panel with the seasons a build may not train on
    removed (`data.seasons.drop_untrainable`) and only rows with possessions;
    `unseen` the seasons no training row may come from (the rated one, plus its neighbours in the test
    setting); `params` each side's LightGBM settings.  Inputs: the shipped list, then chunk_poss, chunk_seasons.
    """
    out = {}
    for side, column in (("O", "offense"), ("D", "defense")):
        names = list(FEATURE_SETS[settings.prior.feature_set][side])
        training = panel[(panel.side == side) & ~panel.season.isin(unseen)]
        train = training_rows(labels[side], training, column, names, settings.prior.chunk_sizes)
        model = fit_out_of_player(train, names + ROW_FEATURES, params[side], settings.prior.n_folds,
                                  settings.prior.fold_assignment)
        rated = rated_rows(panel[(panel.side == side) & (panel.season == season)], names)
        out[side] = SidePrior(values=dict(zip(rated.player_id.to_numpy(), predict(model, rated))), model=model,
                              feature_names=names)
    return out
