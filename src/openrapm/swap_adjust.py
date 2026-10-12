"""Step 5 of 5: each team's credit re-split by what its lineup swaps say, every team's total kept.

After step 4, each player's rating says how good he was.  Lineup swaps in other seasons say which kinds of player
such ratings give too much or too little credit, compared with the teammates they swap with.  This step moves half
of that predicted miss onto each player and takes it back inside his own team, so only the split inside a team
changes (the owner's "team version", adopted 2026-10-01; notes/DECISIONS.md, "The swap adjustment with the spread held:
it passes": on the year-over-year test a paired t-statistic of -5.2, better in 39 of its 56 predictions, each
season 1998-2025 predicted from the season before and from the season after).

Words used below
    lineup swap   two lineups of one team in one season that share four players: the four kept players are
                  held fixed, so the difference between the two lineups' results is evidence about the fifth
                  player of each, `fifth_a` against `fifth_b` (swap_test.py builds them; a lineup's result is
                  pooled over every stint it played that season)
    residual      what the ratings did not predict about a swap, in points per 100 possessions:
                      r = (observed_a - observed_b) - (predicted_a - predicted_b)
                  observed: the lineup's points per 100 (scored on offence, allowed on defence); predicted: the
                  ten players' ratings plus a level refitted on the season with the ratings held fixed (the
                  intercept, home court, playoffs, playoff home court, garbage time, score margin, margin x share
                  of the game left, minutes into the period, its last two minutes, the clutch minutes:
                  swap_test.CONTEXTS["nofatigue"]), averaged over the lineup's stints by possessions
    information   each swap's weight, h = P_a P_b / (P_a + P_b), P the two lineups' possessions: a swap seen
                  for 2,000 possessions one way and 20 the other knows about as much as 20 possessions
    type features 22 numbers per player-season, each standardised over every player-season of the table that
                  the panel also has (mean 0, standard deviation 1): 20 inputs of the season panel
                  (data/panel.py): 13 padded rates per 100 (threes, twos and free throws made and missed,
                  offensive and defensive rebounds, assists, turnovers, steals, blocks, fouls), share of team
                  possessions, share of games started, garbage-time share, age, seasons before, height, weight;
                  then step 4's prior_off and prior_def (TYPE_FEATURES)
    type model    a ridge regression of each swap's residual on the difference of the two swapped players' type
                  features (the equation below), one per side
    player fold   player_id % 5.  A player's type prediction comes from the model fitted without any swap he is
                  in, so his own swaps never reach his own adjustment
    applied share kappa = 0.5: half of the type prediction is applied
    give-back     taking each team's total of kappa m back from its own players, so that the team's
                  possession-weighted total adjustment is zero (A c = 0 below)
    main team     the team a player played the most possessions for that season (data/roles.main_teams)

The type model, for one side, over the pairs of the training seasons of the same table (every season but the rated
one in the product setting; only seasons more than one away from it in the test setting; never a season still being
played):

    minimise   sum over pairs  h (r - (x_a - x_b)' beta)^2  +  lambda |beta|^2,   lambda = 1e-4 x mean diag(G)

    x_a, x_b   the type features of fifth_a and fifth_b in the season of the swap
    beta       the 22 coefficients; G = sum h (x_a - x_b)(x_a - x_b)', so the penalty means the same at any
               number of seasons
    m_i        = x_i' beta, player i's type prediction (raw sign: on defence, points allowed)

Then, per season and side:

    c = kappa m - A' (A A')^-1 A kappa m        the give-back: A c = 0
    c <- c - sum(w c) / sum(w)                  re-centred: the possession-weighted mean adjustment is zero
    rating' = rating + c, spread held within teams (`hold_spread_within`)

    A        teams x players: A[t, i] = player i's possessions for team t on that end over the sum of all of
             team t's players' possessions (five per possession), so each row sums to 1: a starter on the
             court for 75% of his team's possessions has 0.15.  Rescaling a row changes no c
    w        the step-4 table's possessions on that end (poss_off, poss_def)

c is the nearest point to kappa m, in squared distance, with every team's total zero: each player gives back in
proportion to his share of his team's possessions (notes/DECISIONS.md, experiment 41).  The adjustment widens the
ratings inside teams; holding the spread scales each player's distance from his main team's mean back, one factor
per season and side, until the spread inside teams equals the step-4 table's.  Team means stay, and so does the
order inside each team.  Team totals move only through players on two teams (a median 0.007 per 100 of the team's
possessions when adopted).

Why half, and why the spread is held (notes/DECISIONS.md, "The swap adjustment" and "The swap adjustment with the
spread held: it passes").  On the swap test, half of the type prediction kept most of the gain in ordering
teammates (net order +0.162, z +5.5; full strength +0.191) and tied on the gaps between them (z +0.7), where full
strength made the gaps worse (z +7.9).  Without the spread held, the ratings came out about 12% wider on offence
and 27% wider on defence, when they were already too wide, and failed the year-over-year test (team-game z +3.5).
With the spread held inside teams they passed (z -5.2, better in 39 of 56).

Example: 2026, product setting.  The type model predicts that step 4 gives Wembanyama's offence 2.45 per 100 too
little credit; half of it is +1.23; the Spurs' give-back leaves +0.54; holding the spread (2026's offensive factor
0.80, which also pulls his step-4 rating toward the Spurs' mean) leaves +0.20.  On defence, in points allowed, the
same steps read -1.68, -0.84, -0.94 and -0.39: his defensive rating rises 0.39.  Over the thirty seasons the
spread factors average 0.84 on offence and 0.86 on defence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import BOX_RATES
from .swap_test import Season, aligned, predict, side_rates

__all__ = ["SIDES", "RESIDUAL_CONTEXT", "PANEL_FEATURES", "PRIOR_FEATURES", "TYPE_FEATURES", "residual_pairs",
           "team_possessions", "type_features", "season_features", "player_folds", "PairGrams", "pair_grams",
           "SeasonSwaps", "season_swaps", "type_coefficients", "training_seasons", "type_predictions",
           "possession_shares", "give_back", "season_corrections", "recentre", "hold_spread_within",
           "hold_spread_within_teams", "adjusted_table", "adjust", "type_model_table"]

SIDES = ("offense", "defense")
# the full context without the time-on-court clocks, which came out as tired fives playing BETTER, an artefact
# (notes/DECISIONS.md, the swap test)
RESIDUAL_CONTEXT = "nofatigue"
# the season panel's inputs, all from the rated season.  NOT the games' part u: the swaps are read in the same
# season, so the part of a player's record the ridge held back is still in them, and a model given u learned to
# multiply it by about 2 -- a lighter ridge penalty wearing a "player type" label (notes/DECISIONS.md, the swap
# adjustment).  Order is load-bearing: it is the order of the coefficients.
PANEL_FEATURES = [*BOX_RATES, "team_possession_share", "games_started_share", "garbage_time_share", "age",
                  "seasons_before", "height_in", "weight_lb"]
# the box prior, so the model can say "the prior is too high for this kind of player" if the swaps do
PRIOR_FEATURES = ["prior_off", "prior_def"]
TYPE_FEATURES = PANEL_FEATURES + PRIOR_FEATURES


# ------------------------------------------------------------------------------------------- the swaps
def residual_pairs(season: Season, ratings: pd.DataFrame, context: str = RESIDUAL_CONTEXT) -> dict:
    """{side: DataFrame(team, fifth_a, fifth_b, h, r)}: every swap of the season, its information and residual.

    `ratings` is (player_id, o, d) in raw sign for this season; a player it lacks is rated 0.  On defence r is in
    points ALLOWED: a positive r says fifth_a's lineup allowed more than predicted, so the ratings rate fifth_a
    too well against fifth_b.
    """
    o, d = aligned(season.ids, ratings)
    stint_prediction, _ = predict(season, o, d, context)
    rates = side_rates(season, stint_prediction, o, d)
    out = {}
    for side in SIDES:
        observed, predicted, possessions, _ = rates[side]
        pairs = season.pairs[side]
        lineup_a, lineup_b = pairs.lineup_a.to_numpy(), pairs.lineup_b.to_numpy()
        information = (possessions[lineup_a] * possessions[lineup_b]
                       / (possessions[lineup_a] + possessions[lineup_b]))
        residual = (observed[lineup_a] - observed[lineup_b]) - (predicted[lineup_a] - predicted[lineup_b])
        out[side] = pd.DataFrame({"team": pairs.team.to_numpy(), "fifth_a": pairs.fifth_a.to_numpy(),
                                  "fifth_b": pairs.fifth_b.to_numpy(), "h": information, "r": residual})
    return out


def team_possessions(season: Season) -> dict:
    """{side: DataFrame(player_id, team, poss)}: the possessions each player played for each team on that end."""
    out = {}
    for side in SIDES:
        lineups = season.tables[side]
        long = pd.DataFrame({"player_id": lineups[["p1", "p2", "p3", "p4", "p5"]].to_numpy().ravel(),
                             "team": np.repeat(lineups.team.to_numpy(), 5),
                             "poss": np.repeat(lineups.poss.to_numpy(), 5)})
        out[side] = long.groupby(["player_id", "team"], as_index=False).poss.sum()
    return out


# ------------------------------------------------------------------------------------------- the type model
def type_features(panel: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """TYPE_FEATURES per (player_id, season), each standardised over the player-seasons in both `panel` and `table`.

    `panel` is the whole season panel (offence rows are read: the panel's inputs are the same on both sides);
    `table` the step-4 table, which brings prior_off and prior_def.  Standardised with the population standard
    deviation (ddof 0).
    """
    offence = panel[panel.side == "O"][["player_id", "season", *PANEL_FEATURES]]
    features = offence.merge(table[["player_id", "season", *PRIOR_FEATURES]], on=["player_id", "season"],
                             how="inner")
    values = features[TYPE_FEATURES].astype(float)
    features[TYPE_FEATURES] = (values - values.mean()) / values.std(ddof=0)
    return features.set_index(["player_id", "season"])[TYPE_FEATURES]


def season_features(features: pd.DataFrame, season: int) -> pd.DataFrame:
    """One season's type features, indexed by player_id (empty when the season has none)."""
    if season in features.index.get_level_values("season"):
        return features.xs(season, level="season")
    return features.iloc[:0]


def player_folds(player_ids, n_folds: int) -> pd.Series:
    """player_id -> his fold, player_id % n_folds.  Example: player 2544 is in fold 4 of 5."""
    ids = np.unique(player_ids)
    return pd.Series(ids % n_folds, index=ids)


@dataclass
class PairGrams:
    """One season and side's share of the type model's normal equations.

    G, b            sum over its pairs of h dx dx' and h r dx, dx = x_a - x_b
    G_fold, b_fold  the same over the pairs that involve a player of each fold: what that fold's model leaves out
    n               pairs used (both players had type features)
    """
    G: np.ndarray
    b: np.ndarray
    G_fold: np.ndarray
    b_fold: np.ndarray
    n: int


def pair_grams(pairs: pd.DataFrame, features: pd.DataFrame, fold_of: pd.Series, n_folds: int) -> PairGrams:
    """`features`: one season's type features by player_id; `fold_of`: player_id -> fold.

    A pair is used only if both players have type features.
    """
    xa = features.reindex(pairs.fifth_a.to_numpy()).to_numpy(dtype=float)
    xb = features.reindex(pairs.fifth_b.to_numpy()).to_numpy(dtype=float)
    ok = ~(np.isnan(xa).any(axis=1) | np.isnan(xb).any(axis=1))
    dx, h, r = (xa - xb)[ok], pairs.h.to_numpy(dtype=float)[ok], pairs.r.to_numpy(dtype=float)[ok]
    fold_a = fold_of.reindex(pairs.fifth_a.to_numpy()).to_numpy()[ok]
    fold_b = fold_of.reindex(pairs.fifth_b.to_numpy()).to_numpy()[ok]
    n_features = dx.shape[1]
    gram_fold, rhs_fold = np.zeros((n_folds, n_features, n_features)), np.zeros((n_folds, n_features))
    for fold in range(n_folds):
        mine = (fold_a == fold) | (fold_b == fold)
        gram_fold[fold] = (dx[mine] * h[mine, None]).T @ dx[mine]
        rhs_fold[fold] = dx[mine].T @ (h[mine] * r[mine])
    return PairGrams(G=(dx * h[:, None]).T @ dx, b=dx.T @ (h * r), G_fold=gram_fold, b_fold=rhs_fold,
                     n=int(ok.sum()))


@dataclass
class SeasonSwaps:
    """What step 5 keeps of one season: per side, its swaps (`residual_pairs`), each player's possessions per
    team (`team_possessions`) and its share of the type model's normal equations (`pair_grams`)."""
    pairs: dict
    team_possessions: dict
    grams: dict


def season_swaps(season: Season, ratings: pd.DataFrame, features: pd.DataFrame, fold_of: pd.Series,
                 n_folds: int) -> SeasonSwaps:
    """One season's swaps against its own step-4 ratings (raw sign), with `features` that season's."""
    pairs = residual_pairs(season, ratings, RESIDUAL_CONTEXT)
    return SeasonSwaps(pairs=pairs, team_possessions=team_possessions(season),
                       grams={side: pair_grams(pairs[side], features, fold_of, n_folds) for side in SIDES})


def type_coefficients(grams: list, fold: int | None = None, penalty: float = 1e-4) -> np.ndarray:
    """beta from the summed grams, leaving out `fold`'s pairs (None: every pair).

        (G + lambda I) beta = b,   lambda = penalty x mean diag(G)
    """
    gram = sum(part.G - (part.G_fold[fold] if fold is not None else 0.0) for part in grams)
    rhs = sum(part.b - (part.b_fold[fold] if fold is not None else 0.0) for part in grams)
    ridge = float(penalty) * float(np.mean(np.diag(gram)))
    return np.linalg.solve(gram + ridge * np.eye(len(rhs)), rhs)


def training_seasons(season: int, seasons, trainable: set, exclude_near: int) -> list:
    """The seasons whose swaps train `season`'s type model, in the order given.

    Example: exclude_near 1 and seasons 2023-2026 all trainable: rating 2026 trains on 2023 and 2024.
    """
    return [other for other in seasons if abs(other - season) > exclude_near and other in trainable]


def _fold_predictions(features: pd.DataFrame, grams: list, fold_of: pd.Series, settings) -> pd.Series:
    """Each player's type prediction from the model fitted without his fold's pairs."""
    prediction = pd.Series(np.nan, index=features.index, dtype=float)
    for fold in range(settings.n_folds):
        beta = type_coefficients(grams, fold, settings.type_model_penalty)
        mine = fold_of.reindex(features.index).to_numpy() == fold
        prediction[mine] = features[mine].to_numpy() @ beta
    return prediction


def type_predictions(features: pd.DataFrame, swaps: dict, fold_of: pd.Series, trainable: set, settings) -> dict:
    """{(season, side): player_id -> m, the type prediction in raw sign}.

    `swaps` is {season: SeasonSwaps}; `settings` is settings.SwapSettings.  A season with no training season
    (a table of one season, say) gets a prediction of 0 for everyone, so step 5 leaves its ratings as they are,
    bit for bit (`hold_spread_within` returns ratings it did not move unchanged).
    """
    out = {}
    for season in swaps:
        features_now = season_features(features, season)
        train = training_seasons(season, list(swaps), trainable, settings.exclude_near)
        for side in SIDES:
            if not train:
                out[(season, side)] = pd.Series(0.0, index=features_now.index)
                continue
            out[(season, side)] = _fold_predictions(features_now, [swaps[other].grams[side] for other in train],
                                                    fold_of, settings)
    return out


# ------------------------------------------------------------------------------------------- the give-back
def possession_shares(pairs: pd.DataFrame, team_poss: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(ids, A): every player with possessions or a swap, and A[t, i] = his possessions for team t over the sum of
    all of team t's players' possessions (each row sums to 1; the module docstring).

    A player with a swap but no possessions on this end has a column of zeros and gives nothing back.
    """
    ids = np.unique(np.concatenate([team_poss.player_id.to_numpy(dtype=np.int64),
                                    pairs.fifth_a.to_numpy(dtype=np.int64), pairs.fifth_b.to_numpy(dtype=np.int64)]))
    teams, team_index = np.unique(team_poss.team.to_numpy(), return_inverse=True)
    shares = np.zeros((len(teams), len(ids)))
    np.add.at(shares, (np.asarray(team_index).ravel(),
                       np.searchsorted(ids, team_poss.player_id.to_numpy(dtype=np.int64))),
              team_poss.poss.to_numpy(dtype=float))
    shares /= shares.sum(axis=1, keepdims=True)
    return ids, shares


def give_back(ids: np.ndarray, shares: np.ndarray, mean: pd.Series) -> pd.Series:
    """player_id -> c = m - A'(AA')^-1 A m, the nearest point to `mean` with every team's total zero.

    `mean` is kappa x the type prediction (a player it lacks: 0).

    Example: two players on one team with shares 0.6 and 0.4 and m = (+1.0, 0.0): the team total 0.6 must go,
    and c = m - t (0.6, 0.4) with t = 0.6 / (0.36 + 0.16) = 1.15, so c = (+0.31, -0.46) and 0.6 x 0.31 - 0.4 x 0.46
    = 0.
    """
    m = mean.reindex(ids).fillna(0.0).to_numpy(dtype=float)
    return pd.Series(m - shares.T @ np.linalg.solve(shares @ shares.T, shares @ m), index=ids)


def season_corrections(raw: pd.DataFrame, swaps: dict, predictions: dict, applied_share: float) -> pd.DataFrame:
    """The give-back of kappa m per season and side, as columns c_o, c_d beside `raw`'s rows (raw sign).

    `raw` is the step-4 table as (player_id, season, o, d); rows come out season by season, in `swaps`' order.
    A player of the table with neither possessions nor a swap gets 0.
    """
    parts = []
    for season, kept in swaps.items():
        corrections = {side: give_back(*possession_shares(kept.pairs[side], kept.team_possessions[side]),
                                       applied_share * predictions[(season, side)])
                       for side in SIDES}
        rows = raw[raw.season == season].copy()
        rows["c_o"] = rows.player_id.map(corrections["offense"]).fillna(0.0).to_numpy()
        rows["c_d"] = rows.player_id.map(corrections["defense"]).fillna(0.0).to_numpy()
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def recentre(rows: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """c <- c - sum(w c) / sum(w) per season, for c_o with weights["c_o"] and c_d with weights["c_d"], so each
    season's possession-weighted mean rating does not move."""
    rows = rows.copy()
    seasons = rows.season.to_numpy()
    for column, weight in weights.items():
        mean = (pd.Series(rows[column].to_numpy() * weight).groupby(seasons).sum()
                / pd.Series(weight).groupby(seasons).sum())
        rows[column] = rows[column] - rows.season.map(mean).to_numpy()
    return rows


# ------------------------------------------------------------------------------------------- the spread
def _weighted_spread(values: np.ndarray, centre: np.ndarray, weight: np.ndarray) -> float:
    """sum(w (x - centre)^2) / sum(w)."""
    return float(np.sum(weight * (values - centre) ** 2) / np.sum(weight))


def _team_means(values: np.ndarray, team_key: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Each player's team's weighted mean of `values`; a team with no weight keeps the player's own value."""
    frame = pd.DataFrame({"team": team_key, "weighted": weight * values, "weight": weight})
    sums = frame.groupby("team")[["weighted", "weight"]].transform("sum")
    total = sums.weight.to_numpy()
    return np.where(total > 0, sums.weighted.to_numpy() / np.where(total > 0, total, 1.0), values)


def hold_spread_within(adjusted: np.ndarray, base: np.ndarray, weight: np.ndarray,
                       team: np.ndarray) -> tuple[np.ndarray, float]:
    """One season and side: the adjusted ratings with the spread inside teams held to the base's, and the factor.

        held = mean_t(adjusted) + k (adjusted - mean_t(adjusted)),
        k    = sqrt( spread of base about mean_t(base) / spread of adjusted about mean_t(adjusted) )

    mean_t is the possession-weighted mean over the player's main team, and a spread is the possession-weighted
    mean of the squared distances from it.  The adjusted team mean is kept, so team means and the order inside
    each team do not move.  A player with no main team (NaN) keeps his adjusted number.  A weight that is missing
    or not positive is 0.  When the adjustment moved no one (adjusted equals base), there is no spread to hold:
    the ratings come back as they are, with k = 1, rather than as mean + 1 x (adjusted - mean), which can differ
    from them in the last bit.
    """
    adjusted, base, weight = (np.asarray(x, dtype=float) for x in (adjusted, base, weight))
    if np.array_equal(adjusted, base):
        return adjusted.copy(), 1.0
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
    team = pd.Series(team)
    known = team.notna().to_numpy()
    team_key = team.fillna(-1).to_numpy()
    mean_adjusted = np.where(known, _team_means(adjusted, team_key, weight), adjusted)
    mean_base = np.where(known, _team_means(base, team_key, weight), base)
    factor = np.sqrt(_weighted_spread(base, mean_base, weight) / _weighted_spread(adjusted, mean_adjusted, weight))
    return mean_adjusted + factor * (adjusted - mean_adjusted), float(factor)


def hold_spread_within_teams(rows: pd.DataFrame, weights: dict, team: np.ndarray) -> tuple[pd.DataFrame, list]:
    """`hold_spread_within` for every season and side; the corrections c_o, c_d become held - base.

    Returns (rows, [one {season, side, factor} per season and side]).
    """
    rows = rows.copy()
    factors = []
    for side, column, base_column in (("offense", "c_o", "o"), ("defense", "c_d", "d")):
        weight = weights[column]
        for season in sorted(rows.season.unique()):
            mine = (rows.season == season).to_numpy()
            base = rows[base_column].to_numpy()[mine]
            held, factor = hold_spread_within(base + rows[column].to_numpy()[mine], base, weight[mine], team[mine])
            rows.loc[mine, column] = held - rows[base_column].to_numpy()[mine]
            factors.append(dict(season=season, side=side, factor=factor))
    return rows, factors


# ------------------------------------------------------------------------------------------- the table
def adjusted_table(table: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    """The step-4 table with each rating moved by its correction, in the published schema plus c_off, c_def.

    c_off, c_def are positive-good: rating_off += c_off, rating_def += c_def.  The prior and u columns are step
    4's, so after this step a rating is no longer prior + u: the difference is c.
    """
    out = table.merge(corrections, on=["player_id", "season"], how="left").fillna({"c_o": 0.0, "c_d": 0.0})
    out["rating_off"] = out.rating_off + out.c_o
    out["rating_def"] = out.rating_def - out.c_d
    out["rating_total"] = out.rating_off + out.rating_def
    out["offense"], out["defense"] = out.rating_off, -out.rating_def
    out["c_off"], out["c_def"] = out.c_o, -out.c_d
    return out.drop(columns=["c_o", "c_d"])


def adjust(table: pd.DataFrame, raw: pd.DataFrame, swaps: dict, predictions: dict, main_team: pd.Series,
           settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 5 on a step-4 table: the give-back, the re-centring and the spread held, applied.

    `raw` is `table` as (player_id, season, o, d) in raw sign (swap_test.ratings_in_raw_sign); `swaps` is
    {season: SeasonSwaps}; `predictions` is `type_predictions`; `main_team` maps (player_id, season) to his main
    team; `settings` is settings.SwapSettings.  Returns (the adjusted table, the spread factors).
    """
    rows = season_corrections(raw, swaps, predictions, settings.applied_share)
    key = pd.MultiIndex.from_arrays([rows.player_id, rows.season])
    indexed = table.set_index(["player_id", "season"])
    weights = {"c_o": indexed.poss_off.reindex(key).fillna(0).to_numpy(),
               "c_d": indexed.poss_def.reindex(key).fillna(0).to_numpy()}
    rows = recentre(rows, weights)
    rows, factors = hold_spread_within_teams(rows, weights, main_team.reindex(key).to_numpy(dtype=float))
    return adjusted_table(table, rows[["player_id", "season", "c_o", "c_d"]]), pd.DataFrame(factors)


def type_model_table(swaps: dict, trainable: set, penalty: float) -> pd.DataFrame:
    """The type model fitted on every trainable season's swaps, one row per type feature, for reading.

    Columns: feature, offense, defense_points_allowed, defense_positive_good (points per 100 for one standard
    deviation more of the feature than the teammate he swaps with, beyond what the ratings credited).
    """
    grams = {side: [swaps[season].grams[side] for season in swaps if season in trainable] for side in SIDES}
    out = pd.DataFrame({"feature": TYPE_FEATURES,
                        "offense": type_coefficients(grams["offense"], None, penalty),
                        "defense_points_allowed": type_coefficients(grams["defense"], None, penalty)})
    out["defense_positive_good"] = -out.defense_points_allowed
    return out
