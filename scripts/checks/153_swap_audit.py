"""Check 153, what the swap step does: is its gain a real misrating of certain kinds of players, or something else?

The owner, 2026-10-18: does step 5 fix a real, systematic misrating of certain kinds of players, or does its gain come
from something else?  Step 5 (src/openrapm/swap_adjust.py) moves credit between teammates by what a type model says
about each player: a ridge regression of each lineup swap's residual on the two swapped players' box-score, role,
body and prior numbers, fitted on other seasons; half of its prediction is applied, each team's total is given back
and the spread inside teams is held to step 4's.  notes/PLAN_swap_step.md sets out five explanations of its gain and
the stage that separates each:

    1 real type misses     step 4 gives some kinds of player too little of their team's credit and the model finds
                           which: stages 1 and 3
    2 within-team shrink   holding the spread multiplies every player's distance from his team's mean by about 0.85,
                           and that shrink alone may be the gain: stage 2
    3 same-games luck      the box counts are read in the same games as the swaps: stage 4
    4 coaching context     who replaces whom, and when, is not random: stage 3's context refits
    5 undoing step 4       step 4 multiplies the box-score part by about 0.7, and step 5 may hand part of it back:
                           stage 5

Every stage is read-only: nothing here changes a shipped table.

Words used below
    system         a build at the test setting with its own folds.  `bags10`, the incumbent, reads
                   outputs/season_ratings_bags10_step4.parquet, outputs/season_ratings_bags10.parquet (its step 5) and
                   the folds outputs/within/bags10_early and bags10_within
    fold           one season rated from three quarters of its games, the fit games, and scored on the quarter held
                   back (data/within_folds.py): 30 seasons x 12 folds
    type features  step 5's 22 inputs (swap_adjust.TYPE_FEATURES), standardised with the whole table's means and
                   standard deviations, so a fold's player is read on the scale the type model was fitted on
    arm            a version of step 5, built the same way on a whole-season table and on every fold:
                       step5            as shipped
                       step5_nohold     half strength and the give-back, the spread inside teams not held
                       shrink_only      no type move: each player's distance from his main team's possession-weighted
                                        mean multiplied by the spread factor step5 used on the same table, season and
                                        side, the team means kept
                       step5_leak       (folds only) the type features of the whole season, held-back games included:
                                        the leak control, which should look better than step5
                       placebo_use_K    the shipped type model read on type features shuffled among the players with
                                        seed K: a move of the shipped size that knows nothing about the players
                       placebo_train_K  the type features shuffled in training as well: the type model refitted on noise
                       step5_other      the type model refitted on the folds of the seasons more than one away: each
                                        fold's fit-game features against its HELD-BACK games' swaps, so a box count
                                        never meets the results it was counted in (stage 4's question, as an arm)
                       step5_same       the same refit on each fold's FIT games' swaps: the shipped situation at the
                                        folds' size, the control for step5_other
                       step5_share037   the shipped type model at an applied share of 0.37 instead of 0.5: about
                                        step5_other's typical move, the control for "it only moves players less"
                       step5_single     the shipped type model without seasons played before this one, a career count
                                        (experiment 80, the owner, 2026-10-19: "it must ONLY come from the season in
                                        question")
                       step5_other_single  step5_other without it: experiment 80's candidate
    on a fold      every input step 5 reads comes from the fold's fit games: the box counts, role and playing time
                   from the fold's players file (rebuilt from the fit games by `openrapm folds`), the two prior
                   features from the fold's step-4 prior part, each player's possessions for each team and his main
                   team from the fit games.  The type model is the one the whole-season step 5 rates that season
                   with: other seasons more than one away (the test setting), their whole-season swaps against the
                   step-4 table.  Step 4 on a fold is check 151's: m x prior + g x u, re-centred
    paired reading for each season, the arm's error minus the other's; the mean of the differences, its standard
                   error across seasons, the paired t-statistic (`z`) and the seasons the arm is better in

Subcommands, one per stage:

    reproduce   stage 0: this check's step 5, handed whole seasons, must rebuild the system's step-5 table from its
                step-4 table to 1e-9, both from the build's own sources and from the fold files' whole-season rows
    whole       stage 2's tables: every arm on the whole-season step-4 table, written as
                outputs/season_ratings_audit_<system>_<arm>.parquet for the guard (63_yoy.py), the swap test
                (90_swap_test.py) and the top 20 (66_compare.py)
    folds       stages 1, 2 and 4 on every fold: first the type model's normal equations on the fit games' and the
                held-back games' swaps (grams_<season>.npz), then each arm scored by check 151's decider
                (decider_<season>.csv); one file per season under outputs/swap_audit/<system>/, so a stopped run
                resumes
    decider     stages 1-2's readings: each arm against step 4, step 5 against shrink_only and the placebos, and the
                noise control beside them
    types       stage 3: the type model's coefficients with their season-jackknife standard errors, by era, on odd and
                even seasons, refitted on swaps rebuilt three ways; and how far the odd- and even-season models agree
                player by player
    luck        stage 4: the type model fitted on the folds' fit-game swaps and on their held-back-game swaps
    whose       stage 5: the type model fitted against five rating tables' swap residuals

    python scripts/checks/153_swap_audit.py reproduce --system=bags10
    python scripts/checks/153_swap_audit.py folds --system=bags10 --first=2024 --last=2024
    python scripts/checks/153_swap_audit.py decider --systems=bags10,noise_bags10
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm import prior_shrink
from openrapm import swap_adjust as sa
from openrapm import swap_test as sw
from openrapm.commands.console import one_thread, use_utf8
from openrapm.data.config import load_config, resolve
from openrapm.data.designs import DesignCache
from openrapm.data.panel import panel_and_trainable_seasons
from openrapm.data.roles import main_teams
from openrapm.data.stints import load_season_stints
from openrapm.data.within_folds import fold_games
from openrapm.features import BOX_RATES
from openrapm.prior_shrink import multipliers_by_season
from openrapm.settings import TEST

# the test setting: the type model never trains on the rated season or its neighbours.  Step 5 is off since experiment
# 82 (adopted 2026-10-20); this check audits it at the half strength it was built and shipped with
SWAP = replace(TEST.swap, applied_share=0.5)
TOLERANCE = 1e-9             # stage 0: the largest move of a rating the reproduction allows
STEP4_TOLERANCE = 1e-12      # a fold's step 4 here against check 151's own
SEEDS = (1, 2, 3, 4, 5)      # the placebo arms' shuffles, read as their mean (the plan)
ERAS = ((1997, 2006), (2007, 2016), (2017, 2026))
WEIGHT_CAP = 0.95            # stage 3: each swap's information capped at this quantile of its season and side
WHOLE_SEASON_KEY = 1000      # a whole season's shuffle key, beside a fold's (season, repeat, fold)
SMALLER_SHARE = 0.37         # the arm step5_share037: about step5_other's typical move (smaller_share)
# experiment 80's type features: the shipped 22 without seasons played before this one, a career count (the owner,
# 2026-10-19: "it must ONLY come from the season in question")
# step 5 as it shipped until experiment 80, the version this check audits: 22 type features, seasons played
# before this one among them (swap_adjust.TYPE_FEATURES has 21 since)
AUDITED_PANEL = [*BOX_RATES, "team_possession_share", "games_started_share", "garbage_time_share", "age",
                 "seasons_before", "height_in", "weight_lb"]
AUDITED_FEATURES = AUDITED_PANEL + sa.PRIOR_FEATURES
SINGLE_SEASON = [feature for feature in AUDITED_FEATURES if feature != "seasons_before"]
# experiment 87 (`folds --age-curve`): age read as a curve as well, (age - 27)^2 beside age, because the ratings
# over-credit the youngest and the oldest players inside teams (notes/DECISIONS.md, "The ratings' miss by age inside
# teams"); the cache then sits in outputs/swap_audit/<system>_agecurve
AGE_CURVE = "age_squared"
AGE_CENTRE = 27.0
PLAIN_RAPM = "season_ratings_plainrapm3000"     # 90_swap_test.py's plain single-season RAPM, penalty 3,000
NOISE = {"bags10": "noise_bags10"}              # each system's noise control: the same build, players regrouped
PLAIN_NAMES = {"threes_made_per100": "made threes", "threes_missed_per100": "missed threes",
               "twos_made_per100": "made twos", "twos_missed_per100": "missed twos",
               "free_throws_made_per100": "made free throws", "free_throws_missed_per100": "missed free throws",
               "off_rebounds_per100": "offensive rebounds", "def_rebounds_per100": "defensive rebounds",
               "assists_per100": "assists", "turnovers_per100": "turnovers", "steals_per100": "steals",
               "blocks_per100": "blocks", "fouls_per100": "fouls",
               "team_possession_share": "share of team possessions played",
               "games_started_share": "share of games started",
               "garbage_time_share": "share of possessions in garbage time", "age": "age",
               "seasons_before": "seasons played before this one", "height_in": "height", "weight_lb": "weight",
               "prior_off": "box-score prior, offence", "prior_def": "box-score prior, defence"}


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # its dataclasses look their module up while being defined
    spec.loader.exec_module(module)
    return module


@dataclass
class Checks:
    """The decider (check 151) and the held-out audit it reads each fold's parts with (check 138)."""
    wowy: object
    audit: object

    @classmethod
    def load(cls) -> Checks:
        return cls(wowy=_load("wowy", "151_wowy.py"), audit=_load("heldout_audit", "138_heldout_audit.py"))


# ------------------------------------------------------------------------------------------- the whole seasons
def season_rows(cfg: dict, season: int) -> pd.DataFrame:
    """One season's stints as `swap_test.season_rows` builds them for step 5 (commands/build.collect_swaps)."""
    stints = load_season_stints(season, cfg, sw.STINT_COLUMNS)
    return sw.season_rows(stints, margin_clip=float(cfg.get("margin_clip", 25)))


def feature_scale(panel: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(means, population standard deviations) of the type features over every player-season of both the panel and
    the table: what swap_adjust.type_features standardises with."""
    offence = panel[panel.side == "O"][["player_id", "season", *AUDITED_PANEL]]
    both = offence.merge(table[["player_id", "season", *sa.PRIOR_FEATURES]], on=["player_id", "season"], how="inner")
    values = both[AUDITED_FEATURES].astype(float)
    return values.mean(), values.std(ddof=0)


def season_betas(grams: dict, trainable: set) -> dict:
    """{(season, side): [the coefficients of each player fold]}: the type model each season is rated with, from the
    grams of its training seasons in ascending order (swap_adjust.type_predictions); None for a season with none."""
    seasons = list(grams)
    out = {}
    for season in seasons:
        train = sa.training_seasons(season, seasons, trainable, SWAP.exclude_near)
        for side in sa.SIDES:
            parts = [grams[other][side] for other in train]
            out[(season, side)] = ([sa.type_coefficients(parts, fold, SWAP.type_model_penalty)
                                    for fold in range(SWAP.n_folds)] if train else None)
    return out


@dataclass
class TypeModel:
    """A system's whole-season step 5 at the test setting, rebuilt: what every arm and every fold reads."""
    name: str
    step4: pd.DataFrame          # the step-4 test table
    final: pd.DataFrame          # the step-5 test table as the build wrote it
    centre: pd.Series            # the type features' means and standard deviations over the table
    scale: pd.Series
    features: pd.DataFrame       # standardised, indexed (player_id, season)
    swaps: dict                  # {season: swap_adjust.SeasonSwaps} against the step-4 table
    trainable: set
    betas: dict                  # season_betas
    main_team: pd.Series         # (player_id, season) -> his main team, from the roles table

    def season_features(self, season: int) -> pd.DataFrame:
        return sa.season_features(self.features, season)


def with_age_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` with AGE_CURVE, (age - 27)^2, when the age curve is on (`use_age_curve`); unchanged otherwise."""
    if AGE_CURVE not in AUDITED_PANEL:
        return frame
    return frame.assign(**{AGE_CURVE: (frame.age.astype(float) - AGE_CENTRE) ** 2})


def use_age_curve() -> None:
    """Experiment 87: every audited list reads the age curve too, after age."""
    global AUDITED_PANEL, AUDITED_FEATURES, SINGLE_SEASON
    at = AUDITED_PANEL.index("age") + 1
    AUDITED_PANEL = [*AUDITED_PANEL[:at], AGE_CURVE, *AUDITED_PANEL[at:]]
    AUDITED_FEATURES = AUDITED_PANEL + sa.PRIOR_FEATURES
    SINGLE_SEASON = [feature for feature in AUDITED_FEATURES if feature != "seasons_before"]
    PLAIN_NAMES.setdefault(AGE_CURVE, "age, squared distance from 27")


def type_model(cfg: dict, name: str) -> TypeModel:
    """The system's tables, and its step 5 rebuilt as `openrapm build --setting test` builds it."""
    started = time.time()
    outputs = resolve(cfg, "outputs")
    step4 = pd.read_parquet(outputs / f"season_ratings_{name}_step4.parquet")
    final = pd.read_parquet(outputs / f"season_ratings_{name}.parquet")
    panel, trainable = panel_and_trainable_seasons(cfg)
    panel = with_age_curve(panel)
    centre, scale = feature_scale(panel, step4)
    features = sa.type_features(panel, step4, AUDITED_FEATURES)
    raw = sw.ratings_in_raw_sign(step4)
    fold_of = sa.player_folds(raw.player_id, SWAP.n_folds)
    swaps = {season: sa.season_swaps(sw.prepare(season_rows(cfg, season), season), raw[raw.season == season],
                                     sa.season_features(features, season), fold_of, SWAP.n_folds)
             for season in sorted(int(season) for season in raw.season.unique())}
    betas = season_betas({season: kept.grams for season, kept in swaps.items()}, trainable)
    main_team = main_teams(cfg).set_index(["player_id", "season"]).team_id
    print(f"  {name}: step 5's type model rebuilt over {len(swaps)} seasons ({time.time() - started:.0f}s)", flush=True)
    return TypeModel(name, step4, final, centre, scale, features, swaps, trainable, betas, main_team)


# ---------------------------------------------------------------------------------------- step 5, given inputs
def no_pairs() -> pd.DataFrame:
    """A season's swaps with none in them.  The give-back reads the swaps only for their players' ids, and every
    swapped player is in the lineups the possessions come from, so leaving them out changes nothing."""
    return pd.DataFrame({"fifth_a": np.zeros(0, dtype=np.int64), "fifth_b": np.zeros(0, dtype=np.int64)})


def type_prediction(features: pd.DataFrame, betas: list | None) -> pd.Series:
    """player_id -> m, each player's type prediction from the model of his player fold (player_id % 5), as
    swap_adjust.type_predictions gives it; 0 for everyone when the season has no model."""
    if betas is None:
        return pd.Series(0.0, index=features.index)
    prediction = pd.Series(np.nan, index=features.index, dtype=float)
    fold = features.index.to_numpy() % SWAP.n_folds
    for k, beta in enumerate(betas):
        mine = fold == k
        prediction[mine] = features[mine].to_numpy() @ beta
    return prediction


def step5_rows(table: pd.DataFrame, prediction: dict, team_poss: dict, main_team: pd.Series,
               hold: bool = True) -> tuple[pd.DataFrame, dict]:
    """Step 5 on one season's step-4 rows through swap_adjust as it ships: (the adjusted rows, {side: spread factor}).

    `prediction` is {side: player_id -> m}, `team_poss` {side: player_id, team, poss} and `main_team` (player_id,
    season) -> team.  Without `hold`, the give-back and the re-centring alone: the arm step5_nohold, factors 1.
    """
    season = int(table.season.iloc[0])
    raw = sw.ratings_in_raw_sign(table)
    swaps = {season: sa.SeasonSwaps(pairs={side: no_pairs() for side in sa.SIDES}, team_possessions=team_poss,
                                    grams={})}
    predictions = {(season, side): prediction[side] for side in sa.SIDES}
    if hold:
        adjusted, factors = sa.adjust(table, raw, swaps, predictions, main_team, SWAP)
        return adjusted, dict(zip(factors.side, factors.factor))
    rows = sa.season_corrections(raw, swaps, predictions, SWAP.applied_share)
    key = pd.MultiIndex.from_arrays([rows.player_id, rows.season])
    indexed = table.set_index(["player_id", "season"])
    rows = sa.recentre(rows, {"c_o": indexed.poss_off.reindex(key).fillna(0).to_numpy(),
                              "c_d": indexed.poss_def.reindex(key).fillna(0).to_numpy()})
    return sa.adjusted_table(table, rows[["player_id", "season", "c_o", "c_d"]]), {side: 1.0 for side in sa.SIDES}


def shrink_within(values: np.ndarray, weight: np.ndarray, team: np.ndarray, factor: float) -> np.ndarray:
    """mean_t + factor (x - mean_t), mean_t the possession-weighted mean of the player's main team: swap_adjust's
    hold_spread_within with its factor given.  A player with no main team, or on a team with no weight, keeps x."""
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
    known = ~np.isnan(team)
    frame = pd.DataFrame({"team": np.where(known, team, -1.0), "weighted": weight * values, "weight": weight})
    sums = frame.groupby("team")[["weighted", "weight"]].transform("sum")
    total = sums.weight.to_numpy()
    mean = np.where(known & (total > 0), sums.weighted.to_numpy() / np.where(total > 0, total, 1.0), values)
    return mean + factor * (values - mean)


def shrink_only(table: pd.DataFrame, factors: dict, main_team: pd.Series) -> pd.DataFrame:
    """The arm shrink_only: each player's distance from his main team's mean times `factors[side]`, nothing else."""
    out = table.copy()
    team = main_team.reindex(pd.MultiIndex.from_arrays([table.player_id, table.season])).to_numpy(dtype=float)
    for side, column, weight in (("offense", "rating_off", "poss_off"), ("defense", "rating_def", "poss_def")):
        out[column] = shrink_within(table[column].to_numpy(float), table[weight].to_numpy(float), team, factors[side])
    out["c_off"], out["c_def"] = out.rating_off - table.rating_off, out.rating_def - table.rating_def
    out["rating_total"] = out.rating_off + out.rating_def
    out["offense"], out["defense"] = out.rating_off, -out.rating_def
    return out


def shuffled(features: pd.DataFrame, seed: int, key: tuple) -> pd.DataFrame:
    """`features` with its rows dealt to its players in a random order drawn with the seed (seed, *key): each player
    carries another player's 22 numbers."""
    order = np.random.default_rng([int(seed), *(int(part) for part in key)]).permutation(len(features))
    return pd.DataFrame(features.to_numpy()[order], index=features.index, columns=features.columns)


def placebo_betas(model: TypeModel, seed: int) -> dict:
    """The type model refitted with every season's type features shuffled among its players (the seed): season_betas
    on the same swaps."""
    fold_of = sa.player_folds(model.step4.player_id, SWAP.n_folds)
    grams = {}
    for season, kept in model.swaps.items():
        features = shuffled(model.season_features(season), seed, (season, 0))
        grams[season] = {side: sa.pair_grams(kept.pairs[side], features, fold_of, SWAP.n_folds) for side in sa.SIDES}
    return season_betas(grams, model.trainable)


@dataclass
class SeasonInputs:
    """What step 5 reads for one whole season or one fold."""
    table: pd.DataFrame          # step-4 rows: player_id, season, rating_off, rating_def, prior_off, prior_def,
                                 # poss_off, poss_def (positive good), and whatever else the table carries
    features: pd.DataFrame       # the standardised type features, by player_id
    leak_features: pd.DataFrame | None   # the whole season's, for the leak control on a fold; None on a whole season
    team_poss: dict              # {side: player_id, team, poss}
    main_team: pd.Series         # (player_id, season) -> team
    key: tuple                   # seeds the shuffles: (season, repeat, fold), or (season, WHOLE_SEASON_KEY)


def arms(inputs: SeasonInputs, model: TypeModel, placebo: dict, refits: dict | None = None) -> tuple[dict, dict]:
    """({arm: the season's rows after it}, step5's spread factors): every arm of the module docstring.  `placebo` is
    {seed: placebo_betas}; `refits` {arm: (season_betas, the type features it reads)}, `refit_arms`."""
    season = int(inputs.table.season.iloc[0])
    shipped_betas = {side: model.betas[(season, side)] for side in sa.SIDES}

    def step5(features: pd.DataFrame, betas: dict = shipped_betas, hold: bool = True):
        prediction = {side: type_prediction(features, betas[side]) for side in sa.SIDES}
        return step5_rows(inputs.table, prediction, inputs.team_poss, inputs.main_team, hold)

    shipped, factors = step5(inputs.features)
    out = {"step5": shipped, "step5_nohold": step5(inputs.features, hold=False)[0],
           "shrink_only": shrink_only(inputs.table, factors, inputs.main_team)}
    if inputs.leak_features is not None:
        out["step5_leak"] = step5(inputs.leak_features)[0]
    for seed, trained in placebo.items():
        noise = shuffled(inputs.features, seed, inputs.key)
        out[f"placebo_use_{seed}"] = step5(noise)[0]
        out[f"placebo_train_{seed}"] = step5(noise, {side: trained[(season, side)] for side in sa.SIDES})[0]
    for arm, (betas, columns) in (refits or {}).items():
        out[arm] = step5(inputs.features[columns], {side: betas[(season, side)] for side in sa.SIDES})[0]
    return out, factors


def whole_inputs(model: TypeModel, season: int) -> SeasonInputs:
    """A whole season as the build hands it to step 5."""
    table = model.step4[model.step4.season == season].reset_index(drop=True)
    return SeasonInputs(table, model.season_features(season), None, model.swaps[season].team_possessions,
                        model.main_team, (season, WHOLE_SEASON_KEY))


def largest_gap(mine: pd.DataFrame, theirs: pd.DataFrame, columns=("rating_off", "rating_def")) -> float:
    """The largest absolute difference between two tables' columns over the players both have, matched on player
    and season (`unmatched` counts the rest)."""
    both = mine.merge(theirs, on=["player_id", "season"], suffixes=("", "_theirs"), how="inner")
    return float(max(np.nanmax(np.abs(both[column] - both[f"{column}_theirs"])) for column in columns))


def unmatched(mine: pd.DataFrame, theirs: pd.DataFrame) -> int:
    """The rows of either table the other lacks, matched on player and season."""
    both = mine.merge(theirs, on=["player_id", "season"], how="outer", indicator=True)
    return int((both._merge != "both").sum())


# --------------------------------------------------------------------------------------------------- the folds
def fold_factors(checks: Checks, folds: list) -> dict:
    """{season: (m_O, m_D, g_O, g_D)}, step 4's factors fitted as check 151 fits them (the 8 nearest fold seasons, the
    test rule), the defensive ones times the shipped defence factor."""
    wowy = checks.wowy
    table = multipliers_by_season(sorted({fold.season for fold in folds}), folds, "test", wowy.NEAREST,
                                  free_games=wowy.FREE_GAMES).set_index("season")
    return {int(season): (float(row.multiplier_off), wowy.DEFENCE_FACTOR * float(row.multiplier_def),
                          float(row.get("games_multiplier_off", 1.0)),
                          wowy.DEFENCE_FACTOR * float(row.get("games_multiplier_def", 1.0)))
            for season, row in table.iterrows()}


def fold_features(model: TypeModel, rows_offence: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """player_id -> the standardised type features of a fold's rated players, on the audited list's scale
    (swap_adjust.fold_type_features)."""
    return sa.fold_type_features(with_age_curve(rows_offence), table, model.centre, model.scale)


def fold_main_team(rows_offence: pd.DataFrame, season: int) -> pd.Series:
    """(player_id, season) -> the team he played the most possessions for in the fit games (the players file's
    team_id; -1, no team, left out)."""
    rows = rows_offence[rows_offence.team_id >= 0]
    index = pd.MultiIndex.from_arrays([rows.player_id.to_numpy(np.int64), np.full(len(rows), season, dtype=np.int64)])
    return pd.Series(rows.team_id.to_numpy(float), index=index)


@dataclass
class FoldWorld:
    """One fold made ready for step 5: its step-4 ratings and the fit games' and held-back games' lineups."""
    fold: object
    table: pd.DataFrame
    fit: sw.Season
    held: sw.Season


def fold_world(fold, directory: Path, rows: pd.DataFrame, factors: tuple) -> FoldWorld:
    fit_games, held_games = fold_games(directory, fold)
    game = rows.game_id.astype(str)
    fit = sw.prepare(rows[game.isin(fit_games).to_numpy()].reset_index(drop=True), fold.season)
    held = sw.prepare(rows[game.isin(held_games).to_numpy()].reset_index(drop=True), fold.season)
    return FoldWorld(fold, prior_shrink.shrink_fold(fold.rows, fold.season, factors), fit, held)


def check_step4(checks: Checks, world: FoldWorld, factors: tuple) -> None:
    """This check's step 4 on a fold must be check 151's, so every arm is read against the decider's own step 4."""
    fold = world.fold
    parts = checks.audit.RatingParts.of(fold)
    theirs = checks.wowy.step4_parts(checks.audit, fold, parts, factors)
    mine = world.table.set_index("player_id")
    gap = max(np.max(np.abs(mine.rating_off.reindex(fold.rows["O"].player_id).to_numpy() - sum(theirs["O"]))),
              np.max(np.abs(mine.rating_def.reindex(fold.rows["D"].player_id).to_numpy() - sum(theirs["D"]))))
    if not gap <= STEP4_TOLERANCE:
        raise SystemExit(f"{fold.stem}: step 4 here misses check 151's by {gap:.1e}")


def fold_inputs(model: TypeModel, world: FoldWorld) -> SeasonInputs:
    fold = world.fold
    return SeasonInputs(world.table, fold_features(model, fold.rows["O"], world.table),
                        model.season_features(fold.season), sa.team_possessions(world.fit),
                        fold_main_team(fold.rows["O"], fold.season), (fold.season, fold.repeat, fold.fold))


def as_stage(table: pd.DataFrame, fold) -> dict:
    """An arm's ratings as check 151's `extra` stages read them: positive good, in the fold's row order."""
    by_id = table.set_index("player_id")
    stage = {"O": by_id.rating_off.reindex(fold.rows["O"].player_id.to_numpy()).to_numpy(float),
             "D": by_id.rating_def.reindex(fold.rows["D"].player_id.to_numpy()).to_numpy(float)}
    if np.isnan(stage["O"]).any() or np.isnan(stage["D"]).any():
        raise SystemExit(f"{fold.stem}: an arm lost a rated player")
    return stage


@dataclass
class Equations:
    """The type model's normal equations over a set of swaps: G = sum h dx dx', b = sum h r dx (swap_adjust.PairGrams'
    G and b), n the swaps used."""
    G: np.ndarray
    b: np.ndarray
    n: int


def equations(pairs: pd.DataFrame, features: pd.DataFrame) -> Equations:
    """`pairs` is residual pairs (fifth_a, fifth_b, h, r); a swap is used only if both players have type features."""
    xa = features.reindex(pairs.fifth_a.to_numpy()).to_numpy(dtype=float)
    xb = features.reindex(pairs.fifth_b.to_numpy()).to_numpy(dtype=float)
    ok = ~(np.isnan(xa).any(axis=1) | np.isnan(xb).any(axis=1))
    dx, h, r = (xa - xb)[ok], pairs.h.to_numpy(dtype=float)[ok], pairs.r.to_numpy(dtype=float)[ok]
    return Equations((dx * h[:, None]).T @ dx, dx.T @ (h * r), int(ok.sum()))


GAMES = ("same", "other")       # stage 4: the swaps of the fold's fit games, or of its held-back games


def luck_grams(world: FoldWorld, features: pd.DataFrame) -> dict:
    """{(games, side): swap_adjust.PairGrams} on the fold's swaps against its step-4 ratings, the features from its
    fit games: games "same" reads the fit games' swaps (the shipped situation), "other" the held-back games'.  The
    player folds are kept, so a type model fitted on them can leave each player's own swaps out, as step 5's does."""
    raw = sw.ratings_in_raw_sign(world.table)
    fold_of = sa.player_folds(features.index.to_numpy(), SWAP.n_folds)
    out = {}
    for games, lineups in (("same", world.fit), ("other", world.held)):
        pairs = sa.residual_pairs(lineups, raw)
        for side in sa.SIDES:
            out[(games, side)] = sa.pair_grams(pairs[side], features, fold_of, SWAP.n_folds)
    return out


def save_grams(path: Path, grams: dict) -> None:
    arrays = {}
    for (games, side), gram in grams.items():
        key = f"{games}_{side}"
        arrays.update({f"{key}_G": gram.G, f"{key}_b": gram.b, f"{key}_G_fold": gram.G_fold,
                       f"{key}_b_fold": gram.b_fold, f"{key}_n": gram.n})
    np.savez(path, **arrays)


def load_grams(path: Path) -> dict:
    """{(games, side): PairGrams}, as `save_grams` wrote them."""
    stored = np.load(path)
    return {(games, side): sa.PairGrams(G=stored[f"{games}_{side}_G"], b=stored[f"{games}_{side}_b"],
                                        G_fold=stored[f"{games}_{side}_G_fold"],
                                        b_fold=stored[f"{games}_{side}_b_fold"], n=int(stored[f"{games}_{side}_n"]))
            for games in GAMES for side in sa.SIDES}


def kept_columns(gram: sa.PairGrams, columns) -> sa.PairGrams:
    """`gram` with only the type features `columns`: the normal equations of a type model fitted on them alone."""
    index = [AUDITED_FEATURES.index(column) for column in columns]
    return sa.PairGrams(G=gram.G[np.ix_(index, index)], b=gram.b[index],
                        G_fold=gram.G_fold[:, index][:, :, index], b_fold=gram.b_fold[:, index], n=gram.n)


def refit_betas(out_dir: Path, seasons, trainable: set, columns=tuple(AUDITED_FEATURES)) -> dict:
    """{arm: season_betas}: the type model refitted on the folds' swaps of the seasons more than one away, the
    features from the fit games, each player's own swaps left out: `step5_other` on the held-back games' swaps, so a
    box count meets results it was not counted in; `step5_same` on the fit games' own, the shipped situation at the
    folds' size.  On the type features `columns` alone.  Empty when a season's equations are missing."""
    paths = {season: out_dir / f"grams_{season}.npz" for season in sorted(seasons)}
    if not all(path.exists() for path in paths.values()):
        return {}
    grams = {season: load_grams(path) for season, path in paths.items()}
    return {f"step5_{games}": season_betas({season: {side: kept_columns(parts[(games, side)], columns)
                                                     for side in sa.SIDES}
                                            for season, parts in grams.items()}, trainable)
            for games in GAMES}


def smaller_share(betas: dict) -> dict:
    """The shipped type model applied at SMALLER_SHARE instead of half: the control for step5_other, which moves a
    player with 1,000+ possessions a median 0.26 per 100 against the shipped 0.35."""
    ratio = SMALLER_SHARE / SWAP.applied_share
    return {key: None if values is None else [ratio * beta for beta in values] for key, values in betas.items()}


def refit_arms(model: TypeModel, out_dir: Path, seasons) -> dict:
    """{arm: (season_betas, the type features it reads)}: the refits on the folds' swaps, the smaller-share control,
    and experiment 80's single-season versions without `seasons_before` (the owner, 2026-10-19: "it must ONLY come
    from the season in question"): `step5_single` the shipped model on SINGLE_SEASON, `step5_other_single` the
    held-back refit on them."""
    every = list(AUDITED_FEATURES)
    arms = {"step5_share037": (smaller_share(model.betas), every),
            "step5_single": (season_betas({season: {side: kept_columns(kept.grams[side], SINGLE_SEASON)
                                                    for side in sa.SIDES}
                                           for season, kept in model.swaps.items()}, model.trainable), SINGLE_SEASON)}
    for columns, suffix in ((every, ""), (SINGLE_SEASON, "_single")):
        for arm, betas in refit_betas(out_dir, seasons, model.trainable, columns).items():
            arms[f"{arm}{suffix}"] = (betas, columns)
    return arms


def fold_summary(world: FoldWorld, tables: dict, factors: dict) -> dict:
    """How far step 5 moved the fold's players, and its spread factors, beside the fold's size."""
    fold, step5 = world.fold, tables["step5"]
    weight = step5.poss_off.to_numpy(float)
    move = np.abs(step5.c_off.to_numpy() + step5.c_def.to_numpy())
    return dict(stem=fold.stem, season=fold.season, players=len(step5), factor_off=factors["offense"],
                factor_def=factors["defense"], median_move=float(np.median(move[weight >= 1000])),
                largest_move=float(move.max()), fit_pairs=len(world.fit.pairs["offense"]),
                held_pairs=len(world.held.pairs["offense"]))


def into(totals: dict, errors: dict) -> None:
    for key, (top, bottom) in errors.items():
        total = totals.setdefault(key, [0.0, 0.0])
        total[0] += top
        total[1] += bottom


@dataclass
class FoldRun:
    """One system's folds and what scoring them needs."""
    checks: Checks
    model: TypeModel
    placebo: dict
    folds: dict                  # {season: its folds}
    where: dict                  # {stem: its folder}
    factors: dict                # {season: step 4's factors}
    out_dir: Path
    refits: dict | None = None   # refit_betas, once every season's equations are cached


def season_grams(run: FoldRun, cfg: dict, season: int) -> dict:
    """Stage 4's equations of one season, summed over its folds: {(games, side): PairGrams}."""
    rows, parts = season_rows(cfg, season), {}
    for fold in run.folds[season]:
        world = fold_world(fold, run.where[fold.stem], rows, run.factors[season])
        for key, gram in luck_grams(world, fold_features(run.model, fold.rows["O"], world.table)).items():
            parts.setdefault(key, []).append(gram)
    return {key: sa.add_grams(grams) for key, grams in parts.items()}


def score_fold(run: FoldRun, fold, rows: pd.DataFrame, design) -> tuple[dict, dict]:
    """(the decider's sums for every arm, the fold's summary) for one fold."""
    factors = run.factors[fold.season]
    world = fold_world(fold, run.where[fold.stem], rows, factors)
    check_step4(run.checks, world, factors)
    tables, spread = arms(fold_inputs(run.model, world), run.model, run.placebo, run.refits)
    extra = {arm: as_stage(table, fold) for arm, table in tables.items()}
    teams = run.checks.wowy.team_arrays(run.where[fold.stem], fold.stem)
    errors = run.checks.wowy.stint_fold_errors(run.checks.audit, fold, teams, design, factors, extra)
    return errors, fold_summary(world, tables, spread)


def write_season(out_dir: Path, season: int, errors: dict, summaries: list) -> None:
    """summary_<season>.csv, then decider_<season>.csv, which marks the season done."""
    pd.DataFrame(summaries).to_csv(out_dir / f"summary_{season}.csv", index=False)
    pd.DataFrame([dict(season=season, stage=stage, slice=slice_, top=top, bottom=bottom)
                  for (stage, slice_), (top, bottom) in errors.items()]).to_csv(out_dir / f"decider_{season}.csv",
                                                                                 index=False)


def run_season(run: FoldRun, cfg: dict, season: int, cache: DesignCache, started: float) -> None:
    rows, design = season_rows(cfg, season), cache.season(season, "pts")
    errors, summaries = {}, []
    for fold in run.folds[season]:
        fold_errors, summary = score_fold(run, fold, rows, design)
        into(errors, fold_errors)
        summaries.append(summary)
    write_season(run.out_dir, season, errors, summaries)
    moved = np.median([summary["median_move"] for summary in summaries])
    print(f"  {run.model.name} {season}: {len(summaries)} folds, step 5 moved a player with 1,000+ possessions a "
          f"median {moved:.3f} per 100 ({time.time() - started:.0f}s)", flush=True)


def fold_run(cfg: dict, checks: Checks, name: str, seeds) -> FoldRun:
    folds, where = checks.wowy.system_folds(resolve(cfg, "within"), name)
    model = type_model(cfg, name)
    placebo = {seed: placebo_betas(model, seed) for seed in seeds}
    by_season: dict = {}
    for fold in folds:
        by_season.setdefault(fold.season, []).append(fold)
    out_dir = resolve(cfg, "outputs") / "swap_audit" / (f"{name}_agecurve" if AGE_CURVE in AUDITED_PANEL else name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return FoldRun(checks, model, placebo, by_season, where, fold_factors(checks, folds), out_dir)


def run_folds(args, cfg: dict) -> int:
    """`folds`: first stage 4's equations of every season not cached (the refitted arms need all of them), then the
    arms of every season of `--first` to `--last` not already done."""
    started = time.time()
    run = fold_run(cfg, Checks.load(), args.system, SEEDS)
    for season in sorted(run.folds):
        path = run.out_dir / f"grams_{season}.npz"
        if not path.exists():
            save_grams(path, season_grams(run, cfg, season))
            print(f"  {args.system} {season}: stage 4's equations ({time.time() - started:.0f}s)", flush=True)
    run.refits = refit_arms(run.model, run.out_dir, run.folds)
    cache = DesignCache(cfg, keep=1)
    for season in sorted(run.folds):
        if args.first <= season <= args.last and not (run.out_dir / f"decider_{season}.csv").exists():
            run_season(run, cfg, season, cache, started)
    print(f"done in {time.time() - started:.0f}s")
    return 0


# ---------------------------------------------------------------------------------------------- stage 0
def reproduce_from_build(model: TypeModel) -> tuple[float, float]:
    """(the build's own step 5 on the step-4 table, this check's season by season): each one's largest gap from the
    step-5 table the build wrote."""
    fold_of = sa.player_folds(model.step4.player_id, SWAP.n_folds)
    predictions = sa.type_predictions(model.features, model.swaps, fold_of, model.trainable, SWAP)
    rebuilt, _ = sa.adjust(model.step4, sw.ratings_in_raw_sign(model.step4), model.swaps, predictions,
                           model.main_team, SWAP)
    mine = pd.concat([arms(whole_inputs(model, season), model, {})[0]["step5"] for season in model.swaps])
    return largest_gap(rebuilt, model.final), largest_gap(mine, model.final)


def whole_season_world(cfg: dict, model: TypeModel, season: int, factors: tuple) -> tuple[SeasonInputs, float, int]:
    """The season handed to the fold code with every game as its fit games: (its inputs, how far its step 4 is from
    the step-4 table, the table's players it does not rate).  The fold code rates a player only with offensive
    possessions; the table also keeps one who played defence alone (2010: one player, one possession)."""
    within = resolve(cfg, "within")
    folder = next(within / f"{model.name}_{part}" for part in ("early", "within")
                  if (within / f"{model.name}_{part}" / f"players_{season}_all.parquet").exists())
    players = pd.read_parquet(folder / f"players_{season}_all.parquet")
    rated = players[players.poss_off > 0]          # the rated players, as data/within_folds.read_fold keeps them
    rows = {side: rated[rated.side == side].reset_index(drop=True) for side in ("O", "D")}
    table = prior_shrink.shrink_fold(rows, season, factors)
    games = set(pd.read_parquet(folder / f"folds_{season}.parquet").index.astype(str))
    every = season_rows(cfg, season)
    lineups = sw.prepare(every[every.game_id.astype(str).isin(games).to_numpy()].reset_index(drop=True), season)
    inputs = SeasonInputs(table, fold_features(model, rows["O"], table), None, sa.team_possessions(lineups),
                          fold_main_team(rows["O"], season), (season, WHOLE_SEASON_KEY))
    return inputs, largest_gap(table, model.step4[model.step4.season == season]), unmatched(
        table[["player_id", "season"]], model.step4[model.step4.season == season][["player_id", "season"]])


def reproduce_fold_path(cfg: dict, checks: Checks, model: TypeModel) -> pd.DataFrame:
    """One row per season: the fold code handed every game, its step 4 and its step 5 against the build's tables."""
    folds, _ = checks.wowy.system_folds(resolve(cfg, "within"), model.name)
    factors = fold_factors(checks, folds)
    rows = []
    for season in sorted(model.swaps):
        inputs, step4_gap, missing = whole_season_world(cfg, model, season, factors[season])
        step5 = arms(inputs, model, {})[0]["step5"]
        rows.append(dict(season=season, players=len(inputs.table), not_rated_here=missing, step4_gap=step4_gap,
                         step5_gap=largest_gap(step5, model.final[model.final.season == season]),
                         features_gap=float(np.nanmax(np.abs(inputs.features.to_numpy()
                                                             - model.season_features(season)
                                                             .reindex(inputs.features.index).to_numpy())))))
    return pd.DataFrame(rows)


def run_reproduce(args, cfg: dict) -> int:
    """`reproduce`: stage 0.  Stops with an error if a rating moves by more than TOLERANCE."""
    model = type_model(cfg, args.system)
    build_gap, mine_gap = reproduce_from_build(model)
    print(f"  the build's step 5 rebuilt: largest gap {build_gap:.1e}; season by season through this check: "
          f"{mine_gap:.1e}", flush=True)
    table = reproduce_fold_path(cfg, Checks.load(), model)
    out = resolve(cfg, "outputs") / "swap_audit"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"reproduce_{args.system}.csv", index=False)
    print(table.to_string(index=False))
    whole = table[table.not_rated_here == 0]
    worst = max(build_gap, mine_gap, float(whole[["step4_gap", "step5_gap"]].to_numpy().max()))
    print(f"largest gap over every season the fold code rates whole: {worst:.1e} against a tolerance of "
          f"{TOLERANCE:g}")
    for row in table[table.not_rated_here > 0].itertuples():
        print(f"  {row.season}: {row.not_rated_here} player(s) of the table without offensive possessions, so not "
              f"rated by the fold code; the rest move by {row.step5_gap:.1e}")
    return 0 if worst <= TOLERANCE else 1


# ---------------------------------------------------------------------------------------------- stage 2's tables
def run_whole(args, cfg: dict) -> int:
    """`whole`: every arm on the whole-season step-4 table, one table per arm."""
    model = type_model(cfg, args.system)
    placebo = {seed: placebo_betas(model, seed) for seed in SEEDS}
    refits = refit_arms(model, resolve(cfg, "outputs") / "swap_audit" / args.system, model.swaps)
    if "step5_other" not in refits:
        print("  the folds' equations are not all cached (`folds` writes them): no refitted arms", flush=True)
    tables: dict = {}
    for season in sorted(model.swaps):
        for arm, rows in arms(whole_inputs(model, season), model, placebo, refits)[0].items():
            tables.setdefault(arm, []).append(rows)
    outputs = resolve(cfg, "outputs")
    columns = list(model.final.columns)
    for arm, parts in tables.items():
        if arm == "step5":
            gap = largest_gap(pd.concat(parts), model.final)
            print(f"  step5 against the build's table: largest gap {gap:.1e}")
            continue
        path = outputs / f"season_ratings_audit_{args.system}_{arm}.parquet"
        pd.concat(parts, ignore_index=True)[columns].to_parquet(path, index=False)
        print(f"wrote {path}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------- the decider
def decider_seasons(cfg: dict, systems: list) -> pd.DataFrame:
    """system, season, stage, slice, error: every cached season of every system, with the placebo seeds' mean as the
    stages placebo_use and placebo_train."""
    frames = []
    for name in systems:
        for path in sorted((resolve(cfg, "outputs") / "swap_audit" / name).glob("decider_*.csv")):
            frames.append(pd.read_csv(path).assign(system=name))
    table = pd.concat(frames, ignore_index=True)
    table = table[table.stage != "share"].assign(error=lambda frame: frame.top / frame.bottom)
    seeds = table.stage.str.extract(r"^(placebo_(?:use|train))_\d+$")[0]
    means = (table[seeds.notna()].assign(stage=seeds[seeds.notna()])
             .groupby(["system", "season", "stage", "slice"], as_index=False).error.mean())
    return pd.concat([table[["system", "season", "stage", "slice", "error"]], means], ignore_index=True)


def paired_row(table: pd.DataFrame, mine: tuple, theirs: tuple, slice_: str, seasons=(1997, 2026)) -> dict:
    """mine and theirs are (system, stage): the paired reading over the seasons both have."""
    def errors(system, stage):
        rows = table[(table.system == system) & (table.stage == stage) & (table.slice == slice_)
                     & table.season.between(*seasons)]
        return rows.set_index("season").error
    diff = (errors(*mine) - errors(*theirs)).dropna()
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan
    return dict(system=mine[0], stage=mine[1], against_system=theirs[0], against_stage=theirs[1], slice=slice_,
                first=seasons[0], last=seasons[1], mean_change=float(diff.mean()), se=se,
                z=float(diff.mean() / se) if se and se > 0 else np.nan, seasons_better=int((diff < 0).sum()),
                seasons=len(diff))


def comparisons(systems: list) -> list:
    """[(mine, theirs)]: each arm against step 4 and step 5 against its rivals, on each system; and each stage of a
    noise control against its system's."""
    arms_read = ("step5", "step5_nohold", "shrink_only", "step5_leak", "placebo_use", "placebo_train", "step5_other",
                 "step5_same", "step5_share037", "step5_single", "step5_other_single")
    pairs = []
    for name in systems:
        pairs += [((name, arm), (name, "step4")) for arm in arms_read]
        pairs += [((name, "step5"), (name, rival)) for rival in arms_read[1:]]
    for name, noise in NOISE.items():
        if name in systems and noise in systems:
            pairs += [((noise, stage), (name, stage)) for stage in ("step4", "step5")]
    return pairs


def run_decider(args, cfg: dict) -> int:
    """`decider`: the paired readings of stages 1 and 2 from the cached folds."""
    systems = [name.strip() for name in args.systems.split(",") if name.strip()]
    table = decider_seasons(cfg, systems)
    rows = [paired_row(table, mine, theirs, slice_, era)
            for mine, theirs in comparisons(systems) for slice_ in ("all", "dnp")
            for era in ((1997, 2026), *ERAS)]
    summary = pd.DataFrame(rows).dropna(subset=["mean_change"])
    out = resolve(cfg, "outputs") / "swap_audit"
    table.to_csv(out / f"decider_{args.tag}_seasons.csv", index=False)
    summary.to_csv(out / f"decider_{args.tag}.csv", index=False)
    pd.set_option("display.width", 220)
    whole = summary[(summary["first"] == 1997) & (summary["last"] == 2026)]
    print("each stage minus the other, season by season, the error over the held-back stints; below 0 is better")
    print(whole.drop(columns=["first", "last"]).round(4).to_string(index=False))
    return 0


# ---------------------------------------------------------------------------------------------- the guard and swap test
def with_placebo_means(table: pd.DataFrame, keys: list) -> pd.DataFrame:
    """`table` (arm, *keys, value) with the placebo seeds' mean added as the arms placebo_use and placebo_train."""
    seeds = table.arm.str.extract(r"^(placebo_(?:use|train))_\d+$")[0]
    means = (table[seeds.notna()].assign(arm=seeds[seeds.notna()]).groupby(["arm", *keys], as_index=False)
             .value.mean())
    return pd.concat([table, means], ignore_index=True)


def guard_observations(cfg: dict, tag: str) -> pd.DataFrame:
    """arm, held_out, direction, value: the guard's team-game mean squared error per observation (63_yoy.py)."""
    yoy = pd.read_parquet(resolve(cfg, "outputs") / f"yoy_{tag}.parquet")
    yoy = yoy[(yoy.split == "all") & (yoy.group == "all")]
    name = yoy.system.str.rsplit(":", n=1)
    table = pd.DataFrame({"arm": name.str[0], "held_out": yoy.held_out.to_numpy(), "direction": name.str[1],
                          "value": yoy.tg.to_numpy()})
    return with_placebo_means(table, ["held_out", "direction"])


def swap_observations(cfg: dict, tag: str, score: str) -> pd.DataFrame:
    """arm, scored, rated, value: the swap test's `score` on the net side, every pair (90_swap_test.py)."""
    swaps = pd.read_parquet(resolve(cfg, "outputs") / f"swaptest_{tag}.parquet")
    swaps = swaps[(swaps.side == "net") & (swaps.group == "all")]
    table = pd.DataFrame({"arm": swaps.ranking.to_numpy(), "scored": swaps.scored.to_numpy(),
                          "rated": swaps.rated.to_numpy(), "value": swaps[score].to_numpy()})
    return with_placebo_means(table, ["scored", "rated"])


def observation_pair(table: pd.DataFrame, keys: list, mine: str, theirs: str, higher_is_better: bool = False) -> dict:
    """The paired reading of arm `mine` against arm `theirs` over the observations both have."""
    def values(arm):
        return table[table.arm == arm].set_index(keys).value
    diff = (values(mine) - values(theirs)).dropna()
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    better = (diff > 0) if higher_is_better else (diff < 0)
    return dict(arm=mine, against=theirs, mean_change=float(diff.mean()), se=se, z=float(diff.mean() / se),
                observations_better=int(better.sum()), observations=len(diff))


def run_readings(args, cfg: dict) -> int:
    """`readings`: stage 2's guard and swap test, paired: each arm against step 4, step 5 against its rivals."""
    readings = {"guard, team-game error": (guard_observations(cfg, args.tag), ["held_out", "direction"], False),
                "swap test, net order": (swap_observations(cfg, args.tag, "order"), ["scored", "rated"], True),
                "swap test, net gaps": (swap_observations(cfg, args.tag, "gaps"), ["scored", "rated"], False)}
    rows = []
    for reading, (table, keys, higher) in readings.items():
        arms_read = [arm for arm in table.arm.unique()
                     if arm != "step4" and not re.fullmatch(r"placebo_(?:use|train)_\d+", arm)]
        pairs = [(arm, "step4") for arm in arms_read] + [("step5", rival) for rival in arms_read if rival != "step5"]
        rows += [dict(reading=reading, **observation_pair(table, keys, mine, theirs, higher)) for mine, theirs in pairs]
    summary = pd.DataFrame(rows)
    summary.to_csv(resolve(cfg, "outputs") / "swap_audit" / f"readings_{args.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    return 0


# ---------------------------------------------------------------------------------------------- stage 3
def coefficients(parts: list) -> np.ndarray:
    """beta from summed normal equations, with step 5's ridge (swap_adjust.type_coefficients, every pair)."""
    return sa.type_coefficients(parts, None, SWAP.type_model_penalty)


def jackknife(parts: dict) -> tuple[np.ndarray, np.ndarray]:
    """(beta on every season of `parts`, its season-jackknife standard error): beta refitted with each season left
    out, se = sqrt((n - 1) / n x sum over seasons of (beta_-s - their mean)^2)."""
    seasons = list(parts)
    drops = np.array([coefficients([parts[other] for other in seasons if other != season]) for season in seasons])
    n = len(seasons)
    se = np.sqrt((n - 1) / n * ((drops - drops.mean(axis=0)) ** 2).sum(axis=0))
    return coefficients([parts[season] for season in seasons]), se


def shipped_equations(model: TypeModel, side: str) -> dict:
    """{season: Equations} of every trainable season's whole-season swaps, as the type model reads them."""
    return {season: Equations(kept.grams[side].G, kept.grams[side].b, kept.grams[side].n)
            for season, kept in model.swaps.items() if season in model.trainable}


def capped(pairs: pd.DataFrame) -> pd.DataFrame:
    """The swaps with each one's information capped at the WEIGHT_CAP quantile of the season's."""
    return pairs.assign(h=np.minimum(pairs.h.to_numpy(), np.quantile(pairs.h.to_numpy(), WEIGHT_CAP)))


def context_equations(cfg: dict, model: TypeModel) -> dict:
    """{(refit, side): {season: Equations}} for the three context refits: garbage-time stints out of the lineups'
    results, the time-on-court clocks back in the level refit, each swap's information capped."""
    raw = sw.ratings_in_raw_sign(model.step4)
    out: dict = {}
    for season in sorted(model.trainable & set(model.swaps)):
        rows, ratings, features = season_rows(cfg, season), raw[raw.season == season], model.season_features(season)
        no_garbage = sa.residual_pairs(sw.prepare(rows[rows.is_gt.to_numpy() < 0.5].reset_index(drop=True), season),
                                       ratings)
        with_clocks = sa.residual_pairs(sw.prepare(rows, season), ratings, "full")
        for side in sa.SIDES:
            for refit, pairs in (("without_garbage_time", no_garbage[side]), ("with_time_on_court", with_clocks[side]),
                                 ("information_capped", capped(model.swaps[season].pairs[side]))):
                out.setdefault((refit, side), {})[season] = equations(pairs, features)
    return out


def subset_coefficients(parts: dict) -> dict:
    """{column: beta} on each era and on the odd and the even seasons."""
    out = {f"era_{first}_{last}": coefficients([part for season, part in parts.items() if first <= season <= last])
           for first, last in ERAS}
    out["odd_seasons"] = coefficients([part for season, part in parts.items() if season % 2 == 1])
    out["even_seasons"] = coefficients([part for season, part in parts.items() if season % 2 == 0])
    return out


def side_table(side: str, parts: dict, refits: dict) -> pd.DataFrame:
    """One row per type feature: every reading of stage 3 for one side.  Defence is shown positive good: + means a
    player with more of the feature than the teammate he swaps with allowed fewer points than the ratings said."""
    sign = 1.0 if side == "offense" else -1.0
    beta, se = jackknife(parts)
    table = pd.DataFrame({"side": side, "feature": AUDITED_FEATURES,
                          "name": [PLAIN_NAMES[feature] for feature in AUDITED_FEATURES],
                          "coefficient": sign * beta, "jackknife_se": se, "t": sign * beta / se})
    for column, values in subset_coefficients(parts).items():
        table[column] = sign * values
    for refit in ("without_garbage_time", "with_time_on_court", "information_capped"):
        table[refit] = sign * coefficients(list(refits[(refit, side)].values()))
    return table


def weighted_correlation(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    x, y = x - np.average(x, weights=weight), y - np.average(y, weights=weight)
    return float(np.sum(weight * x * y) / np.sqrt(np.sum(weight * x * x) * np.sum(weight * y * y)))


def stability(model: TypeModel) -> pd.DataFrame:
    """Per side: the correlation over player-seasons of each player's type prediction from the odd-season model and
    from the even-season model, weighted by his possessions, for everyone and for players with 1,000+."""
    rows = []
    known = model.features.dropna()
    poss = model.step4.set_index(["player_id", "season"])[["poss_off", "poss_def"]].reindex(known.index)
    for side, column in (("offense", "poss_off"), ("defense", "poss_def")):
        parts = shipped_equations(model, side)
        betas = subset_coefficients(parts)
        odd, even = known.to_numpy() @ betas["odd_seasons"], known.to_numpy() @ betas["even_seasons"]
        weight = poss[column].fillna(0.0).to_numpy()
        regular = weight >= 1000
        rows.append(dict(side=side, player_seasons=len(known), correlation=weighted_correlation(odd, even, weight),
                         players_1000=int(regular.sum()),
                         correlation_1000=weighted_correlation(odd[regular], even[regular], weight[regular]),
                         sd_odd=float(np.sqrt(np.average((odd - np.average(odd, weights=weight)) ** 2,
                                                         weights=weight)))))
    return pd.DataFrame(rows)


def run_types(args, cfg: dict) -> int:
    """`types`: stage 3's table, every feature on both sides, and the player-level stability."""
    model = type_model(cfg, args.system)
    refits = context_equations(cfg, model)
    table = pd.concat([side_table(side, shipped_equations(model, side), refits) for side in sa.SIDES],
                      ignore_index=True)
    out = resolve(cfg, "outputs") / "swap_audit"
    table.to_csv(out / f"types_{args.system}.csv", index=False)
    agreement = stability(model)
    agreement.to_csv(out / f"stability_{args.system}.csv", index=False)
    pd.set_option("display.width", 260)
    print(table.drop(columns=["feature"]).round(3).to_string(index=False))
    print(agreement.round(3).to_string(index=False))
    return 0


# ---------------------------------------------------------------------------------------------- stage 4
def luck_parts(cfg: dict, name: str, trainable: set) -> dict:
    """{(games, side): {season: Equations}} from the cached folds of every trainable season."""
    out: dict = {}
    for path in sorted((resolve(cfg, "outputs") / "swap_audit" / name).glob("grams_*.npz")):
        season = int(path.stem.split("_")[1])
        if season not in trainable:
            continue
        for key, gram in load_grams(path).items():
            out.setdefault(key, {})[season] = Equations(gram.G, gram.b, gram.n)
    return out


def run_luck(args, cfg: dict) -> int:
    """`luck`: stage 4's table, the type model on the fit games' swaps beside the held-back games'."""
    _, trainable = panel_and_trainable_seasons(cfg)
    parts = luck_parts(cfg, args.system, trainable)
    frames = []
    for side in sa.SIDES:
        sign = 1.0 if side == "offense" else -1.0
        same, same_se = jackknife(parts[("same", side)])
        other, other_se = jackknife(parts[("other", side)])
        frames.append(pd.DataFrame({"side": side, "name": [PLAIN_NAMES[feature] for feature in AUDITED_FEATURES],
                                    "same_games": sign * same, "same_games_se": same_se, "other_games": sign * other,
                                    "other_games_se": other_se, "ratio": other / same,
                                    "swaps_same": sum(part.n for part in parts[("same", side)].values()),
                                    "swaps_other": sum(part.n for part in parts[("other", side)].values())}))
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(resolve(cfg, "outputs") / "swap_audit" / f"luck_{args.system}.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    return 0


# ---------------------------------------------------------------------------------------------- stage 5
def whose_tables(cfg: dict, model: TypeModel) -> dict:
    """{name: ratings in raw sign}: the five tables whose swap residuals the type model is fitted against."""
    outputs = resolve(cfg, "outputs")
    step3 = pd.read_parquet(outputs / f"season_ratings_{model.name}_step3.parquet")
    plain = pd.read_parquet(outputs / f"{PLAIN_RAPM}.parquet")
    return {"plain_rapm": sw.ratings_in_raw_sign(plain), "box_score_part": sw.ratings_in_raw_sign(step3, "prior"),
            "step3": sw.ratings_in_raw_sign(step3), "step4": sw.ratings_in_raw_sign(model.step4),
            "step5": sw.ratings_in_raw_sign(model.final)}


def within_team_spread(model: TypeModel, beta: np.ndarray, side: str) -> float:
    """The possession-weighted standard deviation of the type prediction about each player's main-team mean, pooled
    over every season: how far apart the model would move teammates."""
    known = model.features.dropna()
    prediction = known.to_numpy() @ beta
    frame = pd.DataFrame({"m": prediction, "team": model.main_team.reindex(known.index).to_numpy(dtype=float),
                          "season": known.index.get_level_values("season"),
                          "w": model.step4.set_index(["player_id", "season"])[
                              "poss_off" if side == "offense" else "poss_def"].reindex(known.index).fillna(0.0)
                          .to_numpy()})
    frame = frame[frame.team.notna() & (frame.w > 0)]
    mean = (frame.w * frame.m).groupby([frame.season, frame.team]).transform("sum") / frame.w.groupby(
        [frame.season, frame.team]).transform("sum")
    return float(np.sqrt(np.average((frame.m - mean) ** 2, weights=frame.w)))


def run_whose(args, cfg: dict) -> int:
    """`whose`: stage 5's table, the type model against each table's swap residuals, every feature on both sides."""
    model = type_model(cfg, args.system)
    tables = whose_tables(cfg, model)
    parts: dict = {}
    for season in sorted(model.trainable & set(model.swaps)):
        lineups, features = sw.prepare(season_rows(cfg, season), season), model.season_features(season)
        for name, ratings in tables.items():
            pairs = sa.residual_pairs(lineups, ratings[ratings.season == season])
            for side in sa.SIDES:
                parts.setdefault((name, side), {})[season] = equations(pairs[side], features)
    frames, spreads = [], []
    for side in sa.SIDES:
        sign = 1.0 if side == "offense" else -1.0
        frame = pd.DataFrame({"side": side, "name": [PLAIN_NAMES[feature] for feature in AUDITED_FEATURES]})
        for name in tables:
            beta, se = jackknife(parts[(name, side)])
            frame[name], frame[f"{name}_se"] = sign * beta, se
            spreads.append(dict(side=side, table=name, within_team_sd=within_team_spread(model, beta, side)))
        frames.append(frame)
    out = resolve(cfg, "outputs") / "swap_audit"
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(out / f"whose_{args.system}.csv", index=False)
    pd.DataFrame(spreads).to_csv(out / f"whose_spread_{args.system}.csv", index=False)
    pd.set_option("display.width", 260)
    print(table.round(3).to_string(index=False))
    print(pd.DataFrame(spreads).round(3).to_string(index=False))
    return 0


# ---------------------------------------------------------------------------------------------- the command line
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="What the swap step does (notes/PLAN_swap_step.md).")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("reproduce", "stage 0: rebuild the system's step-5 table"),
                       ("whole", "stage 2's tables: every arm on the whole-season table"),
                       ("folds", "stages 1, 2 and 4 on every fold"),
                       ("types", "stage 3: the type model's coefficients"),
                       ("luck", "stage 4: fit-game swaps against held-back-game swaps"),
                       ("whose", "stage 5: the type model against five tables")):
        command = commands.add_parser(name, help=text)
        command.add_argument("--system", default="bags10", help="the system's tables and folds (default %(default)s)")
        if name == "folds":
            command.add_argument("--first", type=int, default=1997)
            command.add_argument("--last", type=int, default=2026)
            command.add_argument("--age-curve", action="store_true",
                                 help="experiment 87: step 5 also reads (age - 27)^2; cached under <system>_agecurve")
    decider = commands.add_parser("decider", help="the readings of stages 1 and 2 from the cached folds")
    decider.add_argument("--systems", default="bags10,noise_bags10", help="comma-separated (default %(default)s)")
    decider.add_argument("--tag", default="swap_audit", help="names the output files (default %(default)s)")
    readings = commands.add_parser("readings", help="stage 2's guard and swap test, paired")
    readings.add_argument("--tag", default="swap_audit", help="reads outputs/yoy_<TAG>.parquet and "
                                                              "outputs/swaptest_<TAG>.parquet (default %(default)s)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    use_utf8()
    args = parse_args(argv)
    if getattr(args, "age_curve", False):
        use_age_curve()
    commands = {"reproduce": run_reproduce, "whole": run_whole, "folds": run_folds, "decider": run_decider,
                "readings": run_readings, "types": run_types, "luck": run_luck, "whose": run_whose}
    with one_thread():
        return commands[args.command](args, load_config())


if __name__ == "__main__":
    raise SystemExit(main())
