"""`openrapm folds`: each season rated from three quarters of its games and scored on the quarter it never saw.

    openrapm folds --models NAME --base NAME [--seasons 2017-2026] [--tag TAG] [--reproduce-only] [--pieces]
                   [--whole-season-inputs]

Why (the owner, 2026-10-03).  The prior shrink (step 4) needs the ratings' misses on games they never saw.  Misses
on the neighbouring seasons' games would teach it that a peak season comes back down the year after, and it would
shrink every peak season; so the misses come from the season's own games, held out.  notes/WITHIN_SEASON_LEDGER.md lists
every input, what it is built from and the check that proves it.

How.  A season's games (regular season and playoffs) are assigned to four folds at random, three times over (three
repeats), each assignment drawn with the seed (season, repeat).  For each fold:

    fit games    every game not in the fold.  Every input the rating reads (box score, playing time, starts,
                 tenure, on-court record) is rebuilt from these games alone (`data.panel.season_panel`, the function
                 `openrapm data panel` builds whole seasons with); the saved boosters of `--models` answer for the
                 rebuilt rows (step 2), and the ridge pulls the fit games' RAPM toward them (step 3), as
                 `openrapm rate` does for a whole season.  `leak_checks` proves nothing else was read.
    test games   the fold.  Nothing about them reaches the rating.  Stored: the test team-games (points per 100
                 as scored, each player's share of the possessions) and the rating's error on them.

The reproduction check comes first: each season rebuilt from all of its games must give the row of the base table
(`--base`, written by the same `openrapm rate --save-models` run), so the part-season ratings come from the shipped
code path and nothing else.  The rebuilt inputs are compared with the season panel, column by column.

Writes, under outputs/within/<tag>/ (paths.within):
    folds_<season>.parquet                 game_id -> fold, one column per repeat
    players_<season>_all.parquet           one row per player and side: inputs, rating and its parts, whole season
    players_<season>_r<r>_f<k>.parquet     the same from the fit games, plus his possessions in the test games
    test_<season>_r<r>_f<k>.npz            the test team-games: shares, points per 100, possessions, controls, teams
    fold_<season>_r<r>_f<k>.json           the fold's error on its test games and every leak check
    reproduce_<season>.json                the reproduction check
    summary.parquet, checks.parquet        all folds, gathered at the end
and, with --pieces (experiment 61's screen, check 145):
    pieces_<season>_r<r>_f<k>.parquet      each rated player's games' part u split over the rows it came from (on court,
                                           sat, missed, team SOS) and over the points it is made of (twos, threes,
                                           free throws), with its posterior standard deviation; one row per player
                                           and side
A finished fold is skipped on a rerun, so a stopped run resumes where it stopped; a failed check stops the run.

The second prior (experiment 81).  Boosters saved by `openrapm rate --second-prior TAG` carry the tag in their meta;
each fold then reads its context numbers from the fit games' stint rows and answers, and step 3 pulls toward the
second prior (second_prior.py), its line the season's from outputs/second_prior/<TAG>/ at the test setting.  The
leak arm, --whole-season-context: the context numbers come from the whole season's stint rows and box scores,
held-back games included, while the player's own box-score prior and the ridge read the fit games only.  The decider on
those folds against the honest ones says how much of the second prior's gain a leak would explain.  Folds built this way
say `context_inputs` "whole season" and are never step 4's.

The leak control, --whole-season-inputs (2026-10-15).  An input that exists only as a season total, such as the
tracking and hustle dashboards, cannot be rebuilt from the fit games, so a fold's prior would read the held-back games
too.  With this flag every input the boosters read comes from the whole season, held-back games included; the ridge
still reads the fit games only.  The decider (scripts/checks/151_wowy.py) on these folds against the honest ones gives
how much such a leak is worth.  Every box-score input leaks here, so it bounds a few leaked columns from above.  Folds
built this way are never step 4's: each fold's summary says `prior_inputs` "whole season".
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import box_prior, decomposition, prior_rapm, second_prior, trade_set
from ..data import legacy_names
from ..data.careers import careers_before
from ..data.config import resolve
from ..data.designs import DesignCache
from ..data.panel import (
    DEFENCE_ALL,
    OFFENCE,
    SeasonPanel,
    SeasonSources,
    SeasonTables,
    fit_designs,
    season_panel,
    share_cap,
    trainable_panel,
)
from ..data.roles import player_season_inputs, roles_path
from ..data.seasons import trainable
from ..data.second_prior_files import line_setup, load_grams
from ..data.shots import DEFENSE_TARGETS, OPPONENT_THREE_LUCK_KEPT, expected_threes
from ..data.tables import team_of_rows
from ..features import CLOSENESS, OFF_COURT, ON_MINUS_OFF, PANEL_KEYS, PRIOR_FEATURES, add_features
from ..scoring import Ratings, armse, predict_season, score
from ..settings import ROW_LABELS, TEST, Targets
from . import parallel
from .build import season_lineup_rows

SIDES = ("O", "D")
REPEATS = 3
N_FOLDS = 4
SCORE_TARGET = "pts"                                        # the test games are scored on the points as scored
# a player the rating never saw: the year-over-year test's fill, 0.25 of the mean rating of players under 500
# possessions
FILL = (500.0, 0.25)
REPRODUCE_TOLERANCE = 1e-6
# the boosters were saved by a test-setting build; the ridge and the targets are the same in both settings
SETTINGS = TEST
# every design a season world rebuilds from its fit games: the points as scored, the panel's two (data/panel.py) and
# the ratings' two (settings.Targets), in the order the leak checks list them
WORLD_TARGETS = tuple(dict.fromkeys((SCORE_TARGET, OFFENCE, DEFENCE_ALL, SETTINGS.targets.offence,
                                     SETTINGS.targets.defence)))


def world_targets(settings) -> tuple:
    """WORLD_TARGETS for a run whose ratings are fitted on `settings.targets` (experiment 84 put the defence on the
    offence's target): the points as scored, the panel's two, then the ratings' two, each once."""
    return tuple(dict.fromkeys((SCORE_TARGET, OFFENCE, DEFENCE_ALL, settings.targets.offence,
                                settings.targets.defence)))


def fold_settings(meta: dict):
    """The test setting with the targets the saved boosters' build was fitted on (its meta's settings), so a build on
    another defensive target is rated on its own target; a file without them is read on the shipped targets."""
    targets = (meta.get("settings") or {}).get("targets")
    return TEST if not targets else replace(TEST, targets=Targets(**targets))
# what a saved file's meta must say: the test setting's labels and training rows, the shipped rows and folds.  How
# the training rows were labelled (`chunk_label`, settings.ROW_LABELS) is free: the folds only read the boosters
EXPECTED_META = dict(exclude_neighbours=1, rows="chunks", unshrink=["defense"], player_folds=5)
# every column of the player tables: the prior's possible inputs, the off-court record, the score state
PLAYER_COLUMNS = list(dict.fromkeys(PRIOR_FEATURES + OFF_COURT + ON_MINUS_OFF + CLOSENESS))
# the share of the opponents' three-point luck each defensive target keeps (data/shots.py), for the outcome pieces
LUCK_KEPT = {"points_with_expected_opponent_threes": OPPONENT_THREE_LUCK_KEPT,
             "points_with_expected_opponent_threes_all": 0.0, "points_with_expected_opponent_threes_half": 0.5}
PIECES_TOLERANCE = 1e-9         # the pieces add up to the games' part, and the refit reproduces it, to this
PIECES_JITTER_TOLERANCE = 1e-6  # the same in a season whose solve needed rapm.solved_ridge's jitter (2020)


# ------------------------------------------------------------------------------------------- the saved models
def load_models(outputs: Path, models: str, seasons=None) -> dict:
    """outputs/prior_models_<models>.pkl under the plain names: a file saved before 2026-10-11 names its inputs,
    input list and targets the old way, and is translated as it is read (data/legacy_names.py), with one line
    saying so.  With `seasons`, only those seasons' boosters and the meta are kept: thirty seasons of boosters take
    about 2 GB, and a process folding a few seasons holds no more than it reads."""
    saved, renamed = legacy_names.saved_models(pd.read_pickle(outputs / f"prior_models_{models}.pkl"))
    if renamed:
        print(f"prior_models_{models}.pkl uses the names from before 2026-10-11; read under the plain names "
              "(each booster's inputs keep their order)", flush=True)
    if seasons is not None:
        wanted = {"meta", *seasons}
        saved = {key: value for key, value in saved.items() if key in wanted}
    return saved


def _step3_mismatches(meta: dict) -> dict:
    """{name: (saved, expected)} for the step-3 settings a schema-2 file records and this command does not use.

    The folds rate every season with `SETTINGS`' ridge penalty and the saved build's own targets (`fold_settings`);
    boosters saved by a build with another penalty would be rated on settings their base table was not built with.
    """
    saved = meta.get("settings") or {}
    expected = {"ridge": asdict(SETTINGS.ridge)}
    return {name: (saved[name], value) for name, value in expected.items() if name in saved and saved[name] != value}


def check_models(meta: dict) -> None:
    """Refuse boosters that were not saved by a test-setting build of the shipped steps.

    `crossfit` "scale" is the earlier builds' cross-fitted scale: its fold priors equalled the full prior value
    for value (checked 2026-10-09), so the full-season fit here reproduces it.  "honest" rebuilt every input per
    cross-fitting fold, which this command does not.
    """
    wrong = {key: (meta.get(key), value) for key, value in EXPECTED_META.items() if meta.get(key) != value}
    if meta.get("chunk_label") not in ROW_LABELS:
        wrong["chunk_label"] = (meta.get("chunk_label"), ROW_LABELS)
    wrong.update(_step3_mismatches(meta))
    if meta.get("crossfit") == "honest":
        raise SystemExit("these boosters priced the prior scale with --crossfit=honest, which `openrapm folds` "
                         "does not rebuild; save them again with `openrapm rate --setting test --save-models`")
    if meta.get("crossfit") != "scale":
        wrong["crossfit"] = (meta.get("crossfit"), "scale")
    if wrong:
        raise SystemExit(f"the saved models were not built at the shipped test settings: {wrong}")


def season_models(saved_all: dict, season: int) -> dict:
    """{"O", "D"}: the season's saved boosters, after checking they never trained on it or a neighbour."""
    if season not in saved_all:
        raise SystemExit(f"no saved models for {season}")
    saved = saved_all[season]
    for side in SIDES:
        if saved[side]["groupings"] != 1 or len(saved[side]["folds"]) != saved[side]["n_folds"]:
            raise SystemExit(f"{season} {side}: the boosters average several fold groupings, which `predict` "
                             "does not; save them with one grouping")
    unseen = set(saved["O"]["unseen"]) & set(saved["D"]["unseen"])
    if not {season - 1, season, season + 1} <= unseen:
        raise SystemExit(f"{season}: the boosters trained on the season or a neighbour (unseen {sorted(unseen)})")
    return saved


def saved_booster(saved: dict):
    """One side's saved boosters as the box_prior object that built them: an OutOfPlayerPrior, or a BaggedPrior when
    the file holds bags."""
    members = [box_prior.OutOfPlayerPrior(full=bag["full"], fold_models=list(bag["folds"]),
                                          held_out_players=list(bag["excluded"]), inputs=list(saved["model_feats"]))
               for bag in saved.get("bags", [saved])]
    return members[0] if "bags" not in saved else box_prior.BaggedPrior(members, list(saved["model_feats"]))


def prior_rows(panel: pd.DataFrame, side: str, saved: dict) -> pd.DataFrame:
    """The panel rows one side's saved boosters answer for: that side's rows with possessions, derived."""
    return box_prior.rated_rows(panel[(panel.side == side) & (panel.possessions > 0)], saved["feats"])


def saved_prior(panel: pd.DataFrame, side: str, saved: dict) -> dict:
    """{player_id: prior} from the saved boosters, each player answered by the fold model that never saw him."""
    rows = prior_rows(panel, side, saved)
    return dict(zip(rows.player_id.to_numpy(), box_prior.predict(saved_booster(saved), rows)))


# ------------------------------------------------------------------------------------------------ the folds
def assign_folds(game_ids, n_folds: int, season: int, repeat: int) -> pd.Series:
    """game_id -> fold: the season's games put in a random order drawn with the seed (season, repeat, 9701) and
    assigned to folds 0, 1, 2, 3, 0, 1, ... in that order, so every fold has the same number of games to within one.

    Example: 1,315 games and four folds give folds of 329, 329, 329 and 328 games.
    """
    ids = np.sort(np.asarray(pd.unique(pd.Series(game_ids).astype(str))))
    rng = np.random.default_rng([int(season), int(repeat), 9701])
    return pd.Series(rng.permutation(len(ids)) % int(n_folds), index=ids, name=f"r{repeat}")


# --------------------------------------------------------------------------------------- one season world
@dataclass
class SeasonWorld:
    """A season rebuilt from its fit games: their designs, the panel rows and role rows built from them, and the
    leak checks that prove nothing else was read."""
    season: int
    fit_idx: np.ndarray          # the fit games' game_idx values, sorted
    designs: dict                # {target: the fit games' design}, WORLD_TARGETS
    inputs: SeasonPanel          # data.panel.season_panel on those designs
    checks: dict

    @property
    def panel(self) -> pd.DataFrame:
        """The rebuilt panel rows (features.SEASON_PANEL_COLUMNS)."""
        return self.inputs.rows

    @property
    def roles(self) -> pd.DataFrame:
        """The season's role rows rebuilt from the fit games, one per player and team."""
        return self.inputs.roles


def design_checks(designs: dict, full: dict, fit_idx: np.ndarray) -> dict:
    """Every rebuilt design reads only the fit games and has the points design's rows; each defensive target also
    records how far its response moved from the whole-season design's on the same rows: 0 for the whole season (the
    rebuild is that design), above 0 for part of one (the shooters were repriced from the fit games)."""
    checks, reference, fit = {}, designs[SCORE_TARGET].rows, set(fit_idx)
    for name, design in designs.items():
        checks[f"design {name}: rows only from fit games"] = set(design.rows["game_idx"].unique()) <= fit
        checks[f"design {name}: box and possession tables only from fit games"] = (
            set(design.game_box["game_idx"].unique()) <= fit and set(design.game_poss["game_idx"].unique()) <= fit)
        checks[f"design {name}: same rows as the points design"] = (
            np.array_equal(design.rows["game_idx"].to_numpy(), reference["game_idx"].to_numpy())
            and np.array_equal(design.rows["is_home_off"].to_numpy(), reference["is_home_off"].to_numpy()))
    for name in (target for target in designs if target in DEFENSE_TARGETS):
        whole_y = full[name].y[np.isin(full[name].rows["game_idx"].to_numpy(), fit_idx)]
        checks[f"design {name}: response against the whole-season design (max change)"] = float(
            np.max(np.abs(designs[name].y - whole_y)))
    return checks


def role_checks(stored_roles: pd.DataFrame, rebuilt: pd.DataFrame, season: int) -> dict:
    """The whole season's rebuilt role rows against the stored roles table's rows for the season."""
    stored = stored_roles[stored_roles.season == season].sort_values(["player_id", "team_id"]).reset_index(drop=True)
    mine = rebuilt[list(stored_roles.columns)].sort_values(["player_id", "team_id"]).reset_index(drop=True)
    same_rows = len(stored) == len(mine) and np.array_equal(stored[["player_id", "team_id"]].to_numpy(),
                                                             mine[["player_id", "team_id"]].to_numpy())
    checks = {"roles: whole-season rebuild has the cached rows": bool(same_rows)}
    if same_rows:
        for column in ("games", "starts", "minutes", "poss_on", "team_poss", "age"):
            checks[f"roles: whole-season rebuild, max difference in {column}"] = float(
                np.nanmax(np.abs(mine[column].to_numpy(float) - stored[column].to_numpy(float))))
    return checks


def leak_checks(inputs: SeasonPanel, designs: dict, full: dict, fit_ids: set, fit_idx: np.ndarray,
                stored_roles: pd.DataFrame, whole: bool) -> dict:
    """Every check of a rebuilt season, in the order the fold files list them: the designs, the games the role rows
    read, the role rows themselves against the stored table (whole season only), the games the score state read."""
    checks = design_checks(designs, full, fit_idx)
    checks["roles: box scores read only from fit games"] = inputs.games_read["role box scores"] <= fit_ids
    checks["roles: stints read only from fit games"] = inputs.games_read["role stints"] <= fit_ids
    if whole:
        checks.update(role_checks(stored_roles, inputs.roles, inputs.season))
    checks["context: stints read only from fit games"] = inputs.games_read["score-state stints"] <= fit_ids
    checks["play-by-play parts: counts read only from fit games"] = inputs.games_read["event counts"] <= fit_ids
    # the per-game hustle and tracking box scores (the key kept its first name)
    checks["hustle: per-game counts read only from fit games"] = inputs.games_read.get("hustle games", set()) <= fit_ids
    return checks


# ---------------------------------------------------------------------------------------- one rated world
@dataclass
class RatedWorld:
    """A season rebuilt from its fit games, and steps 2-3 on it."""
    world: SeasonWorld
    table: pd.DataFrame          # step 3's centred table, raw sign (prior_rapm.season_table's columns)
    prior: dict                  # {"O", "D"}: {player_id: the prior step 3 read, before the prior scale}
    box: dict | None = None      # {"O", "D"}: {player_id: the boosters' answer}, when step 3 read the second prior


@dataclass
class ContextInputs:
    """What one rating's second prior reads (second_prior.py)."""
    coefficients: dict           # {"O", "D"}: the season's line, second_prior.season_coefficients
    stints: pd.DataFrame         # the stint rows the context numbers read: the fit games', or the whole season's
    panel: pd.DataFrame | None   # the panel rows the context's boosters answer for; None: the boosters' own panel
    n_folds: int                 # the line's player folds
    team_totals: str = "held"    # second_prior.TEAM_TOTALS
    type_scale: dict | None = None   # with the line on step 5's numbers too, their standardisation
    input_set: str | None = None     # PriorSettings.second_prior_inputs; None reads it from type_scale


def rate_world(world: SeasonWorld, saved: dict, prior_panel: pd.DataFrame | None = None,
               context: ContextInputs | None = None, settings=SETTINGS) -> RatedWorld:
    """Steps 2-3 on the world's rebuilt inputs: the saved boosters' priors, both ridges, the centred table.

    `prior_panel`, when given, is what the boosters read instead of the world's own rows (the leak control, the
    module docstring); the ridge reads the world's designs either way.  With `context`, step 3 reads the second prior
    built from it instead of the boosters' answers.  `settings` names the targets the ridge reads (`fold_settings`).
    Adds one check to the world's: the ridge's possessions are the fit games' (five players on every row).
    """
    panel = world.panel if prior_panel is None else prior_panel
    box = {side: saved_prior(panel, side, saved[side]) for side in SIDES}
    prior = box
    if context is not None:
        context_panel = panel if context.panel is None else context.panel
        answers = {side: second_prior.booster_answers(saved_booster(saved[side]),
                                                      prior_rows(context_panel, side, saved[side])) for side in SIDES}
        # step 5's numbers from the offence rows with possessions, the rows the build reads (a player with
        # defensive possessions and no offensive ones has none there, and reads 0, the mean)
        types = (None if context.type_scale is None else second_prior.standardised_types(
            panel[(panel.side == "O") & (panel.possessions > 0)], context.type_scale))
        prior, _ = second_prior.season_second_priors(context.stints, answers, box, context.coefficients,
                                                     context.n_folds, context.team_totals, types, context.input_set)
    designs = {target: world.designs[target] for target in settings.targets.distinct()}
    table = prior_rapm.rate_season(designs, prior, settings, world.season)
    world.checks["ridge: possessions are the fit games' possessions"] = bool(np.isclose(
        table.possessions.sum(), float(world.designs[settings.targets.offence].rows["poss"].sum()) * 5.0, rtol=1e-9))
    return RatedWorld(world=world, table=table, prior=prior, box=None if context is None else box)


def player_table(rated: RatedWorld) -> pd.DataFrame:
    """One row per player and side with possessions: every rebuilt input (derived columns built), the rating in
    the published positive-good schema, the boosters' answer before the scale, and his main team."""
    world, table = rated.world, rated.table
    frame = add_features(world.panel[world.panel.possessions > 0].copy(), PLAYER_COLUMNS)
    ratings = pd.DataFrame({
        "player_id": table.player_id, "rating_off": table.offense, "rating_def": -table.defense,
        "prior_off": table.prior_offense, "prior_def": -table.prior_defense,
        "poss_off": table.possessions, "poss_def": table.poss_def,
        "prior_scale_off": table.prior_scale_off, "prior_scale_def": table.prior_scale_def})
    ratings["rating_total"] = ratings.rating_off + ratings.rating_def
    ratings["u_off"], ratings["u_def"] = ratings.rating_off - ratings.prior_off, ratings.rating_def - ratings.prior_def
    box = rated.prior if rated.box is None else rated.box
    ratings["prior_raw_off"] = ratings.player_id.map(box["O"])
    ratings["prior_raw_def"] = -ratings.player_id.map(box["D"])
    if rated.box is not None:
        # the second prior step 3 read, before the prior scale (second_prior.py)
        ratings["second_prior_raw_off"] = ratings.player_id.map(rated.prior["O"])
        ratings["second_prior_raw_def"] = -ratings.player_id.map(rated.prior["D"])
    # his main team in these games (most possessions on the court), for the team-level inputs of step 4's readers
    played = world.roles[world.roles.poss_on > 0]
    main = (played.sort_values(["player_id", "poss_on"], ascending=[True, False])
            .drop_duplicates("player_id").set_index("player_id").team_id)
    ratings["team_id"] = ratings.player_id.map(main).fillna(-1).astype(np.int64)
    return frame.merge(ratings, on="player_id", how="left")


# ------------------------------------------------------------------------------------------- the test games
def score_test_games(rated: RatedWorld, full_points, test_idx: np.ndarray, cfg: dict) -> tuple:
    """The fold's test games on the points as scored: the rating's team-game error on them (the year-over-year
    test's scoring, the level and home edge refitted on the scored games), and the team-games pooled the trade
    set's way for step 4 to fit on.  Returns (pooled, teams, test possessions, summary, checks)."""
    keep = np.isin(full_points.rows["game_idx"].to_numpy(), test_idx)
    design = full_points.subset(keep)
    table = rated.table
    ratings = pd.DataFrame({"player_id": table.player_id, "o": table.offense, "d": table.defense,
                            "poss": table.possessions})
    ratings = ratings[ratings.poss > 0].reset_index(drop=True)
    fill_o, fill_d = trade_set.replacement_fill(ratings, max_poss=FILL[0], shrink=FILL[1])
    result = score(predict_season(Ratings(ratings, fill_o=fill_o, fill_d=fill_d), design))
    pooled = trade_set.team_game_design(design)
    teams = team_of_rows(pooled.keys, cfg)
    n_players = pooled.n_players
    checks = {"test: rows only from test games": set(design.rows["game_idx"].unique()) <= set(test_idx),
              "test: no test game among the fit games": not (set(test_idx) & set(rated.world.fit_idx)),
              "test: shares sum to five": bool(np.allclose(
                  np.asarray(pooled.Z[:, :n_players].sum(axis=1)).ravel(), 5.0, atol=1e-6))}
    with_poss = np.asarray(pooled.Z.T @ pooled.w).ravel()
    test_poss = pd.DataFrame({"player_id": pooled.player_ids, "test_poss_off": with_poss[:n_players],
                              "test_poss_def": with_poss[n_players:]})
    summary = dict(tg=float(result["tg"]), tg_base=float(result["tg_base"]), tg_n=float(result["tg_n"]),
                   game_armse=float(armse(result["tg"])), scale_off=float(result["scale_off"]),
                   scale_def=float(result["scale_def"]), fill_o=fill_o, fill_d=fill_d,
                   test_team_games=int(pooled.Z.shape[0]))
    return pooled, teams, test_poss, summary, checks


def save_test(path: Path, pooled, teams: pd.DataFrame) -> None:
    """The test team-games as a compressed .npz: the share matrix in CSR parts, targets, weights, controls, keys."""
    shares = pooled.Z.tocsr()
    keys = pooled.keys
    np.savez_compressed(path, Z_data=shares.data, Z_indices=shares.indices, Z_indptr=shares.indptr,
                        Z_shape=np.array(shares.shape), y=pooled.y, w=pooled.w, F=pooled.F,
                        control_names=np.array(pooled.control_names), player_ids=pooled.player_ids,
                        game_idx=keys.game_idx.to_numpy(), is_home_off=keys.is_home_off.to_numpy(),
                        game_id=keys.game_id.astype(str).to_numpy(), phase=keys.phase.astype(str).to_numpy(),
                        team_off=teams.team_off.to_numpy(), team_def=teams.team_def.to_numpy())


# ---------------------------------------------------------------------------------- the reproduction check
def failed_checks(checks: dict) -> list:
    """The pass/fail checks that failed (the numeric ones are evidence, read in the ledger)."""
    return [name for name, ok in checks.items() if isinstance(ok, (bool, np.bool_)) and not ok]


def _panel_differences(world: SeasonWorld, panel: pd.DataFrame) -> tuple[dict, int]:
    """({column: largest absolute difference}, stored rows) of the rebuilt panel rows against the stored ones."""
    stored = panel[panel.season == world.season]
    columns = [column for column in world.panel.columns if column in stored.columns
               and column not in PANEL_KEYS]
    both = world.panel.merge(stored, on=["player_id", "side"], suffixes=("", "_panel"))
    return ({column: float(np.nanmax(np.abs(both[column].to_numpy(float) - both[f"{column}_panel"].to_numpy(float))))
             for column in columns if pd.api.types.is_numeric_dtype(both[column])}, int(len(stored)))


def reproduce(rated: RatedWorld, base: pd.DataFrame, panel: pd.DataFrame, saved: dict) -> dict:
    """The whole-season world against the base table's rows for the season, its boosters' answers against the
    priors the build saved, and its rebuilt inputs against the stored season panel."""
    world, mine = rated.world, rated.table
    theirs = base[base.season == world.season]
    joined = mine.merge(theirs[["player_id", "offense", "defense", "prior_offense", "prior_defense"]],
                        on="player_id", suffixes=("", "_base"))
    # "board_players" is the key the earlier files used for the base table's row count
    out = dict(season=world.season, players=int(len(mine)), board_players=int(len(theirs)), joined=int(len(joined)))
    for column in ("offense", "defense", "prior_offense", "prior_defense"):
        out[f"max_diff_{column}"] = float(np.max(np.abs(joined[column] - joined[f"{column}_base"])))
    box = rated.prior if rated.box is None else rated.box
    for side in SIDES:
        built = pd.Series(saved[side]["prior"])
        rebuilt = pd.Series(box[side]).reindex(built.index)
        out[f"max_diff_raw_prior_{side}"] = float(np.nanmax(np.abs(rebuilt - built)))
        out[f"raw_prior_missing_{side}"] = int(rebuilt.isna().sum())
        if rated.box is not None:
            built = pd.Series(saved[side]["second_prior"])
            rebuilt = pd.Series(rated.prior[side]).reindex(built.index)
            out[f"max_diff_second_prior_{side}"] = float(np.nanmax(np.abs(rebuilt - built)))
            out[f"second_prior_missing_{side}"] = int(rebuilt.isna().sum())
    differences, stored_rows = _panel_differences(world, panel)
    out["panel_rows"], out["panel_rows_stored"] = int(len(world.panel)), stored_rows
    out["panel_max_diff"] = differences
    out["world_checks"] = {key: (bool(value) if isinstance(value, (bool, np.bool_)) else value)
                           for key, value in world.checks.items()}
    out["passed"] = bool(len(joined) == len(theirs) == len(mine)
                         and max(out["max_diff_offense"], out["max_diff_defense"]) < REPRODUCE_TOLERANCE
                         and not failed_checks(world.checks))
    return out


# ------------------------------------------------------------------------------------------------ the run
@dataclass
class SeasonRun:
    """What every fold of one season shares."""
    season: int
    cfg: dict
    full: dict                   # {target: the whole season's design}, WORLD_TARGETS
    sources: SeasonSources       # the season's stints and box scores, the roles table of every season
    saved: dict                  # {"O", "D"}: the season's saved boosters
    out_dir: Path
    box_files: object = None     # commands.site.BoxScoreFiles when `--pieces` is on (the rows' teams), else None
    prior_panel: pd.DataFrame | None = None   # the whole season's panel rows with --whole-season-inputs, else None
    line: dict | None = None     # {"O", "D"}: the season's second-prior line when the boosters carry one, else None
    line_folds: int = 5          # the line's player folds
    line_team_totals: str = "held"            # second_prior.TEAM_TOTALS
    line_type_scale: dict | None = None       # with the line on step 5's numbers too
    line_input_set: str | None = None         # the line's inputs, PriorSettings.second_prior_inputs
    stints: pd.DataFrame | None = None        # the season's stint rows (swap_test.season_rows), with the line
    context_panel: pd.DataFrame | None = None  # the whole season's panel rows with --whole-season-context, else None
    settings: object = SETTINGS  # the test setting with the saved build's own targets (`fold_settings`)

    @property
    def game_ids(self) -> pd.Series:
        """game_idx -> game_id (text) for every game of the season."""
        games = self.full[SCORE_TARGET].games
        return pd.Series(games["game_id"].astype(str).to_numpy(), index=games["game_idx"].to_numpy())

    def world(self, fit_ids) -> SeasonWorld:
        """The season rebuilt from the games `fit_ids`, with its leak checks.  Handed every game, it is the whole
        season through the same code."""
        fit = {str(game) for game in fit_ids}
        every_game = set(self.game_ids)
        if not fit <= every_game:
            raise ValueError(f"{self.season}: fit games that are not this season's")
        fit_idx = np.sort(self.game_ids.index[self.game_ids.isin(fit)].to_numpy())
        designs = fit_designs(self.season, self.cfg, self.full, fit)
        inputs = season_panel(self.season, self.cfg, designs, self.sources, fit)
        checks = leak_checks(inputs, designs, self.full, fit, fit_idx, self.sources.roles, fit == every_game)
        return SeasonWorld(self.season, fit_idx, designs, inputs, checks)

    def rate(self, fit_ids) -> RatedWorld:
        world = self.world(fit_ids)
        context = None
        if self.line is not None:
            fit = {str(game) for game in fit_ids}
            stints = self.stints
            if self.context_panel is None:
                stints = stints[stints.game_id.astype(str).isin(fit).to_numpy()]
                world.checks["second prior: stint rows only from fit games"] = set(stints.game_id.astype(str)) <= fit
            context = ContextInputs(self.line, stints, self.context_panel, self.line_folds, self.line_team_totals,
                                    self.line_type_scale, self.line_input_set)
        return rate_world(world, self.saved, self.prior_panel, context, self.settings)

    @property
    def prior_inputs(self) -> str:
        """Where the boosters' inputs come from: "fit games", or "whole season" for the leak control."""
        return "fit games" if self.prior_panel is None else "whole season"

    @property
    def context_inputs(self) -> str:
        """Where the second prior's context numbers come from: "none" without one, "fit games", or "whole season" for
        its leak arm."""
        if self.line is None:
            return "none"
        return "fit games" if self.context_panel is None else "whole season"


def run_reproduction(run: SeasonRun, base: pd.DataFrame, panel: pd.DataFrame, started: float) -> None:
    """The season rebuilt from all of its games, held to the base table; stops the run if it does not match."""
    path = run.out_dir / f"reproduce_{run.season}.json"
    rated = run.rate(set(run.game_ids))
    result = reproduce(rated, base, panel, run.saved)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    player_table(rated).assign(season=run.season, repeat=-1, fold=-1).to_parquet(
        run.out_dir / f"players_{run.season}_all.parquet", index=False)
    largest = sorted(result["panel_max_diff"].items(), key=lambda item: -item[1])[:4]
    print(f"  {run.season} reproduction: rating max diff {result['max_diff_offense']:.1e} / "
          f"{result['max_diff_defense']:.1e}, raw prior {result['max_diff_raw_prior_O']:.1e} / "
          f"{result['max_diff_raw_prior_D']:.1e}; {result['joined']} of {result['board_players']} players; largest "
          f"input differences {[(key, f'{value:.1e}') for key, value in largest]}  ({time.time() - started:.0f}s)",
          flush=True)
    if not result["passed"]:
        raise SystemExit(f"{run.season}: the whole-season rebuild does not reproduce the base table; see {path}")


def season_folds(run: SeasonRun) -> pd.DataFrame:
    """The season's folds, one column per repeat, written to folds_<season>.parquet (checked against any earlier
    file: the assignment is seeded, so a rerun must draw the same folds)."""
    path = run.out_dir / f"folds_{run.season}.parquet"
    folds = pd.concat([assign_folds(run.game_ids.to_numpy(), N_FOLDS, run.season, repeat)
                       for repeat in range(REPEATS)], axis=1)
    if path.exists():
        earlier = pd.read_parquet(path)
        common = [column for column in earlier.columns if column in folds.columns]
        if not earlier[common].equals(folds.loc[earlier.index, common]):
            raise SystemExit(f"{path} holds other folds than the seeded ones")
    folds.to_parquet(path)
    return folds


def fold_checks(run: SeasonRun, rated: RatedWorld, test_checks: dict, fit_ids: set, test_ids: set) -> dict:
    """Every leak check of one fold, in the order the fold file lists them."""
    unseen = set(run.saved["O"]["unseen"]) & set(run.saved["D"]["unseen"])
    checks = dict(test_checks)
    checks.update(rated.world.checks)
    checks["folds: fit and test games disjoint"] = not (fit_ids & test_ids)
    checks["folds: fit and test games cover the season"] = (fit_ids | test_ids) == set(run.game_ids)
    checks["prior models: never trained on this season or its neighbours"] = (
        {run.season - 1, run.season, run.season + 1} <= unseen)
    return checks


def fold_pieces(run: SeasonRun, rated: RatedWorld, fit_ids: set) -> tuple[pd.DataFrame, dict]:
    """One fold's games' part split into pieces on both sides (decomposition.split_pieces): (one row per player and
    side, the checks).

    Step 3 is solved again on each target's fit-games design with the same boosters' priors, through the code the
    rating used, so the refit's games' part is the fold's own; the checks say by how much (they must be 0, up to the
    solver's last digits).  The defence's outcome pieces price the opponents' threes at the shooters' expected makes
    from the fit games alone, as the fold's defensive target did (data/panel.fit_designs).
    """
    world, table, season = rated.world, rated.table, run.season
    teams = run.box_files.teams(season)
    points = world.designs[SCORE_TARGET]
    keep = {int(season): sorted(fit_ids)}
    by_side, checks, retried = [], {}, False
    for side, key, target, sign in (("O", "off", run.settings.targets.offence, 1.0),
                                    ("D", "def", run.settings.targets.defence, -1.0)):
        design = world.designs[target]
        fit = decomposition.refit(design, rated.prior["O"], rated.prior["D"], SETTINGS.ridge.penalty)
        lineup_o, lineup_d = decomposition.lineups(fit.players, fit.n_players)
        keys = decomposition.row_keys(design, lineup_o, lineup_d, teams)
        x3, luck_kept = None, 0.0
        if side == "D":
            x3 = expected_threes([int(season)], run.cfg, points, prev=0, k3=450.0, keep=keep)[0]
            luck_kept = LUCK_KEPT[target]
        outcomes, _ = decomposition.outcome_responses(design.counters, points.y, x3, luck_kept)
        frame, misses = decomposition.split_pieces(fit, keys, key, outcomes)
        rating = table.set_index("player_id")
        own, prior = ("offense", "prior_offense") if side == "O" else ("defense", "prior_defense")
        deviation = sign * (rating[own] - rating[prior])
        retried = retried or misses["retried"]
        checks[f"pieces {key}: refit games' part against the rating's (max change)"] = float(
            np.abs(frame.u.to_numpy() - deviation.reindex(frame.player_id).to_numpy()).max())
        checks[f"pieces {key}: outcome pieces against the response (max change)"] = float(
            np.abs(sum(outcomes.values()) - design.y).max())
        checks[f"pieces {key}: outcome pieces against the games' part (max change)"] = misses["outcomes"]
        checks[f"pieces {key}: row pieces against the games' part (max change)"] = float(
            np.abs(frame[list(decomposition.ROW_PIECES)].sum(axis=1) - frame.u).max())
        checks[f"pieces {key}: games' part is the sum of weight x response (max change)"] = misses["sum_hy"]
        by_side.append(frame.assign(side=side))
    tolerance = PIECES_JITTER_TOLERANCE if retried else PIECES_TOLERANCE
    worst = {name: value for name, value in checks.items() if value >= tolerance}
    if worst:
        raise SystemExit(f"{season}: the pieces do not add up (tolerance {tolerance:g}): {worst}")
    return pd.concat(by_side, ignore_index=True), checks


def run_fold(run: SeasonRun, folds: pd.DataFrame, repeat: int, fold: int, started: float) -> None:
    """Rate the season from every game outside the fold, score the fold, write the fold's three files (and its pieces,
    with --pieces)."""
    stem = f"{run.season}_r{repeat}_f{fold}"
    fold_started = time.time()
    test_ids = set(folds.index[folds[f"r{repeat}"] == fold])
    fit_ids = set(folds.index) - test_ids
    games = run.full[SCORE_TARGET].games
    test_idx = np.sort(games.loc[games["game_id"].astype(str).isin(test_ids), "game_idx"].to_numpy())
    rated = run.rate(fit_ids)
    pooled, teams, test_poss, summary, test_checks = score_test_games(rated, run.full[SCORE_TARGET], test_idx,
                                                                      run.cfg)
    checks = fold_checks(run, rated, test_checks, fit_ids, test_ids)
    if run.box_files is not None:
        pieces, piece_checks = fold_pieces(run, rated, fit_ids)
        checks.update(piece_checks)
        pieces.assign(season=run.season, repeat=repeat, fold=fold).to_parquet(
            run.out_dir / f"pieces_{stem}.parquet", index=False)
    failed = failed_checks(checks)
    table = (player_table(rated).merge(test_poss, on="player_id", how="left")
             .fillna({"test_poss_off": 0.0, "test_poss_def": 0.0})
             .assign(season=run.season, repeat=repeat, fold=fold))
    table.to_parquet(run.out_dir / f"players_{stem}.parquet", index=False)
    save_test(run.out_dir / f"test_{stem}.npz", pooled, teams)
    # rate_from, split and cut name the fold shape, the only one left: kept so the files read as before
    summary.update(season=run.season, repeat=repeat, fold=fold, rate_from="rest", fit_games=len(fit_ids),
                   split="random", cut=None, test_games=len(test_ids), prior_inputs=run.prior_inputs,
                   context_inputs=run.context_inputs,
                   players=int((rated.table.possessions > 0).sum()),
                   prior_scale_off=float(rated.table.prior_scale_off.iloc[0]),
                   prior_scale_def=float(rated.table.prior_scale_def.iloc[0]),
                   seconds=round(time.time() - fold_started, 1))
    record = dict(summary=summary, checks={key: (bool(value) if isinstance(value, (bool, np.bool_)) else value)
                                           for key, value in checks.items()})
    (run.out_dir / f"fold_{stem}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"  {stem}: {summary['fit_games']} fit / {summary['test_games']} test games, {summary['players']} "
          f"players; test error {summary['game_armse']:.3f} (base {armse(summary['tg_base']):.3f}); test scales "
          f"{summary['scale_off']:.2f} / {summary['scale_def']:.2f}; prior scale {summary['prior_scale_off']:.2f} / "
          f"{summary['prior_scale_def']:.2f}; checks {len(checks) - len(failed)} of {len(checks)} passed "
          f"({summary['seconds']:.0f}s, {time.time() - started:.0f}s total)", flush=True)
    if failed:
        raise SystemExit(f"{stem}: leak checks failed: {failed}")


def run_season(run: SeasonRun, base: pd.DataFrame, panel: pd.DataFrame, reproduce_only: bool,
               started: float) -> None:
    """The reproduction check (unless already on disk), then every fold not already on disk."""
    if not (run.out_dir / f"reproduce_{run.season}.json").exists():
        run_reproduction(run, base, panel, started)
    if reproduce_only:
        return
    folds = season_folds(run)
    for repeat in range(REPEATS):
        for fold in range(N_FOLDS):
            if not (run.out_dir / f"fold_{run.season}_r{repeat}_f{fold}.json").exists():
                run_fold(run, folds, repeat, fold, started)


def gather(out_dir: Path) -> None:
    """summary.parquet and checks.parquet: every fold's summary, and every check one row each.  A child of
    `folds_in_parts` leaves it to its parent, which runs it once every child is done."""
    if parallel.is_child():
        return
    paths = sorted(out_dir.glob("fold_*.json"))
    if not paths:
        return
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    pd.DataFrame([record["summary"] for record in loaded]).to_parquet(out_dir / "summary.parquet", index=False)
    # a check is either pass/fail or a measured size: one column each, or one file cannot hold them
    checks = pd.DataFrame([dict(season=record["summary"]["season"], repeat=record["summary"]["repeat"],
                                fold=record["summary"]["fold"], check=key,
                                passed=value if isinstance(value, bool) else None,
                                measure=None if isinstance(value, bool) else float(value))
                           for record in loaded for key, value in record["checks"].items()])
    checks.to_parquet(out_dir / "checks.parquet", index=False)
    failed = checks[checks.passed.eq(False)]
    print(f"checks: {int(checks.passed.notna().sum())} pass/fail checks over {len(loaded)} folds, {len(failed)} failed")
    print(f"\nwrote {out_dir}/summary.parquet and checks.parquet: {len(loaded)} folds")


def second_prior_lines(cfg: dict, meta: dict) -> tuple[dict | None, int, float, str, dict | None]:
    """({season: {side: PairGrams}} or None, the line's player folds, its penalty, its team totals, the standardisation
    of step 5's numbers when it reads them): the second prior the saved boosters' build read (its tag in the meta's
    settings), at the test setting the folds rate at.  Boosters saved before the team totals were a setting applied
    the line as it is, "free"."""
    prior = (meta.get("settings") or {}).get("prior") or {}
    tag = prior.get("second_prior") or ""
    folds, penalty = int(prior.get("second_prior_folds", 5)), float(prior.get("second_prior_penalty", 1e-4))
    totals = prior.get("second_prior_team_totals", "free")
    if not tag:
        return None, folds, penalty, totals, None
    record = line_setup(cfg, tag, second_prior.INPUT_SETS[prior.get("second_prior_inputs", "context")])
    # the nearest-seasons count rides on the record as well: SeasonRun reads it from there
    record = {**record, "nearest": int(prior.get("second_prior_nearest", 0))}
    return load_grams(cfg, tag), folds, penalty, totals, record


def folds_in_parts(cfg: dict, seasons, tag: str, jobs: int, argv) -> bool:
    """`openrapm folds --jobs N`: when more than one process fits (`parallel.affordable_jobs`), the seasons folded in
    that many `openrapm folds` children at once, each writing its own seasons' files into outputs/within/<tag>/, and
    the gather once at the end; True when it ran, False when the command should run in its own process."""
    fits = parallel.affordable_jobs(jobs) if jobs > 1 and len(seasons) > 1 else 1
    if fits < jobs:
        print(f"{jobs} processes asked for, free memory holds {fits}", flush=True)
    if fits < 2:
        return False
    parallel.run_children(list(argv), parallel.season_chunks(seasons, fits), lambda k: [],
                          resolve(cfg, "outputs") / "parallel" / tag)
    gather(resolve(cfg, "within") / tag)
    return True


def run_folds(cfg: dict, models: str, base: str, seasons, tag: str, reproduce_only: bool = False,
              pieces: bool = False, whole_season_inputs: bool = False, whole_season_context: bool = False) -> int:
    """`openrapm folds`: the reproduction check and the folds of every season, then the gathered files.  With
    `pieces`, each fold also writes its pieces_<stem>.parquet (`fold_pieces`); with `whole_season_inputs`, every
    fold's boosters read the whole season's inputs (the leak control, the module docstring); with
    `whole_season_context`, every fold's second prior reads the whole season's context numbers (its leak arm).
    `folds_in_parts` runs it in several processes at once."""
    box_files = None
    if pieces:
        from .site import BoxScoreFiles
        box_files = BoxScoreFiles(cfg)
    outputs = resolve(cfg, "outputs")
    out_dir = resolve(cfg, "within") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_all = load_models(outputs, models, seasons)
    meta = saved_all.get("meta", {})
    check_models(meta)
    settings = fold_settings(meta)
    base_table = pd.read_parquet(outputs / f"{base}.parquet")
    panel = trainable_panel(cfg)
    roles = pd.read_parquet(roles_path(cfg))
    inputs = player_season_inputs(roles, cap=share_cap(cfg))
    debut = careers_before(cfg, roles)            # the seasons before 1997, as the season panel read them
    designs = DesignCache(cfg)
    print(f"models {models} ({meta.get('argv')}); seasons {seasons[0]}-{seasons[-1]}; {REPEATS} repeats of "
          f"{N_FOLDS} folds; targets {settings.targets.offence} / {settings.targets.defence}; writing {out_dir}",
          flush=True)
    if whole_season_inputs:
        print("the leak control: every fold's boosters read the WHOLE season's inputs, held-back games included",
              flush=True)
    grams, line_folds, line_penalty, team_totals, record = second_prior_lines(cfg, meta)
    type_scale = None if record is None else record.get("type_scale")
    line_nearest = 0 if record is None else record["nearest"]
    if whole_season_context and grams is None:
        raise SystemExit("--whole-season-context is the second prior's leak arm; these boosters were saved without one")
    if grams is not None:
        print(f"the second prior: step 3 reads it, each season's line from the grams of "
              f"{meta['settings']['prior']['second_prior']} more than {SETTINGS.exclude_neighbours} season away, "
              f"team totals {team_totals}"
              + ("; the leak arm: its context numbers from the WHOLE season" if whole_season_context else ""),
              flush=True)
    trainable_seasons = set(trainable(cfg).seasons)
    started = time.time()
    for season in seasons:
        saved = season_models(saved_all, season)
        full = {target: designs.season(season, target) for target in world_targets(settings)}
        run = SeasonRun(season, cfg, full, SeasonSources(SeasonTables(season, cfg), roles, inputs, debut), saved,
                        out_dir, box_files, settings=settings)
        if whole_season_inputs:
            run.prior_panel = run.world(set(run.game_ids)).panel
        if grams is not None:
            run.line = second_prior.season_coefficients(grams, season, trainable_seasons, SETTINGS.exclude_neighbours,
                                                        line_folds, line_penalty, line_nearest)
            run.line_folds, run.line_team_totals, run.line_type_scale = line_folds, team_totals, type_scale
            run.line_input_set = ((meta.get("settings") or {}).get("prior") or {}).get("second_prior_inputs")
            run.stints = season_lineup_rows(cfg, season)
            if run.line is None:
                raise SystemExit(f"{season}: no season of the grams may train its second prior's line")
            if whole_season_context:
                run.context_panel = run.world(set(run.game_ids)).panel
        run_season(run, base_table, panel, reproduce_only, started)
        designs.clear()
    gather(out_dir)
    return 0
