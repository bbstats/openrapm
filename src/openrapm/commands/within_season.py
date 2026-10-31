"""`openrapm folds`: each season rated from three quarters of its games and scored on the quarter it never saw.

    openrapm folds --models NAME --base NAME [--seasons 2017-2026] [--tag TAG] [--reproduce-only] [--pieces]
                   [--whole-season-inputs] [--cut SHARE | --so-far SHARE]

Why (the owner, 2026-10-03).  The prior shrink (step 4) needs the ratings' misses on games they never saw.  Misses
on the neighbouring seasons' games would teach it that a peak season comes back down the year after, and it would
shrink every peak season; so the misses come from the season's own games, held out.  notes/WITHIN_SEASON_LEDGER.md lists
every input, what it is built from and the check that proves it.

How.  A season's games (regular season and playoffs) are assigned to four folds at random, three times over (three
repeats), each assignment drawn with the seed (season, repeat); `--folds-per-repeat` and `--repeats` change both,
and step 4's folds are built with the four and the three.  For each fold:

    fit games    every game not in the fold.  Every input the rating reads (box score, playing time, starts,
                 tenure, on-court record) is rebuilt from these games alone (`data.panel.season_panel`, the function
                 `openrapm data panel` builds whole seasons with); the saved boosters of `--models` answer for the
                 rebuilt rows (step 2), and the ridge pulls the fit games' RAPM toward them (step 3), as
                 `openrapm rate` does for a whole season.  `leak_checks` proves nothing else was read.
    test games   the fold.  Nothing about them reaches the rating.  Stored: the test team-games (points per 100
                 as scored, each player's share of the possessions) and the rating's error on them.

The season so far, --cut SHARE (2026-10-23; the owner: "want to support 'at any point in the season' really").  In
place of the random folds, one fold a season: it is rated from the season so far on the day the first SHARE of its
regular-season games had been played (`data.panel.season_to_date`), every input rebuilt from those games as above, and
scored on every game after that day, the rest of the regular season and the playoffs.  Example: 2024 with --cut 0.25 is
rated from its first 308 games, through 2023-12-07, and scored on the other 1,004.  Boosters trained with part-season
rows (settings.PriorSettings.part_season_shares) are told the share of the season the rows cover (their `row_seasons`
input); every other booster reads 1, as for a whole season.  Folds built this way say `split` "date" and are never
step 4's (data/within_folds.py refuses them unless asked).

The season so far drawn at random, --so-far SHARE (experiment 108; the owner, 2026-10-23: "Love it, let's try it").
The games of the season so far, as --cut takes them, are put in folds as a whole season's are, four folds a repeat
and three repeats by default, and every game after that day is left out of the fold set altogether (`OUTSIDE`, -2 in
folds_<season>.parquet): each fold is rated from three quarters of the season so far and scored on the quarter of it
held back.  The shrink by share fits a season so far's multipliers on these, as step 4 fits a whole season's on the
random folds.  Example: 2024 at 0.25: its first 308 games in four folds of 77, rated from 231 and scored on 77, three
times over.  Folds built this way say `split` "so_far".

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
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import box_prior, decomposition, game_prior, game_rates, prior_rapm, second_prior, trade_set
from ..data import legacy_names
from ..data.careers import careers_before
from ..data.config import resolve
from ..data.designs import DesignCache
from ..data.game_rate_files import load_network
from ..data.panel import (
    DEFENCE_ALL,
    OFFENCE,
    SeasonPanel,
    SeasonSources,
    SeasonTables,
    fit_designs,
    season_panel,
    season_to_date,
    share_cap,
    trainable_panel,
)
from ..data.player_games import season_games
from ..data.roles import player_season_inputs, roles_path
from ..data.seasons import trainable
from ..data.second_prior_files import line_setup, load_grams
from ..data.shots import DEFENSE_TARGETS, OPPONENT_THREE_LUCK_KEPT, expected_threes
from ..data.tables import team_of_rows
from ..data.within_folds import OUTSIDE
from ..features import CLOSENESS, OFF_COURT, ON_MINUS_OFF, PANEL_KEYS, PRIOR_FEATURES, add_features
from ..scoring import Ratings, armse, predict_season, score
from ..settings import ROW_LABELS, TEST, UNSHRINK_LABELS, Targets
from . import parallel
from .build import game_box_priors, line_weights, season_lineup_rows

SIDES = ("O", "D")
REPEATS = 3                 # the repeats of a season's assignment, as step 4's folds are built (`--repeats`)
N_FOLDS = 4                 # the folds of a repeat: a quarter of the games held back (`--folds-per-repeat`)
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
# the training rows were labelled (`chunk_label`, settings.ROW_LABELS) is free: the folds only read the boosters.  So
# is which labels were un-shrunk, among settings.UNSHRINK_LABELS (experiment 102), for the same reason
EXPECTED_META = dict(exclude_neighbours=1, rows="chunks", player_folds=5)
UNSHRINK_ALLOWED = [list(columns) for columns in UNSHRINK_LABELS.values()]
# every column of the player tables: the prior's possible inputs, the off-court record, the score state
PLAYER_COLUMNS = list(dict.fromkeys(PRIOR_FEATURES + OFF_COURT + ON_MINUS_OFF + CLOSENESS))
# the share of the opponents' three-point luck each defensive target keeps (data/shots.py), for the outcome pieces
LUCK_KEPT = {"points_with_expected_opponent_threes": OPPONENT_THREE_LUCK_KEPT,
             "points_with_expected_opponent_threes_late_as_scored": OPPONENT_THREE_LUCK_KEPT,
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
    Only the penalty is compared: the ridge's other settings, the held prior scale of a season in progress
    (settings.RidgeSettings.held_scale_through, 2026-10-30), never reach a fold of a finished season.
    """
    saved = (meta.get("settings") or {}).get("ridge")
    if saved is not None and saved.get("penalty") != SETTINGS.ridge.penalty:
        return {"ridge": (saved, {"penalty": SETTINGS.ridge.penalty})}
    return {}


def check_models(meta: dict) -> None:
    """Refuse boosters that were not saved by a test-setting build of the shipped steps.

    `crossfit` "scale" is the earlier builds' cross-fitted scale: its fold priors equalled the full prior value
    for value (checked 2026-10-09), so the full-season fit here reproduces it.  "honest" rebuilt every input per
    cross-fitting fold, which this command does not.
    """
    wrong = {key: (meta.get(key), value) for key, value in EXPECTED_META.items() if meta.get(key) != value}
    if meta.get("chunk_label") not in ROW_LABELS:
        wrong["chunk_label"] = (meta.get("chunk_label"), ROW_LABELS)
    if meta.get("unshrink") not in UNSHRINK_ALLOWED:
        wrong["unshrink"] = (meta.get("unshrink"), UNSHRINK_ALLOWED)
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
        # an out-of-bag prior has no player folds (box_prior.OutOfBagPrior)
        folds_kept = "out_of_bag" in saved[side] or len(saved[side]["folds"]) == saved[side]["n_folds"]
        if saved[side]["groupings"] != 1 or not folds_kept:
            raise SystemExit(f"{season} {side}: the boosters average several fold groupings, which `predict` "
                             "does not; save them with one grouping")
    unseen = set(saved["O"]["unseen"]) & set(saved["D"]["unseen"])
    if not {season - 1, season, season + 1} <= unseen:
        raise SystemExit(f"{season}: the boosters trained on the season or a neighbour (unseen {sorted(unseen)})")
    return saved


def saved_booster(saved: dict):
    """One side's saved boosters as the box_prior object that built them: an OutOfPlayerPrior, a BaggedPrior when
    the file holds bags, or an OutOfBagPrior when it holds out-of-bag boosters (build.booster_layout)."""
    if "out_of_bag" in saved:
        return box_prior.OutOfBagPrior(models=[bag["model"] for bag in saved["out_of_bag"]],
                                       drawn_players=[set(bag["drawn"]) for bag in saved["out_of_bag"]],
                                       inputs=list(saved["model_feats"]))
    members = [box_prior.OutOfPlayerPrior(full=bag["full"], fold_models=list(bag["folds"]),
                                          held_out_players=list(bag["excluded"]), inputs=list(saved["model_feats"]))
               for bag in saved.get("bags", [saved])]
    return members[0] if "bags" not in saved else box_prior.BaggedPrior(members, list(saved["model_feats"]))


def prior_rows(panel: pd.DataFrame, side: str, saved: dict, share: float = 1.0) -> pd.DataFrame:
    """The panel rows one side's saved boosters answer for: that side's rows with possessions, derived, each covering
    `share` of a season (box_prior.rated_rows)."""
    return box_prior.rated_rows(panel[(panel.side == side) & (panel.possessions > 0)], saved["feats"], share)


def saved_prior(panel: pd.DataFrame, side: str, saved: dict, share: float = 1.0) -> dict:
    """{player_id: prior} from the saved boosters, each player answered by the fold model that never saw him."""
    rows = prior_rows(panel, side, saved, share)
    return dict(zip(rows.player_id.to_numpy(), box_prior.predict(saved_booster(saved), rows)))


def trained_on_part_seasons(meta: dict) -> bool:
    """Whether the saved boosters' training rows held part-season rows (settings.PriorSettings.part_season_shares), so
    their `row_seasons` input can say how much of a season a rated row covers."""
    return bool(((meta.get("settings") or {}).get("prior") or {}).get("part_season_shares"))


# ------------------------------------------------------------------------------------------------ the folds
def assign_folds(game_ids, n_folds: int, season: int, repeat: int) -> pd.Series:
    """game_id -> fold: the season's games put in a random order drawn with the seed (season, repeat, 9701) and
    assigned to folds 0, 1, 2, 3, 0, 1, ... in that order, so every fold has the same number of games to within one.

    Example: 1,315 games and four folds give folds of 329, 329, 329 and 328 games.
    """
    ids = np.sort(np.asarray(pd.unique(pd.Series(game_ids).astype(str))))
    rng = np.random.default_rng([int(season), int(repeat), 9701])
    return pd.Series(rng.permutation(len(ids)) % int(n_folds), index=ids, name=f"r{repeat}")


def so_far_folds(games: pd.DataFrame, share: float, n_folds: int, season: int, repeat: int) -> pd.Series:
    """game_id -> fold for --so-far: the games of the season so far (`season_to_date`) put in folds 0 to n_folds - 1
    as `assign_folds` puts a whole season's, with the same seed, and every game after that day `OUTSIDE`.

    Example: 2024 at 0.25 with four folds: its first 308 games in four folds of 77, the other 1,004 read -2.
    """
    ids = np.sort(np.asarray(pd.unique(games["game_id"].astype(str))))
    so_far, _ = season_to_date(games, share)
    drawn = assign_folds(sorted(so_far), n_folds, season, repeat)
    return pd.Series(ids, index=ids).map(drawn).fillna(OUTSIDE).astype(int).rename(f"r{repeat}")


def date_folds(games: pd.DataFrame, share: float) -> pd.Series:
    """game_id -> fold for the season so far (--cut): -1 for a game of the season so far (`season_to_date`), which
    the rating is fitted on, and 0 for every game after it, held back as fold 0 of repeat 0.

    Example: 2024 at 0.25: its first 308 games read -1, the other 1,004, playoffs included, read 0.
    """
    ids = np.sort(np.asarray(pd.unique(games["game_id"].astype(str))))
    so_far, _ = season_to_date(games, share)
    return pd.Series(np.where(np.isin(ids, sorted(so_far)), -1, 0), index=ids, name="r0")


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
    checks["self-creation: counts read only from fit games"] = inputs.games_read["self-creation counts"] <= fit_ids
    checks["fouls and blocked shots: counts read only from fit games"] = inputs.games_read["foul counts"] <= fit_ids
    # the per-game hustle and tracking box scores (the key kept its first name)
    checks["hustle: per-game counts read only from fit games"] = inputs.games_read.get("hustle games", set()) <= fit_ids
    checks["matchups: per-game pairs read only from fit games"] = (inputs.games_read.get("matchup games", set())
                                                                    <= fit_ids)
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
               context: ContextInputs | None = None, settings=SETTINGS, share: float = 1.0,
               scale=None, types_panel: pd.DataFrame | None = None, held_weight: float = 0.0) -> RatedWorld:
    """Steps 2-3 on the world's rebuilt inputs: the saved boosters' priors, both ridges, the centred table.

    `prior_panel`, when given, is what the boosters read instead of the world's own rows (the leak control, the
    module docstring); the ridge reads the world's designs either way.  With `context`, step 3 reads the second prior
    built from it instead of the boosters' answers.  `settings` names the targets the ridge reads (`fold_settings`).
    `share` is the share of a season the boosters are told the rows cover (`prior_rows`); `scale`, (s_O, s_D), holds
    the ridge's prior scales (prior_rapm.held_prior_scale, experiment 109); `types_panel`, the rows the second prior's
    step-5 numbers are read from when not the boosters' (the network's whole-season rows, `rate_world_by_game`);
    `held_weight`, the share of the way from the held scales to the fit's own (prior_rapm.rate_season_blended), 0 held.
    Adds one check to the world's: the ridge's possessions are the fit games' (five players on every row).
    """
    panel = world.panel if prior_panel is None else prior_panel
    box = {side: saved_prior(panel, side, saved[side], share) for side in SIDES}
    prior = box
    if context is not None:
        context_panel = panel if context.panel is None else context.panel
        answers = {side: second_prior.booster_answers(saved_booster(saved[side]),
                                                      prior_rows(context_panel, side, saved[side], share))
                   for side in SIDES}
        # step 5's numbers from the offence rows with possessions, the rows the build reads (a player with
        # defensive possessions and no offensive ones has none there, and reads 0, the mean)
        typed = panel if types_panel is None else types_panel
        types = (None if context.type_scale is None else second_prior.standardised_types(
            typed[(typed.side == "O") & (typed.possessions > 0)], context.type_scale))
        prior, _ = second_prior.season_second_priors(context.stints, answers, box, context.coefficients,
                                                     context.n_folds, context.team_totals, types, context.input_set)
    designs = {target: world.designs[target] for target in settings.targets.distinct()}
    table = prior_rapm.rate_season_blended(designs, prior, settings, world.season, scale, held_weight)
    world.checks["ridge: possessions are the fit games' possessions"] = bool(np.isclose(
        table.possessions.sum(), float(world.designs[settings.targets.offence].rows["poss"].sum()) * 5.0, rtol=1e-9))
    return RatedWorld(world=world, table=table, prior=prior, box=None if context is None else box)


def rate_world_by_game(world: SeasonWorld, saved: dict, network, games: pd.DataFrame, least_share: float,
                       context: ContextInputs | None = None, settings=SETTINGS, scale=None,
                       backward=None, whole_season: bool = False) -> RatedWorld:
    """`rate_world` with the box-score prior game by game (game_prior.py), as `openrapm rate --game-prior` rates a
    season (build.rate_one_season_by_game): the network's state before every fit game from the fit games alone, the
    saved boosters' answer for each, each player's average over his fit games, the second prior on that average and
    spread over his games by the line's box-score coefficient, and step 3 pulling every row toward its own game's
    prior.  The held-back games leave every player's sequence, the opponents' allowed rates and the league levels.
    With `backward`, the season's backward network, the first half of each player's fit games take their state from
    the fit games after them (game_prior.splice), as `openrapm rate --game-prior-backward` rates the season.  With
    `whole_season`, one prior a season, as `--game-prior-whole-season` rates it: the saved boosters' answer for the
    network's state after each player's last fit game, read by `rate_world`, the second prior's step-5 numbers from the
    world's own panel as the build reads them from the season panel.

    Adds two checks to the world's: every game row read is a fit game, and the network never trained on the season or
    its neighbours."""
    fit = set(world.designs[SCORE_TARGET].games.set_index("game_idx").loc[world.fit_idx, "game_id"].astype(str))
    rows = games[games.game_id.astype(str).isin(fit).to_numpy()].reset_index(drop=True)
    world.checks["game rows: only from fit games"] = set(rows.game_id.astype(str)) <= fit
    world.checks["network: never trained on this season or its neighbours"] = not (
        {world.season - 1, world.season, world.season + 1} & {int(s) for s in network.seasons})
    states_games, states = game_prior.season_states(rows, network)
    # the season panel's rows with possessions, as `openrapm rate` hands them over (build.rate_table): a player with
    # no possessions on one end reads no listing there in either, so the whole-season rebuild reproduces the build
    base = world.panel[world.panel.possessions > 0]
    whole, _ = game_prior.network_panels(states_games, states, base, ())
    if whole_season:
        whole = whole[whole.possessions > 0].reset_index(drop=True)
        return rate_world(world, saved, whole, context, settings, 1.0, scale, types_panel=world.panel)
    game_rows = game_prior.game_rows(states_games, states, base)
    if backward is not None:
        game_rows = game_prior.splice(game_rows, game_prior.backward_game_rows(rows, backward, base))
        world.checks["backward network: never trained on this season or its neighbours"] = not (
            {world.season - 1, world.season, world.season + 1} & {int(s) for s in backward.seasons})
    boosters = {side: box_prior.SidePrior(values={}, model=saved_booster(saved[side]),
                                          feature_names=list(saved[side]["feats"])) for side in SIDES}
    game_rows["box_game"] = game_box_priors(game_rows, boosters, least_share)
    box = game_prior.season_average(game_rows, "box_game")
    prior = box
    if context is not None:
        answers = {side: second_prior.booster_answers(boosters[side].model, prior_rows(whole, side, saved[side]))
                   for side in SIDES}
        types = (None if context.type_scale is None else second_prior.standardised_types(
            world.panel[(world.panel.side == "O") & (world.panel.possessions > 0)], context.type_scale))
        prior, _ = second_prior.season_second_priors(context.stints, answers, box, context.coefficients,
                                                     context.n_folds, context.team_totals, types, context.input_set)
        weights = line_weights(context.coefficients, context.n_folds, game_rows)
        game_rows["prior_game"] = game_prior.per_game_second(game_rows, "box_game", prior, box, weights)
    else:
        game_rows["prior_game"] = game_rows["box_game"]
    designs = {target: world.designs[target] for target in settings.targets.distinct()}
    tables = {target: game_prior.game_tables(design, game_rows, "prior_game", prior)
              for target, design in designs.items()}
    table = prior_rapm.rate_season(designs, prior, settings, world.season, scale, game_tables=tables)
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
    n_folds: int = N_FOLDS       # the folds of each repeat: 1 / n_folds of the season's games held back
    repeats: int = REPEATS
    settings: object = SETTINGS  # the test setting with the saved build's own targets (`fold_settings`)
    cut: float | None = None     # --cut or --so-far: the season so far at this share of the regular season, else None
    split: str = "random"        # data.within_folds.SPLITS: "date" with --cut, "so_far" with --so-far
    part_season_rows: bool = False   # the boosters trained with part-season rows (`trained_on_part_seasons`)
    held_scale: tuple | None = None  # --hold-prior-scale: (s_O, s_D) every fold's ridge holds, the whole season never
    held_weight: float = 0.0     # --blend-through: the share of the way from those to the fold's own fitted scales
    network: object = None       # with the box-score prior game by game, the season's network (game_prior.py)
    backward: object = None      # and its backward network, when the first half of each player's games read the rest
    games: pd.DataFrame | None = None    # and the season's game rows (data/player_games.py)
    least_share: float = 1.0     # and the smallest part-season share its boosters trained on (build.least_share)
    whole_season: bool = False   # and one prior a season from its state after the last game (game_prior_whole_season)

    @property
    def season_share(self) -> float | None:
        """With --cut or --so-far, the share of the regular season the season so far holds (`season_to_date`); else
        None."""
        return None if self.cut is None else season_to_date(self.full[SCORE_TARGET].games, self.cut)[1]

    def prior_share(self, fit_ids) -> float:
        """The share of a season the boosters are told the fit games cover: the season so far's share for boosters
        trained with part-season rows, 1 otherwise and for the whole season."""
        whole = {str(game) for game in fit_ids} == set(self.game_ids)
        return 1.0 if whole or self.cut is None or not self.part_season_rows else float(self.season_share)

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
        whole = {str(game) for game in fit_ids} == set(self.game_ids)
        if self.network is not None:
            return rate_world_by_game(world, self.saved, self.network, self.games, self.least_share, context,
                                      self.settings, None if whole else self.held_scale, self.backward,
                                      self.whole_season)
        return rate_world(world, self.saved, self.prior_panel, context, self.settings, self.prior_share(fit_ids),
                          None if whole else self.held_scale, held_weight=self.held_weight)

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
    file: the assignment is seeded, so a rerun must draw the same folds); with --cut, the one date fold
    (`date_folds`); with --so-far, the season so far's (`so_far_folds`)."""
    path = run.out_dir / f"folds_{run.season}.parquet"
    games = run.full[SCORE_TARGET].games
    if run.split == "date":
        folds = date_folds(games, run.cut).to_frame()
    elif run.split == "so_far":
        folds = pd.concat([so_far_folds(games, run.cut, run.n_folds, run.season, repeat)
                           for repeat in range(run.repeats)], axis=1)
    else:
        folds = pd.concat([assign_folds(run.game_ids.to_numpy(), run.n_folds, run.season, repeat)
                           for repeat in range(run.repeats)], axis=1)
    if path.exists():
        earlier = pd.read_parquet(path)
        common = [column for column in earlier.columns if column in folds.columns]
        if not earlier[common].equals(folds.loc[earlier.index, common]):
            raise SystemExit(f"{path} holds other folds than the seeded ones")
    folds.to_parquet(path)
    return folds


def fold_checks(run: SeasonRun, rated: RatedWorld, test_checks: dict, fit_ids: set, test_ids: set,
                universe: set | None = None) -> dict:
    """Every leak check of one fold, in the order the fold file lists them.  `universe` is every game of the fold set,
    the whole season unless --so-far leaves the games after the season so far out; the check keeps its name."""
    unseen = set(run.saved["O"]["unseen"]) & set(run.saved["D"]["unseen"])
    checks = dict(test_checks)
    checks.update(rated.world.checks)
    checks["folds: fit and test games disjoint"] = not (fit_ids & test_ids)
    checks["folds: fit and test games cover the season"] = (fit_ids | test_ids) == (
        set(run.game_ids) if universe is None else universe)
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
    column = folds[f"r{repeat}"]
    test_ids = set(folds.index[column == fold])
    universe = set(folds.index[column != OUTSIDE])
    fit_ids = universe - test_ids
    games = run.full[SCORE_TARGET].games
    test_idx = np.sort(games.loc[games["game_id"].astype(str).isin(test_ids), "game_idx"].to_numpy())
    rated = run.rate(fit_ids)
    pooled, teams, test_poss, summary, test_checks = score_test_games(rated, run.full[SCORE_TARGET], test_idx,
                                                                      run.cfg)
    checks = fold_checks(run, rated, test_checks, fit_ids, test_ids, universe)
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
    # rate_from, split and cut name the fold shape: rated from the rest of the fold set, its games drawn at random
    # over the season ("random", cut None), the season so far against the rest ("date", cut the share asked for,
    # season_share the share it holds) or drawn at random over the season so far ("so_far")
    summary.update(season=run.season, repeat=repeat, fold=fold, folds_per_repeat=run.n_folds, rate_from="rest",
                   fit_games=len(fit_ids), split=run.split, cut=run.cut,
                   season_share=run.season_share, rated_share=run.prior_share(fit_ids),
                   test_games=len(test_ids), prior_inputs=run.prior_inputs,
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
    for repeat in range(run.repeats):
        for fold in range(run.n_folds):
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


@dataclass
class FoldLines:
    """The second prior's grams and settings, as the saved boosters' build read them (`second_prior_lines`)."""
    grams: dict
    folds: int
    penalty: float
    team_totals: str
    type_scale: dict | None
    nearest: int
    input_set: str | None
    trainable: set               # the seasons a line may be fitted on


def fold_lines(cfg: dict, meta: dict, whole_season_context: bool) -> FoldLines | None:
    """The second prior the folds rate with, None when the boosters were saved without one; says which."""
    grams, line_folds, line_penalty, team_totals, record = second_prior_lines(cfg, meta)
    if whole_season_context and grams is None:
        raise SystemExit("--whole-season-context is the second prior's leak arm; these boosters were saved without one")
    if grams is None:
        return None
    print(f"the second prior: step 3 reads it, each season's line from the grams of "
          f"{meta['settings']['prior']['second_prior']} more than {SETTINGS.exclude_neighbours} season away, "
          f"team totals {team_totals}"
          + ("; the leak arm: its context numbers from the WHOLE season" if whole_season_context else ""),
          flush=True)
    return FoldLines(grams, line_folds, line_penalty, team_totals, record.get("type_scale"), record["nearest"],
                     ((meta.get("settings") or {}).get("prior") or {}).get("second_prior_inputs"),
                     set(trainable(cfg).seasons))


def attach_line(run: SeasonRun, lines: FoldLines, whole_season_context: bool) -> None:
    """The season's second-prior line and the stint rows its context numbers read, on `run`."""
    run.line = second_prior.season_coefficients(lines.grams, run.season, lines.trainable, SETTINGS.exclude_neighbours,
                                                lines.folds, lines.penalty, lines.nearest)
    run.line_folds, run.line_team_totals, run.line_type_scale = lines.folds, lines.team_totals, lines.type_scale
    run.line_input_set = lines.input_set
    run.stints = season_lineup_rows(run.cfg, run.season)
    if run.line is None:
        raise SystemExit(f"{run.season}: no season of the grams may train its second prior's line")
    if whole_season_context:
        run.context_panel = run.world(set(run.game_ids)).panel


def cut_words(cut: float | None, part_season_rows: bool, split: str = "date") -> str:
    """The fold design in words, for the run's first lines."""
    if cut is None:
        return ""
    told = ("its share of the season" if part_season_rows
            else "1, a whole season: they trained without part-season rows")
    if split == "so_far":
        return (f"the season so far drawn at random: each season's games through the day {cut:g} of its regular season "
                f"had been played, in folds; every later game left out; the boosters are told {told}")
    return (f"the season so far: each season rated from its games through the day {cut:g} of its regular season had "
            f"been played, scored on the rest; the boosters are told {told}")


def game_prior_tag(meta: dict, pieces: bool, whole_season_inputs: bool, whole_season_context: bool) -> str:
    """The multi-stat networks' tag when the saved boosters' build read the box-score prior game by game, else "";
    said once.  --pieces, --whole-season-inputs and --whole-season-context rebuild one prior per player-season and are
    refused with it."""
    tag = ((meta.get("settings") or {}).get("prior") or {}).get("game_prior") or ""
    if tag and (pieces or whole_season_inputs or whole_season_context):
        raise SystemExit("--pieces, --whole-season-inputs and --whole-season-context read one prior per player-season; "
                         "these boosters' build read the box-score prior game by game")
    if tag and ((meta.get("settings") or {}).get("prior") or {}).get("game_prior_whole_season"):
        print(f"the box-score prior once a season from the networks of outputs/game_rates/{tag}/: each fold's state "
              f"after every player's last fit game, the fit games alone", flush=True)
    elif tag:
        print(f"the box-score prior game by game: each fold's state before every fit game from the networks of "
              f"outputs/game_rates/{tag}/, the fit games alone", flush=True)
    return tag


def attach_networks(run: SeasonRun, meta: dict, tag: str) -> None:
    """The season's networks of the box-score prior game by game, forward and, when the build read one, backward, its
    game rows and the smallest part-season share its boosters trained on, on `run`."""
    prior = (meta.get("settings") or {}).get("prior") or {}
    run.network = load_network(run.cfg, tag, run.season, game_rates.input_names(), game_rates.NAMES)
    if prior.get("game_prior_backward"):
        run.backward = load_network(run.cfg, prior["game_prior_backward"], run.season, game_rates.input_names(),
                                    game_rates.NAMES)
    run.games = season_games(run.cfg, run.season)
    shares = prior.get("part_season_shares") or ()
    run.least_share = float(min(shares)) if shares else 1.0
    run.whole_season = bool(prior.get("game_prior_whole_season"))


def held_scale_table(outputs: Path, name: str | None) -> pd.DataFrame | None:
    """--hold-prior-scale: outputs/<name>.parquet, whose seasons' prior scales the folds hold; None without."""
    if name is None:
        return None
    print(f"every fold's prior scales held at the mean of {name}'s 8 seasons nearest its own", flush=True)
    return pd.read_parquet(outputs / f"{name}.parquet")


def season_held_scale(scales: pd.DataFrame | None, season: int) -> tuple | None:
    """(s_O, s_D) a season's folds hold (prior_rapm.held_prior_scale), said once; None without a table."""
    if scales is None:
        return None
    held = prior_rapm.held_prior_scale(scales, season)
    print(f"  {season}: prior scales held at {held[0]:.3f} / {held[1]:.3f}", flush=True)
    return held


def blend_run(run: "SeasonRun", blend_through: float | None, blend_from: float = 0.0) -> None:
    """--blend-through: the run's folds hold their prior scales a straight line from the held ones toward their own
    fitted ones, by the season so far's share from `blend_from` to `blend_through` (`SeasonRun.held_weight`), said
    once.  Only with held scales and a season so far (--hold-prior-scale, --cut)."""
    if blend_through is None or run.held_scale is None or run.cut is None:
        return
    run.held_weight = prior_rapm.blend_weight(run.season_share, blend_through, blend_from)
    print(f"  {run.season}: the prior scales {run.held_weight:.0%} of the way from held to the fit's own, "
          f"{run.season_share:.1%} of the season played", flush=True)


def boosters_identity(outputs: Path, models: str) -> dict | None:
    """The saved boosters a fold folder is built from: outputs/prior_models_<models>.pkl's name, size and modified
    time; None when the file is not there under that name."""
    path = outputs / f"prior_models_{models}.pkl"
    if not path.exists():
        return None
    stat = path.stat()
    return {"boosters": path.name, "bytes": int(stat.st_size), "modified_ns": int(stat.st_mtime_ns)}


def check_fold_boosters(out_dir: Path, identity: dict | None) -> None:
    """One fold folder holds the folds of one set of boosters (2026-10-30).  A fold already on disk is skipped, and
    its reproduction check with it, so a rerun under one tag after the boosters were saved again would have mixed two
    builds without a word.  The first run writes boosters.json; a later run with other boosters stops.  A folder built
    before the record existed takes the record of its first run since."""
    if identity is None:
        return
    record = out_dir / "boosters.json"
    try:
        with open(record, "x", encoding="utf-8") as handle:       # the first process of a run writes it
            handle.write(json.dumps(identity))
        return
    except FileExistsError:
        pass
    # the processes of one run (`folds_in_parts`) start together: a record another one is still writing reads empty
    # or, on Windows, refuses to open (2026-10-30, the first night: PermissionError stopped a part)
    for _ in range(50):
        try:
            held = json.loads(record.read_text(encoding="utf-8"))
            break
        except (OSError, ValueError):
            time.sleep(0.1)
    else:
        return
    if held != identity:
        raise SystemExit(f"{out_dir} holds folds built from {held.get('boosters')} as it was then "
                         f"({held.get('bytes')} bytes); {identity['boosters']} is another file now "
                         f"({identity['bytes']} bytes).  Use another --tag, or remove the folder")


def run_folds(cfg: dict, models: str, base: str, seasons, tag: str, reproduce_only: bool = False,
              pieces: bool = False, whole_season_inputs: bool = False, whole_season_context: bool = False,
              n_folds: int = N_FOLDS, repeats: int = REPEATS, cut: float | None = None,
              so_far: float | None = None, hold_prior_scale: str | None = None,
              blend_through: float | None = None, blend_from: float = 0.0) -> int:
    """`openrapm folds`: the reproduction check and the folds of every season, then the gathered files.  With
    `pieces`, each fold also writes its pieces_<stem>.parquet (`fold_pieces`); with `whole_season_inputs`, every
    fold's boosters read the whole season's inputs (the leak control, the module docstring); with
    `whole_season_context`, every fold's second prior reads the whole season's context numbers (its leak arm).
    `folds_in_parts` runs it in several processes at once.  `n_folds` and `repeats` are the fold design, four and
    three as step 4's folds are built; with `cut`, one date fold a season, the season so far at that share of its
    regular season against the rest; with `so_far`, the season so far at that share drawn at random into the fold
    design (the module docstring).  With `hold_prior_scale`, a table under outputs/, every fold's ridge holds the
    prior scales at the mean of that table's 8 seasons nearest its season (prior_rapm.held_prior_scale, experiment
    109); the whole season, the reproduction, is fitted as shipped.  With `blend_through` too, the scales go a straight
    line from held at a season's first game to its fit's own at that share (prior_rapm.blend_weight)."""
    if cut is not None and so_far is not None:
        raise SystemExit("--cut and --so-far are two fold designs; ask for one")
    split = "date" if cut is not None else "so_far" if so_far is not None else "random"
    cut = so_far if so_far is not None else cut
    box_files = None
    if pieces:
        from .site import BoxScoreFiles
        box_files = BoxScoreFiles(cfg)
    outputs = resolve(cfg, "outputs")
    out_dir = resolve(cfg, "within") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    check_fold_boosters(out_dir, boosters_identity(outputs, models))
    saved_all = load_models(outputs, models, seasons)
    meta = saved_all.get("meta", {})
    check_models(meta)
    settings = fold_settings(meta)
    part_season_rows = trained_on_part_seasons(meta)
    game_tag = game_prior_tag(meta, pieces, whole_season_inputs, whole_season_context)
    if split == "date":
        n_folds, repeats = 1, 1
    base_table = pd.read_parquet(outputs / f"{base}.parquet")
    panel = trainable_panel(cfg)
    roles = pd.read_parquet(roles_path(cfg))
    inputs = player_season_inputs(roles, cap=share_cap(cfg))
    debut = careers_before(cfg, roles)            # the seasons before 1997, as the season panel read them
    designs = DesignCache(cfg)
    print(f"models {models} ({meta.get('argv')}); seasons {seasons[0]}-{seasons[-1]}; {repeats} repeats of "
          f"{n_folds} folds; targets {settings.targets.offence} / {settings.targets.defence}; writing {out_dir}",
          flush=True)
    if whole_season_inputs:
        print("the leak control: every fold's boosters read the WHOLE season's inputs, held-back games included",
              flush=True)
    if cut is not None:
        print(cut_words(cut, part_season_rows, split), flush=True)
    scales = held_scale_table(outputs, hold_prior_scale)
    lines = fold_lines(cfg, meta, whole_season_context)
    started = time.time()
    for season in seasons:
        saved = season_models(saved_all, season)
        full = {target: designs.season(season, target) for target in world_targets(settings)}
        run = SeasonRun(season, cfg, full, SeasonSources(SeasonTables(season, cfg), roles, inputs, debut), saved,
                        out_dir, box_files, settings=settings, n_folds=n_folds, repeats=repeats, cut=cut,
                        split=split, part_season_rows=part_season_rows,
                        held_scale=season_held_scale(scales, season))
        blend_run(run, blend_through, blend_from)
        if whole_season_inputs:
            run.prior_panel = run.world(set(run.game_ids)).panel
        if lines is not None:
            attach_line(run, lines, whole_season_context)
        if game_tag:
            attach_networks(run, meta, game_tag)
        run_season(run, base_table, panel, reproduce_only, started)
        designs.clear()
    gather(out_dir)
    return 0
