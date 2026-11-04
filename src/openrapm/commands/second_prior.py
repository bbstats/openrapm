"""`openrapm second-prior`: the swaps the second prior's line is fitted on (second_prior.py), from the held-back games
of within-season folds.

    openrapm second-prior --models NAME --folds TAG[,TAG] --tag TAG [--seasons 1997-2026] [--inputs context_and_types]

For every fold of every season in the folds named:

    1. Every player of the fold's fit games is answered for by every booster the fold was rated with (`--models`,
       outputs/prior_models_<NAME>.pkl).  His box-score prior from those answers must equal the one the fold stored
       (prior_raw_off, prior_raw_def) to 1e-9, or the run stops: the folds were rated with other boosters.
    2. His eight context numbers, from the fit games' stint rows, every prior in them from the boosters that never saw
       him (second_prior.context_numbers).
    With --inputs context_and_types (experiment 82), or context_and_types_centred (experiment 93, the default: the
    same inputs, centred only when the line is applied), every player also carries his other end's box-score prior and
    step 5's 19 numbers from the fold's players file, the fit games', standardised with the trainable panel's means
    and standard deviations, which run.json keeps for the build.
    With --inputs context_types_team_rates_centred (experiment 128), also his team's 16 rates with him on the court and
    off it, from the fit games' stint rows (second_prior.team_rate_numbers).
    3. The swaps of the held-back games, scored against the fold's step-4 ratings at the test setting's multipliers,
       give the fold's share of the line's normal equations (second_prior.held_back_grams).  The fit games and the
       held-back games never share a game, so a shot counted in a box score never meets the lineup result it scored.
       With --target prior_part (experiment 130) only the fold's prior parts are added back to each swap's residual,
       not its whole ratings.

With --target rating (rating_line.py, experiment 134) step 3 replaces the swaps: each fold's ridge on its fit games and
its held-back stints give the fold's share of one set of normal equations for both ends, at the fold season's within
layer of step 4.  Writes sums_<season>.npz in place of the grams.

Writes outputs/second_prior/<TAG>/ (data/second_prior_files.py): one grams_<season>.npz and inputs_<season>.parquet per
season, and run.json.  A season already written is skipped, so a stopped run resumes where it stopped.

Example: `openrapm second-prior --models bags10 --folds bags10_early,bags10_within --tag ctx81` reads the incumbent's
360 folds, 1997-2026; `openrapm rate --setting test --second-prior ctx81` then rates every season on the line.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import prior_rapm, prior_shrink, rating_line, second_prior, swap_adjust, swap_test
from ..data.config import resolve
from ..data.designs import DesignCache
from ..data.legacy_names import translate_frame
from ..data.panel import fit_designs, trainable_panel
from ..data.second_prior_files import (
    grams_directory,
    grams_path,
    rating_sums_path,
    write_grams,
    write_rating_sums,
    write_run,
)
from ..data.within_folds import fold_games
from ..rapm import design_blocks
from ..settings import PRODUCT, TEST
from .build import season_lineup_rows, step4_folds, step4_multipliers, within_layer_multipliers
from .within_season import check_models, load_models, prior_rows, saved_booster, season_models

SIDES = ("O", "D")
PRIOR_TOLERANCE = 1e-9          # the boosters' answers against the prior the fold stored
RIDGE_TOLERANCE = 1e-3          # the fit designs' ridge against the ratings the fold stored (`check_fit_designs`)


def fold_players(directory: Path, stem: str) -> pd.DataFrame:
    """A fold's players_<stem>.parquet under the plain names: one row per player and side of its fit games."""
    return translate_frame(pd.read_parquet(Path(directory) / f"players_{stem}.parquet"))


def stored_box(players: pd.DataFrame) -> dict:
    """{"O", "D"}: {player_id: the box-score prior the fold rated with}, raw sign (on defence, points allowed)."""
    offence = players[(players.side == "O") & players.prior_raw_off.notna()]
    defence = players[(players.side == "D") & players.prior_raw_def.notna()]
    return {"O": pd.Series(offence.prior_raw_off.to_numpy(float), index=offence.player_id.to_numpy()),
            "D": pd.Series(-defence.prior_raw_def.to_numpy(float), index=defence.player_id.to_numpy())}


def fold_answers(players: pd.DataFrame, saved: dict) -> tuple[dict, float]:
    """({"O", "D"}: every saved booster's answer for the fold's players, second_prior.BoosterAnswers), the largest gap
    between the box-score prior those answers give and the one the fold stored."""
    answers = {side: second_prior.booster_answers(saved_booster(saved[side]), prior_rows(players, side, saved[side]))
               for side in SIDES}
    box = stored_box(players)
    gap = max(float(np.nanmax(np.abs(second_prior.own_priors(answers[side]).reindex(box[side].index).to_numpy()
                                     - box[side].to_numpy()))) for side in SIDES)
    return answers, gap


def fold_grams(fold, directory: Path, stints: pd.DataFrame, saved: dict, factors: tuple, n_folds: int,
               scale: dict | None = None, input_set: str | None = None, target: str = "whole_rating"):
    """One fold: ({side: PairGrams}, its players' inputs in one frame, the prior check's largest gap, swaps used)."""
    inputs, gap, fit_stints, held_ids = fold_inputs(fold, directory, stints, saved, scale, input_set)
    game = stints.game_id.astype(str)
    held = swap_test.prepare(stints[game.isin(held_ids).to_numpy()].reset_index(drop=True), fold.season)
    table = prior_shrink.shrink_fold(fold.rows, fold.season, factors)
    ratings = swap_test.ratings_in_raw_sign(table)
    added_back = swap_test.ratings_in_raw_sign(table, "prior") if target == "prior_part" else None
    grams = second_prior.held_back_grams(held, ratings, inputs, n_folds, added_back)
    return grams, inputs_frame(inputs, fold), gap, {side: grams[side].n for side in grams}


def inputs_frame(inputs: dict, fold) -> pd.DataFrame:
    """A fold's line inputs in one frame: season, repeat, fold, side, player_id and the inputs."""
    frame = pd.concat([inputs[side].reset_index().assign(side=side) for side in SIDES], ignore_index=True)
    return frame.assign(season=fold.season, repeat=fold.repeat, fold=fold.fold)


def fold_inputs(fold, directory: Path, stints: pd.DataFrame, saved: dict, scale: dict | None, input_set: str | None):
    """One fold's line inputs: ({"O", "D"}: from its fit games, the prior check's largest gap, the fit games' stint
    rows, the held-back game ids)."""
    fit_ids, held_ids = fold_games(directory, fold)
    players = fold_players(directory, fold.stem)
    answers, gap = fold_answers(players, saved)
    game = stints.game_id.astype(str)
    fit_stints = stints[game.isin(fit_ids).to_numpy()]
    context = second_prior.context_numbers(fit_stints, answers["O"], answers["D"])
    box = stored_box(players)
    types = None if scale is None else second_prior.standardised_types(players[players.side == "O"], scale)
    team_rates = second_prior.team_rate_numbers(fit_stints) if second_prior.reads_team_rates(input_set) else None
    return second_prior.side_inputs(box, context, types, input_set, team_rates), gap, fit_stints, held_ids


@dataclass
class RatingSeason:
    """What every fold of one season needs for the line fitted on the rating."""
    cfg: dict
    whole: dict                  # {target: the season's design}: "pts" and the two targets step 3 fits
    players: object              # the "pts" design's player columns, every stint of the season
    within: tuple                # the season's within layer, prior_shrink.WITHIN_COLUMNS' order
    penalty: float


def held_back_stints(season: RatingSeason, directory: Path, fold) -> rating_line.HeldBack:
    """The fold's held-back stints, keyed by its test file as the within-season test keys them
    (scripts/checks/151_wowy.py)."""
    full = season.whole["pts"]
    test = np.load(Path(directory) / f"test_{fold.stem}.npz", allow_pickle=True)
    key = pd.DataFrame({"game_id": test["game_id"].astype(str), "is_home_off": test["is_home_off"].astype(bool),
                        "team_off": test["team_off"], "team_def": test["team_def"]})
    games = full.games.set_index("game_idx").game_id.astype(str)
    rows = full.rows.assign(game_id=full.rows.game_idx.map(games).to_numpy())
    rows = rows.reset_index().merge(key, on=["game_id", "is_home_off"], how="inner")
    index = rows["index"].to_numpy()
    ids = full.spec.players["player_id"].to_numpy(dtype=np.int64)
    rated = np.concatenate([np.isin(ids, fold.rows["O"].player_id.to_numpy()),
                            np.isin(ids, fold.rows["D"].player_id.to_numpy())])
    levels = rating_line.team_levels(rows.team_off, rows.team_def, rows.is_home_off, rows.phase == "PO")
    return rating_line.HeldBack(players=season.players[index], y=np.asarray(full.y, dtype=float)[index],
                                w=np.asarray(full.w, dtype=float)[index], levels=levels, rated=rated,
                                stand_in=(fold.fill_o, fold.fill_d))


def check_fit_designs(fold, designs: dict, penalty: float) -> float:
    """The largest gap between step 3 refitted on `designs` and the ratings the fold stored, each end's level aside:
    these are the fold's own fit games and targets, or the sums would read another ridge than the folds rate with."""
    worst = 0.0
    offence, defence = fold.rows["O"].set_index("player_id"), fold.rows["D"].set_index("player_id")
    prior_o, prior_d = offence.second_prior_raw_off.to_dict(), (-defence.second_prior_raw_def).to_dict()
    for target, stored, column in ((TEST.targets.offence, offence.prior_off + offence.u_off, "offense"),
                                   (TEST.targets.defence, -(defence.prior_def + defence.u_def), "defense")):
        fitted = prior_rapm.fit(designs[target], prior_o, prior_d, penalty).ratings.set_index("player_id")[column]
        gap = (fitted.reindex(stored.index) - stored).dropna()
        worst = max(worst, float((gap - gap.mean()).abs().max()))
    return worst


def fold_rating_sums(fold, directory: Path, stints: pd.DataFrame, saved: dict, season: RatingSeason,
                     scale: dict | None, input_set: str | None, check: bool):
    """One fold: (its rating_line.RatingSums, its players' inputs in one frame, the prior check's largest gap)."""
    inputs, gap, fit_stints, _ = fold_inputs(fold, directory, stints, saved, scale, input_set)
    fit_ids, _ = fold_games(directory, fold)
    designs = fit_designs(fold.season, season.cfg, season.whole, fit_ids)
    if check:
        ridge_gap = check_fit_designs(fold, designs, season.penalty)
        if not ridge_gap < RIDGE_TOLERANCE:
            raise SystemExit(f"{fold.stem}: step 3 refitted here misses the fold's ratings by {ridge_gap:.1e}")
    design = designs[TEST.targets.offence]
    players, context = design_blocks(design)
    ids = design.spec.players["player_id"].to_numpy(dtype=np.int64)
    team_poss = {side: second_prior.team_possessions(fit_stints, side) for side in SIDES}
    priors = rating_line.prior_columns(inputs, ids, team_poss)
    offence, defence, explained = rating_line.ridge_parts(
        players, context, np.asarray(design.w, dtype=float), np.asarray(design.y, dtype=float),
        np.asarray(designs[TEST.targets.defence].y, dtype=float), priors, season.penalty)
    sums = rating_line.held_back_sums(held_back_stints(season, directory, fold), (offence, defence), priors, explained,
                                      season.within)
    return sums, inputs_frame(inputs, fold), gap


def rating_season(cfg: dict, season: int, season_folds: list, where: dict, saved: dict, within: pd.DataFrame,
                  scale: dict | None, input_set: str, out_dir: Path, designs: DesignCache) -> float:
    """One season's sums_<season>.npz and inputs_<season>.parquet, the line fitted on the rating; its largest gap."""
    stints = season_lineup_rows(cfg, season)
    whole = {target: designs.season(season, target) for target in ("pts", TEST.targets.offence, TEST.targets.defence)}
    row = within.loc[season]
    shared = RatingSeason(cfg, whole, design_blocks(whole["pts"])[0],
                          tuple(float(row[column]) for column in prior_shrink.WITHIN_COLUMNS), TEST.ridge.penalty)
    total, frames, worst = None, [], 0.0
    for number, fold in enumerate(season_folds):
        sums, frame, gap = fold_rating_sums(fold, where[fold.stem], stints, saved, shared, scale, input_set,
                                            number == 0)
        total = sums if total is None else total + sums
        frames.append(frame)
        worst = max(worst, gap)
    pd.concat(frames, ignore_index=True).to_parquet(out_dir / f"inputs_{season}.parquet", index=False)
    write_rating_sums(rating_sums_path(out_dir, season), total)
    return worst


def run_second_prior(cfg: dict, models: str, folds_tag: str, tag: str, seasons=None, argv=(),
                     inputs: str = "context", target: str = "whole_rating") -> int:
    """`openrapm second-prior`: every season's grams and inputs, then run.json (the module docstring)."""
    started = time.time()
    n_folds = PRODUCT.prior.second_prior_folds
    out_dir = grams_directory(cfg, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_all = load_models(resolve(cfg, "outputs"), models)
    check_models(saved_all.get("meta", {}))
    if ((saved_all.get("meta", {}).get("settings") or {}).get("prior") or {}).get("game_prior"):
        raise SystemExit("these boosters' build read the box-score prior game by game; the second prior's swaps are "
                         "read from folds rated one prior per player-season")
    shrink = replace(TEST.shrink, folds_tag=folds_tag)
    folds, where = step4_folds(cfg, shrink)
    fold_seasons = sorted({fold.season for fold in folds})
    multipliers = step4_multipliers(fold_seasons, folds, shrink)
    chosen = [season for season in fold_seasons if seasons is None or season in set(seasons)]
    scale = None
    # every set that reads step 5's numbers needs their standardisation, the centred one too: without it the line
    # would be fitted on the eight context numbers alone under the set's name (notes/DECISIONS.md, the bug-fix
    # pass of 2026-10-22)
    if second_prior.reads_types(inputs):
        panel = trainable_panel(cfg)
        scale = second_prior.type_scale(panel[panel.side == "O"])
    print(f"second prior: {len(folds)} folds of {folds_tag}, boosters {models}; seasons {chosen[0]}-{chosen[-1]}; "
          f"step 4 at the test setting; fitted on {target}; writing {out_dir}", flush=True)
    worst_gap = 0.0
    within, designs = None, None
    if target == "rating":
        within = within_layer_multipliers(cfg, fold_seasons, folds, where, shrink).set_index("season")
        designs = DesignCache(cfg, keep=1)
    for season in chosen:
        if target == "rating":
            if rating_sums_path(out_dir, season).exists():
                continue
            saved = season_models(saved_all, season)
            season_folds = [fold for fold in folds if fold.season == season]
            worst_gap = max(worst_gap, rating_season(cfg, season, season_folds, where, saved, within, scale, inputs,
                                                     out_dir, designs))
            print(f"  {season}: priors within {worst_gap:.1e} of the folds' ({time.time() - started:.0f}s)", flush=True)
            continue
        path = grams_path(out_dir, season)
        if path.exists():
            continue
        saved = season_models(saved_all, season)
        stints = season_lineup_rows(cfg, season)
        factors = prior_shrink.season_factors(multipliers, season)
        parts, frames, used = [], [], {"offense": 0, "defense": 0}
        for fold in (fold for fold in folds if fold.season == season):
            grams, frame, gap, n = fold_grams(fold, where[fold.stem], stints, saved, factors, n_folds, scale, inputs,
                                              target)
            if not gap < PRIOR_TOLERANCE:
                raise SystemExit(f"{fold.stem}: the boosters of {models} give priors {gap:.1e} from the fold's own; "
                                 "these folds were rated with other boosters")
            worst_gap = max(worst_gap, gap)
            parts.append(grams)
            frames.append(frame)
            used = {side: used[side] + n[side] for side in used}
        summed = {side: swap_adjust.add_grams([part[side] for part in parts]) for side in swap_adjust.SIDES}
        pd.concat(frames, ignore_index=True).to_parquet(out_dir / f"inputs_{season}.parquet", index=False)
        write_grams(path, summed)
        print(f"  {season}: {len(parts)} folds, {used['offense']:,} offensive and {used['defense']:,} defensive swaps; "
              f"priors within {worst_gap:.1e} of the folds' ({time.time() - started:.0f}s)", flush=True)
    write_run(out_dir, dict(argv=list(argv), models=models, folds=folds_tag, seasons=chosen, n_folds=n_folds,
                            inputs=second_prior.INPUT_SETS[inputs], type_scale=scale, padded=True, target=target,
                            step4_rule=shrink.rule,
                            prior_check=PRIOR_TOLERANCE))
    print(f"wrote {out_dir} ({time.time() - started:.0f}s)", flush=True)
    return 0
