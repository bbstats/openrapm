"""`openrapm second-prior`: the swaps the second prior's line is fitted on (second_prior.py), from the held-back games
of within-season folds.

    openrapm second-prior --models NAME --folds TAG[,TAG] --tag TAG [--seasons 1997-2026] [--inputs context_and_types]

For every fold of every season in the folds named:

    1. Every player of the fold's fit games is answered for by every booster the fold was rated with (`--models`,
       outputs/prior_models_<NAME>.pkl).  His box-score prior from those answers must equal the one the fold stored
       (prior_raw_off, prior_raw_def) to 1e-9, or the run stops: the folds were rated with other boosters.
    2. His eight context numbers, from the fit games' stint rows, every prior in them from the boosters that never saw
       him (second_prior.context_numbers).
    With --inputs context_and_types (experiment 82) every player also carries his other end's box-score prior and
    step 5's 19 numbers from the fold's players file, the fit games', standardised with the trainable panel's means
    and standard deviations, which run.json keeps for the build.
    3. The swaps of the held-back games, scored against the fold's step-4 ratings at the test setting's multipliers,
       give the fold's share of the line's normal equations (second_prior.held_back_grams).  The fit games and the
       held-back games never share a game, so a shot counted in a box score never meets the lineup result it scored.

Writes outputs/second_prior/<TAG>/ (data/second_prior_files.py): one grams_<season>.npz and inputs_<season>.parquet per
season, and run.json.  A season already written is skipped, so a stopped run resumes where it stopped.

Example: `openrapm second-prior --models bags10 --folds bags10_early,bags10_within --tag ctx81` reads the incumbent's
360 folds, 1997-2026; `openrapm rate --setting test --second-prior ctx81` then rates every season on the line.
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import prior_shrink, second_prior, swap_adjust, swap_test
from ..data.config import resolve
from ..data.legacy_names import translate_frame
from ..data.panel import trainable_panel
from ..data.second_prior_files import grams_directory, grams_path, write_grams, write_run
from ..data.within_folds import fold_games
from ..settings import PRODUCT, TEST
from .build import season_lineup_rows, step4_folds, step4_multipliers
from .within_season import check_models, load_models, prior_rows, saved_booster, season_models

SIDES = ("O", "D")
PRIOR_TOLERANCE = 1e-9          # the boosters' answers against the prior the fold stored


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
               scale: dict | None = None, input_set: str | None = None):
    """One fold: ({side: PairGrams}, its players' inputs in one frame, the prior check's largest gap, swaps used)."""
    fit_ids, held_ids = fold_games(directory, fold)
    players = fold_players(directory, fold.stem)
    answers, gap = fold_answers(players, saved)
    game = stints.game_id.astype(str)
    fit_stints = stints[game.isin(fit_ids).to_numpy()]
    context = second_prior.context_numbers(fit_stints, answers["O"], answers["D"])
    box = stored_box(players)
    types = None if scale is None else second_prior.standardised_types(players[players.side == "O"], scale)
    inputs = second_prior.side_inputs(box, context, types, input_set)
    held = swap_test.prepare(stints[game.isin(held_ids).to_numpy()].reset_index(drop=True), fold.season)
    ratings = swap_test.ratings_in_raw_sign(prior_shrink.shrink_fold(fold.rows, fold.season, factors))
    grams = second_prior.held_back_grams(held, ratings, inputs, n_folds)
    frame = pd.concat([inputs[side].reset_index().assign(side=side) for side in SIDES], ignore_index=True)
    frame = frame.assign(season=fold.season, repeat=fold.repeat, fold=fold.fold)
    return grams, frame, gap, {side: grams[side].n for side in grams}


def run_second_prior(cfg: dict, models: str, folds_tag: str, tag: str, seasons=None, argv=(),
                     inputs: str = "context") -> int:
    """`openrapm second-prior`: every season's grams and inputs, then run.json (the module docstring)."""
    started = time.time()
    n_folds = PRODUCT.prior.second_prior_folds
    out_dir = grams_directory(cfg, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_all = load_models(resolve(cfg, "outputs"), models)
    check_models(saved_all.get("meta", {}))
    shrink = replace(TEST.shrink, folds_tag=folds_tag)
    folds, where = step4_folds(cfg, shrink)
    fold_seasons = sorted({fold.season for fold in folds})
    multipliers = step4_multipliers(fold_seasons, folds, shrink)
    chosen = [season for season in fold_seasons if seasons is None or season in set(seasons)]
    scale = None
    if inputs == "context_and_types":
        panel = trainable_panel(cfg)
        scale = second_prior.type_scale(panel[panel.side == "O"])
    print(f"second prior: {len(folds)} folds of {folds_tag}, boosters {models}; seasons {chosen[0]}-{chosen[-1]}; "
          f"step 4 at the test setting; writing {out_dir}", flush=True)
    worst_gap = 0.0
    for season in chosen:
        path = grams_path(out_dir, season)
        if path.exists():
            continue
        saved = season_models(saved_all, season)
        stints = season_lineup_rows(cfg, season)
        factors = prior_shrink.season_factors(multipliers, season)
        parts, frames, used = [], [], {"offense": 0, "defense": 0}
        for fold in (fold for fold in folds if fold.season == season):
            grams, frame, gap, n = fold_grams(fold, where[fold.stem], stints, saved, factors, n_folds, scale, inputs)
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
                            inputs=second_prior.INPUT_SETS[inputs], type_scale=scale, padded=True,
                            step4_rule=shrink.rule,
                            prior_check=PRIOR_TOLERANCE))
    print(f"wrote {out_dir} ({time.time() - started:.0f}s)", flush=True)
    return 0
