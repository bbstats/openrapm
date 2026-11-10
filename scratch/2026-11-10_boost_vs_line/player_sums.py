"""Read-only: each fold's share of the within-season test's normal equations with one column per PLAYER.

The line fitted on the rating (src/openrapm/rating_line.py) keeps its sums over the line's inputs, 57 an end, so only
a straight line in those inputs can be scored from them.  Here every player is his own column on each end: a prior
that gives player i the value p_i on an end, however it was made, a booster's output included, is then scored exactly
as the line's is, since a rating is a straight line in its prior.

For a fold with n players the columns are [offence's fitted part, defence's fitted part, n offence, n defence], and

    error(w, p) = (yy - 2 [w p]' c + [w p]' Q [w p]) / sw

is the fold's share of the test's error at the fitted parts' weights w and the prior p.  The line's own sums are these
with p = P beta, P the fold's `rating_line.prior_columns`: the script checks that, season by season, against the sums
the shipped line was cut from (outputs/second_prior/zones139_line), before anything is read from its files.

Everything is the run that made those sums: padbins127's folds, the test setting's ridge, the within layer of the
shipped settings, and each fold's inputs as that run saved them (outputs/second_prior/zones139/inputs_<season>.parquet),
so no booster is loaded.

Writes outputs/second_prior/zones139_players/<fold>.npz: Q, c, yy, sw, the players' ids, P for the shipped 57 inputs,
each end's raw inputs with their ids, and each end's (player, team, possessions) of the fit games.  A fold already
written is skipped.

    .venv/Scripts/python scratch/2026-11-10_boost_vs_line/player_sums.py [SEASON ...]
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm import prior_shrink, rating_line, second_prior
from openrapm.commands.build import season_lineup_rows, step4_folds, within_layer_multipliers
from openrapm.commands.second_prior import RatingSeason, held_back_stints
from openrapm.data.config import load_config, resolve
from openrapm.data.designs import DesignCache
from openrapm.data.panel import fit_designs
from openrapm.data.second_prior_files import load_grams
from openrapm.data.within_folds import fold_games
from openrapm.rapm import design_blocks
from openrapm.settings import TEST

FOLDS = "padbins127_early,padbins127_within"
SCREEN, LINE, OUT = "zones139", "zones139_line", "zones139_players"
SIDES = ("O", "D")
F = rating_line.FITTED_PARTS


def fold_inputs(saved: pd.DataFrame, fold, names: list) -> dict:
    """{"O", "D"}: the fold's inputs as the screen's run saved them, player_id -> the shipped line's inputs."""
    rows = saved[(saved.repeat == fold.repeat) & (saved.fold == fold.fold)]
    return {side: rows[rows.side == side].set_index("player_id")[names] for side in SIDES}


def fold_sums(fold, directory: Path, stints: pd.DataFrame, season: RatingSeason, inputs: dict) -> dict:
    """One fold's arrays (the module docstring)."""
    fit_ids, _ = fold_games(directory, fold)
    fit_stints = stints[stints.game_id.astype(str).isin(fit_ids).to_numpy()]
    designs = fit_designs(fold.season, season.cfg, season.whole, fit_ids)
    design = designs[TEST.targets.offence]
    players, context = design_blocks(design)
    ids = design.spec.players["player_id"].to_numpy(dtype=np.int64)
    team_poss = {side: second_prior.team_possessions(fit_stints, side) for side in SIDES}
    weights, y_off = np.asarray(design.w, dtype=float), np.asarray(design.y, dtype=float)
    y_def = np.asarray(designs[TEST.targets.defence].y, dtype=float)
    one_each = np.eye(2 * len(ids))
    offence, defence, explained = rating_line.ridge_parts(players, context, weights, y_off, y_def, one_each,
                                                          season.penalty)
    held = held_back_stints(season, directory, fold)
    sums = rating_line.held_back_sums(held, (offence, defence), one_each, explained, season.within)
    # the line's own columns through the same ridge, for the check: P's explained part is the players' times P
    priors = rating_line.prior_columns(inputs, ids, team_poss)
    return dict(Q=sums.Q, c=sums.c, yy=sums.yy, sw=sums.sw, ids=ids, P=priors,
                **{f"ids_{side}": inputs[side].index.to_numpy(dtype=np.int64) for side in SIDES},
                **{f"X_{side}": inputs[side].to_numpy(dtype=float) for side in SIDES},
                **{f"team_poss_{side}": team_poss[side][["player_id", "team", "poss"]].to_numpy(dtype=float)
                   for side in SIDES})


def projected(arrays: dict) -> rating_line.RatingSums:
    """The fold's sums over the line's own columns, [fitted parts | P]."""
    n = 2 * len(arrays["ids"])
    across = np.zeros((F + n, F + arrays["P"].shape[1]))
    across[:F, :F] = np.eye(F)
    across[F:, F:] = arrays["P"]
    return rating_line.RatingSums(Q=across.T @ arrays["Q"] @ across, c=across.T @ arrays["c"], yy=float(arrays["yy"]),
                                  sw=float(arrays["sw"]))


def main() -> None:
    started = time.time()
    cfg = load_config()
    asked = {int(season) for season in sys.argv[1:]}
    out = resolve(cfg, "outputs") / "second_prior" / OUT
    out.mkdir(parents=True, exist_ok=True)
    shrink = replace(TEST.shrink, folds_tag=FOLDS)
    folds, where = step4_folds(cfg, shrink)
    fold_seasons = sorted({fold.season for fold in folds})
    within = within_layer_multipliers(cfg, fold_seasons, folds, where, shrink).set_index("season")
    names = second_prior.INPUT_SETS["context_types_team_zones_centred"]
    shipped = load_grams(cfg, LINE)
    designs = DesignCache(cfg, keep=1)
    print(f"{len(folds)} folds of {FOLDS}; {len(names)} inputs an end; writing {out}", flush=True)
    for season in fold_seasons:
        if asked and season not in asked:
            continue
        season_folds = [fold for fold in folds if fold.season == season]
        missing = [fold for fold in season_folds if not (out / f"{fold.stem}.npz").exists()]
        if missing:
            stints = season_lineup_rows(cfg, season)
            whole = {target: designs.season(season, target)
                     for target in ("pts", TEST.targets.offence, TEST.targets.defence)}
            row = within.loc[season]
            shared = RatingSeason(cfg, whole, design_blocks(whole["pts"])[0],
                                  tuple(float(row[column]) for column in prior_shrink.WITHIN_COLUMNS),
                                  TEST.ridge.penalty)
            saved = pd.read_parquet(resolve(cfg, "outputs") / "second_prior" / SCREEN / f"inputs_{season}.parquet")
            for fold in missing:
                arrays = fold_sums(fold, where[fold.stem], stints, shared, fold_inputs(saved, fold, names))
                written = out / f"{fold.stem}.npz.part"
                with open(written, "wb") as handle:
                    np.savez(handle, **arrays)
                written.replace(out / f"{fold.stem}.npz")
        # the check: these folds, projected on the line's columns, are the season's shipped sums
        total = None
        for fold in season_folds:
            with np.load(out / f"{fold.stem}.npz") as arrays:
                part = projected({key: arrays[key] for key in ("Q", "c", "yy", "sw", "ids", "P")})
            total = part if total is None else total + part
        scale = np.sqrt(np.outer(np.diag(shipped[season].Q), np.diag(shipped[season].Q)))
        scale[scale == 0] = 1.0
        gap_q = float(np.abs((total.Q - shipped[season].Q) / scale).max())
        gap_c = float(np.abs(total.c - shipped[season].c).max() / np.abs(shipped[season].c).max())
        gap = max(gap_q, gap_c, abs(total.yy - shipped[season].yy) / shipped[season].yy,
                  abs(total.sw - shipped[season].sw) / shipped[season].sw)
        print(f"  {season}: {len(season_folds)} folds, {len(missing)} written; against the shipped line's sums the "
              f"largest relative gap {gap:.1e} ({time.time() - started:.0f}s)", flush=True)
        designs.clear()


if __name__ == "__main__":
    main()
