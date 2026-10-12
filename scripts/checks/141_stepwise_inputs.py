"""Check 141, forward stepwise selection of the box-score prior's inputs, on one half of the seasons.

The owner, 2026-10-12: rather than test each play-by-play split one at a time (the order would decide what is
adopted, and two splits could fix the same lean twice), select from all of them at once and see what sticks; with
forward stepwise selection, because it asks what a cut adds to the inputs already chosen (Boruta keeps every input
related to the label, a total and all its parts together).

Words used below
    half          1997-2011 ("early") or 2012-2026 ("late").  Everything the selection reads comes from one half: the
                  labels (step 1's RAPM, over that half's seasons only), the training rows and the rated rows, so the
                  build that tests the selected inputs can be scored on the other half, which the selection never saw
    screen season one of four rated seasons of the half (`SCREEN`); its neighbours are left out of the labels and
                  the training rows, as in the test setting
    cell          one screen season under one assignment of players to the five player folds (0 is the shipped
                  assignment, 1 and 2 regroup the players: box_prior.balanced_player_folds); 4 x 3 = 12 cells
    held-out error  in a cell, each rated player's prior comes from the booster fitted without his fold (the shipped
                  LightGBM settings, linear leaves); the error is the possession-weighted mean squared miss of his
                  label after the best straight line through the priors (`rescaled_error`), because step 3 fits the
                  prior's scale and intercept anyway
    pool          the candidate inputs: every play-by-play part (features.EVENT_RATES) plus made free throws and
                  free-throw %, the other two inputs of the stat-splits experiments' lists

Each round, every remaining candidate is added in turn to the side's current list and refitted in all 12 cells; its
change in held-out error against the current list is paired by cell.  The candidate with the largest mean drop is
added if its paired t-statistic (the mean change over its standard error, 11 degrees of freedom) is -3 or below;
otherwise the selection stops.  A part that only helps beside another enters on a later round, once the other is in.

Known blind spot: the error weights players by their possessions, so the bench barely counts (the screen of
experiments 32 and 44 had the same, notes/DECISIONS.md, trap 29); the full test that follows reads every player.

Reads the season panel (paths.panel), the design pieces (data/cache), params/booster_params_lgb1.json.  Writes
outputs/csv/stepwise_<half>.csv (every candidate in every round) and outputs/stepwise_<half>.json (the selected list
per side, in the order the inputs entered, after the shipped ones), and prints each round.

    python scripts/checks/141_stepwise_inputs.py --half=early --jobs=10      (about 30-40 minutes)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm.commands.console import one_thread, use_utf8

HALVES = {"early": (1997, 2011), "late": (2012, 2026)}
SCREEN = {"early": (2000, 2003, 2006, 2009), "late": (2015, 2018, 2021, 2024)}
ASSIGNMENTS = (0, 1, 2)
N_FOLDS = 5
Z_TO_ENTER = -3.0
SIDES = (("O", "offense"), ("D", "defense"))
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

_CELLS: dict = {}                 # in each worker: {(side, season): (training rows, rated rows)}
_PARAMS: dict = {}                # in each worker: {side: LightGBM settings}


# ------------------------------------------------------------------------------------------ what is read
def pool_inputs() -> list:
    from openrapm.features import EVENT_RATES
    return [*EVENT_RATES, "free_throws_made_per100", "free_throw_pct"]


def prepare(half: str, cfg: dict) -> dict:
    """{(side, season): (training rows, rated rows)} for every screen season, with every shipped and pool input.

    The labels are step 1's RAPM over the half's trainable seasons only, the screen season and its neighbours left
    out; the rated rows are the screen season's own, each with its label (players without one are dropped)."""
    from openrapm.box_prior import rated_rows, training_rows
    from openrapm.commands.build import accumulate_rapm, unseen_seasons
    from openrapm.data.designs import DesignCache
    from openrapm.data.panel import trainable_panel
    from openrapm.data.seasons import trainable
    from openrapm.features import SHIPPED
    from openrapm.rapm import career_labels
    from openrapm.settings import TEST
    first, last = HALVES[half]
    seasons = [season for season in trainable(cfg).seasons if first <= season <= last]
    sums = accumulate_rapm(DesignCache(cfg), seasons, TEST)
    panel = trainable_panel(cfg)
    panel = panel[(panel.possessions > 0) & panel.season.between(first, last)].reset_index(drop=True)
    cells = {}
    for season in SCREEN[half]:
        unseen = unseen_seasons(season, TEST)
        labels = career_labels(sums, unseen, TEST.labels, TEST.targets)
        for side, column in SIDES:
            names = list(dict.fromkeys([*SHIPPED[side], *pool_inputs()]))
            mine = panel[panel.side == side]
            train = training_rows(labels[side], mine[~mine.season.isin(unseen)], column, names,
                                  TEST.prior.chunk_sizes)
            rated = rated_rows(mine[mine.season == season], names)
            label = labels[side].set_index("player_id")[column]
            cells[(side, season)] = (train, rated.assign(label=rated.player_id.map(label)).dropna(subset=["label"]))
        print(f"  {season}: labels and rows ready", flush=True)
    return cells


# ---------------------------------------------------------------------------------------- one cell's error
def rescaled_error(label: np.ndarray, prior: np.ndarray, weight: np.ndarray) -> float:
    """The weighted mean squared miss of `label` after the best straight line a + b x prior (weighted least
    squares), weights the rated possessions.

    Example: priors twice as wide as the labels but in the right order miss by nothing after the line."""
    design = np.column_stack([np.ones_like(prior), prior])
    root = np.sqrt(weight)
    coefficients = np.linalg.lstsq(design * root[:, None], label * root, rcond=None)[0]
    miss = label - design @ coefficients
    return float(np.sum(weight * miss ** 2) / np.sum(weight))


def held_out_priors(train: pd.DataFrame, rated: pd.DataFrame, inputs: list, params: dict,
                    assignment: int) -> np.ndarray:
    """Each rated player's prior from the booster fitted without his player fold (NaN for a player in none)."""
    from openrapm.box_prior import balanced_player_folds, fit_lightgbm
    from openrapm.features import ROW_FEATURES
    columns = [*inputs, *ROW_FEATURES]
    fold = balanced_player_folds(train, N_FOLDS, assignment)
    matrix, label, weight = (train[columns].to_numpy(float), train.label.to_numpy(float),
                             train.row_weight.to_numpy(float))
    rated_matrix, rated_ids = rated[columns].to_numpy(float), rated.player_id.to_numpy()
    out = np.full(len(rated), np.nan)
    for left_out in range(N_FOLDS):
        keep = fold != left_out
        mine = np.isin(rated_ids, train.index[~keep].unique())
        if mine.any():
            out[mine] = fit_lightgbm(params, matrix[keep], label[keep], weight[keep]).predict(rated_matrix[mine])
    return out


def cell_error(task: tuple) -> float:
    """One (side, screen season, assignment, inputs): its held-out error.  Runs in a worker."""
    side, season, assignment, inputs = task
    train, rated = _CELLS[(side, season)]
    prior = held_out_priors(train, rated, list(inputs), _PARAMS[side], assignment)
    seen = ~np.isnan(prior)
    return rescaled_error(rated.label.to_numpy(float)[seen], prior[seen], rated.possessions.to_numpy(float)[seen])


def _start_worker(data_path: str) -> None:
    for name in THREADS:
        os.environ[name] = "1"
    global _CELLS, _PARAMS
    with open(data_path, "rb") as handle:
        _CELLS, _PARAMS = pickle.load(handle)


# --------------------------------------------------------------------------------------------- the rounds
def cells_of(half: str) -> list:
    return [(season, assignment) for season in SCREEN[half] for assignment in ASSIGNMENTS]


def errors(executor, half: str, side: str, lists: dict) -> dict:
    """{name: the held-out error in every cell} for each input list in `lists` ({name: inputs}), all at once."""
    keys = [(name, cell) for name in lists for cell in cells_of(half)]
    tasks = [(side, season, assignment, tuple(lists[name])) for name, (season, assignment) in keys]
    values = list(executor.map(cell_error, tasks, chunksize=1))
    out: dict = {}
    for (name, _cell), value in zip(keys, values):
        out.setdefault(name, []).append(value)
    return {name: np.asarray(value) for name, value in out.items()}


def paired(trial: np.ndarray, current: np.ndarray) -> dict:
    """The change against the current list, paired by cell: mean, standard error, t-statistic, cells better."""
    change = trial - current
    se = change.std(ddof=1) / np.sqrt(len(change))
    return dict(mean_change=float(change.mean()), se=float(se), z=float(change.mean() / se) if se > 0 else 0.0,
                cells_better=int((change < 0).sum()), cells=len(change))


def select_side(executor, half: str, side: str, started: float) -> tuple[list, list]:
    """Forward stepwise for one side: (the selected list, one row per candidate per round)."""
    from openrapm.features import SHIPPED, meaning
    current = list(SHIPPED[side])
    remaining = [name for name in pool_inputs() if name not in current]
    base = errors(executor, half, side, {"current": current})["current"]
    print(f"{side}: shipped list, held-out error {base.mean():.4f} over {len(base)} cells", flush=True)
    rows, round_number = [], 0
    while remaining:
        round_number += 1
        trials = errors(executor, half, side, {name: current + [name] for name in remaining})
        table = pd.DataFrame([dict(half=half, side=side, round=round_number, candidate=name, meaning=meaning(name),
                                   **paired(trials[name], base)) for name in remaining])
        best = table.sort_values("mean_change").iloc[0]
        enters = bool(best.z <= Z_TO_ENTER and best.mean_change < 0)
        rows += table.assign(entered=table.candidate.eq(best.candidate) & enters).to_dict("records")
        print(f"  round {round_number}: best {best.meaning} {best.mean_change:+.5f} (z {best.z:+.1f}, "
              f"{best.cells_better} of {best.cells} cells) -> {'ENTERS' if enters else 'stop'} "
              f"({time.time() - started:.0f}s)", flush=True)
        if not enters:
            break
        current.append(best.candidate)
        remaining.remove(best.candidate)
        base = trials[best.candidate]
    return current, rows


def run(args: argparse.Namespace) -> int:
    from openrapm.data.config import booster_params, load_config, resolve
    from openrapm.settings import TEST
    cfg = load_config()
    started, outputs = time.time(), resolve(cfg, "outputs")
    print(f"the {args.half} half {HALVES[args.half]}: labels and rows", flush=True)
    data_path = outputs / f"stepwise_{args.half}_cells.pkl"
    with open(data_path, "wb") as handle:
        pickle.dump((prepare(args.half, cfg), booster_params(cfg, TEST.prior.booster_file)), handle)
    selected, rows = {}, []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_start_worker, initargs=(str(data_path),)) as pool:
        for side, _column in SIDES:
            selected[side], side_rows = select_side(pool, args.half, side, started)
            rows += side_rows
    (outputs / "csv").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(outputs / "csv" / f"stepwise_{args.half}.csv", index=False)
    Path(outputs / f"stepwise_{args.half}.json").write_text(json.dumps(selected, indent=1), encoding="utf-8")
    data_path.unlink()
    from openrapm.features import SHIPPED
    print(f"selected, {args.half} half: " + "; ".join(f"{side} + {selected[side][len(SHIPPED[side]):]}"
                                                      for side, _ in SIDES), flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)
    return 0


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="Forward stepwise selection of the prior's inputs, one half.")
    parser.add_argument("--half", choices=sorted(HALVES), required=True)
    parser.add_argument("--jobs", type=int, default=10, help="worker processes, one thread each (default %(default)s)")
    args = parser.parse_args(argv)
    with one_thread():
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
