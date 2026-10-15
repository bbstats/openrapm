"""Check 143, the warm-start screen: the pooled prior, then a few more trees fitted on rows weighted toward the rated
season, scored on held-out players before any build (notes/DECISIONS.md, "The warm-start screen").

The defensive prices drift between eras, and weighting the training rows toward the rated season lost at every
half-life (check 142): the rows it gives up cost more than the drift it follows.  A warm start keeps every row.

Words used below
    base         the booster as shipped: fitted on every training row (box_prior.fit_lightgbm)
    extra trees  a second booster with the same settings but `n_estimators` = the extra trees, fitted on the same rows
                 weighted by their era kernel (box_prior.era_kernel: a row counts half as much for every `half-life`
                 seasons between its seasons and the rated one, beyond the nearest usable season), starting from the
                 base's own predictions on those rows (LightGBM `init_score`).  Its trees therefore fit what the base
                 missed where the kernel puts the weight
    warm prior   base prediction + extra trees' prediction
    arm          one (half-life, extra trees) pair; `today` is the base alone
    cell, held-out error, yardstick   as in check 142 (fold assignments 0, 1 and 2 of the shipped player folds; each
                 rated player's prior from the models fitted without his fold; the possessions-weighted squared miss of
                 the label after the best straight line through the priors; the career label and the neighbouring
                 seasons' RAPM)

Everything else is check 142's: the ten screen seasons at the test setting, the prepared rows (`prepare`), the paired
statistics over seasons (`paired`) and the gate (`passes`).  The gate: an arm passes when it beats today on the career
label at t -2 or below on at least one side, is worse at t +2 or beyond on neither side on either yardstick, is not
worse on the neighbours yardstick on a side that passed, and no era loses in every one of its screen seasons.  Of the
passing arms the one with the largest defensive gain on the career label is the best; the chosen one is the arm with
the fewest extra trees whose defensive change is within one standard error of the best's.

Writes outputs/csv/warm_screen_cells.csv (every cell's error), warm_screen.csv (each arm against today, paired by
season) and outputs/warm_screen_decision.json.

    python scripts/checks/143_warm_start_screen.py --jobs=10      (about 10 minutes)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm.commands.console import one_thread, use_utf8


def _check_142():
    """Check 142 as a module: its prepared rows, error, paired statistics and gate are used here unchanged."""
    spec = importlib.util.spec_from_file_location("era_screen", Path(__file__).with_name("142_era_screen.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCREEN = _check_142()
HALF_LIVES = (4, 8)
EXTRA_TREES = (25, 75)
GATE_ARMS = [f"warm_h{half_life}_e{trees}" for half_life in HALF_LIVES for trees in EXTRA_TREES]

_CELLS: dict = {}                   # in each worker: {(side, season): (training rows, rated rows)}
_PARAMS: dict = {}                  # in each worker: {side: LightGBM settings}


# ------------------------------------------------------------------------------------------------ the arms
def input_lists() -> dict:
    """{side: the input list} every arm reads: the shipped prior's (`PRODUCT.prior.feature_set`)."""
    from openrapm.features import FEATURE_SETS
    from openrapm.settings import PRODUCT
    return FEATURE_SETS[PRODUCT.prior.feature_set]


def warm_arms() -> dict:
    """{arm: (the training rows' kernel column, the extra trees)}; `today` is left out."""
    return {f"warm_h{half_life}_e{trees}": (f"kernel_{half_life}", trees)
            for half_life in HALF_LIVES for trees in EXTRA_TREES}


# ---------------------------------------------------------------------------------------------- the models
def fit_extra(params: dict, matrix: np.ndarray, label: np.ndarray, weight: np.ndarray, kernel: np.ndarray,
              start: np.ndarray, trees: int):
    """The extra trees: the shipped settings with `trees` estimators, weights rescaled to mean 1 and times the era
    kernel (as box_prior.fit_lightgbm does), starting from `start`, the base's predictions on these rows."""
    import lightgbm

    from openrapm.box_prior import LIGHTGBM_FIXED
    weights = np.asarray(weight, dtype=float)
    weights = weights / weights.mean() * np.asarray(kernel, dtype=float)
    model = lightgbm.LGBMRegressor(**{**params, **LIGHTGBM_FIXED, "n_estimators": int(trees)})
    return model.fit(matrix, label, sample_weight=weights, init_score=start)


def cell_priors(train: pd.DataFrame, rated: pd.DataFrame, inputs: list, params: dict, assignment: int) -> dict:
    """{arm: each rated player's prior from the models fitted without his player fold} for `today` and every warm arm.

    A player in no fold takes the model fitted on every row, as in the build.  The base is fitted once per fold and
    shared by the arms; each arm adds its own extra trees on the same rows."""
    from openrapm.box_prior import balanced_player_folds, fit_lightgbm
    from openrapm.features import ROW_FEATURES
    columns = [*inputs, *ROW_FEATURES]
    fold = balanced_player_folds(train, SCREEN.N_FOLDS, assignment)
    matrix, label, weight = (train[columns].to_numpy(float), train.label.to_numpy(float),
                             train.row_weight.to_numpy(float))
    kernels = {arm: train[column].to_numpy(float) for arm, (column, _) in warm_arms().items()}
    rated_matrix, rated_ids = rated[columns].to_numpy(float), rated.player_id.to_numpy()

    def predictions(rows) -> dict:
        base = fit_lightgbm(params, matrix[rows], label[rows], weight[rows])
        start = base.predict(matrix[rows])
        out = {"today": lambda x: base.predict(x)}
        for arm, (_, trees) in warm_arms().items():
            extra = fit_extra(params, matrix[rows], label[rows], weight[rows], kernels[arm][rows], start, trees)
            out[arm] = (lambda x, extra=extra: base.predict(x) + extra.predict(x))
        return out

    everyone = predictions(np.ones(len(train), dtype=bool))
    out = {arm: predict(rated_matrix) for arm, predict in everyone.items()}
    for left_out in range(SCREEN.N_FOLDS):
        keep = fold != left_out
        mine = np.isin(rated_ids, train.index[~keep].unique())
        if mine.any():
            for arm, predict in predictions(keep).items():
                out[arm][mine] = predict(rated_matrix[mine])
    return out


def cell_task(task: tuple) -> dict:
    """One (side, screen season, assignment): every arm's errors and priors.  Runs in a worker."""
    side, season, assignment = task
    train, rated = _CELLS[(side, season)]
    priors = cell_priors(train, rated, list(input_lists()[side]), _PARAMS[side], assignment)
    return dict(task=task, errors={arm: SCREEN.errors_of(rated, prior) for arm, prior in priors.items()})


def _start_worker(data_path: str) -> None:
    for name in SCREEN.THREADS:
        os.environ[name] = "1"
    global _CELLS, _PARAMS
    with open(data_path, "rb") as handle:
        _CELLS, _PARAMS = pickle.load(handle)


# ---------------------------------------------------------------------------------------------- the reading
def decide(per_season: pd.DataFrame) -> dict:
    """The gate for every warm arm; the best and the chosen passing arm (module docstring), None when none passes."""
    verdicts = {arm: SCREEN.passes(per_season, arm) for arm in GATE_ARMS}
    career = {(arm, side): SCREEN.paired(per_season, arm, "today", side, "career", "all")
              for arm in GATE_ARMS for side, _ in SCREEN.SIDES}
    passing = [arm for arm, (ok, _, _) in verdicts.items() if ok]
    out = dict(verdicts={arm: dict(passes=ok, sides=won, why=why) for arm, (ok, won, why) in verdicts.items()},
               best=None, chosen=None, half_life=None, extra_trees=None)
    if passing:
        side = "D" if any("D" in verdicts[arm][1] for arm in passing) else "O"
        best = min(passing, key=lambda arm: career[(arm, side)]["mean_change"])
        bar = career[(best, side)]["mean_change"] + career[(best, side)]["se"]
        close = [arm for arm in passing if career[(arm, side)]["mean_change"] <= bar]
        chosen = min(close, key=lambda arm: (warm_arms()[arm][1], -int(warm_arms()[arm][0].removeprefix("kernel_"))))
        out.update(best=best, chosen=chosen, dial_side=side,
                   half_life=int(warm_arms()[chosen][0].removeprefix("kernel_")), extra_trees=warm_arms()[chosen][1])
    return out


def report(summary: pd.DataFrame, decision: dict) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    shown = ["arm", "side", "yardstick", "mean_change", "se", "t", "seasons_better", "seasons",
             *[f"{era} change" for era in SCREEN.ERAS], *[f"{era} worse" for era in SCREEN.ERAS]]
    for group, title in (("all", "every rated player"), ("bench", "players under 1,000 rated possessions")):
        print(f"\n=== each arm minus today, {title}: held-out error, below 0 is better ===", flush=True)
        print(summary[summary.group == group][shown].round(5).to_string(index=False), flush=True)
    print("\n=== the gate ===", flush=True)
    for arm, verdict in decision["verdicts"].items():
        print(f"  {arm}: {'passes' if verdict['passes'] else 'fails'} {verdict['why']}", flush=True)
    print(f"  best: {decision['best']}  chosen: {decision['chosen']}", flush=True)


def run(args: argparse.Namespace) -> int:
    from openrapm.data.config import booster_params, load_config, resolve
    from openrapm.settings import TEST
    cfg = load_config()
    started, outputs = time.time(), resolve(cfg, "outputs")
    print("labels and rows for the screen seasons", flush=True)
    cells = SCREEN.prepare(cfg, {"today": (input_lists(), None)})
    data_path = outputs / "warm_screen_cells.pkl"
    with open(data_path, "wb") as handle:
        pickle.dump((cells, booster_params(cfg, TEST.prior.booster_file)), handle)
    tasks = [(side, season, assignment) for side, _ in SCREEN.SIDES for season in SCREEN.SCREEN
             for assignment in SCREEN.ASSIGNMENTS]
    print(f"{len(tasks)} cells ({time.time() - started:.0f}s)", flush=True)
    error_rows = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_start_worker, initargs=(str(data_path),)) as pool:
        for done, result in enumerate(pool.map(cell_task, tasks, chunksize=1), start=1):
            side, season, assignment = result["task"]
            for arm, errors in result["errors"].items():
                error_rows += [dict(side=side, season=season, assignment=assignment, arm=arm, yardstick=yardstick,
                                    group=group, error=value) for (yardstick, group), value in errors.items()]
            if done % 12 == 0:
                print(f"  {done} of {len(tasks)} ({time.time() - started:.0f}s)", flush=True)
    data_path.unlink()
    (outputs / "csv").mkdir(exist_ok=True)
    errors = pd.DataFrame(error_rows)
    errors.to_csv(outputs / "csv" / "warm_screen_cells.csv", index=False)
    per_season = errors.groupby(["side", "arm", "yardstick", "group", "season"], as_index=False).error.mean()
    summary = pd.DataFrame([SCREEN.paired(per_season, arm, "today", side, yardstick, group)
                            for arm in GATE_ARMS for side, _ in SCREEN.SIDES
                            for yardstick, _ in SCREEN.YARDSTICKS for group in SCREEN.GROUPS])
    summary.to_csv(outputs / "csv" / "warm_screen.csv", index=False)
    decision = decide(per_season)
    (outputs / "warm_screen_decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    report(summary, decision)
    print(f"done in {time.time() - started:.0f}s", flush=True)
    return 0


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="The warm-start screen: the pooled prior plus a few local trees.")
    parser.add_argument("--jobs", type=int, default=10, help="worker processes, one thread each (default %(default)s)")
    args = parser.parse_args(argv)
    with one_thread():
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
