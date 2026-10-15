"""Check 150, the column-subsampling screen: the box-score prior's booster with each tree (or each split) shown only
part of its inputs, scored on held-out players before any build.

Why: checks 148 and 149 found that adding any column to the offensive prior, even one with nothing in it, lowered its
held-out error (a shuffled column -0.012, plain noise -0.009), and the defensive prior moved the other way.  A booster
that gets better when handed a useless column is one that leans too hard on the inputs it has, and the usual cure is to
show each tree a random part of them (LightGBM's `colsample_bytree`), or each split (`feature_fraction_bynode`).  The
shipped settings show every tree every input (box_prior.LIGHTGBM_FIXED).  This is tuning, so trap 30 applies: a win here
is against today's settings on today's inputs and is read on the year-over-year test before anything is adopted.

Words used below
    screen season, cell, held-out error, yardstick    as in check 142, on its ten screen seasons spread over the eras
    arm             one setting of the subsampling, the same on both sides (the reason is the same; the owner's rule):
                    `today` (every input in every tree), `tree_0.8`, `tree_0.6`, `tree_0.4` (that share of the inputs
                    drawn for each tree) and `split_0.6` (60% drawn for each split).  The draws are seeded (random_state
                    0), so a cell gives the same answer every run

The inputs are today's (features.FEATURE_SETS["no_career_counts"]) for every arm; only the booster's draw of them
changes.

The gate is check 142's, registered before the first run (notes/DECISIONS.md, "Check 150"): an arm passes when (a) on
the career yardstick it beats today at t -2 or below on at least one side, (b) neither side is worse at t +2 or beyond
on either yardstick, (c) its mean change on the neighbours yardstick is not above 0 on a side that passed (a), (d) no
era is worse in every one of its screen seasons.  Of the passing arms the one with the largest summed career-yardstick
gain over the two sides is the best; the chosen one is the mildest (the largest share drawn) whose summed gain is within
one standard error of the best's.  Nothing passing, nothing is built.

Writes outputs/csv/column_subsampling_screen_cells.csv, column_subsampling_screen.csv and
outputs/column_subsampling_screen_decision.json.

    python scripts/checks/150_column_subsampling_screen.py --jobs=10      (about 10 minutes)
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


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCREEN = _load("era_screen", "142_era_screen.py")
SEASONS = SCREEN.SCREEN
# {arm: what it changes in box_prior.LIGHTGBM_FIXED}, mildest first after today
SETTINGS = {"today": {}, "tree_0.8": {"colsample_bytree": 0.8}, "tree_0.6": {"colsample_bytree": 0.6},
            "tree_0.4": {"colsample_bytree": 0.4}, "split_0.6": {"feature_fraction_bynode": 0.6}}
SHARE = {"today": 1.0, "tree_0.8": 0.8, "tree_0.6": 0.6, "tree_0.4": 0.4, "split_0.6": 0.6}
GATE_ARMS = tuple(arm for arm in SETTINGS if arm != "today")

_CELLS: dict = {}
_PARAMS: dict = {}
_FIXED: dict = {}


def arms() -> dict:
    """{name: ({side: input list}, None)}: every arm reads today's inputs."""
    from openrapm.features import FEATURE_SETS
    return {arm: (FEATURE_SETS["no_career_counts"], None) for arm in SETTINGS}


def cell_task(task: tuple) -> dict:
    """One (side, screen season, assignment, arm): its errors, the arm's subsampling set for the fits.  In a worker."""
    from openrapm import box_prior
    side, season, assignment, arm = task
    box_prior.LIGHTGBM_FIXED = {**_FIXED, **SETTINGS[arm]}
    train, rated = _CELLS[(side, season)]
    inputs = list(arms()[arm][0][side])
    prior, _ = SCREEN.held_out_priors(train, rated, inputs, _PARAMS[side], assignment, None)
    return dict(task=task, errors=SCREEN.errors_of(rated, prior), spread=float(np.std(prior)))


def _start_worker(data_path: str) -> None:
    for name in SCREEN.THREADS:
        os.environ[name] = "1"
    from openrapm import box_prior
    global _CELLS, _PARAMS, _FIXED
    with open(data_path, "rb") as handle:
        _CELLS, _PARAMS = pickle.load(handle)
    _FIXED = dict(box_prior.LIGHTGBM_FIXED)


def decide(per_season: pd.DataFrame) -> dict:
    """The gate for every arm; the best and the chosen passing arm (module docstring), None when none passes."""
    verdicts = {arm: SCREEN.passes(per_season, arm) for arm in GATE_ARMS}
    career = {(arm, side): SCREEN.paired(per_season, arm, "today", side, "career", "all")
              for arm in GATE_ARMS for side, _ in SCREEN.SIDES}
    summed = {arm: sum(career[(arm, side)]["mean_change"] for side, _ in SCREEN.SIDES) for arm in GATE_ARMS}
    spread = {arm: float(np.hypot(*[career[(arm, side)]["se"] for side, _ in SCREEN.SIDES])) for arm in GATE_ARMS}
    passing = [arm for arm in GATE_ARMS if verdicts[arm][0]]
    out = dict(verdicts={arm: dict(passes=ok, sides=won, why=why) for arm, (ok, won, why) in verdicts.items()},
               summed_change=summed, best=None, chosen=None)
    if passing:
        best = min(passing, key=lambda arm: summed[arm])
        close = [arm for arm in passing if summed[arm] <= summed[best] + spread[best]]
        out.update(best=best, chosen=max(close, key=lambda arm: (SHARE[arm], -summed[arm])))
    return out


def run(args: argparse.Namespace) -> int:
    from openrapm.data.config import booster_params, load_config, resolve
    from openrapm.settings import TEST
    cfg = load_config()
    started, outputs = time.time(), resolve(cfg, "outputs")
    print(f"labels and rows for the screen seasons {', '.join(map(str, SEASONS))}", flush=True)
    cells = SCREEN.prepare(cfg, arms())
    data_path = outputs / "column_subsampling_screen_cells.pkl"
    with open(data_path, "wb") as handle:
        pickle.dump((cells, booster_params(cfg, TEST.prior.booster_file)), handle)
    tasks = [(side, season, assignment, arm) for arm in SETTINGS for side, _ in SCREEN.SIDES for season in SEASONS
             for assignment in SCREEN.ASSIGNMENTS]
    print(f"{len(tasks)} cells ({time.time() - started:.0f}s)", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_start_worker, initargs=(str(data_path),)) as pool:
        for done, result in enumerate(pool.map(cell_task, tasks, chunksize=1), start=1):
            side, season, assignment, arm = result["task"]
            rows += [dict(side=side, season=season, assignment=assignment, arm=arm, yardstick=yardstick, group=group,
                          error=value, prior_sd=result["spread"])
                     for (yardstick, group), value in result["errors"].items()]
            if done % 50 == 0:
                print(f"  {done} of {len(tasks)} ({time.time() - started:.0f}s)", flush=True)
    data_path.unlink()
    errors = pd.DataFrame(rows)
    (outputs / "csv").mkdir(exist_ok=True)
    errors.to_csv(outputs / "csv" / "column_subsampling_screen_cells.csv", index=False)
    per_season = errors.groupby(["side", "arm", "yardstick", "group", "season"], as_index=False).error.mean()
    summary = pd.DataFrame([SCREEN.paired(per_season, arm, "today", side, yardstick, group)
                            for arm in GATE_ARMS for side, _ in SCREEN.SIDES
                            for yardstick, _ in SCREEN.YARDSTICKS for group in ("all", "bench")])
    summary.to_csv(outputs / "csv" / "column_subsampling_screen.csv", index=False)
    decision = decide(per_season)
    (outputs / "column_subsampling_screen_decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    pd.set_option("display.width", 220)
    shown = ["arm", "side", "yardstick", "group", "mean_change", "se", "t", "seasons_better", "seasons"]
    print(summary[shown].round(5).to_string(index=False), flush=True)
    print("\nthe spread of the held-out priors, each arm and side (points per 100):", flush=True)
    print(errors.groupby(["arm", "side"]).prior_sd.mean().unstack().round(3).to_string(), flush=True)
    print("\n=== the gate ===", flush=True)
    for arm, verdict in decision["verdicts"].items():
        print(f"  {arm}: {'passes' if verdict['passes'] else 'fails'} (sides {verdict['sides'] or 'none'}) "
              f"{verdict['why']}", flush=True)
    print(f"  best: {decision['best']}  chosen: {decision['chosen']}", flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)
    return 0


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="The column-subsampling screen: the booster shown part of its inputs.")
    parser.add_argument("--jobs", type=int, default=10, help="worker processes, one thread each (default %(default)s)")
    args = parser.parse_args(argv)
    with one_thread():
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
