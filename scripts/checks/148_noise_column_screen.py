"""Check 148, the noise-column screen: what did check 146's placebo gain from?  A diagnostic, no gate, no build.

Check 146's placebo (the dashboard columns shuffled among the players of a season and side, the missing values kept
where they were) gained on the offensive prior (career yardstick -0.015, t -2.6) while carrying no information.  Check
147 put an era flag in by name and found nothing, so the placebo's gain is not the era.  This separates two other
explanations on the same nine screen seasons (2018-2026) and the same machinery:

    noise_missing   the placebo again: the 30 shuffled dashboard columns, missing before 2014 and 2016 (a reproduction
                    check against check 146: the same code and seeds should give the same errors)
    noise_dense     30 columns of independent standard normal noise with nothing missing, split 15 and 15 between the
                    sides as the dashboard arms' lists are

If only `noise_missing` gains, the booster is using the missing pattern some way an explicit era flag does not (the
linear leaves, `linear_tree`, treat a row with a missing input differently); if `noise_dense` gains too, added columns
of any kind move the offensive prior, and today is one draw among equals.

Writes outputs/csv/noise_column_screen_cells.csv and noise_column_screen.csv.

    python scripts/checks/148_noise_column_screen.py --jobs=10      (about 20 minutes)
"""
from __future__ import annotations

import argparse
import importlib.util
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
DASHBOARD = _load("dashboard_screen", "146_dashboard_screen.py")
SEASONS = DASHBOARD.SCREEN_SEASONS
N_DENSE = 15                       # noise columns per side, as the combined dashboard arm adds

_CELLS: dict = {}
_PARAMS: dict = {}
_ARMS: dict = {}


def dense_names() -> list:
    return [f"noise_{i}" for i in range(N_DENSE)]


def arms() -> dict:
    """{name: ({side: input list}, None)}: today, the shuffled dashboard columns, and plain noise columns."""
    from openrapm.features import FEATURE_SETS
    today = FEATURE_SETS["no_career_counts"]
    out = {"today": (today, None), "noise_missing": DASHBOARD.arms()["placebo"]}
    out["noise_dense"] = ({side: today[side] + dense_names() for side in ("O", "D")}, None)
    return out


def prepare(cfg: dict) -> dict:
    """check 142's `prepare` on the screen seasons, both kinds of noise columns in the panel in memory only."""
    from openrapm import features
    from openrapm.data import panel as panel_module
    real = panel_module.trainable_panel
    shuffled = [f"{DASHBOARD.PLACEBO_PREFIX}{name}" for name in features.DASHBOARD_RATES]

    def with_noise(cfg):
        panel = DASHBOARD.shuffle_within(real(cfg), list(features.DASHBOARD_RATES))
        rng = np.random.default_rng(0)
        return panel.assign(**{name: rng.normal(size=len(panel)) for name in dense_names()})
    panel_module.trainable_panel = with_noise
    added = [*shuffled, *dense_names()]
    features.PANEL_INPUTS.extend(added)
    features.MISSING_OK.extend(shuffled)
    SCREEN.SCREEN = SEASONS
    try:
        return SCREEN.prepare(cfg, arms())
    finally:
        panel_module.trainable_panel = real
        del features.PANEL_INPUTS[-len(added):]
        del features.MISSING_OK[-len(shuffled):]


def cell_task(task: tuple) -> dict:
    side, season, assignment, arm = task
    train, rated = _CELLS[(side, season)]
    prior, _ = SCREEN.held_out_priors(train, rated, list(_ARMS[arm][0][side]), _PARAMS[side], assignment, None)
    return dict(task=task, errors=SCREEN.errors_of(rated, prior))


def _start_worker(data_path: str) -> None:
    for name in SCREEN.THREADS:
        os.environ[name] = "1"
    global _CELLS, _PARAMS, _ARMS
    with open(data_path, "rb") as handle:
        _CELLS, _PARAMS = pickle.load(handle)
    _ARMS = arms()


def run(args: argparse.Namespace) -> int:
    from openrapm.data.config import booster_params, load_config, resolve
    from openrapm.settings import TEST
    cfg = load_config()
    started, outputs = time.time(), resolve(cfg, "outputs")
    print(f"labels and rows for the screen seasons {SEASONS[0]}-{SEASONS[-1]}", flush=True)
    cells = prepare(cfg)
    data_path = outputs / "noise_column_screen_cells.pkl"
    with open(data_path, "wb") as handle:
        pickle.dump((cells, booster_params(cfg, TEST.prior.booster_file)), handle)
    tasks = [(side, season, assignment, arm) for arm in arms() for side, _ in SCREEN.SIDES for season in SEASONS
             for assignment in SCREEN.ASSIGNMENTS]
    print(f"{len(tasks)} cells ({time.time() - started:.0f}s)", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_start_worker, initargs=(str(data_path),)) as pool:
        for done, result in enumerate(pool.map(cell_task, tasks, chunksize=1), start=1):
            side, season, assignment, arm = result["task"]
            rows += [dict(side=side, season=season, assignment=assignment, arm=arm, yardstick=yardstick, group=group,
                          error=value) for (yardstick, group), value in result["errors"].items()]
            if done % 40 == 0:
                print(f"  {done} of {len(tasks)} ({time.time() - started:.0f}s)", flush=True)
    data_path.unlink()
    errors = pd.DataFrame(rows)
    (outputs / "csv").mkdir(exist_ok=True)
    errors.to_csv(outputs / "csv" / "noise_column_screen_cells.csv", index=False)
    per_season = errors.groupby(["side", "arm", "yardstick", "group", "season"], as_index=False).error.mean()
    pairs = [("noise_missing", "today"), ("noise_dense", "today"), ("noise_dense", "noise_missing")]
    summary = pd.DataFrame([SCREEN.paired(per_season, arm, reference, side, yardstick, group)
                            for arm, reference in pairs for side, _ in SCREEN.SIDES
                            for yardstick, _ in SCREEN.YARDSTICKS for group in ("all", "bench")])
    summary.to_csv(outputs / "csv" / "noise_column_screen.csv", index=False)
    pd.set_option("display.width", 200)
    shown = ["arm", "reference", "side", "yardstick", "group", "mean_change", "se", "t", "seasons_better", "seasons"]
    print(summary[shown].round(5).to_string(index=False), flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)
    return 0


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="What the placebo gained from: missing values or extra columns.")
    parser.add_argument("--jobs", type=int, default=10, help="worker processes, one thread each (default %(default)s)")
    args = parser.parse_args(argv)
    with one_thread():
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
