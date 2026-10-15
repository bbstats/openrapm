"""Check 146, the dashboard screen: the box-score prior with the tracking and hustle dashboards added, scored on
held-out players before any build (notes/DECISIONS.md, experiment 62).

The owner, 2026-10-14: "let's look for hustle, tracking, synergy play types, and matchups".  The tracking and hustle
dashboards (data/dashboards.py, data/dashboard_rates.py) count what the box score cannot: rim shots defended, shots made
against a defender above expected, rebound chances, contested shots, box-outs, deflections, charges, drives, potential
assists, touches.  This asks whether they let the prior predict a player better than the box score and his body do.

Words used below
    screen season   one of nine rated seasons, 2018-2026, at the test setting: it and its neighbours are left out of the
                    labels and of the training rows (every screen season and its neighbours have both dashboards)
    cell            one screen season under one assignment of players to the five player folds (0 is the shipped
                    assignment, 1 and 2 the noise control's regroupings)
    arm             one way to fit the prior: its input list
    held-out error, yardstick   as in check 142: each rated player's prior comes from the booster fitted without his
                    player fold, scored against his career label (step 1's RAPM over every season but the screen
                    season and its neighbours) and against his neighbours' RAPM, possession-weighted, after the best
                    straight line through the priors
    placebo         the `tracking_and_hustle` inputs with every dashboard column shuffled among the players of its
                    season and side, the missing values kept where they are: the same columns with nothing in them.
                    It joins the panel in memory only

The arms: `today` (features.FEATURE_SETS["no_career_counts"]); `tracking`; `hustle`; `tracking_and_hustle`; `placebo`.
Training rows from the seasons before a dashboard starts have its columns missing (tracking before 2014, hustle before
2016), as they would in a build; the prior reads a missing value as missing (box_prior.py, features.MISSING_OK).

Statistics and the gate are check 142's, registered before the first run (notes/DECISIONS.md, experiment 62): an arm
passes when (a) on the career yardstick it beats today at t -2 or below on at least one side, (b) neither side is worse
at t +2 or beyond on either yardstick, (c) its mean change on the neighbours yardstick is not above 0 on a side that
passed (a), (d) no era is worse in every one of its screen seasons.  The placebo must fail (a).  Of the passing arms the
best has the largest career-yardstick gain on the side that passed (defence when it did); the chosen one is the arm with
the fewest added inputs whose change is within one standard error of the best's.  With nothing passing nothing is built.

Known blind spot: the error weights players by possessions, so the bench barely counts (trap 29); it is read separately
(players under 1,000 rated possessions) and not gated.

Writes outputs/csv/dashboard_screen_cells.csv (every cell's error), dashboard_screen.csv (each arm against today and the
combined arm against each part, paired by season), dashboard_screen_priors.parquet (every held-out prior) and
outputs/dashboard_screen_decision.json (what the gate chose and its `openrapm rate` flag).

    python scripts/checks/146_dashboard_screen.py --jobs=10      (about 10 minutes)
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
    """Check 142 as a module: its prepared rows, held-out priors, error, paired statistics and gate are used here."""
    spec = importlib.util.spec_from_file_location("era_screen", Path(__file__).with_name("142_era_screen.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCREEN = _check_142()
SCREEN_SEASONS = tuple(range(2018, 2027))
REAL_ARMS = ("tracking", "hustle", "tracking_and_hustle")
GATE_ARMS = (*REAL_ARMS, "placebo")
PLACEBO_PREFIX = "shuffled_"

_CELLS: dict = {}                   # in each worker: {(side, season): (training rows, rated rows)}
_PARAMS: dict = {}                  # in each worker: {side: LightGBM settings}
_ARMS: dict = {}                    # in each worker: arms()


# ------------------------------------------------------------------------------------------------ the arms
def arms() -> dict:
    """{name: ({side: input list}, None)}: today, the three dashboard arms and the placebo (the same inputs,
    shuffled)."""
    from openrapm.features import FEATURE_SETS
    out = {"today": (FEATURE_SETS["no_career_counts"], None)}
    out.update({name: (FEATURE_SETS[name], None) for name in REAL_ARMS})
    both = FEATURE_SETS["tracking_and_hustle"]
    out["placebo"] = ({side: [f"{PLACEBO_PREFIX}{name}" if name.endswith(("_vs_league_avg", "_per_shot")) else name
                              for name in both[side]] for side in ("O", "D")}, None)
    return out


def added_inputs(arm: str) -> int:
    """How many inputs the arm adds to today's lists, both sides together."""
    from openrapm.features import FEATURE_SETS
    return sum(len(inputs) - len(FEATURE_SETS["no_career_counts"][side]) for side, inputs in arms()[arm][0].items())


def shuffle_within(panel: pd.DataFrame, columns: list, seed: int = 0) -> pd.DataFrame:
    """`panel` with a `shuffled_<column>` beside each of `columns`: its values permuted among the rows of the same
    season and side that have one (a missing value stays where it is)."""
    rng = np.random.default_rng(seed)
    out = panel.copy()
    for column in columns:
        values = out[column].to_numpy(float).copy()
        for _, rows in out.groupby(["season", "side"]).indices.items():
            known = rows[np.isfinite(values[rows])]
            values[known] = values[rng.permutation(known)]
        out[f"{PLACEBO_PREFIX}{column}"] = values
    return out


def prepare(cfg: dict, seasons=SCREEN_SEASONS) -> dict:
    """check 142's `prepare` on the screen seasons with the five arms' lists, the placebo's columns in the panel.

    The shuffled columns exist in memory only: the panel read is wrapped, and the feature lists the readers average
    over (features.PANEL_INPUTS) and may find missing (features.MISSING_OK) are extended for this run."""
    from openrapm import features
    from openrapm.data import panel as panel_module
    placebo = [name for name in features.DASHBOARD_RATES]
    shuffled = [f"{PLACEBO_PREFIX}{name}" for name in placebo]
    real = panel_module.trainable_panel

    def with_placebo(cfg):
        return shuffle_within(real(cfg), placebo)
    panel_module.trainable_panel = with_placebo
    features.PANEL_INPUTS.extend(shuffled)
    features.MISSING_OK.extend(shuffled)
    SCREEN.SCREEN = seasons
    try:
        return SCREEN.prepare(cfg, arms())
    finally:
        panel_module.trainable_panel = real
        del features.PANEL_INPUTS[-len(shuffled):]
        del features.MISSING_OK[-len(shuffled):]


# ---------------------------------------------------------------------------------------- one cell's errors
def cell_task(task: tuple) -> dict:
    """One (side, screen season, assignment, arm): its errors and held-out priors.  Runs in a worker."""
    side, season, assignment, arm = task
    train, rated = _CELLS[(side, season)]
    prior, _ = SCREEN.held_out_priors(train, rated, list(_ARMS[arm][0][side]), _PARAMS[side], assignment, None)
    return dict(task=task, errors=SCREEN.errors_of(rated, prior), prior=prior)


def _start_worker(data_path: str) -> None:
    for name in SCREEN.THREADS:
        os.environ[name] = "1"
    global _CELLS, _PARAMS, _ARMS
    with open(data_path, "rb") as handle:
        _CELLS, _PARAMS = pickle.load(handle)
    _ARMS = arms()


# ---------------------------------------------------------------------------------------------- the reading
def decide(per_season: pd.DataFrame) -> dict:
    """The gate for every arm (module docstring); the best and the chosen passing arm, None when none passes."""
    verdicts = {arm: SCREEN.passes(per_season, arm) for arm in GATE_ARMS}
    career = {(arm, side): SCREEN.paired(per_season, arm, "today", side, "career", "all")
              for arm in GATE_ARMS for side, _ in SCREEN.SIDES}
    passing = [arm for arm in REAL_ARMS if verdicts[arm][0]]
    out = dict(verdicts={arm: dict(passes=ok, sides=won, why=why) for arm, (ok, won, why) in verdicts.items()},
               placebo_clean=not verdicts["placebo"][0] and all(
                   career[("placebo", side)]["t"] > -SCREEN.T_GATE for side, _ in SCREEN.SIDES),
               best=None, chosen=None, features=None)
    if passing:
        side = "D" if any("D" in verdicts[arm][1] for arm in passing) else "O"
        best = min(passing, key=lambda arm: career[(arm, side)]["mean_change"])
        bar = career[(best, side)]["mean_change"] + career[(best, side)]["se"]
        close = [arm for arm in passing if career[(arm, side)]["mean_change"] <= bar]
        chosen = min(close, key=lambda arm: (added_inputs(arm), career[(arm, side)]["mean_change"]))
        out.update(best=best, chosen=chosen, dial_side=side, features=chosen)
    return out


def report(summary: pd.DataFrame, decision: dict) -> None:
    """Each arm against today, every player, both yardsticks; then the bench and the pre-1997 careers; the verdict."""
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    shown = ["arm", "reference", "side", "yardstick", "mean_change", "se", "t", "seasons_better", "seasons"]
    for group, title in (("all", "every rated player"), ("bench", "players under 1,000 rated possessions"),
                         ("pre1997", "players whose careers began before 1997")):
        print(f"\n=== each arm minus its reference, {title}: held-out error, below 0 is better ===", flush=True)
        print(summary[(summary.group == group)][shown].round(5).to_string(index=False), flush=True)
    print("\n=== the gate ===", flush=True)
    for arm, verdict in decision["verdicts"].items():
        print(f"  {arm}: {'passes' if verdict['passes'] else 'fails'} (sides {verdict['sides'] or 'none'}) "
              f"{verdict['why']}", flush=True)
    print(f"  the placebo is clean: {decision['placebo_clean']}", flush=True)
    print(f"  best: {decision['best']}  chosen: {decision['chosen']}", flush=True)


def run(args: argparse.Namespace) -> int:
    from openrapm.data.config import booster_params, load_config, resolve
    from openrapm.settings import TEST
    cfg = load_config()
    started, outputs = time.time(), resolve(cfg, "outputs")
    seasons = tuple(range(args.first, args.last + 1))
    print(f"labels and rows for the screen seasons {seasons[0]}-{seasons[-1]}", flush=True)
    cells = prepare(cfg, seasons)
    data_path = outputs / "dashboard_screen_cells.pkl"
    with open(data_path, "wb") as handle:
        pickle.dump((cells, booster_params(cfg, TEST.prior.booster_file)), handle)
    tasks = [(side, season, assignment, arm) for arm in arms() for side, _ in SCREEN.SIDES
             for season in seasons for assignment in SCREEN.ASSIGNMENTS]
    print(f"{len(tasks)} cells, {len(tasks) * (SCREEN.N_FOLDS + 1)} fits ({time.time() - started:.0f}s)", flush=True)
    error_rows, prior_rows = [], []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_start_worker, initargs=(str(data_path),)) as pool:
        for done, result in enumerate(pool.map(cell_task, tasks, chunksize=1), start=1):
            side, season, assignment, arm = result["task"]
            key = dict(side=side, season=season, assignment=assignment, arm=arm)
            error_rows += [dict(key, yardstick=yardstick, group=group, error=value)
                           for (yardstick, group), value in result["errors"].items()]
            rated = cells[(side, season)][1]
            prior_rows.append(pd.DataFrame(dict(key, player_id=rated.player_id.to_numpy(), prior=result["prior"])))
            if done % 40 == 0:
                print(f"  {done} of {len(tasks)} ({time.time() - started:.0f}s)", flush=True)
    data_path.unlink()
    (outputs / "csv").mkdir(exist_ok=True)
    errors = pd.DataFrame(error_rows)
    errors.to_csv(outputs / "csv" / "dashboard_screen_cells.csv", index=False)
    pd.concat(prior_rows, ignore_index=True).to_parquet(outputs / "csv" / "dashboard_screen_priors.parquet",
                                                         index=False)
    per_season = errors.groupby(["side", "arm", "yardstick", "group", "season"], as_index=False).error.mean()
    comparisons = [(arm, "today") for arm in GATE_ARMS] + [("tracking_and_hustle", "tracking"),
                                                           ("tracking_and_hustle", "hustle")]
    summary = pd.DataFrame([SCREEN.paired(per_season, arm, reference, side, yardstick, group)
                            for arm, reference in comparisons for side, _ in SCREEN.SIDES
                            for yardstick, _ in SCREEN.YARDSTICKS for group in SCREEN.GROUPS])
    summary.to_csv(outputs / "csv" / "dashboard_screen.csv", index=False)
    decision = decide(per_season)
    (outputs / "dashboard_screen_decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    report(summary, decision)
    print(f"done in {time.time() - started:.0f}s", flush=True)
    return 0


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="The dashboard screen: tracking and hustle inputs, held out.")
    parser.add_argument("--jobs", type=int, default=10, help="worker processes, one thread each (default %(default)s)")
    parser.add_argument("--first", type=int, default=SCREEN_SEASONS[0],
                        help="first screen season (default %(default)s)")
    parser.add_argument("--last", type=int, default=SCREEN_SEASONS[-1], help="last screen season (default %(default)s)")
    args = parser.parse_args(argv)
    with one_thread():
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
