"""Check 147, the era-flag screen: the box-score prior with an explicit era flag as an input, scored on held-out players
before any build (notes/DECISIONS.md, experiment 62's "Next").

Why.  Check 146's placebo (the dashboard columns shuffled among players, the missing values kept where they were) gained
on offence: its missing pattern told the booster which training rows come from before the dashboards started, an
indicator of the modern era, and the prior priced the modern rated seasons a little better for knowing it (career
yardstick -0.015, t -2.6, with no information in it).  This puts that indicator in by name, to see whether it is the
flag and how large.

Words used below
    era flag      1 for a season from a given year on and 0 before, a panel column in memory only (the panel file is not
                  touched); a training row that averages several seasons averages the flag over them by possessions (the
                  share of its seasons in the era), the rated season's row carries 0 or 1
    placebo       `odd_seasons`: 1 for an odd-numbered season, a split with no basketball meaning, which should
                  gain nothing
    screen season, cell, arm, yardstick, held-out error    as in check 142 (its ten screen seasons, 1999-2026)

The arms: `today` (features.FEATURE_SETS["no_career_counts"]), `era_from_2005`, `era_from_2014`, `both_flags`,
`odd_seasons`; each flag is appended to both sides' lists.  Statistics and the gate are check 142's, registered before
the first run (notes/DECISIONS.md, experiment 62): an arm passes when (a) on the career yardstick it beats today at t -2
or below on at least one side, (b) neither side is worse at t +2 or beyond on either yardstick, (c) its change on the
neighbours yardstick is not above 0 on a side that passed (a), (d) no era is worse in every one of its screen seasons;
the placebo must fail (a), and a real arm must beat the placebo at t -2 or below on a side that passed.  Nothing
passing: nothing is built.

Writes outputs/csv/era_flag_screen_cells.csv, era_flag_screen.csv (each arm against today and against the placebo,
paired by season), era_flag_screen_priors.parquet and outputs/era_flag_screen_decision.json.

    python scripts/checks/147_era_flag_screen.py --jobs=10      (about 25 minutes)
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

import pandas as pd

from openrapm.commands.console import one_thread, use_utf8


def _check_142():
    """Check 142 as a module: its prepared rows, held-out priors, error, paired statistics and gate are used here."""
    spec = importlib.util.spec_from_file_location("era_screen", Path(__file__).with_name("142_era_screen.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCREEN = _check_142()
FLAGS = {"era_from_2005": lambda season: (season >= 2005).astype(float),
         "era_from_2014": lambda season: (season >= 2014).astype(float),
         "odd_seasons": lambda season: (season % 2 == 1).astype(float)}
ARM_FLAGS = {"era_from_2005": ["era_from_2005"], "era_from_2014": ["era_from_2014"],
             "both_flags": ["era_from_2005", "era_from_2014"], "odd_seasons": ["odd_seasons"]}
GATE_ARMS = tuple(ARM_FLAGS)

_CELLS: dict = {}                   # in each worker: {(side, season): (training rows, rated rows)}
_PARAMS: dict = {}                  # in each worker: {side: LightGBM settings}
_ARMS: dict = {}                    # in each worker: arms()


def arms() -> dict:
    """{name: ({side: input list}, None)}: today and each flag arm (today's lists with the flags appended)."""
    from openrapm.features import FEATURE_SETS
    today = FEATURE_SETS["no_career_counts"]
    out = {"today": (today, None)}
    for name, flags in ARM_FLAGS.items():
        out[name] = ({side: today[side] + flags for side in ("O", "D")}, None)
    return out


def with_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """`panel` with every era flag as a column, from its season."""
    out = panel.copy()
    season = out.season.to_numpy()
    for name, make in FLAGS.items():
        out[name] = make(season)
    return out


def prepare(cfg: dict) -> dict:
    """check 142's `prepare` on its ten screen seasons with these arms, the flags in the panel in memory only."""
    from openrapm import features
    from openrapm.data import panel as panel_module
    real = panel_module.trainable_panel
    panel_module.trainable_panel = lambda cfg: with_flags(real(cfg))
    features.PANEL_INPUTS.extend(FLAGS)
    try:
        return SCREEN.prepare(cfg, arms())
    finally:
        panel_module.trainable_panel = real
        del features.PANEL_INPUTS[-len(FLAGS):]


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


def decide(per_season: pd.DataFrame) -> dict:
    """The gate for every arm; the placebo must fail (a), a real arm must also beat the placebo; which arm is built."""
    verdicts = {arm: SCREEN.passes(per_season, arm) for arm in GATE_ARMS}
    versus = {(arm, side): SCREEN.paired(per_season, arm, "odd_seasons", side, "career", "all")
              for arm in GATE_ARMS if arm != "odd_seasons" for side, _ in SCREEN.SIDES}
    career = {(arm, side): SCREEN.paired(per_season, arm, "today", side, "career", "all")
              for arm in GATE_ARMS for side, _ in SCREEN.SIDES}
    placebo_clean = not verdicts["odd_seasons"][0] and all(
        career[("odd_seasons", side)]["t"] > -SCREEN.T_GATE for side, _ in SCREEN.SIDES)
    real = [arm for arm in GATE_ARMS if arm != "odd_seasons"]
    passing = [arm for arm in real if verdicts[arm][0] and any(
        versus[(arm, side)]["t"] <= -SCREEN.T_GATE for side in verdicts[arm][1])]
    out = dict(verdicts={arm: dict(passes=ok, sides=won, why=why) for arm, (ok, won, why) in verdicts.items()},
               placebo_clean=placebo_clean,
               beats_placebo={f"{arm} {side}": versus[(arm, side)]["t"] for arm, side in versus},
               chosen=None, features=None)
    if passing:
        side = "D" if any("D" in verdicts[arm][1] for arm in passing) else "O"
        best = min(passing, key=lambda arm: career[(arm, side)]["mean_change"])
        bar = career[(best, side)]["mean_change"] + career[(best, side)]["se"]
        close = [arm for arm in passing if career[(arm, side)]["mean_change"] <= bar]
        chosen = min(close, key=lambda arm: (len(ARM_FLAGS[arm]), career[(arm, side)]["mean_change"]))
        out.update(chosen=chosen, best=best, dial_side=side, features=ARM_FLAGS[chosen])
    return out


def report(summary: pd.DataFrame, decision: dict) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    shown = ["arm", "reference", "side", "yardstick", "mean_change", "se", "t", "seasons_better", "seasons"]
    for group, title in (("all", "every rated player"), ("bench", "players under 1,000 rated possessions")):
        print(f"\n=== each arm minus its reference, {title}: held-out error, below 0 is better ===", flush=True)
        print(summary[summary.group == group][shown].round(5).to_string(index=False), flush=True)
    print("\n=== the gate ===", flush=True)
    for arm, verdict in decision["verdicts"].items():
        print(f"  {arm}: {'passes' if verdict['passes'] else 'fails'} (sides {verdict['sides'] or 'none'}) "
              f"{verdict['why']}", flush=True)
    print(f"  the placebo is clean: {decision['placebo_clean']}; against the placebo, career t: "
          f"{ {k: round(v, 2) for k, v in decision['beats_placebo'].items()} }", flush=True)
    print(f"  chosen: {decision['chosen']}", flush=True)


def run(args: argparse.Namespace) -> int:
    from openrapm.data.config import booster_params, load_config, resolve
    from openrapm.settings import TEST
    cfg = load_config()
    started, outputs = time.time(), resolve(cfg, "outputs")
    print(f"labels and rows for the screen seasons {SCREEN.SCREEN[0]}-{SCREEN.SCREEN[-1]}", flush=True)
    cells = prepare(cfg)
    data_path = outputs / "era_flag_screen_cells.pkl"
    with open(data_path, "wb") as handle:
        pickle.dump((cells, booster_params(cfg, TEST.prior.booster_file)), handle)
    tasks = [(side, season, assignment, arm) for arm in arms() for side, _ in SCREEN.SIDES
             for season in SCREEN.SCREEN for assignment in SCREEN.ASSIGNMENTS]
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
            if done % 50 == 0:
                print(f"  {done} of {len(tasks)} ({time.time() - started:.0f}s)", flush=True)
    data_path.unlink()
    (outputs / "csv").mkdir(exist_ok=True)
    errors = pd.DataFrame(error_rows)
    errors.to_csv(outputs / "csv" / "era_flag_screen_cells.csv", index=False)
    pd.concat(prior_rows, ignore_index=True).to_parquet(outputs / "csv" / "era_flag_screen_priors.parquet", index=False)
    per_season = errors.groupby(["side", "arm", "yardstick", "group", "season"], as_index=False).error.mean()
    comparisons = [(arm, "today") for arm in GATE_ARMS] + [(arm, "odd_seasons") for arm in GATE_ARMS
                                                           if arm != "odd_seasons"]
    summary = pd.DataFrame([SCREEN.paired(per_season, arm, reference, side, yardstick, group)
                            for arm, reference in comparisons for side, _ in SCREEN.SIDES
                            for yardstick, _ in SCREEN.YARDSTICKS for group in SCREEN.GROUPS])
    summary.to_csv(outputs / "csv" / "era_flag_screen.csv", index=False)
    decision = decide(per_season)
    (outputs / "era_flag_screen_decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    report(summary, decision)
    print(f"done in {time.time() - started:.0f}s", flush=True)
    return 0


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="The era-flag screen: an explicit era flag in the prior, held out.")
    parser.add_argument("--jobs", type=int, default=10, help="worker processes, one thread each (default %(default)s)")
    args = parser.parse_args(argv)
    with one_thread():
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
