"""Check 149, the matchup screen: the box-score prior with each player's matchup assignment added, scored on held-out
players before any build (notes/DECISIONS.md, experiment 62, "Matchups").

The owner, 2026-10-14: "let's look for hustle, tracking, synergy play types, and matchups".  Who a defender guarded, not
how the matchups went (the points a matchup produced are outcome data and are not read).  A defender's input is the
scoring of the attackers he guarded, an attacker's the stinginess of the defenders who guarded him
(data/dashboard_rates.py, features.MATCHUP_RATES).  A read-only look found the defender's version lined up with the
defensive games' part at +0.12 after his own prior, height and possessions, 9 of 9 seasons, and the attacker's at
nothing; one input per side is built anyway, the same method on both sides (the owner's rule), so the offensive side
tests the mirror.  This asks whether the defensive prior predicts a player better with the assignment than without.

Words used below
    screen season, cell, arm, held-out error, yardstick, placebo    as in check 146; the placebo here is the two matchup
                    columns shuffled among the players of a season and side, the missing values kept where they are
    assignment      the scoring of the attackers a defender guarded, each attacker's rate without the rows against that
                    defender, padded toward the league's; its mirror for an attacker (data/dashboard_rates.py)

The arms: `today` (features.FEATURE_SETS["no_career_counts"]); `matchups` (features.FEATURE_SETS["matchups"], one input
added per side); `placebo`.  The nine screen seasons are 2018-2026, the first nine with the matchups dashboard whole
(the 2017 file is a fragment); training rows from 2017 and earlier have the columns missing, as they would in a build.

The gate is check 146's, registered before the first run (notes/DECISIONS.md): `matchups` passes when (a) on the career
yardstick it beats today at t -2 or below on at least one side, (b) neither side is worse at t +2 or beyond on either
yardstick, (c) its mean change on the neighbours yardstick is not above 0 on a side that passed (a), (d) no era is
worse in every one of its screen seasons.  The placebo must fail (a): when it does not, a pass of `matchups` is not
credited to the assignment and nothing is built (`decide`).

Known blind spot: the error weights players by possessions, so the bench barely counts (trap 29); it is read separately
(players under 1,000 rated possessions) and not gated.

Writes outputs/csv/matchup_screen_cells.csv, matchup_screen.csv and outputs/matchup_screen_decision.json.

    python scripts/checks/149_matchup_screen.py --jobs=10      (about 10 minutes)
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


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCREEN = _load("era_screen", "142_era_screen.py")
DASHBOARD = _load("dashboard_screen", "146_dashboard_screen.py")
SEASONS = DASHBOARD.SCREEN_SEASONS
GATE_ARMS = ("matchups", "placebo")

_CELLS: dict = {}                   # in each worker: {(side, season): (training rows, rated rows)}
_PARAMS: dict = {}
_ARMS: dict = {}


def arms() -> dict:
    """{name: ({side: input list}, None)}: today, the matchup arm, and the same inputs shuffled."""
    from openrapm.features import FEATURE_SETS, MATCHUP_RATES
    matchups = FEATURE_SETS["matchups"]
    shuffled = {name: f"{DASHBOARD.PLACEBO_PREFIX}{name}" for name in MATCHUP_RATES}
    return {"today": (FEATURE_SETS["no_career_counts"], None),
            "matchups": (matchups, None),
            "placebo": ({side: [shuffled.get(name, name) for name in matchups[side]] for side in ("O", "D")}, None)}


def prepare(cfg: dict) -> dict:
    """check 142's `prepare` on the screen seasons, the shuffled matchup columns in the panel in memory only."""
    from openrapm import features
    from openrapm.data import panel as panel_module
    real = panel_module.trainable_panel
    shuffled = [f"{DASHBOARD.PLACEBO_PREFIX}{name}" for name in features.MATCHUP_RATES]

    def with_placebo(cfg):
        return DASHBOARD.shuffle_within(real(cfg), list(features.MATCHUP_RATES))
    panel_module.trainable_panel = with_placebo
    features.PANEL_INPUTS.extend(shuffled)
    features.MISSING_OK.extend(shuffled)
    SCREEN.SCREEN = SEASONS
    try:
        return SCREEN.prepare(cfg, arms())
    finally:
        panel_module.trainable_panel = real
        del features.PANEL_INPUTS[-len(shuffled):]
        del features.MISSING_OK[-len(shuffled):]


def cell_task(task: tuple) -> dict:
    """One (side, screen season, assignment, arm): its errors.  Runs in a worker."""
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


def decide(per_season: pd.DataFrame) -> dict:
    """The gate for the matchup arm and the placebo (module docstring)."""
    verdicts = {arm: SCREEN.passes(per_season, arm) for arm in GATE_ARMS}
    career = {(arm, side): SCREEN.paired(per_season, arm, "today", side, "career", "all")
              for arm in GATE_ARMS for side, _ in SCREEN.SIDES}
    passes = bool(verdicts["matchups"][0])
    clean = not verdicts["placebo"][0] and all(
        career[("placebo", side)]["t"] > -SCREEN.T_GATE for side, _ in SCREEN.SIDES)
    # the registered rule: the placebo must fail (a), or a pass of the real arm is not credited to the assignment
    chosen = "matchups" if passes and clean else None
    return dict(verdicts={arm: dict(passes=ok, sides=won, why=why) for arm, (ok, won, why) in verdicts.items()},
                placebo_clean=clean, blocked_by_placebo=passes and not clean,
                best="matchups" if passes else None, chosen=chosen, features=chosen)


def run(args: argparse.Namespace) -> int:
    from openrapm.data.config import booster_params, load_config, resolve
    from openrapm.settings import TEST
    cfg = load_config()
    started, outputs = time.time(), resolve(cfg, "outputs")
    print(f"labels and rows for the screen seasons {SEASONS[0]}-{SEASONS[-1]}", flush=True)
    cells = prepare(cfg)
    data_path = outputs / "matchup_screen_cells.pkl"
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
    errors.to_csv(outputs / "csv" / "matchup_screen_cells.csv", index=False)
    per_season = errors.groupby(["side", "arm", "yardstick", "group", "season"], as_index=False).error.mean()
    summary = pd.DataFrame([SCREEN.paired(per_season, arm, "today", side, yardstick, group)
                            for arm in GATE_ARMS for side, _ in SCREEN.SIDES
                            for yardstick, _ in SCREEN.YARDSTICKS for group in SCREEN.GROUPS])
    summary.to_csv(outputs / "csv" / "matchup_screen.csv", index=False)
    decision = decide(per_season)
    (outputs / "matchup_screen_decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    DASHBOARD.report(summary, decision)
    print(f"done in {time.time() - started:.0f}s", flush=True)
    return 0


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="The matchup screen: each assignment as a prior input, held out.")
    parser.add_argument("--jobs", type=int, default=10, help="worker processes, one thread each (default %(default)s)")
    args = parser.parse_args(argv)
    with one_thread():
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
