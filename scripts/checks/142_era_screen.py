"""Check 142, the era screen: the box-score prior with its training rows weighted toward the rated season, or with
its ratios against the league, scored on held-out players before any build (notes/DECISIONS.md, experiment 56).

The owner, 2026-10-12: "any way to get era-level prior adjustments? one thought is moving the statistical stuff to
'vs league average' but that only solves part of the equation - the 'bias', not the 'slope'."  The prior learns one
price per input from all thirty seasons, and the defensive prices drift between eras (notes/DECISIONS.md, "Era drift").

Words used below
    screen season   one of ten rated seasons (`SCREEN`), at the test setting: it and its neighbours are left out of the
                    labels and of the training rows
    cell            one screen season under one assignment of players to the five player folds (0 is the shipped
                    assignment, 1 and 2 the noise control's regroupings: box_prior.balanced_player_folds)
    arm             one way to fit the prior: its input list and how much each training row counts (`arms`)
    half-life       a training row counts half as much for every that many seasons between its seasons and the rated
                    one, beyond the nearest season the fit may use (box_prior.era_kernel)
    yardstick       what a held-out prior is scored against.  "career": the player's step-1 label, his RAPM over every
                    season but the screen season and its neighbours (what the prior is trained on).  "neighbours": his
                    RAPM over the two neighbouring seasons alone, weighted by his possessions there (what the
                    year-over-year test scores, read player by player)
    held-out error  each rated player's prior comes from the booster fitted without his player fold (a player in no
                    fold from the booster fitted on every row, as in the build); the error is the possession-weighted
                    mean squared miss of the yardstick after the best straight line through the priors, the line
                    fitted on every rated player and the miss read on a group of them
    pre-1997 career a player first seen in 1997 at age 24 or older: his seasons played and career possessions are
                    counted from 1997 (data/roles.py, career_inputs), so they read low

The arms: today (the shipped prior); the ratios against the league (features.VS_LEAGUE, each in its ratio's place);
the shipped inputs at half-lives 2, 4, 8 and 16; both together at each half-life; and a diagnostic, today without the
pre-1997 careers' rows, which says whether those players' career counts, rather than drift, are what a weighting
would fix.  The player folds are drawn from the shipped row weights, so every arm splits the players the same way.

Statistics: each arm's error minus today's, the three assignments averaged within a season, then paired over the ten
seasons (t = the mean change over its standard error, 9 degrees of freedom); by era, the change and the seasons worse.

The gate, registered before any result (notes/DECISIONS.md, experiment 56).  An arm passes when
    (a) on the career yardstick it beats today at t -2 or below on at least one side,
    (b) neither side is worse at t +2 or above on either yardstick,
    (c) its mean change on the neighbours yardstick is not above 0 on a side that passed (a),
    (d) no era is worse in every one of its screen seasons, on either side, on the career yardstick.
The half-life: the longest passing one whose change (on defence; on offence if no passing one passed (a) on
defence) is within one standard error of the best passing one's; 2 is eligible only when 4 beats both 8 and 16.  The
ratios against the league join it only if both together pass and beat the half-life alone at t -2 or below on a side,
worse on neither side and neither yardstick at t +2.  With no half-life passing, the ratios alone are built if they
pass; with nothing passing, nothing is built.

Known blind spot: the error weights players by possessions, so the bench barely counts (notes/DECISIONS.md, trap 29);
it is read separately (players under 1,000 rated possessions) and not gated.

Writes outputs/csv/era_screen_cells.csv (every cell's error), outputs/csv/era_screen.csv (each arm against today,
paired by season), outputs/csv/era_screen_prices.csv (each arm's price per within-season standard deviation of the
inputs whose prices drift, from its booster fitted on every row), outputs/csv/era_screen_priors.parquet (every held-out
prior) and outputs/era_screen_decision.json (what the gate chose and its `openrapm rate` flags).

    python scripts/checks/142_era_screen.py --jobs=10      (about 15-20 minutes)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from openrapm.commands.console import one_thread, use_utf8

SCREEN = (1999, 2002, 2005, 2009, 2012, 2015, 2019, 2022, 2025, 2026)
ERAS = {"1997-2006": (1997, 2006), "2007-2016": (2007, 2016), "2017-2026": (2017, 2026)}
ASSIGNMENTS = (0, 1, 2)
N_FOLDS = 5
HALF_LIVES = (2, 4, 8, 16)
T_GATE = 2.0
BENCH = 1000.0                      # rated possessions below which a player is read as the bench
PRE_1997_AGE = 24.0                 # first seen in 1997 at this age or older: a career that began before 1997
SIDES = (("O", "offense"), ("D", "defense"))
YARDSTICKS = (("career", "possessions"), ("neighbours", "neighbours_possessions"))
GROUPS = ("all", "bench", "pre1997", "since1997")
# the inputs whose prices are read, each side: on defence the ones the era-drift test found moving, on offence the
# shooting ones whose league level moved most (a ratio is read through its twin in an arm that has the twin)
PRICED = {"O": ["three_point_attempt_share", "true_shooting_pct", "points_per100", "assists_per100",
                "off_rebounds_per100"],
          "D": ["height_in", "def_rebounds_per100", "used_possessions_per100", "steals_per100", "blocks_per100",
                "points_per100"]}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

_CELLS: dict = {}                   # in each worker: {(side, season): (training rows, rated rows)}
_PARAMS: dict = {}                  # in each worker: {side: LightGBM settings}
_ARMS: dict = {}                    # in each worker: arms()


# ------------------------------------------------------------------------------------------------ the arms
def arms() -> dict:
    """{name: ({side: input list}, the training rows' column that weighs each row, or None for the row weight
    alone)}."""
    from openrapm.features import FEATURE_SETS
    shipped, versus = FEATURE_SETS["box_score_and_bio"], FEATURE_SETS["ratios_vs_league"]
    out = {"today": (shipped, None), "vs_league": (versus, None), "pre1997_out": (shipped, "pre1997_out")}
    for half_life in HALF_LIVES:
        out[f"half_life_{half_life}"] = (shipped, f"kernel_{half_life}")
        out[f"vs_league_half_life_{half_life}"] = (versus, f"kernel_{half_life}")
    return out


def kernel_columns() -> list:
    return [f"kernel_{half_life}" for half_life in HALF_LIVES] + ["pre1997_out"]


# ------------------------------------------------------------------------------------------ what is read
def pre_1997_players(panel: pd.DataFrame) -> set:
    """Players first seen in 1997 at `PRE_1997_AGE` or older: their careers began before the data does."""
    first = panel.sort_values("season").groupby("player_id")[["season", "age"]].first()
    return set(first.index[(first.season == 1997) & (first.age >= PRE_1997_AGE)])


def prepare(cfg: dict, every_arm: dict | None = None) -> dict:
    """{(side, season): (training rows, rated rows)} for every screen season, with every arm's inputs and weights.

    `every_arm` is {name: ({side: input list}, ...)}, `arms()` unless given (check 143 hands over its own lists).

    The labels are step 1's RAPM over every trainable season but the screen season and its neighbours (the test
    setting); the neighbours yardstick is the same fit over the two neighbouring seasons alone.  Only the columns the
    workers read are kept."""
    from openrapm.box_prior import era_kernel, rated_rows, training_rows
    from openrapm.commands.build import accumulate_rapm, unseen_seasons
    from openrapm.data.designs import DesignCache
    from openrapm.data.panel import trainable_panel
    from openrapm.data.seasons import trainable
    from openrapm.features import PANEL_INPUTS, ROW_FEATURES
    from openrapm.rapm import career_labels
    from openrapm.settings import TEST
    seasons = list(trainable(cfg).seasons)
    sums = accumulate_rapm(DesignCache(cfg), seasons, TEST)
    panel = trainable_panel(cfg)
    panel = panel[panel.possessions > 0].reset_index(drop=True)
    pre1997 = pre_1997_players(panel)
    print(f"  {len(pre1997)} players first seen in 1997 at {PRE_1997_AGE:g} or older", flush=True)
    every_arm = arms() if every_arm is None else every_arm
    cells = {}
    for season in SCREEN:
        unseen = unseen_seasons(season, TEST)
        labels = career_labels(sums, unseen, TEST.labels, TEST.targets)
        apart = [other for other in seasons if other not in (season - 1, season + 1)]
        neighbours = career_labels(sums, apart, TEST.labels, TEST.targets)
        for side, column in SIDES:
            names = list(dict.fromkeys(name for inputs, _ in every_arm.values() for name in inputs[side]))
            mine = panel[panel.side == side]
            training = mine[~mine.season.isin(unseen)]
            weights = {f"kernel_{half_life}": era_kernel(training.season, season, half_life,
                                                         TEST.exclude_neighbours + 1) for half_life in HALF_LIVES}
            weights["pre1997_out"] = (~training.player_id.isin(pre1997)).to_numpy(float)
            train = training_rows(labels[side], training.assign(**weights), column, names, TEST.prior.chunk_sizes,
                                  [*PANEL_INPUTS, *weights])
            train = train[[*names, *ROW_FEATURES, "label", "row_weight", *weights]]
            rated = rated_rows(mine[mine.season == season], names)
            career = labels[side].set_index("player_id")[column]
            near = neighbours[side].set_index("player_id")
            rated = rated.assign(career=rated.player_id.map(career), neighbours=rated.player_id.map(near[column]),
                                 neighbours_possessions=rated.player_id.map(near.possessions),
                                 pre1997=rated.player_id.isin(pre1997))
            rated = rated[["player_id", "possessions", *names, *ROW_FEATURES, "career", "neighbours",
                           "neighbours_possessions", "pre1997"]].reset_index(drop=True)
            cells[(side, season)] = (train, rated)
        print(f"  {season}: labels and rows ready", flush=True)
    return cells


# ---------------------------------------------------------------------------------------- one cell's errors
def fitted_line(label: np.ndarray, prior: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """(a, b) of the weighted least-squares line label = a + b x prior."""
    design = np.column_stack([np.ones_like(prior), prior])
    root = np.sqrt(weight)
    return np.linalg.lstsq(design * root[:, None], label * root, rcond=None)[0]


def errors_of(rated: pd.DataFrame, prior: np.ndarray) -> dict:
    """{(yardstick, group): the weighted mean squared miss after the line fitted on every rated player}."""
    out = {}
    possessions = rated.possessions.to_numpy(float)
    pre1997 = rated.pre1997.to_numpy(bool)
    for yardstick, weight_column in YARDSTICKS:
        label, weight = rated[yardstick].to_numpy(float), rated[weight_column].to_numpy(float)
        usable = np.isfinite(label) & np.isfinite(weight) & (weight > 0) & np.isfinite(prior)
        a, b = fitted_line(label[usable], prior[usable], weight[usable])
        miss = label - (a + b * prior)
        groups = {"all": usable, "bench": usable & (possessions < BENCH), "pre1997": usable & pre1997,
                  "since1997": usable & ~pre1997}
        for group, mask in groups.items():
            out[(yardstick, group)] = (float(np.sum(weight[mask] * miss[mask] ** 2) / np.sum(weight[mask]))
                                       if mask.any() else np.nan)
    return out


def held_out_priors(train: pd.DataFrame, rated: pd.DataFrame, inputs: list, params: dict, assignment: int,
                    weighing: str | None):
    """(each rated player's prior from the booster fitted without his player fold, the booster fitted on every row).

    `weighing` names the training rows' column each row's weight is multiplied by after the rescale, as
    box_prior.fit_out_of_player does with the era kernel; a row it weighs 0 is left out."""
    from openrapm.box_prior import balanced_player_folds, fit_lightgbm
    from openrapm.features import ROW_FEATURES
    columns = [*inputs, *ROW_FEATURES]
    fold = balanced_player_folds(train, N_FOLDS, assignment)
    matrix, label, weight = (train[columns].to_numpy(float), train.label.to_numpy(float),
                             train.row_weight.to_numpy(float))
    kernel = None if weighing is None else train[weighing].to_numpy(float)
    used = np.ones(len(train), dtype=bool) if kernel is None else kernel > 0

    def fit(rows):
        rows = rows & used
        return fit_lightgbm(params, matrix[rows], label[rows], weight[rows], None if kernel is None else kernel[rows])

    full = fit(np.ones(len(train), dtype=bool))
    rated_matrix, rated_ids = rated[columns].to_numpy(float), rated.player_id.to_numpy()
    out = full.predict(rated_matrix)
    for left_out in range(N_FOLDS):
        keep = fold != left_out
        mine = np.isin(rated_ids, train.index[~keep].unique())
        if mine.any():
            out[mine] = fit(keep).predict(rated_matrix[mine])
    return out, full


def prices_of(model, rated: pd.DataFrame, inputs: list, side: str) -> list:
    """For each input in `PRICED`, the mean change in the booster's answer over the rated rows when that input
    alone moves up one standard deviation of the season's rated rows; also as a share of the answers' spread."""
    from openrapm.features import RATIOS, ROW_FEATURES
    columns = [*inputs, *ROW_FEATURES]
    matrix = rated[columns].to_numpy(float)
    base = model.predict(matrix)
    out = []
    for name in PRICED[side]:
        column = f"{name}_vs_league" if name in RATIOS and f"{name}_vs_league" in inputs else name
        if column not in inputs:
            continue
        shifted = matrix.copy()
        shifted[:, columns.index(column)] += float(rated[column].std())
        change = float(np.mean(model.predict(shifted) - base))
        out.append(dict(input=name, read_as=column, per_sd=change, share_of_spread=change / float(base.std())))
    return out


def cell_task(task: tuple) -> dict:
    """One (side, screen season, assignment, arm): its errors, priors, and (assignment 0) its prices.  Runs in a
    worker."""
    side, season, assignment, arm = task
    train, rated = _CELLS[(side, season)]
    inputs, weighing = _ARMS[arm][0][side], _ARMS[arm][1]
    prior, full = held_out_priors(train, rated, list(inputs), _PARAMS[side], assignment, weighing)
    return dict(task=task, errors=errors_of(rated, prior), prior=prior,
                prices=prices_of(full, rated, list(inputs), side) if assignment == 0 else [])


def _start_worker(data_path: str) -> None:
    for name in THREADS:
        os.environ[name] = "1"
    global _CELLS, _PARAMS, _ARMS
    with open(data_path, "rb") as handle:
        _CELLS, _PARAMS = pickle.load(handle)
    _ARMS = arms()


# ---------------------------------------------------------------------------------------------- the reading
def era_of(season: int) -> str:
    return next(name for name, (first, last) in ERAS.items() if first <= season <= last)


def paired(per_season: pd.DataFrame, arm: str, reference: str, side: str, yardstick: str, group: str) -> dict:
    """`arm` minus `reference`, the assignments already averaged, paired over the screen seasons."""
    pick = per_season[(per_season.side == side) & (per_season.yardstick == yardstick) & (per_season.group == group)]
    change = (pick[pick.arm == arm].set_index("season").error
              - pick[pick.arm == reference].set_index("season").error).dropna()
    n = len(change)
    se = float(change.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    mean = float(change.mean()) if n else np.nan
    row = dict(arm=arm, reference=reference, side=side, yardstick=yardstick, group=group, mean_change=mean, se=se,
               t=mean / se if se and se > 0 else np.nan, seasons_better=int((change < 0).sum()), seasons=n)
    for era in ERAS:
        part = change[[era_of(season) == era for season in change.index]]
        row[f"{era} change"] = float(part.mean()) if len(part) else np.nan
        row[f"{era} worse"] = f"{int((part > 0).sum())} of {len(part)}"
        row[f"{era} worse in every season"] = bool(len(part)) and bool((part > 0).all())
    return row


def passes(per_season: pd.DataFrame, arm: str, reference: str = "today") -> tuple[bool, list, str]:
    """(the gate's verdict for `arm` against `reference`, the sides that passed (a), why not when it fails)."""
    rows = {(side, yardstick): paired(per_season, arm, reference, side, yardstick, "all")
            for side, _ in SIDES for yardstick, _ in YARDSTICKS}
    won = [side for side, _ in SIDES
           if rows[(side, "career")]["t"] <= -T_GATE and rows[(side, "career")]["mean_change"] < 0]
    if not won:
        return False, [], "(a) beats the reference at t -2 on neither side"
    worse = [f"{side} {yardstick}" for (side, yardstick), row in rows.items() if row["t"] >= T_GATE]
    if worse:
        return False, won, f"(b) worse at t +2: {', '.join(worse)}"
    lost = [side for side in won if rows[(side, "neighbours")]["mean_change"] > 0]
    if lost:
        return False, won, f"(c) worse on the neighbours yardstick on {', '.join(lost)}"
    swept = [f"{side} {era}" for side, _ in SIDES for era in ERAS
             if rows[(side, "career")][f"{era} worse in every season"]]
    if swept:
        return False, won, f"(d) worse in every screen season of {', '.join(swept)}"
    return True, won, ""


def decide(per_season: pd.DataFrame) -> dict:
    """The gate in the module docstring: which arm is built, with its `openrapm rate` flags."""
    verdicts = {arm: passes(per_season, arm) for arm in arms() if arm not in ("today", "pre1997_out")}
    career = {(arm, side): paired(per_season, arm, "today", side, "career", "all")
              for arm in verdicts for side, _ in SIDES}
    four_best = all(career[("half_life_4", "D")]["mean_change"] < career[(f"half_life_{other}", "D")]["mean_change"]
                    for other in (8, 16))
    eligible = [half_life for half_life in HALF_LIVES if half_life != 2 or four_best]
    passing = [half_life for half_life in eligible if verdicts[f"half_life_{half_life}"][0]]
    out = dict(verdicts={arm: dict(passes=ok, sides=won, why=why) for arm, (ok, won, why) in verdicts.items()},
               half_life_2_eligible=four_best, chosen=None, half_life=None, with_vs_league=False, flags=None)
    if passing:
        side = "D" if any("D" in verdicts[f"half_life_{half_life}"][1] for half_life in passing) else "O"
        best = min(passing, key=lambda half_life: career[(f"half_life_{half_life}", side)]["mean_change"])
        bar = career[(f"half_life_{best}", side)]["mean_change"] + career[(f"half_life_{best}", side)]["se"]
        chosen = max(half_life for half_life in passing
                     if career[(f"half_life_{half_life}", side)]["mean_change"] <= bar)
        both = f"vs_league_half_life_{chosen}"
        joins = verdicts[both][0] and passes(per_season, both, f"half_life_{chosen}")[0]
        out.update(chosen=both if joins else f"half_life_{chosen}", half_life=chosen, with_vs_league=joins,
                   dial_side=side, flags=f"--era-halflife {chosen}" + (" --features ratios_vs_league" if joins else ""))
    elif verdicts["vs_league"][0]:
        out.update(chosen="vs_league", with_vs_league=True, flags="--features ratios_vs_league")
    return out


# --------------------------------------------------------------------------------------------------- the run
def run(args: argparse.Namespace) -> int:
    from openrapm.data.config import booster_params, load_config, resolve
    from openrapm.settings import TEST
    cfg = load_config()
    started, outputs = time.time(), resolve(cfg, "outputs")
    print("labels and rows for the screen seasons", flush=True)
    cells = prepare(cfg)
    data_path = outputs / "era_screen_cells.pkl"
    with open(data_path, "wb") as handle:
        pickle.dump((cells, booster_params(cfg, TEST.prior.booster_file)), handle)
    tasks = [(side, season, assignment, arm) for arm in arms() for side, _ in SIDES for season in SCREEN
             for assignment in ASSIGNMENTS]
    print(f"{len(tasks)} cells x arms, {len(tasks) * (N_FOLDS + 1)} fits ({time.time() - started:.0f}s)", flush=True)
    error_rows, prior_rows, price_rows = [], [], []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_start_worker, initargs=(str(data_path),)) as pool:
        for done, result in enumerate(pool.map(cell_task, tasks, chunksize=1), start=1):
            side, season, assignment, arm = result["task"]
            key = dict(side=side, season=season, assignment=assignment, arm=arm)
            error_rows += [dict(key, yardstick=yardstick, group=group, error=value)
                           for (yardstick, group), value in result["errors"].items()]
            rated = cells[(side, season)][1]
            prior_rows.append(pd.DataFrame(dict(key, player_id=rated.player_id.to_numpy(), prior=result["prior"])))
            price_rows += [dict(side=side, season=season, era=era_of(season), arm=arm, **price)
                           for price in result["prices"]]
            if done % 60 == 0:
                print(f"  {done} of {len(tasks)} ({time.time() - started:.0f}s)", flush=True)
    data_path.unlink()
    (outputs / "csv").mkdir(exist_ok=True)
    errors = pd.DataFrame(error_rows)
    errors.to_csv(outputs / "csv" / "era_screen_cells.csv", index=False)
    pd.concat(prior_rows, ignore_index=True).to_parquet(outputs / "csv" / "era_screen_priors.parquet", index=False)
    pd.DataFrame(price_rows).to_csv(outputs / "csv" / "era_screen_prices.csv", index=False)
    per_season = errors.groupby(["side", "arm", "yardstick", "group", "season"], as_index=False).error.mean()
    summary = pd.DataFrame([paired(per_season, arm, "today", side, yardstick, group)
                            for arm in arms() if arm != "today" for side, _ in SIDES
                            for yardstick, _ in YARDSTICKS for group in GROUPS])
    summary.to_csv(outputs / "csv" / "era_screen.csv", index=False)
    decision = decide(per_season)
    (outputs / "era_screen_decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    report(summary, decision)
    print(f"done in {time.time() - started:.0f}s", flush=True)
    return 0


def report(summary: pd.DataFrame, decision: dict) -> None:
    """Each arm against today, every player, both yardsticks; then the bench, the pre-1997 careers, the verdict."""
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    shown = ["arm", "side", "yardstick", "mean_change", "se", "t", "seasons_better", "seasons",
             *[f"{era} change" for era in ERAS], *[f"{era} worse" for era in ERAS]]
    for group, title in (("all", "every rated player"), ("bench", "players under 1,000 rated possessions"),
                         ("pre1997", "players whose careers began before 1997"),
                         ("since1997", "players whose careers began in 1997 or later")):
        print(f"\n=== each arm minus today, {title}: held-out error, below 0 is better ===", flush=True)
        print(summary[summary.group == group][shown].round(5).to_string(index=False), flush=True)
    print("\n=== the gate ===", flush=True)
    for arm, verdict in decision["verdicts"].items():
        print(f"  {arm}: {'passes' if verdict['passes'] else 'fails'} {verdict['why']}", flush=True)
    print(f"  chosen: {decision['chosen']}  flags: {decision['flags']}", flush=True)


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="The era screen: era-weighted and league-relative priors, held out.")
    parser.add_argument("--jobs", type=int, default=10, help="worker processes, one thread each (default %(default)s)")
    args = parser.parse_args(argv)
    with one_thread():
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
