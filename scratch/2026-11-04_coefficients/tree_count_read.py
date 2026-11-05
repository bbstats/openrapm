"""Read-only: why 130 trees?  Three reads, for the owner's question of 2026-11-05: "Why did we choose that tree number?"

1. The guarded search that chose 130: what its best trials chose for defense's trees and how far they reached.
2. A tree-count curve: defense's boosters fitted to 1,000 trees, the same rows, resamples and player folds as the saved
   ones, read at each count on the LightGBM's own goal, the career RAPM of seasons it did not see, among regulars.
3. How much of the LightGBM's output a straight line on its own inputs reproduces, by side and era.

Usage: tree_count_read.py 2000,2005,2010,2015,2022,2026 [processes at once, default 6]
"""
import os
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import optuna
import pandas as pd

from openrapm import box_prior, rapm
from openrapm.commands.build import accumulate_rapm, part_season_panel, trainable_panel
from openrapm.data.config import booster_params, load_config
from openrapm.data.designs import DesignCache
from openrapm.data.seasons import trainable
from openrapm.features import FEATURE_SETS, PANEL_INPUTS, ROW_FEATURES
from openrapm.settings import TEST

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
SEASONS = [int(s) for s in sys.argv[1].split(",")]
JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
COUNTS = (50, 100, 130, 200, 330, 500, 750, 1000)


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


def slope(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / np.average((x - mx) ** 2, weights=w))


def correlation(x, y, w):
    return slope(x, y, w) * wsd(x, w) / wsd(y, w)


def search_read():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = "sqlite:///outputs/tune/tune1/guarded.db"
    names = [s.study_name for s in optuna.get_all_study_summaries(storage)]
    print("studies:", names)
    study = optuna.load_study(study_name=names[0], storage=storage)
    rows = []
    for trial in study.trials:
        if trial.value is None:
            continue
        p = trial.params
        guard = trial.user_attrs.get("guard_team_game", {})
        rows.append(dict(trial=trial.number, decider=trial.user_attrs.get("vs_reference", {}).get("mean_change", np.nan), value=trial.value,
                         guard_z=guard.get("z", np.nan), D_trees=p["D_n_estimators"], D_rate=p["D_learning_rate"],
                         D_reach=1 - (1 - p["D_learning_rate"]) ** p["D_n_estimators"], D_rows_per_leaf=p["D_min_child_samples"],
                         O_trees=p["O_n_estimators"], O_rate=p["O_learning_rate"],
                         O_reach=1 - (1 - p["O_learning_rate"]) ** p["O_n_estimators"]))
    t = pd.DataFrame(rows)
    t.to_csv("outputs/csv/tree_count_search.csv", index=False)
    legal = t[t.guard_z <= 1.0].sort_values("value")
    print(f"{len(t)} trials, {len(legal)} inside the year-over-year limit; best 12 inside it:")
    print(legal.head(12).round(4).to_string(index=False))
    print("all trials inside the limit, score against defense's trees and reach, Spearman:",
          legal[["value", "D_trees", "D_reach", "O_trees", "O_reach", "D_rows_per_leaf"]].corr(method="spearman")["value"].round(3).to_dict())
    best = legal.value.min()
    close = legal[legal.value <= best + 0.03]
    print(f"trials within 0.03 of the best, the swing a booster's seed alone moved this test: {len(close)}; their defense trees "
          f"{close.D_trees.min()} to {close.D_trees.max()}, reach {close.D_reach.min():.2f} to {close.D_reach.max():.2f}")
    close = legal[legal.value <= best + 0.1]
    print(f"trials within 0.10 of the best: {len(close)}; their defense trees {close.D_trees.min()} to {close.D_trees.max()}, "
          f"reach {close.D_reach.min():.2f} to {close.D_reach.max():.2f}")


def answer(prior: box_prior.BaggedPrior, rows: pd.DataFrame, trees: int) -> np.ndarray:
    matrix = rows[prior.inputs].to_numpy(float)
    ids = rows.player_id.to_numpy()
    total = np.zeros(len(rows))
    for member in prior.members:
        out = member.full.predict(matrix, num_iteration=trees)
        for model, held_out in zip(member.fold_models, member.held_out_players):
            mine = np.fromiter((pid in held_out for pid in ids), dtype=bool, count=len(ids))
            if mine.any():
                out[mine] = model.predict(matrix[mine], num_iteration=trees)
        total += out
    return total / len(prior.members)


def curve_season(season, train, rows, inputs, params, label, table):
    settings = TEST.prior
    prior = box_prior.fit_prior(train, inputs, params, settings.n_folds, settings.fold_assignment, None, settings.bags,
                                settings.prediction, settings.split_per_bag, settings.bag_draw)
    answers = {count: pd.Series(answer(prior, rows, count), index=rows.player_id.to_numpy()) for count in COUNTS}
    keep = table.index[table.poss_def >= 1000].intersection(rows.player_id).intersection(label.index)
    w = table.loc[keep, "poss_def"].to_numpy(float)
    y = label.loc[keep, "defense"].to_numpy(float)
    base = answers[130].loc[keep].to_numpy(float)
    out = []
    for count in COUNTS:
        x = answers[count].loc[keep].to_numpy(float)
        out.append(dict(season=season, trees=count, output_sd=wsd(x, w), target_per_point=slope(x, y, w), target_correlation=correlation(x, y, w),
                        target_rmse=float(np.sqrt(np.average((y - x) ** 2, weights=w))), correlation_with_130=correlation(base, x, w)))
    print(f"{season} done", flush=True)
    return out


def main():
    search_read()
    cfg = load_config()
    designs = DesignCache(cfg)
    sums = accumulate_rapm(designs, trainable(cfg).seasons, TEST)
    panel = trainable_panel(cfg)
    panel = panel[panel.possessions > 0].reset_index(drop=True)
    parts = part_season_panel(cfg, TEST)
    published = pd.read_parquet("outputs/season_ratings_mid136.parquet")
    params = {**booster_params(cfg, TEST.prior.booster_file)["D"], "n_estimators": max(COUNTS)}
    names = list(FEATURE_SETS[TEST.prior.feature_set]["D"])
    tasks = []
    for season in SEASONS:
        unseen = [season - 1, season, season + 1]
        labels = rapm.career_labels(sums, unseen, TEST.labels, TEST.targets)["D"]
        train = box_prior.training_rows(labels, box_prior.training_seasons(panel, "D", unseen, season, TEST), "defense", names,
                                        TEST.prior.chunk_sizes, PANEL_INPUTS, None, box_prior.training_seasons(parts, "D", unseen, season, TEST))
        rows = box_prior.rated_rows(panel[(panel.side == "D") & (panel.season == season)], names, 1.0, box_prior.least_share(TEST))
        tasks.append((season, train, rows, names + ROW_FEATURES, params, labels.set_index("player_id"),
                      published[published.season == season].set_index("player_id")))
    del sums, designs
    with ThreadPoolExecutor(JOBS) as pool:
        results = [f.result() for f in [pool.submit(curve_season, *task) for task in tasks]]
    curve = pd.DataFrame([row for season in results for row in season])
    curve.to_csv("outputs/csv/tree_count_curve.csv", index=False)
    print(curve.groupby("trees")[["output_sd", "target_per_point", "target_correlation", "target_rmse", "correlation_with_130"]].mean().round(4).to_string())
    print(curve.pivot(index="season", columns="trees", values="target_rmse").round(3).to_string())
    print(curve.pivot(index="season", columns="trees", values="target_per_point").round(2).to_string())

    models = pd.read_pickle("outputs/prior_models_rating134.pkl")
    rows = []
    for era, low, high in (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026)):
        for side in ("O", "D"):
            names = list(FEATURE_SETS[TEST.prior.feature_set][side])
            frames = []
            for season in range(low, high + 1):
                source = box_prior.rated_rows(panel[(panel.side == side) & (panel.season == season)], names, 1.0, box_prior.least_share(TEST))
                source = source.assign(output=source.player_id.map(pd.Series(models[season][side]["prior"], dtype=float)), season=season)
                frames.append(source)
            frame = pd.concat(frames, ignore_index=True).dropna(subset=["output"])
            frame = frame[frame.possessions >= 1000]
            columns = [c for c in names if frame[c].notna().mean() > 0.5 and frame[c].std() > 0]
            matrix = frame[columns].fillna(frame[columns].mean()).to_numpy(float)
            matrix = np.column_stack([np.ones(len(frame)), (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)])
            w = np.sqrt(frame.possessions.to_numpy(float))
            y = frame.output.to_numpy(float)
            half = frame.player_id.to_numpy() % 2 == 0
            fit = np.linalg.lstsq(matrix[half] * w[half, None], y[half] * w[half], rcond=None)[0]
            held = ~half
            miss = y[held] - matrix[held] @ fit
            weights = frame.possessions.to_numpy(float)[held]
            share = 1 - np.average(miss ** 2, weights=weights) / np.average((y[held] - np.average(y[held], weights=weights)) ** 2, weights=weights)
            rows.append(dict(era=era, side=side, regulars=len(frame), inputs=len(columns), line_share_of_output=share))
    print("share of the LightGBM output's variance a straight line on its inputs gets, on players the line did not see")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
