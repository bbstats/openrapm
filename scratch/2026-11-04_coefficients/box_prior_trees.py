"""Read-only: does fitting the defensive boosters further change the shape of their output, or only its scale?

Today's defensive boosters stop at 130 trees.  Here each is fitted to 330 on the training rows, resamples and player
folds of the saved test-setting boosters (experiment 134), and read twice: at its first 130 trees and at all 330.
LightGBM adds trees in order, so the first 130 are the 130-tree fit; the first column of the result checks that against
the saved answers.

A change of scale cannot move a rating (notes/HANDOFF.md, "Next job"), so the number to read is what is left of the
330-tree answer after the best straight line on the 130-tree answer, in points per 100 of the Box score column: that
residual's standard deviation x Box Adj's weight x the prior scale x the shrink / the straight line's slope.

Usage: box_prior_trees.py 2000,2005,2010,2015,2022,2026 [processes at once, default 6]
Each season is written to outputs/box_prior_trees/ as it finishes, so a stopped run goes on from there.
"""
import os
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm import box_prior, rapm
from openrapm.commands.build import accumulate_rapm, part_season_panel, trainable_panel
from openrapm.data.config import booster_params, load_config
from openrapm.data.designs import DesignCache
from openrapm.data.seasons import trainable
from openrapm.features import FEATURE_SETS, PANEL_INPUTS, ROW_FEATURES
from openrapm.settings import TEST

sys.stdout.reconfigure(encoding="utf-8")
TODAY, FURTHER = 130, 330
SEASONS = [int(s) for s in sys.argv[1].split(",")]
JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
FOLDER = Path("outputs/box_prior_trees")


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


def slope(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / np.average((x - mx) ** 2, weights=w))


def correlation(x, y, w):
    return slope(x, y, w) * wsd(x, w) / wsd(y, w)


def answer(prior: box_prior.BaggedPrior, rows: pd.DataFrame, trees: int) -> np.ndarray:
    """box_prior.predict at each booster's first `trees` trees."""
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


def read_season(season, train, rows, inputs, params, label, table, net, saved):
    """Fit one season's defensive boosters to 330 trees and write its row and its players."""
    started = time.time()
    settings = TEST.prior
    prior = box_prior.fit_prior(train, inputs, params, settings.n_folds, settings.fold_assignment, None, settings.bags,
                                settings.prediction, settings.split_per_bag, settings.bag_draw)
    today = pd.Series(answer(prior, rows, TODAY), index=rows.player_id.to_numpy())
    further = pd.Series(answer(prior, rows, FURTHER), index=rows.player_id.to_numpy())
    keep = table.index[table.poss_def >= 1000].intersection(today.index)
    w = table.loc[keep, "poss_def"].to_numpy(float)
    a, b = today.loc[keep].to_numpy(float), further.loc[keep].to_numpy(float)
    both = keep.intersection(label.index)
    wb = table.loc[both, "poss_def"].to_numpy(float)
    y = label.loc[both, "defense"].to_numpy(float)
    ax, bx = today.loc[both].to_numpy(float), further.loc[both].to_numpy(float)
    rescale = slope(a, b, w)
    residual = (b - np.average(b, weights=w)) - rescale * (a - np.average(a, weights=w))
    row = dict(season=season, regulars=len(keep), players_with_target=len(both),
               largest_gap_from_saved_answers=float((today - pd.Series(saved).reindex(today.index)).abs().max()),
               correlation_130_with_330=correlation(a, b, w),
               correlation_every_player=float(np.corrcoef(today, further)[0, 1]),
               output_sd_130=wsd(a, w), output_sd_330=wsd(b, w), target_sd=wsd(y, wb),
               target_per_output_point_130=slope(ax, y, wb), target_per_output_point_330=slope(bx, y, wb),
               target_correlation_130=correlation(ax, y, wb), target_correlation_330=correlation(bx, y, wb),
               straight_line_slope=rescale, residual_sd=wsd(residual, w), net=net,
               box_column_sd=net * wsd(a, w), box_column_move=wsd(residual, w) * net / rescale)
    frame = pd.DataFrame(dict(player_id=today.index, season=season, trees_130=today.to_numpy(), trees_330=further.to_numpy()))
    written = FOLDER / f"{season}.csv.part"
    pd.DataFrame([row]).to_csv(written, index=False)
    written.replace(FOLDER / f"{season}.csv")
    frame.to_csv(FOLDER / f"players_{season}.csv", index=False)
    print(f"{season}: {time.time() - started:.0f}s, 130 against 330 correlate {row['correlation_130_with_330']:.4f}, "
          f"largest gap from the saved answers {row['largest_gap_from_saved_answers']:.2e}", flush=True)


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    missing = [s for s in SEASONS if not (FOLDER / f"{s}.csv").exists()]
    if missing:
        cfg = load_config()
        models = pd.read_pickle("outputs/prior_models_rating134.pkl")
        saved = {s: (dict(models[s]["D"]["prior"]), list(models[s]["D"]["unseen"])) for s in missing}
        del models
        designs = DesignCache(cfg)
        sums = accumulate_rapm(designs, trainable(cfg).seasons, TEST)
        panel = trainable_panel(cfg)
        panel = panel[panel.possessions > 0].reset_index(drop=True)
        parts = part_season_panel(cfg, TEST)
        published = pd.read_parquet("outputs/season_ratings_product.parquet")
        stages = pd.read_csv("outputs/csv/box_prior_stages.csv")
        nets = stages[stages.side == "D"].set_index("season").net
        params = {**booster_params(cfg, TEST.prior.booster_file)["D"], "n_estimators": FURTHER}
        names = list(FEATURE_SETS[TEST.prior.feature_set]["D"])
        inputs = names + ROW_FEATURES
        tasks = []
        for season in missing:
            prior_values, unseen = saved[season]
            labels = rapm.career_labels(sums, unseen, TEST.labels, TEST.targets)["D"]
            train = box_prior.training_rows(labels, box_prior.training_seasons(panel, "D", unseen, season, TEST), "defense",
                                            names, TEST.prior.chunk_sizes, PANEL_INPUTS, None,
                                            box_prior.training_seasons(parts, "D", unseen, season, TEST))
            rows = box_prior.rated_rows(panel[(panel.side == "D") & (panel.season == season)], names, 1.0,
                                        box_prior.least_share(TEST))
            table = published[published.season == season].set_index("player_id")
            tasks.append((season, train, rows, inputs, params, labels.set_index("player_id"), table, float(nets.loc[season]),
                          prior_values))
            print(f"{season}: {len(train):,} training rows, {len(rows)} rated players", flush=True)
        del sums, designs
        with ThreadPoolExecutor(JOBS) as pool:
            for finished in [pool.submit(read_season, *task) for task in tasks]:
                finished.result()
    frames = pd.concat([pd.read_csv(FOLDER / f"{s}.csv") for s in SEASONS], ignore_index=True)
    frames.to_csv("outputs/csv/box_prior_trees.csv", index=False)
    pd.set_option("display.width", 240)
    print(frames.round(4).T.to_string())


if __name__ == "__main__":
    main()
