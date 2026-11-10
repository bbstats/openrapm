"""Read-only: experiment 140's two reads from outputs/second_prior/zones139_line/sums_<season>.npz (notes/DECISIONS.md,
"Experiment 140").

Box Adj's weights for a rated season are solved on the seasons it may learn from, every column scaled to one standard
deviation, with a ridge penalty of 0.0001 x the mean diagonal (rating_line._solve), close to none.  Two settings have
never been read on the within-season test:

    the penalty                   0.0001 as shipped, 0.001, 0.01, 0.03, 0.1, 0.3, 1; on every column as coded, and
                                  again with each end's fitted part left free, since those two columns stand where
                                  step 4's own multipliers stand
    the seasons the weights use   all of them as shipped, or the 8, 12 or 16 nearest the rated season

Each rated season is scored on its own held-back stints (RatingSums.error) with weights from the seasons more than one
away from it, the test setting's rule, and paired by season against the shipped setting, by era.

Writes outputs/csv/box_adj_weights140_seasons.csv and outputs/csv/box_adj_weights140.csv and prints both tables.
Run: .venv/Scripts/python scratch/2026-11-09_improvements/box_adj_weights.py
"""
import sys

import numpy as np
import pandas as pd

from openrapm import rating_line, second_prior as sp, swap_adjust
from openrapm.data.config import load_config
from openrapm.data.seasons import trainable
from openrapm.data.second_prior_files import load_grams
from openrapm.settings import PRODUCT

sys.stdout.reconfigure(encoding="utf-8")
F = rating_line.FITTED_PARTS
TAG = "zones139_line"
PENALTIES = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
NEAREST = (8, 12, 16)
ERAS = (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("all 30", 1997, 2026))


def solve(total, penalty: float, free_fitted_parts: bool = False) -> np.ndarray:
    """rating_line._solve on every column of `total`; with `free_fitted_parts`, no penalty on the first two columns."""
    mean_square = np.diag(total.Q) / total.sw
    has_spread = mean_square > rating_line.NO_SPREAD
    scale = np.where(has_spread, 1.0 / np.sqrt(np.where(has_spread, mean_square, 1.0)), 0.0)
    gram = total.Q * np.outer(scale, scale)
    ridge = float(penalty) * float(np.mean(np.diag(gram))) * np.ones(len(scale))
    if free_fitted_parts:
        ridge[:F] = 0.0
    return np.linalg.solve(gram + np.diag(ridge), total.c * scale) * scale


def added(sums: dict, seasons: list):
    return sum([sums[season] for season in seasons[1:]], sums[seasons[0]])


def nearest_seasons(train: list, season: int, count: int) -> list:
    """second_prior._line_seasons' pick: the `count` nearest, the nearer first and on an equal distance the earlier."""
    return sorted(sorted(train, key=lambda other: (abs(other - season), other))[:count])


def main() -> None:
    cfg, prior = load_config(), PRODUCT.prior
    sums = load_grams(cfg, TAG)
    seasons, finished = sorted(sums), set(trainable(cfg).seasons)
    names = sp.INPUT_SETS[prior.second_prior_inputs]
    columns = sums[seasons[0]].Q.shape[0]
    assert columns == F + 2 * len(names), (columns, len(names))
    print(f"{TAG}: {len(seasons)} seasons, {len(names)} inputs an end, {columns} columns")

    # the code must give back the shipped weights before anything else is read: this solve against rating_line's own,
    # and the line of the season in progress, which trains on every finished season, against the morning's log
    everything = added(sums, seasons)
    shipped = rating_line._solve(everything, np.arange(columns), prior.second_prior_penalty)
    gap = float(np.abs(solve(everything, prior.second_prior_penalty) - shipped).max())
    print(f"this solve against rating_line._solve on every season: largest gap {gap:.1e}")
    assert gap == 0.0, gap
    line = dict(zip(names, shipped[F:F + len(names)]))
    print("offense's weights as the build of 2027 logs them: "
          + ", ".join(f"{name} {line[name]:+.3f}" for name in names[:7]))

    rows = []
    for season in seasons:
        train = swap_adjust.training_seasons(season, seasons, finished, 1)
        total = added(sums, train)
        record = {"season": season, "training_seasons": len(train)}
        for penalty in PENALTIES:
            record[f"penalty {penalty:g}"] = sums[season].error(solve(total, penalty))
            record[f"penalty {penalty:g}, fitted parts free"] = sums[season].error(solve(total, penalty, True))
        for count in NEAREST:
            near = added(sums, nearest_seasons(train, season, count))
            record[f"nearest {count}"] = sums[season].error(solve(near, prior.second_prior_penalty))
        rows.append(record)
    table = pd.DataFrame(rows)
    today = f"penalty {prior.second_prior_penalty:g}"
    table.to_csv("outputs/csv/box_adj_weights140_seasons.csv", index=False)

    summary = []
    for column in table.columns[2:]:
        if column == today:
            continue
        for era, first, last in ERAS:
            change = (table[column] - table[today])[(table.season >= first) & (table.season <= last)].to_numpy()
            error = change.std(ddof=1) / np.sqrt(len(change))
            summary.append(dict(setting=column, era=era, change=round(float(change.mean()), 4),
                                standard_error=round(float(error), 4),
                                paired_z=round(float(change.mean() / error), 2) if error > 0 else 0.0,
                                seasons_better=int((change < 0).sum()), seasons=len(change)))
    summary = pd.DataFrame(summary)
    summary.to_csv("outputs/csv/box_adj_weights140.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\nthe shipped setting, {today}, every season its training seasons: mean error {table[today].mean():.4f}")
    print("each setting minus the shipped one, season by season; below 0 is better")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
