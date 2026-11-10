"""Read-only: experiment 140's third read (notes/DECISIONS.md, "Experiment 140, a third read").

Box Adj's weights for a rated season are solved on every season it may learn from, each counted the same.  Here a
training season d seasons away from the rated one is counted 0.5 ** (d / H), H the half-life in seasons, so the far
seasons stay in and weigh less.  Everything else is experiment 140's read: the saved sums of the shipped line, the
training seasons more than one away, the shipped penalty, the rated season's own held-back stints as the score, paired
by season against the shipped setting, by era.

Writes outputs/csv/box_adj_fade140_seasons.csv and outputs/csv/box_adj_fade140.csv and prints both tables.
Run: .venv/Scripts/python scratch/2026-11-09_improvements/box_adj_fade.py
"""
import sys

import numpy as np
import pandas as pd

from openrapm import rating_line, swap_adjust
from openrapm.data.config import load_config
from openrapm.data.seasons import trainable
from openrapm.data.second_prior_files import load_grams
from openrapm.settings import PRODUCT

sys.stdout.reconfigure(encoding="utf-8")
TAG = "zones139_line"
HALF_LIVES = (4, 6, 8, 12, 16, 24)
ERAS = (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("all 30", 1997, 2026))
SHIPPED = "no fade"


def faded(sums: dict, train: list, season: int, half_life: float | None) -> rating_line.RatingSums:
    """The training seasons' sums added, each times 0.5 ** (its distance from `season` / `half_life`); every season
    counted 1 without a half-life."""
    total = None
    for other in train:
        weight = 1.0 if half_life is None else 0.5 ** (abs(other - season) / float(half_life))
        part = sums[other]
        part = rating_line.RatingSums(weight * part.Q, weight * part.c, weight * part.yy, weight * part.sw)
        total = part if total is None else total + part
    return total


def seasons_worth(train: list, season: int, half_life: float) -> float:
    """How many equally counted seasons the faded ones are worth: (sum of weights) squared over the sum of squares."""
    weights = np.array([0.5 ** (abs(other - season) / float(half_life)) for other in train])
    return float(weights.sum() ** 2 / (weights ** 2).sum())


def main() -> None:
    cfg, prior = load_config(), PRODUCT.prior
    sums = load_grams(cfg, TAG)
    seasons, finished = sorted(sums), set(trainable(cfg).seasons)
    columns = np.arange(sums[seasons[0]].Q.shape[0])
    rows = []
    for season in seasons:
        train = swap_adjust.training_seasons(season, seasons, finished, 1)
        record = {"season": season, "training_seasons": len(train),
                  SHIPPED: sums[season].error(rating_line._solve(faded(sums, train, season, None), columns,
                                                                 prior.second_prior_penalty))}
        for half_life in HALF_LIVES:
            weights = rating_line._solve(faded(sums, train, season, half_life), columns, prior.second_prior_penalty)
            record[f"half-life {half_life}"] = sums[season].error(weights)
            record[f"seasons' worth at {half_life}"] = seasons_worth(train, season, half_life)
        rows.append(record)
    table = pd.DataFrame(rows)

    # the code must give back experiment 140's shipped column before anything else is read
    before = pd.read_csv("outputs/csv/box_adj_weights140_seasons.csv").set_index("season")
    shipped_column = f"penalty {prior.second_prior_penalty:g}"
    gap = float((table.set_index("season")[SHIPPED] - before[shipped_column]).abs().max())
    print(f"{TAG}: {len(seasons)} seasons; with no fade, the largest gap from experiment 140's shipped column "
          f"{gap:.1e}")
    assert gap < 1e-9, gap
    table.to_csv("outputs/csv/box_adj_fade140_seasons.csv", index=False)

    summary = []
    for half_life in HALF_LIVES:
        column = f"half-life {half_life}"
        worth = table[f"seasons' worth at {half_life}"]
        for era, first, last in ERAS:
            inside = (table.season >= first) & (table.season <= last)
            change = (table[column] - table[SHIPPED])[inside].to_numpy()
            error = change.std(ddof=1) / np.sqrt(len(change))
            summary.append(dict(setting=column, era=era, change=round(float(change.mean()), 4),
                                standard_error=round(float(error), 4),
                                paired_z=round(float(change.mean() / error), 2) if error > 0 else 0.0,
                                seasons_better=int((change < 0).sum()), seasons=len(change),
                                seasons_worth=round(float(worth[inside].mean()), 1)))
    summary = pd.DataFrame(summary)
    summary.to_csv("outputs/csv/box_adj_fade140.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\nthe shipped setting, every training season counted the same: mean error {table[SHIPPED].mean():.4f}, "
          f"{table.training_seasons.mean():.1f} training seasons a rated season")
    print("each half-life minus the shipped setting, season by season; below 0 is better")
    print(summary.to_string(index=False))
    by_season = table[["season", SHIPPED]].copy()
    for half_life in HALF_LIVES:
        by_season[f"H{half_life}"] = (table[f"half-life {half_life}"] - table[SHIPPED]).round(3)
    print("\nseason by season, each half-life minus the shipped setting")
    print(by_season.drop(columns=SHIPPED).to_string(index=False))


if __name__ == "__main__":
    main()
