"""A standard error for every published rating, from the ratings rebuilt with part of the season's games held back.

The question: had the season's games come out as another draw of the same schedule, how far would a player's published
rating typically land from the one shown?  The answer is the delete-a-group jackknife over games:

    se^2 = (g - 1) / g  x  sum over the g folds of one repeat of (rating_fold - mean rating over those folds)^2

with g the folds a repeat (10 in the tenth-held-back sets, 4 in the quarter-held-back ones), each fold's ratings the
season rated without that fold's games, every input rebuilt from the games kept, then shrunk with the season's
published step-4 multipliers (prior_shrink.shrink_fold).  The three repeats' se^2 are averaged.

What it holds: the noise of the games in everything a rating reads from them, box score, Box Adj and the ridge.
What it leaves out: how far the box-score prior itself can miss a player whatever games are played.

Example: a player whose ten folds rate him 5.0, 5.4, 4.6, 5.2, 4.8, 5.0, 5.3, 4.7, 5.1 and 4.9 has a sum of squares
of 0.60, so se = sqrt(0.9 x 0.60) = 0.73.

    .venv/Scripts/python scratch/2026-11-26_rating_se/jackknife.py 1997 2026
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm import prior_shrink
from openrapm.data import within_folds

OUT = Path("outputs/rating_se")
MULTIPLIERS = Path("outputs/csv/season_ratings_product_step4_multipliers.csv")
SETS = {"tenth": ("rd143_k10",), "quarter": ("rd143_early", "rd143_within")}
# the three ratings, then each side's two parts, the prior's and the games', to see where the noise comes from
SIDES = ("rating_off", "rating_def", "rating_total", "prior_off", "u_off", "prior_def", "u_def")


def season_folds(tags: tuple, season: int) -> list:
    """The season's folds from whichever of `tags` holds them."""
    for tag in tags:
        directory = Path("outputs/within") / tag
        if list(directory.glob(f"fold_{season}_*.json")):
            return within_folds.load_folds(directory, (season, season))
    raise ValueError(f"no folds for {season} in {tags}")


def jackknife(folds: list, season: int, factors: tuple) -> pd.DataFrame:
    """One row per player: se_off, se_def, se_total, the same for each side's parts (se_prior_off, se_u_off, se_prior_def,
    se_u_def), and folds, the number of folds that rated him."""
    tables = []
    for fold in folds:
        table = prior_shrink.shrink_fold(fold.rows, season, factors)[["player_id", *SIDES]]
        tables.append(table.assign(repeat=fold.repeat, fold=fold.fold))
    ratings = pd.concat(tables, ignore_index=True)
    per_repeat = ratings.groupby("repeat").fold.nunique()
    groups = int(per_repeat.iloc[0])
    grouped = ratings.groupby(["player_id", "repeat"])
    count = grouped.fold.count()
    # a player missing from a fold, all his games held back, has no rating there: that repeat is left out for him
    sums = grouped[list(SIDES)].agg(lambda values: ((values - values.mean()) ** 2).sum())[count == groups]
    variance = (groups - 1) / groups * sums.groupby("player_id").mean()
    out = np.sqrt(variance).rename(columns={side: "se_" + side.replace("rating_", "") for side in SIDES})
    out["folds"] = ratings.groupby("player_id").fold.count()
    return out.reset_index().assign(season=season)


def main() -> None:
    first, last = int(sys.argv[1]), int(sys.argv[2])
    multipliers = pd.read_csv(MULTIPLIERS)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, tags in SETS.items():
        seasons = []
        for season in range(first, last + 1):
            factors = prior_shrink.season_factors(multipliers, season)
            seasons.append(jackknife(season_folds(tags, season), season, factors))
            print(name, season, len(seasons[-1]), flush=True)
        pd.concat(seasons, ignore_index=True).to_parquet(OUT / f"jackknife_{name}_{first}_{last}.parquet")


if __name__ == "__main__":
    main()
