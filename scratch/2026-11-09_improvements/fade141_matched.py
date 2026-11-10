"""Read-only: experiment 141 at matched width (the measurement traps: a year-over-year gain or loss can be width).

The candidate's ratings with each side's spread, season by season, multiplied to equal the incumbent's among regulars
of 1,000 or more possessions on that side, written as outputs/season_ratings_matched141.parquet for the year-over-year
test against the incumbent's ratings.  Prints the mean multiplier by era and side.

    .venv/Scripts/python scratch/2026-11-09_improvements/fade141_matched.py
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
INCUMBENT, CANDIDATE, OUT = "zones139", "fade141", "matched141"


def spread(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.average((values - np.average(values, weights=weights)) ** 2, weights=weights)))


def main() -> None:
    today = pd.read_parquet(f"outputs/season_ratings_{INCUMBENT}.parquet")
    candidate = pd.read_parquet(f"outputs/season_ratings_{CANDIDATE}.parquet")
    parts, ratios = [], []
    for season, group in candidate.groupby("season"):
        group = group.copy()
        base = today[today.season == season].set_index("player_id")
        for column, poss in (("rating_off", "poss_off"), ("rating_def", "poss_def")):
            regulars = group[group[poss] >= 1000].set_index("player_id")
            keep = regulars.index.intersection(base.index)
            weights = regulars.loc[keep, poss].to_numpy(float)
            ratio = (spread(base.loc[keep, column].to_numpy(float), weights)
                     / spread(regulars.loc[keep, column].to_numpy(float), weights))
            group[column] = group[column] * ratio
            ratios.append(dict(season=season, column=column, ratio=ratio))
        group["rating_total"] = group.rating_off + group.rating_def
        parts.append(group)
    pd.concat(parts, ignore_index=True).to_parquet(f"outputs/season_ratings_{OUT}.parquet", index=False)
    ratios = pd.DataFrame(ratios)
    eras = pd.cut(ratios.season, [1996, 2006, 2016, 2026], labels=["1997-2006", "2007-2016", "2017-2026"])
    print("the multiplier that matches the candidate's spread to the incumbent's, mean by era")
    print(ratios.assign(era=eras).groupby(["era", "column"], observed=True).ratio.mean().round(4).to_string())
    print(f"all seasons: offense {ratios[ratios.column == 'rating_off'].ratio.mean():.4f}, "
          f"defense {ratios[ratios.column == 'rating_def'].ratio.mean():.4f}")


if __name__ == "__main__":
    main()
