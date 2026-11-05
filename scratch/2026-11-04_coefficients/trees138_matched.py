"""Read-only: experiment 138 at matched width (trap 39).  The candidate's ratings with each side's spread, season by season,
multiplied to equal today's among regulars of 1,000+ possessions, for the year-over-year test against today's ratings.
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
today = pd.read_parquet("outputs/season_ratings_mid136.parquet")
candidate = pd.read_parquet("outputs/season_ratings_trees138_line.parquet")


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


parts, ratios = [], []
for season, group in candidate.groupby("season"):
    group = group.copy()
    base = today[today.season == season].set_index("player_id")
    for column, poss in (("rating_off", "poss_off"), ("rating_def", "poss_def")):
        regulars = group[group[poss] >= 1000].set_index("player_id")
        keep = regulars.index.intersection(base.index)
        w = regulars.loc[keep, poss].to_numpy(float)
        ratio = wsd(base.loc[keep, column].to_numpy(float), w) / wsd(regulars.loc[keep, column].to_numpy(float), w)
        group[column] = group[column] * ratio
        ratios.append(dict(season=season, column=column, ratio=ratio))
    group["rating_total"] = group.rating_off + group.rating_def
    parts.append(group)
pd.concat(parts, ignore_index=True).to_parquet("outputs/season_ratings_matched138.parquet", index=False)
ratios = pd.DataFrame(ratios)
print(ratios.assign(era=pd.cut(ratios.season, [1996, 2006, 2016, 2026], labels=["1997-2006", "2007-2016", "2017-2026"]))
      .groupby(["era", "column"], observed=True).ratio.mean().round(4).to_string())
