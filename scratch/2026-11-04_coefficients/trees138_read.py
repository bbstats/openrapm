"""Read-only: experiment 138's registered check and its reported columns (notes/DECISIONS.md).

First the check: the saved defensive answers of the candidate's and the first build's boosters against the read-only look's
330-tree answers.  Then, for the candidate against rating134, Box Adj's weight on the box-score prior and the target per
output point, by era and side, over 30 seasons; offense's output must not have moved.
"""
import sys

import numpy as np
import pandas as pd

from openrapm import rapm
from openrapm.commands.build import accumulate_rapm
from openrapm.data.config import load_config
from openrapm.data.designs import DesignCache
from openrapm.data.seasons import trainable
from openrapm.settings import TEST

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
LOOK = [2000, 2005, 2010, 2015, 2022, 2026]
ERAS = (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("all", 1997, 2026))


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


def slope(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / np.average((x - mx) ** 2, weights=w))


def extract(name):
    """{season: {side: (prior, line weight on the box-score prior, unseen)}} from a saved booster file."""
    models = pd.read_pickle(f"outputs/prior_models_{name}.pkl")
    out = {}
    for season, saved in models.items():
        if season == "meta":
            continue
        out[season] = {side: (pd.Series(saved[side]["prior"], dtype=float), float(np.asarray(saved[side]["line"])[-1][0]),
                              list(saved[side]["unseen"])) for side in ("O", "D")}
    trees = {side: models[2026][side]["params"]["n_estimators"] for side in ("O", "D")}
    del models
    return out, trees


today, today_trees = extract("rating134")
first, first_trees = extract("trees138")
candidate, candidate_trees = extract("trees138_line")
print("trees, rating134:", today_trees, "trees138:", first_trees, "trees138_line:", candidate_trees)

rows = []
for season in LOOK:
    look = pd.read_csv(f"outputs/box_prior_trees/players_{season}.csv").set_index("player_id").trees_330
    for label, build in (("trees138", first), ("trees138_line", candidate)):
        saved = build[season]["D"][0]
        gap = (saved.reindex(look.index) - look).abs()
        rows.append(dict(season=season, build=label, players=len(look), missing=int(gap.isna().sum()), largest_gap=float(gap.max())))
print("the registered check: saved defensive answers against the look's 330-tree answers")
print(pd.DataFrame(rows).to_string(index=False))

print("offense's output, candidate against rating134, largest gap over 30 seasons:",
      max(float((candidate[s]["O"][0] - today[s]["O"][0].reindex(candidate[s]["O"][0].index)).abs().max()) for s in today))

cfg = load_config()
designs = DesignCache(cfg)
sums = accumulate_rapm(designs, trainable(cfg).seasons, TEST)
published = pd.read_parquet("outputs/season_ratings_mid136.parquet")
reads = []
for season in sorted(today):
    table = published[published.season == season].set_index("player_id")
    labels = rapm.career_labels(sums, today[season]["O"][2], TEST.labels, TEST.targets)
    for side, poss, column in (("O", "poss_off", "offense"), ("D", "poss_def", "defense")):
        label = labels[side].set_index("player_id")
        row = dict(season=season, side=side, weight_today=today[season][side][1], weight_candidate=candidate[season][side][1])
        for name, build in (("today", today), ("candidate", candidate)):
            prior = build[season][side][0]
            keep = table.index[table[poss] >= 1000].intersection(prior.index).intersection(label.index)
            w = table.loc[keep, poss].to_numpy(float)
            x, y = prior.loc[keep].to_numpy(float), label.loc[keep, column].to_numpy(float)
            row[f"output_sd_{name}"] = wsd(x, w)
            row[f"target_per_point_{name}"] = slope(x, y, w)
        reads.append(row)
reads = pd.DataFrame(reads)
reads["weight_ratio"] = reads.weight_today / reads.weight_candidate
reads["spread_ratio"] = reads.output_sd_candidate / reads.output_sd_today
reads["weight_x_sd_today"] = reads.weight_today * reads.output_sd_today
reads["weight_x_sd_candidate"] = reads.weight_candidate * reads.output_sd_candidate
reads.to_csv("outputs/csv/trees138_reads.csv", index=False)
columns = ["weight_today", "weight_candidate", "weight_ratio", "output_sd_today", "output_sd_candidate", "spread_ratio",
           "weight_x_sd_today", "weight_x_sd_candidate", "target_per_point_today", "target_per_point_candidate"]
for era, low, high in ERAS:
    part = reads[(reads.season >= low) & (reads.season <= high)]
    print(era)
    print(part.groupby("side")[columns].mean().round(3).T.to_string())
