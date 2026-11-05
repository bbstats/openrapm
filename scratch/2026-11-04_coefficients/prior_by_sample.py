"""Read-only: is the LightGBM's output narrow because of small samples?  Its output and career target by the rated season's
possessions, 30 seasons, players its boosters did not see, weighted by possessions; season by season, then the mean."""
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
pd.set_option("display.width", 200)
TIERS = ((0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 1e9))
models = pd.read_pickle("outputs/prior_models_rating134.pkl")
cfg = load_config()
sums = accumulate_rapm(DesignCache(cfg), trainable(cfg).seasons, TEST)
table = pd.read_parquet("outputs/season_ratings_mid136.parquet")


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


def slope(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / np.average((x - mx) ** 2, weights=w))


rows = []
for season in sorted(k for k in models if k != "meta"):
    labels = rapm.career_labels(sums, list(models[season]["O"]["unseen"]), TEST.labels, TEST.targets)
    rated = table[table.season == season].set_index("player_id")
    for side, poss, column in (("O", "poss_off", "offense"), ("D", "poss_def", "defense")):
        prior = pd.Series(models[season][side]["prior"], dtype=float)
        label = labels[side].set_index("player_id")
        ids = rated.index.intersection(prior.index)
        for low, high in TIERS:
            keep = ids[(rated.loc[ids, poss] >= low) & (rated.loc[ids, poss] < high)]
            with_target = keep.intersection(label.index)
            if len(with_target) < 15:
                continue
            w = rated.loc[with_target, poss].to_numpy(float) + 1.0
            x, y = prior.loc[with_target].to_numpy(float), label.loc[with_target, column].to_numpy(float)
            rows.append(dict(season=season, side=side, possessions=f"{low:,.0f}+", low=low, players=len(with_target), output_sd=wsd(x, w), target_sd=wsd(y, w),
                             correlation=slope(x, y, w) * wsd(x, w) / wsd(y, w), target_per_point=slope(x, y, w)))
t = pd.DataFrame(rows)
t.to_csv("outputs/csv/prior_by_sample.csv", index=False)
print(t.groupby(["side", "low"])[["players", "output_sd", "target_sd", "correlation", "target_per_point"]].mean().round(3).to_string())
