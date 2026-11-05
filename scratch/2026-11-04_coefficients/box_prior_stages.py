"""Read-only: is the box-score prior narrow because of the LightGBM or because of what comes after it?

For regulars of a rated season: the career RAPM the LightGBM trains on, from seasons its boosters did not see for that
season, against the LightGBM's output; then the output's spread through Box Adj's weight, the prior scale and the shrink.
"""
import sys

import numpy as np
import pandas as pd

from openrapm import rapm, second_prior as sp
from openrapm.commands.build import accumulate_rapm
from openrapm.data.config import load_config
from openrapm.data.designs import DesignCache
from openrapm.data.seasons import trainable
from openrapm.data.second_prior_files import load_grams
from openrapm.settings import PRODUCT, TEST

sys.stdout.reconfigure(encoding="utf-8")
cfg = load_config()
SEASONS = [int(s) for s in sys.argv[1].split(",")]
models = pd.read_pickle("outputs/prior_models_rating134.pkl")
designs = DesignCache(cfg)
sums = accumulate_rapm(designs, trainable(cfg).seasons, TEST)
grams = load_grams(cfg, "rating134")
finished = set(trainable(cfg).seasons)
published = pd.read_parquet("outputs/season_ratings_product.parquet")
multipliers = pd.read_csv("outputs/csv/season_ratings_product_step4_multipliers.csv").set_index("season")


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


def slope(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / np.average((x - mx) ** 2, weights=w))


rows = []
for season in SEASONS:
    saved = models[season]
    unseen = set(saved["O"]["unseen"])
    labels = rapm.career_labels(sums, unseen, TEST.labels, TEST.targets)
    table = published[published.season == season].set_index("player_id")
    line = sp.season_coefficients(grams, season, finished, 0, PRODUCT.prior.second_prior_folds,
                                  PRODUCT.prior.second_prior_penalty, PRODUCT.prior.second_prior_nearest)
    for side, poss, column, scale, within, sign in (("O", "poss_off", "offense", "prior_scale_off", "within_multiplier_off", 1.0),
                                                    ("D", "poss_def", "defense", "prior_scale_def", "within_multiplier_def", -1.0)):
        prior = pd.Series(saved[side]["prior"], dtype=float)
        label = labels[side].set_index("player_id")
        keep = table.index[table[poss] >= 1000].intersection(prior.index)
        w = table.loc[keep, poss].to_numpy(float)
        output = prior.loc[keep].to_numpy(float)
        both = keep.intersection(label.index)
        wb = table.loc[both, poss].to_numpy(float)
        x, y = prior.loc[both].to_numpy(float), label.loc[both, column].to_numpy(float)
        weight = float(line[side][-1][0])
        prior_scale = float(table[scale].iloc[0])
        shrink = float(multipliers.loc[season, within])
        rows.append(dict(season=season, side=side, players_with_label=len(both), label_sd=wsd(y, wb), output_sd_same_players=wsd(x, wb),
                         label_on_output_slope=slope(x, y, wb), correlation=float(np.corrcoef(x, y)[0, 1]), output_sd=wsd(output, w),
                         line_weight=weight, after_weight=weight * wsd(output, w), prior_scale=prior_scale,
                         after_scale=weight * prior_scale * wsd(output, w), shrink=shrink,
                         box_column=weight * prior_scale * shrink * wsd(output, w), net=weight * prior_scale * shrink,
                         median_label_possessions=float(label.loc[both, "possessions"].median())))
pd.set_option("display.width", 220)
t = pd.DataFrame(rows).set_index(["season", "side"]).round(3)
print(t.T.to_string())
