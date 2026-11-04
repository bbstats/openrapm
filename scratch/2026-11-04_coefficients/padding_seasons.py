"""Read-only: over every season, how much of each team rate survives its padding for a regular, and what the two rows that
pad flat in 2026 do in the seasons they do not."""
import sys

import numpy as np
import pandas as pd

from openrapm import second_prior as sp
from openrapm.commands.build import season_lineup_rows
from openrapm.data.config import load_config
from openrapm.data.seasons import trainable
from openrapm.data.second_prior_files import load_grams
from openrapm.settings import PRODUCT

sys.stdout.reconfigure(encoding="utf-8")
cfg, settings = load_config(), PRODUCT.prior
grams = load_grams(cfg, settings.second_prior)
finished = set(trainable(cfg).seasons)
names = sp.INPUT_SETS[settings.second_prior_inputs]
full = pd.read_parquet("outputs/season_ratings_product.parquet")
mult = pd.read_csv("outputs/csv/season_ratings_product_step4_multipliers.csv").set_index("season")
rows = []
for season in range(1997, 2027):
    stints = season_lineup_rows(cfg, season)
    padded = sp.team_rate_numbers(stints, pad=True)
    raw = sp.team_rate_numbers(stints, pad=False)
    line = sp.season_coefficients(grams, season, finished, 0, settings.second_prior_folds, settings.second_prior_penalty, settings.second_prior_nearest)
    table = full[full.season == season].set_index("player_id")
    for side, poss, scale, within, sign in (("O", "poss_off", "prior_scale_off", "within_multiplier_off", 1.0), ("D", "poss_def", "prior_scale_def", "within_multiplier_def", -1.0)):
        page = float(table[scale].iloc[0] * mult.loc[season, within])
        weights = dict(zip(names, line[side][-1]))
        team_poss = sp.team_possessions(stints, side)
        keep = table.index[table[poss] >= 1000].intersection(padded[side].index)
        w = table.loc[keep, poss].to_numpy(float)

        def sd(series):
            held = sp.hold_team_totals(series, pd.Series(0.0, index=series.index), team_poss).loc[keep].to_numpy(float)
            return float(np.sqrt(np.average((held - np.average(held, weights=w)) ** 2, weights=w)))

        for name in sp.TEAM_RATES:
            sd_padded, sd_raw = sd(padded[side][name]), sd(raw[side][name])
            rows.append(dict(season=season, side=side, input=name, sd_padded=sd_padded, sd_raw=sd_raw, coefficient=sign * page * weights[name],
                             impact=sign * page * weights[name] * sd_padded))
    print(season, end=" ", flush=True)
print()
t = pd.DataFrame(rows)
t["kept"] = t.sd_padded / t.sd_raw
t["flat"] = t.sd_padded < 1e-3
pd.set_option("display.width", 220)
summary = t.groupby(["side", "input"]).agg(sd_raw=("sd_raw", "mean"), sd_padded=("sd_padded", "mean"), kept=("kept", "mean"), seasons_flat=("flat", "sum"), coefficient=("coefficient", "mean"), impact_abs=("impact", lambda x: x.abs().mean())).round(3)
print(summary.to_string())
for key in (("D", "team_off_rebounds_off"), ("D", "team_twos_missed_off")):
    s = t[(t.side == key[0]) & (t.input == key[1])].set_index("season")
    print(key, "by season, spread after padding:", s.sd_padded.round(3).to_dict())
print("share of the sum of squares of all 16 rows by side, by era: not computed")
t.to_csv("outputs/csv/box_adj_team_rate_padding.csv", index=False)
