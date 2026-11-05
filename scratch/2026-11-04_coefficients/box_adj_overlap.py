"""Read-only: which part of Box Adj repeats the box-score prior?  Each group of Box Adj's rows against the Box score
column, among teammates through the model's own hold of team totals, regulars of a rated season, on the published line."""
import sys

import numpy as np
import pandas as pd

from openrapm import second_prior as sp
from openrapm.commands.build import season_lineup_rows
from openrapm.commands.within_season import prior_rows, saved_booster
from openrapm.data.config import load_config
from openrapm.data.seasons import trainable
from openrapm.data.second_prior_files import line_setup, load_grams
from openrapm.settings import PRODUCT

sys.stdout.reconfigure(encoding="utf-8")
SEASONS = [int(s) for s in sys.argv[1].split(",")]
cfg, settings = load_config(), PRODUCT.prior
names = sp.INPUT_SETS[settings.second_prior_inputs]
record, grams = line_setup(cfg, settings.second_prior, names), load_grams(cfg, settings.second_prior)
finished = set(trainable(cfg).seasons)
models = pd.read_pickle("outputs/prior_models_rating134.pkl")
panel = pd.read_parquet("outputs/season_panel.parquet")
published = pd.read_parquet("outputs/season_ratings_product.parquet")
multipliers = pd.read_csv("outputs/csv/season_ratings_product_step4_multipliers.csv").set_index("season")
scale = record["type_scale"]
BOX_RATES = [n for n in names if n.endswith("_per100")]
ROLE_BODY = [n for n in names if n in scale and n not in BOX_RATES]
OTHERS = [n for n in names if n not in scale and not n.startswith("team_") and n != "box_prior"]
TEAM = [n for n in names if n.startswith("team_") and n != "team_possession_share"]
GROUPS = {"his box rates": BOX_RATES, "his role and body": ROLE_BODY, "others' priors and his other end": OTHERS, "team rates": TEAM,
          "all of Box Adj": BOX_RATES + ROLE_BODY + OTHERS + TEAM}


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


def wcorr(x, y, w):
    return float(np.average((x - np.average(x, weights=w)) * (y - np.average(y, weights=w)), weights=w) / (wsd(x, w) * wsd(y, w)))


rows = []
for season in SEASONS:
    saved = models[season]
    season_panel = panel[panel.season == season]
    answers = {side: sp.booster_answers(saved_booster(saved[side]), prior_rows(season_panel[season_panel.side == side], side, saved[side]))
               for side in ("O", "D")}
    box = {side: saved[side]["prior"] for side in ("O", "D")}
    types = sp.standardised_types(season_panel[season_panel.side == "O"], scale)
    stints = season_lineup_rows(cfg, season)
    line = sp.season_coefficients(grams, season, finished, 0, settings.second_prior_folds, settings.second_prior_penalty, settings.second_prior_nearest)
    _, inputs = sp.season_second_priors(stints, answers, box, line, settings.second_prior_folds, settings.second_prior_team_totals, types,
                                        settings.second_prior_inputs)
    table = published[published.season == season].set_index("player_id")
    for side, poss, scale_column, within, sign in (("O", "poss_off", "prior_scale_off", "within_multiplier_off", 1.0),
                                                   ("D", "poss_def", "prior_scale_def", "within_multiplier_def", -1.0)):
        page = float(table[scale_column].iloc[0] * multipliers.loc[season, within])
        weights = dict(zip(names, line[side][-1]))
        frame = inputs[side]
        team_poss = sp.team_possessions(stints, side)
        keep = table.index[table[poss] >= 1000].intersection(frame.index)
        w = table.loc[keep, poss].to_numpy(float)

        def held(values):
            return sp.hold_team_totals(values, pd.Series(0.0, index=values.index), team_poss).loc[keep].to_numpy(float)

        box_part = held(sign * page * weights["box_prior"] * frame["box_prior"])
        for group, members in GROUPS.items():
            part = held(sum(sign * page * weights[n] * frame[n] for n in members))
            rows.append(dict(season=season, side=side, group=group, spread=wsd(part, w), correlation_with_box_score=wcorr(part, box_part, w)))
        rows.append(dict(season=season, side=side, group="Box score column", spread=wsd(box_part, w), correlation_with_box_score=1.0))
t = pd.DataFrame(rows)
t.to_csv("outputs/csv/box_adj_overlap.csv", index=False)
pd.set_option("display.width", 200)
print(t.groupby(["side", "group"], sort=False)[["spread", "correlation_with_box_score"]].mean().round(3).to_string())
print(t[t.season == SEASONS[-1]].set_index(["side", "group"])[["spread", "correlation_with_box_score"]].round(3).to_string())
