"""Read-only: the prior's two columns regrouped, the owner's idea of 2026-11-05: everything read from a player's own stats in
the box-score column, and Box Adj only who he played with and against and how his team did.  The same weights, so no rating
moves.  Among teammates through the model's hold of team totals, regulars of 1,000+ possessions, test-setting tables."""
import sys

import numpy as np
import pandas as pd

from openrapm import second_prior as sp
from openrapm.commands.build import season_lineup_rows
from openrapm.commands.within_season import prior_rows, saved_booster
from openrapm.data.config import load_config
from openrapm.data.second_prior_files import line_setup
from openrapm.settings import PRODUCT

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
SEASONS = [2000, 2005, 2010, 2015, 2022, 2026]
cfg, settings = load_config(), PRODUCT.prior
names = sp.INPUT_SETS[settings.second_prior_inputs]
scale = line_setup(cfg, "rating134", names)["type_scale"]
table = pd.read_parquet("outputs/season_ratings_mid136.parquet")
multipliers = pd.read_csv("outputs/csv/season_ratings_mid136_step4_multipliers.csv").set_index("season")
panel = pd.read_parquet("outputs/season_panel.parquet")
models = pd.read_pickle("outputs/prior_models_rating134.pkl")
TEAM = [n for n in names if n.startswith("team_") and n != "team_possession_share"]
OTHERS = [n for n in names if n.startswith(("teammates_", "opponents_"))]
OWN_ADDED = [n for n in names if n not in TEAM + OTHERS + ["box_prior"]]
print("own stats added to the box-score column:", OWN_ADDED)


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


def wcorr(x, y, w):
    return float(np.average((x - np.average(x, weights=w)) * (y - np.average(y, weights=w)), weights=w) / (wsd(x, w) * wsd(y, w)))


rows = []
for season in SEASONS:
    saved = models[season]
    season_panel = panel[panel.season == season]
    answers = {side: sp.booster_answers(saved_booster(saved[side]), prior_rows(season_panel[season_panel.side == side], side, saved[side])) for side in "OD"}
    box = {side: saved[side]["prior"] for side in "OD"}
    types = sp.standardised_types(season_panel[season_panel.side == "O"], scale)
    stints = season_lineup_rows(cfg, season)
    line = {side: np.asarray(saved[side]["line"], dtype=float) for side in "OD"}
    _, inputs = sp.season_second_priors(stints, answers, box, line, settings.second_prior_folds, settings.second_prior_team_totals, types, settings.second_prior_inputs)
    rated = table[table.season == season].set_index("player_id")
    for side, poss, scale_column, within, sign in (("O", "poss_off", "prior_scale_off", "within_multiplier_off", 1.0), ("D", "poss_def", "prior_scale_def", "within_multiplier_def", -1.0)):
        page = float(rated[scale_column].iloc[0] * multipliers.loc[season, within])
        weights = dict(zip(names, line[side][-1]))
        frame = inputs[side]
        team_poss = sp.team_possessions(stints, side)
        keep = rated.index[rated[poss] >= 1000].intersection(frame.index)
        w = rated.loc[keep, poss].to_numpy(float)

        def held(members):
            values = sum(sign * page * weights[n] * frame[n] for n in members)
            return sp.hold_team_totals(values, pd.Series(0.0, index=values.index), team_poss).loc[keep].to_numpy(float)

        old_box, old_adj = held(["box_prior"]), held(OWN_ADDED + OTHERS + TEAM)
        new_box, new_adj = held(["box_prior"] + OWN_ADDED), held(OTHERS + TEAM)
        rows.append(dict(season=season, side=side, box_today=wsd(old_box, w), adj_today=wsd(old_adj, w), corr_today=wcorr(old_box, old_adj, w),
                         box_regrouped=wsd(new_box, w), adj_regrouped=wsd(new_adj, w), corr_regrouped=wcorr(new_box, new_adj, w),
                         whole_prior=wsd(new_box + new_adj, w)))
t = pd.DataFrame(rows)
t.to_csv("outputs/csv/regroup_columns.csv", index=False)
print(t.round(3).to_string(index=False))
print(t.groupby("side").mean(numeric_only=True).drop(columns="season").round(3).T.to_string())
