"""Read-only: experiment 138's ratings against mid136: spread, how far they moved, the 2026 top 20, the move against height,
and each group of Box Adj's rows against the Box score column, the form of box_adj_overlap.py, on each build's own saved line
and test-setting tables, so the two are like for like.
"""
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
pd.set_option("display.width", 220)
SEASONS = [2000, 2005, 2010, 2015, 2022, 2026]
ERAS = (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("2026", 2026, 2026))
TODAY, CANDIDATE = "mid136", "trees138_line"
cfg, settings = load_config(), PRODUCT.prior
names = sp.INPUT_SETS[settings.second_prior_inputs]


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


def wcorr(x, y, w):
    return float(np.average((x - np.average(x, weights=w)) * (y - np.average(y, weights=w)), weights=w) / (wsd(x, w) * wsd(y, w)))


today = pd.read_parquet(f"outputs/season_ratings_{TODAY}.parquet")
candidate = pd.read_parquet(f"outputs/season_ratings_{CANDIDATE}.parquet")
both = today.merge(candidate, on=["season", "player_id"], suffixes=("_today", "_new"))
panel = pd.read_parquet("outputs/season_panel.parquet")
body = panel[panel.side == "D"][["season", "player_id", "height_in", "weight_lb"]]
both = both.merge(body, on=["season", "player_id"], how="left")

rows = []
for era, low, high in ERAS:
    part = both[(both.season >= low) & (both.season <= high) & (both.poss_off_today >= 1000) & (both.poss_def_today >= 1000)]
    row = dict(era=era, regulars=len(part))
    for column, poss in (("rating_off", "poss_off_today"), ("rating_def", "poss_def_today"), ("rating_total", "poss_off_today")):
        sds_today, sds_new = [], []
        for _, g in part.groupby("season"):
            w = g[poss].to_numpy(float)
            sds_today.append(wsd(g[f"{column}_today"].to_numpy(float), w) ** 2)
            sds_new.append(wsd(g[f"{column}_new"].to_numpy(float), w) ** 2)
        row[f"{column}_sd_today"] = np.sqrt(np.mean(sds_today))
        row[f"{column}_sd_new"] = np.sqrt(np.mean(sds_new))
        move = (part[f"{column}_new"] - part[f"{column}_today"]).abs()
        row[f"{column}_move_median"] = move.median()
        row[f"{column}_move_90th"] = move.quantile(0.9)
        row[f"{column}_move_largest"] = move.max()
    rows.append(row)
print(pd.DataFrame(rows).round(3).set_index("era").T.to_string())

rows = []
for era, low, high in ERAS:
    part = both[(both.season >= low) & (both.season <= high) & (both.poss_def_today >= 1000)].dropna(subset=["height_in"])
    corr = [wcorr(g.height_in.to_numpy(float), (g.rating_def_new - g.rating_def_today).to_numpy(float), g.poss_def_today.to_numpy(float))
            for _, g in part.groupby("season")]
    rows.append(dict(era=era, defense_move_with_height=np.mean(corr)))
print(pd.DataFrame(rows).round(3).to_string(index=False))

latest = both[(both.season == 2026) & (both.poss_off_today >= 1000)].copy()
latest["rank_today"] = latest.rating_total_today.rank(ascending=False).astype(int)
latest["rank_new"] = latest.rating_total_new.rank(ascending=False).astype(int)
latest["move_total"] = latest.rating_total_new - latest.rating_total_today
latest["move_off"] = latest.rating_off_new - latest.rating_off_today
latest["move_def"] = latest.rating_def_new - latest.rating_def_today
show = ["player_name_today", "rank_today", "rank_new", "rating_total_today", "rating_total_new", "move_total", "move_off", "move_def"]
print("2026, top 20 today or new, 1,000+ possessions")
print(latest[(latest.rank_today <= 20) | (latest.rank_new <= 20)].sort_values("rank_new")[show].round(2).to_string(index=False))
print("2026, largest moves")
print(latest.reindex(latest.move_total.abs().sort_values(ascending=False).index).head(12)[show].round(2).to_string(index=False))

today_multipliers = pd.read_csv(f"outputs/csv/season_ratings_{TODAY}_step4_multipliers.csv").set_index("season")
new_multipliers = pd.read_csv(f"outputs/csv/season_ratings_{CANDIDATE}_step4_multipliers.csv").set_index("season")
for label, frame in (("today", today_multipliers), ("candidate", new_multipliers)):
    print(label, "within multipliers mean:", frame[["within_multiplier_off", "within_multiplier_def"]].mean().round(4).to_dict())
for label, name, table in (("today", TODAY, today), ("candidate", CANDIDATE, candidate)):
    group = table[table.poss_off >= 1000]
    print(label, "prior scales mean:", table.groupby("season")[["prior_scale_off", "prior_scale_def"]].first().mean().round(4).to_dict())

for label, tag, models_name, table, multipliers in (("today", "rating134", "rating134", today, today_multipliers),
                                                    ("candidate", CANDIDATE, CANDIDATE, candidate, new_multipliers)):
    record = line_setup(cfg, tag, names)
    scale = record["type_scale"]
    BOX_RATES = [n for n in names if n.endswith("_per100")]
    ROLE_BODY = [n for n in names if n in scale and n not in BOX_RATES]
    OTHERS = [n for n in names if n not in scale and not n.startswith("team_") and n != "box_prior"]
    TEAM = [n for n in names if n.startswith("team_")]
    GROUPS = {"his box rates": BOX_RATES, "his role and body": ROLE_BODY, "others' priors and his other end": OTHERS, "team rates": TEAM,
              "all of Box Adj": BOX_RATES + ROLE_BODY + OTHERS + TEAM}
    models = pd.read_pickle(f"outputs/prior_models_{models_name}.pkl")
    rows = []
    for season in SEASONS:
        saved = models[season]
        season_panel = panel[panel.season == season]
        answers = {side: sp.booster_answers(saved_booster(saved[side]), prior_rows(season_panel[season_panel.side == side], side, saved[side]))
                   for side in ("O", "D")}
        box = {side: saved[side]["prior"] for side in ("O", "D")}
        types = sp.standardised_types(season_panel[season_panel.side == "O"], scale)
        stints = season_lineup_rows(cfg, season)
        line = {side: np.asarray(saved[side]["line"], dtype=float) for side in ("O", "D")}
        _, inputs = sp.season_second_priors(stints, answers, box, line, settings.second_prior_folds, settings.second_prior_team_totals, types,
                                            settings.second_prior_inputs)
        rated = table[table.season == season].set_index("player_id")
        for side, poss, scale_column, within, sign in (("O", "poss_off", "prior_scale_off", "within_multiplier_off", 1.0),
                                                       ("D", "poss_def", "prior_scale_def", "within_multiplier_def", -1.0)):
            page = float(rated[scale_column].iloc[0] * multipliers.loc[season, within])
            weights = dict(zip(names, line[side][-1]))
            frame = inputs[side]
            team_poss = sp.team_possessions(stints, side)
            keep = rated.index[rated[poss] >= 1000].intersection(frame.index)
            w = rated.loc[keep, poss].to_numpy(float)

            def held(values):
                return sp.hold_team_totals(values, pd.Series(0.0, index=values.index), team_poss).loc[keep].to_numpy(float)

            box_part = held(sign * page * weights["box_prior"] * frame["box_prior"])
            for group, members in GROUPS.items():
                part = held(sum(sign * page * weights[n] * frame[n] for n in members))
                rows.append(dict(season=season, side=side, group=group, spread=wsd(part, w), correlation_with_box_score=wcorr(part, box_part, w)))
            rows.append(dict(season=season, side=side, group="Box score column", spread=wsd(box_part, w), correlation_with_box_score=1.0))
    del models
    frame = pd.DataFrame(rows)
    frame.to_csv(f"outputs/csv/trees138_overlap_{label}.csv", index=False)
    print(f"\nBox Adj's groups, {label}, mean over {SEASONS}")
    print(frame.groupby(["side", "group"], sort=False)[["spread", "correlation_with_box_score"]].mean().round(3).to_string())
    print(f"Box Adj's groups, {label}, 2026")
    print(frame[frame.season == 2026].set_index(["side", "group"])[["spread", "correlation_with_box_score"]].round(3).to_string())
