"""Read-only: the Box Adj coefficients of 2026 in the sheet's form, every step checked.

    python coefs134.py OUT.json LINE INPUT_SET MODELS TABLE
"""
import json
import os
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
OUT, LINE, INPUT_SET, MODELS, TABLE = sys.argv[1:6]
SEASON = 2026
cfg, settings = load_config(), PRODUCT.prior
names = sp.INPUT_SETS[INPUT_SET]
record, grams = line_setup(cfg, LINE, names), load_grams(cfg, LINE)
finished = set(trainable(cfg).seasons)


def line(near):
    return sp.season_coefficients(grams, SEASON, finished, near, settings.second_prior_folds,
                                  settings.second_prior_penalty, settings.second_prior_nearest)


def flipped_input(side, name):
    """Is this input in points allowed, so that a higher number means a worse defender?"""
    if name == "box_prior":
        return side == "D"
    if name == "box_prior_other_end":
        return side == "O"
    return "_defence_" in name


def good(side, name, raw):
    """The rating's points for one step of the input, in its good direction where it describes defense."""
    return raw * (1.0 if side == "O" else -1.0) * (-1.0 if flipped_input(side, name) else 1.0)


models = pd.read_pickle(f"outputs/prior_models_{MODELS}.pkl")
saved = models[SEASON]
print("check 1, the test-setting line against the saved one, largest gap:",
      max(float(np.abs(line(1)[side] - np.array(saved[side]["line"])).max()) for side in ("O", "D")))
panel = pd.read_parquet("outputs/season_panel.parquet")
rows = panel[panel.season == SEASON]
answers = {side: sp.booster_answers(saved_booster(saved[side]), prior_rows(rows[rows.side == side], side, saved[side]))
           for side in ("O", "D")}
box = {side: saved[side]["prior"] for side in ("O", "D")}
types = sp.standardised_types(rows[rows.side == "O"], record["type_scale"])
stints = season_lineup_rows(cfg, SEASON)
test_values, _ = sp.season_second_priors(stints, answers, box, line(1), settings.second_prior_folds,
                                         settings.second_prior_team_totals, types, INPUT_SET)
print("check 2, the rebuilt test-setting second prior against the saved one, largest gap:",
      {side: float((pd.Series(test_values[side]) - pd.Series(saved[side]["second_prior"])).abs().max()) for side in ("O", "D")})
coefficients = line(0)
values, inputs = sp.season_second_priors(stints, answers, box, coefficients, settings.second_prior_folds,
                                         settings.second_prior_team_totals, types, INPUT_SET)
adjust = sp.adjustments(values, inputs, coefficients, settings.second_prior_folds)
print("the line, offense, first weights:", np.round(coefficients["O"][-1][:6], 3).tolist())

table = pd.read_parquet(f"outputs/season_ratings_{TABLE}.parquet")
table = table[table.season == SEASON].set_index("player_id")
mult = pd.read_csv(f"outputs/csv/season_ratings_{TABLE}_step4_multipliers.csv").set_index("season").loc[SEASON]
page = {"O": float(table.prior_scale_off.iloc[0] * mult.within_multiplier_off),
        "D": float(table.prior_scale_def.iloc[0] * mult.within_multiplier_def)}
print("prior scale", round(float(table.prior_scale_off.iloc[0]), 4), round(float(table.prior_scale_def.iloc[0]), 4),
      "within", round(float(mult.within_multiplier_off), 4), round(float(mult.within_multiplier_def), 4),
      "page scale", {k: round(v, 4) for k, v in page.items()})
scale = record["type_scale"]
out = {"page": page, "O": [], "D": []}
for side, poss, column, sign in (("O", "poss_off", "box_adj_off", 1.0), ("D", "poss_def", "box_adj_def", -1.0)):
    frame = inputs[side]
    weight = table[poss].reindex(frame.index).fillna(0.0)
    keep = weight >= 1000
    print(side, "players with 1,000 possessions:", int(keep.sum()))
    got = pd.Series(adjust[side]).reindex(table.index)
    both = (table[poss] >= 1000) & got.notna()
    print("  check 3, the rebuilt Box Adj, sign applied, against the table's: correlation",
          round(float(np.corrcoef(sign * got[both], table.loc[both, column])[0, 1]), 4),
          " page scale x rebuilt against the table: root mean square gap",
          round(float(np.sqrt(np.mean((sign * page[side] * got[both] - table.loc[both, column]) ** 2))), 4))
    direct = sign * (frame.to_numpy() @ coefficients[side][-1])
    good_inputs = np.column_stack([frame[n].to_numpy() * (-1.0 if flipped_input(side, n) else 1.0) for n in names])
    rule = good_inputs @ np.array([good(side, n, w) for n, w in zip(names, coefficients[side][-1])])
    print("  check 4, the sign rule against the raw form, largest gap:", float(np.abs(direct - rule).max()))
    # what a row does among teammates: each player's deviation from his main team's possession-weighted mean
    team_poss = sp.team_possessions(stints, side)
    main = team_poss.sort_values("poss").groupby("player_id").tail(1).set_index("player_id")
    team_col = [c for c in team_poss.columns if c not in ("player_id", "poss")][0]
    teams = main[team_col].reindex(frame.index)
    poss_all = main.poss.reindex(frame.index).fillna(0.0)

    def inside(values):
        total = (values * poss_all).groupby(teams).transform("sum")
        return values - total / poss_all.groupby(teams).transform("sum")

    def spread(values):
        x, w = values[keep].to_numpy(float), weight[keep].to_numpy(float)
        return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))

    pieces = {}
    for name, raw in zip(names, coefficients[side][-1]):
        sd = spread(frame[name])
        sd_inside = spread(inside(frame[name]))
        step = float(scale[name][1]) if name in scale else 1.0
        g = good(side, name, float(raw))
        out[side].append({"input": name, "raw": float(raw), "good": g, "sd": sd, "sd_inside": sd_inside, "step": step,
                          "coefficient": g * page[side] / step, "impact": g * sd * page[side],
                          "impact_inside": g * sd_inside * page[side]})
        pieces[name] = sign * float(raw) * page[side] * frame[name]
    pieces = pd.DataFrame(pieces)
    box_rows = [n for n in names if n in scale]
    context_rows = [n for n in names if n not in scale and not n.startswith("team_") and n != "box_prior"]
    team_rows = [n for n in names if n.startswith("team_")]
    groups = {"own box rates, role, body": box_rows, "prior of others and other end": context_rows, "16 team rates": team_rows,
              "all of Box Adj": box_rows + context_rows + team_rows}
    print("  rating points per standard deviation among teammates, each group together:",
          {k: round(spread(inside(pieces[v].sum(axis=1))), 2) for k, v in groups.items() if v})
    if team_rows:
        for count, members in (("offensive rebounds", [n for n in team_rows if "off_rebounds" in n]),
                               ("threes", [n for n in team_rows if "threes" in n]), ("twos", [n for n in team_rows if "twos" in n]),
                               ("free throws", [n for n in team_rows if "free_throws" in n]), ("turnovers", [n for n in team_rows if "turnovers" in n]),
                               ("on floor", [n for n in team_rows if n.endswith("_on")]), ("on bench", [n for n in team_rows if n.endswith("_off")])):
            print("    team rates,", count, round(spread(inside(pieces[members].sum(axis=1))), 2))
        top = pieces[[n for n in names if n != "box_prior"]].sum(axis=1)
        one = pieces["team_off_rebounds_off"]
        print("    correlation among teammates of the bench offensive-rebound row with the rest of Box Adj:",
              round(float(np.corrcoef(inside(one)[keep], inside(top - one)[keep])[0, 1]), 3))
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("wrote", OUT)
