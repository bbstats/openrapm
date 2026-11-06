"""Read-only: what the published Box Adj of 2026 is made of, by group of inputs, for the owner's top ten and across regulars.

Box Adj = the line on every input less the first stage on his own stats, both held to team totals of 0, times the prior scale
and step 4's within multiplier.  Each group's share is the same with only that group's weights."""
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
pd.set_option("display.width", 250)
SEASON = 2026
cfg, settings = load_config(), PRODUCT
prior = settings.prior
names = sp.INPUT_SETS[prior.second_prior_inputs]
type_scale = line_setup(cfg, prior.second_prior, names)["type_scale"]
saved = pd.read_pickle("outputs/prior_models_boxadj_read.pkl")[SEASON]
grams = load_grams(cfg, prior.second_prior)
first = sp.season_first_stage(grams, SEASON, set(trainable(cfg).seasons), settings.exclude_neighbours, prior.second_prior_folds,
                              prior.second_prior_penalty, prior.second_prior_nearest, prior.second_prior_inputs)
panel = pd.read_parquet("outputs/season_panel.parquet")
rows = panel[panel.season == SEASON]
answers = {side: sp.booster_answers(saved_booster(saved[side]), prior_rows(rows[rows.side == side], side, saved[side])) for side in "OD"}
box = {side: saved[side]["prior"] for side in "OD"}
types = sp.standardised_types(rows[rows.side == "O"], type_scale)
stints = season_lineup_rows(cfg, SEASON)
line = {side: np.asarray(saved[side]["line"], dtype=float) for side in "OD"}
_, inputs = sp.season_second_priors(stints, answers, box, line, prior.second_prior_folds, prior.second_prior_team_totals, types, prior.second_prior_inputs)
table = pd.read_parquet("outputs/season_ratings_product.parquet").query("season == @SEASON").set_index("player_id")
mult = pd.read_csv("outputs/csv/season_ratings_product_step4_multipliers.csv").set_index("season").loc[SEASON]

own = [names[at] for at in sp.box_score_positions(prior.second_prior_inputs)]
GROUPS = {"teammates' priors, him on": ["teammates_offence_on", "teammates_defence_on"],
          "teammates' priors, him sitting": ["teammates_offence_bench", "teammates_defence_bench"],
          "opponents' priors, him on": ["opponents_offence_on", "opponents_defence_on"],
          "opponents' priors, him sitting": ["opponents_offence_bench", "opponents_defence_bench"],
          "team rates, him on": [n for n in names if n.startswith("team_") and n.endswith("_on")],
          "team rates, him sitting": [n for n in names if n.startswith("team_") and n.endswith("_off")]}
parts, checks = {}, {}
for side, sign, poss, scale_column, within in (("O", 1.0, "poss_off", "prior_scale_off", "within_multiplier_off"), ("D", -1.0, "poss_def", "prior_scale_def", "within_multiplier_def")):
    frame = inputs[side]
    team_poss = sp.team_possessions(stints, side)
    page = sign * float(table[scale_column].iloc[0]) * float(mult[within])
    joint, alone = dict(zip(names, line[side][-1])), dict(zip(names, first[side][-1]))

    def held(weights):
        values = sum(weights[n] * frame[n] for n in weights)
        return page * sp.hold_team_totals(values, pd.Series(0.0, index=values.index), team_poss)

    side_parts = {label: held({n: joint[n] for n in members}) for label, members in GROUPS.items()}
    side_parts["his own stats, read again"] = held({n: joint[n] - alone[n] for n in own})
    for n in ("blocks_per100", "height_in", "def_rebounds_per100", "off_rebounds_per100"):
        side_parts[f"  of which {n}"] = held({n: joint[n] - alone[n]})
    for n in GROUPS["team rates, him on"] + GROUPS["team rates, him sitting"]:
        side_parts[f"  row {n}"] = held({n: joint[n]})
    total = sum(v for k, v in side_parts.items() if not k.startswith("  "))
    published = table[f"box_adj_{'off' if side == 'O' else 'def'}"].reindex(total.index)
    regulars = table[poss].reindex(total.index) >= 1000
    checks[side] = float((total - published)[regulars].abs().max())
    parts[side] = pd.DataFrame(side_parts).assign(total=total, published=published)
print("this sum against the published Box Adj, regulars, largest gap:", {k: round(v, 3) for k, v in checks.items()}, "(step 4's team layer on traded players is left out)")

top = ["Jaren Jackson Jr.", "Tony Bradley", "Donovan Clingan", "Draymond Green", "Kyshawn George", "Ryan Kalkbrenner", "Kevon Looney", "Khaman Maluach", "Drew Eubanks", "Walker Kessler"]
ids = table.index[table.player_name.isin(top)]
for side in "OD":
    shown = parts[side].reindex(ids).assign(player=table.player_name.reindex(ids)).set_index("player").reindex(top)
    print(f"\n{'offense' if side == 'O' else 'defense'}, points per 100")
    print(shown.round(2).T.to_string())

# across regulars: does Box Adj follow blocks, and which group carries its spread
reg = table.index[(table.poss_off >= 1000)]
frame = pd.DataFrame({"box_adj_total": parts["O"].total.reindex(reg) + parts["D"].total.reindex(reg)})
frame["blocks"] = rows[rows.side == "D"].set_index("player_id").reindex(reg)["blocks_per100"] if "blocks_per100" in rows.columns else np.nan
frame["height"] = types.reindex(reg)["height_in"]
w = table.poss_off.reindex(reg).to_numpy(float)


def wcorr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y, ww = x[ok], y[ok], w[ok]
    mx, my = np.average(x, weights=ww), np.average(y, weights=ww)
    return float(np.average((x - mx) * (y - my), weights=ww) / np.sqrt(np.average((x - mx) ** 2, weights=ww) * np.average((y - my) ** 2, weights=ww)))


print("\nregulars of 1,000+ possessions, correlation with total Box Adj: blocks per 100", round(wcorr(frame.box_adj_total, frame.blocks), 3), " height", round(wcorr(frame.box_adj_total, frame.height), 3))
print("spread among regulars of each group's total-side contribution, and its correlation with total Box Adj:")
for label in parts["O"].columns:
    if label in ("total", "published"):
        continue
    both = parts["O"][label].reindex(reg) + parts["D"][label].reindex(reg)
    sd = float(np.sqrt(np.average((both - np.average(both, weights=w)) ** 2, weights=w)))
    print(f"  {label}: spread {sd:.2f}, correlation {wcorr(both, frame.box_adj_total):+.2f}, with blocks {wcorr(both, frame.blocks):+.2f}")

summary = pd.DataFrame(index=ids)
threes = [f"  row team_threes_{k}_{w}" for k in ("made", "missed") for w in ("on", "off")]
summary["Box Adj"] = parts["O"].total.reindex(ids) + parts["D"].total.reindex(ids)
summary["opponents' twos made, him on"] = parts["D"]["  row team_twos_made_on"].reindex(ids)
summary["three-point rows, both ends"] = parts["O"][threes].sum(axis=1).reindex(ids) + parts["D"][threes].sum(axis=1).reindex(ids)
summary["his blocks"] = parts["O"]["  of which blocks_per100"].reindex(ids) + parts["D"]["  of which blocks_per100"].reindex(ids)
summary["everything else"] = summary["Box Adj"] - summary.iloc[:, 1:4].sum(axis=1)
summary.index = table.player_name.reindex(ids)
print(summary.reindex(top).round(2).to_string())
both = {k: parts["O"][k].reindex(reg) + parts["D"][k].reindex(reg) for k in parts["O"].columns}
three_all = sum(both[k] for k in threes)
for label, values in (("opponents' twos made, him on", parts["D"]["  row team_twos_made_on"].reindex(reg)), ("three-point rows", three_all), ("his blocks", both["  of which blocks_per100"])):
    sd = float(np.sqrt(np.average((values - np.average(values, weights=w)) ** 2, weights=w)))
    print(f"regulars: {label}: spread {sd:.2f}, correlation with total Box Adj {wcorr(values, frame.box_adj_total):+.2f}, with his blocks per 100 {wcorr(values, frame.blocks):+.2f}")
