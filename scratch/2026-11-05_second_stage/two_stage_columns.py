"""Read-only: each arm's prior scale from step 3 solved again on each season, the test at those scales, and the prior's two
columns among teammates, today's weights against the two stages'.  Needs two_stage.py's outputs/csv/two_stage_lines.npz."""
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from openrapm import prior_rapm, second_prior as sp
from openrapm.commands.build import season_lineup_rows, step4_folds, within_layer_multipliers
from openrapm.commands.within_season import prior_rows, saved_booster
from openrapm.data.config import load_config
from openrapm.data.designs import DesignCache
from openrapm.data.second_prior_files import line_setup, load_grams
from openrapm.settings import PRODUCT, TEST

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
SIX = [2000, 2005, 2010, 2015, 2022, 2026]
F, K = 2, 45
cfg, settings = load_config(), PRODUCT.prior
names = sp.INPUT_SETS[settings.second_prior_inputs]
scale = line_setup(cfg, "rating134", names)["type_scale"]
L = np.load("outputs/csv/two_stage_lines.npz")
sums = load_grams(cfg, "rating134")
table = pd.read_parquet("outputs/season_ratings_mid136.parquet")
step3 = pd.read_parquet("outputs/season_ratings_mid136_step3.parquet")
multipliers = pd.read_csv("outputs/csv/season_ratings_mid136_step4_multipliers.csv").set_index("season")
panel = pd.read_parquet("outputs/season_panel.parquet")
models = pd.read_pickle("outputs/prior_models_rating134.pkl")
designs = DesignCache(cfg, keep=1)
ADJ = [n for n in names if n.startswith(("teammates_", "opponents_")) or (n.startswith("team_") and n != "team_possession_share")]
OWN = [n for n in names if n not in ADJ]
ARMS = {"today": 0, "two": 2}
started = time.time()

# the within layer the sums were written with: padbins127's folds at the factors of the day (experiment 134, before 136)
shrink = replace(TEST.shrink, folds_tag="padbins127_early,padbins127_within", within_factors=(0.75, 1.0), defence_factors=((0, 0.89), (2017, 0.84)))
folds, where = step4_folds(cfg, shrink)
layer = within_layer_multipliers(cfg, sorted({f.season for f in folds}), folds, where, shrink).set_index("season")
layer.to_csv("outputs/csv/two_stage_sums_layer.csv")
print(f"the sums' within layer read ({time.time() - started:.0f}s)", flush=True)


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


def wcorr(x, y, w):
    return float(np.average((x - np.average(x, weights=w)) * (y - np.average(y, weights=w)), weights=w) / (wsd(x, w) * wsd(y, w)))


scales, columns = [], []
for season in sorted(sums):
    saved = models[season]
    rows = panel[panel.season == season]
    answers = {side: sp.booster_answers(saved_booster(saved[side]), prior_rows(rows[rows.side == side], side, saved[side])) for side in "OD"}
    box = {side: saved[side]["prior"] for side in "OD"}
    types = sp.standardised_types(rows[rows.side == "O"], scale)
    stints = season_lineup_rows(cfg, season)
    off_design, def_design = designs.season(season, TEST.targets.offence), designs.season(season, TEST.targets.defence)
    arm_line, arm_values, inputs, arm_scale = {}, {}, None, {}
    for arm, row in ARMS.items():
        vector = L[str(season)][row]
        line = {"O": np.tile(vector[F:F + K], (settings.second_prior_folds + 1, 1)), "D": np.tile(vector[F + K:], (settings.second_prior_folds + 1, 1))}
        values, inputs = sp.season_second_priors(stints, answers, box, line, settings.second_prior_folds, settings.second_prior_team_totals, types, settings.second_prior_inputs)
        s_o = prior_rapm.fit(off_design, values["O"], values["D"], TEST.ridge.penalty).prior_scale[0]
        s_d = prior_rapm.fit(def_design, values["O"], values["D"], TEST.ridge.penalty).prior_scale[1]
        arm_line[arm], arm_values[arm], arm_scale[arm] = line, values, (float(s_o), float(s_d))
    published = step3[step3.season == season].iloc[0]
    scales.append(dict(season=season, s_off_table=published.prior_scale_off, s_def_table=published.prior_scale_def, s_off_today=arm_scale["today"][0], s_def_today=arm_scale["today"][1], s_off_two=arm_scale["two"][0], s_def_two=arm_scale["two"][1]))
    print(season, {k: round(v, 3) for k, v in scales[-1].items() if k != "season"}, f"({time.time() - started:.0f}s)", flush=True)
    if season not in SIX:
        continue
    rated = table[table.season == season].set_index("player_id")
    for side, poss, within, sign, end in (("O", "poss_off", "within_multiplier_off", 1.0, 0), ("D", "poss_def", "within_multiplier_def", -1.0, 1)):
        frame = inputs[side]
        team_poss = sp.team_possessions(stints, side)
        keep = rated.index[rated[poss] >= 1000].intersection(frame.index)
        w = rated.loc[keep, poss].to_numpy(float)
        m = float(multipliers.loc[season, within])

        def held(arm, members, page):
            weights = dict(zip(names, arm_line[arm][side][-1]))
            values = sum(sign * page * weights[n] * frame[n] for n in members)
            return sp.hold_team_totals(values, pd.Series(0.0, index=values.index), team_poss).loc[keep].to_numpy(float)

        today_page = arm_scale["today"][end] * m
        today_own, today_adj = held("today", OWN, today_page), held("today", ADJ, today_page)
        for arm, label, page in (("today", "today's weights", today_page), ("two", "two stages, today's scale", today_page), ("two", "two stages, own scale", arm_scale["two"][end] * m)):
            own, adj = held(arm, OWN, page), held(arm, ADJ, page)
            whole = own + adj
            columns.append(dict(season=season, side=side, arm=label, own=wsd(own, w), adjustment=wsd(adj, w), corr=wcorr(own, adj, w), whole=wsd(whole, w), corr_whole_today=wcorr(whole, today_own + today_adj, w),
                                move_sd=wsd(whole - today_own - today_adj, w), corr_own_today=wcorr(own, today_own, w)))
s = pd.DataFrame(scales)
s.to_csv("outputs/csv/two_stage_scales.csv", index=False)
c = pd.DataFrame(columns)
c.to_csv("outputs/csv/two_stage_columns.csv", index=False)
print("regrouped at today's weights against outputs/csv/regroup_columns.csv:")
old = pd.read_csv("outputs/csv/regroup_columns.csv")
today = c[c.arm == "today's weights"].merge(old, on=["season", "side"])
print({k: float((today[a] - today[b]).abs().max()) for k, a, b in (("own", "own", "box_regrouped"), ("adjustment", "adjustment", "adj_regrouped"), ("corr", "corr", "corr_regrouped"), ("whole", "whole", "whole_prior"))})
print("table scale against step 3 solved here, largest gap", float(max((s.s_off_table - s.s_off_today).abs().max(), (s.s_def_table - s.s_def_today).abs().max())))
print(c.round(3).to_string(index=False))
print(c.groupby(["side", "arm"], sort=False).mean(numeric_only=True).drop(columns="season").round(3).to_string())


# the test with each arm at its own scale and the sums' within layer on the fitted parts
def paired(d, label):
    print("\n" + label)
    for era, lo, hi in (("1997-2006", 0, 10), ("2007-2016", 10, 20), ("2017-2026", 20, 30), ("all 30", 0, 30)):
        x = d[lo:hi]
        print(f"  {era}: change {x.mean():+.3f}, standard error {x.std(ddof=1) / np.sqrt(len(x)):.3f}, z {x.mean() / (x.std(ddof=1) / np.sqrt(len(x))):+.2f}, better {int((x < 0).sum())} of {len(x)}")


years = sorted(sums)
s = s.set_index("season")
check = pd.read_csv("outputs/csv/wowy_rating134_seasons.csv")
check = check[(check.system == "rating134") & (check.stage == "step4") & (check.slice == "all")].set_index("season").error


def at(season, arm, scaled=True):
    x = L[str(season)][ARMS[arm]].copy()
    x[:2] = layer.loc[season, ["within_games_multiplier_off", "within_games_multiplier_def"]].to_numpy(float)
    if scaled:
        x[F:F + K] *= s.loc[season, f"s_off_{arm}"]
        x[F + K:] *= s.loc[season, f"s_def_{arm}"]
    return x


for scaled, words in ((False, "scale 1"), (True, "each arm's own scale")):
    gap = np.array([sums[y].error(at(y, "today", scaled)) - check[y] for y in years])
    print(f"\ntoday's line, the sums' within layer, {words}, formula minus check 151: mean {gap.mean():+.3f}, standard deviation {gap.std(ddof=1):.3f}, largest {np.abs(gap).max():.3f}, by era {gap[:10].mean():+.3f} {gap[10:20].mean():+.3f} {gap[20:].mean():+.3f}")
    paired(np.array([sums[y].error(at(y, "two", scaled)) - sums[y].error(at(y, "today", scaled)) for y in years]), f"two stages minus today's, the sums' within layer, {words}")
