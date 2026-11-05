"""Read-only: the two-stage difference at the setting that reproduces check 151 best, and the LightGBM's weight times the prior
scale, which sets a team's total in the prior."""
import numpy as np
import pandas as pd

from openrapm.data.config import load_config
from openrapm.data.second_prior_files import load_grams

F, K = 2, 45
sums = load_grams(load_config(), "rating134")
years = sorted(sums)
L = np.load("outputs/csv/two_stage_lines.npz")
s = pd.read_csv("outputs/csv/two_stage_scales.csv").set_index("season")
layers = {"the sums' layer": pd.read_csv("outputs/csv/two_stage_sums_layer.csv").set_index("season"),
          "rating134's layer": pd.read_csv("outputs/csv/season_ratings_rating134_step4_multipliers.csv").set_index("season")}
check = pd.read_csv("outputs/csv/wowy_rating134_seasons.csv")
check = check[(check.system == "rating134") & (check.stage == "step4") & (check.slice == "all")].set_index("season").error
ARMS = {"today": 0, "two": 2}
G = ["within_games_multiplier_off", "within_games_multiplier_def"]
M = ["within_multiplier_off", "within_multiplier_def"]
for name, layer in layers.items():
    print(name, "mean prior part, games part:", layer[M].mean().round(3).tolist(), layer[G].mean().round(3).tolist())


def at(y, arm, layer):
    x = L[str(y)][ARMS[arm]].copy()
    x[:2] = layer.loc[y, G].to_numpy(float)
    x[F:F + K] *= s.loc[y, f"s_off_{arm}"]
    x[F + K:] *= s.loc[y, f"s_def_{arm}"]
    return x


def paired(d):
    for era, lo, hi in (("1997-2006", 0, 10), ("2007-2016", 10, 20), ("2017-2026", 20, 30), ("all 30", 0, 30)):
        x = d[lo:hi]
        se = x.std(ddof=1) / np.sqrt(len(x))
        print(f"  {era}: change {x.mean():+.3f}, standard error {se:.3f}, z {x.mean() / se:+.2f}, better {int((x < 0).sum())} of {len(x)}")


layer = layers["rating134's layer"]
gap = np.array([sums[y].error(at(y, "today", layer)) - check[y] for y in years])
print(f"own scale, rating134's games part, formula minus check 151: mean {gap.mean():+.3f}, sd {gap.std(ddof=1):.3f}, by era {gap[:10].mean():+.3f} {gap[10:20].mean():+.3f} {gap[20:].mean():+.3f}")
paired(np.array([sums[y].error(at(y, "two", layer)) - sums[y].error(at(y, "today", layer)) for y in years]))

rows = []
for y in years:
    today, two = L[str(y)][0], L[str(y)][2]
    rows.append(dict(season=y, w_off_today=today[F], w_off_two=two[F], w_def_today=today[F + K], w_def_two=two[F + K],
                     total_off_today=today[F] * s.loc[y, "s_off_today"], total_off_two=two[F] * s.loc[y, "s_off_two"],
                     total_def_today=today[F + K] * s.loc[y, "s_def_today"], total_def_two=two[F + K] * s.loc[y, "s_def_two"]))
t = pd.DataFrame(rows).set_index("season")
print(t.mean().round(3).to_dict())
print("2026", t.loc[2026].round(3).to_dict())
print("team-total factor, two over today, offense then defense:", round((t.total_off_two / t.total_off_today).mean(), 3), round((t.total_def_two / t.total_def_today).mean(), 3))
print(" by era", [(round((t.total_off_two / t.total_off_today).loc[a:b].mean(), 3), round((t.total_def_two / t.total_def_today).loc[a:b].mean(), 3)) for a, b in ((1997, 2006), (2007, 2016), (2017, 2026))])
