"""Read-only: why the formula's error at today's line misses check 151's: the prior scale, the within layer, the fold's own scale."""
import glob, sys
import numpy as np, pandas as pd
from openrapm.data.config import load_config
from openrapm.data.second_prior_files import load_grams
cfg = load_config(); sums = load_grams(cfg, "rating134"); F = 2; K = 45
L = np.load("outputs/csv/two_stage_lines.npz"); seasons = sorted(sums)
check = pd.read_csv("outputs/csv/wowy_rating134_seasons.csv")
c = check[(check.system == "rating134") & (check.stage == "step4") & (check.slice == "all")].set_index("season").error
c3 = check[(check.system == "rating134") & (check.stage == "step3") & (check.slice == "all")].set_index("season").error
print("files with a table:", sorted(glob.glob("outputs/season_ratings_rating134*.parquet")))
t3 = pd.read_parquet("outputs/season_ratings_rating134_step3.parquet")
s = t3.groupby("season")[["prior_scale_off", "prior_scale_def"]].first()
print(s.describe().loc[["mean", "min", "max"]].round(3))
print(s.loc[[1997,2005,2015,2019,2022,2026]].round(3))
m = pd.read_csv("outputs/csv/season_ratings_rating134_step4_multipliers.csv").set_index("season")
p = pd.read_csv("outputs/csv/season_ratings_padbins127_step4_multipliers.csv").set_index("season")
def report(label, f, target=c):
    d = np.array([sums[y].error(f(y)) - target[y] for y in seasons])
    print(f"{label}: mean {d.mean():+.3f}, sd {d.std(ddof=1):.3f}, largest {np.abs(d).max():.3f}, eras {d[:10].mean():+.3f} {d[10:20].mean():+.3f} {d[20:].mean():+.3f}")
def w(y, scale, mult):
    x = L[str(y)][0].copy()
    if mult is not None:
        x[:2] = mult.loc[y, ["within_games_multiplier_off", "within_games_multiplier_def"]].to_numpy(float)
    if scale:
        x[F:F+K] *= s.loc[y, "prior_scale_off"]; x[F+K:] *= s.loc[y, "prior_scale_def"]
    return x
report("scale 1, g rating134", lambda y: w(y, False, m))
report("scale 1, g padbins127", lambda y: w(y, False, p))
report("own scale, g rating134", lambda y: w(y, True, m))
report("own scale, g padbins127", lambda y: w(y, True, p))
report("own scale, fitted parts solved", lambda y: w(y, True, None))
