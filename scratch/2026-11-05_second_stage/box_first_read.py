"""Read-only: 2026 through the real code with the split on, against the published table and the preview, then the columns and the
2026 top 20 before and after."""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
pub = pd.read_parquet("outputs/season_ratings_product_step3.parquet").query("season == 2026").set_index("player_id")
real = pd.read_parquet("outputs/season_ratings_boxfirst_check_step3.parquet").query("season == 2026").set_index("player_id")
side = pd.read_parquet("outputs/season_ratings_product_boxfirst_step3.parquet").query("season == 2026").set_index("player_id")
print("players", len(pub), len(real), "same ids", set(pub.index) == set(real.index))
real = real.reindex(pub.index)
side = side.reindex(pub.index)
for column in ("rating_off", "rating_def", "rating_total", "prior_off", "prior_def", "u_off", "u_def"):
    print(f"  {column}: real code with the split on minus published, largest {float((real[column] - pub[column]).abs().max()):.2e}")
for column in ("box_adj_off", "box_adj_def"):
    print(f"  {column}: real code minus preview, largest {float((real[column] - side[column]).abs().max()):.2e}; real code minus published, largest {float((real[column] - pub[column]).abs().max()):.3f}")

old = pd.read_parquet("outputs/season_ratings_product.parquet")
new = pd.read_parquet("outputs/season_ratings_product_boxfirst.parquet")
key = ["season", "player_id"]
both = old.merge(new[key + ["box_adj_off", "box_adj_def"]], on=key, suffixes=("_old", "_new"))
for s in ("off", "def"):
    both[f"box_{s}_old"] = both[f"prior_{s}"] - both[f"box_adj_{s}_old"]
    both[f"box_{s}_new"] = both[f"prior_{s}"] - both[f"box_adj_{s}_new"]


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


def wcorr(x, y, w):
    return float(np.average((x - np.average(x, weights=w)) * (y - np.average(y, weights=w)), weights=w) / (wsd(x, w) * wsd(y, w)))


rows = []
for label, lo, hi in (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("2026", 2026, 2026)):
    for s, poss in (("off", "poss_off"), ("def", "poss_def")):
        per = []
        for season, g in both[(both.season >= lo) & (both.season <= hi) & (both[poss] >= 1000)].groupby("season"):
            w = g[poss].to_numpy(float)
            per.append(dict(box_old=wsd(g[f"box_{s}_old"], w), adj_old=wsd(g[f"box_adj_{s}_old"], w), corr_old=wcorr(g[f"box_{s}_old"], g[f"box_adj_{s}_old"], w),
                            box_new=wsd(g[f"box_{s}_new"], w), adj_new=wsd(g[f"box_adj_{s}_new"], w), corr_new=wcorr(g[f"box_{s}_new"], g[f"box_adj_{s}_new"], w), prior=wsd(g[f"prior_{s}"], w)))
        rows.append(dict(era=label, side=s, **pd.DataFrame(per).mean().round(2).to_dict()))
print(pd.DataFrame(rows).to_string(index=False))

y = both[(both.season == 2026) & (both.poss_off >= 1000)].sort_values("rating_total", ascending=False).head(20)
top = pd.DataFrame({"player": y.player_name, "rating": y.rating_total, "O box was": y.box_off_old, "O box now": y.box_off_new, "O adj was": y.box_adj_off_old, "O adj now": y.box_adj_off_new,
                    "D box was": y.box_def_old, "D box now": y.box_def_new, "D adj was": y.box_adj_def_old, "D adj now": y.box_adj_def_new})
print(top.round(2).to_string(index=False))
d = both[(both.season == 2026) & (both.poss_off >= 1000)]
moved = pd.concat([d.box_adj_off_new - d.box_adj_off_old, d.box_adj_def_new - d.box_adj_def_old])
print("regulars 2026, points moved between the columns: median", round(float(moved.abs().median()), 2), "ninetieth percentile", round(float(moved.abs().quantile(0.9)), 2), "largest", round(float(moved.abs().max()), 2))
