"""The wins columns and the site's files of the final tables, against the published ones.  Read-only."""
import json
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
NEW = "product136"
old = pd.read_parquet("outputs/season_ratings_product.parquet")
new = pd.read_parquet(f"outputs/season_ratings_{NEW}.parquet")
r_new = pd.read_parquet(f"outputs/season_ratings_{NEW}_regular.parquet")
r_flag = pd.read_parquet("outputs/season_ratings_product134_regular.parquet")
r_old = pd.read_parquet("outputs/season_ratings_product_regular.parquet")
print("regular-season tables, rows:", len(r_new), len(r_flag), len(r_old))
for season in (2002, 2016, 2026):
    a = new[(new.season == season) & (new.poss_off >= 1000)].set_index("player_id").rating_total
    for label, t in (("final", r_new), ("the flagged build's", r_flag), ("published", r_old)):
        b = t[t.season == season].set_index("player_id").rating_total.reindex(a.index)
        print(f"  {season}, final whole-season ratings against the {label} regular-season table: correlation {a.corr(b):.4f}, root mean square gap {np.sqrt(((a - b) ** 2).mean()):.3f}")
keep = ["player_id", "season", "player_name", "wosa", "wosa_off", "wosa_def", "wosa_pace", "leverage", "starts_rs", "rating_total", "poss_off"]
m = old[keep].merge(new[keep], on=["player_id", "season", "player_name"], suffixes=("_old", "_new"))
print("rows with wins:", len(m), "missing new:", int(m.wosa_new.isna().sum()), "missing old:", int(m.wosa_old.isna().sum()))
print("leverage gap, largest:", float((m.leverage_old - m.leverage_new).abs().max()), " starts gap, largest:", float((m.starts_rs_old - m.starts_rs_new).abs().max()))
for season, top in ((2026, 12), (2025, 6), (2027, 8)):
    s = m[m.season == season].copy()
    s["rank_old"] = s.wosa_old.rank(ascending=False).astype(int)
    s["rank_new"] = s.wosa_new.rank(ascending=False).astype(int)
    t = s.sort_values("rank_new").head(top)
    print(f"\n{season}, wins over starter average")
    print(t[["player_name", "wosa_old", "wosa_new", "rank_old", "rank_new"]].round(1).to_string(index=False))
    d = (s.wosa_new - s.wosa_old)
    print(f"  move median {d.abs().median():.2f}, largest {d.abs().max():.2f}; sum published {s.wosa_old.sum():.0f}, new {s.wosa_new.sum():.0f}")
for name, path in (("ratings", "ratings.json"), ("detail", "decompose_openrapm.json"), ("power", "power.json")):
    a = json.load(open(f"outputs/site_check136/data/{path}", encoding="utf-8"))
    b = json.load(open(f"A:/code/spmm_site/docs/data/{path}", encoding="utf-8"))
    print(name, "same keys:", set(a) == set(b), "meta same keys:", set(a["meta"]) == set(b["meta"]), {k: a["meta"][k] for k in a["meta"] if k in ("built", "default_season", "through", "penalty")})
a = json.load(open("outputs/site_check136/data/decompose_openrapm.json", encoding="utf-8"))
worst = 0.0
for season, rows in a["rows"].items():
    for r in rows:
        for side in (r[3], r[4], r[5]):
            worst = max(worst, abs(sum(side[:5]) - side[5]))
print("largest gap, the five pieces on the page against the rating, after rounding:", round(worst, 3))
pw = pd.DataFrame(json.load(open("outputs/site_check136/data/power.json", encoding="utf-8"))["teams"])
print("power, teams", len(pw), "mean of totals", round(float(pw[[c for c in pw.columns if c in ("t", "total", "power")][0]].mean()), 3) if any(c in pw.columns for c in ("t", "total", "power")) else list(pw.columns)[:8])
