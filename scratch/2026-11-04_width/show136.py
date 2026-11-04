"""The final moves: the published table against the final tables, read-only."""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
NEW = sys.argv[1] if len(sys.argv) > 1 else "product136"
old = pd.read_parquet("outputs/season_ratings_product.parquet")
new = pd.read_parquet(f"outputs/season_ratings_{NEW}.parquet")
keep = ["player_id", "season", "player_name", "rating_off", "rating_def", "rating_total", "poss_off", "poss_def"]
both = old[keep].merge(new[keep[:2] + keep[3:]], on=["player_id", "season"], suffixes=("_old", "_new"), how="outer", indicator=True)
print("rows published", len(old), "new", len(new), both._merge.value_counts().to_dict())
both = both[both._merge == "both"].copy()
print("possessions differ in rows:", int((both.poss_off_old != both.poss_off_new).sum()))
both["change"] = both.rating_total_new - both.rating_total_old
both["change_o"] = both.rating_off_new - both.rating_off_old
both["change_d"] = both.rating_def_new - both.rating_def_old


def table(season, minimum=60, top=20):
    s = both[(both.season == season) & (both.poss_off_new >= minimum)].copy()
    s["rank_old"] = s.rating_total_old.rank(ascending=False).astype(int)
    s["rank_new"] = s.rating_total_new.rank(ascending=False).astype(int)
    t = s.sort_values("rank_new").head(top)
    print(f"\n{season}, top {top} under the new version, {minimum}+ possessions")
    print(t[["player_name", "rating_total_old", "rating_total_new", "change", "change_o", "change_d", "rank_old", "rank_new", "poss_off_new"]].round(2).to_string(index=False))
    tops = (set(s.nsmallest(top, "rank_old").player_id), set(s.nsmallest(top, "rank_new").player_id))
    names = s.set_index("player_id").player_name
    print("  in:", [(names[i], round(float(s.set_index("player_id").change[i]), 2)) for i in tops[1] - tops[0]],
          " out:", [(names[i], round(float(s.set_index("player_id").change[i]), 2)) for i in tops[0] - tops[1]])
    reg = s[s.poss_off_new >= (1000 if season < 2027 else 60)]
    d = reg.change.abs()
    print(f"  {len(reg)} players: move median {d.median():.2f}, 90th percentile {d.quantile(.9):.2f}, largest {d.max():.2f}, correlation {reg.rating_total_old.corr(reg.rating_total_new):.3f}")
    print("  up most:", [(r.player_name, round(r.change, 2)) for r in reg.nlargest(6, "change").itertuples()])
    print("  down most:", [(r.player_name, round(r.change, 2)) for r in reg.nsmallest(6, "change").itertuples()])


table(2026)
table(2027)
print("\nregulars, 1,000+ possessions, by era: move median, 90th percentile, largest")
for era, a, b in (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026)):
    s = both[(both.season >= a) & (both.season <= b) & (both.poss_off_new >= 1000)]
    d = s.change.abs()
    print(f"  {era}: {d.median():.2f}, {d.quantile(.9):.2f}, {d.max():.2f}, mean change {s.change.mean():+.3f}")
for label, a, b in (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("2026", 2026, 2026)):
    out = []
    for suffix in ("old", "new"):
        s = both[(both.season >= a) & (both.season <= b) & (both.poss_off_new >= 1000) & (both.poss_def_new >= 1000)]
        sd = []
        for col, w in ((f"rating_off_{suffix}", "poss_off_new"), (f"rating_def_{suffix}", "poss_def_new")):
            v = [np.average((g[col] - np.average(g[col], weights=g[w])) ** 2, weights=g[w]) for _, g in s.groupby("season")]
            sd.append(float(np.sqrt(np.mean(v))))
        out.append((round(sd[0], 3), round(sd[1], 3), round(sd[1] / sd[0], 3)))
    print(f"  width {label}: published offense, defense, defense over offense {out[0]}; new {out[1]}")
s = both[both.poss_off_new >= 1000].copy()
s["rank_old"] = s.rating_total_old.rank(ascending=False).astype(int)
s["rank_new"] = s.rating_total_new.rank(ascending=False).astype(int)
print("\nbest seasons since 1997, 1,000+ possessions")
print(s.sort_values("rank_new").head(12)[["player_name", "season", "rating_total_old", "rating_total_new", "change", "rank_old", "rank_new"]].round(2).to_string(index=False))
