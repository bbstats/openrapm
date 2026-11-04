"""Read-only: experiment 134's widths, 2026 top 20 and 2027 today against the incumbent, each like for like."""
import numpy as np
import pandas as pd

GROUP = "Ajay Mitchell|Hugo Gonz|Oso Ighodaro|Jimmy Butler|Nikola Joki|Zach Edey"
inc = pd.read_parquet("outputs/season_ratings_padbins127.parquet")
new = pd.read_parquet("outputs/season_ratings_rating134.parquet")


def sd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


rows = []
for era, (lo, hi) in {"1997-2006": (1997, 2006), "2007-2016": (2007, 2016), "2017-2026": (2017, 2026), "2026": (2026, 2026)}.items():
    row = {"era": era}
    for label, table in (("today", inc), ("new", new)):
        p = table[(table.season >= lo) & (table.season <= hi) & (table.poss_off >= 1000) & (table.poss_def >= 1000)]
        o, d = sd(p.rating_off, p.poss_off), sd(p.rating_def, p.poss_def)
        row.update({f"{label}_O": round(o, 3), f"{label}_D": round(d, 3), f"{label}_DO": round(d / o, 3)})
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))


def compare(old, cand, season, minimum, label):
    both = old[old.season == season].set_index("player_id").join(cand[cand.season == season].set_index("player_id"), rsuffix="_new")
    both = both[both.poss_off >= minimum].copy()
    both["ri"] = both.rating_total.rank(ascending=False).astype(int)
    both["rn"] = both.rating_total_new.rank(ascending=False).astype(int)
    both["move"] = both.rating_total_new - both.rating_total
    both["mo"], both["md"] = both.rating_off_new - both.rating_off, both.rating_def_new - both.rating_def
    show = ["player_name", "rating_total", "rating_total_new", "move", "mo", "md", "ri", "rn", "poss_off"]
    print(f"\n{label}: {len(both)} players; top 20 under the new line")
    print(both[both.rn <= 20].sort_values("rn")[show].round(2).to_string(index=False))
    print("left the top 20:")
    print(both[(both.ri <= 20) & (both.rn > 20)].sort_values("ri")[show].round(2).to_string(index=False))
    regulars = both[both.poss_off >= (1000 if season == 2026 else 60)]
    print(f"move median {regulars.move.abs().median():.3f}, 90th percentile {regulars.move.abs().quantile(0.9):.3f}, "
          f"largest {regulars.move.abs().max():.3f}; correlation {np.corrcoef(regulars.rating_total, regulars.rating_total_new)[0, 1]:.3f}; "
          f"spread {regulars.rating_total.std():.3f} to {regulars.rating_total_new.std():.3f}")
    ordered = regulars.sort_values("move")
    print("largest falls:", [(a, round(b, 2)) for a, b in ordered[["player_name", "move"]].head(6).to_numpy()])
    print("largest rises:", [(a, round(b, 2)) for a, b in ordered[["player_name", "move"]].tail(6).to_numpy()])
    return both


top = compare(inc, new, 2026, 60, "2026")
print("the ruling's group:", [(a, round(b, 2)) for a, b in top[top.player_name.str.contains(GROUP)][["player_name", "move"]].to_numpy()])
now_old = pd.read_parquet("outputs/season_ratings_padbins127now27.parquet")
now_new = pd.read_parquet("outputs/season_ratings_rating134now27.parquet")
compare(now_old, now_new, 2027, 60, "2027 today")
for name, table in (("today", now_old), ("new", now_new)):
    t = table[table.season == 2027]
    print(name, "2027 prior scales", round(float(t.prior_scale_off.iloc[0]), 3), round(float(t.prior_scale_def.iloc[0]), 3))
