"""2027 before and after a change: the top 20 by total, how far each regular moved, and the panel's finished seasons.
python scratch/2026-10-30_bugfix/compare_2027.py BEFORE_TABLE AFTER_TABLE [BEFORE_PANEL AFTER_PANEL]"""
import sys

import numpy as np
import pandas as pd

from openrapm import features

before, after = (pd.read_parquet(path) for path in sys.argv[1:3])
cols = ["player_id", "player_name", "rating_off", "rating_def", "rating_total", "poss_off"]
b = before[before.season == 2027][cols].set_index("player_id")
a = after[after.season == 2027][cols].set_index("player_id")
both = a.join(b[["rating_off", "rating_def", "rating_total"]], rsuffix="_before", how="inner")
shown = both[both.poss_off >= 60]
for part in ("off", "def", "total"):
    move = (shown[f"rating_{part}"] - shown[f"rating_{part}_before"]).abs()
    print(f"2027, {len(shown)} players with 60+ possessions, {part}: median move {move.median():.2f} per 100, "
          f"90th percentile {move.quantile(0.9):.2f}, largest {move.max():.2f}")
shown = shown.assign(rank_after=shown.rating_total.rank(ascending=False).astype(int),
                     rank_before=shown.rating_total_before.rank(ascending=False).astype(int))
top = shown.sort_values("rating_total", ascending=False).head(20)
table = pd.DataFrame({"player": top.player_name, "offense": top.rating_off.round(2),
                      "defense": top.rating_def.round(2), "total": top.rating_total.round(2),
                      "total_before": top.rating_total_before.round(2),
                      "change": (top.rating_total - top.rating_total_before).round(2),
                      "rank_before": top.rank_before, "possessions": top.poss_off.astype(int)})
print(table.to_string(index=False))
left = shown[(shown.rank_before <= 20) & (shown.rank_after > 20)]
print("out of the top 20:", ", ".join(f"{r.player_name} {r.rank_before} to {r.rank_after}" for r in left.itertuples()))
print("finished seasons' ratings unchanged:",
      bool(np.allclose(before[before.season <= 2026].sort_values(["season", "player_id"]).rating_total.to_numpy(),
                       after[after.season <= 2026].sort_values(["season", "player_id"]).rating_total.to_numpy(),
                       atol=0, rtol=0)))
if len(sys.argv) > 4:
    p_before, p_after = (pd.read_parquet(path) for path in sys.argv[3:5])
    o, n = (p[p.season <= 2026].sort_values(["player_id", "season", "side"]).reset_index(drop=True)
            for p in (p_before, p_after))
    print("1997-2026 panel rows identical:", all(o[c].equals(n[c]) for c in o.columns if c in n.columns))
    hustle = list(features.HUSTLE_RATES)
    q = p_after[(p_after.season == 2027) & (p_after.side == "O") & (p_after.possessions >= 60)]
    print("2027 players with 60+ possessions and no hustle value:", int(q[hustle].isna().all(axis=1).sum()), "of",
          len(q))
