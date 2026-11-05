"""Read-only: where the rebuilt 2027 Box Adj differs from the preview's, recovered from the table."""
import numpy as np
import pandas as pd

key = ["season", "player_id"]
now = pd.read_parquet("outputs/season_ratings_product.parquet").query("season == 2027").set_index("player_id")
old = pd.read_parquet("outputs/season_ratings_product_before_boxfirst.parquet").query("season == 2027").set_index("player_id")
pre = pd.read_parquet("outputs/season_ratings_product_boxfirst.parquet").query("season == 2027").set_index("player_id").reindex(now.index)
step3 = pd.read_parquet("outputs/season_ratings_product_step3.parquet").query("season == 2027").iloc[0]
print("prior scales in the table", step3.prior_scale_off, step3.prior_scale_def)
for s in ("off", "def"):
    gap = now[f"box_adj_{s}"] - pre[f"box_adj_{s}"]
    moved = now[f"box_adj_{s}"] - old[f"box_adj_{s}"]
    poss = now[f"poss_{s}"]
    print(s, "gap: rms", round(float(np.sqrt(np.mean(gap ** 2))), 4), "median abs", round(float(gap.abs().median()), 4), "corr with the column's move", round(float(np.corrcoef(gap, moved)[0, 1]), 3),
          "ratio gap to move, regression", round(float((gap * moved).sum() / (moved * moved).sum()), 4))
    w = now.assign(gap=gap, moved=moved, new=now[f"box_adj_{s}"], preview=pre[f"box_adj_{s}"], poss=poss)
    print(w.reindex(gap.abs().sort_values(ascending=False).index)[["player_name", "poss", "gap", "moved", "new", "preview"]].head(6).round(3).to_string())
    t = now.groupby(now.index % 1).size()
