"""Writes the split learned first into the published tables for the finished seasons, from box_first_tables.py's copies of the backup.
2027 keeps the rows its real rebuild wrote.  Checks 2005 against the real code, and every other number against the backup."""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
key = ["season", "player_id"]
real = pd.read_parquet("outputs/season_ratings_boxfirst_check2005_step3.parquet").query("season == 2005").set_index("player_id")
copy = pd.read_parquet("outputs/season_ratings_product_before_boxfirst_boxfirst_step3.parquet").query("season == 2005").set_index("player_id").reindex(real.index)
backup = pd.read_parquet("outputs/season_ratings_product_before_boxfirst_step3.parquet").query("season == 2005").set_index("player_id").reindex(real.index)
for column in ("rating_off", "rating_def", "prior_off", "prior_def"):
    print(f"2005, real code minus the published step-3 table, {column}: largest {float((real[column] - backup[column]).abs().max()):.2e}")
for column, poss in (("box_adj_off", "poss_off"), ("box_adj_def", "poss_def")):
    gap = (real[column] - copy[column]).abs()
    print(f"2005, {column}, real code minus the copy: largest {float(gap.max()):.2e}; with 100+ possessions {float(gap[real[poss] >= 100].max()):.2e}")

for label in ("", "_step3", "_step4"):
    now = pd.read_parquet(f"outputs/season_ratings_product{label}.parquet")
    copy = pd.read_parquet(f"outputs/season_ratings_product_before_boxfirst_boxfirst{label}.parquet").set_index(key)
    was = pd.read_parquet(f"outputs/season_ratings_product_before_boxfirst{label}.parquet").set_index(key).sort_index()
    finished = now.season.to_numpy() <= 2026
    index = pd.MultiIndex.from_arrays([now.season, now.player_id])
    for column in ("box_adj_off", "box_adj_def"):
        values = now[column].to_numpy(float).copy()
        values[finished] = copy[column].reindex(index).to_numpy(float)[finished]
        now[column] = values
    now.to_parquet(f"outputs/season_ratings_product{label}.parquet", index=False)
    after = now.set_index(key).sort_index()
    numeric = [c for c in was.columns if pd.api.types.is_numeric_dtype(was[c]) and c not in ("box_adj_off", "box_adj_def")]
    gap = (after[numeric] - was[numeric]).abs().max()
    print(f"{label or 'final'}: same players {after.index.equals(was.index)}, largest change in any other number {float(gap.max()):.2e}; prior and rating columns {float(gap[[c for c in numeric if 'rating' in c or 'prior' in c]].max()):.2e}")
