"""The rebuilt published table against the ratings the site shows, docs/data/ratings.json as of 2026-11-16.

    python scratch/2026-11-16_cleanup/against_site.py [table]     (default outputs/season_ratings_product.parquet)

Per season: players matched, the correlation and the mean and largest gap of offense, defense and the total, in
points per 100, and of WOSA.  Then the 2026 top 20 by total on the site beside the rebuild.  The site rounds to two
decimals, so a gap under 0.005 is rounding.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
table = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else ROOT / "outputs/season_ratings_product.parquet")
site = pd.DataFrame(json.loads((ROOT / "docs/data/ratings.json").read_text(encoding="utf-8"))["rows"])
site = site.rename(columns={"s": "season", "n": "name", "o": "off_site", "d": "def_site", "t": "tot_site",
                            "id": "player_id", "w": "wosa_site"})
mine = table[["player_id", "season", "rating_off", "rating_def", "rating_total"]
             + (["wosa"] if "wosa" in table.columns else [])].rename(
    columns={"rating_off": "off_new", "rating_def": "def_new", "rating_total": "tot_new", "wosa": "wosa_new"})
both = site.merge(mine, on=["player_id", "season"], how="left")
print(f"site rows {len(site)}, matched {both.off_new.notna().sum()}")

rows = []
for season, g in both.dropna(subset=["off_new"]).groupby("season"):
    row = {"season": season, "players": len(g)}
    for key in ("off", "def", "tot", "wosa"):
        if f"{key}_new" not in g or g[f"{key}_new"].isna().all():
            continue
        gap = (g[f"{key}_new"] - g[f"{key}_site"]).abs()
        row[f"{key}_corr"] = round(float(np.corrcoef(g[f"{key}_new"], g[f"{key}_site"])[0, 1]), 4)
        row[f"{key}_mean_gap"] = round(float(gap.mean()), 3)
        row[f"{key}_max_gap"] = round(float(gap.max()), 2)
    rows.append(row)
seasons = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(seasons.to_string(index=False))
finished = seasons[seasons.season <= 2026]
print("\nfinished seasons, medians:", finished.drop(columns=["season", "players"]).median().round(4).to_dict())

top = both[both.season == 2026].sort_values("tot_site", ascending=False).head(20).copy()
ranked = both[both.season == 2026].dropna(subset=["tot_new"]).sort_values("tot_new", ascending=False)
new_rank = {pid: k + 1 for k, pid in enumerate(ranked.player_id)}
top["site_rank"] = range(1, 21)
top["new_rank"] = top.player_id.map(new_rank)
top["change"] = (top.tot_new - top.tot_site).round(2)
print("\n2026 top 20 on the site, beside the rebuild:")
print(top[["site_rank", "name", "off_site", "def_site", "tot_site", "tot_new", "change", "new_rank"]]
      .round(2).to_string(index=False))
