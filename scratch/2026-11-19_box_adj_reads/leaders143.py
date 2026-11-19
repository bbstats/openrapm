"""Read-only: experiment 143's product build, `product143`, against the published tables, `product`: every season's
top-rated player and WOSA leader in both, the 2026 and 2027 top 20 by rating with who moved, and how far regulars moved.

Writes outputs/csv/leaders143.csv (one row a season) and outputs/csv/moves143.csv (every player-season in both).
    .venv/Scripts/python scratch/2026-11-19_box_adj_reads/leaders143.py
"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
KEEP = ["season", "player_id", "player_name", "rating_off", "rating_def", "rating_total", "poss_off", "wosa"]


def main() -> None:
    today = pd.read_parquet("outputs/season_ratings_product.parquet")[KEEP]
    new = pd.read_parquet("outputs/season_ratings_product143.parquet")[KEEP]
    both = today.merge(new.drop(columns=["player_name", "poss_off"]), on=["season", "player_id"],
                       suffixes=("_today", "_new"))
    both.to_csv("outputs/csv/moves143.csv", index=False)
    rows = []
    for season, group in both.groupby("season"):
        record = {"season": season}
        for what in ("rating_total", "wosa"):
            for side in ("today", "new"):
                top = group.loc[group[f"{what}_{side}"].idxmax()]
                record[f"{what} leader {side}"] = top.player_name
                record[f"{what} {side}"] = round(float(top[f"{what}_{side}"]), 2)
        rows.append(record)
    leaders = pd.DataFrame(rows)
    leaders.to_csv("outputs/csv/leaders143.csv", index=False)
    for what in ("rating_total", "wosa"):
        changed = leaders[leaders[f"{what} leader today"] != leaders[f"{what} leader new"]]
        print(f"\nseasons whose {what} leader changes: {len(changed)}")
        if len(changed):
            print(changed[["season", f"{what} leader today", f"{what} today", f"{what} leader new", f"{what} new"]]
                  .to_string(index=False))
            for _, row in changed.iterrows():
                group = both[both.season == row.season].set_index("player_name")
                for name in (row[f"{what} leader today"], row[f"{what} leader new"]):
                    print(f"    {row.season} {name}: {group.loc[name, f'{what}_today']:.2f} to "
                          f"{group.loc[name, f'{what}_new']:.2f}")
    regulars = both[(both.poss_off >= 1000) & (both.season <= 2026)]
    move = (regulars.rating_total_new - regulars.rating_total_today).abs()
    print(f"\nregulars of 1,000 or more possessions, 1997-2026: total moves a median {move.median():.3f}, ninetieth "
          f"percentile {move.quantile(0.9):.3f}, largest {move.max():.3f}")
    for season in (2026, 2027):
        group = both[both.season == season].copy()
        group["rank today"] = group.rating_total_today.rank(ascending=False).astype(int)
        group["rank new"] = group.rating_total_new.rank(ascending=False).astype(int)
        group["change"] = group.rating_total_new - group.rating_total_today
        top = group[(group["rank today"] <= 20) | (group["rank new"] <= 20)].sort_values("rank new")
        print(f"\n{season} top 20, either table")
        print(top[["player_name", "rating_total_today", "rating_total_new", "change", "rank today", "rank new",
                   "wosa_today", "wosa_new"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
