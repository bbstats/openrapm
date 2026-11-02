"""The published table against the one kept before experiment 127: the movement by era for players with 60+ possessions,
the 2026 and 2027 top 20.  2027 mixes the fix with new games (the table kept was the 6 AM job's of 2026-11-01)."""
import numpy as np
import pandas as pd

new = pd.read_parquet("outputs/season_ratings_product.parquet")
old = pd.read_parquet("outputs/season_ratings_product_before_padbins127.parquet")
print(list(new.columns)[:20])
keys = ["season", "player_id"]
both = new.merge(old, on=keys, suffixes=("", "_old"), how="inner")
both = both[both.poss_off >= 60]
both["move"] = (both.rating_total - both.rating_total_old).abs()
rows = []
for name, part in (("1997-2006", both[both.season.between(1997, 2006)]), ("2007-2016", both[both.season.between(2007, 2016)]),
                   ("2017-2026", both[both.season.between(2017, 2026)]), ("2026", both[both.season == 2026]), ("2027", both[both.season == 2027])):
    rows.append(dict(seasons=name, players=len(part), median_move=part.move.median(), p90_move=part.move.quantile(.9), largest_move=part.move.max(),
                     sd_before=part.groupby("season").rating_total_old.std().mean(), sd_after=part.groupby("season").rating_total.std().mean(),
                     correlation=np.corrcoef(part.rating_total, part.rating_total_old)[0, 1]))
print(pd.DataFrame(rows).round(3).to_string(index=False))
for season in (2026, 2027):
    a = both[both.season == season].copy()
    a["rank"], a["rank_old"] = a.rating_total.rank(ascending=False).astype(int), a.rating_total_old.rank(ascending=False).astype(int)
    a["change"] = a.rating_total - a.rating_total_old
    print(season, "top 20 kept names:", len(set(a[a["rank"] <= 20].player_id) & set(a[a["rank_old"] <= 20].player_id)))
    print(a.sort_values("rank").head(21)[["rank", "rank_old", "player_name", "rating_total", "rating_total_old", "change", "poss_off", "poss_off_old"]].round(2).to_string(index=False))
big = both.sort_values("move", ascending=False).head(8)
print(big[["season", "player_name", "rating_total", "rating_total_old", "poss_off"]].round(2).to_string(index=False))
