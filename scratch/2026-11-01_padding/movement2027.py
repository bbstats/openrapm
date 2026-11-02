"""2027 on the published tables: the players without a game since the table kept, so the fix alone, or nearly."""
import pandas as pd

new = pd.read_parquet("outputs/season_ratings_product.parquet")
old = pd.read_parquet("outputs/season_ratings_product_before_padbins127.parquet")
both = new[new.season == 2027].merge(old[old.season == 2027], on="player_id", suffixes=("", "_old"))
both = both[both.poss_off >= 60]
both["move"] = (both.rating_total - both.rating_total_old).abs()
same = both[both.poss_off == both.poss_off_old]
played = both[both.poss_off != both.poss_off_old]
for name, part in (("no new game", same), ("new games", played)):
    print(name, len(part), "median", round(part.move.median(), 3), "p90", round(part.move.quantile(.9), 3), "largest", round(part.move.max(), 3))
print(same.sort_values("move", ascending=False).head(6)[["player_name", "rating_total", "rating_total_old", "poss_off"]].round(2).to_string(index=False))
under = new[new.season <= 2026].merge(old[old.season <= 2026], on=["season", "player_id"], suffixes=("", "_old"))
under["change"] = under.rating_total - under.rating_total_old
under = under[under.poss_off >= 60]
for low, high in ((60, 200), (200, 500), (500, 1500), (1500, 4444), (4444, 99999)):
    part = under[under.poss_off.between(low, high, inclusive="left")]
    print(f"possessions {low}-{high}: {len(part)} players, median move {part.change.abs().median():.3f}, p90 {part.change.abs().quantile(.9):.3f}, largest {part.change.abs().max():.2f}, mean change {part.change.mean():+.3f}")
