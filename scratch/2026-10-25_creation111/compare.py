"""Experiment 111: the new panel against the old one, and the two self-creations side by side on whole seasons."""
import numpy as np
import pandas as pd

from openrapm.features import CREATION, CREATION_ACTUAL_MAKES, add_features

new = pd.read_parquet("outputs/season_panel.parquet")
old = pd.read_parquet("outputs/season_panel_before_creation111.parquet")
key = ["side", "player_id", "season"]
m = old.merge(new, on=key, suffixes=("_old", ""))
shared = [c for c in old.columns if c not in key]
worst = max(((c, np.nanmax(np.abs(m[c + "_old"].to_numpy(float) - m[c].to_numpy(float)))) for c in shared),
            key=lambda t: t[1])
print(f"rows old {len(old)} new {len(new)} matched {len(m)}; columns added {sorted(set(new.columns) - set(old.columns))}")
print(f"largest change in an older column: {worst[0]} {worst[1]:.2e}")

o = new[new.side == "O"].reset_index(drop=True)
o = add_features(o, CREATION + CREATION_ACTUAL_MAKES)
makes = o.twos_made_per100_uncentred + o.threes_made_per100_uncentred
raw = (o.unassisted_points_per100_uncentred - o.unassisted_threes_made_per100_uncentred) / 2
print(f"unassisted makes below 0: {(raw < 0).mean():.4f}, above his makes: {(raw > makes).mean():.4f} of offense rows")
big = o[o.possessions >= 1500]
print(f"1,500+ possessions: correlation of the two {np.corrcoef(big[CREATION[0]], big[CREATION_ACTUAL_MAKES[0]])[0, 1]:.3f}; "
      f"mean {big[CREATION[0]].mean():.2f} vs {big[CREATION_ACTUAL_MAKES[0]].mean():.2f}; "
      f"sd {big[CREATION[0]].std():.2f} vs {big[CREATION_ACTUAL_MAKES[0]].std():.2f}")
names = pd.read_parquet("outputs/season_ratings_product.parquet")[["player_id", "season", "player_name"]].drop_duplicates()
t = o[(o.season == 2026) & (o.possessions >= 1500)].merge(names, on=["player_id", "season"])
t = t.assign(makes=makes[t.index] if False else t.twos_made_per100_uncentred + t.threes_made_per100_uncentred,
             unassisted_makes=(t.unassisted_points_per100_uncentred - t.unassisted_threes_made_per100_uncentred) / 2)
t["actual_share_assisted"] = 1 - t.unassisted_makes / t.makes
level, a, b = 0.73, 0.03, 0.01
t["estimated_share_assisted"] = np.clip(level - a * t.assists_per100_uncentred - b * t.makes, 0, 1)
t["change"] = t[CREATION_ACTUAL_MAKES[0]] - t[CREATION[0]]
cols = ["player_name", "makes", "actual_share_assisted", "estimated_share_assisted", CREATION[0], CREATION_ACTUAL_MAKES[0], "change"]
print("\n2026, top 20 by the new self-creation")
print(t.sort_values(CREATION_ACTUAL_MAKES[0], ascending=False)[cols].head(20).round(2).to_string(index=False))
print("\n2026, largest moves")
print(t.reindex(t.change.abs().sort_values(ascending=False).index)[cols].head(12).round(2).to_string(index=False))
lb = t[t.player_name.str.contains("LeBron")]
print("\nLeBron 2026:", lb[["unassisted_points_per100_uncentred", "unassisted_threes_made_per100_uncentred", "makes",
      "twos_missed_per100_uncentred", "threes_missed_per100_uncentred", "free_throws_made_per100_uncentred",
      "free_throws_missed_per100_uncentred", "assists_per100_uncentred", CREATION[0], CREATION_ACTUAL_MAKES[0]]].round(4).T)
