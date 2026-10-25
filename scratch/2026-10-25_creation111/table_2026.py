"""Experiment 111: the new self-creation per 100 for 2026, against experiment 110's, players with 1,500+ offensive
possessions.  New: unassisted makes + misses not off a pass + free-throw trips not off a pass, from the season's
counts over his offensive possessions, an and-one's free throw left to its make; experiment 110's from the panel's padded rates."""
import numpy as np
import pandas as pd

from openrapm.features import CREATION, add_features

players = pd.read_parquet("outputs/creation111/players.parquet")
players["passed_other_trips"] = players.passed_trips - players.assisted_and_ones
s = players.groupby(["shooter", "season"])[["makes", "assisted_makes", "misses", "passed_misses", "trips",
                                            "other_trips", "technical", "passed_other_trips"]].sum().reset_index()
panel = pd.read_parquet("outputs/season_panel.parquet")
o = panel[panel.side == "O"].reset_index(drop=True)
o = add_features(o, CREATION)
s = s.merge(o[["player_id", "season", "possessions", CREATION[0], "free_throws_made_per100_uncentred",
               "free_throws_missed_per100_uncentred"]], left_on=["shooter", "season"],
            right_on=["player_id", "season"])
per = 100 / s.possessions
s["unassisted_makes"] = (s.makes - s.assisted_makes) * per
s["self_created_misses"] = (s.misses - s.passed_misses) * per
s["self_created_trips"] = (s.other_trips - s.passed_other_trips) * per
s["self_creation_new"] = s.unassisted_makes + s.self_created_misses + s.self_created_trips
s["self_creation_old"] = s[CREATION[0]]
s["change"] = s.self_creation_new - s.self_creation_old
s["miss_share_off_pass"] = s.passed_misses / s.misses
s["trip_share_off_pass"] = s.passed_other_trips / s.other_trips
big = s[s.possessions >= 1500]
print(f"1,500+ possessions, every season: correlation {np.corrcoef(big.self_creation_old, big.self_creation_new)[0, 1]:.3f}, "
      f"mean {big.self_creation_old.mean():.2f} to {big.self_creation_new.mean():.2f}, "
      f"sd {big.self_creation_old.std():.2f} to {big.self_creation_new.std():.2f}; "
      f"trips without a make per 0.44 free-throw attempts {(big.other_trips / (0.44 * (big.free_throws_made_per100_uncentred + big.free_throws_missed_per100_uncentred) * big.possessions / 100)).median():.3f}")
names = pd.read_parquet("outputs/season_ratings_product.parquet")[["player_id", "season", "player_name"]].drop_duplicates()
t = big[big.season == 2026].merge(names, on=["player_id", "season"])
cols = ["player_name", "unassisted_makes", "self_created_misses", "self_created_trips", "miss_share_off_pass",
        "trip_share_off_pass", "self_creation_old", "self_creation_new", "change"]
t[cols].sort_values("self_creation_new", ascending=False).round(2).to_csv("outputs/creation111/top_2026.csv", index=False)
print(t[cols].sort_values("self_creation_new", ascending=False).head(20).round(2).to_string(index=False))
print()
print(t.reindex(t.change.abs().sort_values(ascending=False).index)[cols].head(10).round(2).to_string(index=False))
