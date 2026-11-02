"""The published table against the tables the experiment reported: the candidate's and the incumbent's, every finished season."""
import numpy as np
import pandas as pd

def load(name):
    return pd.read_parquet(f"outputs/{name}.parquet")

new, before = load("season_ratings_product"), load("season_ratings_product_before_padbins127")
cand, inc = load("season_ratings_padbins127"), load("season_ratings_ctx110")
def gap(a, b, name):
    both = a[a.season <= 2026].merge(b[b.season <= 2026], on=["season", "player_id"], suffixes=("", "_b"), how="outer", indicator=True)
    only = (both["_merge"] != "both").sum()
    shared = both[both["_merge"] == "both"]
    diff = (shared.rating_total - shared.rating_total_b).abs()
    print(f"{name}: {len(shared)} shared, {only} unmatched, largest {diff.max():.2e}, median {diff.median():.2e}, by season over 1e-9: {shared[diff > 1e-9].season.nunique()}")
    return shared.assign(diff=diff)
gap(new, cand, "published now vs the candidate")
gap(before, inc, "published before vs the incumbent ctx110")
for name, t in (("published now", new), ("candidate", cand), ("before", before), ("incumbent", inc)):
    a = t[(t.season == 2026)].copy()
    a = a.sort_values("rating_total", ascending=False).reset_index(drop=True)
    print(name, a.head(22).player_name.tolist()[17:22] if "player_name" in a else a.head(22).player_id.tolist()[17:22])
    if "player_name" in a:
        print("  Ausar", a[a.player_name.eq("Ausar Thompson")][["rating_total", "poss_off"]].round(3).values.tolist(), "rank", a.index[a.player_name.eq("Ausar Thompson")].tolist())
