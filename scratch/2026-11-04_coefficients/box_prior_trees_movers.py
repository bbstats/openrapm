"""Read-only, after box_prior_trees.py: who moves when defense's boosters run 330 trees instead of 130.

The move is the Box score column's, beyond a rescale: what is left of the 330-tree output after the best straight line
on the 130-tree output, x that season's Box Adj weight, prior scale and shrink / the straight line's slope.  Regulars
of 1,000+ possessions, weighted by possessions as in box_prior_trees.py.
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SEASONS = [2000, 2005, 2010, 2015, 2022, 2026]
pd.set_option("display.width", 200)
published = pd.read_parquet("outputs/season_ratings_product.parquet")
summary = pd.read_csv("outputs/csv/box_prior_trees.csv").set_index("season")
name = next(c for c in ("player_name", "name", "player") if c in published.columns)
rows, players = [], []
for season in SEASONS:
    p = pd.read_csv(f"outputs/box_prior_trees/players_{season}.csv").set_index("player_id")
    t = published[published.season == season].set_index("player_id")
    keep = t.index[t.poss_def >= 1000].intersection(p.index)
    w = t.loc[keep, "poss_def"].to_numpy(float)
    a, b = p.loc[keep, "trees_130"].to_numpy(float), p.loc[keep, "trees_330"].to_numpy(float)
    line, net = summary.loc[season, "straight_line_slope"], summary.loc[season, "net"]
    move = ((b - np.average(b, weights=w)) - line * (a - np.average(a, weights=w))) * net / line
    size = np.abs(move)
    high, low = np.quantile(a, 0.9), np.quantile(a, 0.1)
    rows.append(dict(season=season, median_size=np.median(size), size_95th=np.quantile(size, 0.95), largest=size.max(),
                     share_at_025=np.mean(size >= 0.25), top_tenth_mean=move[a >= high].mean(),
                     middle_mean=move[(a > low) & (a < high)].mean(), bottom_tenth_mean=move[a <= low].mean()))
    players.append(pd.DataFrame(dict(season=season, player=t.loc[keep, name].to_numpy(), possessions=w, output_130=a,
                                     box_column_today=(a - np.average(a, weights=w)) * net, move=move,
                                     defense=t.loc[keep].filter(regex="^(defense|def_|rating_def|defence)").iloc[:, 0].to_numpy()
                                     if t.filter(regex="^(defense|def_|rating_def|defence)").shape[1] else np.nan)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
every = pd.concat(players, ignore_index=True)
every.to_csv("outputs/csv/box_prior_trees_movers.csv", index=False)
latest = every[every.season == 2026].sort_values("move")
print(latest.tail(6).round(3).to_string(index=False))
print(latest.head(6).round(3).to_string(index=False))
print(latest.sort_values("box_column_today").tail(3)[["player", "box_column_today"]].round(2).to_string(index=False))
print(latest.sort_values("box_column_today").head(3)[["player", "box_column_today"]].round(2).to_string(index=False))
print([c for c in published.columns if "def" in c.lower()][:20])
