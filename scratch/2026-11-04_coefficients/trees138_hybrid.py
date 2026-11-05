"""Read-only: where experiment 138's year-over-year gain comes from: each team's total, or how credit is split inside it.

Two mixed tables, through the model's own hold of team totals (second_prior.hold_team_totals, trap 45), both scored on the
year-over-year test against today's ratings:
    totals138    today's split inside each team, with the candidate's team totals
    split138     the candidate's split inside each team, with today's team totals
Then how far team totals moved, per team and season, in points per 100 for five players on the court.
"""
import sys

import numpy as np
import pandas as pd

from openrapm import second_prior as sp
from openrapm.commands.build import season_lineup_rows
from openrapm.data.config import load_config

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
cfg = load_config()
today = pd.read_parquet("outputs/season_ratings_mid136.parquet")
candidate = pd.read_parquet("outputs/season_ratings_trees138_line.parquet")
totals, split, moves = [], [], []
for season in sorted(today.season.unique()):
    a = today[today.season == season].set_index("player_id")
    b = candidate[candidate.season == season].set_index("player_id").reindex(a.index)
    stints = season_lineup_rows(cfg, int(season))
    mixed_totals, mixed_split = a.copy(), a.copy()
    for side, column in (("O", "rating_off"), ("D", "rating_def")):
        team_poss = sp.team_possessions(stints, side)
        mixed_totals[column] = sp.hold_team_totals(a[column], b[column], team_poss)
        mixed_split[column] = sp.hold_team_totals(b[column], a[column], team_poss)
        for team, group in team_poss.groupby("team"):
            mine = group[group.player_id.isin(a.index)]
            w = mine.poss.to_numpy(float)
            five = 5.0 * w / w.sum()
            ids = mine.player_id.to_numpy()
            moves.append(dict(season=season, side=side, team=team, today=float((five * a.loc[ids, column].to_numpy(float)).sum()),
                              candidate=float((five * b.loc[ids, column].to_numpy(float)).sum())))
    for mixed in (mixed_totals, mixed_split):
        mixed["rating_total"] = mixed.rating_off + mixed.rating_def
    totals.append(mixed_totals.reset_index())
    split.append(mixed_split.reset_index())
pd.concat(totals, ignore_index=True).to_parquet("outputs/season_ratings_totals138.parquet", index=False)
pd.concat(split, ignore_index=True).to_parquet("outputs/season_ratings_split138.parquet", index=False)
moves = pd.DataFrame(moves)
moves["move"] = moves.candidate - moves.today
moves.to_csv("outputs/csv/trees138_team_totals.csv", index=False)
for era, low, high in (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026)):
    part = moves[(moves.season >= low) & (moves.season <= high)]
    print(era, part.groupby("side").apply(lambda g: pd.Series(dict(teams=len(g), team_total_sd_today=g.today.std(), team_total_sd_new=g.candidate.std(),
                                                                  move_sd=g.move.std(), move_largest=g.move.abs().max(),
                                                                  correlation=np.corrcoef(g.today, g.candidate)[0, 1]))).round(3).to_string())
