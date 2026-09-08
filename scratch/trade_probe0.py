"""Trade calibration, probe 0: what the panel and the roles table already know about movers.

Prints the panel's columns, then per season how many players' main team changed since the previous
season, and how much of the league's possessions they carry.  Read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

panel = pd.read_parquet(ROOT / "outputs" / "role_panel.parquet")
print("panel:", panel.shape)
print("columns:", list(panel.columns))
print(panel.head(3).T.to_string())
print()
print("rows per side:", panel.side.value_counts().to_dict())
print("windows:", sorted(panel.window.unique()))

roles = pd.read_parquet(ROOT / "data" / "cache" / "roles.parquet")
print("\nroles:", roles.shape, list(roles.columns))
# main team per player-season = the team with the most on-floor possessions
r = roles.sort_values("poss_on", ascending=False).drop_duplicates(["player_id", "season"])
r = r[["player_id", "season", "team_id", "poss_on"]].sort_values(["player_id", "season"])
tot = roles.groupby(["player_id", "season"], as_index=False).agg(poss_all=("poss_on", "sum"),
                                                                 n_teams=("team_id", "nunique"))
r = r.merge(tot, on=["player_id", "season"])
r["prev_team"] = r.groupby("player_id")["team_id"].shift(1)
r["prev_season"] = r.groupby("player_id")["season"].shift(1)
r["moved"] = (r.prev_team.notna()) & (r.prev_team != r.team_id) & (r.prev_season == r.season - 1)
r["returning"] = r.prev_season == r.season - 1
r["midseason"] = r.n_teams > 1
lg = r.groupby("season").agg(players=("player_id", "size"),
                              returning=("returning", "sum"),
                              moved=("moved", "sum"),
                              midseason=("midseason", "sum"),
                              poss_all=("poss_all", "sum"))
mv = r[r.moved].groupby("season")["poss_all"].sum()
lg["moved_poss_share"] = (mv / lg.poss_all).round(3)
lg["moved_frac_of_returning"] = (lg.moved / lg.returning).round(3)
print("\nper season (main team = most on-floor possessions; moved = main team differs from last season's):")
print(lg.drop(columns="poss_all").to_string())
print("\nover all seasons: returning players", int(lg.returning.sum()), "of whom moved", int(lg.moved.sum()),
      f"({lg.moved.sum() / lg.returning.sum():.1%}); mid-season multi-team player-seasons", int(lg.midseason.sum()))
