"""Experiment 111: each estimate summed by team against the NBA's own counts, regular season 2014-2026.

Misses: potential assists minus assists, the NBA's misses that came off a pass.  Free throws: free-throw assists,
trips off a pass with a free throw made, and-ones out (the make's assist is an assist).  Each NBA count sits on the
passer's last team of the season, each estimate on the shooter's team at the time.  The reading: the correlation
across the 30 teams within each season, averaged over the seasons of each tracking provider, and the ratio of the
league totals, estimate over the NBA's.
"""
import numpy as np
import pandas as pd

players = pd.read_parquet("outputs/creation111/players.parquet")
rs = players[(players.phase == "RS") & (players.season >= 2014)].copy()
season_share = players.groupby(["shooter", "season"]).apply(
    lambda g: g.assisted_makes.sum() / max(g.makes.sum(), 1), include_groups=False)
rs["makes_share"] = season_share.reindex(pd.MultiIndex.from_arrays([rs.shooter, rs.season])).to_numpy()
rs["owner_share"] = rs.owner_share.fillna(rs.makes_share)
one_made = 1 - (1 - rs.ft_pct) ** 2
rs["new_trips"] = (rs.passed_trips - rs.assisted_and_ones) * one_made
estimates = {
    "misses": {"owner's formula": rs.owner_share * rs.misses, "his makes' assisted share": rs.makes_share * rs.misses,
               "cells alone": rs.cells_misses, "new estimate": rs.passed_misses},
    "trips": {"owner's formula": rs.owner_share * rs.other_trips * one_made,
              "his makes' assisted share": rs.makes_share * rs.other_trips * one_made,
              "the same share for everyone": rs.other_trips * one_made, "new estimate": rs.new_trips}}
nba = []
for season in range(2014, 2027):
    p = pd.read_parquet(f"data/raw/dashboards/tracking/pt_passing/{season}_RS.parquet")
    nba.append(p.groupby("TEAM_ID")[["AST", "POTENTIAL_AST", "FT_AST"]].sum().assign(season=season))
nba = pd.concat(nba).reset_index().rename(columns={"TEAM_ID": "team"})
nba["misses"], nba["trips"] = nba.POTENTIAL_AST - nba.AST, nba.FT_AST

rows = []
for kind, table in estimates.items():
    for name, value in table.items():
        team = rs.assign(value=value).groupby(["season", "team"]).value.sum().rename("estimate").reset_index()
        both = team.merge(nba[["season", "team", kind]], on=["season", "team"])
        for era, seasons in (("2014-2017", range(2014, 2018)), ("2018-2026", [s for s in range(2018, 2027)
                                                                              if not (kind == "trips" and s == 2025)])):
            part = both[both.season.isin(seasons)]
            r = part.groupby("season").apply(lambda g: np.corrcoef(g.estimate, g[kind])[0, 1], include_groups=False)
            rows.append({"count": kind, "estimate": name, "seasons": era, "mean_correlation": r.mean(),
                         "lowest_season": r.min(), "estimate_over_nba": part.estimate.sum() / part[kind].sum()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
