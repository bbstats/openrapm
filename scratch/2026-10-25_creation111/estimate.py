"""Experiment 111's estimate for every season, and its check against the NBA's potential assists.

For each player-season, from the play-by-play alone (data/shotframe, the free-throw trips of trips.py):
    misses off a pass   sum over his misses of p(cell) moved by the cell's miss factor (miss_factor.py), times his
                        lean in that shot's spot group: inside (rim, paint), other twos, threes, where
                        lean = (assisted makes + 10) / (expected assisted makes + 10), expected from p(cell) of his
                        makes there; each shot's chance held at 1
    trips off a pass    and-ones: the make's own assist flag; every other trip: his inside lean x the season's mean
                        chance a rim miss came off a pass x one league factor, fitted so that the trips off a pass
                        with a free throw made add up to the NBA's free-throw assists over 2014-2017
Writes outputs/creation111/players.parquet and prints the team check: within each season, the correlation across
teams of each estimate with the NBA's count, potential assists minus assists for the misses and free-throw assists
for the trips, by era of the tracking provider.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cells import CELL_PAD, cell_rates, expit, load_shots, logit

SEASONS = range(1997, 2027)
LEAN_PAD = 10.0
GROUP = {"rim": "inside", "paint": "inside", "mid": "two", "long_two": "two", "corner_three": "three",
         "above_break_three": "three"}
FT_FIT = range(2014, 2018)          # the free-throw factor's seasons: before the provider changed


def season_rows(season: int, prior: pd.Series, factor: dict) -> tuple[pd.DataFrame, float]:
    """(one row per player, team and phase: his counts and estimates; the season's rim base)."""
    shots = load_shots(season)
    rate = cell_rates(shots[shots.made == 1], prior)
    shots["p_cell"] = shots.cell.map(rate).fillna(prior.mean())
    shots["group"] = shots.spot.map(GROUP)
    made = shots[shots.made == 1]
    lean = made.groupby(["shooter", "group"]).agg(a=("post_assisted", "sum"), e=("p_cell", "sum"))
    lean = (lean.a + LEAN_PAD) / (lean.e + LEAN_PAD)
    miss = shots[shots.made == 0].copy()
    miss["p_miss"] = expit(logit(miss.p_cell) + miss.cell.map(factor).fillna(0.0))
    key = pd.MultiIndex.from_arrays([miss.shooter, miss.group])
    miss["lean"] = lean.reindex(key).fillna(1.0).to_numpy()
    miss["passed"] = (miss.p_miss * miss.lean).clip(upper=1.0)
    rim_base = float(miss.loc[miss.spot == "rim", "p_miss"].mean())

    trips = pd.read_parquet(f"outputs/creation111/trips_{season}.parquet")
    phase_team = shots.groupby(["game_id", "shooter"]).team.first()
    trips["team"] = phase_team.reindex(pd.MultiIndex.from_arrays([trips.game_id, trips.player_id])).to_numpy()
    by = ["shooter", "team", "phase"]
    out = made.groupby(by).agg(makes=("made", "size"), assisted_makes=("post_assisted", "sum"),
                               and_ones=("post_and1", "sum"))
    out["assisted_and_ones"] = (made.post_and1 * made.post_assisted).groupby([made.shooter, made.team,
                                                                              made.phase]).sum()
    out = out.join(miss.groupby(by).agg(misses=("made", "size"), passed_misses=("passed", "sum"),
                                        cells_misses=("p_miss", "sum")), how="outer")
    t = trips.dropna(subset=["team"]).astype({"team": int}).rename(columns={"player_id": "shooter"})
    out = out.join(t.groupby(by).agg(trips=("trip", "sum"), technical=("technical", "sum")), how="outer")
    out = out.fillna(0.0).reset_index()
    out["lean_inside"] = lean.xs("inside", level="group").reindex(out.shooter).fillna(1.0).to_numpy()
    out["season"] = season
    return out, rim_base


def main() -> None:
    factor = pd.read_parquet("outputs/creation111/miss_factor.parquet").set_index("cell").factor.to_dict()
    pooled = []
    for season in SEASONS:
        s = load_shots(season)
        pooled.append(s[s.made == 1].groupby("cell").post_assisted.agg(["sum", "size"]))
    pooled = pd.concat(pooled).groupby(level=0).sum()
    prior = pooled["sum"] / pooled["size"]

    frames, bases = [], {}
    for season in SEASONS:
        rows, bases[season] = season_rows(season, prior, factor)
        frames.append(rows)
        print(f"  {season}: {len(rows)} player-team rows, rim base {bases[season]:.3f}", flush=True)
    players = pd.concat(frames, ignore_index=True)
    players["rim_base"] = players.season.map(bases)

    panel = pd.read_parquet("outputs/season_panel.parquet")
    o = panel[panel.side == "O"].set_index(["player_id", "season"])
    fgm = o.twos_made_per100_uncentred + o.threes_made_per100_uncentred
    owner = np.clip(0.73 - 0.03 * o.assists_per100_uncentred - 0.01 * fgm, 0, 1)
    ft_pct = o.free_throws_made_per100_uncentred / (o.free_throws_made_per100_uncentred
                                                     + o.free_throws_missed_per100_uncentred)
    key = pd.MultiIndex.from_arrays([players.shooter, players.season])
    players["owner_share"] = owner.reindex(key).to_numpy()
    players["ft_pct"] = ft_pct.reindex(key).fillna(0.75).to_numpy()
    players["other_trips"] = (players.trips - players.and_ones).clip(lower=0)
    one_made = 1 - (1 - players.ft_pct) ** 2

    # the free-throw factor: the NBA's free-throw assists over 2014-2017 against the trips' base
    nba = []
    for season in range(2014, 2027):
        p = pd.read_parquet(f"data/raw/dashboards/tracking/pt_passing/{season}_RS.parquet")
        nba.append(p.groupby("TEAM_ID")[["AST", "POTENTIAL_AST", "FT_AST"]].sum().assign(season=season))
    nba = pd.concat(nba).reset_index().rename(columns={"TEAM_ID": "team"})
    base = players.other_trips * players.lean_inside * players.rim_base * one_made
    fit = players.season.isin(FT_FIT) & (players.phase == "RS")      # the NBA's counts are the regular season's
    ft_factor = nba[nba.season.isin(FT_FIT)].FT_AST.sum() / base[fit].sum()
    print(f"\nfree-throw factor {ft_factor:.3f}: a non-and-one trip comes off a pass "
          f"{(players.lean_inside * players.rim_base * ft_factor)[fit].mean():.3f} of the time on average, 2014-2017")
    players["passed_trips"] = players.assisted_and_ones + (players.other_trips * (players.lean_inside * players.rim_base
                                                                                  * ft_factor).clip(upper=1))
    players.to_parquet("outputs/creation111/players.parquet", index=False)
    print(f"wrote outputs/creation111/players.parquet: {len(players)} rows")


if __name__ == "__main__":
    main()
