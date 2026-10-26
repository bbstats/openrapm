"""Experiment 112: each candidate's split-half reliability, odd against even games within a season.

Per player and season, every count of the candidate summed over his odd-numbered and his even-numbered games
(regular season and playoffs, in date order of his games), each half's rate per 36 minutes, and the correlation of
the two halves over the players with 1,000+ minutes, in one season of each era: 2002, 2012, 2022.  Per minute, not per
100 possessions, so the halves need only the box score; the candidate made from several counts (potential assists,
the passes-estimated self-creation) is built from each half's per-minute counts with the panel's formula.
"""
import numpy as np
import pandas as pd

from openrapm.data.box_scores import season_box
from openrapm.data.config import load_config
from openrapm.features import (CREATION_ESTIMATED, CREATION_GROUPS, DERIVED, SCREEN_CANDIDATES, add_features)

SEASONS = (2002, 2012, 2022)
MIN_MINUTES = 1000.0


def halves(season: int, cfg: dict) -> pd.DataFrame:
    box = season_box([season], ["RS", "PO"], cfg)
    box = box[box.minutes > 0].sort_values(["player_id", "game_id"])
    box["half"] = box.groupby("player_id").cumcount() % 2
    events = pd.read_parquet(f"data/event_counts/{season}.parquet")
    creation = pd.read_parquet(f"data/event_counts/{season}_creation.parquet")
    keys = ["game_id", "player_id"]
    for table in (events, creation):
        table["game_id"] = table.game_id.astype(str)
    counts = (box.merge(events.drop(columns=["season", "phase"]), on=keys, how="left")
              .merge(creation.drop(columns=["season", "phase"]), on=keys, how="left").fillna(0.0))
    numeric = [c for c in counts.columns if c not in ("game_id", "player_id", "team_id", "player_name", "season",
                                                      "phase", "half")]
    summed = counts.groupby(["player_id", "half"])[numeric].sum()
    return summed[summed.groupby(level="player_id").minutes.transform("sum") >= MIN_MINUTES]


def candidate_values(rates: pd.DataFrame) -> pd.DataFrame:
    """Each candidate from one half's per-36 counts, named as the prior reads it."""
    out = pd.DataFrame(index=rates.index)
    out["potential_assists_estimated_per100"] = sum(weight * rates[rate.removesuffix("_per100")]
                                                    for rate, weight in DERIVED[
                                                        "potential_assists_estimated_per100"].items())
    for name in ("bad_pass_steals", "lost_ball_steals", "russell_blocks", "trips_without_make", "late_clock_attempts",
                 "transition_attempts", "putback_attempts", "and_ones",
                 *(f"unassisted_makes_{g}" for g in CREATION_GROUPS)):
        out[f"{name}_per100"] = rates[name]
    # the passes-estimated self-creation through the panel's own formula, the per-36 counts in place of per 100
    frame = pd.DataFrame(index=rates.index)
    for name in ("unassisted_points", "unassisted_threes_made"):
        frame[f"{name}_per100_uncentred"] = rates[name]
    frame["twos_made_per100_uncentred"], frame["threes_made_per100_uncentred"] = rates.fg2m, rates.fg3m
    for group in CREATION_GROUPS:
        for stem in ("assisted_makes", "expected_assisted_makes", "misses", "expected_passed_misses"):
            frame[f"{stem}_{group}_per100_uncentred"] = rates[f"{stem}_{group}"]
    frame["trips_without_make_per100_uncentred"] = rates.trips_without_make
    frame["expected_passed_trips_per100_uncentred"] = rates.expected_passed_trips
    out[CREATION_ESTIMATED[0]] = add_features(frame, CREATION_ESTIMATED)[CREATION_ESTIMATED[0]]
    return out


def main() -> None:
    cfg = load_config()
    names = [*SCREEN_CANDIDATES["O"], *SCREEN_CANDIDATES["D"]]
    rows = []
    for season in SEASONS:
        summed = halves(season, cfg)
        rates = summed.div(summed.minutes, axis=0) * 36.0
        values = candidate_values(rates)
        first, second = values.xs(0, level="half"), values.xs(1, level="half")
        both = first.index.intersection(second.index)
        for name in names:
            rows.append({"input": name, "season": season, "players": len(both),
                         "reliability": float(np.corrcoef(first.loc[both, name], second.loc[both, name])[0, 1])})
        print(f"  {season}: {len(both)} players", flush=True)
    table = pd.DataFrame(rows).pivot(index="input", columns="season", values="reliability").reindex(names)
    table.columns = [f"reliability_{season}" for season in table.columns]
    print(table.round(3).to_string())
    table.to_csv("outputs/csv/screen112_reliability.csv")


if __name__ == "__main__":
    main()
