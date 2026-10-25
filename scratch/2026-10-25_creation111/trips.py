"""Free-throw trips per game, player and phase from the raw play-by-play: the first free throw of each trip
("Free Throw 1 of 1", "1 of 2", "1 of 3", flagrant ones included), technical free throws counted apart.
Writes outputs/creation111/trips_<season>.parquet.  Usage: trips.py FIRST LAST"""
import sys
from pathlib import Path

import pandas as pd

first, last = int(sys.argv[1]), int(sys.argv[2])
for season in range(first, last + 1):
    rows = []
    for path in sorted(Path(f"data/raw/pbp/{season}").glob("*.parquet")):
        game = path.stem
        if game[2] not in "24":          # 002 regular season, 004 playoffs
            continue
        p = pd.read_parquet(path, columns=["personId", "actionType", "subType"])
        ft = p[p.actionType == "Free Throw"]
        sub = ft.subType.astype(str)
        trip = sub.str.contains(r"1 of [123]$") & ~sub.str.contains("Technical")
        tech = sub.str.contains("Technical")
        out = pd.DataFrame({"player_id": ft.personId.astype(int), "trip": trip.astype(int), "technical": tech.astype(int)})
        out = out.groupby("player_id", as_index=False).sum()
        out["game_id"], out["phase"] = game, "RS" if game[2] == "2" else "PO"
        rows.append(out)
    table = pd.concat(rows, ignore_index=True)
    table.to_parquet(f"outputs/creation111/trips_{season}.parquet", index=False)
    print(season, len(table), int(table.trip.sum()), int(table.technical.sum()), flush=True)
