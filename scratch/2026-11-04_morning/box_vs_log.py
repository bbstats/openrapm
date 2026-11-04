"""Every 2027 player-game: the fresh game log against the box score stored when the game was first fetched."""
import glob
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
log = pd.read_parquet("data/raw/gamelog/2027_RS.parquet")
box = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/raw/box/2027/*.parquet"))], ignore_index=True)
pairs = {"FGM": "fieldGoalsMade", "FGA": "fieldGoalsAttempted", "FG3M": "threePointersMade", "FG3A": "threePointersAttempted",
         "FTM": "freeThrowsMade", "FTA": "freeThrowsAttempted", "OREB": "reboundsOffensive", "DREB": "reboundsDefensive",
         "AST": "assists", "STL": "steals", "BLK": "blocks", "TOV": "turnovers", "PF": "foulsPersonal", "PTS": "points",
         "PLUS_MINUS": "plusMinusPoints"}
box["GAME_ID"] = box.gameId.astype(str).str.zfill(10)
log["GAME_ID"] = log.GAME_ID.astype(str).str.zfill(10)
both = log.merge(box, left_on=["GAME_ID", "PLAYER_ID"], right_on=["GAME_ID", "personId"], how="left", indicator=True)
print("game-log rows", len(log), "matched", int((both._merge == "both").sum()))
both = both[both._merge == "both"]
rows = []
for a, b in pairs.items():
    differ = both[pd.to_numeric(both[a], errors="coerce").fillna(0) != pd.to_numeric(both[b], errors="coerce").fillna(0)]
    for r in differ.itertuples():
        rows.append((r.GAME_ID, str(r.GAME_DATE)[:10], r.MATCHUP, r.PLAYER_NAME, a, getattr(r, b), getattr(r, a)))
out = pd.DataFrame(rows, columns=["game", "date", "matchup", "player", "stat", "stored_box", "game_log_now"])
print("differences:", len(out))
print(out.sort_values(["date", "game", "player"]).to_string(index=False))
