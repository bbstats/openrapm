"""Read-only probe, 2026-10-30: every 2027 hustle box score asked again (not cached) and summed beside the cached files
and the season dashboard.  Question: do the per-game box scores fill in over the days after a game?"""
import glob
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import BoxScoreHustleV2

from openrapm.data.dashboards import HUSTLE_GAME_COLUMNS
from openrapm.data.config import load_config
from openrapm.data.ingest import game_table, load_gamelog

cfg = load_config()
dates = game_table(load_gamelog(2027, "RS", cfg)).set_index("game_id")["game_date"]
rows = []
for path in sorted(glob.glob("data/raw/dashboards/hustle/games/2027/*.parquet")):
    game = Path(path).stem
    cached = pd.read_parquet(path)
    fresh = BoxScoreHustleV2(game_id=game, timeout=30).get_data_frames()[0]
    for label, frame in (("cached", cached), ("fresh", fresh)):
        frame = frame[pd.to_numeric(frame.personId) > 0]
        rows.append(dict(game=game, date=str(dates.get(game, ""))[:10], copy=label,
                         **{k: pd.to_numeric(frame[v]).sum() for k, v in HUSTLE_GAME_COLUMNS.items()}))
    time.sleep(0.4)
d = pd.DataFrame(rows)
d.to_csv("scratch/2026-10-30_bugfix/hustle_refetch_probe.csv", index=False)
season = pd.read_parquet("data/raw/dashboards/hustle/players/2027_RS.parquet")
print("totals over 71 games:")
for k in HUSTLE_GAME_COLUMNS:
    print(f"  {k:24s} cached {d[d['copy'] == 'cached'][k].sum():6.0f}  fresh {d[d['copy'] == 'fresh'][k].sum():6.0f}  "
          f"season dashboard {pd.to_numeric(season[k]).sum():6.0f}")
fresh = d[d['copy'] == 'fresh'].groupby("date")[["DEFLECTIONS", "DEF_BOXOUTS", "SCREEN_AST_PTS"]].mean().round(1)
print("fresh copies, per game, by game date:")
print(fresh.to_string())
