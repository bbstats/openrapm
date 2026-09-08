"""Trade calibration, probe 1: schemas of the stints, the gamelog table, the xrapm panel and a track dump."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.ingest import game_table, load_gamelog  # noqa: E402

cfg = load_config()
st = pd.read_parquet(ROOT / "data" / "stints" / "2015_RS.parquet")
print("stints 2015:", st.shape)
print("columns (first 40):", list(st.columns)[:40])
print(st[[c for c in st.columns if c in ("game_id", "h1", "a1", "poss_h", "poss_a", "period", "season")]].head(3).to_string())
g = game_table(load_gamelog(2015, "RS", cfg))
print("\ngame table:", g.shape, list(g.columns))
print(g.head(3).to_string())
xp = ROOT / "outputs" / "xrapm_panel.parquet"
if xp.exists():
    x = pd.read_parquet(xp)
    print("\nxrapm_panel:", x.shape, list(x.columns)[:30])
for name in ("ratings_track_tune501_b7.parquet", "holdout_track_tune501_b7.parquet", "calmap_track_tune501_b7.parquet"):
    p = ROOT / "outputs" / name
    if p.exists():
        d = pd.read_parquet(p)
        print(f"\n{name}: {d.shape}", list(d.columns)[:24])
        print(d.head(2).to_string())
