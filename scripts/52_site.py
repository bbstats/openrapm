"""OpenRAPM: the one-page site.  Exports the board to docs/data/ratings.json for docs/index.html.

One table, one row per player per SEASON, offense and defense in points per 100 possessions with
positive good on both ends.  A season's rating is fit on that season's games -- regular season and
playoffs together -- so the latest row is the season in progress and re-running this updates it.

Reads outputs/season_ratings.parquet (scripts/60_season_board.py), falling back to the shipped copy
in artifacts/ so that a fresh clone can build the page without refitting anything.

usage: python scripts/52_site.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402

cfg = load_config()
root = Path(cfg["_root"])

CANDIDATES = [root / "outputs" / "season_ratings.parquet", root / "artifacts" / "season_ratings.parquet"]
src = next((p for p in CANDIDATES if p.exists()), None)
if src is None:
    raise SystemExit("no season board found.  Run `python scripts/60_season_board.py` first, or "
                     "restore artifacts/season_ratings.parquet.")

rat = pd.read_parquet(src)
cols = {"season": "s", "player_name": "n", "rating_off": "o", "rating_def": "d",
        "rating_total": "t", "poss_season": "p"}
missing = [c for c in cols if c not in rat.columns]
if missing:
    raise SystemExit(f"{src.name} is missing {missing}; rebuild it with scripts/60_season_board.py")

d = rat[list(cols)].rename(columns=cols).copy()
d["n"] = d["n"].fillna("").astype(str)
d = d[d.p > 0].sort_values(["s", "t"], ascending=[True, False])
rows = [dict(s=int(r.s), n=r.n, o=round(float(r.o), 2), d=round(float(r.d), 2),
             t=round(float(r.t), 2), p=int(r.p))
        for r in d.itertuples(index=False)]

meta = dict(seasons=sorted(int(s) for s in d.s.unique()),
            built=pd.Timestamp.now("UTC").strftime("%Y-%m-%d"),
            n_players=int(rat.player_id.nunique()) if "player_id" in rat.columns else 0)

out = root / "docs" / "data"
out.mkdir(parents=True, exist_ok=True)
path = out / "ratings.json"
path.write_text(json.dumps(dict(meta=meta, rows=rows), separators=(",", ":")), encoding="utf-8")
print(f"wrote {path.relative_to(root)}: {len(rows)} rows over {len(meta['seasons'])} seasons "
      f"({meta['seasons'][0]}-{meta['seasons'][-1]}), {path.stat().st_size / 1e6:.1f} MB, from {src.name}")
