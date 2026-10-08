"""OpenRAPM: the one-page site.  Exports the board to docs/data/ratings.json for docs/index.html.

One table, one row per player per SEASON, offense and defense in points per 100 possessions with
positive good on both ends.  A season's rating is fit on that season's games -- regular season and
playoffs together -- so the latest row is the season in progress and re-running this updates it.

Reads outputs/season_ratings.parquet (scripts/60_season_board.py), falling back to the shipped copy
in artifacts/ so that a fresh clone can build the page without refitting anything.  `OPENRAPM_BOARD`
overrides both -- the same convention tests/test_vs_consensus.py uses -- so a candidate board can be
published without overwriting the shipped artifact the rest of the scripts read.

usage: python scripts/52_site.py [--portable=0]
       OPENRAPM_BOARD=outputs/season_ratings_sy.parquet python scripts/52_site.py
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config  # noqa: E402

check_flags()

cfg = load_config()
root = Path(cfg["_root"])

# First path that exists wins, so the order IS the decision about what gets published.  The product
# table goes first because it is what ships; `artifacts/season_ratings.parquet` is the older
# scripts/60 system, kept for the tests, and publishing it would quietly replace the live rankings
# with a superseded set (it rates Jokic 4th in 2026 where the shipped table has him 2nd).
CANDIDATES = ([Path(os.environ["OPENRAPM_BOARD"])] if os.environ.get("OPENRAPM_BOARD") else
              [root / "outputs" / "season_ratings_product.parquet",
               root / "outputs" / "season_ratings.parquet",
               root / "artifacts" / "season_ratings.parquet"])
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

# The portable rating (experiment 40): the box-prior part kept and the part beyond it (games part + swap adjustment)
# times one ratio fitted on players who changed teams; scripts/112_portable.py writes the ratio, and it is applied here
# to the table being published, so the column cannot fall out of step with the ratings beside it.
portable_json = root / "outputs" / "portable" / "production.json"
has_parts = {"prior_off", "u_off", "c_off", "prior_def", "u_def", "c_def"} <= set(rat.columns)
# --portable=0 (2026-10-08): leave the column out.  The LightGBM ratings of experiment 45 were published without it:
# the owner had not approved it for the site, and its ratio was fitted on the earlier ratings.
with_portable = flag("portable", "1") not in ("0", "no", "false")
if with_portable and portable_json.exists() and has_parts:
    from eracoef.portable import portable_table  # noqa: E402
    ratio_beyond = float(json.loads(portable_json.read_text(encoding="utf-8"))["ratio_beyond"])
    port = portable_table(rat, ratio_beyond)
    rat = rat.assign(portable_off=port.portable_off.to_numpy(), portable_def=port.portable_def.to_numpy(),
                     portable_total=port.portable_total.to_numpy())
    cols.update(portable_off="po", portable_def="pd", portable_total="pt")
else:
    ratio_beyond = None

d = rat[list(cols)].rename(columns=cols).copy()
d["id"] = rat["player_id"].to_numpy() if "player_id" in rat.columns else 0   # the NBA id: in the file and the
d["n"] = d["n"].fillna("").astype(str)                                         # CSV download, not on the page
d = d[d.p > 0].sort_values(["s", "t"], ascending=[True, False])
rows = [dict(s=int(r.s), n=r.n, o=round(float(r.o), 2), d=round(float(r.d), 2),
             t=round(float(r.t), 2), p=int(r.p), id=int(r.id),
             **(dict(po=round(float(r.po), 2), pd=round(float(r.pd), 2), pt=round(float(r.pt), 2))
                if ratio_beyond is not None else {}))
        for r in d.itertuples(index=False)]

# "Built" is when the ratings were built -- the source table's date -- so re-exporting the same table (to add a
# field, say) does not tell readers the ratings changed
meta = dict(seasons=sorted(int(s) for s in d.s.unique()),
            built=pd.Timestamp(src.stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%d"),
            n_players=int(rat.player_id.nunique()) if "player_id" in rat.columns else 0)
if ratio_beyond is not None:
    meta["portable_ratio"] = round(ratio_beyond, 2)

out = root / "docs" / "data"
out.mkdir(parents=True, exist_ok=True)
path = out / "ratings.json"
path.write_text(json.dumps(dict(meta=meta, rows=rows), separators=(",", ":")), encoding="utf-8")
print(f"wrote {path.relative_to(root)}: {len(rows)} rows over {len(meta['seasons'])} seasons "
      f"({meta['seasons'][0]}-{meta['seasons'][-1]}), {path.stat().st_size / 1e6:.1f} MB, "
      f"from {src.relative_to(root) if src.is_relative_to(root) else src}")
