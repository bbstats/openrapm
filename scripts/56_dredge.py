"""Build the Dredge counter tables: one parquet per season of per-player play-by-play event counts.

Reads `data/raw/pbp/{season}/{game_id}.parquet` for every regular-season game the stints were built
from and writes `data/dredge/{season}_RS.parquet` with, per player: the counters of
`eracoef.dredge.COUNTERS` and his offensive and defensive possessions from the stints.  See
`src/eracoef/dredge.py` for what each counter is and which of them our v3 feed can honestly attribute.

Then prints the two reports that decide whether any of it may become a feature:

  * per season, the league rate of every counter per 100 possessions, so an ERA TREND is visible
    before it is learned as "old era" (`season` is a feature of the prior, and FINDINGS 22.2 is what
    happens when a column names the rows instead of describing them).  Watch `goaltend`, which
    HANDOFF 3.1 flags as disagreeing with Justin Willard's published 1997 count.
  * a cross-check against the BOX SCORE: the play-by-play's made field goals and turnovers, summed
    over the league, against the same totals from the game logs.  These count the same events from
    two different feeds and must agree to a fraction of a percent.

usage: python scripts/56_dredge.py [first] [last] [--force]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.dredge import COUNTERS, season_dredge  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
(OUT / "csv").mkdir(parents=True, exist_ok=True)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FORCE = "--force" in sys.argv
first = int(args[0]) if args else int(cfg["first_season"])
last = int(args[1]) if len(args) > 1 else int(cfg["last_season"])

t0 = time.time()
rows = []
for s in range(first, last + 1):
    d = season_dredge(s, cfg, force=FORCE, verbose=True)
    box = season_box([s], ["RS"], cfg)
    r = dict(season=s, players=len(d), poss_off=float(d.poss_off.sum()), poss_def=float(d.poss_def.sum()))
    per100 = 100.0 / max(float(d.poss_off.sum()), 1.0)
    for c in COUNTERS:
        r[c] = float(d[c].sum()) * per100
    r["unast_share"] = float(d.fgm_unast.sum() / max(d.fgm_unast.sum() + d.fgm_ast.sum(), 1.0))
    r["russell_share"] = float(d.blk_rus.sum() / max(d.blk.sum(), 1.0))
    r["rim_share"] = float(d.blk_rim.sum() / max(d.blk.sum(), 1.0))
    r["three_share"] = float(d.blk_3.sum() / max(d.blk.sum(), 1.0))
    r["stolen_share"] = float(d.tov_stolen.sum() / max(d.tov_all.sum(), 1.0))
    ast_tot = float(d.fgm_ast.sum())                     # one assist per assisted make, by definition
    r["ast_resolved"] = float(d.ast_res.sum()) / max(ast_tot, 1.0)
    zsum = float(d[["ast_rim", "ast_smr", "ast_lmr", "ast_c3", "ast_ab3"]].to_numpy().sum())
    r["ast_located"] = zsum / max(float(d.ast_res.sum()), 1.0)
    for z in ("rim", "smr", "lmr", "c3", "ab3"):
        r[f"ast_{z}_sh"] = float(d[f"ast_{z}"].sum()) / max(zsum, 1.0)
    r["blk_smr_sh"] = float(d.blk_smr.sum()) / max(float(d.blk.sum()), 1.0)
    r["blk_lmr_sh"] = float(d.blk_lmr.sum()) / max(float(d.blk.sum()), 1.0)
    r["pbp_ast"] = ast_tot
    r["box_ast"] = float(box.ast.sum())
    # the box-score cross-check: the same events, counted from the game logs instead
    r["pbp_fgm"] = float(d.fgm_unast.sum() + d.fgm_ast.sum())
    r["box_fgm"] = float((box.fg2m + box.fg3m).sum())
    r["pbp_tov"] = float(d.tov_all.sum())
    r["box_tov"] = float(box.tov.sum())
    r["pbp_blk"] = float(d.blk.sum())
    r["box_blk"] = float(box.blk.sum())
    r["pbp_stl"] = float(d.stl.sum())
    r["box_stl"] = float(box.stl.sum())
    rows.append(r)
R = pd.DataFrame(rows)

print("\n=== league rate per 100 possessions, per season (an era trend here is a warning, not a feature)")
print(R[["season", "players", *COUNTERS]].to_string(index=False))

print("\n=== the ASSIST attribution: what fraction we could name a passer for (a surname in the "
      "SHOOTER's row, not a personId), and where the shots they created came from")
print(R[["season", "ast_resolved", "ast_located", "ast_rim_sh", "ast_smr_sh", "ast_lmr_sh",
         "ast_c3_sh", "ast_ab3_sh", "blk_smr_sh", "blk_lmr_sh"]].to_string(index=False))

print("\n=== the shares the features are really made of")
print(R[["season", "unast_share", "russell_share", "rim_share", "three_share", "stolen_share"]].to_string(index=False))

print("\n=== cross-check against the box score (the same events from the game logs; ratios must be ~1)")
chk = R[["season"]].copy()
for c in ("fgm", "tov", "blk", "stl", "ast"):
    chk[c] = R[f"pbp_{c}"] / R[f"box_{c}"].replace(0, np.nan)
print(chk.to_string(index=False))
bad = chk[[c for c in ("fgm", "tov", "blk", "stl", "ast")]].sub(1.0).abs().max()
print("\nworst |ratio - 1| per counter:", {c: f"{v:.4f}" for c, v in bad.items()})

R.to_csv(OUT / "csv" / "dredge_seasons.csv", index=False)
print(f"\nwrote data/dredge/{{{first}..{last}}}_RS.parquet and outputs/csv/dredge_seasons.csv "
      f"({time.time() - t0:.0f}s)")
