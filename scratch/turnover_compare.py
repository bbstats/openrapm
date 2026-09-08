"""The board at target_pct_new_teammates = 0 (every teammate familiar) and = 1 (every teammate new), and the
difference: the offensive prior (the SPM), the raw and mapped ratings, for one window.  Not the actual turnover
of anyone -- the same context for everyone, twice.

    python scratch/turnover_compare.py [window=2024-2026]

Rebuilds the board twice through scripts/08_ratings.py with config gbdt_turn.ref set to 0 and 1, restores
config.yaml and the shipped board afterwards, and writes
outputs/csv/target_pct_new_teammates_<window>.csv (one row per player, sorted by the total's delta).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

WIN = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "2024-2026"
CFG = Path("config.yaml")
BOARD = Path("outputs/player_ratings.parquet")
BAK_CFG, BAK_BOARD = Path("scratch/config_before_turncmp.yaml"), Path("scratch/board_before_turncmp.parquet")
shutil.copy(CFG, BAK_CFG)
shutil.copy(BOARD, BAK_BOARD)
py = sys.executable
boards = {}
try:
    for ref in (0.0, 1.0):
        s = BAK_CFG.read_text(encoding="utf-8")
        s2 = re.sub(r"^  gbdt_turn: .*$", f"  gbdt_turn: {{sides: [O], ref: {ref:g}}}", s, flags=re.M)
        assert s2 != s, "config.yaml has no gbdt_turn line"
        CFG.write_text(s2, encoding="utf-8")
        r = subprocess.run([py, "scripts/08_ratings.py"], stdout=open(f"scratch/ratings_turn{ref:g}.log", "w"),
                           stderr=subprocess.STDOUT)
        assert r.returncode == 0, f"08_ratings failed at ref {ref}; see scratch/ratings_turn{ref:g}.log"
        b = pd.read_parquet(BOARD)
        boards[ref] = b[b.window == WIN].copy()
        print(f"  ref {ref:g}: {len(boards[ref])} players in {WIN}", flush=True)
finally:
    shutil.copy(BAK_CFG, CFG)
    shutil.copy(BAK_BOARD, BOARD)
    print("config.yaml and outputs/player_ratings.parquet restored")

cols = ["prior_off", "prior_def", "rating_off_raw", "rating_def_raw", "rating_off", "rating_def", "rating_total"]
a = boards[0.0].set_index("player_id")
b = boards[1.0].set_index("player_id")
out = pd.DataFrame({"player_name": a.player_name, "poss": a.poss_off})
for c in cols:
    out[f"{c}_new0"] = a[c]
    out[f"{c}_new1"] = b[c]
    out[f"{c}_delta"] = b[c] - a[c]
out = out.sort_values("rating_total_delta", key=lambda s: s.abs(), ascending=False).reset_index()
path = Path("outputs/csv") / f"target_pct_new_teammates_{WIN}.csv"
out.to_csv(path, index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.precision", 2)
print(f"\nwrote {path}: {len(out)} players.  Deltas are (all new teammates) minus (all familiar).")
q = out[out.poss >= 1000]
print(f"players with 1000+ possessions: prior_off delta mean {q.prior_off_delta.mean():+.3f}, sd {q.prior_off_delta.std():.3f}; "
      f"rating_off (mapped) delta mean {q.rating_off_delta.mean():+.3f}, sd {q.rating_off_delta.std():.3f}; "
      f"prior_def delta max |{q.prior_def_delta.abs().max():.4f}| (defense does not carry the feature)")
show = ["player_name", "poss", "prior_off_new0", "prior_off_new1", "prior_off_delta", "rating_off_new0", "rating_off_new1",
        "rating_off_delta", "rating_total_new0", "rating_total_new1", "rating_total_delta"]
print("\nthe 15 most HURT by new teammates (1000+ possessions):")
print(q.sort_values("rating_total_delta").head(15)[show].to_string(index=False))
print("\nthe 15 most HELPED by new teammates:")
print(q.sort_values("rating_total_delta").tail(15).iloc[::-1][show].to_string(index=False))
