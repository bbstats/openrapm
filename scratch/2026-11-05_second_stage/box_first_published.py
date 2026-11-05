"""Read-only: the published tables after the switch against the backup from before it.  With a site folder, also the site's
three files against what main holds.

    python box_first_published.py [SITE]
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
key = ["season", "player_id"]
preview = pd.read_parquet("outputs/season_ratings_product_boxfirst.parquet").set_index(key).sort_index()
for label in ("", "_step3", "_step4"):
    now = pd.read_parquet(f"outputs/season_ratings_product{label}.parquet").set_index(key).sort_index()
    was = pd.read_parquet(f"outputs/season_ratings_product_before_boxfirst{label}.parquet").set_index(key).sort_index()
    same_ids = now.index.equals(was.index)
    numeric = [c for c in was.columns if pd.api.types.is_numeric_dtype(was[c]) and c not in ("box_adj_off", "box_adj_def")]
    gap = (now[numeric] - was[numeric]).abs().max()
    print(f"{label or 'final'}: rows {len(now)}, same players {same_ids}, largest change in any other number {float(gap.max()):.2e}, on {gap.idxmax() if gap.max() > 0 else 'none'}")
now = pd.read_parquet("outputs/season_ratings_product.parquet").set_index(key).sort_index()
for season, rows in ((2027, now.xs(2027, level="season")), (2026, now.xs(2026, level="season"))):
    other = preview.xs(season, level="season").reindex(rows.index)
    print(season, "Box Adj against the preview, largest gap", f"{float(max((rows.box_adj_off - other.box_adj_off).abs().max(), (rows.box_adj_def - other.box_adj_def).abs().max())):.2e}")

if len(sys.argv) > 1:
    site = sys.argv[1]

    def head(file):
        return json.loads(subprocess.run(["git", "-C", site, "show", f"HEAD:{file}"], capture_output=True, text=True, encoding="utf-8", check=True).stdout)

    for file in ("docs/data/ratings.json", "docs/data/power.json", "docs/data/decompose_openrapm.json"):
        new, old = json.load(open(f"{site}/{file}", encoding="utf-8")), head(file)
        new_meta, old_meta = {k: v for k, v in new.get("meta", {}).items() if k != "built"}, {k: v for k, v in old.get("meta", {}).items() if k != "built"}
        print(file, "same beyond the date built:", {**new, "meta": new_meta} == {**old, "meta": old_meta}, "keys", list(new)[:6])
    new, old = json.load(open(f"{site}/docs/data/decompose_openrapm.json", encoding="utf-8")), head("docs/data/decompose_openrapm.json")
    print("meta", {k: (v if not isinstance(v, list) else v[:30]) for k, v in new["meta"].items()})
    print("row keys", {k: type(v).__name__ + (f" {len(v)}" if hasattr(v, "__len__") else "") for k, v in new.items()})
