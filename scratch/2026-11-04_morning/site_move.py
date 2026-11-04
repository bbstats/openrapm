"""How far each 2027 row moved between two commits of the site, and whether its counts moved."""
import json
import subprocess
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SITE = "A:/code/spmm_site"


def rows(commit):
    raw = subprocess.run(["git", "-C", SITE, "show", f"{commit}:docs/data/ratings.json"], capture_output=True, check=True).stdout
    doc = json.loads(raw.decode("utf-8"))
    return doc["meta"], pd.DataFrame(doc["rows"])


meta_a, a = rows(sys.argv[1])
meta_b, b = rows(sys.argv[2])
print("meta keys", {k: (meta_a.get(k), meta_b.get(k)) for k in meta_b if meta_a.get(k) != meta_b.get(k)})
print("columns", list(a.columns))
key = [c for c in a.columns if c in ("id", "pid", "player_id", "s", "season", "y")]
print("key", key)
both = a.merge(b, on=key, suffixes=("_a", "_b"), how="outer", indicator=True)
print(both._merge.value_counts().to_dict())
both = both[both._merge == "both"]
for c in a.columns:
    if c in key:
        continue
    x, y = both[f"{c}_a"], both[f"{c}_b"]
    differ = (x != y) & ~(x.isna() & y.isna())
    if differ.any():
        if pd.api.types.is_numeric_dtype(x):
            d = (y - x)[differ].abs()
            print(f"{c}: {differ.sum()} rows differ, largest {d.max():.3f}, seasons {sorted(both.loc[differ, key[-1]].unique())[:4]}")
        else:
            print(f"{c}: {differ.sum()} rows differ, for example", both.loc[differ, [f'{c}_a', f'{c}_b']].head(3).values.tolist())
