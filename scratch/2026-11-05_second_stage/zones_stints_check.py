"""Read-only: the stints rebuilt with the zones against data/stints, every existing column of every season and phase, and
the zones adding up to the buckets they split."""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
worst, problems = 0.0, []
for path in sorted(glob.glob("data/stints/*_[RP][SO].parquet")):
    name = os.path.basename(path)
    for suffix in ("", "_diag", "_shots", "_names"):
        old_path = path.replace(".parquet", f"{suffix}.parquet")
        new_path = os.path.join("data/stints_zones139", os.path.basename(old_path))
        if not os.path.exists(old_path):
            continue
        if not os.path.exists(new_path):
            problems.append(f"missing {new_path}")
            continue
        old, new = pd.read_parquet(old_path), pd.read_parquet(new_path)
        if len(old) != len(new):
            problems.append(f"{os.path.basename(old_path)}: rows {len(old)} against {len(new)}")
            continue
        missing = [c for c in old.columns if c not in new.columns]
        if missing:
            problems.append(f"{os.path.basename(old_path)}: lacks {missing[:5]}")
        for column in old.columns:
            if column not in new.columns:
                continue
            a, b = old[column], new[column]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                gap = float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float)))) if len(a) else 0.0
                same_nan = bool((a.isna() == b.isna()).all())
                if gap > 0 or not same_nan:
                    problems.append(f"{os.path.basename(old_path)} {column}: largest gap {gap}")
                worst = max(worst, gap)
            elif not a.astype(str).equals(b.astype(str)):
                problems.append(f"{os.path.basename(old_path)} {column}: differs")
    new = pd.read_parquet(os.path.join("data/stints_zones139", name))
    for side in ("h", "a"):
        twos = new[f"fga_{side}"] - new[f"fga_thr_{side}"]
        if not np.array_equal(new[f"fga_rim_{side}"] + new[f"fga_smid_{side}"] + new[f"fga_lmid_{side}"], twos):
            problems.append(f"{name}: the two-point zones do not add to the twos on side {side}")
        if not np.array_equal(new[f"fga_smid_{side}"] + new[f"fga_lmid_{side}"], new[f"fga_mid_{side}"]):
            problems.append(f"{name}: short and long mid do not add to mid on side {side}")
        if (new[f"fga_lthr_{side}"] > new[f"fga_thr_{side}"]).any() or (new[f"fgm_lthr_{side}"] > new[f"fga_lthr_{side}"]).any():
            problems.append(f"{name}: long threes exceed threes or long makes exceed long attempts on side {side}")
    print(name, "checked", flush=True)
print("largest gap on an existing column:", worst)
print("problems:", len(problems))
for line in problems[:40]:
    print(" ", line)
