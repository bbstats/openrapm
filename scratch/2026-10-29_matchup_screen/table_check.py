"""The refactored table against the tables written by the old code (screen112, screen119): every old column equal."""
import importlib.util
import sys

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("screen155", "scripts/checks/155_input_screen.py")
m = importlib.util.module_from_spec(spec)
sys.modules["screen155"] = m
spec.loader.exec_module(m)
for tag in ("screen112", "screen119"):
    new, _ = m.screen_table(tag)
    old = pd.read_csv(f"outputs/csv/screen_{tag}_table.csv")
    key = ["arm", "side", "input"]
    both = old.merge(new, on=key, suffixes=("_old", "_new"))
    worst = 0.0
    for column in old.columns:
        if column in key or column in ("plain_name", "verdict") or old[column].dtype == object:
            continue
        a, b = both[f"{column}_old"].to_numpy(float), both[f"{column}_new"].to_numpy(float)
        worst = max(worst, float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0)
    same = (both.verdict_old.fillna("") == both.verdict_new.fillna("")).all()
    print(tag, len(old), len(new), len(both), "largest difference", worst, "verdicts same", same,
          [c for c in new.columns if c.startswith("vs_shadow_")])
