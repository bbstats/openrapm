"""Probe: what one rd143 fold's player rows carry."""
from pathlib import Path
import pandas as pd
from openrapm.data.within_folds import SPLITS, load_folds
folds = load_folds(Path("outputs/within/rd143_within"), (2026, 2026), splits=SPLITS)
f = folds[0]
print(f.stem, f.season, len(folds))
for side in ("O", "D"):
    r = f.rows[side]
    print(side, r.shape)
    print(sorted(r.columns))
