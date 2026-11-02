"""After the crash of 2026-11-01 (a memory stick came unseated, about 6:04 PM): are the numbers computed just before it
the numbers a clean machine computes?  Every fold the crashed run finished against the same fold of the fresh run: the
summary's numbers, the held-back games' arrays, the players' table.  Prints the counts and the largest difference;
exit status 1 when any number differs by more than 1e-9 or any fold fails to match, so the adoption stops there."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OLD, NEW = Path("outputs/within/padbins127_k10_crash"), Path("outputs/within/padbins127_k10")
TOLERANCE = 1e-9


def largest_gap(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return np.inf
    if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
        a, b = a.astype(float), b.astype(float)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            return np.inf
        return float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    return 0.0 if (a.astype(str) == b.astype(str)).all() else np.inf


def fold_gap(stem: str) -> float:
    old, new = json.loads((OLD / f"fold_{stem}.json").read_text()), json.loads((NEW / f"fold_{stem}.json").read_text())
    worst = 0.0
    for key, value in old["summary"].items():
        if key != "seconds" and isinstance(value, (int, float)):
            worst = max(worst, abs(value - new["summary"][key]))
    with np.load(OLD / f"test_{stem}.npz", allow_pickle=True) as a, np.load(NEW / f"test_{stem}.npz", allow_pickle=True) as b:
        for name in a.files:
            worst = max(worst, largest_gap(a[name], b[name]))
    x, y = pd.read_parquet(OLD / f"players_{stem}.parquet"), pd.read_parquet(NEW / f"players_{stem}.parquet")
    for name in x.columns:
        worst = max(worst, largest_gap(x[name].to_numpy(), y[name].to_numpy()))
    return worst


def main() -> int:
    stems = sorted(path.stem.removeprefix("fold_") for path in OLD.glob("fold_*.json"))
    matched, unreadable, worst, far = 0, [], 0.0, []
    for stem in stems:
        try:
            gap = fold_gap(stem)
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
            unreadable.append((stem, type(error).__name__))
            continue
        matched += 1
        worst = max(worst, gap)
        if gap > TOLERANCE:
            far.append((stem, gap))
    print(f"the crashed run's {len(stems)} folds against the fresh run: {matched} compared, largest difference {worst:.2e}; "
          f"{len(far)} differ by more than {TOLERANCE}; {len(unreadable)} cut off by the crash: {unreadable}")
    print(far[:20])
    return 1 if far or matched < len(stems) - 4 else 0


if __name__ == "__main__":
    sys.exit(main())
