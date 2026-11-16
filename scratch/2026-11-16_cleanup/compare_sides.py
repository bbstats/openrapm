"""The cleanup's proof that no number moved: every file the before side wrote against the after side's.

    python scratch/2026-11-16_cleanup/compare_sides.py

Tables (parquet): the same bytes, or the same values when only the file's own bookkeeping differs; a `seconds` column,
which records how long a fold took, is left out.  Arrays (npz): every array equal, since a zip records the time it
was written.  JSON: equal once `seconds`, `argv` and `started` are taken out.  CSV and the site's files: the same
bytes.  Prints one line a file and the count; exits 1 when any pair differs or a file is missing on one side.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("A:/code/spmm/outputs")
TIMING = {"seconds", "argv", "started", "elapsed"}


def same_parquet(a: Path, b: Path) -> tuple[bool, str]:
    if a.read_bytes() == b.read_bytes():
        return True, "same bytes"
    x, y = pd.read_parquet(a), pd.read_parquet(b)
    x, y = (frame.drop(columns=[c for c in frame.columns if c in TIMING]) for frame in (x, y))
    try:
        pd.testing.assert_frame_equal(x, y, check_exact=True)
    except AssertionError as error:
        return False, str(error).splitlines()[0][:200]
    return True, "same values"


def same_npz(a: Path, b: Path) -> tuple[bool, str]:
    x, y = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    if sorted(x.files) != sorted(y.files):
        return False, f"arrays {sorted(x.files)} against {sorted(y.files)}"
    for name in x.files:
        if not np.array_equal(x[name], y[name], equal_nan=True):
            return False, f"array {name} differs"
    return True, "same arrays"


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in TIMING}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def same_json(a: Path, b: Path) -> tuple[bool, str]:
    if a.read_bytes() == b.read_bytes():
        return True, "same bytes"
    x, y = (strip(json.loads(path.read_text(encoding="utf-8"))) for path in (a, b))
    return (x == y), ("same once timings are out" if x == y else "content differs")


def same_bytes(a: Path, b: Path) -> tuple[bool, str]:
    return (a.read_bytes() == b.read_bytes()), "same bytes" if a.read_bytes() == b.read_bytes() else (
        f"{hashlib.sha256(a.read_bytes()).hexdigest()[:12]} against {hashlib.sha256(b.read_bytes()).hexdigest()[:12]}")


RULES = {".parquet": same_parquet, ".npz": same_npz, ".json": same_json}


def pairs() -> list:
    """(label, before path, after path) of everything both sides write."""
    out = []
    for stem in ("test", "prod", "morning"):
        for suffix in ("_step3", "_step4", "", "_regular"):
            name = f"season_ratings_cu_{{}}_{stem}{suffix}.parquet"
            if (OUT / name.format("b")).exists() or suffix != "_regular":
                out.append((name.format("S"), OUT / name.format("b"), OUT / name.format("a")))
        name = f"csv/season_ratings_cu_{{}}_{stem}_step4_multipliers.csv"
        out.append((name.format("S"), OUT / name.format("b"), OUT / name.format("a")))
    out.append(("csv/wins_cu_S_morning_seasons.csv", OUT / "csv/wins_cu_b_morning_seasons.csv",
                OUT / "csv/wins_cu_a_morning_seasons.csv"))
    folders = [*(f"within/cu_{{}}_{tag}" for tag in ("plain", "cut10", "cut50", "pieces")), "second_prior/cu_{}_sp",
               "cu_{}_site"]
    for folder in folders:
        before, after = OUT / folder.format("b"), OUT / folder.format("a")
        names = sorted({p.relative_to(before).as_posix() for p in before.rglob("*") if p.is_file()}
                       | {p.relative_to(after).as_posix() for p in after.rglob("*") if p.is_file()})
        out += [(f"{folder.format('S')}/{name}", before / name, after / name) for name in names
                if not name.endswith(".log") and name != "boosters.json"]
    return out


def main() -> int:
    bad = 0
    found = pairs()
    for label, before, after in found:
        if not before.exists() or not after.exists():
            print(f"MISSING     {label}: before {before.exists()}, after {after.exists()}")
            bad += 1
            continue
        same, note = RULES.get(before.suffix, same_bytes)(before, after)
        print(f"{'same' if same else 'DIFFERENT':10s}  {label}: {note}")
        bad += not same
    print(f"\n{len(found) - bad} of {len(found)} files the same; {bad} not")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
