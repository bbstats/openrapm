"""`openrapm compare A B`: are two parquet tables the same, and if not, where do they differ?

    openrapm compare A.parquet B.parquet [--season 2026] [--key player_id,season] [--ignore a,b]

The first line is the verdict:
    IDENTICAL-BYTES    the two files are the same bytes
    IDENTICAL-VALUES   the same columns in the same order, the same rows, the same data types and every value
                       equal (NaN equal to NaN), though the bytes differ
    DIFFERENT          anything else; each difference follows on its own line

`--season` keeps one season of each table (a one-season rebuild against a full table).  When the rows come in
another order they are aligned on `--key` before the values are compared, and the reordering is reported.

Example: a one-season rebuild of 2026 against the published table,
    openrapm compare outputs/season_ratings_check_step3.parquet outputs/season_ratings_product.parquet --season 2026
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def file_hash(path) -> str:
    """The file's SHA-256, in hexadecimal."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path, season: int | None, ignore) -> pd.DataFrame:
    table = pd.read_parquet(path)
    if season is not None:
        table = table[table.season == int(season)].reset_index(drop=True)
    return table.drop(columns=list(ignore), errors="ignore")


def _shape_notes(a: pd.DataFrame, b: pd.DataFrame) -> list:
    """Differences in the columns and the number of rows."""
    notes = []
    if list(a.columns) != list(b.columns):
        notes.append(f"columns differ: only A {sorted(set(a.columns) - set(b.columns))}, only B "
                     f"{sorted(set(b.columns) - set(a.columns))}, same set {set(a.columns) == set(b.columns)}")
    if len(a) != len(b):
        notes.append(f"rows differ: {len(a)} vs {len(b)}")
    return notes


def _aligned(a: pd.DataFrame, b: pd.DataFrame, key) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """The two tables in the same row order: as they are when their keys already match, else sorted on the key."""
    key = [column for column in key if column in a.columns and column in b.columns]
    if not key or (len(a) == len(b) and a[key].equals(b[key])):
        return a, b, []
    return (a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True),
            ["row order differs; aligning on " + ",".join(key)])


def _column_note(x: pd.Series, y: pd.Series) -> tuple[str | None, float]:
    """(a note when the column's values differ, the largest absolute difference); NaN equals NaN."""
    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        left, right = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
        nan_mismatch = int((np.isnan(left) != np.isnan(right)).sum())
        gap = left - right
        largest = float(np.nanmax(np.abs(gap))) if np.isfinite(gap).any() else 0.0
        largest = 0.0 if np.isnan(largest) else largest
        note = f"max |diff| {largest:.3g}, NaN mismatches {nan_mismatch}" if largest > 0 or nan_mismatch else None
        return note, largest
    both_missing = x.isna().to_numpy() & y.isna().to_numpy()
    differ = int(((x.astype(str) != y.astype(str)).to_numpy() & ~both_missing).sum())
    return (f"{differ} values differ" if differ else None), 0.0


def compare(a_path, b_path, season: int | None = None, ignore=(), key=("player_id", "season")) -> tuple:
    """(verdict, notes, rows compared, largest numeric difference) for two parquet files."""
    same_bytes = file_hash(a_path) == file_hash(b_path)
    a, b = _read(a_path, season, ignore), _read(b_path, season, ignore)
    notes = _shape_notes(a, b)
    a, b, order = _aligned(a, b, key)
    notes += order
    worst = 0.0
    for column in [column for column in a.columns if column in b.columns]:
        if str(a[column].dtype) != str(b[column].dtype):
            notes.append(f"dtype {column}: {a[column].dtype} vs {b[column].dtype}")
        if len(a) == len(b):
            note, largest = _column_note(a[column], b[column])
            worst = max(worst, largest)
            if note:
                notes.append(f"{column}: {note}")
    verdict = "IDENTICAL-BYTES" if same_bytes else "IDENTICAL-VALUES" if not notes else "DIFFERENT"
    return verdict, notes, len(a), worst


def run_compare(a_path, b_path, season: int | None = None, ignore: str = "", key: str = "player_id,season") -> int:
    """`openrapm compare`: print the verdict and each difference; 1 when the tables differ, else 0."""
    verdict, notes, rows, worst = compare(a_path, b_path, season=season,
                                          ignore=[column for column in ignore.split(",") if column],
                                          key=[column for column in key.split(",") if column])
    print(f"{verdict}  {a_path}  vs  {b_path}  ({rows} rows, worst numeric diff {worst:.3g})")
    for note in notes:
        print("   ", note)
    return 0 if verdict != "DIFFERENT" else 1
