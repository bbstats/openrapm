"""`openrapm compare` (src/openrapm/commands/compare.py): the three verdicts and what each difference says."""
import numpy as np
import pandas as pd

from openrapm.commands.compare import compare, run_compare


def _table():
    return pd.DataFrame({"player_id": [1, 2, 3], "season": [2026, 2026, 2025], "rating": [1.5, -0.25, np.nan],
                         "player_name": ["A", None, "C"]})


def test_the_same_file_is_identical_bytes(tmp_path):
    path = tmp_path / "a.parquet"
    _table().to_parquet(path, index=False)
    verdict, notes, rows, worst = compare(path, path)
    assert (verdict, notes, rows, worst) == ("IDENTICAL-BYTES", [], 3, 0.0)


def test_the_same_values_in_other_bytes_are_identical_values(tmp_path):
    _table().to_parquet(tmp_path / "a.parquet", index=False, compression="snappy")
    _table().to_parquet(tmp_path / "b.parquet", index=False, compression="gzip")
    verdict, notes, _, _ = compare(tmp_path / "a.parquet", tmp_path / "b.parquet")
    assert verdict == "IDENTICAL-VALUES" and notes == []


def test_a_moved_value_a_reordering_and_a_new_dtype_are_each_reported(tmp_path, capsys):
    _table().to_parquet(tmp_path / "a.parquet", index=False)
    other = _table().iloc[[1, 0, 2]].reset_index(drop=True)
    other.loc[other.player_id == 1, "rating"] = 1.75
    other["season"] = other.season.astype("int32")
    other.to_parquet(tmp_path / "b.parquet", index=False)
    verdict, notes, _, worst = compare(tmp_path / "a.parquet", tmp_path / "b.parquet")
    assert verdict == "DIFFERENT" and worst == 0.25
    assert "row order differs; aligning on player_id,season" in notes
    assert "dtype season: int64 vs int32" in notes
    assert "rating: max |diff| 0.25, NaN mismatches 0" in notes
    assert run_compare(tmp_path / "a.parquet", tmp_path / "b.parquet", season=2025, ignore="season") == 0
    assert run_compare(tmp_path / "a.parquet", tmp_path / "b.parquet") == 1
    assert capsys.readouterr().out.splitlines()[-4].startswith("DIFFERENT")
