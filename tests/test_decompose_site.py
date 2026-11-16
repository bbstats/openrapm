"""The published ratings carry an NBA id on every row.

docs/data/ratings.json is written by `openrapm site`.  The ids are in the files and the CSV downloads only.
"""
import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
FIELDS = ["n", "p", "m", "t", "o", "x", "r", "a", "h", "w", "g", "e", "f", "id"]
ROUNDING = 0.031        # six numbers at two decimals, the five pieces and the total, each off by up to 0.005


def test_ratings_carry_ids():
    path = DATA / "ratings.json"
    if not path.exists():
        pytest.skip("ratings.json is not built")
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    missing = [r for r in rows if not (isinstance(r.get("id"), int) and r["id"] > 0)]
    assert not missing, f"{len(missing)} rating rows have no NBA id, e.g. {missing[:2]}"
    keys = [(r["s"], r["id"]) for r in rows]
    assert len(set(keys)) == len(keys), "a player appears twice in one season"
