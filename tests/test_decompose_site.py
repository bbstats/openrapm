"""The published breakdown page's data: both splits of every row add up to its RAPM.

docs/data/decompose_1y.json and decompose_3y.json are written by scripts/83_decompose_site.py, which checks both
identities to 1e-9 before it writes; this checks what was actually committed, at the two decimals the page shows.
"""
import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
FIELDS = ["n", "p", "m", "t", "o", "x", "r", "a", "h", "w", "g", "e", "f"]
ROUNDING = 0.031        # six numbers at two decimals, the five pieces and the total, each off by up to 0.005


@pytest.mark.parametrize("name", ["decompose_1y.json", "decompose_3y.json"])
def test_rows_add_up(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} is not built")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["meta"]["fields"] == FIELDS
    assert list(doc["rows"]) == doc["meta"]["periods"]
    by_player = by_possession = 0.0
    for period, rows in doc["rows"].items():
        assert rows, f"{period} has no players"
        for n, p, m, t, o, x, r, a, h, w, g, e, f in rows:
            assert n and p > 0, f"{period}: a row without a name or possessions"
            assert f is None or isinstance(f, (int, float)), f"{period}: {n}'s off-court rating is not a number"
            by_player = max(by_player, abs(m + t + o + x + r - a))
            by_possession = max(by_possession, abs(h + w + g + e - a))
    assert by_player <= ROUNDING, f"a row's five pieces miss its RAPM by {by_player:.3f}"
    assert by_possession <= ROUNDING, f"a row's four possession groups miss its RAPM by {by_possession:.3f}"
