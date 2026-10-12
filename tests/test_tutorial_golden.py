"""examples/tutorial.py --small against tests/golden/tutorial_small.json: steps 1-3 on a small simulated league give the
numbers they gave when the file was written, each to a relative 1e-6.

The tutorial runs the real step modules (rapm, box_prior, prior_rapm) on a league from openrapm.simulate, so a change
anywhere along that path that moves a number fails here, in a few seconds and with no data on disk.  A change that is
meant to move them (a new shipped setting, a different simulated league) rewrites the file on purpose:

    python examples/tutorial.py --small --write-golden

and the diff of the JSON then shows what moved.  The tutorial runs in its own process, as a reader would run it.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "examples" / "tutorial.py"
GOLDEN = ROOT / "tests" / "golden" / "tutorial_small.json"
RELATIVE = 1e-6
ABSOLUTE = 1e-12          # for a number that is 0 in the golden file


@pytest.fixture(scope="module")
def small_tour(tmp_path_factory):
    """(what the small tour printed, every number it wrote)."""
    out = tmp_path_factory.mktemp("tutorial") / "small.json"
    result = subprocess.run([sys.executable, str(TUTORIAL), "--small", "--json", str(out)], capture_output=True,
                            text=True, encoding="utf-8", cwd=ROOT, timeout=600)
    assert result.returncode == 0, result.stderr
    return result.stdout, json.loads(out.read_text(encoding="utf-8"))


def _leaves(tree, path=()) -> dict:
    """{path: value} for every number, string and flag in a nested JSON tree."""
    if isinstance(tree, dict):
        return {key: value for name, child in tree.items() for key, value in _leaves(child, (*path, name)).items()}
    if isinstance(tree, list):
        return {key: value for index, child in enumerate(tree)
                for key, value in _leaves(child, (*path, index)).items()}
    return {path: tree}


def test_every_number_matches_the_golden_file(small_tour):
    _, got = small_tour
    want = _leaves(json.loads(GOLDEN.read_text(encoding="utf-8")))
    got = _leaves(got)
    assert got.keys() == want.keys(), "the tour writes other numbers than the golden file holds; regenerate it"
    moved = [f"{'/'.join(map(str, key))}: {got[key]!r} (golden {value!r})" for key, value in want.items()
             if not _same(got[key], value)]
    assert not moved, f"{len(moved)} of {len(want)} numbers moved, the first: {moved[:5]}"


def _same(got, want) -> bool:
    if isinstance(want, float) or isinstance(got, float):
        return (math.isnan(got) and math.isnan(want)) or math.isclose(got, want, rel_tol=RELATIVE, abs_tol=ABSOLUTE)
    return got == want


def test_the_prior_improves_on_plain_rapm_where_the_box_score_sees_the_impact(small_tour):
    """The story the tutorial tells holds on the small league too: on offence, where the box score explains most of
    a player's impact, step 3 beats step 1 on the same season."""
    _, got = small_tour
    summary = got["summary"]
    before, after = summary["step1"]["one_season"]["offence"], summary["step3"]["offence"]
    assert after["correlation"] > before["correlation"] and after["error"] < before["error"]


def test_the_tour_tells_each_step_with_its_numbers(small_tour):
    printed, got = small_tour
    for heading in ("The league:", "Step 1: plain RAPM", "Step 2: the box-score prior", "Step 3: prior-informed RAPM"):
        assert heading in printed
    summary = got["summary"]
    for number in (summary["step1"]["one_season"]["offence"]["correlation"], summary["step2"]["offence"]["correlation"],
                   summary["step3"]["offence"]["correlation"]):
        assert f"{number:.2f}" in printed
