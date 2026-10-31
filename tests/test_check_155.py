"""scripts/checks/155_input_screen.py: an input added to a side's list keeps the number of inputs each tree is shown,
so the rule's comparison with today's lists is not charged for LightGBM's rounding (2026-10-30)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "155_input_screen.py"


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("input_screen_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def widest_tree(share: float, n_inputs: int) -> int:
    """The most distinct inputs any tree of a LightGBM fit uses, every input informative, at column share `share`."""
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, n_inputs))
    y = x.sum(axis=1) + 0.1 * rng.normal(size=4000)
    model = lgb.LGBMRegressor(n_estimators=40, num_leaves=63, min_child_samples=5, colsample_bytree=share,
                              verbose=-1, random_state=0).fit(x, y)
    trees = model.booster_.trees_to_dataframe()
    return int(trees.dropna(subset=["split_feature"]).groupby("tree_index").split_feature.nunique().max())


def test_an_added_input_keeps_the_inputs_each_tree_is_shown(screen):
    """The defensive share 0.367 shows a tree 7 of 20 inputs but 8 of 21; the added arms are given 7 of 21."""
    shipped_share = 0.3671956372541044
    assert widest_tree(shipped_share, 20) == 7 and widest_tree(shipped_share, 21) == 8
    fixed = screen.same_inputs_per_tree({"colsample_bytree": shipped_share, "num_leaves": 7}, 20, 21)
    assert fixed["num_leaves"] == 7 and fixed["colsample_bytree"] == pytest.approx(7 / 21)
    assert widest_tree(fixed["colsample_bytree"], 21) == 7
    offence = screen.same_inputs_per_tree({"colsample_bytree": 0.5361093535529958}, 20, 21)
    assert round(21 * offence["colsample_bytree"]) == round(21 * 0.5361093535529958) == 11
    assert screen.same_inputs_per_tree({"learner": "kumo_tabular"}, 20, 21) == {"learner": "kumo_tabular"}
