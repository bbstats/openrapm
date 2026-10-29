"""How many inputs a LightGBM tree is shown at the defensive booster's column share, 0.3672: 20 inputs against 21.
The shipped defensive list is 18 inputs plus the two row columns; a screen candidate makes 21."""
import lightgbm
import numpy as np


def splits(node):
    if "split_feature" in node:
        yield node
        yield from splits(node["left_child"])
        yield from splits(node["right_child"])


rng = np.random.default_rng(0)
for n_inputs in (20, 21):
    x = rng.normal(size=(5000, n_inputs))
    y = x.sum(axis=1) + rng.normal(size=5000)
    model = lightgbm.LGBMRegressor(n_estimators=40, num_leaves=63, min_child_samples=5,
                                   colsample_bytree=0.3671956372541044, verbose=-1, random_state=1).fit(x, y)
    used = [len({node["split_feature"] for node in splits(tree["tree_structure"])})
            for tree in model.booster_.dump_model()["tree_info"]]
    print(n_inputs, "inputs: the most any tree split on", max(used))
