"""The box-score feature definitions (src/openrapm/gbdt_prior.py): the playing-time interactions `add_derived`
builds from a feature list."""
import numpy as np
import pandas as pd

from openrapm.gbdt_prior import add_derived, interaction_parts


def test_interaction_features_are_the_products_asked_for():
    """`gs_pct_x_<f>` / `poss_pct_x_<f>`: how much he played, times what he did (owner ruling, 2026-09-10).
    The product is the product, and only the names asked for are built."""
    assert interaction_parts("poss_pct_x_past_apm") == ("poss_pct", "past_apm")
    assert interaction_parts("blk") is None and interaction_parts("gsx") is None

    df = pd.DataFrame({"gs_pct": [0.0, 0.5, 1.0], "poss_pct": [0.1, 0.2, 0.4], "blk": [1.0, -2.0, 3.0]})
    out = add_derived(df.copy(), ["gs_pct_x_blk", "poss_pct_x_blk"])
    assert np.allclose(out.gs_pct_x_blk, df.gs_pct * df.blk)
    assert np.allclose(out.poss_pct_x_blk, df.poss_pct * df.blk)
    assert "gs_pct_x_gs_pct" not in out.columns, "only the names asked for are built"
