"""The single-year prior's inputs: the feature list, and the order the aggregation happens in.

Two things here have bitten already.  The board's hand-picked list left out every ratio and every
shot-quality column -- the features an axis-aligned tree cannot build for itself -- and nothing failed,
because `add_derived` skips a family whose inputs are absent instead of raising.  And a possession-weighted
mean of a RATIO is not the ratio of the means, so the aggregation order is a real choice and not a detail.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef import singleyear as sy
from eracoef.bio import PLAYER_INPUTS
from eracoef.gbdt_prior import DERIVED, RATIOS, SHOTQ


def _panel(n=40, seed=0):
    """A frame with every `INPUT_COLUMNS` name, two seasons per player, plausible magnitudes."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n):
        for season in (2014, 2015):
            r = {c: float(rng.normal(0, 2)) for c in sy.INPUT_COLUMNS}
            r.update(player_id=pid, season=season, side="O",
                     poss=float(rng.uniform(300, 4000)))
            for c in sy.INPUT_COLUMNS:
                if c.startswith("raw_"):
                    r[c] = float(rng.uniform(1, 12))
            r.update(shot_fg2a=float(rng.uniform(50, 900)), shot_fg3a=float(rng.uniform(10, 600)),
                     shot_lg2=0.485, shot_lg3=0.35, shot_lgpps=0.99)
            r.update(shot_fg2m=r["shot_fg2a"] * 0.5, shot_xl2=r["shot_fg2a"] * 0.49,
                     shot_fg3m=r["shot_fg3a"] * 0.36, shot_xl3=r["shot_fg3a"] * 0.35)
            rows.append(r)
    return pd.DataFrame(rows)


def test_the_bio_names_have_not_drifted():
    """`singleyear` repeats `bio.PLAYER_INPUTS` because it may not import a data-layer module."""
    assert sy.BIO == list(PLAYER_INPUTS)


def test_the_feature_list_carries_what_a_tree_cannot_build():
    assert "season" not in sy.PRIOR_FEATURES, "a row pooled over twelve seasons has no season"
    for family, name in ((DERIVED, "DERIVED"), (RATIOS, "RATIOS"), (SHOTQ, "SHOTQ")):
        assert set(family) <= set(sy.PRIOR_FEATURES), f"{name} missing from the single-year prior"
    assert len(sy.PRIOR_FEATURES) == len(set(sy.PRIOR_FEATURES)) == 54


def test_aggregate_builds_every_requested_feature():
    out = sy.aggregate(_panel())
    assert list(out.index.names) == ["player_id"]
    assert not [f for f in sy.PRIOR_FEATURES if f not in out.columns]
    assert out[sy.PRIOR_FEATURES].notna().all().all()


def test_a_missing_input_raises_instead_of_shortening_the_list():
    """The failure mode the guard exists for: drop the raw rates and the ratios quietly vanish."""
    panel = _panel().drop(columns=[c for c in sy.INPUT_COLUMNS if c.startswith("raw_")])
    with pytest.raises(KeyError, match="add_derived did not build"):
        sy.aggregate(panel)


def test_the_ratios_are_built_on_the_average_not_averaged():
    """`ts` of the mean, not the mean of `ts` -- the two differ, and the first is the one we want."""
    panel = _panel()
    got = sy.aggregate(panel)["ts"]

    per_season = sy.season_frame(panel)
    w = per_season.poss
    averaged = (per_season.ts * w).groupby(per_season.player_id).sum() / w.groupby(per_season.player_id).sum()

    num, den, pad, target = RATIOS["ts"]
    mean_raw = {c: (panel[f"raw_{c}"] * panel.poss).groupby(panel.player_id).sum()
                   / panel.poss.groupby(panel.player_id).sum() for c in set(num) | set(den)}
    expected = ((sum(k * mean_raw[c] for c, k in num.items()) + pad * target)
                / (sum(k * mean_raw[c] for c, k in den.items()) + pad))
    assert np.allclose(got.to_numpy(), expected.to_numpy())
    assert not np.allclose(got.to_numpy(), averaged.to_numpy())


def test_prior_rows_joins_an_external_target_instead_of_pooling_one():
    """The target is already a leave-one-out quantity; pooling it again would count it twice."""
    panel = _panel()
    target = pd.DataFrame({"player_id": np.arange(0, 40, 2), "offense": np.linspace(-3, 3, 20),
                           "possessions": np.linspace(500, 9000, 20)})
    rows = sy.prior_rows(target, panel, "offense")
    assert len(rows) == 20
    assert np.allclose(rows.target.to_numpy(), rows.offense.to_numpy())
    assert np.allclose(rows.weight.to_numpy(), rows.possessions.to_numpy())
