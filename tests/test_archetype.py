"""The archetype mixture (src/eracoef/archetype.py) on data whose clusters are known by construction.

The statistic these tests pin is `spread`: the sd, across clusters, of the mean board-minus-blend gap on the
TOTAL rating. It replaced a hand-made bigness axis in tests/test_vs_consensus.py, so it needs its own
fixture-level guarantees: that it is near zero when no type is mis-rated, that it rises with the size of a
type-shaped distortion, and that a cluster too small to mean anything cannot drive it.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef.archetype import RATES, cluster_gaps, fit_clusters, gap_columns, per36, spread


@pytest.fixture(scope="module")
def world():
    """Three well-separated types (a big, a wing, a guard), 300 players, plus a rating and a blend that
    agree up to noise. `truth` is the type each player was drawn from."""
    rng = np.random.default_rng(7)
    centres = {0: dict(pts=12, fg3m=0.3, fg3_miss=0.7, ftm=3.0, ast=1.5, orb=3.5, drb=7.0, stl=0.6, blk=1.8, tov=1.5, pf=3.5),
               1: dict(pts=15, fg3m=2.0, fg3_miss=3.5, ftm=2.0, ast=2.5, orb=1.0, drb=4.0, stl=1.0, blk=0.5, tov=1.4, pf=2.5),
               2: dict(pts=20, fg3m=2.5, fg3_miss=4.5, ftm=4.0, ast=6.5, orb=0.6, drb=3.0, stl=1.2, blk=0.2, tov=2.6, pf=2.0)}
    truth = rng.integers(0, 3, 300)
    rows = []
    for t in truth:
        rows.append({c: max(0.0, centres[int(t)][c] * (1.0 + 0.10 * rng.standard_normal())) for c in RATES})
    X = pd.DataFrame(rows)
    skill = rng.standard_normal(len(truth))
    d = X.assign(player_id=np.arange(len(truth)),
                 rating_off=skill + 0.15 * rng.standard_normal(len(truth)),
                 rating_def=-skill + 0.15 * rng.standard_normal(len(truth)),
                 adj_offense=skill + 0.15 * rng.standard_normal(len(truth)),
                 adj_defense=-skill + 0.15 * rng.standard_normal(len(truth)))
    d["rating_total"] = d.rating_off + d.rating_def
    d["adj_overall"] = d.adj_offense + d.adj_defense
    return d, truth


def _spread_of(d, k=6, seed=0):
    labels, _ = fit_clusters(d[RATES].to_numpy(dtype=float), k=k, seed=seed)
    return spread(cluster_gaps(labels, gap_columns(d, d))), labels


def test_the_mixture_recovers_the_types(world):
    """k is an upper bound: six components on three types must not shatter them.  Scored by purity --
    labels are arbitrary, so what matters is that each cluster is nearly all one type."""
    d, truth = world
    labels, weights = fit_clusters(d[RATES].to_numpy(dtype=float), k=6, seed=0)
    used = [c for c in np.unique(labels) if (labels == c).sum() >= 10]
    assert 3 <= len(used) <= 4, f"three types became {len(used)} clusters of any size"
    purity = sum(pd.Series(truth[labels == c]).value_counts().iloc[0] for c in used) / sum((labels == c).sum() for c in used)
    assert purity > 0.95, f"clusters mix the types: purity {purity:.2f}"
    assert weights.sum() == pytest.approx(1.0)


def test_agreement_reads_near_zero(world):
    """No type is mis-rated here, so the statistic is noise around zero -- and must be small against the
    0.30 floor tests/test_vs_consensus.py sets on it."""
    d, _ = world
    got, _ = _spread_of(d)
    assert got < 0.10, f"an unbiased board reads {got:.3f}"


@pytest.mark.parametrize("bump", [0.5, 1.0, 2.0])
def test_a_type_shaped_distortion_raises_it_monotonically(world, bump):
    """Hand one type a flat bonus on the total and the statistic has to see it, larger for a larger bonus.
    This is what makes the floor convertible into points: on the real boards the slope is 0.16 per point
    per 100 (2026-09-10)."""
    d, truth = world
    base, _ = _spread_of(d)
    hit = d.copy()
    hit.loc[truth == 0, "rating_total"] = hit.loc[truth == 0, "rating_total"] + bump
    got, _ = _spread_of(hit)
    assert got > base, f"a {bump} bump on one type left the statistic at {got:.3f} against {base:.3f}"
    if bump >= 1.0:
        assert got > 0.30, f"a {bump}-point archetype bias reads {got:.3f}, under the shipped floor"


def test_a_tiny_cluster_cannot_drive_the_statistic(world):
    """`min_n` exists because the mixture emits a handful-of-players component whenever k is generous, and
    a five-player mean gap is noise.  A fabricated tiny cluster with an absurd gap must be ignored."""
    d, _ = world
    labels, _ = fit_clusters(d[RATES].to_numpy(dtype=float), k=6, seed=0)
    gaps = gap_columns(d, d)
    labels = np.asarray(labels).copy()
    labels[:4] = labels.max() + 1
    t = cluster_gaps(labels, gaps)
    t.loc[t.n < 10, "gap_total"] = 99.0
    assert spread(t, min_n=10) < 0.10
    assert spread(t, min_n=1) > 1.0, "the guard is doing nothing; min_n is not what excluded it"


def test_per36_is_a_rate_not_a_total():
    box = pd.DataFrame({"player_id": [1, 1, 2], "minutes": [36.0, 36.0, 18.0],
                        **{c: [1.0, 3.0, 1.0] for c in RATES}})
    g = per36(box).set_index("player_id")
    assert g.loc[1, "pts"] == pytest.approx(2.0)      # 4 points in 72 minutes
    assert g.loc[2, "pts"] == pytest.approx(2.0)      # 1 point in 18 minutes


def test_the_gap_is_scale_free(world):
    """Both sides are standardised, so doubling the board's spread cannot by itself create a gap."""
    d, _ = world
    a = gap_columns(d, d)
    b = gap_columns(d.assign(rating_total=2 * d.rating_total, rating_off=2 * d.rating_off,
                             rating_def=2 * d.rating_def), d)
    assert np.allclose(a.to_numpy(), b.to_numpy())
