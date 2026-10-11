"""The player-type mixture (src/openrapm/player_types.py) on data whose clusters are known by construction.

The statistic these tests pin is `spread`: the sd, across clusters, of the mean ratings-minus-blend gap on the
TOTAL rating. It replaced a hand-made bigness axis in tests/test_vs_consensus.py, so it needs its own
fixture-level guarantees: that it is near zero when no type is mis-rated, that it rises with the size of a
type-shaped distortion, and that a cluster too small to mean anything cannot drive it.
"""
import numpy as np
import pandas as pd
import pytest

from openrapm.player_types import (
    RATES,
    bias_by_type,
    cluster_gaps,
    describe_types,
    fit_clusters,
    gap_columns,
    per36,
    signature,
    spread,
)


@pytest.fixture(scope="module")
def world():
    """Three well-separated types (a big, a wing, a guard), 300 players, plus a rating and a blend that
    agree up to noise. `truth` is the type each player was drawn from."""
    rng = np.random.default_rng(7)
    centres = {0: dict(pts=12, fg3m=0.3, fg3_miss=0.7, ftm=3.0, ast=1.5, orb=3.5, drb=7.0, stl=0.6, blk=1.8,
                       tov=1.5, pf=3.5),
               1: dict(pts=15, fg3m=2.0, fg3_miss=3.5, ftm=2.0, ast=2.5, orb=1.0, drb=4.0, stl=1.0, blk=0.5,
                       tov=1.4, pf=2.5),
               2: dict(pts=20, fg3m=2.5, fg3_miss=4.5, ftm=4.0, ast=6.5, orb=0.6, drb=3.0, stl=1.2, blk=0.2,
                       tov=2.6, pf=2.0)}
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
    largest_share = sum(pd.Series(truth[labels == c]).value_counts().iloc[0] for c in used)
    purity = largest_share / sum((labels == c).sum() for c in used)
    assert purity > 0.95, f"clusters mix the types: purity {purity:.2f}"
    assert weights.sum() == pytest.approx(1.0)


def test_agreement_reads_near_zero(world):
    """No type is mis-rated here, so the statistic is noise around zero -- and must be small against the
    0.30 limit tests/test_vs_consensus.py once set on it."""
    d, _ = world
    got, _ = _spread_of(d)
    assert got < 0.10, f"unbiased ratings read {got:.3f}"


@pytest.mark.parametrize("bump", [0.5, 1.0, 2.0])
def test_a_type_shaped_distortion_raises_it_monotonically(world, bump):
    """Hand one type a flat bonus on the total and the statistic has to see it, larger for a larger bonus.
    This is what makes the limit convertible into points: on the real ratings the slope is 0.16 per point
    per 100 (2026-09-10)."""
    d, truth = world
    base, _ = _spread_of(d)
    hit = d.copy()
    hit.loc[truth == 0, "rating_total"] = hit.loc[truth == 0, "rating_total"] + bump
    got, _ = _spread_of(hit)
    assert got > base, f"a {bump} bump on one type left the statistic at {got:.3f} against {base:.3f}"
    if bump >= 1.0:
        assert got > 0.30, f"a {bump}-point archetype bias reads {got:.3f}, over the shipped limit"


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
    """Both sides are standardised, so doubling the ratings' spread cannot by itself create a gap."""
    d, _ = world
    a = gap_columns(d, d)
    b = gap_columns(d.assign(rating_total=2 * d.rating_total, rating_off=2 * d.rating_off,
                             rating_def=2 * d.rating_def), d)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_a_signature_is_the_four_largest_z_scores_with_their_signs():
    """The bias page's Rim-protecting centers, 2026: orb +2.25, threes missed -1.67, blk +1.60, threes made -1.55."""
    profile = pd.Series({"pts": -0.40, "fg3m": -1.55, "fg3_miss": -1.67, "ftm": -0.01, "ast": -0.72, "orb": 2.25,
                         "drb": 1.09, "stl": -0.22, "blk": 1.60, "tov": -0.18, "pf": 0.96})
    assert signature(profile) == "+orb -fg3_miss +blk -fg3m"
    assert signature(profile, n=2) == "+orb -fg3_miss"


def _typed_players():
    """Two types of twelve players and one of three: type 0 rated 1.0 above the source on 100 possessions each,
    type 1 rated 0.5 below on 300 each."""
    rng = np.random.default_rng(3)
    groups = np.repeat([0, 1, 2], [12, 12, 3])
    players = pd.DataFrame({"group": groups, "player_name": [f"player {i}" for i in range(len(groups))],
                            "poss": np.where(groups == 1, 300.0, 100.0) + np.arange(len(groups)),
                            "ours": rng.standard_normal(len(groups))})
    players["theirs"] = players.ours - np.select([groups == 0, groups == 1], [1.0, -0.5], 0.0)
    players["bias"] = players.ours - players.theirs
    z = pd.DataFrame({rate: rng.standard_normal(len(groups)) + 2.0 * (groups == 0) * (rate == "orb")
                      for rate in RATES})
    return players, z


def test_each_type_is_named_from_its_signature_or_left_unnamed():
    players, z = _typed_players()
    named = describe_types(players, z, {signature(z.loc[players.group == 0, RATES].mean()): "Big men"})
    assert named[0]["name"] == "Big men" and named[0]["signature"].startswith("+orb")
    assert named[1]["name"] == "UNNAMED"
    assert list(named[0]) == ["signature", "profile", "name", "most_played", "most_typical", "least_typical"]
    assert named[0]["most_played"][0] == "player 11"          # the type-0 player with the most possessions
    assert len(named[0]["most_typical"]) == 6 and len(named[0]["least_typical"]) == 3


def test_bias_by_type_is_possession_weighted_ordered_by_reach_and_skips_tiny_types():
    """Type 1's bias is half type 0's but covers three times the possessions, so it comes first; type 2 has
    three players, under the ten a row needs."""
    players, z = _typed_players()
    rows = bias_by_type(players, describe_types(players, z, {}), "ours", "theirs", "bias")
    assert [row["group"] for row in rows] == [1, 0]
    assert rows[0]["bias"] == pytest.approx(-0.5) and rows[1]["bias"] == pytest.approx(1.0)
    assert rows[0]["cost"] == pytest.approx(0.5 * rows[0]["possessions"])
    weights = players.poss[players.group == 0]
    assert rows[1]["ours"] == pytest.approx(np.average(players.ours[players.group == 0], weights=weights))
