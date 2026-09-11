"""Rebalanced leave-one-out (src/eracoef/rloocv.py).

The claim under test is arithmetic before it is statistical: holding fold j out moves the remaining
weighted mean AWAY from fold j's own label, by an amount identity (A) gives exactly, and the partner
drop is supposed to move it back.  So the tests check the identity, check the partner shrinks the shift,
check the splitter is still a partition of the data, and check the artifact the paper is about --
a spurious negative correlation between a leave-one-out quantity and its own held-out label -- gets
smaller on data built with no real signal in it at all.
"""
import numpy as np
import pytest

from eracoef.rloocv import (RebalancedLeaveOneGroupOut, loo_mean_shift, rebalance_partners,
                            rebalanced_splits)


def _loo_mean(label, w, keep):
    return float((w[keep] * label[keep]).sum() / w[keep].sum())


# ------------------------------------------------------------------ the identity the whole thing rests on
def test_leaving_a_row_out_moves_the_mean_away_from_it():
    """(A): p_loo(j) - pbar == -n_j (p_j - pbar) / (S_n - n_j), exactly.  No noise, no model."""
    rng = np.random.default_rng(0)
    label, w = rng.normal(100, 12, 40), rng.uniform(50, 150, 40)
    S, W = (w * label).sum(), w.sum()
    M = S / W
    for j in range(40):
        keep = np.arange(40) != j
        lhs = _loo_mean(label, w, keep) - M
        rhs = -w[j] * (label[j] - M) / (W - w[j])
        assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-12)


# ------------------------------------------------------------------ the partner rule
def test_the_partner_moves_the_mean_back_without_crossing_it():
    rng = np.random.default_rng(1)
    label, w = rng.normal(0, 1, 30), rng.uniform(1, 4, 30)
    sh = loo_mean_shift(label, w)
    pj = sh["partner"]
    assert sh["dropped"] > 0
    for j in np.flatnonzero(pj >= 0):
        assert abs(sh["rebalanced"][j]) < abs(sh["plain"][j])                 # closer to the full mean
        assert np.sign(sh["rebalanced"][j]) * np.sign(sh["plain"][j]) >= 0    # and did not cross it
    for j in np.flatnonzero(pj < 0):
        assert sh["rebalanced"][j] == pytest.approx(sh["plain"][j])           # left as plain LOO
    assert sh["max_abs_rebalanced"] < sh["max_abs_plain"]
    assert (pj != np.arange(len(pj))).all()                                   # never its own partner


def test_a_fold_never_partners_with_itself_and_tiny_problems_opt_out():
    assert (rebalance_partners([1.0, 2.0]) == -1).all()            # fewer than three folds: no partner
    p = rebalance_partners(np.arange(10.0))
    assert (p != np.arange(10)).all()


def test_groups_make_the_fold_a_season_not_a_row():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(12), 25)
    label = rng.normal(0, 1, groups.size) + np.repeat(rng.normal(0, 0.6, 12), 25)
    w = rng.uniform(1, 3, groups.size)
    pj = rebalance_partners(label, w, groups)
    assert pj.size == 12                                            # one entry per GROUP, not per row
    sh = loo_mean_shift(label, w, groups)
    assert sh["max_abs_rebalanced"] < sh["max_abs_plain"]
    # every row of a group is held out together
    for train, test in rebalanced_splits(label, w, groups):
        assert np.unique(groups[test]).size == 1


# ------------------------------------------------------------------ the splitter
def test_the_test_folds_still_partition_the_data():
    """The partner is removed from TRAIN only.  If it were dropped from the test side too, a
    cross_val_predict over this splitter would silently leave rows unpredicted."""
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(9), 11)
    y = rng.normal(0, 1, groups.size)
    cv = RebalancedLeaveOneGroupOut()
    seen = []
    for train, test in cv.split(np.zeros((groups.size, 2)), y, groups):
        assert not set(train) & set(test)
        seen.append(test)
    allseen = np.concatenate(seen)
    assert np.array_equal(np.sort(allseen), np.arange(groups.size))
    assert cv.get_n_splits(groups=groups) == 9


def test_rebalance_false_is_plain_leave_one_group_out():
    groups = np.repeat(np.arange(8), 5)
    y = np.arange(groups.size, dtype=float)
    for train, test in RebalancedLeaveOneGroupOut(rebalance=False).split(None, y, groups):
        assert len(train) + len(test) == groups.size          # nothing extra dropped
    got = [len(tr) for tr, _ in RebalancedLeaveOneGroupOut().split(None, y, groups)]
    assert min(got) < groups.size - 5                          # rebalancing does drop a partner


def test_the_splitter_drops_the_partner_from_training_only():
    groups = np.repeat(np.arange(10), 4)
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, groups.size)
    pj = rebalance_partners(y, None, groups)
    for g, (train, test) in enumerate(RebalancedLeaveOneGroupOut().split(None, y, groups)):
        missing = set(np.unique(groups)) - set(np.unique(groups[train]))
        assert g in missing
        assert missing == ({g, int(pj[g])} if pj[g] >= 0 else {g})


# ------------------------------------------------------------------ the artifact itself
def test_it_shrinks_the_spurious_correlation_on_data_with_no_signal():
    """Groups drawn from one distribution, so a group's own mean carries NO information about the
    others: the honest correlation between a leave-one-group-out mean and the held-out group's own mean
    is zero.  Plain LOO manufactures a negative one; rebalancing should shrink it."""
    rng = np.random.default_rng(7)
    n_g, per = 24, 30
    groups = np.repeat(np.arange(n_g), per)
    label = rng.normal(0, 1, n_g * per)                 # iid: no between-group structure at all
    w = rng.uniform(0.5, 4.0, n_g * per)
    own = np.array([np.average(label[groups == g], weights=w[groups == g]) for g in range(n_g)])
    sh = loo_mean_shift(label, w, groups)
    r_plain = np.corrcoef(sh["plain"], own)[0, 1]
    r_reb = np.corrcoef(sh["rebalanced"], own)[0, 1]
    assert r_plain < -0.9                                # the artifact, and it is nearly deterministic
    assert r_reb > r_plain                               # rebalancing pulls it back toward zero
    assert abs(r_reb) < abs(r_plain)


def test_teamloo_still_gets_the_same_partners_it_always_did():
    """`teamloo.rebalance_partners` is now an import from here; one game per fold must be unchanged."""
    from eracoef.teamloo import rebalance_partners as tl
    rng = np.random.default_rng(11)
    label, w = rng.normal(110, 9, 41), rng.uniform(90, 110, 41)
    assert np.array_equal(tl(label, w), rebalance_partners(label, w))
