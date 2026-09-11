"""Rebalanced leave-one-out: a cross-validator that does not move the mean under the fold it holds out.

Austin, Pe'er and Korem (*Distributional bias compromises leave-one-out cross-validation*, Science
Advances 2025).  Hold fold j out and the remaining weighted mean of the label shifts AWAY from fold j's
own label, by construction.  With weights n_j, the fold's mean label p_j and the full weighted mean pbar:

    p_loo(j) - pbar  ==  -n_j (p_j - pbar) / (S_n - n_j)                              (A)

identically -- no noise, no model, just arithmetic.  So anything computed from "everything except j"
carries a mechanical negative correlation with j's own label.  Train a model leave-one-season-out and it
is fitted on a training set whose mean is tilted away from the season you are about to predict; the
tilt is largest exactly where the fold is largest or the label most extreme.

The paper's fix, implemented here: drop one FURTHER fold alongside j, chosen so the remaining mean lands
back on the full mean without crossing it.  `rebalance_partners` picks it, `RebalancedLeaveOneGroupOut`
is the sklearn-shaped splitter that uses it, so it drops into `cross_val_predict`, `cross_val_score`,
`GridSearchCV` or a plain `for train, test in cv.split(...)` loop.

    from eracoef.rloocv import RebalancedLeaveOneGroupOut
    cv = RebalancedLeaveOneGroupOut()
    for train, test in cv.split(X, y, groups=panel.season):
        ...                                   # `train` is missing BOTH the test season and its partner

**It is a heuristic, and this project has measured it over-correcting.** On `teamloo`'s method-of-moments
estimate of between-team variance the rebalanced figure runs 9-23% above the split-half truth while plain
LOO agrees to 4-9%, so the constant there is taken from the method of moments and the rebalanced rates are
used only where the attenuation actually bites (the matchup regression).  Rebalance where a leave-one-out
QUANTITY feeds a regression; do not assume it is free.  The honest check is always the same one: does the
answer change, and does it change in the direction the bias predicts?

Cost of the correction: one fold's worth of training rows, per fold.  With 30 seasons that is ~3% of the
data, and `partner_of` returns -1 for a fold where no partner improves on leaving it out alone -- those
folds are left exactly as plain LOO.
"""
from __future__ import annotations

import numpy as np

__all__ = ["rebalance_partners", "RebalancedLeaveOneGroupOut", "rebalanced_splits", "loo_mean_shift"]


def _fold_totals(label, w, groups):
    """(keys, per-fold weighted label sum, per-fold weight sum, per-fold row index array)."""
    label = np.asarray(label, dtype=float)
    n = label.size
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    if w.size != n:
        raise ValueError(f"label and w disagree: {n} vs {w.size}")
    if groups is None:
        keys = np.arange(n)
        return keys, w * label, w.copy(), [np.array([i]) for i in range(n)]
    groups = np.asarray(groups)
    keys, inv = np.unique(groups, return_inverse=True)
    S = np.bincount(inv, weights=w * label, minlength=keys.size)
    W = np.bincount(inv, weights=w, minlength=keys.size)
    idx = [np.flatnonzero(inv == g) for g in range(keys.size)]
    return keys, S, W, idx


def rebalance_partners(label, w=None, groups=None) -> np.ndarray:
    """For each fold, the other fold whose removal alongside it puts the remaining weighted mean of
    `label` closest to the full weighted mean WITHOUT crossing it; -1 when no partner beats leaving the
    fold out alone.  The rule of Austin, Pe'er and Korem (2025) for a continuous label.

    `groups=None` makes every row its own fold (plain leave-one-OUT).  Otherwise folds are the distinct
    values of `groups`, in `np.unique` order, and the returned array is indexed by fold, not by row.

    The fold table is built outright, so this is O(folds^2) in memory: fine for 30 seasons or 82 games,
    not for 100k rows with `groups=None`.
    """
    _, S_f, W_f, _ = _fold_totals(label, w, groups)
    G = S_f.size
    if G < 3:
        return np.full(G, -1, dtype=np.int64)
    S, W = float(S_f.sum()), float(W_f.sum())
    M = S / W
    d1 = (S - S_f) / np.maximum(W - W_f, 1e-9) - M                 # what leaving the fold out alone does
    num = S - S_f[:, None] - S_f[None, :]
    den = W - W_f[:, None] - W_f[None, :]
    d2 = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), np.nan) - M
    np.fill_diagonal(d2, np.nan)
    # keep only partners that move the mean back toward the full mean without overshooting it
    ok = np.isfinite(d2) & (np.sign(d2) * np.sign(d1)[:, None] >= 0) & (np.abs(d2) < np.abs(d1)[:, None])
    cost = np.where(ok, np.abs(d2), np.inf)
    partner = np.argmin(cost, axis=1).astype(np.int64)
    return np.where(np.isfinite(cost[np.arange(G), partner]), partner, -1)


def loo_mean_shift(label, w=None, groups=None) -> dict:
    """How far the training mean moves under each fold, before and after rebalancing -- the diagnostic
    that says whether the correction was worth its cost.  `plain` and `rebalanced` are per fold, in the
    same order as `rebalance_partners`; `dropped` is how many folds got a partner."""
    _, S_f, W_f, _ = _fold_totals(label, w, groups)
    S, W = float(S_f.sum()), float(W_f.sum())
    M = S / W
    plain = (S - S_f) / np.maximum(W - W_f, 1e-9) - M
    pj = rebalance_partners(label, w, groups)
    take = pj >= 0
    k = np.maximum(pj, 0)
    num = S - S_f - np.where(take, S_f[k], 0.0)
    den = W - W_f - np.where(take, W_f[k], 0.0)
    reb = np.where(take, num / np.maximum(den, 1e-9) - M, plain)
    return dict(full_mean=M, plain=plain, rebalanced=reb, partner=pj, dropped=int(take.sum()),
                max_abs_plain=float(np.abs(plain).max()), max_abs_rebalanced=float(np.abs(reb).max()))


class RebalancedLeaveOneGroupOut:
    """sklearn-shaped CV splitter: leave one group out, and drop its rebalancing partner group too.

    `split(X, y, groups)` yields `(train_idx, test_idx)`.  `test_idx` is the held-out group exactly as
    `LeaveOneGroupOut` would give it -- the partner is removed from TRAIN only, never scored -- so the
    test folds still partition the data and a `cross_val_predict` over this splitter covers every row.

    The label the balance is struck on is `y` by default.  Pass `balance_on=` to use something else (the
    paper's rule is stated on the label; `teamloo` measured balancing on a component's own rate to be
    slightly better, which bounds how much of identity (A) a label-chosen partner removes).

        cv = RebalancedLeaveOneGroupOut()
        for train, test in cv.split(X, y, groups=panel.season):
            model.fit(X.iloc[train], y.iloc[train], sample_weight=w[train])
            pred[test] = model.predict(X.iloc[test])
    """

    def __init__(self, sample_weight=None, balance_on=None, rebalance: bool = True):
        self.sample_weight = sample_weight
        self.balance_on = balance_on
        self.rebalance = bool(rebalance)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if groups is None:
            raise ValueError("groups is required")
        return int(np.unique(np.asarray(groups)).size)

    def split(self, X=None, y=None, groups=None):
        if groups is None:
            raise ValueError("groups is required")
        groups = np.asarray(groups)
        lab = self.balance_on if self.balance_on is not None else y
        if lab is None:
            raise ValueError("need y (or balance_on) to choose a rebalancing partner")
        lab = np.asarray(lab, dtype=float)
        w = None if self.sample_weight is None else np.asarray(self.sample_weight, dtype=float)
        keys, _, _, idx = _fold_totals(lab, w, groups)
        pj = rebalance_partners(lab, w, groups) if self.rebalance else np.full(keys.size, -1, dtype=np.int64)
        n = groups.size
        for g in range(keys.size):
            test = idx[g]
            drop = np.zeros(n, dtype=bool)
            drop[test] = True
            if pj[g] >= 0:
                drop[idx[pj[g]]] = True              # the partner leaves TRAIN but is never tested
            yield np.flatnonzero(~drop), test


def rebalanced_splits(label, w=None, groups=None, rebalance: bool = True) -> list:
    """`RebalancedLeaveOneGroupOut` as a plain list of `(train_idx, test_idx)`, for a loop of your own."""
    cv = RebalancedLeaveOneGroupOut(sample_weight=w, balance_on=label, rebalance=rebalance)
    g = np.arange(np.asarray(label).size) if groups is None else groups
    return list(cv.split(None, label, g))
