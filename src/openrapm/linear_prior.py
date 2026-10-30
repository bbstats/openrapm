"""The box-score prior as a straight line on every input and every product of two inputs (experiment 121; the owner,
2026-10-30: "I want to try maybe combining algorithms? So I want to use a linear model so like elastic net or
something, and I want to do combination features in the linear model, combine that with the gradient boosting model,
and test a couple weightings of the two", then "the feature combinations is basically just Max of two interactions,
everything (including things we have eliminated via boruta)").

Words used below
    input         one column the line reads: a box-score rate, a role, a play-by-play count, a matchup average, the
                  row's possessions and seasons; missing where its data does not reach, as the hustle counts before 2017
    standardized  minus its mean over the training rows, over its standard deviation, both weighted by the rows'
                  weights; a missing value reads 0, the mean, and a flag column (1 where missing) says it was missing
    product       two standardized inputs multiplied, every pair and every input with itself, then standardized again
    elastic net   the coefficients b and intercept c that minimize, over the training rows weighted by w,
                      sum of w (y - c - x b)^2 / (2 sum of w) + alpha (l1 sum of |b| + (1 - l1) / 2 sum of b^2),
                  the objective of scikit-learn's ElasticNet with sample weights; solved coordinate by coordinate on
                  the columns' weighted cross-products (their Gram matrix), never on the rows themselves
    sums          what a set of rows contributes to the Gram matrix and its neighbours (`Sums`); the sums of a set of
                  players' rows are subtracted from the training rows' to fit without them, so a held-out player's
                  line never saw his label

The standardization and the flags are read off all of a fit's training rows, inputs only: a held-out player's inputs
shift a column's mean and spread by a fraction of a percent, never his label.

Example: 61 inputs and the two row columns make 63 standardized inputs, 2,016 products and a flag for each input that
is ever missing, about 2,100 columns.  69,000 training rows make one 2,100 x 2,100 Gram matrix, about 15 seconds; every
fit after that works on the matrix alone, a few seconds each.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LEARNER", "Expansion", "Sums", "LinePrior", "ColumnPick", "fit_expansion", "row_sums", "solve",
           "line_from_sums", "held_out_error", "alpha_max"]

LEARNER = "expanded_elastic_net"
CHUNK = 8000                     # rows expanded at a time: 8,000 x 2,100 doubles is about 135 MB
TOLERANCE = 1e-8                 # the largest coefficient move, standardized units, at which the descent stops
MAX_EPOCHS = 20000


@dataclass
class Expansion:
    """How raw inputs become the line's columns: each input standardized (missing to 0), a flag per input ever missing
    in training, every product of two standardized inputs; every column then standardized, and columns with no spread
    in training dropped."""
    means: np.ndarray            # per input
    scales: np.ndarray
    flags: np.ndarray            # the inputs that get a missing flag
    pair_i: np.ndarray           # the products, input i times input j, i <= j
    pair_j: np.ndarray
    column_means: np.ndarray     # per column of the expanded matrix, before the drop
    column_scales: np.ndarray
    keep: np.ndarray             # the columns kept

    @property
    def width(self) -> int:
        return int(self.keep.sum())

    def raw(self, matrix: np.ndarray) -> np.ndarray:
        """The expanded columns before their own standardization: standardized inputs, flags, products."""
        x = np.asarray(matrix, dtype=float)
        missing = np.isnan(x)
        z = np.where(missing, 0.0, (x - self.means) / self.scales)
        return np.hstack([z, missing[:, self.flags].astype(float), z[:, self.pair_i] * z[:, self.pair_j]])

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        out = (self.raw(matrix) - self.column_means) / self.column_scales
        return out[:, self.keep]


def _weighted_moments(values: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = weight.sum()
    mean = weight @ values / total
    variance = weight @ (values - mean) ** 2 / total
    return mean, np.sqrt(np.maximum(variance, 0.0))


def fit_expansion(matrix: np.ndarray, weight: np.ndarray) -> Expansion:
    """The expansion read off a fit's training rows: weighted means and spreads of every input and every column."""
    x = np.asarray(matrix, dtype=float)
    w = np.asarray(weight, dtype=float)
    present = ~np.isnan(x)
    means = np.array([w[present[:, k]] @ x[present[:, k], k] / w[present[:, k]].sum() if present[:, k].any() else 0.0
                      for k in range(x.shape[1])])
    centred = np.where(present, x - means, 0.0)
    variance = np.array([w[present[:, k]] @ centred[present[:, k], k] ** 2 / w[present[:, k]].sum()
                         if present[:, k].any() else 0.0 for k in range(x.shape[1])])
    scales = np.where(variance > 0, np.sqrt(variance), 1.0)
    flags = np.flatnonzero((~present).any(axis=0))
    pair_i, pair_j = np.triu_indices(x.shape[1])
    partial = Expansion(means, scales, flags, pair_i, pair_j, np.zeros(0), np.ones(0), np.zeros(0, dtype=bool))
    total = 0.0
    first = second = None
    for start in range(0, x.shape[0], CHUNK):
        block = partial.raw(x[start:start + CHUNK])
        wb = w[start:start + CHUNK]
        first = wb @ block if first is None else first + wb @ block
        second = wb @ block ** 2 if second is None else second + wb @ block ** 2
        total += wb.sum()
    column_means = first / total
    column_variance = np.maximum(second / total - column_means ** 2, 0.0)
    keep = column_variance > 1e-12
    return Expansion(means, scales, flags, pair_i, pair_j, column_means,
                     np.where(keep, np.sqrt(column_variance), 1.0), keep)


@dataclass
class Sums:
    """A set of rows' contribution, in the expansion's columns X, label y and weights w: X'WX, X'w, X'Wy, the sum of
    w, of w y and of w y^2.  Sums add and subtract like the rows they come from."""
    gram: np.ndarray
    xw: np.ndarray
    xy: np.ndarray
    w: float
    wy: float
    wyy: float

    def __add__(self, other: "Sums") -> "Sums":
        return Sums(self.gram + other.gram, self.xw + other.xw, self.xy + other.xy, self.w + other.w,
                    self.wy + other.wy, self.wyy + other.wyy)

    def __sub__(self, other: "Sums") -> "Sums":
        return Sums(self.gram - other.gram, self.xw - other.xw, self.xy - other.xy, self.w - other.w,
                    self.wy - other.wy, self.wyy - other.wyy)

    def centred(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """(the weighted covariance of the columns, of the columns with y, the columns' weighted means, y's)."""
        mean = self.xw / self.w
        mean_y = self.wy / self.w
        return self.gram / self.w - np.outer(mean, mean), self.xy / self.w - mean * mean_y, mean, mean_y


def row_sums(expansion: Expansion, matrix: np.ndarray, label: np.ndarray, weight: np.ndarray, groups: np.ndarray,
             n_groups: int) -> list:
    """[Sums of each group's rows], `groups` a group number per row (a player's fold), one pass over the rows."""
    width = expansion.width
    out = [Sums(np.zeros((width, width)), np.zeros(width), np.zeros(width), 0.0, 0.0, 0.0) for _ in range(n_groups)]
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(label, dtype=float)
    w = np.asarray(weight, dtype=float)
    g = np.asarray(groups)
    for start in range(0, x.shape[0], CHUNK):
        block = expansion.transform(x[start:start + CHUNK])
        yb, wb, gb = y[start:start + CHUNK], w[start:start + CHUNK], g[start:start + CHUNK]
        for group in np.unique(gb):
            rows = gb == group
            xb, wr, yr = block[rows], wb[rows], yb[rows]
            out[group] = out[group] + Sums(xb.T @ (xb * wr[:, None]), xb.T @ wr, xb.T @ (wr * yr), float(wr.sum()),
                                           float(wr @ yr), float(wr @ (yr * yr)))
    return out


def _descend_python(gram, q, l1_penalty, l2_penalty, coef, max_epochs, tolerance):
    gb = gram @ coef
    for epoch in range(max_epochs):
        largest = 0.0
        for j in range(q.shape[0]):
            old = coef[j]
            r = q[j] - gb[j] + gram[j, j] * old
            new = np.sign(r) * max(abs(r) - l1_penalty, 0.0) / (gram[j, j] + l2_penalty)
            if new != old:
                gb += (new - old) * gram[:, j]
                coef[j] = new
                largest = max(largest, abs(new - old))
        if largest < tolerance:
            return coef, epoch + 1
    return coef, max_epochs


try:                               # the same descent compiled; numba ships with the project's environment
    import numba

    @numba.njit(cache=True)
    def _descend(gram, q, l1_penalty, l2_penalty, coef, max_epochs, tolerance):
        m = q.shape[0]
        gb = gram @ coef
        for epoch in range(max_epochs):
            largest = 0.0
            for j in range(m):
                old = coef[j]
                r = q[j] - gb[j] + gram[j, j] * old
                shrunk = abs(r) - l1_penalty
                new = 0.0
                if shrunk > 0.0:
                    new = (shrunk if r > 0.0 else -shrunk) / (gram[j, j] + l2_penalty)
                if new != old:
                    step = new - old
                    for k in range(m):
                        gb[k] += step * gram[k, j]
                    coef[j] = new
                    if abs(step) > largest:
                        largest = abs(step)
            if largest < tolerance:
                return coef, epoch + 1
        return coef, max_epochs
except ImportError:                # pragma: no cover
    _descend = _descend_python


def solve(sums: Sums, alpha: float, l1: float, coef: np.ndarray | None = None) -> tuple[np.ndarray, float, int]:
    """(coefficients, intercept, epochs) of the elastic net on `sums` at `alpha` and `l1`, from `coef` when given (a
    warm start along a path of alphas)."""
    covariance, cross, mean, mean_y = sums.centred()
    start = np.zeros(cross.shape[0]) if coef is None else np.array(coef, dtype=float)
    out, epochs = _descend(np.ascontiguousarray(covariance), cross, alpha * l1, alpha * (1.0 - l1), start,
                           MAX_EPOCHS, TOLERANCE)
    return out, float(mean_y - mean @ out), int(epochs)


def alpha_max(sums: Sums, l1: float) -> float:
    """The smallest alpha at which every coefficient is 0."""
    _, cross, _, _ = sums.centred()
    return float(np.abs(cross).max() / max(l1, 1e-3))


def held_out_error(sums: Sums, coef: np.ndarray, intercept: float) -> float:
    """The weighted squared miss of the line on the rows of `sums`, summed, from the sums alone."""
    return float(sums.wyy - 2 * coef @ sums.xy - 2 * intercept * sums.wy + coef @ sums.gram @ coef
                 + 2 * intercept * (coef @ sums.xw) + intercept ** 2 * sums.w)


@dataclass
class LinePrior:
    """One fitted line: answers like a booster, `predict(matrix)` on the raw inputs in the order it was fitted."""
    expansion: Expansion
    coef: np.ndarray
    intercept: float

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return self.intercept + self.expansion.transform(matrix) @ self.coef


def line_from_sums(expansion: Expansion, sums: Sums, alpha: float, l1: float) -> LinePrior:
    coef, intercept, _ = solve(sums, alpha, l1)
    return LinePrior(expansion, coef, intercept)


@dataclass
class ColumnPick:
    """A booster fitted on some of a wider matrix's columns: `predict` reads those columns, in its order."""
    model: object
    columns: np.ndarray

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(matrix, dtype=float)[:, self.columns])
