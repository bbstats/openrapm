"""The linear box-score prior of experiment 121 (linear_prior.py): the descent on the Gram matrix gives scikit-learn's
elastic net, the sums give the rows' own numbers, and a line fitted without a group of rows never reads them."""
import numpy as np
import pytest

from openrapm import linear_prior as lp


def _problem(n=400, p=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    y = x @ rng.normal(size=p) + 0.3 * x[:, 0] * x[:, 1] + rng.normal(scale=0.5, size=n)
    w = rng.uniform(0.2, 3.0, size=n)
    return x, y, w


def _direct_sums(x, y, w):
    return lp.Sums(x.T @ (x * w[:, None]), x.T @ w, x.T @ (w * y), float(w.sum()), float(w @ y), float(w @ (y * y)))


@pytest.mark.parametrize("alpha,l1", [(0.05, 0.5), (0.01, 0.9), (0.2, 0.1)])
def test_descent_matches_scikit_learn(alpha, l1):
    from sklearn.linear_model import ElasticNet
    x, y, w = _problem()
    coef, intercept, _ = lp.solve(_direct_sums(x, y, w), alpha, l1)
    theirs = ElasticNet(alpha=alpha, l1_ratio=l1, tol=1e-12, max_iter=200000).fit(x, y, sample_weight=w)
    assert np.allclose(coef, theirs.coef_, atol=1e-6)
    assert intercept == pytest.approx(theirs.intercept_, abs=1e-6)


def test_compiled_descent_matches_the_plain_one():
    x, y, w = _problem(seed=1)
    covariance, cross, _, _ = _direct_sums(x, y, w).centred()
    fast, _ = lp._descend(covariance, cross, 0.01, 0.02, np.zeros(cross.size), 50000, 1e-12)
    slow, _ = lp._descend_python(covariance, cross, 0.01, 0.02, np.zeros(cross.size), 50000, 1e-12)
    assert np.allclose(fast, slow, atol=1e-10)


def test_expansion_flags_products_and_standardizes():
    x, _, w = _problem(n=200, p=3, seed=2)
    x[::7, 1] = np.nan
    expansion = lp.fit_expansion(x, w)
    assert list(expansion.flags) == [1]
    columns = expansion.transform(x)
    assert columns.shape == (200, 3 + 1 + 6)            # three inputs, one flag, six products with the squares
    mean = w @ columns / w.sum()
    spread = np.sqrt(w @ (columns - mean) ** 2 / w.sum())
    assert np.allclose(mean, 0.0, atol=1e-9) and np.allclose(spread, 1.0, atol=1e-9)
    assert np.isfinite(columns).all()


def test_sums_are_the_rows_own_numbers():
    x, y, w = _problem(n=300, p=4, seed=3)
    expansion = lp.fit_expansion(x, w)
    groups = np.arange(300) % 3
    parts = lp.row_sums(expansion, x, y, w, groups, 3)
    columns = expansion.transform(x)
    for group, part in enumerate(parts):
        rows = groups == group
        direct = _direct_sums(columns[rows], y[rows], w[rows])
        assert np.allclose(part.gram, direct.gram) and np.allclose(part.xy, direct.xy)
        coef = np.linspace(-0.2, 0.3, columns.shape[1])
        miss = y[rows] - 0.1 - columns[rows] @ coef
        assert lp.held_out_error(part, coef, 0.1) == pytest.approx(float(w[rows] @ miss ** 2), rel=1e-9)


def test_a_line_without_a_group_never_reads_its_rows():
    x, y, w = _problem(n=300, p=4, seed=4)
    expansion = lp.fit_expansion(x, w)
    groups = np.arange(300) % 3
    parts = lp.row_sums(expansion, x, y, w, groups, 3)
    total = parts[0] + parts[1] + parts[2]
    without = lp.line_from_sums(expansion, total - parts[1], 0.02, 0.5)
    y_changed = y.copy()
    y_changed[groups == 1] += 100.0                       # the held-out rows' labels cannot reach the line
    changed = lp.row_sums(expansion, x, y_changed, w, groups, 3)
    again = lp.line_from_sums(expansion, changed[0] + changed[2], 0.02, 0.5)
    assert np.allclose(without.coef, again.coef, atol=1e-9)
    assert without.intercept == pytest.approx(again.intercept, abs=1e-9)


def test_column_pick_reads_its_columns():
    class Echo:
        def predict(self, matrix):
            return matrix[:, 0] * 10 + matrix[:, 1]
    pick = lp.ColumnPick(Echo(), np.array([2, 0]))
    assert list(pick.predict(np.array([[1.0, 5.0, 3.0]]))) == [31.0]
