"""Player archetypes fitted from the box profile, and the board's disagreement with an outside blend per type.

`tests/test_vs_consensus.py` guarded one hand-built axis, `bigness` = per-36 (orb + blk + 0.3 drb - 0.5 ast -
0.4 fg3m), because that is where the board's original defect ran.  A formula with five chosen weights in it is
a guess about what a player type is, and it answers only the question it was shaped for.  This fits the types
instead: a Bayesian Gaussian mixture over the standardised per-36 box profile, which prunes the components it
does not need, and then reports the gap between the board and the blend per cluster.

The statistic is `spread(...)`: the standard deviation, across clusters, of the mean TOTAL gap.  Read the
total first.  A cluster whose offensive and defensive gaps are large and opposite is an attribution
disagreement -- the two sources agree about how good that type of player is and disagree about which side of
the ball it comes from -- and only the total says a rating is wrong.  On 2024-2026 the three-season board
reads 0.229 and the single-season board 0.133, which is the opposite ranking to the `bigness` axis.

Model layer: everything here takes frames.  The caller loads the box scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The per-36 profile the mixture clusters on: volume, shot diet, playmaking, the glass, defensive events.
# Every column is in boxtable.season_box's schema, so the caller hands that frame straight in.
RATES = ["pts", "fg3m", "fg3_miss", "ftm", "ast", "orb", "drb", "stl", "blk", "tov", "pf"]


def per36(box: pd.DataFrame, rates: list | None = None) -> pd.DataFrame:
    """One row per player: his per-36 rates over whatever rows `box` holds.  `minutes` is kept."""
    rates = list(rates or RATES)
    g = box.groupby("player_id", as_index=False)[["minutes", *rates]].sum()
    for c in rates:
        g[c] = g[c] / g.minutes.clip(lower=1) * 36.0
    return g


def fit_clusters(X: np.ndarray, k: int = 8, seed: int = 0, n_init: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Standardise, fit a Bayesian Gaussian mixture, return (label per row, weight per component).

    Bayesian, not plain EM, so `k` is an upper bound rather than a choice: a component the data does not
    support has its weight driven toward zero instead of splitting a real cluster in two."""
    from sklearn.mixture import BayesianGaussianMixture

    Z = np.asarray(X, dtype=float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    bgm = BayesianGaussianMixture(n_components=int(k), covariance_type="full",
                                  weight_concentration_prior=1.0 / int(k), max_iter=1000,
                                  n_init=int(n_init), random_state=int(seed)).fit(Z)
    return bgm.predict(Z), bgm.weights_


def cluster_gaps(labels: np.ndarray, gaps: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean gap per cluster.  `gaps`: columns `gap_off`, `gap_def`, `gap_total`, one row per player, in
    `labels` order.  `extra`: any per-player columns to average beside them (the per-36 rates, a name)."""
    d = gaps.reset_index(drop=True).assign(cluster=np.asarray(labels))
    if extra is not None:
        d = pd.concat([d, extra.reset_index(drop=True)], axis=1)
    num = [c for c in d.columns if c != "cluster" and pd.api.types.is_numeric_dtype(d[c])]
    out = d.groupby("cluster", as_index=False)[num].mean()
    out.insert(1, "n", d.groupby("cluster").size().to_numpy())
    return out.sort_values("gap_total").reset_index(drop=True)


def spread(table: pd.DataFrame, col: str = "gap_total", min_n: int = 10) -> float:
    """The statistic: sd across clusters of the mean gap, over clusters with at least `min_n` players.

    Clusters below `min_n` are dropped rather than weighted -- a five-player component's mean gap is noise,
    and the mixture produces one whenever `k` is set above what the data supports."""
    v = table.loc[table.n >= int(min_n), col].to_numpy(dtype=float)
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def gap_columns(ours: pd.DataFrame, theirs: pd.DataFrame, pairs=(("rating_off", "adj_offense"),
                                                                 ("rating_def", "adj_defense"),
                                                                 ("rating_total", "adj_overall"))) -> pd.DataFrame:
    """`gap_<side>` = our z minus their z, side by side.  Both are standardised because the two sources are
    on different scales: the board's offensive spread is 1.19 per 100 against the blend's 1.96, and an
    unstandardised difference would read that as disagreement about every player."""
    def z(v):
        v = pd.Series(np.asarray(v, dtype=float))
        return (v - v.mean()) / v.std()

    return pd.DataFrame({f"gap_{side}": z(ours[mine]) - z(theirs[yours])
                         for (mine, yours), side in zip(pairs, ("off", "def", "total"))})
