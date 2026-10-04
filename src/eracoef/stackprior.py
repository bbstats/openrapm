"""The stacked prior: an elastic net on the plus-minus columns and a booster on everything else, blended.

The owner, 2026-10-02: "a stacking regressor of the +/- features into an elastic net, and all other features into
boruta -> chimeraboost quality=5 (so, 2 models ensembled)".

**What it is.**  sklearn's `StackingRegressor` with two base models and a linear blend on top:

  - the LINEAR half: `ElasticNetCV` on the plus-minus columns, `PLUS_MINUS`: his own season's on-court points
    per 100 on offence and defence (`singleyear.ONC`) and his team's with him OFF the court (`singleyear.OFFC`),
    padded, and the possessions behind each.  On/off (`NET`) is on minus off, so a linear model already has it.
    Standardised by the training rows' weighted mean and sd; its penalty and L1 share are chosen by its own
    cross-validation over players;
  - the BOOSTER half: chimeraboost at `quality=5` (a bag of eight) on every other column the caller passes;
  - the BLEND: a non-negative least-squares line, `blend = a + w_linear x linear + w_booster x booster`, fitted on
    each base model's OUT-OF-FOLD predictions, so it weighs the halves by how well they predict players they have
    not seen.  Non-negative because a negative weight on one half is the stack fitting noise between two
    correlated predictions, not a real trade.

**The one departure from sklearn, and why it is the same estimator.**  `StackingRegressor` cross-validates its
base models to get the out-of-fold predictions and then refits them on everything.  The prior already fits every
model once on everything and once per PLAYER fold, and hands each player the fold fit that never saw his rows
(`OutOfPlayerSPM` in scripts/62).  Those fold fits ARE the out-of-fold predictions the blend needs, so the stack
reuses them: the five folds that make the prior out-of-player are also the stack's `cv`.  sklearn's default `cv`
(five plain row folds) would be wrong here twice over -- a player's career row and chunk rows carry one label,
so a row fold leaks the label to the booster through his other rows and the blend would trust the booster for
remembering players, not for predicting them.

The blend's three numbers are fitted once, on all the out-of-fold predictions, and applied to the full fit and to
every fold fit alike; three numbers learned partly from a player's own rows is a leak too small to matter, and
nesting the stack inside each fold would cost five times the boosters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import singleyear as sy

__all__ = ["StackedSPM", "weighted_standardise", "PLUS_MINUS"]

# the owner, 2026-10-02: "did the elastic net not use off court data too?" -- on-court AND off-court
PLUS_MINUS = sy.ONC + sy.OFFC


def weighted_standardise(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The weighted mean and sd of each column (sd 1 where a column is constant)."""
    w = np.asarray(w, dtype=float) / np.sum(w)
    mean = w @ X
    sd = np.sqrt(w @ (X - mean) ** 2)
    return mean, np.where(sd > 0, sd, 1.0)


class _Linear:
    """`ElasticNetCV` on standardised columns, its own folds over players."""

    def __init__(self, l1_ratios, n_inner: int = 5):
        self.l1_ratios, self.n_inner = list(l1_ratios), int(n_inner)

    def fit(self, X, y, w, groups):
        from sklearn.linear_model import ElasticNetCV
        from sklearn.model_selection import GroupKFold

        self.mean_, self.sd_ = weighted_standardise(X, w)
        splits = list(GroupKFold(n_splits=self.n_inner).split(X, y, groups))
        self.model_ = ElasticNetCV(l1_ratio=self.l1_ratios, cv=splits, max_iter=20000, n_jobs=1)
        self.model_.fit((X - self.mean_) / self.sd_, y, sample_weight=w)
        return self

    def predict(self, X):
        return self.model_.predict((X - self.mean_) / self.sd_)

    def describe(self, names) -> str:
        coef = self.model_.coef_ / self.sd_          # per raw unit of each column
        return (f"alpha {self.model_.alpha_:.3g}, l1 {self.model_.l1_ratio_:g}; per raw unit "
                + ", ".join(f"{n} {c:+.4g}" for n, c in zip(names, coef)))


class StackedSPM:
    """The out-of-player prior as a two-model stack.  Same `fit(train, model_feats)` / `predict(frame,
    model_feats)` as scripts/62's `OutOfPlayerSPM`, so the rest of the pipeline (the cross-fitted fold priors
    included) cannot tell them apart.  `linear` names the columns that go to the elastic net; every other name
    in `model_feats` goes to the booster.  `n_folds` must be 2 or more: the folds are the stack's `cv`."""

    def __init__(self, booster_params: dict, n_folds: int, linear=tuple(PLUS_MINUS), quality: int | None = 5,
                 l1_ratios=(0.1, 0.5, 0.9, 1.0)):
        if int(n_folds) < 2:
            raise ValueError("the stack needs player folds (--player_folds=5): they are its cross-validation")
        params = dict(booster_params)
        if quality is not None:
            params.pop("n_ensembles", None)          # quality 4 / 5 sets the bag size itself
            params["quality"] = int(quality)
        params["ensemble_n_jobs"] = 1                # never fork inside a build
        self.params, self.n_folds = params, int(n_folds)
        self.linear, self.l1_ratios = list(linear), list(l1_ratios)

    def _split(self, model_feats):
        lin = [f for f in model_feats if f in self.linear]
        boost = [f for f in model_feats if f not in self.linear]
        if not lin or not boost:
            raise ValueError(f"the stack needs both halves: linear {lin}, booster {boost}")
        return lin, boost

    def _fit_pair(self, X_lin, X_boost, y, w, groups):
        from chimeraboost import ChimeraBoostRegressor

        lin = _Linear(self.l1_ratios).fit(X_lin, y, w, groups)
        boost = ChimeraBoostRegressor(random_state=0, **self.params).fit(X_boost, y, sample_weight=w)
        return lin, boost

    def fit(self, train: pd.DataFrame, model_feats: list) -> "StackedSPM":
        from sklearn.linear_model import LinearRegression

        self.lin_names_, self.boost_names_ = self._split(model_feats)
        X_lin = train[self.lin_names_].to_numpy(float)
        X_boost = train[self.boost_names_].to_numpy(float)
        y, w = train.target.to_numpy(float), train.weight.to_numpy(float)
        groups = train.index.to_numpy()
        self.full_ = self._fit_pair(X_lin, X_boost, y, w, groups)
        fold = sy.stratified_player_folds(train, self.n_folds)
        self.shift_ = sy.fold_mean_shift(train, fold)
        oof = np.full((len(train), 2), np.nan)
        self.fold_models_, self.excluded_ = [], []
        for f in range(self.n_folds):
            keep = fold != f
            pair = self._fit_pair(X_lin[keep], X_boost[keep], y[keep], w[keep], groups[keep])
            oof[~keep, 0] = pair[0].predict(X_lin[~keep])
            oof[~keep, 1] = pair[1].predict(X_boost[~keep])
            self.fold_models_.append(pair)
            self.excluded_.append(set(train.index[~keep].tolist()))
        self.blend_ = LinearRegression(positive=True).fit(oof, y, sample_weight=w)
        # how well each piece predicts players it never saw, weighted, for the log
        resid = {"linear": oof[:, 0], "booster": oof[:, 1], "stack": self.blend_.predict(oof)}
        self.oof_rmse_ = {k: float(np.sqrt(np.average((y - v) ** 2, weights=w))) for k, v in resid.items()}
        return self

    def _predict_pair(self, pair, X_lin, X_boost):
        return self.blend_.predict(np.column_stack([pair[0].predict(X_lin), pair[1].predict(X_boost)]))

    def predict(self, frame: pd.DataFrame, model_feats: list) -> np.ndarray:
        X_lin = frame[self.lin_names_].to_numpy(float)
        X_boost = frame[self.boost_names_].to_numpy(float)
        out = self._predict_pair(self.full_, X_lin, X_boost)
        ids = frame.player_id.to_numpy()
        for pair, excluded in zip(self.fold_models_, self.excluded_):
            idx = np.fromiter((p in excluded for p in ids), dtype=bool, count=len(ids))
            if idx.any():
                out[idx] = self._predict_pair(pair, X_lin[idx], X_boost[idx])
        return out

    def describe(self) -> str:
        a, (w_lin, w_boost) = self.blend_.intercept_, self.blend_.coef_
        r = self.oof_rmse_
        return (f"blend {a:+.3f} + {w_lin:.3f} x linear + {w_boost:.3f} x booster; out-of-fold error (weighted "
                f"rmse) linear {r['linear']:.3f}, booster {r['booster']:.3f}, stack {r['stack']:.3f}; linear "
                + self.full_[0].describe(self.lin_names_))
