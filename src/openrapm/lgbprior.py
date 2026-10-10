"""LightGBM as the box-score prior's learner, and the leave-one-player-out prior built with it.

The owner, 2026-10-08 ("go! (lightgbm mode)"), after experiment 43 measured where the prior's training-group noise
comes from: about half of the defensive spread across groupings was chimeraboost's own randomness (its row and column
subsampling and its random early-stopping split), the rest which players were left out with the rated one.  This
learner removes the first part by construction and leave-one-player-out removes the second:

* **Deterministic.**  No row or column subsampling, a fixed number of trees (no early-stopping split), one thread per
  fit and `deterministic=True`, so the same training rows always give the same model (`FIXED`, applied over whatever
  settings are passed).  Measured on the 2015 rows: five seeds give identical predictions.
* **Weights normalised to mean one** inside every fit, so `reg_lambda` and the minimum leaf weight mean the same thing
  whatever the possession scale of the rows.  The tuning screen (scripts/136) and the build fit through `fit` alike.
* **Leave one player out** (`fit_leave_one_out`): one model per rated player, each trained on every other player's
  rows, fitted in parallel processes (one thread each; about 0.8 s a fit on 34,000 rows).
"""
from __future__ import annotations

import numpy as np

FIXED = dict(subsample=1.0, subsample_freq=0, colsample_bytree=1.0, n_jobs=1, deterministic=True,
             force_row_wise=True, verbose=-1, random_state=0)


def make(params: dict):
    """An unfitted LGBMRegressor with `params`, made deterministic (`FIXED` wins over anything passed)."""
    import lightgbm as lgb
    return lgb.LGBMRegressor(**{**params, **FIXED})


def fit(params: dict, X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """One deterministic fit; the weights rescaled to mean one."""
    w = np.asarray(w, dtype=float)
    return make(params).fit(X, y, sample_weight=w / w.mean())


def _fit_without(params: dict, X: np.ndarray, y: np.ndarray, w: np.ndarray, players: np.ndarray, player):
    keep = players != player
    return fit(params, X[keep], y[keep], w[keep])


def _fit_predict_without(params: dict, X: np.ndarray, y: np.ndarray, w: np.ndarray, players: np.ndarray, player,
                         Xq: np.ndarray) -> np.ndarray:
    return _fit_without(params, X, y, w, players, player).predict(Xq)


def predict_leave_one_out(params: dict, X: np.ndarray, y: np.ndarray, w: np.ndarray, players: np.ndarray,
                          Xq: np.ndarray, q_players: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    """Each query row predicted by the model fitted without its player's training rows; NaN where the player has
    no training rows (the caller asks the full fit).  One model per player, fitted, asked about his query rows and
    dropped inside its worker, so nothing large comes back (a model can be 14 MB; 500 of them would not fit)."""
    players, q_players = np.asarray(players), np.asarray(q_players)
    have = set(players.tolist())
    tasks = [(p, np.flatnonzero(q_players == p)) for p in dict.fromkeys(q_players.tolist()) if p in have]
    if n_jobs <= 1:
        got = [_fit_predict_without(params, X, y, w, players, p, Xq[idx]) for p, idx in tasks]
    else:
        from joblib import Parallel, delayed
        got = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fit_predict_without)(params, X, y, w, players, p, Xq[idx]) for p, idx in tasks)
    out = np.full(len(q_players), np.nan)
    for (_, idx), pred in zip(tasks, got):
        out[idx] = pred
    return out


def fit_leave_one_out(params: dict, X: np.ndarray, y: np.ndarray, w: np.ndarray, players: np.ndarray,
                      needed, n_jobs: int = 1) -> dict:
    """{player: the model fitted on every row that is not his}, for each player in `needed` who has rows.

    Fitted in `n_jobs` processes (joblib's loky backend; the arrays are memory-mapped to the workers once)."""
    players = np.asarray(players)
    have = set(players.tolist())
    todo = [p for p in dict.fromkeys(np.asarray(needed).tolist()) if p in have]
    if n_jobs <= 1:
        models = [_fit_without(params, X, y, w, players, p) for p in todo]
    else:
        from joblib import Parallel, delayed
        models = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fit_without)(params, X, y, w, players, p) for p in todo)
    return dict(zip(todo, models))
