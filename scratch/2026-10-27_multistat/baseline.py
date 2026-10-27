"""The baseline: the per-stat recursion, each of the 13 stats on its own, with k fitted per stat.

    w          = e / (e + k)                  e = the game's exposure: possessions for a rate, attempts for a percentage
    pred_{t+1} = w x observed_t + (1 - w) x pred_t,     pred_0 = the league value of the stat

Words used below
    NLL        negative log-likelihood of one game's outcome under a prediction, without the terms that do not
               depend on the prediction:
                   a rate (count y over p possessions, mean m = pred x p / 100):   m - y log m
                   a percentage (made y of a attempts, probability pi):              -(y log pi + (a - y) log(1 - pi))
               Summed over games it is the quantity both the baseline's k and the network are fitted to, and lower
               is better.  A game with no exposure (no possessions, or no attempts for a percentage) adds 0.

Example: a 30-possession game with 2 assists, predicted at 5.0 per 100: m = 1.5, NLL = 1.5 - 2 log 1.5 = 0.689.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar

from sim_league import ATTEMPTS_OF, IS_PCT, STATS

EPS = 1e-9


def exposures(d):
    """(rows x 13) each stat's exposure: possessions for a rate, attempts for a percentage."""
    e = np.repeat(d["poss"][:, None], 13, axis=1).astype(np.float64)
    for jp, ja in ATTEMPTS_OF.items():
        e[:, jp] = d["obs"][:, ja]
    return e


def observed(d, e):
    """(rows x 13) each stat's observed value: per 100 possessions for a rate, makes / attempts for a percentage;
    0 where the exposure is 0 (never read there)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(IS_PCT, d["obs"] / e, 100.0 * d["obs"] / e)
    return np.where(e > 0, v, 0.0)


@njit(cache=True)
def recursion(player, e, v, k, x0):
    """Pre-game predictions of one stat for every row (rows sorted by player, then time)."""
    n = player.size
    pred = np.empty(n)
    x = x0
    for i in range(n):
        if i == 0 or player[i] != player[i - 1]:
            x = x0
        pred[i] = x
        if e[i] > 0:
            w = e[i] / (e[i] + k)
            x = x + w * (v[i] - x)
    return pred


def nll(j, pred, d, e, rows):
    """Per-row NLL of stat j (see the module docstring) on the rows selected by `rows`."""
    y = d["obs"][rows, j]
    if IS_PCT[j]:
        pi = np.clip(pred[rows], EPS, 1 - EPS)
        a = e[rows, j]
        return -(y * np.log(pi) + (a - y) * np.log1p(-pi))
    m = np.maximum(pred[rows], EPS) * d["poss"][rows] / 100.0
    return m - y * np.log(m)


def league_values(d, e, rows):
    """The league value of each stat over `rows`: exposure-weighted."""
    v = 100.0 * d["obs"][rows].sum(0) / d["poss"][rows].sum()
    for jp, ja in ATTEMPTS_OF.items():
        v[jp] = d["obs"][rows, jp].sum() / d["obs"][rows, ja].sum()
    return v


def fit(d, train_rows, bounds=(1.0, 1e5)):
    """k per stat minimizing that stat's NLL on the training rows; x0 = the league value on the training rows."""
    e = exposures(d)
    v = observed(d, e)
    x0 = league_values(d, e, train_rows & (d["poss"] > 0))
    ks = np.empty(13)
    for j in range(13):
        def loss(lk, j=j):
            p = recursion(d["player"], e[:, j], v[:, j], np.exp(lk), x0[j])
            return nll(j, p, d, e, train_rows).mean()
        r = minimize_scalar(loss, bounds=np.log(bounds), method="bounded", options={"xatol": 1e-3})
        ks[j] = np.exp(r.x)
    return ks, x0


def predict(d, ks, x0):
    """(rows x 13) the baseline's pre-game predictions."""
    e = exposures(d)
    v = observed(d, e)
    return np.column_stack([recursion(d["player"], e[:, j], v[:, j], ks[j], x0[j]) for j in range(13)])


if __name__ == "__main__":
    import time
    from sim_league import simulate
    d = simulate()
    played = d["poss"] > 0
    train = played & (d["season"] <= 10)
    test = played & (d["season"] >= 13)
    t0 = time.perf_counter(); ks, x0 = fit(d, train); t = time.perf_counter() - t0
    pred = predict(d, ks, x0)
    e = exposures(d)
    print(f"fitted k per stat in {t:.1f} s (possessions for rates, attempts for percentages):")
    for j, s in enumerate(STATS):
        base = nll(j, pred[:, j], d, e, test).mean()
        orc = nll(j, d["true"][:, j], d, e, test).mean()
        print(f"  {s:8s} k = {ks[j]:8.1f}   test NLL baseline {base:.5f}  truth {orc:.5f}")
