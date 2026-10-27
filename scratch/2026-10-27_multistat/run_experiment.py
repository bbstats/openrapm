"""Simulate a league, fit the per-stat baseline, train the multi-stat network, score both on held-out seasons.

Seasons 0-10 train, 11-12 validate (early stopping), 13-14 test: nothing fitted, standardized or chosen has seen
the test seasons.  Every number below is over the test seasons' games played.

Scores
    NLL gain      baseline NLL minus network NLL per game, per stat (above 0: the network predicts better)
    gap closed    (baseline NLL - network NLL) / (baseline NLL - truth NLL): the share of what a perfect predictor of
                  the true rates would gain over the baseline that the network gains
    error vs truth  root of the exposure-weighted mean squared difference between a prediction and the true rate, in
                  the stat's units (per 100 possessions, or percentage points); the reduction is 1 - network / baseline
"""
from __future__ import annotations

import sys
import time

import numpy as np

import baseline as B
import model as M
from inputs import build_inputs, input_names
from sim_league import IS_PCT, STATS, simulate

TRAIN_LAST, VAL_LAST = 10, 12


def prepare(d, ks=None, x0=None, train_last=TRAIN_LAST, taus=None):
    played = d["poss"] > 0
    train = played & (d["season"] <= train_last)
    if ks is None:
        ks, x0 = B.fit(d, train)
    bp = B.predict(d, ks, x0)
    X = build_inputs(d, bp, x0) if taus is None else build_inputs(d, bp, x0, taus=taus)
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    e = B.exposures(d)
    b = np.clip(bp, 1e-9, None)
    off = np.where(IS_PCT, np.log(np.clip(b, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(b, 1e-6, 1 - 1e-6)), np.log(b))
    log_scale = np.log(np.maximum(d["poss"], 1e-9) / 100.0)[:, None] * np.ones((1, 13))
    att = np.where(IS_PCT, e, 0.0)
    data = (Xs, off.astype(np.float32), d["obs"].astype(np.float32), att.astype(np.float32), log_scale.astype(np.float32))
    return dict(data=data, bp=bp, ks=ks, x0=x0, X=X, mu=mu, sd=sd, e=e, played=played)


def to_values(link):
    return np.where(IS_PCT, 1.0 / (1.0 + np.exp(-link)), np.exp(link))


def score(d, P, pred, rows):
    """Per stat: NLL, and exposure-weighted root mean squared error against the true rate."""
    e = P["e"]
    out = {}
    for j in range(13):
        nll = B.nll(j, pred[:, j], d, e, rows).mean()
        w = e[rows, j]
        diff = pred[rows, j] - d["true"][rows, j]
        out[j] = (nll, np.sqrt((w * diff ** 2).sum() / w.sum()))
    return out


def report(d, P, net_pred, rows, label="test seasons"):
    sb, sn, st = score(d, P, P["bp"], rows), score(d, P, net_pred, rows), score(d, P, d["true"], rows)
    print(f"\n{label}: {rows.sum():,} games played")
    print(f"  {'stat':8s} {'NLL gain':>9s} {'gap closed':>10s} {'baseline err':>12s} {'network err':>11s} {'reduction':>9s}")
    tot_b = tot_n = tot_t = 0.0
    for j, s in enumerate(STATS):
        gain = sb[j][0] - sn[j][0]
        gap = sb[j][0] - st[j][0]
        scale = 100.0 if IS_PCT[j] else 1.0
        tot_b += sb[j][0]; tot_n += sn[j][0]; tot_t += st[j][0]
        print(f"  {s:8s} {gain:9.5f} {gain / gap:10.1%} {sb[j][1] * scale:12.3f} {sn[j][1] * scale:11.3f} "
              f"{1 - sn[j][1] / sb[j][1]:9.1%}")
    print(f"  {'all 13':8s} {tot_b - tot_n:9.5f} {(tot_b - tot_n) / (tot_b - tot_t):10.1%}")
    return sb, sn, st


def main(seed=0, net_seed=0):
    t0 = time.perf_counter()
    d = simulate(seed=seed)
    P = prepare(d)
    played = P["played"]
    tr = np.flatnonzero(played & (d["season"] <= TRAIN_LAST))
    va = np.flatnonzero(played & (d["season"] > TRAIN_LAST) & (d["season"] <= VAL_LAST))
    print(f"league seed {seed}: {len(tr):,} training games, {len(va):,} validation games; "
          f"setup {time.perf_counter() - t0:.1f} s")
    t1 = time.perf_counter()
    params, hist = M.train(P["data"], tr, va, seed=net_seed, max_epochs=60, patience=5, **M.BEST)
    print(f"  trained in {time.perf_counter() - t1:.0f} s")
    net = to_values(M.predict_link(params, P["data"][0], P["data"][1]))
    test = played & (d["season"] > VAL_LAST)
    report(d, P, net, test)
    return d, P, params, net


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
