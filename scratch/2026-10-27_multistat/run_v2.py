"""The full-size test: a 30-season league (OpenRAPM's span, 1997-2026), baseline vs network vs ceiling.

Seasons 0-24 train, 25-27 validate (early stopping), 28-29 test.  Settings were chosen on the 15-season league's
validation seasons (tune.py); nothing here is chosen on the test seasons.

Rows of the report, per stat, over the test seasons' games played:
    gain          baseline NLL minus the model's NLL per game (above 0: better than the per-stat recursion)
    gap closed    gain / (baseline NLL - truth NLL)
    err red.      1 - (model's error vs the true rate) / (baseline's), error = root exposure-weighted mean square
"""
from __future__ import annotations

import sys
import time

import numpy as np

import baseline as B
import model as M
from kf_ceiling import kf_predict
from run_experiment import prepare, score, to_values
from sim_league import IS_PCT, STATS, simulate

N_SEASONS, TRAIN_LAST, VAL_LAST = 30, 24, 27
TAUS = (100.0, 400.0, 1500.0, 5000.0)
CONFIG = M.BEST


def nll_by_stat(d, P, pred, rows):
    return np.array([B.nll(j, pred[:, j], d, P["e"], rows).mean() for j in range(13)])


def err_by_stat(d, P, pred, rows):
    s = score(d, P, pred, rows)
    return np.array([s[j][1] for j in range(13)])


def table(d, P, preds, rows, title):
    nb, nt = nll_by_stat(d, P, P["bp"], rows), nll_by_stat(d, P, d["true"], rows)
    eb = err_by_stat(d, P, P["bp"], rows)
    res = {n: (nll_by_stat(d, P, p, rows), err_by_stat(d, P, p, rows)) for n, p in preds.items()}
    print(f"\n{title}: {rows.sum():,} games played")
    print(f"  {'':8s}  {'gap closed':^{12 * len(res)}s}  {'error vs truth reduction':^{12 * len(res)}s}")
    print(f"  {'stat':8s}  " + "".join(f"{n[:11]:>12s}" for n in res) + "  " + "".join(f"{n[:11]:>12s}" for n in res))
    for j, s in enumerate(STATS):
        gaps = "".join(f"{(nb[j] - r[0][j]) / (nb[j] - nt[j]):12.1%}" for r in res.values())
        errs = "".join(f"{1 - r[1][j] / eb[j]:12.1%}" for r in res.values())
        print(f"  {s:8s}  {gaps}  {errs}")
    print(f"  {'all 13':8s}  " + "".join(f"{(nb.sum() - r[0].sum()) / (nb.sum() - nt.sum()):12.1%}" for r in res.values()))


def main(seed=0, n_ens=5, n_seasons=N_SEASONS, train_last=TRAIN_LAST, val_last=VAL_LAST, max_epochs=60):
    t0 = time.perf_counter()
    d = simulate(n_seasons=n_seasons, seed=seed)
    P = prepare(d, train_last=train_last, taus=TAUS)
    played = P["played"]
    tr = np.flatnonzero(played & (d["season"] <= train_last))
    va = np.flatnonzero(played & (d["season"] > train_last) & (d["season"] <= val_last))
    test = played & (d["season"] > val_last)
    print(f"league seed {seed}: {len(d['poss']):,} player-games, {len(tr):,} training games, {len(va):,} validation, "
          f"{test.sum():,} test; {P['data'][0].shape[1]} inputs; setup {time.perf_counter() - t0:.0f} s", flush=True)
    nets = []
    for k in range(n_ens):
        t1 = time.perf_counter()
        params, hist = M.train(P["data"], tr, va, max_epochs=max_epochs, patience=5, seed=k, verbose=False, **CONFIG)
        nets.append(to_values(M.predict_link(params, P["data"][0], P["data"][1])))
        print(f"  network {k}: best pass {int(np.argmin([h[2] for h in hist]))} of {len(hist)}, "
              f"{time.perf_counter() - t1:.0f} s", flush=True)
        if k == 0:
            # everything prediction needs: the layers (arr_0 = W1, arr_1 = b1, ...), the input standardization and
            # the baseline's k and starting values (its predictions are the network's offset and 13 of its inputs)
            np.savez_compressed(f"net_seed{seed}.npz", *[np.asarray(a) for W in params for a in W],
                                mu=P["mu"], sd=P["sd"], ks=P["ks"], x0=P["x0"], taus=np.asarray(TAUS))
    t1 = time.perf_counter()
    kp = kf_predict(d)
    print(f"  Kalman ceiling: {time.perf_counter() - t1:.0f} s", flush=True)
    preds = {"network": nets[0], f"ensemble of {n_ens}": np.mean(nets, axis=0), "ceiling": kp}
    va_mask = np.zeros_like(played); va_mask[va] = True
    table(d, P, preds, va_mask, f"validation seasons {train_last + 1}-{val_last}")
    table(d, P, preds, test, f"TEST seasons {val_last + 1}-{n_seasons - 1}")
    # noise control: the spread of single networks' overall gap closed on the test seasons
    nb, nt = nll_by_stat(d, P, P["bp"], test).sum(), nll_by_stat(d, P, d["true"], test).sum()
    singles = [(nb - nll_by_stat(d, P, n, test).sum()) / (nb - nt) for n in nets]
    print(f"\n  single networks, test gap closed by seed: " + ", ".join(f"{g:.1%}" for g in singles))
    # by experience: career possessions before the game
    car = np.expm1(P["X"][:, [i for i, nm in enumerate(__import__("inputs").input_names(TAUS))
                              if nm == "career_log_poss"][0]])
    print("\n  test gap closed by career possessions before the game:")
    print(f"  {'tier':>14s} {'games':>8s} {'network':>9s} {'ensemble':>9s} {'ceiling':>9s}")
    for lo, hi in [(0, 500), (500, 2000), (2000, 6000), (6000, 1e12)]:
        rows = test & (car >= lo) & (car < hi)
        nb_, nt_ = nll_by_stat(d, P, P["bp"], rows).sum(), nll_by_stat(d, P, d["true"], rows).sum()
        g = [(nb_ - nll_by_stat(d, P, preds[n], rows).sum()) / (nb_ - nt_) for n in preds]
        lab = f"{lo:,.0f}-{hi:,.0f}" if hi < 1e12 else f"{lo:,.0f}+"
        print(f"  {lab:>14s} {rows.sum():8,d} " + " ".join(f"{x:9.1%}" for x in g))
    keep = np.flatnonzero(test)                          # test-season games only, float32: about 20 MB, not 400
    np.savez_compressed(f"preds_seed{seed}.npz", rows=keep, player=d["player"][keep], season=d["season"][keep],
                        poss=d["poss"][keep].astype(np.float32), obs=d["obs"][keep].astype(np.float32),
                        **{n: a[keep].astype(np.float32) for n, a in
                           [("baseline", P["bp"]), ("network", nets[0]), ("ensemble", np.mean(nets, axis=0)),
                            ("ceiling", kp), ("true", d["true"])]})
    return d, P, preds


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0, n_ens=int(sys.argv[2]) if len(sys.argv) > 2 else 5)
