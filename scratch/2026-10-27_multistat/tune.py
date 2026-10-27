"""Choose training settings on the VALIDATION seasons only (11-12); the test seasons are not scored here."""
import sys
import time

import numpy as np

import baseline as B
import model as M
from run_experiment import TRAIN_LAST, VAL_LAST, prepare, to_values
from sim_league import simulate


def val_scores(d, P, pred, rows):
    e = P["e"]
    return np.array([B.nll(j, pred[:, j], d, e, rows).mean() for j in range(13)])


if __name__ == "__main__":
    d = simulate(seed=0)
    P = prepare(d)
    played = P["played"]
    tr = np.flatnonzero(played & (d["season"] <= TRAIN_LAST))
    va_mask = played & (d["season"] > TRAIN_LAST) & (d["season"] <= VAL_LAST)
    va = np.flatnonzero(va_mask)
    nb = val_scores(d, P, P["bp"], va_mask).sum()
    nt = val_scores(d, P, d["true"], va_mask).sum()
    configs = [
        dict(name="lr 5e-4, wd 3e-4", lr=5e-4, widths=(256, 256), wd=3e-4, batch=4096),
        dict(name="lr 5e-4, wd 1e-3", lr=5e-4, widths=(256, 256), wd=1e-3, batch=4096),
        dict(name="lr 5e-4, wd 3e-3", lr=5e-4, widths=(256, 256), wd=3e-3, batch=4096),
        dict(name="lr 5e-4, wd 1e-4, dropout 0.1", lr=5e-4, widths=(256, 256), wd=1e-4, batch=4096, drop=0.1),
        dict(name="lr 5e-4, wd 1e-4, dropout 0.3", lr=5e-4, widths=(256, 256), wd=1e-4, batch=4096, drop=0.3),
        dict(name="lr 1e-3, wd 1e-3, 512x512", lr=1e-3, widths=(512, 512), wd=1e-3, batch=4096),
        dict(name="as run: lr 2e-3, 256x256", lr=2e-3, widths=(256, 256), wd=1e-5, batch=4096),
        dict(name="lr 5e-4", lr=5e-4, widths=(256, 256), wd=1e-5, batch=4096),
        dict(name="lr 5e-4, batch 16384", lr=5e-4, widths=(256, 256), wd=1e-5, batch=16384),
        dict(name="lr 5e-4, wd 1e-4", lr=5e-4, widths=(256, 256), wd=1e-4, batch=4096),
        dict(name="lr 5e-4, 128x128", lr=5e-4, widths=(128, 128), wd=1e-5, batch=4096),
        dict(name="lr 5e-4, 512x512", lr=5e-4, widths=(512, 512), wd=1e-5, batch=4096),
        dict(name="lr 2e-4, 256x256x256", lr=2e-4, widths=(256, 256, 256), wd=1e-5, batch=4096),
    ]
    only = sys.argv[1:] and [int(a) for a in sys.argv[1:]]
    for i, c in enumerate(configs):
        if only and i not in only:
            continue
        t0 = time.perf_counter()
        params, hist = M.train(P["data"], tr, va, widths=c["widths"], lr=c["lr"], wd=c["wd"], batch=c["batch"],
                               max_epochs=60, patience=5, seed=0, verbose=False,
                               drop=c.get("drop", 0.0))
        net = to_values(M.predict_link(params, P["data"][0], P["data"][1]))
        nn = val_scores(d, P, net, va_mask).sum()
        best_ep = int(np.argmin([h[2] for h in hist]))
        print(f"  [{i}] {c['name']:28s} validation gap closed {(nb - nn) / (nb - nt):6.1%}  "
              f"best pass {best_ep:2d} of {len(hist):2d}  {time.perf_counter() - t0:4.0f} s", flush=True)
