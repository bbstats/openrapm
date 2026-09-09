"""Does player-grouped cross-fitting kill the identity gain?

FINDINGS 26.2: fine height + weight is the largest offline gain any column has produced here (-0.14 on the
defensive prior's own leave-window-out fit) and it COSTS +0.20 on the criterion at z 4.7.  The proposed
mechanism is memorisation: the fine pair names the player, and the model finds his OWN other rows in the
training set and reads the target off them instead of learning a box-line-to-value map.

This tests that mechanism directly, at the shipped defensive operating point (config features_full_D,
params_def, gbdt_target_def, gbdt_win_decay_def) under the bench's leave-window-out protocol
(scratch/prior_bench.py), with one thing changed:

    off      the shipped fit: each window's model trains on every other window's rows, the scored player's
             own among them
    row      CONTROL: the same number of rows held out at random, so the training set is 10% smaller but the
             player's own rows are still in it
    player   the scoring player's every row is held out, so no feature can identify him to a row of his own

The gap between `off` and `row` is only the smaller training set.  The gap between `row` and `player` is
memorisation.  `pid` (the raw player id as a feature) is the positive control: it must look brilliant under
`row` and worthless under `player`, or the test has no power.

What this does NOT fix: a feature that predicts how QUIET a row's target is (career experience, FINDINGS
22.2) survives any fold structure, because that is a property of the row and not of memorisation.

    python scratch/foldtest.py [--side=D] [--folds=10] [--sets=base,hw,hwb,pid]
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from eracoef import gbdt_prior as G
from eracoef.config import load_config


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


cfg = load_config()
SIDE = flag("side", "D").upper()
NFOLD = int(flag("folds", 10))
P = pd.read_parquet(f"{cfg['_root']}/outputs/role_panel.parquet")
P["pid"] = P.player_id.astype(float)                 # the positive control, a pure identifier
PR = cfg["ratings_prior"]
GB = cfg["gbdt"]
base = list(GB["features_full_D"] if SIDE == "D" else GB["features_full_O"])
base = [f for f in base if f not in G.PAST]          # pooled rows refuse the past block
SETS = {
    "base": base,                                    # exactly what ships on this side
    "hw": [*base, "height", "weight"],               # the fine pair: 26.2's largest offline gain
    "hwb": [*base, "height2", "weight15"],           # the same, binned: the form that ships elsewhere
    "pid": [*base, "pid"],                           # the positive control
}
want = (flag("sets") or "base,hw,hwb,pid").split(",")
params = dict(GB["params_def"] if SIDE == "D" else GB["params"])
tc = PR["gbdt_target_def"] if SIDE == "D" else PR["gbdt_target"]
if tc.startswith("blend"):
    w = float(tc[5:])
    P = P.assign(**{tc: w * P["apm"].to_numpy(float) + (1 - w) * P["rapm1"].to_numpy(float)})
tcol = tc if tc != "rapm1" else "rapm1"
win_decay = float(PR["gbdt_win_decay_def"] if SIDE == "D" else PR["gbdt_win_decay"])
wins = sorted(P.window.unique())
# one stable fold per player, shared by every feature set and window, so the only thing that changes between
# the columns is whether the scored player's rows were in the training set
ids = np.sort(P.player_id.unique())
FOLD = pd.Series(np.arange(len(ids)) % NFOLD, index=ids)
print(f"side {SIDE}, target {tcol}, win_decay {win_decay}, {len(ids)} players in {NFOLD} folds, "
      f"booster depth {params.get('depth')} lr {params.get('learning_rate'):.3g}", flush=True)


def run(feats, mode: str):
    ref = G.reference_mean(P, SIDE, target_col=tcol, win_decay=win_decay)
    rng = np.random.default_rng(11)
    pred, truth, wt = [], [], []
    for lab in wins:
        rows = G.training_rows(P, SIDE, {lab}, feats, target_col=tcol, win_decay=win_decay)
        rows, _ = G.counterbalance(rows, ref, 0.02)
        held = G.training_rows(P, SIDE, (), feats, target_col=tcol, win_decay=win_decay)
        held = held[held.window == lab]
        X_h = held[feats].to_numpy(dtype=float)
        if mode == "off":
            m = G.fit_gbdt(rows, feats, seed=0, thread_count=6, **params)
            p = np.asarray(m.predict(X_h), dtype=float)
        else:
            if mode == "player":
                hf = held.player_id.map(FOLD).to_numpy()
                rf = rows.player_id.map(FOLD).to_numpy()
            else:                                    # "row": the same training size, his rows still there
                hf = rng.integers(0, NFOLD, len(held))
                rf = rng.integers(0, NFOLD, len(rows))
            p = np.zeros(len(held))
            for f in range(NFOLD):
                sel = hf == f
                if not sel.any():
                    continue
                m = G.fit_gbdt(rows[rf != f], feats, seed=0, thread_count=6, **params)
                p[sel] = np.asarray(m.predict(X_h[sel]), dtype=float)
        pred.append(p)
        truth.append(held["target"].to_numpy(dtype=float))
        wt.append(held["weight"].to_numpy(dtype=float))
    p, y, w = np.concatenate(pred), np.concatenate(truth), np.concatenate(wt)
    mse = float(np.average((y - p) ** 2, weights=w))
    pm, ym = np.average(p, weights=w), np.average(y, weights=w)
    cov = float(np.average((p - pm) * (y - ym), weights=w))
    vp = float(np.average((p - pm) ** 2, weights=w))
    return mse, (cov / vp if vp > 0 else np.nan), float(np.sqrt(vp))


t0, out = time.time(), []
for name in want:
    for mode in ("off", "row", "player"):
        mse, slope, sd = run(SETS[name], mode)
        out.append(dict(set=name, folds=mode, mse=mse, slope=slope, sd_pred=sd, n_feat=len(SETS[name])))
        print(f"  {name:5s} folds {mode:6s}: mse {mse:.4f}  slope {slope:.3f}  sd(pred) {sd:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

R = pd.DataFrame(out)
R.to_csv(f"{cfg['_root']}/outputs/csv/foldtest_{SIDE}.csv", index=False)
piv = R.pivot(index="set", columns="folds", values="mse")
cols = [c for c in ("off", "row", "player") if c in piv.columns]
for c in cols:
    piv[f"gain {c}"] = piv[c] - piv.loc["base", c]
print("\n=== what each block buys the prior's own fit, under three training regimes\n")
print("  gain = its mse minus base's IN THE SAME COLUMN (negative = it helps)\n")
print(piv[cols + [f"gain {c}" for c in cols]].round(4).to_string())
if {"hw", "base"} <= set(piv.index) and {"row", "player"} <= set(cols):
    g_row, g_pl = piv.loc["hw", "gain row"], piv.loc["hw", "gain player"]
    keep = 100.0 * g_pl / g_row if g_row else np.nan
    print(f"\n  the FINE pair keeps {keep:.0f}% of its gain when no model has seen the player "
          f"({g_row:+.4f} -> {g_pl:+.4f}), against the equal-sized random control.")
if "pid" in piv.index and {"row", "player"} <= set(cols):
    print(f"  the positive control `pid`: {piv.loc['pid', 'gain row']:+.4f} with his rows in, "
          f"{piv.loc['pid', 'gain player']:+.4f} with them out.  If the second is not ~0 the test has no power.")
