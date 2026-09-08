"""Trade calibration, step 2: the trade-weighted SPM -- which box-score stats port when the teammates change?

    .venv/Scripts/python scratch/trade_spm.py [O|D] [--target=apm|rapm1|blend0.7] [--dist=1|all] [--pen=1.0]

Rows are ORDERED PAIRS of the panel's windows (w -> w'): the player's box line in w, his target in w', the
possessions behind the target as the weight (discounted by 0.5 per window of distance beyond the first when
--dist=all), and the teammate turnover of w' with respect to w (outputs/csv/turnover_windows.csv).

Two weighted ridges, leave-window-out (the held-out window is out of BOTH ends of every training pair):
  base    y = b . x
  trade   y = b . x + g0 turn + (g . x) turn         x standardised, turn in [0, 1]
so g_j is how much stat j's value changes per unit of turnover, in points per 100 per standard deviation of
the stat, and g0 is the level shift of a fully turned-over player that no stat explains (selection, not
portability).  Reported: the pooled leave-window-out MSE of both, the per-window paired difference, and g
with a cluster-bootstrap z (resampling players).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eracoef import gbdt_prior as G  # noqa: E402

pd.set_option("display.width", 200, "display.precision", 4)


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


SIDE = next((a for a in sys.argv[1:] if a in ("O", "D")), "O")
TARGET = flag("target", "apm")
DIST = flag("dist", "1")
PEN = float(flag("pen", "1.0"))
DECAY = 0.5
FEATS = [f for f in G.FULL_FEATURES if f != "season"] + ["usage", "astr", "tovr", "ts", "p3r", "ftr", "orbsh"]

panel = pd.read_parquet(ROOT / "outputs" / "role_panel.parquet")
if TARGET.startswith("blend"):
    wgt = float(TARGET[5:])
    panel[TARGET] = wgt * panel.apm + (1 - wgt) * panel.rapm1
p = panel[panel.side == SIDE].copy()
G.add_derived(p)
wins = sorted(p.window.unique())
idx = {w: i for i, w in enumerate(wins)}
turn = pd.read_csv(ROOT / "outputs" / "csv" / "turnover_windows.csv")

# ---- pair rows
feat = p[["player_id", "window", *FEATS]]
tgt = p[["player_id", "window", "poss", TARGET]].rename(columns={"window": "window_to", "poss": "poss_to", TARGET: "y"})
rows = feat.merge(tgt, on="player_id")
rows = rows[rows.window != rows.window_to]
rows["dist"] = (rows.window_to.map(idx) - rows.window.map(idx)).abs()
if DIST != "all":
    rows = rows[rows.dist <= int(DIST)]
rows = rows.merge(turn[["player_id", "window", "window_to", "turnover"]], on=["player_id", "window", "window_to"], how="inner")
rows["weight"] = rows.poss_to * DECAY ** np.maximum(rows.dist - 1, 0)
rows = rows[(rows.weight > 0) & rows.turnover.notna()].reset_index(drop=True)
print(f"side {SIDE}, target {TARGET}, pairs at distance {DIST}: {len(rows)} rows, {rows.player_id.nunique()} players, "
      f"turnover mean {np.average(rows.turnover, weights=rows.weight):.3f}")

X0 = rows[FEATS].to_numpy(dtype=float)
w_all = rows.weight.to_numpy(dtype=float)
mu = np.average(X0, axis=0, weights=w_all)
sd = np.sqrt(np.average((X0 - mu) ** 2, axis=0, weights=w_all))
sd[sd == 0] = 1.0
Xs = (X0 - mu) / sd
t = rows.turnover.to_numpy(dtype=float)
y = rows.y.to_numpy(dtype=float)
tc = t - np.average(t, weights=w_all)          # centred, so b . x keeps its meaning at the average turnover


def design(Xs, tc, trade: bool):
    ones = np.ones((len(Xs), 1))
    if not trade:
        return np.hstack([ones, Xs])
    return np.hstack([ones, Xs, tc[:, None], Xs * tc[:, None]])


def ridge(A, y, w, pen):
    P = pen * np.eye(A.shape[1])
    P[0, 0] = 0.0
    AtW = (A * w[:, None]).T
    return np.linalg.solve(AtW @ A + P, AtW @ y)


def lwo(trade: bool):
    """Leave-window-out predictions: the held-out window is out of both ends of every training pair."""
    pred = np.full(len(rows), np.nan)
    A = design(Xs, tc, trade)
    for wlab in wins:
        test = (rows.window == wlab).to_numpy()
        train = ~((rows.window == wlab) | (rows.window_to == wlab)).to_numpy()
        beta = ridge(A[train], y[train], w_all[train], PEN)
        pred[test] = A[test] @ beta
    return pred


t0 = time.time()
p0, p1 = lwo(False), lwo(True)
e0, e1 = (y - p0) ** 2, (y - p1) ** 2
print(f"\nleave-window-out weighted MSE:  base {np.average(e0, weights=w_all):.4f}   trade {np.average(e1, weights=w_all):.4f}   "
      f"diff {np.average(e1 - e0, weights=w_all):+.4f}")
per = rows.assign(d=e1 - e0).groupby("window").apply(lambda d: np.average(d.d, weights=d.weight), include_groups=False)
z = per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))
print(f"per-window paired: mean {per.mean():+.4f}, z {z:+.2f}, trade wins {(per < 0).sum()}/{len(per)}")
# the same, split by turnover: does the trade model help where the turnover is high?
for lo, hi in ((0.0, 0.5), (0.5, 0.9), (0.9, 1.01)):
    m = (t >= lo) & (t < hi)
    if m.sum():
        print(f"  turnover [{lo:.1f}, {hi:.1f}): n={m.sum():5d}  base {np.average(e0[m], weights=w_all[m]):.4f}  "
              f"trade {np.average(e1[m], weights=w_all[m]):.4f}  diff {np.average((e1 - e0)[m], weights=w_all[m]):+.4f}")

# ---- the coefficients on all rows, with a cluster bootstrap over players
A = design(Xs, tc, True)
beta = ridge(A, y, w_all, PEN)
k = len(FEATS)
rng = np.random.default_rng(0)
pids = rows.player_id.to_numpy()
upid, inv = np.unique(pids, return_inverse=True)
B = 200
boots = np.zeros((B, len(beta)))
for b in range(B):
    draw = rng.integers(0, len(upid), len(upid))
    cnt = np.bincount(draw, minlength=len(upid))
    wb = w_all * cnt[inv]
    boots[b] = ridge(A, y, wb, PEN)
se = boots.std(axis=0, ddof=1)
tab = pd.DataFrame({"stat": FEATS, "b_at_mean_turn": beta[1:k + 1], "g_per_turn": beta[k + 2:], "se": se[k + 2:]})
tab["z"] = tab.g_per_turn / tab.se
tab["value_if_stayed(t=0.35)"] = tab.b_at_mean_turn + (0.35 - np.average(t, weights=w_all)) * tab.g_per_turn
tab["value_if_traded(t=1.0)"] = tab.b_at_mean_turn + (1.0 - np.average(t, weights=w_all)) * tab.g_per_turn
print(f"\nturnover level g0 = {beta[k + 1]:+.3f} (se {se[k + 1]:.3f}, z {beta[k + 1] / se[k + 1]:+.2f}) points per 100 "
      f"per unit of turnover, holding the box line fixed (selection, not portability)")
print("\ncoefficients in points per 100 per standard deviation of the stat (raw sign; positive = good on offense, "
      "and on defense too, since the panel is raw sign on both sides)")
print(tab.sort_values("z", key=np.abs, ascending=False).to_string(index=False))
print(f"\n{time.time() - t0:.0f}s")
