"""Does the PLUS-MINUS part of a rating travel worse than the BOX part when the teammates change?

    .venv/Scripts/python scratch/trade_resid.py [O|D]

A player's rating in window w is box_w (what the boosted box prior says his stat line is worth, fitted
leave-window-out) plus resid_w (what his on/off data moved it: rapm1_w - box_w).  On ordered adjacent window
pairs (w -> w'), weighted by the possessions behind the target in w', with the teammate turnover of w' with
respect to w:

    y_w' = a + b_box * box_w + b_pm * resid_w + turn * (c0 + c_box * box_w + c_pm * resid_w)

so per unit of turnover the box part keeps (b_box + c_box) / b_box of its value and the plus-minus part
(b_pm + c_pm) / b_pm.  The owner's claim is c_pm < c_box.  Cluster bootstrap over players for the z.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eracoef import gbdt_prior as G  # noqa: E402
from eracoef.config import load_config  # noqa: E402

cfg = load_config()
SIDE = next((a for a in sys.argv[1:] if a in ("O", "D")), "O")
TARGET = "blend0.7" if SIDE == "O" else "rapm1"
gcfg = cfg["gbdt"]
FEATS = list(gcfg["features_full_O"] if SIDE == "O" else gcfg["features_full_D"])
PARAMS = {**dict(gcfg["params_def"]), "thread_count": int(gcfg.get("thread_count", 3))}   # the cheap booster, both sides

panel = pd.read_parquet(ROOT / "outputs" / "role_panel.parquet")
panel["blend0.7"] = 0.7 * panel.apm + 0.3 * panel.rapm1
p = panel[panel.side == SIDE].copy()
G.add_derived(p)
wins = sorted(p.window.unique())
idx = {w: i for i, w in enumerate(wins)}

# 1. the box part of every player-window, leave-window-out (the pooled prior exactly as shipped)
p["box"] = np.nan
for w in wins:
    rows = G.training_rows(p, SIDE, exclude={w}, features=FEATS, target_col=TARGET, win_decay=0.5)
    m = G.fit_gbdt(rows, FEATS, seed=0, **PARAMS)
    sel = p.window == w
    p.loc[sel, "box"] = m.predict(p.loc[sel, FEATS].to_numpy(dtype=float))
p["resid"] = p.rapm1 - p.box              # what the on/off data added to the box read
p["resid_apm"] = p.apm - p.box            # the same with the unshrunk plus-minus

# 2. adjacent pairs with turnover
turn = pd.read_csv(ROOT / "outputs" / "csv" / "turnover_windows.csv")
left = p[["player_id", "window", "box", "resid", "resid_apm", "poss"]]
right = p[["player_id", "window", "poss", TARGET]].rename(columns={"window": "window_to", "poss": "poss_to", TARGET: "y"})
rows = left.merge(right, on="player_id")
rows = rows[(rows.window != rows.window_to) & ((rows.window_to.map(idx) - rows.window.map(idx)).abs() == 1)]
rows = rows.merge(turn, on=["player_id", "window", "window_to"], how="inner")
rows = rows[(rows.poss_to > 0) & rows.turnover.notna() & (rows.poss >= 1500)].reset_index(drop=True)
w = rows.poss_to.to_numpy(dtype=float)
y = rows.y.to_numpy(dtype=float)
t = rows.turnover.to_numpy(dtype=float)
print(f"side {SIDE}, target {TARGET}: {len(rows)} adjacent pairs, {rows.player_id.nunique()} players with 1500+ possessions in w")
print(f"  box part sd {np.sqrt(np.average((rows.box - np.average(rows.box, weights=w)) ** 2, weights=w)):.3f}, "
      f"plus-minus part sd {np.sqrt(np.average((rows.resid - np.average(rows.resid, weights=w)) ** 2, weights=w)):.3f}")


def fit(A, y, w):
    AtW = (A * w[:, None]).T
    return np.linalg.solve(AtW @ A + 1e-6 * np.eye(A.shape[1]), AtW @ y)


pids = rows.player_id.to_numpy()
upid, inv = np.unique(pids, return_inverse=True)
rng = np.random.default_rng(0)
for label, rcol in (("shrunk rating minus box (rapm1 - box)", "resid"), ("raw plus-minus minus box (apm - box)", "resid_apm")):
    box, res = rows.box.to_numpy(dtype=float), rows[rcol].to_numpy(dtype=float)
    A = np.column_stack([np.ones(len(y)), box, res, t, t * box, t * res])
    beta = fit(A, y, w)
    boots = np.zeros((300, 6))
    for b in range(300):
        cnt = np.bincount(rng.integers(0, len(upid), len(upid)), minlength=len(upid))
        boots[b] = fit(A, y, w * cnt[inv])
    se = boots.std(axis=0, ddof=1)
    diff = boots[:, 5] - boots[:, 4]
    b_box, b_pm, c0, c_box, c_pm = beta[1:]
    print(f"\n  plus-minus part = {label}")
    print(f"    a stayer (turnover 0):  box part counts {b_box:+.3f}   plus-minus part counts {b_pm:+.3f}")
    print(f"    per unit of turnover:   box part changes {c_box:+.3f} (z {c_box / se[4]:+.2f})   "
          f"plus-minus part changes {c_pm:+.3f} (z {c_pm / se[5]:+.2f})   level {c0:+.3f} (z {c0 / se[3]:+.2f})")
    keep_box = (b_box + c_box) / b_box if b_box else np.nan
    keep_pm = (b_pm + c_pm) / b_pm if b_pm else np.nan
    print(f"    fully traded (turnover 1): box part keeps {keep_box:.0%} of its weight, plus-minus part keeps {keep_pm:.0%}; "
          f"difference c_pm - c_box = {c_pm - c_box:+.3f} (z {(c_pm - c_box) / diff.std(ddof=1):+.2f})")
