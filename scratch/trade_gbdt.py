"""Trade calibration, step 4: the shipped booster on pair rows, with teammate turnover as a feature.

    .venv/Scripts/python scratch/trade_gbdt.py [O|D] [--target=blend0.7|rapm1|apm] [--dist=all|1] [--cheap]

Rows are ordered window pairs (w -> w') as in trade_spm.py: the player's shipped feature line in w, his target
in w', weight = the possessions behind the target times 0.5 per window of distance beyond the first, and the
teammate turnover of w' with respect to w.  Two boosters per held-out window, the held-out window out of both
ends of every training pair: the shipped feature list, and the same plus `turn`.  Reported: the leave-window-out
weighted MSE of both, overall and by turnover bin; and the TRADE DELTA of the turn model -- its prediction at
turn = 1.0 minus at turn = 0.35 for every player-window -- with its spread and what it correlates with.
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
from eracoef.config import load_config  # noqa: E402

pd.set_option("display.width", 200, "display.precision", 4)


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


cfg = load_config()
SIDE = next((a for a in sys.argv[1:] if a in ("O", "D")), "O")
TARGET = flag("target", "blend0.7" if SIDE == "O" else "rapm1")
DIST = flag("dist", "all")
DECAY = 0.5
gcfg = cfg["gbdt"]
FEATS = list(gcfg["features_full_O"] if SIDE == "O" else gcfg["features_full_D"])
PARAMS = dict(gcfg["params"] if SIDE == "O" else gcfg["params_def"])
if "--cheap" in sys.argv:
    PARAMS = dict(gcfg["params_def"])
PARAMS.setdefault("thread_count", int(gcfg.get("thread_count", 3)))

panel = pd.read_parquet(ROOT / "outputs" / "role_panel.parquet")
if TARGET.startswith("blend"):
    wgt = float(TARGET[5:])
    panel[TARGET] = wgt * panel.apm + (1 - wgt) * panel.rapm1
p = panel[panel.side == SIDE].copy()
G.add_derived(p)
missing = [f for f in FEATS if f not in p.columns]
assert not missing, missing
wins = sorted(p.window.unique())
idx = {w: i for i, w in enumerate(wins)}
turn = pd.read_csv(ROOT / "outputs" / "csv" / "turnover_windows.csv")

feat = p[["player_id", "window", *FEATS]]
tgt = p[["player_id", "window", "poss", TARGET]].rename(columns={"window": "window_to", "poss": "poss_to", TARGET: "target"})
rows = feat.merge(tgt, on="player_id")
rows = rows[rows.window != rows.window_to]
rows["dist"] = (rows.window_to.map(idx) - rows.window.map(idx)).abs()
if DIST != "all":
    rows = rows[rows.dist <= int(DIST)]
rows = rows.merge(turn[["player_id", "window", "window_to", "turnover"]], on=["player_id", "window", "window_to"], how="inner")
rows["weight"] = rows.poss_to * DECAY ** np.maximum(rows.dist - 1, 0)
rows = rows[(rows.weight > 0) & rows.turnover.notna()].reset_index(drop=True)
rows["turn"] = rows.turnover
print(f"side {SIDE}, target {TARGET}, pairs at distance {DIST}: {len(rows)} rows, {rows.player_id.nunique()} players; "
      f"{len(FEATS)} features; params {PARAMS}")

t0 = time.time()
pred0 = np.full(len(rows), np.nan)
pred1 = np.full(len(rows), np.nan)
d_hi = np.full(len(rows), np.nan)          # prediction at turn = 1.0
d_lo = np.full(len(rows), np.nan)          # prediction at turn = 0.35
for wlab in wins:
    test = (rows.window == wlab).to_numpy()
    train = ~((rows.window == wlab) | (rows.window_to == wlab)).to_numpy()
    tr = rows[train]
    m0 = G.fit_gbdt(tr, FEATS, seed=0, **PARAMS)
    m1 = G.fit_gbdt(tr, [*FEATS, "turn"], seed=0, **PARAMS)
    te = rows[test]
    pred0[test] = m0.predict(te[FEATS].to_numpy(dtype=float))
    X1 = te[[*FEATS, "turn"]].copy()
    pred1[test] = m1.predict(X1.to_numpy(dtype=float))
    X1["turn"] = 1.0
    d_hi[test] = m1.predict(X1.to_numpy(dtype=float))
    X1["turn"] = 0.35
    d_lo[test] = m1.predict(X1.to_numpy(dtype=float))
    print(f"  {wlab} done ({time.time() - t0:.0f}s)", flush=True)

y, w = rows.target.to_numpy(dtype=float), rows.weight.to_numpy(dtype=float)
t = rows.turn.to_numpy(dtype=float)
e0, e1 = (y - pred0) ** 2, (y - pred1) ** 2
print(f"\nleave-window-out weighted MSE:  base {np.average(e0, weights=w):.4f}   +turn {np.average(e1, weights=w):.4f}   "
      f"diff {np.average(e1 - e0, weights=w):+.4f}")
per = rows.assign(d=e1 - e0).groupby("window").apply(lambda d: np.average(d.d, weights=d.weight), include_groups=False)
z = per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))
print(f"per-window paired: mean {per.mean():+.4f}, z {z:+.2f}, +turn wins {(per < 0).sum()}/{len(per)}")
for lo, hi in ((0.0, 0.5), (0.5, 0.9), (0.9, 1.01)):
    m = (t >= lo) & (t < hi)
    if m.sum():
        print(f"  turnover [{lo:.1f}, {hi:.1f}): n={m.sum():5d}  base {np.average(e0[m], weights=w[m]):.4f}  "
              f"+turn {np.average(e1[m], weights=w[m]):.4f}  diff {np.average((e1 - e0)[m], weights=w[m]):+.4f}")

# ---- the trade delta: one per player-window (the pair rows repeat the feature line; take the first)
delta = rows.assign(delta=d_hi - d_lo, p_lo=d_lo, p_hi=d_hi).drop_duplicates(["player_id", "window"])
feat_w = p.set_index(["player_id", "window"])
delta = delta.set_index(["player_id", "window"])
print(f"\ntrade delta = prediction at turnover 1.0 minus at 0.35, per player-window (n={len(delta)}), raw sign "
      f"({'positive = good' if SIDE == 'O' else 'positive = WORSE (points allowed)'}):")
dd = delta.delta
print(f"  mean {dd.mean():+.3f}  sd {dd.std():.3f}  p10 {dd.quantile(0.1):+.3f}  med {dd.median():+.3f}  p90 {dd.quantile(0.9):+.3f}")
big = delta[feat_w.loc[delta.index, "poss"].to_numpy() >= 4500]
print(f"  starters (4500+ possessions, n={len(big)}): mean {big.delta.mean():+.3f}  sd {big.delta.std():.3f}  "
      f"p10 {big.delta.quantile(0.1):+.3f}  p90 {big.delta.quantile(0.9):+.3f}")
print("\nwhat the delta correlates with (weighted by the window's possessions):")
cand = [c for c in ("usage", "poss_pct", "gs_pct", "age", "ast", "astr", "fg3m", "p3r", "orb", "drb", "blk", "stl", "tov", "ts", "pts",
                    "bigness", "creation", "q3", "m3", "xps") if c in feat_w.columns]
pw = feat_w.loc[delta.index, "poss"].to_numpy(dtype=float)
out = []
for c in cand:
    x = feat_w.loc[delta.index, c].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(dd.to_numpy())
    xm, ym = np.average(x[ok], weights=pw[ok]), np.average(dd.to_numpy()[ok], weights=pw[ok])
    cov = np.average((x[ok] - xm) * (dd.to_numpy()[ok] - ym), weights=pw[ok])
    r = cov / np.sqrt(np.average((x[ok] - xm) ** 2, weights=pw[ok]) * np.average((dd.to_numpy()[ok] - ym) ** 2, weights=pw[ok]))
    out.append((c, r))
print("  " + "  ".join(f"{c} {r:+.2f}" for c, r in sorted(out, key=lambda t: -abs(t[1]))))
delta.reset_index()[["player_id", "window", "p_lo", "p_hi", "delta"]].to_csv(ROOT / "outputs" / "csv" / f"trade_delta_{SIDE}.csv", index=False)
print(f"\nwritten outputs/csv/trade_delta_{SIDE}.csv  ({time.time() - t0:.0f}s)")
