"""Trade calibration, step 1: build the teammate table and describe teammate turnover.

    .venv/Scripts/python scratch/trade_turnover.py [--force]

Prints: season-to-season turnover for movers and stayers (main-team rule), the same between the panel's
3-season windows (the contrast the prior's training pairs would see), and between a K = 3 training block and
its held-out season (the contrast the criterion sees).  Writes outputs/csv/turnover_season.csv and
outputs/csv/turnover_windows.csv for the next steps.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.turnover import build_teammates, familiar_share, season_turnover, window_pair_turnover  # noqa: E402
from eracoef.windows import window_label, window_seasons  # noqa: E402

pd.set_option("display.width", 200, "display.precision", 3)
cfg = load_config()
OUT = ROOT / "outputs" / "csv"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
tm = build_teammates(cfg, force="--force" in sys.argv)
print(f"teammate table: {len(tm)} rows, {tm.season.nunique()} seasons ({time.time() - t0:.0f}s)")


def q(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "n=0"
    return f"n={len(x):5d}  mean {x.mean():.3f}  p10 {np.percentile(x, 10):.3f}  p25 {np.percentile(x, 25):.3f}  " \
           f"med {np.median(x):.3f}  p75 {np.percentile(x, 75):.3f}  p90 {np.percentile(x, 90):.3f}"


# ---- season to season, split by the main-team rule (moved = most on-floor possessions with a different team)
roles = pd.read_parquet(ROOT / "data" / "cache" / "roles.parquet")
main = roles.sort_values("poss_on", ascending=False).drop_duplicates(["player_id", "season"])[["player_id", "season", "team_id"]]
main = main.sort_values(["player_id", "season"])
main["prev_team"] = main.groupby("player_id")["team_id"].shift(1)
main["prev_season"] = main.groupby("player_id")["season"].shift(1)
main["moved"] = (main.prev_season == main.season - 1) & (main.prev_team != main.team_id)
st = season_turnover(tm).merge(main[["player_id", "season", "moved"]], on=["player_id", "season"], how="left")
st = st[st.poss_new >= 4 * 500]                    # at least 500 on-floor possessions in the new season
st.to_csv(OUT / "turnover_season.csv", index=False)
print("\nseason -> next season, players with 500+ possessions in the new season")
print("  stayers :", q(st.loc[~st.moved.fillna(False), "turnover"]))
print("  movers  :", q(st.loc[st.moved.fillna(False), "turnover"]))
print("  all     :", q(st.turnover))
lo, hi = st.turnover < 0.5, st.turnover >= 0.9
print(f"  under 0.5: {lo.mean():.1%} of player-seasons ({st.loc[lo, 'moved'].fillna(False).mean():.1%} movers); "
      f"0.9 and up: {hi.mean():.1%} ({st.loc[hi, 'moved'].fillna(False).mean():.1%} movers)")
by = st.groupby("season").turnover.agg(["mean", "median"]).T
print("  league mean by season:\n", by.round(3).to_string())

# ---- between the panel's windows: what the prior's leave-window-out pairs would see
wins = [(window_label(list(range(w[0], w[1] + 1))), list(range(w[0], w[1] + 1))) for w in window_seasons(cfg)]
wt = window_pair_turnover(tm, wins)
wt = wt[wt.poss_new >= 4 * 1000]
wt.to_csv(OUT / "turnover_windows.csv", index=False)
idx = {lab: i for i, (lab, _) in enumerate(wins)}
wt["dist"] = (wt.window_to.map(idx) - wt.window.map(idx))
print("\nwindow -> other window, players with 1000+ possessions in the destination window")
for d in (1, -1, 2, -2, 3):
    print(f"  {d:+d} windows:", q(wt.loc[wt.dist == d, "turnover"]))
print("  adjacent (either way):", q(wt.loc[wt.dist.abs() == 1, "turnover"]))

# ---- training block (K = 3: H-2, H-1, H+1) -> held-out season H: what the criterion sees
first, last = int(cfg["first_season"]), int(cfg["last_season"])
parts = []
for h in range(int(cfg["holdout"]["first"]), int(cfg["holdout"]["last"]) + 1):
    block = [s for s in (h - 2, h - 1, h + 1) if first <= s <= last]
    f = familiar_share(tm, block, [h])
    seen = set(tm.loc[tm.season.isin(block), "player_id"].unique())
    f = f[f.player_id.isin(seen)]
    parts.append(f.assign(held_out=h))
bh = pd.concat(parts, ignore_index=True)
bh = bh.merge(main[["player_id", "season", "moved"]].rename(columns={"season": "held_out"}), on=["player_id", "held_out"], how="left")
bh = bh[bh.poss_new >= 4 * 500]
print("\nK = 3 training block -> held-out season, 500+ possessions in H")
print("  stayers :", q(bh.loc[~bh.moved.fillna(False), "turnover"]))
print("  movers  :", q(bh.loc[bh.moved.fillna(False), "turnover"]))
print("  all     :", q(bh.turnover))
print(f"\ndone in {time.time() - t0:.0f}s")
