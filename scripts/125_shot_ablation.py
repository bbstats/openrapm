"""The shot-quality ablation table (stage 9): every arm against the shipped curve on every test, one row an arm.

    python scripts/125_shot_ablation.py

Reads the shot-test outputs already written (outputs/shottest/: s6_ladder_*, s8_*, forward_s6/s8, tracking_taught)
and writes outputs/shottest/ablation.csv.  No winner is picked: the owner chooses the version (G3).

Columns, every difference against the shipped curve (`lp`, a per-season distance curve fitted in season):
    makes_2s / makes_3s     held-out log loss x 1000 on twos / threes, mean over 30 seasons (negative is better)
    won_2s / won_3s         seasons of 30 the arm is better
    gap_2015 / gap_2016     the mean squared gap to the tracking teacher's quality, squared points of make
                            probability (a level, not a difference: lower is closer to tracking)
    contest_left            the slope of player-games' shooting beyond the arm on their share of tightly guarded
                            shots, 2015-2017 mean (a level: nearer zero is better)
    half_*                  other-half error, squared points of make rate, with its z (negative is better)
    fwd_off / fwd_def       the forward test: the arm's team shot quality as the shrink target, minus the shipped
                            luck adjustment, held-out error (negative is better), with z
    arena_post / arena_pre  the arena's signal sd of expected points per 100 attempts, 2011-26 / 1997-2010 means
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef import shottest as st  # noqa: E402
from eracoef.config import load_config  # noqa: E402

# The tracking-taught students are judged by the nested test (scripts/122) alone: the ratings-model versions cannot
# differ from L7 on the tracked seasons (the leakage rule keeps the labels out of 2014-2018), and the first ones were
# taught with labels that had the shooter taken out (fixed 2026-10-06).
WHAT = {
    "flat": "the season's league rate by shot value: no shot quality at all",
    "lp": "shipped: the season's own distance curve (feed distance), fitted in season",
    "loc": "location only from x/y (rim-mapped), trained on other seasons",
    "L1": "+ shooter and arena offsets, zeroed when pricing",
    "L2": "+ how the possession started and seconds into it",
    "L3": "+ putbacks (seconds since own offensive rebound)",
    "L4": "+ rebuilt shot clock (four bands) and its last reset",
    "L5": "+ period, end of period, score margin, garbage time",
    "L6": "+ shooter's time on court",
    "L7": "+ the shot that handed the ball over: THE PLAY-BY-PLAY MODEL",
    "L7fine": "L7 with the shot clock as a fine curve instead of bands",
    "L7nooff": "L7 without the shooter and arena offsets",
    "L7_after": "L7 updated by each shot's own make or miss (the owner's 'after')",
    "L7nooff_after": "L7nooff updated by each shot's own make or miss",
}


def main():
    cfg = load_config()
    d = Path(cfg["_root"]) / "outputs" / "shottest"
    makes = pd.concat([pd.read_csv(d / "s6_ladder_makes.csv"), pd.read_csv(d / "s8_makes.csv")]).drop_duplicates(["season", "value", "arm"])
    half = pd.concat([pd.read_csv(d / "s6_ladder_half.csv"), pd.read_csv(d / "s8_half.csv"),
                      pd.read_csv(d / "s9_vsflat_half.csv")]).drop_duplicates(["season", "unit", "value", "arm"])
    makes = pd.concat([makes, pd.read_csv(d / "s5_loc_makes.csv")]).drop_duplicates(["season", "value", "arm"])
    arena = pd.concat([pd.read_csv(d / "s6_ladder_arena.csv"), pd.read_csv(d / "s8_arena.csv")]).drop_duplicates(["season", "arm"])
    teach = pd.read_csv(d / "s8_teach.csv", dtype={"game_id": str})
    dash = pd.read_csv(d / "s8_dash.csv")
    fwd = pd.concat([pd.read_csv(d / "forward_s6.csv"), pd.read_csv(d / "forward_s8.csv")]).drop_duplicates(["side", "arm"])
    rows = []
    for arm, what in WHAT.items():
        r = dict(arm=arm, what=what)
        for v in (2, 3):
            m = makes[makes.value == v]
            if arm in set(m.arm) and not arm.endswith("_after"):
                p = st.paired(m, arm, "lp") if arm != "lp" else dict(diff=0.0, won=np.nan)
                r[f"makes_{v}s"] = round(1000 * p["diff"], 3)
                r[f"won_{v}s"] = p["won"]
        for s in (2015, 2016):
            t = teach[(teach.arm == arm) & (teach.season == s)]
            if len(t):
                r[f"gap_{s}"] = round(t.gap.mean(), 2)
        c = dash[dash.arm == arm]
        if len(c):
            r["contest_left"] = round(c.slope.mean(), 4)
        for unit in ("shooter", "offence", "defence"):
            for v in (2, 3):
                h = half[(half.unit == unit) & (half.value == v)]
                if arm in set(h.arm) and arm != "lp":
                    p = st.paired_seasons(h[h.arm == arm], h[h.arm == "lp"])
                    r[f"half_{unit}_{v}s"] = round(p["diff"], 3)
                    r[f"half_{unit}_{v}s_z"] = round(p["z"], 1)
        for side in ("off", "def"):
            f = fwd[(fwd.side == side) & (fwd.arm == f"sel_{arm}")]
            if len(f):
                r[f"fwd_{side}"] = round(float(f.vs_sel.iloc[0]), 4)
                r[f"fwd_{side}_z"] = round(float(f.z_sel.iloc[0]), 2)
        a = arena[arena.arm == arm]
        if len(a):
            r["arena_post"] = round(a[a.season >= 2011].xp_arena_sd.mean(), 2)
            r["arena_pre"] = round(a[a.season < 2011].xp_arena_sd.mean(), 2)
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(d / "ablation.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(T.drop(columns="what").to_string(index=False))
    tt = pd.read_csv(d / "tracking_taught.csv")
    print("\nthe nested test (scripts/122): each held-out tracked season, every arm on the same rows")
    print(tt.to_string(index=False))


if __name__ == "__main__":
    main()
