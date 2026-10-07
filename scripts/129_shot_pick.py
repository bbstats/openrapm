"""The search's shortlist: every scored trial against trial 0 and the placebo, per family, with the plateau.

    python scripts/129_shot_pick.py [--families=glm,xgb,...] [--top=8]

Reads outputs/shotsearch/trials.parquet (scripts/127).  Per trial: SALL minus trial 0 (the current model on the same
rows), per 1000 attempts, paired over the 15 search seasons (z, seasons won), and per sub-model.  The placebo
(the current model on other row samples) gives the noise floor.  Per family: the plateau, every trial within one
season-paired standard error of the family's best; the protocol keeps its simplest member and the best.  Printed,
and written to outputs/shotsearch/pick.csv; the shortlist itself is registered by hand in DECISIONS.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import shotsearch as ss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def per_season(t: pd.DataFrame) -> pd.Series:
    return t.groupby("season").apply(lambda g: np.average(g["sall"], weights=g["n"]), include_groups=False)


def main():
    check_flags()
    T = pd.read_parquet(ROOT / "outputs" / "shotsearch" / "trials.parquet")
    base = T[(T.family == "baseline") & (T.trial == "L7_seed0")]
    b = per_season(base)
    fams = [f for f in flag("families", "").split(",") if f] or sorted(set(T.family) - {"baseline"})
    rows = []
    for (fam, trial), g in T[T.family.isin(fams)].groupby(["family", "trial"]):
        if g["season"].nunique() < 15:
            continue                                  # pruned or unfinished
        d = (per_season(g) - b.reindex(per_season(g).index)).dropna()
        r = dict(family=fam, trial=trial, diff_per_1000=1000 * d.mean(), se_per_1000=1000 * d.std(ddof=1) / np.sqrt(len(d)),
                 won=int((d < 0).sum()), params=g["params"].iloc[0], seconds=g["seconds"].iloc[0])
        r["z"] = r["diff_per_1000"] / r["se_per_1000"] if r["se_per_1000"] > 0 else np.nan
        for sub in ("rim", "mid", "three"):
            p = ss.paired(g, base, sub=sub)
            r[f"{sub}_per_1000"] = 1000 * p["diff"]
        rows.append(r)
    R = pd.DataFrame(rows).sort_values("diff_per_1000")
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    top = int(flag("top", "8"))
    for fam, g in R.groupby("family"):
        best = g.iloc[0]
        g = g.assign(plateau=g["diff_per_1000"] <= best["diff_per_1000"] + best["se_per_1000"])
        print(f"\n=== {fam}: {len(g)} complete trials; best #{best['trial']} {best['diff_per_1000']:+.3f}/1000 "
              f"(z {best['z']:+.1f}, {best['won']}/15); plateau {int(g.plateau.sum())} trials")
        print(g.head(top)[["trial", "diff_per_1000", "z", "won", "rim_per_1000", "mid_per_1000", "three_per_1000",
                           "plateau", "seconds", "params"]].round(3).to_string(index=False))
    R.to_csv(ROOT / "outputs" / "shotsearch" / "pick.csv", index=False)


if __name__ == "__main__":
    main()
