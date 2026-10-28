"""Experiment 118's readout from check 155: each Kumo variant's draws against the shipped prior's five, on the
within-season WOWY test, before the shrink (step 3) and after it (step 4), by era; the next-season check beside it;
and the box-score prior by playing time against the shipped prior's, from each variant's first draw.

    .venv/Scripts/python scratch/2026-10-28_kumo/read_screen.py [--variants kumo_small,kumo_medium,kumo_small_equal]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

SCREEN = Path("outputs/screen/screen112")
FOLDS = Path("outputs/within/screen/screen112")
ERAS = {"1997-2006": (1998, 2001, 2004), "2007-2016": (2007, 2010, 2013, 2016), "2017-2026": (2019, 2022, 2025)}
BANDS = [-1, 250, 1000, 2500, 1e9]
BAND_NAMES = ["under 250", "250-1000", "1000-2500", "2500+"]


def paired(mine: pd.Series, theirs: pd.Series, by_era: bool = True) -> dict:
    diff = (mine - theirs).dropna()
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    out = dict(change=diff.mean(), z=diff.mean() / se, seasons_better=int((diff < 0).sum()), seasons=len(diff))
    if by_era:
        out.update({era: diff.reindex(list(seasons)).mean() for era, seasons in ERAS.items()})
    return out


def test_table(variants: list) -> pd.DataFrame:
    shipped = pd.read_csv(SCREEN / "trials.csv", keep_default_na=False)
    shipped = shipped[shipped.arm == "base"]
    guard = pd.read_csv(SCREEN / "guard.csv", keep_default_na=False)
    guard = guard[guard.arm == "base"].groupby(["rated", "scored"]).tg.mean()
    rows = []
    for variant in variants:
        path = SCREEN / f"trials_{variant}.csv"
        if not path.exists():
            continue
        mine = pd.read_csv(path, keep_default_na=False)
        for stage in ("step3", "step4"):
            pick = (mine.stage == stage) & (mine.slice == "all")
            ours = mine[pick].groupby("season").error.mean()
            theirs = shipped[(shipped.stage == stage) & (shipped.slice == "all")].groupby("season").error.mean()
            rows.append(dict(variant=variant, stage=stage, draws=mine[pick].seed.nunique(), **paired(ours, theirs)))
        mine_guard = pd.read_csv(SCREEN / f"guard_{variant}.csv").groupby(["rated", "scored"]).tg.mean()
        next_season = paired(mine_guard, guard, by_era=False)
        rows.append(dict(variant=variant, stage="next-season check", draws=rows[-1]["draws"],
                         change=next_season["change"], z=next_season["z"],
                         seasons_better=next_season["seasons_better"], seasons=next_season["seasons"]))
    return pd.DataFrame(rows)


def band_table(variant: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(FOLDS / f"base__-__-__s0__{variant}" / "players_*_r0_f0.parquet"))):
        name = Path(path).name
        mine, theirs = pd.read_parquet(path), pd.read_parquet(FOLDS / "base__-__-__s0" / name)
        rows.append(mine.merge(theirs, on=["player_id", "side"], suffixes=("_kumo", "_today")))
    both = pd.concat(rows)
    out = []
    for side, prior, poss in (("O", "prior_raw_off", "poss_off"), ("D", "prior_raw_def", "poss_def")):
        part = both[both.side == side].copy()
        part["band"] = pd.cut(part[f"{poss}_kumo"], BANDS, labels=BAND_NAMES)
        for band, group in part.groupby("band", observed=True):
            out.append(dict(side=side, possessions=band, players=len(group),
                            mean_kumo=group[f"{prior}_kumo"].mean(), mean_today=group[f"{prior}_today"].mean(),
                            sd_kumo=group[f"{prior}_kumo"].std(), sd_today=group[f"{prior}_today"].std(),
                            correlation=group[f"{prior}_kumo"].corr(group[f"{prior}_today"])))
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="kumo_small,kumo_medium,kumo_small_equal")
    args = parser.parse_args()
    variants = args.variants.split(",")
    pd.set_option("display.width", 220)
    print("check 155, the 10 test seasons, each variant's draws averaged against the shipped prior's five draws; "
          "below 0 better")
    print(test_table(variants).round(4).to_string(index=False))
    for variant in variants:
        if (FOLDS / f"base__-__-__s0__{variant}").exists():
            print(f"\n{variant}: the box-score prior by possessions, positive good, one held-back game set a season")
            print(band_table(variant).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
