"""The noise control read closely: how far the ratings move when only the box-score prior's deal changes, and
whether the average of the two deals beats both on the within-season test.

    moves      creation110 against noise_creation110, whole seasons after step 4 (the published schema): each
               player's offense, defense and total, by era, players with 1,500+ offensive possessions; the same for
               experiments 110 and 112 against their incumbents, for scale
    average    a third system, avg_deals, whose every fold is the two builds' folds averaged column by column (the
               ratings, their prior parts and games' parts, the boosters' answers), written as `openrapm folds`
               writes a fold; check 151 then reads all three against creation110
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

WITHIN = Path("outputs/within")
AVERAGED = ["rating_off", "rating_def", "prior_off", "prior_def", "u_off", "u_def", "rating_total", "prior_raw_off",
            "prior_raw_def", "second_prior_raw_off", "second_prior_raw_def"]
ERAS = (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026))


def moves(a: str, b: str, label: str) -> list:
    one = pd.read_parquet(f"outputs/season_ratings_{a}.parquet")
    two = pd.read_parquet(f"outputs/season_ratings_{b}.parquet")
    m = one.merge(two, on=["player_id", "season"], suffixes=("_a", "_b"))
    m = m[m.poss_off_a >= 1500]
    rows = []
    for era, first, last in ERAS:
        e = m[m.season.between(first, last)]
        for side in ("off", "def", "total"):
            d = (e[f"rating_{side}_b"] - e[f"rating_{side}_a"]).abs()
            rows.append(dict(comparison=label, era=era, side=side, players=len(e), median=d.median(),
                             p90=d.quantile(0.9), largest=d.max(), share_over_0_1=(d > 0.1).mean(),
                             share_over_0_25=(d > 0.25).mean()))
    return rows


def average_folds(name: str) -> None:
    for part in ("early", "within"):
        a, b, out = WITHIN / f"creation110_{part}", WITHIN / f"noise_creation110_{part}", WITHIN / f"{name}_{part}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        for path in sorted(a.glob("fold_*.json")):
            stem = path.stem[len("fold_"):]
            pa = pd.read_parquet(a / f"players_{stem}.parquet")
            pb = pd.read_parquet(b / f"players_{stem}.parquet").set_index(["player_id", "side"])
            key = pd.MultiIndex.from_arrays([pa.player_id, pa.side])
            if not pb.index.equals(pb.index.unique()) or len(pb) != len(pa) or not key.isin(pb.index).all():
                raise SystemExit(f"{stem}: the two builds rate different players")
            for column in AVERAGED:
                if column in pa.columns:
                    pa[column] = (pa[column].to_numpy(float) + pb[column].reindex(key).to_numpy(float)) / 2
            pa.to_parquet(out / f"players_{stem}.parquet", index=False)
            shutil.copy(a / f"test_{stem}.npz", out / f"test_{stem}.npz")
            ra = json.loads(path.read_text())
            rb = json.loads((b / path.name).read_text())
            for fill in ("fill_o", "fill_d"):
                ra["summary"][fill] = (float(ra["summary"][fill]) + float(rb["summary"][fill])) / 2
            (out / path.name).write_text(json.dumps(ra, indent=1))
        for folds in a.glob("folds_*.parquet"):
            shutil.copy(folds, out / folds.name)
        print(f"  {out}: {len(list(out.glob('fold_*.json')))} folds", flush=True)


def main() -> None:
    rows = moves("creation110", "noise_creation110", "the deal, creation110 vs its control")
    rows += moves("creation110", "screen112", "experiment 112, one input added")
    rows += moves("parts107", "creation110", "experiment 110, self-creation added")
    table = pd.DataFrame(rows)
    table.to_csv("outputs/csv/deal_noise_moves.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    average_folds("avg_deals")


if __name__ == "__main__":
    main()
