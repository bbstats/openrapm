"""Experiment 119's screen by era: each candidate's change in the within-season WOWY test after the shrink, the three
random draws averaged within each season, against its shuffled copy and against the shipped lists, per era.

    .venv/Scripts/python scratch/2026-10-28_fouls/eras.py
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
from openrapm.features import meaning  # noqa: E402

ERAS = {"1998-2004": (1998, 2001, 2004), "2007-2016": (2007, 2010, 2013, 2016), "2019-2025": (2019, 2022, 2025)}
rows = pd.read_csv("outputs/screen/screen119/trials.csv", keep_default_na=False)
rows = rows[(rows.stage == "step4") & (rows.slice == "all")]
table = pd.read_csv("outputs/csv/screen_screen119_table.csv")


def means(frame, seeds):
    return frame[frame.seed.isin(seeds)].groupby("season").error.mean()


base = rows[rows.arm == "base"]
out = []
for (side, name), group in rows[rows.arm == "add"].groupby(["side", "input"]):
    seeds = sorted(group.seed.unique())
    shadow = rows[(rows.arm == "add_shadow") & (rows.side == side) & (rows.input == name)]
    mine, copy, shipped = means(group, seeds), means(shadow, seeds), means(base, seeds)
    row = dict(input=meaning(name), side={"O": "offense", "D": "defense"}[side])
    for era, seasons in ERAS.items():
        row[f"vs_copy_{era}"] = float((mine - copy).reindex(seasons).mean())
    for era, seasons in ERAS.items():
        row[f"vs_shipped_{era}"] = float((mine - shipped).reindex(seasons).mean())
    row["eras_better_vs_shipped"] = sum(row[f"vs_shipped_{era}"] < 0 for era in ERAS)
    out.append(row)
eras = pd.DataFrame(out)
pd.set_option("display.width", 250)
print(eras.round(4).to_string(index=False))
eras.to_csv("outputs/csv/screen_screen119_eras.csv", index=False)
print("\nwrote outputs/csv/screen_screen119_eras.csv")
