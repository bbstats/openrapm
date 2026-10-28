"""Experiment 119's panel against the one before it: every older column the same, and the new columns read.

    .venv/Scripts/python scratch/2026-10-28_fouls/panel_check.py
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
from openrapm.features import FOUL_RATES, GOALTENDS_FROM, OVER_EXPECTED  # noqa: E402

before = pd.read_parquet("scratch/2026-10-28_fouls/season_panel_before.parquet")
after = pd.read_parquet("outputs/season_panel.parquet")
keys = ["side", "player_id", "season"]
both = before.merge(after, on=keys, suffixes=("_before", ""), how="outer", indicator=True)
print("rows before", len(before), "after", len(after), "in one only", int((both._merge != "both").sum()))
worst = {}
for column in before.columns:
    if column in keys or not pd.api.types.is_numeric_dtype(before[column]):
        continue
    a, b = both[f"{column}_before"].to_numpy(float), both[column].to_numpy(float)
    same_missing = np.isnan(a) == np.isnan(b)
    worst[column] = (float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0, bool(same_missing.all()))
moved = {name: value for name, value in worst.items() if value[0] > 1e-12 or not value[1]}
print("older columns compared", len(worst), "moved", moved or "none")

new = [*FOUL_RATES, *OVER_EXPECTED]
print("\nnew columns, rows with a value, by era")
for era, (low, high) in {"1997-2001": (1997, 2001), "2002-2016": (2002, 2016), "2017-2027": (2017, 2027)}.items():
    rows = after[after.season.between(low, high)]
    print(f"  {era}: " + ", ".join(f"{name.removesuffix('_per100')} {rows[name].notna().mean():.2f}" for name in new
                                   if rows[name].notna().mean() < 1) or f"  {era}: every new column full")
print(f"  goaltends start {GOALTENDS_FROM}: missing in", sorted(after[after.goaltends_per100.isna()].season.unique()))

bio = pd.read_parquet("data/raw/bio/2025_RS.parquet")[["PLAYER_ID", "PLAYER_NAME"]]
names = dict(zip(bio.PLAYER_ID, bio.PLAYER_NAME))
rows = after[(after.season == 2025) & (after.possessions >= 2500)].copy()
rows["name"] = rows.player_id.map(names)
for side, name in [("O", "turnovers_over_expected_from_attempts_and_assists_per100"),
                   ("O", "shots_blocked_over_expected_from_attempts_per100"),
                   ("D", "blocks_over_expected_from_fouls_per100"),
                   ("D", "goaltends_over_expected_from_blocks_per100")]:
    part = rows[rows.side == side].sort_values(name)
    print(f"\n2025 {name}, 2,500+ possessions: lowest", ", ".join(f"{n} {v:+.2f}" for n, v in
                                                              zip(part.name.head(4), part[name].head(4))))
    print("   highest", ", ".join(f"{n} {v:+.2f}" for n, v in zip(part.name.tail(4), part[name].tail(4))))
