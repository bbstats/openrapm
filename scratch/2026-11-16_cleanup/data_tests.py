"""Tests of the data fetched again on 2026-11-17, before any of it is published (the owner: "No publishing till we do
testing on the data").  Reads only.

    python scratch/2026-11-16_cleanup/data_tests.py      prints; the morning job's lock is not needed

1. Stints, every season and phase: the fresh files against data/stints_before139/, the copy the deletion missed, built
   before experiment 139.  Same rows in the same order; every shared column equal but the expected-make families
   (xl*, xv*), which experiment 139's shot curve reprices.
2. Every game: data/stints.check_games for every season, which reads the game log, the stints and their diagnostics,
   and removes nothing: games without stints, games whose play-by-play points differ from the box score.
3. The season panel against the copy in A:/code/spmm_verify/final/outputs/, built 2026-10-12: the same player-seasons,
   and per column the share of rows equal and the largest gap.  Columns built differently since 2026-10-12 differ by
   design; the counts the box scores give (games, minutes, points, rebounds...) must not.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm.data import stints as stints_module
from openrapm.data.config import load_config, resolve

ROOT = Path("A:/code/spmm")
cfg = load_config()
pd.set_option("display.width", 220)

print("== 1. stints against data/stints_before139")
EXPECTED = re.compile(r"^x[lv][23]")
bad = []
pairs = sorted(p for p in (ROOT / "data/stints").glob("*_*.parquet") if re.fullmatch(r"\d{4}_(RS|PO)", p.stem))
for path in pairs:
    old_path = ROOT / "data/stints_before139" / path.name
    if not old_path.exists():
        bad.append((path.stem, "no old copy"))
        continue
    new, old = pd.read_parquet(path).reset_index(drop=True), pd.read_parquet(old_path).reset_index(drop=True)
    if len(new) != len(old):
        # the season in progress had fewer games when the old copy was built: compare the games both hold
        keep = set(old.game_id)
        new = new[new.game_id.isin(keep)].reset_index(drop=True)
        note = f"compared on the {len(keep)} games of the old copy"
        if len(new) != len(old):
            bad.append((path.stem, f"rows {len(new)} against {len(old)}"))
            continue
    else:
        note = ""
    differ = [c for c in old.columns if c in new.columns and not new[c].equals(old[c])]
    unexpected = [c for c in differ if not EXPECTED.match(c)]
    missing = [c for c in old.columns if c not in new.columns]
    if unexpected or missing:
        bad.append((path.stem, f"unexpected columns differ {unexpected[:6]}, missing {missing[:6]}"))
    elif note:
        print(f"  {path.stem}: {note}")
print(f"  {len(pairs)} season-phases; {len(pairs) - len(bad)} pass, {len(bad)} do not")
for item in bad:
    print("  NOT", item)

print("\n== 2. every game: game log, stints, points")
accepted = cfg.get("points_mismatch_accepted", []) or []
rows = []
for season in range(int(cfg["first_season"]), int(cfg["last_season"]) + 1):
    for check in stints_module.check_games(season, cfg, accepted):
        rows.append(dict(season=season, phase=check.phase, games=len(check.logged), no_stints=len(check.no_stints),
                         points_differ=len(check.points_differ)))
        for game, why in list(check.no_stints.items())[:3]:
            print(f"  NO STINTS {season} {check.phase} {game}: {why}")
        for game, why in list(check.points_differ.items())[:3]:
            print(f"  POINTS {season} {check.phase} {game}: {why}")
games = pd.DataFrame(rows)
print(f"  {games.games.sum()} games over {len(games)} season-phases; without stints {games.no_stints.sum()}, "
      f"points differ {games.points_differ.sum()}")

print("\n== 3. the season panel against spmm_verify/final, 2026-10-12")
new = pd.read_parquet(resolve(cfg, "panel"))
old = pd.read_parquet("A:/code/spmm_verify/final/outputs/season_panel.parquet")
key = ["player_id", "season", "side"]
merged = new.merge(old, on=key, how="outer", suffixes=("", "_old"), indicator=True)
print(f"  rows: new {len(new)}, old {len(old)}; in both {int((merged._merge == 'both').sum())}, "
      f"new only {int((merged._merge == 'left_only').sum())}, old only {int((merged._merge == 'right_only').sum())}")
only_new = merged[merged._merge == "left_only"].season.value_counts().sort_index()
print(f"  new-only rows by season: {only_new.to_dict()}")
both = merged[merged._merge == "both"]
report = []
for column in sorted(set(new.columns) & set(old.columns) - set(key)):
    a, b = both[column], both[f"{column}_old"]
    if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
        report.append(dict(column=column, share_equal=float((a.astype(str) == b.astype(str)).mean()), max_gap=np.nan))
        continue
    a, b = a.astype(float), b.astype(float)
    same = (np.isclose(a, b, rtol=1e-9, atol=1e-9) | (a.isna() & b.isna()))
    report.append(dict(column=column, share_equal=float(same.mean()), max_gap=float((a - b).abs().max())))
report = pd.DataFrame(report).sort_values("share_equal")
print(f"  {len(report)} shared columns: {int((report.share_equal == 1).sum())} equal on every row")
print(report[report.share_equal < 1].round(6).to_string(index=False))
print(f"  new columns since: {sorted(set(new.columns) - set(old.columns))}")
print(f"  columns gone since: {sorted(set(old.columns) - set(new.columns))}")
