"""Add each player-season's SCORE-STATE EXPOSURE to the season panel, per side.

    python scripts/69_closeness_panel.py [--first=1997] [--last=2026] [--panel=role_panel_season]
                                         [--dry_run=0]

**Why.**  The box prior credits a player's counting stats at full value whatever the score was.  The
ratings side does not: the design carries a garbage-time column and a margin slope, so APM is already net
of the blowout effect.  That asymmetry is the leading explanation for the prior being FLAT in exposure
where APM is steep -- the prior hands -0.76 to a man with 0-50 possessions and -0.81 to one with
1,000-2,000, while APM runs from -9.94 to -2.15 over the same range (DECISIONS.md, experiment 14).

And the premise holds, measured on the stints (`scratch/gt_share.py`, 2024-2026):

| possessions | 0-50 | 100-200 | 500-1k | 2-4k | 4k+ |
|---|---|---|---|---|---|
| share of his possessions in garbage time | **73.8%** | 49.3% | 26.4% | 13.7% | **6.0%** |
| average score gap while he was on court | 20.2 | 18.4 | 13.4 | 10.5 | 8.5 |

A deep-bench player takes three quarters of his possessions in blowouts; a starter takes 6%.  Monotone
across every tier.  His per-possession rates therefore describe a different game from a starter's, and
nothing in the prior's inputs says so.

**What this adds** (the owner's call, 2026-09-15: give the prior the exposure as a feature and let it
learn the discount itself, rather than imposing one by reweighting the stats):

  `gt_share`     his possessions' share flagged by the garbage-time rule (config.yaml `garbage_time`)
  `closeness`    possession-weighted mean of 1 / max(|margin|, 1) -- the same form the ratings step's own
                 objective uses (`priorridge.team_game_weights`), so "close" means the same thing in both
  `abs_margin`   possession-weighted mean absolute score margin while he was on the floor

Per SIDE, because a player's offensive possessions and his defensive ones need not sit at the same score
state: on offence he is weighted by his own team's possessions, on defence by the opponent's.

These are measured on the RATED season, which is always allowed -- the season's own games are the
evidence.  They are context, not quality: they say what conditions his statistics were compiled under.

Adds the three columns to outputs/<panel>.parquet in place (additive; every existing feature list is
explicit, so nothing else changes).  `--dry_run=1` reports and writes nothing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NEW = ["gt_share", "closeness", "abs_margin"]
# lineup slot -> (his team's possessions, the opponent's possessions)
SLOTS = ([(f"h{i}", "poss_h", "poss_a") for i in range(1, 6)]
         + [(f"a{i}", "poss_a", "poss_h") for i in range(1, 6)])


# The shared command line (scripts/_cli.py): `flag` records every name it is asked for so
# `check_flags` below can refuse one that was never asked for.  A misspelled flag used to be
# ignored silently, which is how a run looks right and is wrong.
from _cli import check_flags, flag as _flag, switch   # noqa: E402


def exposure(seasons) -> pd.DataFrame:
    """One row per (player_id, season, side) with the three exposure columns."""
    out = []
    for season in seasons:
        frames = []
        for phase in ("RS", "PO"):
            path = ROOT / f"data/stints/{season}_{phase}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            print(f"  {season}: no stints, skipped", flush=True)
            continue
        st = pd.concat(frames, ignore_index=True)
        st["abs_margin_"] = st["margin_h"].abs()
        st["closeness_"] = 1.0 / np.maximum(st["abs_margin_"], 1.0)
        st["is_gt_"] = st["is_gt"].astype(float)

        rows = []
        for slot, own, opponent in SLOTS:
            for side, poss_col in (("O", own), ("D", opponent)):
                part = st[[slot, poss_col, "abs_margin_", "closeness_", "is_gt_"]].copy()
                part.columns = ["player_id", "poss", "abs_margin_", "closeness_", "is_gt_"]
                part["side"] = side
                rows.append(part)
        d = pd.concat(rows, ignore_index=True)
        d = d[(d["player_id"] > 0) & (d["poss"] > 0)]
        for column in ("abs_margin_", "closeness_", "is_gt_"):
            d[f"w_{column}"] = d["poss"] * d[column]
        g = d.groupby(["player_id", "side"], as_index=False).agg(
            poss=("poss", "sum"), w_gt=("w_is_gt_", "sum"),
            w_margin=("w_abs_margin_", "sum"), w_close=("w_closeness_", "sum"))
        g["season"] = season
        g["gt_share"] = g["w_gt"] / g["poss"]
        g["abs_margin"] = g["w_margin"] / g["poss"]
        g["closeness"] = g["w_close"] / g["poss"]
        out.append(g[["player_id", "season", "side"] + NEW])
        print(f"  {season}: {len(st):,} stints, {g['player_id'].nunique():,} players", flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    check_flags()      # refuse a flag this script does not understand (scripts/_cli.py)
    first, last = int(_flag("first", 1997)), int(_flag("last", 2026))
    name = _flag("panel", "role_panel_season")
    dry = _flag("dry_run", "0") not in ("0", "no", "false")
    path = ROOT / "outputs" / f"{name}.parquet"

    panel = pd.read_parquet(path)
    clash = [c for c in NEW if c in panel.columns]
    print(f"panel {path.name}: {len(panel):,} rows, columns already present: {clash or 'none'}")

    table = exposure(range(first, last + 1))

    # The panel is written back IN PLACE and there is only one copy of it, so a run covering fewer
    # seasons than the panel holds would drop the three columns for every season and refill them
    # with the league mean -- silently, since the merge is a left join and the fill below is
    # unconditional.  Refuse rather than narrow.  (scripts/65_offcourt_panel.py takes its season
    # list from the panel instead, which is the other way to be safe.)
    short = sorted(int(x) for x in set(panel.season.unique()) - set(table.season.unique()))
    if short:
        columns = ", ".join(NEW)
        raise SystemExit(
            f"{path.name} carries season(s) {short} that this run did not build (--first={first} "
            f"--last={last}).  Writing now would refill their {columns} with the league "
            f"mean.  Rerun with --first/--last covering the panel.")

    merged = panel.drop(columns=clash).merge(table, on=["player_id", "season", "side"], how="left")
    assert len(merged) == len(panel), f"merge changed the row count: {len(panel)} -> {len(merged)}"
    missing = merged[NEW].isna().any(axis=1).sum()
    print(f"\nmerged: {len(merged):,} rows, {missing:,} without a stint match "
          f"({100 * missing / len(merged):.2f}%)")
    # a panel row with no stint row is a player the design never saw; league mean is the neutral fill
    for column in NEW:
        merged[column] = merged[column].fillna(merged[column].mean())

    pd.set_option("display.width", 200, "display.precision", 3)
    view = merged[merged.side == "O"].copy()
    view["tier"] = pd.cut(view["poss"], [0, 50, 100, 200, 500, 1000, 2000, 4000, 1e9])
    print("\n=== the new columns, offensive rows, by possessions")
    print(view.groupby("tier", observed=True)[NEW].mean().round(3).to_string())

    if dry:
        print("\n--dry_run=1: nothing written")
        return
    # after the dry-run check, not before it: this used to be written whatever --dry_run said
    table.to_parquet(ROOT / "outputs/closeness_exposure.parquet", index=False)
    merged.to_parquet(path, index=False)
    print(f"\nwrote {path}: {len(merged):,} rows, {len(merged.columns)} columns "
          f"(added {', '.join(NEW)})")


if __name__ == "__main__":
    main()
