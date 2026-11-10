"""Read-only: what the WOSA fix of 2026-11-09 moves (notes/DECISIONS.md, "The WOSA fix").

Two builds of the same ratings table, one before the regular-season rating stopped reading playoff lineups and one
after: every other column must be the same, and the moves are in the regular-season ratings and the wins columns.

    .venv/Scripts/python scratch/2026-11-09_improvements/wosa_moves.py BEFORE AFTER [SEASON ...]

BEFORE and AFTER are table names, outputs/season_ratings_<name>.parquet with <name>_regular beside each.
"""
import sys

import numpy as np
import pandas as pd

from openrapm import wins
from openrapm.data.config import load_config, resolve

sys.stdout.reconfigure(encoding="utf-8")
KEY = ["season", "player_id"]


def table(outputs, name: str, suffix: str = "") -> pd.DataFrame:
    return pd.read_parquet(outputs / f"season_ratings_{name}{suffix}.parquet")


def main() -> None:
    before_name, after_name, *seasons = sys.argv[1:]
    outputs = resolve(load_config(), "outputs")
    before, after = table(outputs, before_name), table(outputs, after_name)
    seasons = [int(season) for season in seasons] or sorted(int(s) for s in before.season.unique())
    before, after = (frame[frame.season.isin(seasons)].sort_values(KEY).reset_index(drop=True)
                     for frame in (before, after))
    assert before[KEY].equals(after[KEY]), "the two tables do not hold the same player-seasons"
    other = [column for column in before.columns if column not in wins.WINS_COLUMNS]
    moved = [column for column in other if not before[column].equals(after[column])]
    print(f"seasons {seasons[0]} to {seasons[-1]}: {len(before):,} player-seasons; columns outside the wins that "
          f"differ: {moved or 'none'}")
    regular_before, regular_after = (table(outputs, name, "_regular") for name in (before_name, after_name))
    regular = regular_before.merge(regular_after, on=KEY, suffixes=("_before", "_after"))
    regular = regular[regular.season.isin(seasons)]
    for side in ("off", "def", "total"):
        move = (regular[f"rating_{side}_after"] - regular[f"rating_{side}_before"]).abs()
        shown = move[regular.poss_season_after >= 60]
        print(f"  regular-season rating, {side}: median move {shown.median():.3f}, 90th percentile "
              f"{shown.quantile(0.9):.3f}, largest {shown.max():.3f} points per 100, players with 60+ possessions")
    both = before[[*KEY, "player_name", "poss_season", "wosa"]].assign(wosa_after=after.wosa.to_numpy())
    both["move"] = both.wosa_after - both.wosa
    shown = both[both.poss_season >= 60]
    size = shown.move.abs()
    print(f"  WOSA: median move {size.median():.3f} wins, 90th percentile {size.quantile(0.9):.3f}, largest "
          f"{size.max():.3f}; {int((size >= 0.1).sum()):,} of {len(shown):,} move a tenth of a win or more")
    by_season = shown.assign(size=size).groupby("season")["size"].agg(["median", "max"]).round(3)
    if len(seasons) > 1:
        print("\nby season, the median and the largest WOSA move:")
        print(by_season.to_string())
    largest = shown.reindex(size.sort_values(ascending=False).index).head(20)
    pd.set_option("display.width", 200)
    print("\nthe 20 largest WOSA moves:")
    print(largest.assign(wosa=largest.wosa.round(2), wosa_after=largest.wosa_after.round(2), move=largest.move.round(2),
                         poss_season=largest.poss_season.astype(int))
          [["season", "player_name", "poss_season", "wosa", "wosa_after", "move"]].to_string(index=False))
    for season in seasons[-1:]:
        top = shown[shown.season == season].nlargest(10, "wosa_after")
        print(f"\n{season}, the top 10 by WOSA after the fix:")
        print(top.assign(wosa=top.wosa.round(2), wosa_after=top.wosa_after.round(2), move=top.move.round(2))
              [["player_name", "wosa", "wosa_after", "move"]].to_string(index=False))
    np.set_printoptions(suppress=True)


if __name__ == "__main__":
    main()
