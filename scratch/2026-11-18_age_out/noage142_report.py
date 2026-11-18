"""Read-only: who moves when age leaves Box Adj's line (notes/DECISIONS.md, "Experiment 142"; the owner, 2026-11-18:
"I want age removed!").

The test-setting tables of the incumbent and the candidate, joined player by player, with each player's age from the
season panel.  Prints: how far the regulars move; the mean move by age band, by era; the 2026 top 20 of either table
side by side; 2026's largest movers; 2026's youngest and oldest regulars.  A move is the candidate minus the
incumbent, in points per 100, positive good on both ends.

Writes outputs/csv/noage142_moves.csv (every player-season) and outputs/csv/noage142_by_age.csv.
    .venv/Scripts/python scratch/2026-11-18_age_out/noage142_report.py
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
INCUMBENT, CANDIDATE = "rebuilt", "noage142"
REGULAR = 1000.0                 # offensive possessions; 2026's lists use 2,000
BANDS = [(0, 22, "22 and under"), (23, 25, "23 to 25"), (26, 28, "26 to 28"), (29, 31, "29 to 31"),
         (32, 34, "32 to 34"), (35, 99, "35 and over")]
ERAS = (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("all 30", 1997, 2026))
SIDES = ("off", "def", "total")


def band_of(age: pd.Series) -> pd.Series:
    out = pd.Series("", index=age.index, dtype=object)
    for low, high, name in BANDS:
        out[(age.round() >= low) & (age.round() <= high)] = name
    return out


def joined() -> pd.DataFrame:
    columns = ["player_id", "season", "player_name", "poss_off", "rating_off", "rating_def", "rating_total"]
    today = pd.read_parquet(f"outputs/season_ratings_{INCUMBENT}.parquet", columns=columns)
    candidate = pd.read_parquet(f"outputs/season_ratings_{CANDIDATE}.parquet", columns=columns)
    table = today.merge(candidate, on=["player_id", "season"], suffixes=("", "_new"), how="inner")
    assert len(table) == len(today) == len(candidate), (len(table), len(today), len(candidate))
    panel = pd.read_parquet("outputs/season_panel.parquet", columns=["side", "player_id", "season", "age"])
    table = table.merge(panel[panel.side == "O"].drop(columns="side"), on=["player_id", "season"], how="left")
    for side in SIDES:
        table[f"move_{side}"] = table[f"rating_{side}_new"] - table[f"rating_{side}"]
    return table.assign(band=band_of(table.age))


def by_age(table: pd.DataFrame) -> pd.DataFrame:
    regulars = table[table.poss_off >= REGULAR]
    rows = []
    for era, first, last in ERAS:
        inside = regulars[regulars.season.between(first, last)]
        for _, _, name in BANDS:
            group = inside[inside.band == name]
            weights = group.poss_off.to_numpy(float)
            rows.append(dict(era=era, age=name, player_seasons=len(group),
                             **{f"mean_move_{side}": round(float(np.average(group[f"move_{side}"], weights=weights)), 3)
                                for side in SIDES}))
    return pd.DataFrame(rows)


def shown(frame: pd.DataFrame) -> str:
    columns = ["player_name", "age", "poss_off", "rating_off", "rating_def", "rating_total", "rating_total_new",
               "move_off", "move_def", "move_total"]
    return frame[columns].round(2).to_string(index=False)


def main() -> None:
    table = joined()
    table.to_csv("outputs/csv/noage142_moves.csv", index=False)
    regulars = table[table.poss_off >= REGULAR]
    print(f"{len(table)} player-seasons, {len(regulars)} of {REGULAR:,.0f} or more offensive possessions")
    for side in SIDES:
        move = regulars[f"move_{side}"].abs()
        print(f"  {side}: the regulars move a median {move.median():.3f}, the ninetieth percentile "
              f"{move.quantile(0.9):.3f}, the largest {move.max():.3f}; correlation "
              f"{np.corrcoef(regulars[f'rating_{side}'], regulars[f'rating_{side}_new'])[0, 1]:.4f}")
    ages = by_age(table)
    ages.to_csv("outputs/csv/noage142_by_age.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nthe mean move by age among the regulars, weighted by possessions")
    print(ages.to_string(index=False))
    season = table[table.season == 2026]
    top = season[season.rating_total.rank(ascending=False).le(20) | season.rating_total_new.rank(ascending=False).le(20)]
    print("\n2026, the top 20 of either table")
    print(shown(top.sort_values("rating_total_new", ascending=False)))
    starters = season[season.poss_off >= 2000]
    print(f"\n2026, the 12 largest moves among the {len(starters)} players of 2,000 or more offensive possessions")
    print(shown(starters.reindex(starters.move_total.abs().sort_values(ascending=False).index).head(12)))
    print("\n2026, the eight youngest of them")
    print(shown(starters.sort_values(["age", "poss_off"], ascending=[True, False]).head(8)))
    print("\n2026, the eight oldest of them")
    print(shown(starters.sort_values(["age", "poss_off"], ascending=[False, False]).head(8)))
    move = starters.move_total.abs()
    print(f"\n2026, those {len(starters)}: a median move of {move.median():.3f}, the ninetieth percentile "
          f"{move.quantile(0.9):.3f}, the largest {move.max():.3f}")


if __name__ == "__main__":
    main()
