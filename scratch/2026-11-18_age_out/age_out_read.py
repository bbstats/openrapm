"""Read-only: what Box Adj's line loses or gains on the within-season test without age (notes/DECISIONS.md,
"Experiment 142: age out of Box Adj", read 1; the owner, 2026-11-18: "I want age removed!").

From the shipped line's saved sums, outputs/second_prior/<tag>/sums_<season>.npz (src/openrapm/rating_line.py: the
within-season test's normal equations over [each end's fitted part | offense's inputs | defense's inputs]).  For each
rated season the weights are solved on the seasons more than one from it, each counted 0.5 ** (distance / half-life),
as the test setting's line is, once on every column and once without the two age columns, and each set of weights is
scored on the rated season's own sums, `RatingSums.error`.  Paired by season, by era.

Also printed: the line's formula error itself beside experiment 140's 3,660.690 with every season counted alike and
0.207 less at a half-life of 8 (on padbins127's held-back games), reading 4 of "The rebuild read on the two tests".

Writes outputs/csv/age_out_read_seasons.csv and outputs/csv/age_out_read.csv.
    .venv/Scripts/python scratch/2026-11-18_age_out/age_out_read.py
"""
import sys

import numpy as np
import pandas as pd

from openrapm import rating_line, second_prior
from openrapm.data.config import load_config, resolve

sys.stdout.reconfigure(encoding="utf-8")
TAG = "rebuilt_line"
INPUTS = "context_types_team_zones_centred"
PENALTY = 1e-4
HALF_LIFE = 8.0
SEASONS = list(range(1997, 2027))
ERAS = (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("all 30", 1997, 2026))
F = rating_line.FITTED_PARTS


def solved(total: rating_line.RatingSums, at: np.ndarray) -> np.ndarray:
    """The line's ridge on the columns `at`, zeros elsewhere."""
    weights = np.zeros(total.Q.shape[0])
    weights[at] = rating_line._solve(total, at, PENALTY)
    return weights


def training_total(sums: dict, rated: int, half_life: float) -> rating_line.RatingSums:
    """The seasons more than one from `rated`, each counted 0.5 ** (distance / half_life); alike with half_life 0."""
    parts = [sums[season].scaled(0.5 ** (abs(season - rated) / half_life) if half_life > 0 else 1.0)
             for season in SEASONS if abs(season - rated) > 1]
    return sum(parts[1:], parts[0])


def by_era(table: pd.DataFrame, column: str, against: str) -> pd.DataFrame:
    rows = []
    for era, first, last in ERAS:
        change = (table[column] - table[against])[(table.season >= first) & (table.season <= last)].to_numpy()
        error = change.std(ddof=1) / np.sqrt(len(change))
        rows.append(dict(reading=f"{column} minus {against}", era=era, change=round(float(change.mean()), 4),
                         standard_error=round(float(error), 4), paired_z=round(float(change.mean() / error), 2),
                         seasons_better=int((change < 0).sum()), seasons=len(change)))
    return pd.DataFrame(rows)


def main() -> None:
    directory = resolve(load_config(), "outputs") / "second_prior" / TAG
    names = second_prior.INPUT_SETS[INPUTS]
    n_inputs, age = len(names), names.index("age")
    sums = {}
    for season in SEASONS:
        with np.load(directory / f"sums_{season}.npz") as data:
            sums[season] = rating_line.RatingSums(Q=data["Q"], c=data["c"], yy=float(data["yy"]), sw=float(data["sw"]))
    n_columns = sums[SEASONS[0]].Q.shape[0]
    assert n_columns == F + 2 * n_inputs, (n_columns, n_inputs)
    every = np.arange(n_columns)
    without_age = np.array([column for column in every if column not in (F + age, F + n_inputs + age)])
    rows = []
    for rated in SEASONS:
        faded = training_total(sums, rated, HALF_LIFE)
        alike = training_total(sums, rated, 0.0)
        with_age = solved(faded, every)
        rows.append(dict(season=rated,
                         every_season_alike=sums[rated].error(solved(alike, every)),
                         line=sums[rated].error(with_age),
                         line_without_age=sums[rated].error(solved(faded, without_age)),
                         age_weight_offense=with_age[F + age], age_weight_defense=with_age[F + n_inputs + age]))
    table = pd.DataFrame(rows)
    out = resolve(load_config(), "outputs") / "csv"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "age_out_read_seasons.csv", index=False)
    summary = pd.concat([by_era(table, "line_without_age", "line"), by_era(table, "line", "every_season_alike")],
                        ignore_index=True)
    summary.to_csv(out / "age_out_read.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{TAG}: {n_inputs} inputs an end, age is input {age}; half-life {HALF_LIFE:g}, penalty {PENALTY:g}")
    print(f"the formula's mean error over the 30 rated seasons: every season alike "
          f"{table.every_season_alike.mean():.4f}; the line, faded {table.line.mean():.4f}; "
          f"without age {table.line_without_age.mean():.4f}")
    print("\npaired by season; below 0 is better")
    print(summary.to_string(index=False))
    print("\nthe weight on age, per standard deviation of age, test setting, by rated season")
    print(table[["season", "age_weight_offense", "age_weight_defense", "line", "line_without_age"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
