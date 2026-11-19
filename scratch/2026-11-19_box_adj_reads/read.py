"""Read-only: experiment 143's reads 1 to 3 from screen.py's sums (notes/DECISIONS.md, "Experiment 143: the reads on Box
Adj without age").  Every line is solved for each rated season on the seasons more than one from it, each counted
second_prior.season_weight(other, rated, 8), at `rating_line._solve`'s penalty, and scored on the rated season's sums.

Writes outputs/csv/reads143_add_one.csv, reads143_steps.csv, reads143_lists.csv, reads143_seasons.csv.
    .venv/Scripts/python scratch/2026-11-19_box_adj_reads/read.py
"""
import json
import sys

import numpy as np
import pandas as pd

from openrapm import rating_line, second_prior
from openrapm.data.config import load_config, resolve
from openrapm.features import FEATURE_SETS

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
TAG = "reads143"
SEASONS = list(range(1997, 2027))
ERAS = (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026))
HALF_LIFE = 8.0
PENALTY = 1e-4
GAIN = 0.042                     # experiment 140's size over the 30 seasons


class Screen:
    def __init__(self):
        cfg = load_config()
        directory = resolve(cfg, "outputs") / "direct" / TAG
        self.meta = json.loads((directory / "names.json").read_text())
        self.names = self.meta["names"]
        self.at = {name: i for i, name in enumerate(self.names)}
        self.sums = {}
        for season in SEASONS:
            with np.load(directory / f"season_{season}.npz") as data:
                assert float(data["gap"]) < 1e-9, (season, float(data["gap"]))
                self.sums[season] = rating_line.RatingSums(Q=data["Q"], c=data["c"], yy=float(data["yy"]),
                                                           sw=float(data["sw"]))
        self.train = {}
        for rated in SEASONS:
            parts = [self.sums[s].scaled(second_prior.season_weight(s, rated, HALF_LIFE))
                     for s in SEASONS if abs(s - rated) > 1]
            self.train[rated] = sum(parts[1:], parts[0])
        self.out = resolve(cfg, "outputs") / "csv"

    def errors(self, columns, seasons=SEASONS, penalty=PENALTY) -> np.ndarray:
        at = np.array([self.at[c] for c in columns])
        out = []
        for rated in seasons:
            w = rating_line._solve(self.train[rated], at, penalty)
            s = self.sums[rated]
            out.append((s.yy - 2.0 * w @ s.c[at] + w @ s.Q[np.ix_(at, at)] @ w) / s.sw)
        return np.array(out)


def summary(change: np.ndarray, seasons=SEASONS) -> dict:
    seasons = np.asarray(seasons)
    out = {}
    for era, first, last in ERAS:
        part = change[(seasons >= first) & (seasons <= last)]
        if len(part) > 2:
            out[era] = part.mean()
            out[f"z {era}"] = part.mean() / (part.std(ddof=1) / np.sqrt(len(part)))
        else:
            out[era], out[f"z {era}"] = np.nan, np.nan
    out["all"] = change.mean()
    out["z all"] = change.mean() / (change.std(ddof=1) / np.sqrt(len(change)))
    out["seasons better"] = int((change < 0).sum())
    out["seasons"] = len(change)
    return out


def passes(real: dict, null: dict) -> bool:
    return (all(real[e] < 0 for e, _, _ in ERAS) and real["all"] < -GAIN
            and abs(null["z all"]) < 2 and null["all"] > -GAIN)


def both(names):
    return [*[f"O:{n}" for n in names], *[f"D:{n}" for n in names]]


def main() -> None:
    s = Screen()
    meta = s.meta
    fitted = ["O:fitted", "D:fitted"]
    base = [*fitted, *both(meta["today"])]
    base_err = s.errors(base)
    seasons_table = pd.DataFrame({"season": SEASONS, "today": base_err})
    print(f"today's line, 56 inputs an end: mean error {base_err.mean():.4f} over the 30 rated seasons\n")

    # ---------------------------------------------------------------- read 1: lists at once
    candidates = meta["candidates"]
    lists = {"every candidate": candidates,
             "team rates the line lacks": meta["team"],
             "teammates' box rates": meta["mates"],
             "panel columns": meta["panel"],
             "the four built here": meta["built"]}
    rows = []
    for label, names in lists.items():
        real = summary(s.errors([*base, *both(names)]) - base_err)
        null = summary(s.errors([*base, *both([f"shuffled_{n}" for n in names])]) - base_err)
        rows.append(dict(list=label, inputs_an_end=len(names), **{k: real[k] for k in real},
                         shuffled_all=null["all"], shuffled_z=null["z all"]))
    table = pd.DataFrame(rows)
    table.to_csv(s.out / "reads143_lists.csv", index=False)
    print("Read 1. Each list added to today's line on both ends at once, minus today's line; below 0 is better")
    print(table.round(3).to_string(index=False))

    # ---------------------------------------------------------------- read 1: one at a time
    rows = []
    for name in candidates:
        for end in ("O", "D"):
            real = summary(s.errors([*base, f"{end}:{name}"]) - base_err)
            null = summary(s.errors([*base, f"{end}:shuffled_{name}"]) - base_err)
            rows.append(dict(end="offense" if end == "O" else "defense", input=name, **real,
                             shuffled_all=null["all"], shuffled_z=null["z all"], passes=passes(real, null)))
    add_one = pd.DataFrame(rows).sort_values("all")
    add_one.to_csv(s.out / "reads143_add_one.csv", index=False)
    print(f"\nRead 1. One candidate added on one end, {len(add_one)} rows; {int(add_one.passes.sum())} pass the rule; "
          f"shuffled copies' z from {add_one.shuffled_z.min():+.2f} to {add_one.shuffled_z.max():+.2f}")
    print(add_one.head(40).round(3).to_string(index=False))

    # ---------------------------------------------------------------- read 1: forward steps
    chosen, current, current_err, steps = [], list(base), base_err, []
    while True:
        best = None
        for name in candidates:
            for end in ("O", "D"):
                column = f"{end}:{name}"
                if column in chosen:
                    continue
                err = s.errors([*current, column])
                real = summary(err - current_err)
                if not (all(real[e] < 0 for e, _, _ in ERAS) and real["all"] < -GAIN):
                    continue
                if best is None or real["all"] < best[1]["all"]:
                    best = (column, real, err)
        if best is None:
            break
        column, real, err = best
        null = summary(s.errors([*current, f"{column[:2]}shuffled_{column[2:]}"]) - current_err)
        steps.append(dict(step=len(chosen) + 1, added=column, **real, shuffled_all=null["all"],
                          shuffled_z=null["z all"], shuffle_at_nothing=passes(real, null),
                          against_today=(err - base_err).mean()))
        if not passes(real, null):
            print(f"  {column} passes alone but its shuffled copy reads {null['all']:+.3f}, z {null['z all']:+.2f}: stop")
            break
        chosen.append(column)
        current, current_err = [*current, column], err
        print(f"  step {len(chosen)}: {column} {real['all']:+.4f}, z {real['z all']:+.2f}; against today "
              f"{(err - base_err).mean():+.4f}", flush=True)
    steps = pd.DataFrame(steps)
    steps.to_csv(s.out / "reads143_steps.csv", index=False)
    print("\nRead 1. Forward steps, each against the line before it")
    print(steps.round(3).to_string(index=False) if len(steps) else "none passes")
    seasons_table["forward"] = current_err
    if chosen:
        final = summary(current_err - base_err)
        print(f"\nthe forward line against today's: {final['all']:+.4f}, z {final['z all']:+.2f}; eras "
              + ", ".join(f"{final[e]:+.3f} z {final['z ' + e]:+.2f}" for e, _, _ in ERAS))
    print("\nthe line's penalty read again, each against the same list at 0.0001")
    rows = []
    for label, columns, at_default in (("today's line", base, base_err), ("the forward line", current, current_err)):
        for penalty in (1e-3, 1e-2, 1e-1):
            rows.append(dict(line=label, penalty=penalty, **summary(s.errors(columns, penalty=penalty) - at_default)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---------------------------------------------------------------- read 2: the LightGBM beside the line
    print("\nRead 2. The box-score prior out of the line, the LightGBM's own inputs offered in its place; "
          "each against today's line")
    lgb = FEATURE_SETS["hustle_no_three_pct_self_creation"]
    missing = [n for end in ("O", "D") for n in lgb[end] if n not in meta["today"] and n not in candidates]
    print(f"  LightGBM inputs not available to the screen: {missing}")
    boxless = [c for c in base if c[2:] not in ("box_prior", "box_prior_other_end")]
    added = {end: [n for n in lgb[end] if n not in meta["today"] and n in candidates] for end in ("O", "D")}
    hustle = {end: [n for n in added[end] if n.endswith("_vs_league_avg")] for end in added}
    plain = {end: [n for n in added[end] if n not in hustle[end]] for end in added}
    print(f"  offered, offense: {added['O']}\n  offered, defense: {added['D']}")
    late = [y for y in SEASONS if y >= 2017]
    late_base = base_err[-len(late):]
    rows = []
    for label, columns, seasons, against in (
            ("without the box-score prior", boxless, SEASONS, base_err),
            ("without it, its non-hustle inputs offered", [*boxless, *[f"{e}:{n}" for e in plain for n in plain[e]]],
             SEASONS, base_err),
            ("today's line, its non-hustle inputs offered beside it",
             [*base, *[f"{e}:{n}" for e in plain for n in plain[e]]], SEASONS, base_err),
            ("without it, all its inputs offered, 2017-2026",
             [*boxless, *[f"{e}:{n}" for e in added for n in added[e]]], late, late_base),
            ("today's line, all its inputs offered beside it, 2017-2026",
             [*base, *[f"{e}:{n}" for e in added for n in added[e]]], late, late_base)):
        err = s.errors(columns, seasons)
        rows.append(dict(line=label, **summary(err - against, seasons)))
        if seasons == SEASONS:
            seasons_table[label] = err
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---------------------------------------------------------------- read 3: the box-score prior by playing time
    print("\nRead 3. The box-score prior times his centered share of team possessions, against today's line")
    rows = []
    for label, columns in (("offense end", ["O:box_prior_x_share"]), ("defense end", ["D:box_prior_x_share"]),
                           ("both ends", both(["box_prior_x_share"]))):
        real = summary(s.errors([*base, *columns]) - base_err)
        null = summary(s.errors([*base, *[f"{c[:2]}shuffled_{c[2:]}" for c in columns]]) - base_err)
        rows.append(dict(added=label, **real, shuffled_all=null["all"], shuffled_z=null["z all"],
                         passes=passes(real, null)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    seasons_table.to_csv(s.out / "reads143_seasons.csv", index=False)


if __name__ == "__main__":
    main()
