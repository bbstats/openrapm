"""Read-only: experiment 143's line, `rd143_line`, against the read that chose it.

The line's sums (outputs/second_prior/rd143_line/sums_<season>.npz) must be the screen's (outputs/direct/reads143) over
the same columns: today's 56 inputs an end, then his offensive rebound share and his close-game share over his share on
each end, the offense end's two all zero.  Then the line's own formula error against today's line, paired by season, by
era: the read gave -0.102, z -3.52.

    .venv/Scripts/python scratch/2026-11-19_box_adj_reads/line143_check.py
"""
import json
import sys

import numpy as np

from openrapm import rating_line, second_prior
from openrapm.data.config import load_config, resolve

sys.stdout.reconfigure(encoding="utf-8")
SEASONS = list(range(1997, 2027))
LIST = "context_types_no_age_team_zones_defence_only_centred"
F = rating_line.FITTED_PARTS


def load(directory, season):
    with np.load(directory / f"sums_{season}.npz") as data:
        return rating_line.RatingSums(Q=data["Q"], c=data["c"], yy=float(data["yy"]), sw=float(data["sw"]))


def main() -> None:
    out = resolve(load_config(), "outputs")
    names = second_prior.INPUT_SETS[LIST]
    meta = json.loads((out / "direct" / "reads143" / "names.json").read_text())
    at = {name: i for i, name in enumerate(meta["names"])}
    k = len(names)
    screen_columns = ["O:fitted", "D:fitted", *[f"O:{n}" for n in names], *[f"D:{n}" for n in names]]
    worst = 0.0
    line, today = {}, {}
    for season in SEASONS:
        line[season] = load(out / "second_prior" / "rd143_line", season)
        today[season] = load(out / "second_prior" / "noage142_line", season)
        with np.load(out / "direct" / "reads143" / f"season_{season}.npz") as data:
            Q, c = data["Q"], data["c"]
        idx = np.array([at[name] for name in screen_columns])
        zero = np.array([F + names.index(n) for n in second_prior.DEFENCE_ONLY])     # the offense end's two
        expected_q, expected_c = Q[np.ix_(idx, idx)].copy(), c[idx].copy()
        expected_q[zero, :], expected_q[:, zero], expected_c[zero] = 0.0, 0.0, 0.0
        gap = max(np.max(np.abs(line[season].Q - expected_q)) / np.max(np.abs(expected_q)),
                  np.max(np.abs(line[season].c - expected_c)) / np.max(np.abs(expected_c)))
        worst = max(worst, gap)
    print(f"rd143_line's sums against the screen's, largest relative gap over 30 seasons: {worst:.1e}")
    change = []
    for rated in SEASONS:
        errors = []
        for sums in (line, today):
            parts = [sums[s].scaled(second_prior.season_weight(s, rated, 8)) for s in SEASONS if abs(s - rated) > 1]
            total = sum(parts[1:], parts[0])
            errors.append(sums[rated].error(rating_line._solve(total, np.arange(total.Q.shape[0]), 1e-4)))
        change.append(errors[0] - errors[1])
    change = np.array(change)
    for era, first in (("1997-2006", 1997), ("2007-2016", 2007), ("2017-2026", 2017), ("all 30", None)):
        part = change if first is None else change[(np.array(SEASONS) >= first) & (np.array(SEASONS) < first + 10)]
        print(f"  {era}: {part.mean():+.4f}, z {part.mean() / (part.std(ddof=1) / np.sqrt(len(part))):+.2f}, "
              f"{int((part < 0).sum())} of {len(part)} better")
    print(f"weights for 2026 on the two, defense end: see run.json; columns {k} an end")


if __name__ == "__main__":
    main()
