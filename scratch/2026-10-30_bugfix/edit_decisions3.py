"""One-off edit for the bug-fix pass of 2026-10-30: experiment 109 continued, its result, and its row (kept for the
record)."""
from pathlib import Path

import pandas as pd

p = Path("notes/DECISIONS.md")
s = p.read_text(encoding="utf-8")
anchor = """up to the share where the fitted scale wins in every era; if held wins at every share tested, it holds through three
quarters of the regular season.
"""
result = """up to the share where the fitted scale wins in every era; if held wins at every share tested, it holds through three
quarters of the regular season.

### Experiment 109 continued: result (2026-10-30, `outputs/exp109_more.log`, `outputs/csv/wowy_exp109_cutNN_eras.csv`)

The within-season WOWY test on the stints of the rest of each season, held minus fitted, both with the whole season's
step 4; below 0 is better.  Newey-West z floored at the plain one (check 151 since tonight).

| share of season | era | change | standard error | paired z | seasons better |
|---|---|---|---|---|---|
| 0.25 | 1997-2006 | -0.567 | 0.201 | -2.82 | 7 of 10 |
| 0.25 | 2007-2016 | -1.048 | 0.238 | -4.40 | 10 of 10 |
| 0.25 | 2017-2026 | -0.755 | 0.225 | -3.36 | 9 of 10 |
| 0.25 | all seasons | -0.790 | 0.129 | -6.13 | 26 of 30 |
| 0.50 | 1997-2006 | -0.029 | 0.074 | -0.39 | 4 of 10 |
| 0.50 | 2007-2016 | -0.139 | 0.115 | -1.21 | 4 of 10 |
| 0.50 | 2017-2026 | -0.200 | 0.111 | -1.81 | 7 of 10 |
| 0.50 | all seasons | -0.123 | 0.058 | -2.11 | 15 of 30 |
| 0.75 | 1997-2006 | +0.009 | 0.042 | +0.21 | 5 of 10 |
| 0.75 | 2007-2016 | +0.060 | 0.085 | +0.71 | 5 of 10 |
| 0.75 | 2017-2026 | -0.031 | 0.048 | -0.65 | 7 of 10 |
| 0.75 | all seasons | +0.013 | 0.035 | +0.36 | 17 of 30 |

Before the shrink the held scale gains more: -3.43, -0.87 and -0.22 at the three shares, z -8.6, -3.1 and -1.1.

**Reading and decision.**  Held wins in every era through a quarter of a season, still in every era by a little at half
(z -2.1 over all seasons, 15 of 30), and ties at three quarters.  The fitted scale wins in no era at any share tested,
so the registered rule's first part never stops the hold, and held does not win at three quarters, so its second part
does not hold through three quarters either; the season in progress holds its scales until half of its regular season
is played, the last share where held is better in every era, and fits them from there on, as a finished season does
(`RidgeSettings.held_scale_through` 0.50, the owner's answer, "test on, then adopt").
"""
assert s.count(anchor) == 1
s = s.replace(anchor, result)
p.write_text(s, encoding="utf-8", newline="\n")

table_path = Path("experiments/runs.csv")
table = pd.read_csv(table_path, dtype={"exp": str, "commit": str})
row = table.exp == "109"
assert row.sum() == 1
table.loc[row, "verdict"] = "Adopted"
table.loc[row, "notes"] = ("Held beats fitted on the rest of each season at 2%, 5%, 10% and 25% of it in every era, "
                           "at 50% in every era by a little (z -2.1), ties at 75%; adopted 2026-10-30 for the season "
                           "in progress through half its regular season (RidgeSettings.held_scale_through 0.50).")
table.to_csv(table_path, index=False, lineterminator="\n")
print("ok")
