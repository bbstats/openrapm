"""One-off: correct experiment 138's record after the matched-width read and add the tree-count read, 2026-11-05."""
from pathlib import Path

decisions = Path("notes/DECISIONS.md")
text = decisions.read_bytes().decode("utf-8")
wrong = ("The total rating's spread does not change, so trap 39 does not explain it: 2.47, 2.41, 2.36 by era against 2.47, 2.41, 2.36.  "
         "Defense widens a little in the two older eras, 1.80 and 1.70 to 1.84 and 1.73, and offense narrows about 1%.")
right = ("**Corrected the same night: this is width, trap 39, not a gain; see \"Experiment 138 at matched width\" below.**  The total rating's "
         "spread does not change, 2.47, 2.41, 2.36 by era against 2.47, 2.41, 2.36, but each side's does: defense widens in the two older eras, "
         "1.80 and 1.70 to 1.84 and 1.73, and offense narrows about 1%.")
assert wrong in text
text = text.replace(wrong, right)
text = text.rstrip("\r\n") + """

### Experiment 138 at matched width, and why 130 trees (the owner, 2026-11-05: "Let's be patient in our reasoning. Why did we choose that tree number? What are the real issues we are dealing with?")

Read-only.  `scratch/2026-11-04_coefficients/trees138_matched.py`, `trees138_hybrid.py`, `tree_count_read.py`; `outputs/csv/tree_count_search.csv`, `tree_count_curve.csv`, `trees138_team_totals.csv`.

**The year-over-year gain is width.**  The candidate's ratings with each side's spread, season by season, multiplied to today's among regulars of 1,000+ possessions; the multipliers average 0.983, 0.983, 0.995 on defense by era and 1.006, 1.010, 1.010 on offense.

| year over year, both directions | team-game change | team-game z | seasons better | stint change | stint z |
|---|---|---|---|---|---|
| as built | -0.058 | -5.35 | 40 of 56 | -0.036 | -2.18 |
| each side's width matched | -0.008 | -0.67 | 31 of 56 | -0.020 | -1.08 |

A check of the total's spread alone missed this: a side's width moving 1% moves the team-game test more than this candidate's order does.  At matched width the candidate ties on both tests.

**Team totals did not move.**  Today's split with the candidate's totals reads -0.003, z -1.27; the candidate's split with today's totals -0.054, z -5.15, through `second_prior.hold_team_totals`.  A team's total moves a standard deviation of 0.02 to 0.03 points per 100 for five players, against a spread of 2.1 to 3.1.  So a first guess, that the LightGBM's job of setting team totals was what the within-season test cannot see, is not what happened here: the gain sat in the split, and the split's gain is width.

**Why 130.**  Experiment 76's guarded search, 2026-10-16, before today's Box Adj.  Of 100 trials 73 passed its year-over-year limit.  Trial 75, 130 trees at a learning rate of 0.0091, beat trial 24, 150 trees at 0.0105, by 0.026, when a booster's random draw alone moved that test about 0.03.  Eight trials sit within 0.10 of the winner, on 105 to 236 trees, reaching 51% to 92%; the best twelve include 65 trees at 0.0165.  Among the 73 the score worsens with reach, Spearman 0.68.  The winner sits on three limits: rows per leaf 2,500 at its cap, 15 bins at the minimum, the leaf penalty 0.001 at its minimum.  **Why 330:** the handoff's formula for 95% reach, no data; the formula says the spread grows 1.37 times, and it grew 1.61.

**The LightGBM's own goal, by trees.**  One fit to 1,000 trees per booster, the same rows, resamples and player folds, read at each count; defense, regulars with a target, six seasons' mean, on players its boosters did not see.

| trees | root mean squared miss of the target | target per output point | correlation with target | correlation with the 130-tree output |
|---|---|---|---|---|
| 50 | 1.671 | 3.58 | 0.494 | 0.998 |
| 130 | 1.585 | 1.73 | 0.502 | 1.000 |
| 330 | 1.532 | 1.09 | 0.510 | 0.993 |
| 500 | 1.528 | 0.98 | 0.513 | 0.985 |
| 750 | 1.527 | 0.92 | 0.515 | 0.974 |
| 1,000 | 1.528 | 0.90 | 0.516 | 0.965 |

The miss bottoms at 330 trees in 2022 and 2026, about 500 in 2010, and is still falling at 1,000 in 2000 and 2005.  Most of the fall is scale, which Box Adj's weight absorbs; the correlation goes from 0.49 to 0.52, so the limit is the inputs.

**The LightGBM is close to a straight line.**  A straight line on its own inputs, fitted on the players with even ids and read on the odd, over regulars and by era, gets 92.6%, 93.4%, 94.3% of the output's variance on offense and 91.5%, 92.2%, 92.3% on defense.  Inputs under half filled in an era are left out.  Box Adj's own-box-rate rows are another straight line on the same stats, with another target, so the repeat comes with the design.

The within-season reads to set beside it are the direct screen's, on ctx110's line: no prior +8.370, the box-score prior alone +1.171, the 29 inputs refit -0.702; dropping the box-score prior from one end costs +0.082, z +4.81 on defense and +0.060, z +3.06 on offense.  Not read on today's line, and not read year over year.
"""
decisions.write_bytes(text.encode("utf-8"))

handoff = Path("notes/HANDOFF.md")
text = handoff.read_bytes().decode("utf-8")
wrong = ("year over year -0.058, z -5.35; the\n   2026 top 20 the same players.  Box Adj's weight fell 1.66 times against a spread up 1.61 times, so the scale argument held.  Verdict the owner's.**")
right = ("year over year -0.058, z -5.35, which is width: -0.008, z -0.67 with each side's width matched; the\n   2026 top 20 the same players.  Box Adj's weight fell 1.66 times against a spread up 1.61 times, so the scale argument held.  A tie on both tests.  Verdict the owner's.**\n"
         "   **Tree count is not the lever.**  130 won its search by less than a random draw moves it, and 330 is arithmetic.  The LightGBM is 92% to 94% a straight line on its inputs, and Box Adj's own box\n"
         "   rates are another, on another target, so the repeat comes with the design.  DECISIONS.md, \"Experiment 138 at matched width\".  Read a year-over-year change at matched width, each side's, before calling it a gain.")
assert wrong in text
handoff.write_bytes(text.replace(wrong, right).encode("utf-8"))
