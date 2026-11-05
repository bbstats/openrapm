"""One-off: notes/HANDOFF.md's next job, the box-score prior's second stage, 2026-11-05."""
from pathlib import Path

path = Path("notes/HANDOFF.md")
text = path.read_bytes().decode("utf-8")
old_heading = '## Next job: the box-score prior is narrow and Box Adj repeats it (the owner, 2026-11-04: "Write handoff to help us fix this")'
assert old_heading in text
section = '''## Next job: learn the box-score prior's second stage, so Box Adj is a true adjustment (the owner, 2026-11-05)

**The owner's words.**  "if the box prior for the player is getting some weight X, couldn't we roll that up into the initial box prior. more importantly - LEARN it as a 2nd stage of the box prior.  i just would like the box-adj to be a true 'adjustment' and be orthogonal", "orthogonal in concept, not saying it has to be completely decorrelated", "might be a bit slower and complicated but will be significantly more interpretable".  The owner likes today's ratings and thinks there are too many steps.

**The design.**

```text
Box score, the second stage = w0 x the LightGBM's answer + the other end's box-score prior + his 13 box rates + his role and body
Box Adj, the adjustment     = his teammates' and opponents' priors (8) + the 16 team rates
```

`team_possession_share` is his own share of possessions, a role input, in spite of its name.  Team totals stay as they are: the LightGBM's answer sets a team's total, every other input is moved to a team total of 0.

**What is known** (DECISIONS.md, "The prior's columns regrouped by what they read", "Experiment 138 at matched width", "The whole rating, by its columns").

- With today's weights, regrouping alone moves no rating.  Six seasons' mean among teammates, offense then defense: the own-stats column 0.71 and 0.74 to 1.66 and 1.32, Box Adj 1.17 and 0.92 to 0.40 and 0.40, their correlation 0.75 and 0.71 to 0.17 and 0.44.  In 2026 the correlation reads -0.07 and 0.54.
- The prior's two columns are 84% to 89% of a regular's rating variance.
- The LightGBM is 92% to 94% a straight line on its inputs, and Box Adj's own box rates are another straight line on the same stats, so changing one part is absorbed by the others.  Defense's boosters at 330 trees widened the output 1.61 times, Box Adj's weight fell 1.66 times, and both tests tied at matched width.

**The first read, read-only, minutes.**  `outputs/second_prior/rating134/sums_<season>.npz` holds every season's within-season normal equations, `rating_line.RatingSums`, over [offense's fitted part, defense's fitted part, offense's 45 inputs, defense's 45 inputs]; `rating_line.coefficients` shows the solve and `RatingSums.error` the test's error at any weights.  No fold needs rebuilding.  For each rated season, on the seasons it may train on, the test setting's rule (`second_prior.season_coefficients`):

1. Stage one: solve both ends' own-stats columns and the two fitted parts, the same column scaling and penalty as `coefficients`.
2. Stage two: hold those weights, solve the eight others' priors and 16 team rates per end, and solve the fitted parts again, since they stand where the games' multiplier stands.
3. Score the rated season with `RatingSums.error`, paired by season against today's joint weights, by era.

Check first that the code reproduces today's line and its error from the same sums before reading the two-stage number.  Register the read in DECISIONS.md before any result.

**How to read it.**  Against today's joint weights, by era 1997-2006, 2007-2016, 2017-2026, beside the noise control's -0.010, z -1.07.  Report the weights of stage two beside today's, and the two columns' spread and correlation among teammates through `second_prior.hold_team_totals` (`scratch/2026-11-04_coefficients/regroup_columns.py`, which needs the two-stage weights in place of the saved line).  The fit leaves out the ridge's prior scale; on 30 seasons the formula's error misses the full test's by a mean +0.008, standard deviation 0.042, so a gap that size is not a result.  It cannot read year over year.

**Then stop and show the owner.**  Show the owner the read and offer the choice through AskUserQuestion, with its numbers in the question:

- **Regroup only:** today's weights, new columns, no rating moves.  It needs the site's files rebuilt and the owner's word and wording; never draft prose for the site.
- **A full build of the two stages:** only on the owner's word.  `rating_line.coefficients` and `second_prior` need the stage order, then the recipe under the previous job, about 3 hours, outside 5:50 to 7:30 AM.  The within-season test better in every era against `mid136`, `noise_rating134` beside it, year over year not worse at z +2 and read at matched width, each side's spread multiplied to today's (`scratch/2026-11-04_coefficients/trees138_matched.py`), and the 2026 top 20.  The row after 138 is 139.  The settings stay the published ones until the owner's word, since the morning job runs on the working tree.

**Traps met on 2026-11-04 and 2026-11-05.**

- A year-over-year gain can be width: experiment 138 read z -5.35 as built and z -0.67 with each side's width matched, though the total's spread had not moved.  Match each side.
- A read on one column does not predict the rating: 330 trees moved the Box score column toward shorter players, and the final ratings did not follow.
- The within-season test fits the weights and judges them, so a win on it is less independent than it was.
- Say what is known.  Start from the whole rating and give one connected account, not a list of findings (the owner: "you are struggling with the full picture and are instead focusing on parts").

**Waiting on the owner.**  Experiment 138's verdict, row 138, pending; the recommendation is to leave 330 trees off.  Sheet row 66 of "OpenRAPM experiments" holds a stray copy of experiment 127; the tab has blank rows, so write a new experiment's row with update_values at the first free row after the last, never append.

'''
text = text.replace(old_heading, section + old_heading.replace("## Next job:", "## The previous job:"))
path.write_bytes(text.encode("utf-8"))
