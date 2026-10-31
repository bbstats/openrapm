"""One-off edit for the bug-fix pass of 2026-10-30: the hustle fix's result and the rest of the pass in DECISIONS.md
(kept for the record)."""
from pathlib import Path

p = Path("notes/DECISIONS.md")
s = p.read_text(encoding="utf-8")
old = """- **Once 2027's Finals are over**: `openrapm data hustle-games 2027 2027 --force` (and `matchup-games`) before its
  panel, since the files fetched during the season may still be partial.
"""
new = """- **Once 2027's Finals are over**: `openrapm data hustle-games 2027 2027 --force` (and `matchup-games`) before its
  panel, since the files fetched during the season may still be partial.
- **The result**, 2027 on its 71 games against the morning's published ratings, the prior scale fitted in both:
  players with 60+ possessions moved a median 0.17 per 100 in total, the ninetieth percentile 0.45, the largest 0.98
  (offense median 0.10, defense 0.11); 0 of 319 lack a hustle value, against 44; the top 20 the same players but Paul
  George in, 23rd to 18th, and Kyrie Irving out, 19th to 23rd; every finished season's panel row and rating unchanged.

### The rest of the pass (2026-10-30)

- **The held prior scale on 2027** (experiment 109 adopted through 10% of a season first, `RidgeSettings`): held at
  1.116 / 1.000, the mean of 2019-2026 in the published step-3 table, against 1.72 / 1.83 fitted on its 71 games.
  Against the fitted 2027, both with the hustle fix: players with 60+ possessions moved a median 0.70 per 100 in
  total, the ninetieth percentile 1.55, the largest 5.04; the spread of totals 1.78 against 2.81 fitted and 2.54 in
  2026's whole season (offense 1.62 against 2.48, defense 0.93 against 1.66).  The top 20 the same players but Anthony
  Edwards in, 26th to 19th, and Kris Dunn out, 17th to 24th.  The box-score prior's share-of-season input is held at
  0.25, the least its training rows cover (`box_prior.least_share`); the season-so-far folds still read it as is.
- **Test tools** (ccb3d9b): check 151's Newey-West standard error floored at the plain one; 151 scores each system with
  its own shrink (`--shrink-flags`), the by-era defence factor form parsed, the era table printed; the chain refuses a
  name already built unless `REUSE=1`, and opens each run's log with a chain start line; 101 logs the within-season
  test by era and refuses a partial or stale reading; 66 gives no summary from a partial year-over-year file; the input
  screen's added arms keep today's inputs per tree.  `openrapm folds` records its boosters in each fold folder: its
  first version replaced the record while another process of the same run read it, and on Windows the read refused, so
  one part of experiment 109's half-season held run stopped; fixed in 6a3b54f, its 10 seasons rebuilt.
- **Small fixes** (6a3b54f): the season in progress counts the games its stints hold; the power page's teams read
  against the average team, which the plain sums put at +0.60 with 17 of 30 teams above it.
- **Not done: a settings check for the morning job.**  The job rates 2027 with the working tree's code, so a setting
  adopted but not published reaches 2027 the next morning (it happened once: 2027 rated with part-season rows while
  1997-2026 was not).  The published tables are written by three commands, `rate`, `shrink` and `swap`, each with
  its own flags, so a record of what built them needs all three to write theirs, and a check that misfires stops the
  unattended morning.  Open; the owner's call whether the job should run from a checkout of the published commit.
"""
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
