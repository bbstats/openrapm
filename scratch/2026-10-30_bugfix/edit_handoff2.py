"""One-off edit for the bug-fix pass of 2026-10-30: HANDOFF's status (kept for the record)."""
from pathlib import Path

p = Path("notes/HANDOFF.md")
s = p.read_text(encoding="utf-8")
pairs = [
    ("2026-10-27, 2026-10-28 and 2026-10-29; the earlier file is in git history.",
     "2026-10-27, 2026-10-28, 2026-10-29 and 2026-10-30; the earlier file is in git history."),
    ("## Where it stands, 2026-10-29\n",
     """## Where it stands, 2026-10-30 night: the bug-fix and make-better pass

The owner: "Bugfix/make better mode. Go."; four answers at 9 PM, all the plan's recommendations.  DECISIONS.md, "The
bug-fix and make-better pass of 2026-10-30", experiment 109 continued and experiment 123 hold every number.

- **2027's hustle inputs, a live bug since the daily job began, fixed** (6195755, 186424b): the 6 AM job cached the
  league's placeholder hustle box scores, 55 of 71; and each game's box score fills in over days, so the season in
  progress now reads the league's season-to-date dashboards.  Its box scores are never cached empty
  (`dashboards.NotProcessedYet`).  **When 2027's Finals are over: `openrapm data hustle-games 2027 2027 --force`
  (and `matchup-games`) before its panel.**
- **The held prior scale, adopted** (experiment 109 and its continuation; `RidgeSettings.held_scale_through` 0.50):
  the season in progress holds its prior scales at the 8 nearest finished seasons' mean through half its regular
  season, then fits them.  2027 today: 1.116 / 1.000 held, against 1.72 / 1.83 fitted on 71 games.
- **What the 6 AM job of 2026-10-31 publishes**: 2027 with both, against yesterday's published 2027: the top 20 below.
  It is the first morning these reach the site.
- **Test tools fixed** (ccb3d9b): the Newey-West z is floored at the plain one, so the plain z rules; check 151 scores
  each system with its own shrink (`--shrink-flags`) and prints the eras; the full-build script refuses a name already
  built unless `REUSE=1`; `openrapm folds` records its boosters per folder; 101 logs the within-season test by era and
  refuses partial or stale readings; the input screen's added arms keep the inputs per tree.
- **Experiment 123, the steadier within-team multiplier**: see its entry; verdict the owner's.
- **Open, the owner's call**: a settings check for the morning job (DECISIONS.md, "The rest of the pass": the job runs
  the working tree's code, and the published tables come from three commands); experiments 111-122's verdicts.

## Where it stood, 2026-10-29
"""),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
