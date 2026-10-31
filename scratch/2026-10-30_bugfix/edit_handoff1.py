"""One-off edit for the bug-fix pass of 2026-10-30: HANDOFF's standing sections (kept for the record)."""
from pathlib import Path

p = Path("notes/HANDOFF.md")
s = p.read_text(encoding="utf-8")
pairs = [
    ("""| 3 | the rating | `prior_rapm.py` | RAPM pulled toward the prior, penalty 13,037, centred |""",
     """| 3 | the rating | `prior_rapm.py` | RAPM pulled toward the prior, penalty 13,037, centred; the season in progress holds its prior scales at the 8 nearest finished seasons' mean while it is young (experiment 109, `RidgeSettings.held_scale_through`, adopted 2026-10-30) |"""),
    ("""| within-season WOWY test | `scripts/checks/151_wowy.py`, stint unit, step 4 as shipped | better in every era, against the incumbent; the noise control beside it |""",
     """| within-season WOWY test | `scripts/checks/151_wowy.py`, stint unit, each system's own step 4 (`--shrink-flags`, the chain passes them) | better in every era, against the incumbent; the noise control beside it; the era table is printed and written (`wowy_<tag>_eras.csv`) |"""),
    ("""- **The season-page score**: each season, ratings of zero score 0 and the reference 100; the ratings before
  experiment 82 explain 18.2 a season on this test (the 15.7 in older notes is experiment 63's step 4).
  `scratch/2026-10-20_charts/season_scores.py` and `season_page.py`; page https://claude.ai/artifact/L7vwSTAvCCKQnj9jnnhYrq.""",
     """- **The season-page score**: each season, ratings of zero score 0 and the reference 100.  The zero line on today's
  stints is `outputs/csv/wowy_zero_baseline_inputs99_seasons.csv`, column zero, 3,681.6 a season: ratings of zero do
  not depend on the system, and every system is scored on the same held-back stints.  creation110 explains 19.7 a
  season, 14.1 to 26.2.  The older file, `wowy_zero_baseline_seasons.csv`, which
  `scratch/2026-10-20_charts/season_scores.py` reads, predates the parser fixes of 2026-10-22 (3,648.6).
- **Which z rules**: the plain z.  `z_newey_west` is floored at it since 2026-10-30: unfloored, a noise control
  read -2.44 against plain -1.83, so noise passed -2 (DECISIONS.md, "The bug-fix and make-better pass of
  2026-10-30").  Two builds that differ only in the box-score prior's random split differ by about 0.010 on the
  within-season test."""),
    ("""  "OpenRAPM experiments", tab Experiments; the next free row is 111.  Input screens get their own tab ("Input
  screen 112").""",
     """  "OpenRAPM experiments", tab Experiments; the next free row is 124.  Since 2026-10-30 it reads the within-season
  test by era too, and refuses a partial or stale reading.  Input screens get their own tab ("Input screen 112")."""),
    ("""    .venv/Scripts/python scripts/checks/101_log_run.py --name=NAME --exp=N --incumbent=inputs99 --idea=owner --verdict=pending --changed="..." --notes="..." """.rstrip(),
     """    .venv/Scripts/python scripts/checks/101_log_run.py --name=NAME --exp=N --idea=owner --verdict=pending --changed="..." --notes="..." """.rstrip()),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
