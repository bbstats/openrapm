"""One-off: the record of the standard errors' build in notes/DECISIONS.md and notes/HANDOFF.md (run once)."""
from pathlib import Path

DECISIONS = '''
## The standard errors built: every season, one column (the owner, 2026-11-26: "Yes, every season", "One column")

**Asked of the owner with the preview's numbers:** whether the games' noise is the number wanted, against the larger
distance from a player's true value, and where the Ratings page shows it.  The answers: every season, the season in
progress included; one column beside Total holding the total's, the offense's and the defense's in Detail.

**What was built** (commit 423afa0; PIPELINE.md, "The ratings' standard errors").
- `src/openrapm/rating_error.py`, the jackknife, and `openrapm errors`, which writes
  `outputs/rating_errors_<name>.parquet`.  On `rd143_k10` it gives the preview's numbers to 1e-15: 1997-2026 in 22
  seconds.
- `openrapm folds --in-progress`: a season in progress rated 30 more times as its build rates it, by that build's own
  boosters, the boosters told the share of the season played, the prior scales held and blended by it.  2027's 30
  ratings took 119 seconds, every leak check passed.
- The site: the Detail file's rows end with [offense, defense, total] and the list file's with the total's, only when
  the errors file is beside the ratings table.  The Ratings page: a column headed ± beside Total, one decimal, and
  beside OpenRAPM in Detail, two decimals; the downloads carry them.  Without the field in the data the page is as it
  was.
- The morning job: the build keeps its boosters, then the 30 ratings and `openrapm errors --seasons 2027`.  Neither
  call is a step; a morning they fail keeps yesterday's standard errors.  All of it waits for
  `outputs/rating_errors_product.parquet` to exist.

**The season in progress does not rebuild to the last digit.**  The season so far rebuilt game by game sits up to 0.16
from the published 2027 rows, median 0.007 on offense and 0.010 on defense.  The cause: the build reads 2027's hustle
counts from the league's season-to-date dashboards, because a game's hustle box score fills in over the days after it,
and a fold can only sum the games it keeps; the game still waited on, 0022600265, adds two teams' shots.  The 30 ratings
are all rebuilt the same way, so their spread, which is the standard error, is not moved by it.  The command stops
past 0.5.

**2027 on 2026-11-26, 21.9% of the season:** 468 of 482 players have a standard error; medians 0.71 offense, 0.52
defense, 0.88 total.  Wembanyama +10.13 with 1.64, Doncic +6.85 with 1.16, Gilgeous-Alexander +4.51 with 0.53.  Of the
14,345 player-seasons the site shows, 5 have none: every game of theirs fell in one fold.

**Left as they are.**  The header ± and its two hover lines, "Standard error of Total" and "Standard error of
OpenRAPM", are placeholders: wording on the site is the owner's.  On a phone the Ratings table already scrolled
sideways; the new column sits where WOSA began.

**Not published.**  Nothing on the site changes until the errors file is made and main has the page.
'''

HANDOFF_OLD_START = "## Waiting on the owner, 2026-11-26: standard errors on the player ratings"
HANDOFF_NEXT = "## Start here (updated 2026-11-24, afternoon)"
HANDOFF_NEW = '''## Waiting on the owner, 2026-11-26: the word to publish the ratings' standard errors

- **Built and committed on this branch, 423afa0, not published.**  The owner asked for a standard error on each
  player's rating, then chose "Yes, every season" and "One column".  DECISIONS.md, "A standard error for each player's
  rating: the preview" and "The standard errors built"; PIPELINE.md, "The ratings' standard errors".
- **What it is.**  How far a rating would land had the season's games come out as another draw: the season rated 30
  more times, each without a tenth of its games.  A regular reads about 0.8 on his total.  It leaves out how far the
  box-score prior can miss a player whatever games are played.
- **Nothing published changes until two things happen**, both waiting on the owner's word:
  1. the errors file is put in place.  It is ready, made and then moved aside after the dry run of the morning job:
     `mv outputs/rating_se/rating_errors_product_ready.parquet outputs/rating_errors_product.parquet`, or made again
     with `OPENRAPM_WRITE_PUBLISHED=1 .venv/Scripts/python -m openrapm errors`.  From the next morning the 6 AM job
     then rates 2027 thirty more times, about 2 minutes, and writes its rows.
  2. main gets the page: `docs/index.html` of this branch, with the code.  Until then the data files would carry the
     field and the published page would not show it.
- **Wording that is the owner's:** the column header ± and its hover lines "Standard error of Total" and "Standard
  error of OpenRAPM"; a Changelog row; an About line.  None drafted.
- **Side files of the session, safe to delete:** `outputs/season_ratings_se2027*`, `outputs/prior_models_se2027.pkl`,
  `outputs/within/se2027_k10`, `outputs/season_ratings_sepreview*`, `outputs/rating_errors_se_try.parquet`,
  `outputs/rating_errors_sepreview.parquet`, `outputs/csv/season_ratings_se_try_step4_multipliers.csv`.

'''


def main() -> None:
    decisions = Path("notes/DECISIONS.md")
    decisions.write_text(decisions.read_text(encoding="utf-8").rstrip("\n") + "\n" + DECISIONS, encoding="utf-8",
                         newline="\n")
    handoff = Path("notes/HANDOFF.md")
    text = handoff.read_text(encoding="utf-8")
    start, end = text.index(HANDOFF_OLD_START), text.index(HANDOFF_NEXT)
    handoff.write_text(text[:start] + HANDOFF_NEW + text[end:], encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
