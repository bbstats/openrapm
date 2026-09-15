# Handoff: the single-year player rankings, one experiment at a time

**This file is transient.**  It starts the next session and is deleted when Phase 1 ships.  `DECISIONS.md`
is the permanent record and carries every number quoted here.  Do not let this grow into a lab notebook.

Branch `cleanup`; `main` is fast-forwarded to it at each publish.  `pytest -q`: **285 passed, 1 xfailed, ~120 s.**

## How we work (the owner, 2026-09-13/14)

- **Two jobs only: implement the owner's ideas, or propose new ones.**  Never run an idea the owner has
  not said "go" to.  One experiment at a time; never a queue of several overnight.
- **One test decides**: the year-over-year test below.  The consensus checks and the 2026 top 20 (the eye
  test) are read every time and can veto; they are never fitted to.
- **Every experiment ends with the 2026 top 20**, emailed as a phone-readable table when the owner is on a
  phone (`/experiment-comparison`; `scripts/66_compare.py` writes the page and the email body).
- **Plain words.**  No invented labels, no single-letter names, no "board" (say the player rankings), no
  "floor" (say the check).  Define a term the first time it is used.  End every message with a one-sentence
  TL;DR unless the message is itself one sentence.

## The rulings that bind

1. **One rating per player per season, from that season's games only**, regular season and playoffs.
2. **Single year or bust**: a player's own other seasons may not reach his rating, not even through the
   booster's memory (hence the out-of-player priors).  Model coefficients may be learned from other seasons.
3. **Never train on the current season until its Finals are over.**  Loading is always allowed.
4. **The SPM is trained on one row per player** (his career average with the rated season out) PLUS chunk
   rows of his seasons; not one row per player-season.
5. **Keep chimeraboost.  The consensus is a sanity check, never a fitting target**; a gross miss vetoes.
6. Each season is centred at possession-weighted zero per side (the RAPM convention), 2026-09-14.

## What ships (the incumbent, `scripts/62_single_year_board.py` defaults)

1. **Target**: `looseason.LeaveSeasonOutRAPM`, one RAPM per player over every season except the rated one
   (penalties 40,000 / 40,000 / 0, closed).
2. **SPM**: chimeraboost on `singleyear.chunk_rows` (the career row plus contiguous 1-, 2-, 3-season
   chunks, 35,647 rows, two features saying how much evidence a row rests on), features `boruta`
   (21 offence / 17 defence, on-court columns in).  **Out-of-player**: five player folds balanced on the
   label (`rloocv.BalancedGroupKFold`), every player's prior from the fit that never saw his rows.
3. **Rating**: `priorridge.PriorRidgeCV`, `scale x prior + residual`, the scale priced on cross-fitted prior
   columns (`--crossfit=scale`), the residual penalty **fixed at 13,037 on both sides** (adopted 2026-09-15;
   the per-season CV had switched the games off on offence in 2024-2026).  Then centred.

The product table (`outputs/season_ratings_product.parquet`, every other season allowed in the prior) is
what `docs/data/ratings.json` and the site are built from.  `artifacts/season_ratings.parquet` is still the
older system, kept for the tests; do not overwrite it.

Threads: the script pins BLAS to one thread and numba to four before importing anything.  Unpinned, one
booster fit went from 15 s to 45 minutes when a second process was running.  **Never run two builds at
once**, and kill a chain's python children when you stop it (stopping the wrapper leaves them running).

## The test: year-over-year

Rate a season from its own games.  Predict every stint of the season before and the season after from the
ten players' ratings alone, refitting only the intercept and home edge on the scored season.  Score
against actual points per team-game.  28 scored seasons, each predicted twice, paired by scored season.
The prior must not have seen the two scored seasons (`--exclude_neighbours=1`):

    .venv/Scripts/python scripts/62_single_year_board.py --exclude_neighbours=1 --score=0 --boards=<ten seasons> --out=season_ratings_<name>_<first>
        (three chunks of ten seasons, stitched; one chunk at a time keeps memory under control; ~3 min a season)
    .venv/Scripts/python scripts/63_yoy.py --rankings=<name>=outputs/season_ratings_<name>.parquet,incumbent=outputs/season_ratings_sy_lam13037.parquet --ref=incumbent --tag=<name> --splits=
    .venv/Scripts/python scratch/consensus_report.py outputs/season_ratings_<name>.parquet outputs/season_ratings_sy_lam13037.parquet
    .venv/Scripts/python scripts/66_compare.py incumbent=outputs/season_ratings_sy_lam13037.parquet <name>=outputs/season_ratings_<name>.parquet --season=2026 --top=20 --yoy=outputs/yoy_<name>.parquet --ref=incumbent

Read `game_armse` (points per 100 per team-game) and the paired row: `mean_diff` below zero is better,
`z` is the difference over its standard error, `wins` of 56.  **Decision rule:** adopt only if `z` is -2
or below with no gross consensus miss and the 2026 top 20 not worse; ties go to the simpler version.
The row "each side rescaled to the scored season" asks whether the order improved with the spread
removed; `--columns=prior` tests the SPM alone.  `scale_*` below 1 on the neighbouring season is expected
(players change year to year) and is not a target.

**Saved priors**: `--save_priors=<name>` writes every season's priors and their per-game-fold versions;
`--priors_from=<name>` reruns the ridge alone in a second a season, which is how the penalty was swept.

## The record so far (all in `DECISIONS.md`; year-over-year error per team-game, points per 100)

| rankings | error | verdict |
|---|---|---|
| shipped before this branch (`artifacts/`) | 8.587 | uses the banned `past_*` channel on offence |
| career row only, one per player | 8.804 | the starting point |
| + 1-3 season chunks (owner's design) | 8.736 | adopted |
| + scale cross-fitted, penalty not | 8.693 | adopted |
| + out-of-player priors | 8.705 | worse on the test, adopted by ruling 2 (memorised careers were the gain) |
| + fixed penalty 13,037 (**the incumbent**) | 8.697 | tie; consensus 0.777 / 0.757 / 0.771, top five 4 of 5; adopted |
| rejected: every chunk size; booster settings; off-court features; one row per player-season; cross-fitting the penalty; the un-shrunk label (8.667 on the test, but Clingan 3rd and Jokic 12th in 2026) | | |

## Next, in order (the owner, 2026-09-15)

1. **Replacement-level fill for thin players.**  Players with too few possessions to rate sit at 0; they
   should sit at the replacement level of their kind, about -2 to -5 per 100 (experiment 11 measured
   -5.4 under 500 label possessions, -3.9 to 1,500, -2.5 to 4,444 on offence; defence the mirror).  Do it
   OUTSIDE the SPM: a fill or a shrink target for the rating of a player under a possession threshold, by
   possessions and perhaps age, never a change to the label (that route scrambled the top of the list).
   Read the bottom of the 2026 list and the year-over-year test.
2. The Bayesian Gaussian mixture bias check on the incumbent (`scripts/58_archetype.py --board=...`).
3. Offence agreement with the consensus is 0.78 now; the shipped offensive prior is still better on the
   test only through the banned channel.  Ideas welcome; none queued.

## Traps that cost a day

1. `pip install -e` puts this working tree on `sys.path`: a fresh clone silently tests THIS repo's data.
2. Tests cannot reach the network (`tests/conftest.py`).
3. A score on a mask that cuts team-games is not a score; read stint-cutting splits on `mse`.
4. Two builds at once, or an unpinned numba pool, and a 15-second fit takes an hour.
5. `TaskStop` on a chain stops the bash wrapper only; its python children keep running and its `for`
   loop keeps launching.  Check `Get-Process python` before starting anything.
6. Seventeen more in `DECISIONS.md`, "The measurement traps".

## Verify you are where this file says

    .venv/Scripts/python -m pytest tests -q                                  # 285 passed, 1 xfailed, ~120 s
    .venv/Scripts/python scripts/63_yoy.py --rankings=incumbent=outputs/season_ratings_sy_lam13037.parquet,ship=artifacts/season_ratings.parquet --ref=incumbent --tag=verify --splits=
                                                                             # incumbent 8.697; ship -4.0 team-game MSE
    .venv/Scripts/python scratch/consensus_report.py outputs/season_ratings_sy_lam13037.parquet
                                                                             # 0.777 / 0.757 / 0.771, top5 4
