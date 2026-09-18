# Handoff: the single-year player rankings, one experiment at a time

**This file is transient.**  It starts the next session and is deleted when Phase 1 ships.  `DECISIONS.md`
is the permanent record and carries every number quoted here.  Do not let this grow into a lab notebook.

Branch `cleanup`; `main` is fast-forwarded to it at each publish.  `pytest -q`: **330 passed, 1 xfailed, ~165 s** with the scraped data present; **305 passed,
10 skipped, 1 xfailed** on a clone without it (2026-09-17).

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
    .venv/Scripts/python scripts/64_consensus_report.py outputs/season_ratings_<name>.parquet outputs/season_ratings_sy_lam13037.parquet
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

## What 2026-09-15 settled (experiments 11-15, all in `DECISIONS.md`; nothing adopted)

- **The box prior is FLAT in exposure and the truth is steep.**  Centred at possession-weighted zero, the
  prior gives -0.76 to a man with 0-50 possessions and -0.81 to one with 1,000-2,000; APM gives -9.94 and
  -2.15.  Measured on 30 seasons, monotone, z 7 to 15.  **This is the real defect.**
- **About half of it is forecastable.**  The same men read -5.15 the season before and -3.64 the season
  after, playing about 700 possessions in each, so they are properly measured there and it is not ageing.
  The level to aim at is therefore the NEIGHBOURING seasons' APM, not the rated season's:

  | possessions | 0-50 | 50-100 | 100-200 | 200-500 | 500-1k | 1-2k | 2-4k | 4k+ |
  |---|---|---|---|---|---|---|---|---|
  | offence, neighbours | **-4.40** | -3.44 | -3.29 | -3.44 | -2.85 | -2.07 | -1.21 | +0.83 |
  | defence, neighbours (positive = allows more) | **+1.69** | +1.25 | +1.30 | +0.86 | +0.85 | +0.20 | -0.02 | -0.04 |

- **Four ways of applying it all failed, for two reasons.**  Fitting the level against the rated season's
  own APM runs away (the loss weights each player by his own information, so a four-possession man costs
  it nothing: it produced -22.41 per 100).  Adding covariates did not help -- more flexibility only
  changes the shape of the runaway.  Applying a correction AFTER the ridge fails differently: nothing
  re-prices it, the rankings get wider, and the test punishes the extra spread whatever the order did.
- **And a feature cannot teach it.**  `singleyear.chunk_rows` gives every training row of a player the
  SAME label as his career row, so the booster can learn "noisier inputs, regress harder" but never
  "fewer possessions, genuinely worse".  Handing it the garbage-time exposure came in at z -1.73 and left
  the prior's exposure spread unchanged (+2.42 to +2.30 against the +11.2 that is really there).

## Where this stands (2026-09-15, end of session)

**The exposure correction is closed.  The owner: *"whatever our version is as last posted seems to be good
enough."***  Six attempts (experiments 12-17) and not one beat the incumbent; the best available outcome
was the exact tie of the linear ramp at N = 100.  The shipped rankings are unchanged and nothing needs
rebuilding.  The case for closing it is in `DECISIONS.md` -- the short version is that the men being
corrected are 7% of the players and **0.13% of the possessions**, and the site already rates a
four-possession man -1.96 per 100, which is a defensible replacement level.

Do not reopen this without a new reason.  In particular, do not reopen it on the strength of the
measurement alone: the gap between the box prior and APM at low exposure is real and documented, and the
rankings are still better without a patch for it.

**The owner named the next experiment on 2026-09-16: the trade set.  It is built, measured and NOT
shipped; the findings and the reason are below.**

Left in the working tree from this session, all harmless and off by default:

- `scripts/62_single_year_board.py`: `--blend_off=x/N/lin` / `--blend_def=x/N/lin`, the linear ramp
  `w = min(n / N, 1)`.  The rational weight `x/k/a` is unchanged.  No default behaviour moved.
- `outputs/season_ratings_blend*.parquet` and `season_ratings_ramp*.parquet`: the two sweeps, disposable.
- `outputs/priors_sy_base.pkl`: **keep this.**  The saved priors at `--exclude_neighbours=1`, which is what
  makes any ridge-side experiment a one-second-a-season rerun instead of a 90-minute build.
- `scripts/67_blend_apm.py`, `68_exposure_slope.py`, `69_closeness_panel.py`: the measurement scripts
  behind experiments 12-15, untracked.

## The trade set (2026-09-16, the owner's design): findings, and why nothing shipped

**The owner's verdict on reading the ratings: *"basically no movement."*  Not shipped.  But the
dismissal was argued on rank places and the owner's own threshold overturns it, so the measurement is
recorded here in the unit that decides: points per 100, with anything under 0.1 counted as nothing.**

Absolute change against the incumbent, 12,103 eligible player-seasons with absence evidence, 30 seasons:

| | median | ninth decile | largest | over 0.1 | over 0.25 |
|---|---|---|---|---|---|
| the per-player correction alone | **0.273** | 0.784 | 2.43 | **79.1%** | 53.2% |
| the uniform rescale's part | 0.381 | 0.909 | 3.88 | 83.3% | 64.2% |
| both together | 0.512 | 1.244 | 4.61 | 89.7% | 74.4% |

By exposure, the per-player part alone: under 500 possessions median 0.130 and 58.7% over the
threshold; 500 to 1,500, 0.244 and 79.1%; 1,500 to 3,000, 0.306 and 82.8%; over 3,000, 0.356 and 85.0%.

**So the movement is real in size and it is not concentrated in the bench.**  The rank reading that
prompted the dismissal was the wrong instrument, and it was also wrong on the facts: the middle of the
list is not as tightly packed as it looks.  Near the middle of a 2026 season 0.84 points per 100 spans
20% of the players, so the men who moved 18 to 26 places moved a median of **0.243** points per 100.
The top of the list genuinely does not move -- the 2026 top five is the same five men reordered, the top
20 keeps 15.7 of its 20 names over 30 seasons and rank agreement is 0.942 -- but "nothing changed" is
not what the points say.

**Why it still does not ship, and this is the real reason.**  Alpha reads the games of the seasons
either side of the rated one, so it can never be a published rating (ruling 1).  The route to shipping
was to teach the prior what alpha knows, and a season's statistics see **3.7% of alpha on offence and
5.6% on defence**.  The correction is large enough to matter and mostly invisible to the box score.

### What it is

For a rated season, every team-game of that season and the two either side is one row.  A player's
entry is his share of that team's possessions in that game, and it is **zero for a game he missed and
zero for his old team's games once he is traded**.  The rated season's rating is subtracted from every
row and the remainder is ridged back onto the same player columns.  That coefficient is **alpha**: what
a player's comings and goings say that his rating did not already know.  Every teammate has a column
too, which is why this is a regression and not a difference of averages -- a team's record without one
man is mostly a statement about the other four, which is what the off-court record died of.

Three terms, used the same way everywhere:

* **the trade set** -- the pooled team-games of the three seasons, one row per team per game.
* **alpha** -- the per-player correction the trade set puts on top of the rating, points per 100.
* **the trade loss** -- the size of alpha over eligible players, every player counting once, by
  possession tier.  The year-over-year test is a team-game error in which a 200-possession man is a
  rounding error; this is the loss that can see him.  Offence 0.300 and defence 0.256 pooled, against a
  rating spread of 1.67 and 1.12, so the correction is about a quarter of the spread.

### How to run it

    python scripts/70_tradeset.py --team_effects=team --out=tradeset_team   # the alpha table and losses
    python scripts/71_tradeset_features.py --alpha=outputs/tradeset_team_alpha.parquet --out=tradeset_team
    python scripts/72_tradeset_shap.py --out=tradeset_team   # WHICH statistics move a player, and how far
    python scripts/70_tradeset.py --block=0 --out=tradeset_block0            # the two controls
    python scripts/70_tradeset.py --team_effects=team_season --out=tradeset_ts

Nine minutes, three minutes, three minutes.  `src/eracoef/tradeset.py` carries the reasoning;
`tests/test_tradeset.py` is 14 tests including truth recovery on a simulated league.  `scripts/63_yoy.py`
gained `--offset=`.  Every output is a new `tradeset_*` name: no shipped artefact is written and
`role_panel_season.parquet` is read only.

### What it found

1. **The offensive rating is about a quarter too wide.**  The multiplier the three seasons ask for is
   **0.749 on offence and 0.919 on defence**, and correcting only that -- one number per side, no player
   told apart from another -- removes 0.069 of game error two seasons out.  This is the largest single
   thing the trade set found and it is not about any player.  Note the sign flips inside a single season
   (1.03 to 1.12 offence, 1.38 to 1.51 defence): within its own games the ridge has over-shrunk, across
   seasons it has not shrunk enough.
2. **The per-player correction removes 0.053 on top of that**, 53 of 56 observations, with one free
   level per team in the fit.  Without that team term it reads 0.098 and **about half of it is the team
   being good, not the player**.  Under a free level per team per SEASON it is 0.023; that version also
   absorbs a player's standing among his own teammates, so read it as a lower bound.
3. **Nine tenths of the per-player gain needs the neighbouring seasons.**  The same machinery on the
   rated season alone, where there are no absences to see, removes 0.009.  The absences are doing the
   work, not the extra data.
4. **Three statistics are under-credited by the rating**, each pushing a correction up about 0.02 per
   100 at its high end: **three-point rate, shot creation, offensive rebound share**.  This is the most
   directly actionable thing here.
5. **On defence the prior reads team offence as defensive credit.**  On-court plus-minus is the loudest
   real statistic on both sides, and a player whose team scored most while he was on the floor has his
   defensive correction pushed **down 0.062**.  That is a defect with a named mechanism.

Scored through the year-over-year test at two seasons out, the corrected ratings read 8.6784 against the
incumbent's 8.7949, better in 52 of 54 seasons.  Most of that is item 1.

### Why the statistics cannot yet carry it

Out of fold, with the how-much-he-played family removed, a season's statistics account for **3.7% of
alpha on offence and 5.6% on defence**, against 0.8% and 1.7% for the rankings' own rating and the
panel's `rapm1`.  (That second number was written up as "the rankings' prior" and is not: the
feature scripts 71 and 72 use is `rapm1`, the role prior plus the season's own residual, not the
box prior the rankings are centred on.  Corrected 2026-09-17; the measurement is unchanged.)
Three to four and a half times better, and still a few per cent.  Read one player at a time it is
starker: Westbrook is corrected -1.52 and the statistics see -0.06 of it; Dort -1.25 against -0.05;
Durant -1.15 against -0.10.  The best cases are Trey Murphy III at +1.04 against +0.47 and Markkanen at
+1.16 against +0.44.  **Most of what an absence reveals is not in the box score at all.**

So alpha is a good instrument and a poor teacher.  Its value from here is diagnostic: it is the only
thing in this project that measures a per-player error with bench players counted the same as starters.

### Traps, all of them paid for

* **The rating must enter as two free unpenalised columns, one per side** (`--free_scale=1`, the
  default).  Without them the amplitude error becomes a negative alpha for every strong player and a
  positive one for every weak one -- a rescale wearing a per-player costume.  With them the correlation
  between alpha and the rating is -0.001 on both sides.
* **Always pass `--team_effects=team`.**  `none` is the default only because it is the plain form of the
  design.  `outputs/tradeset_*` with no suffix is the uncontrolled arm and must not be quoted;
  `tradeset_team_*` is the one to read, `tradeset_ts_*` the strict bound.
* **The corrected rating is `scale x rating + alpha`, never `rating + alpha`.**  The first scored arm
  omitted the multiplier and was reading half a fit as the whole.
* **The penalty grid must be scored two seasons out** (`--offset=2`).  At the default offset the scored
  games are inside the window alpha was fitted on.
* **Career possessions ranks near the top of every feature table and is measurement trap 8**, not a
  finding: a longer career means a better-measured correction, so the feature predicts alpha's noise.
  The percentages above already exclude that family.
* **A first-year player's whole contrast comes from a season he was not in the league**, so his team's
  previous year does all the work.  That is why Dylan Harper lands 20th in the corrected 2026 list.
  Defensible, but not the same kind of evidence a veteran's missed games give.
* Two bugs were caught before any number was quoted and both are in `DECISIONS.md` 18 and 18c: the
  missing amplitude multiplier above, and a melt that repeated each player's correction once per
  statistic and inflated it 42-fold.

### If this line is continued

In the order I would try them:

1. **Use the trade loss to re-judge the ranking candidates already on disk.**  Dozens of them exist and
   the team-game test called many of them ties because it cannot see the bench.  The trade loss counts
   every player once.  No new modelling, about 20 minutes.
2. **A standard error per player**, from the inverse of the regression's own matrix.  Then read only
   corrections larger than their own error, and weight the feature fit by real precision.  Pascal
   Siakam's +1.71 rests on 448 possessions of absence, and nothing currently marks that.
3. **One column per player per season, penalised on the differences between them.**  Alpha is currently
   a three-season average deviation from a one-season rating, which mixes "the rating missed something"
   with "he was different the next year".
4. The per-component luck-adjusted target (`teamloo`) as the game margin, and the plain played-or-not
   entry, both one run each.

Not worth trying: adopting alpha as a rating (illegal under ruling 1), or training the prior on it
without first raising the 3.7% and 5.6%.

## Traps that cost a day

1. `pip install -e` puts this working tree on `sys.path`: a fresh clone silently tests THIS repo's data.
2. Tests cannot reach the network (`tests/conftest.py`).
3. A score on a mask that cuts team-games is not a score; read stint-cutting splits on `mse`.
4. Two builds at once, or an unpinned numba pool, and a 15-second fit takes an hour.
5. `TaskStop` on a chain stops the bash wrapper only; its python children keep running and its `for`
   loop keeps launching.  Check `Get-Process python` before starting anything.
6. Seventeen more in `DECISIONS.md`, "The measurement traps".

## Filed for the next rebuild (2026-09-18)

**`poss_def` in the shipped parquet is still a copy of `poss_off`.** The board script was fixed on
2026-09-18 — `side_possessions()` had always computed both counts correctly but only ran when a
`--blend` flag asked for it, and it now runs every season — but
`outputs/season_ratings_product.parquet` was built before that and carries the old column in all
14,579 rows. **No rating is affected**: `poss_def` is assigned after the ridge and feeds nothing.
Rebuild whenever the next real run happens; there is no reason to spend 90 minutes on it alone.

What it is worth, measured on 2026: 551 of 582 players move, mean 8.2 possessions — about nine in
twenty-five hundred, as expected. But the per-player ratio runs 0.87 to 1.25, so at the bottom of the
rankings it is not a rounding matter, and the bottom is where the exposure work lives. Anything that
reads a per-side count at low exposure (`scripts/68_exposure_slope.py`, `--match_spread`) wants the
rebuilt table. That script detects the stale column and says so.

## What the 2026-09-17/18 cleanup changed (all in the git log; six commits)

Nothing that moves a rating. The four that would have cost a day each:

1. **Nine consensus checks never ran in CI.** A module-scoped fixture in `tests/test_vs_consensus.py`
   computed `bigness` eagerly, which skips without `data/raw` — and CI asserts `data/` is empty. Every
   PR check reported green on tests that did not execute. The network-block guard was never collected
   at all, because it lived in `conftest.py`.
2. **A misspelled flag was ignored.** Nineteen copies of the same `_flag()`, none of which looked at
   the flags it was *not* given. `--exclude_neighbors=1` ran without the exclusion and produced a
   number that looked like one that had not. `scripts/_cli.py` is the one copy now and refuses an
   unknown name; `tests/test_config.py` checks every flag this file tells you to type.
3. **`scripts/52_site.py` republished the superseded rankings** if you ran the README's build step,
   and printed a filename identical for both candidates, so the swap was invisible.
4. **Five scripts fitted from the season panel with no `drop_untrainable` gate** (50, 62, 67, 71, 72).
   Latent — nothing is in progress — but it is the shape of thing noticed only in the year it starts
   lying. `tests/test_no_current_season.py` is the guard `seasons.py` had claimed for months.

`pytest -q` is **330 passed, 1 xfailed** with the scraped data, **305 passed / 10 skipped / 1 xfailed**
on a clone without it.

## Verify you are where this file says

    .venv/Scripts/python -m pytest tests -q                                  # 330 passed, 1 xfailed, ~165 s
    .venv/Scripts/python scripts/63_yoy.py --rankings=incumbent=outputs/season_ratings_sy_lam13037.parquet,ship=artifacts/season_ratings.parquet --ref=incumbent --tag=verify --splits=
                                                                             # incumbent 8.697; ship -4.0 team-game MSE
    .venv/Scripts/python scripts/64_consensus_report.py outputs/season_ratings_sy_lam13037.parquet
                                                                             # 0.777 / 0.757 / 0.771, top5 4
