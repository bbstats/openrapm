# Handoff

**What this file is.**  Where the work stands, what waits on the owner, and the commands a session runs.  Rewritten
whole on 2026-10-16; the earlier, longer file is in git history at feeb8e6.  `DECISIONS.md` is the permanent record and
carries every number quoted here; read its measurement traps before trusting a number.  `README.md` says what
OpenRAPM is, `PIPELINE.md` how each step works, `CONTRIBUTING.md` the writing rules.

## Where it stands, 2026-10-16

- **Live on the site:** experiment 65, published 2026-10-15, commits 9d98448 and feeb8e6.
- **Working tree:** the overnight work of 2026-10-15/16 sits on `main` uncommitted: code, tests, the panel's new
  columns, params files and notes.  `pytest -q` gives 662 passed and 1 failed, the Curry star-guard ceiling;
  `pytest -m slow` 7 passed; `ruff check .` clean.
- **The best candidate:** experiment 71, plain leaves plus the per-game hustle inputs.  Built and scored, not adopted.
- **The owner's direction for what comes next:** "makes sense so far" to a full hyperparameter tune on the decider, and
  asked about tracking.  Neither is started.

## What ships

| step | module | as shipped |
|---|---|---|
| 1, the label | `rapm.py` | one RAPM per player over every season but the rated one; penalties 40,000 / 40,000 / 0; the defensive label un-shrunk |
| 2, the box-score prior | `box_prior.py` | LightGBM, `params/booster_params_lgb1_cols06.json`: straight-line leaves, each tree shown 60% of its inputs; feature set `no_career_counts`; five player folds, out of player |
| 3, the rating | `prior_rapm.py` | the season's RAPM pulled toward the prior, residual penalty 13,037, centred |
| 4, the shrink | `prior_shrink.py` | both parts multiplied by what holds up on held-back games, from the 8 nearest fold seasons: prior part 0.662-0.751 offence, 0.779-0.953 defence; games' part 1.269-1.375 and 1.397-1.567; folds `cols06_early,cols06_within` |
| 5, the swap step | `swap_adjust.py` | credit re-split inside each team from lineup swaps, team totals fixed, spread inside teams held to step 4's |

- `openrapm build` writes `outputs/season_ratings_product.parquet` with `_step3` and `_step4` beside it.  Previous
  product tables: `outputs/season_ratings_product*_before_gamesfree.parquet`, `*_before_cols06.parquet`.
- `openrapm site --bias --openrapm-decomposition --similarity` writes the site's files.  Commit, fast-forward `main`
  and push only on the owner's word, then check the Pages build:
  `gh api repos/bbstats/openrapm/pages/builds/latest`.
- The incumbent for scoring is `gamesfree`, the test-setting tables `outputs/season_ratings_gamesfree*.parquet`.  Its
  folds are `cols06`'s.

## How an experiment is judged

The owner, 2026-10-15: "within-season is truly the best metric, particularly WOWY".  Three readings, nothing else:

| reading | script | rule |
|---|---|---|
| the decider | `scripts/checks/151_wowy.py`, stint unit, step 4 as shipped | paired z of -2 or below against the incumbent; the noise control beside it |
| the guard | `scripts/checks/63_yoy.py` | year over year not worse at z +2 |
| the eye check | `scripts/checks/66_compare.py` | the 2026 top 20 |

- One build: `bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes "EXTRA RATE FLAGS"`.  About
  50 minutes: steps 1-3 at the test setting, the candidate's own folds for all thirty seasons, steps 4 and 5, then the
  three readings.  `INC_NAME=gamesfree` names the incumbent's tables and `INC_SYSTEM=cols06` its folds.
- The noise control on the incumbent, `noise_cols06`, reads +0.077, z +1.05, 11 of 30 seasons.  On experiment 71's
  base, `noise_flathustle` reads +0.043, z +0.71.
- **Scale.**  On the decider, ratings of zero leave 3,648.6 a season and today's ratings 3,633.0: the ratings explain
  15.7, so a gain of 0.27 is 1.7% more of what ratings explain.  `scratch/2026-10-15_overnight/zero_baseline.py`.
- Every experiment gets a row in `experiments/runs.csv` through `scripts/checks/101_log_run.py`, mirrored to the
  sheet "OpenRAPM experiments", tab Experiments.
- Reports to the owner: four parts, what changed / result / 2026 top 20 / your call; one quantity per column, no
  parentheses; plain words; one question at the end and a one-sentence TL;DR.

## Waiting on the owner

1. **Adopt experiment 71?**  Plain leaves plus per-game hustle, `--booster-params lgb1_cols06_flat --features hustle`.
   Decider -0.272, z -4.87, 23 of 30 seasons; guard +0.012, z +0.30; stint level year over year -0.133, z -2.89.
   Cost in the eye check, product setting: Jordan 1998 3rd to 5th, Dončić 2026 3rd to 9th, Giannis 9th to 18th.  A
   product-setting preview sits beside the live table: `outputs/season_ratings_product_flathustle.parquet`.
2. **The full hyperparameter tune, on the decider.**  Today's booster settings came from a 2026-10-08 tune on the
   prior's error against career labels, on 2015 and 2026 only.  Never tuned on the decider.  The design proposed:
   - Optuna, about 60 trials; linear leaves on or off, tied across the two sides by the same-reason rule; leaves,
     depth, rows per leaf, learning rate and trees, both penalties, column share, bins, smoothing, linear-leaf penalty.
   - The hustle inputs on, the era marker as a switch.
   - Each fold's rebuilt inputs cached once, since they do not depend on the settings; a trial retrains the boosters
     and solves the folds again, about 7 minutes on ten seasons.
   - Tune on 1998, 2001, ... 2025; keep the other twenty seasons untouched; confirm the winner through the full chain
     with its own noise control.
   - About 3 hours of code, then about 7 hours of trials.
3. **Tracking, game by game.**  One-day pulls of the tracking dashboards return one game per player: probed
   2026-10-16, drives on 2024-01-15 gave 229 players with one game each.  So all 21 tracking columns can be rebuilt
   per game.  Everything: about 29,000 requests, 13 hours.  Defence first, defended twos, threes and rim shots plus
   contested rebounds: about 11,700 requests, 5 hours.  Experiment 62 found the tracking signal on defence.
4. Verdicts still pending in the experiment table: 64, 65b, 66, 67-72, 74; recommended not adopted except 71.
5. The decider's bar, z -2, to confirm.
6. The Curry star-guard ceiling test fails: Curry 83rd against a ceiling of 80.

**Filed for later by the owner.**
- Arena-season scorekeeper checks on steals, blocks and assists.  Assists measured, DECISIONS.md "Home and away
  assists": arena generosity had a true spread of about 7% before 2018, gone since the league's stats auditor arrived
  in 2017-18; Utah was stingy; Stockton's home edge was below the era's median.
- Stockton above Jordan in 1997 and 1998.  The offensive prior overshoots at its top: the top 25 offensive box-score
  parts carry a mean residual of -0.75 per 100.  Nothing tried put Jordan above Stockton in 1997: input clip, plain
  leaves, age, the points split.

## The overnight run, 2026-10-15/16

The owner: "do as many tests as you want".  Against the incumbent; DECISIONS.md has a section for each.

| exp | name | change | decider ΔMSE | decider z | guard z | reading |
|---|---|---|---|---|---|---|
| 64 | `vsleague2` | experiment 56's ratios against the league | -0.049 | -0.91 | +0.08 | tie |
| 65b | `sourcesplit` | step 4 by source, built whole | -0.088 on top of 65 | -2.59 | +3.50 | fails the guard |
| 66 | `hustlegames` | per-game hustle inputs | -0.196 | -2.96 | -0.03 | passes; half of it an era split |
| 66b | `hustleplacebo` | hustle shuffled among players | -0.103 | -2.03 | +0.16 | the control |
| 67 | `clip` | inputs clipped at 0.5% and 99.5% | +0.045 | +2.03 | +0.79 | worse |
| 68 | `flat` | plain leaves | -0.102 | -3.19 | +0.42 | passes; 1997-2006 -0.197, z -4.3 |
| 69 | `plusage` | age on both priors | +0.713 | +7.46 | -5.22 | the career-count pattern |
| 70 | `cols04` | each tree 40% of inputs | -0.013 | -0.47 | -0.43 | tie |
| 71 | `flathustle` | plain leaves + hustle | **-0.272** | **-4.87** | +0.30 | the candidate |
| 72 | `pointsparts` | assisted and unassisted points | +0.022 | +0.67 | -0.31 | tie |
| 73 | `flatera` | plain leaves + era marker 2017 | -0.156 | -3.92 | +0.33 | explains 66b |
| 74 | `flathustletrack` | 71 + five per-game tracking columns | -0.291 | -4.75 | -0.28 | tie with 71 |

- The leak control: folds whose prior read whole-season inputs scored -2.84, z -31.  A season-total input cannot be
  judged on the decider.  `openrapm folds --whole-season-inputs`; never for step 4.
- Hustle is real information: -0.31 in 2017-2026 beyond an era split, z -3.2.  Columns missing before 2017 let the
  trees split the eras, which helps the early seasons; a plain marker does the same.

## Data and code added since feeb8e6, uncommitted

- `openrapm data hustle-games` and `openrapm data tracking-games`: per-game box scores, `BoxScoreHustleV2` 2017-2026,
  12,812 games; `BoxScorePlayerTrackV3` 2014-2026, 16,758 games; none empty; under
  `data/raw/dashboards/<source>/games/<season>/`.  The endpoint hangs every few minutes, so both ran under watchdogs
  that restart the download every 10 minutes and resume: `scratch/2026-10-15_hustle/*_watchdog.ps1`.
- The season panel, 132 columns: the nine hustle counts summed over the games a rating is built from, 2016 left
  missing; the five tracking columns the per-game box score holds, summed the same way; the hustle placebo columns;
  the era marker `season_from_2017`.  Every older column unchanged.  Backups `outputs/season_panel_before_*.parquet`.
- A leak check per fold: the per-game counts read only fit games.
- `box_prior.INPUT_CLIP`, `ClippedBooster`; params files `lgb1_cols06_flat`, `lgb1_cols06_clip`, `lgb1_cols04`.
- Feature sets `no_career_ratios_vs_league`, `hustle_placebo`, `no_career_plus_age`, `no_career_points_parts`,
  `no_career_era_marker`, `hustle_and_tracking_games`.
- Check 151 `--pieces`; check 145 `family_ratings`.
- Read-only scripts: `scratch/2026-10-15_assists/`, `scratch/2026-10-15_overnight/`.

## Commands a session runs

    bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes "--features hustle --booster-params lgb1_cols06_flat"
        one experiment, about 50 minutes; log outputs/NAME_chain.log
    .venv/Scripts/python scripts/checks/151_wowy.py --systems=cols06,noise_cols06,NAME --ref=cols06 --tag=NAME
        the decider alone on folds already built, about 3 minutes; --pieces=source,certainty adds step 4 in pieces
    .venv/Scripts/python scripts/checks/63_yoy.py --rankings=NAME=outputs/season_ratings_NAME.parquet,incumbent=outputs/season_ratings_gamesfree.parquet --ref=incumbent --tag=NAME --splits=
        the guard alone
    .venv/Scripts/python scripts/checks/66_compare.py incumbent=outputs/season_ratings_gamesfree.parquet NAME=outputs/season_ratings_NAME.parquet --season=2026 --top=20 --ref=incumbent --open=0
        the 2026 top 20
    .venv/Scripts/python scripts/checks/101_log_run.py --name=NAME --exp=75 --incumbent=gamesfree --idea=owner --verdict=pending --changed="one plain sentence" --notes="the readings"
        the experiment table's row; --sheet prints the rows for the Google Sheet
    .venv/Scripts/python scripts/checks/145_piece_multipliers.py --folds=TAG --first=1997 --last=2026
        how much each piece of a rating holds up, on folds built with openrapm folds --pieces
    .venv/Scripts/python -m openrapm build --name=product_NAME --booster-params=lgb1_cols06_flat --features=hustle --folds=NAME_early,NAME_within
        a product-setting preview written beside the live table, about 12 minutes
    .venv/Scripts/python -m openrapm data hustle-games 2017 2026
        the per-game hustle box scores; tracking-games for the tracking ones; run long ones under the watchdogs

## The rulings that bind

1. One rating per player per season, from that season's games only, regular season and playoffs.
2. Single year or bust: a player's own other seasons may not reach his rating, not even through the booster's memory.
   Model coefficients may be learned from other seasons; this-season facts such as age, height and weight are allowed.
3. Never train on the current season until its Finals are over.  Loading is always allowed.
4. The prior is trained on one row per player, his career with the rated season out, plus chunk rows of his seasons.
5. The consensus is a sanity check, never a fitting target.
6. Each season is centred at possession-weighted zero per side.

## How the work is done

- Two jobs: implement the owner's ideas, or propose new ones.  One experiment at a time unless the owner says
  otherwise, as for the overnight run.
- One build at a time on the owner's machine.  When a chain is stopped, stop its python children too: `TaskStop` stops
  only the wrapper.  Check `Get-Process python` before starting anything.
- Never commit, push or publish without the owner's word.  The site is `docs/`, served from `main`.
- Movement is reported in points per 100, never rank places; under 0.1 is nothing.
- On the site the ratings are "OpenRAPM"; no drafted prose on any page.

## Traps that cost a day

1. `defense` in the ratings tables is points allowed, negative good; `rating_def` is positive good.
2. Stint level and team-game level can disagree; say which a number is.
3. A season-total input leaks into the within-season folds: whole-season inputs read z -31 on the decider.
4. A long bash heredoc fails in this shell: write the script to a file first.
5. The stats.nba.com endpoints hang without timing out: run long downloads under a watchdog that restarts them.
6. A background job is stopped after 2 hours: split long downloads into stretches that end on their own.
7. Thirty-seven more in DECISIONS.md, "The measurement traps", and the memory "era-coefs measurement traps".

## Verify you are where this file says

    .venv/Scripts/python -m pytest -q                       # 662 passed, 1 failed: the Curry star-guard ceiling
    .venv/Scripts/python -m pytest -q -m slow               # 7 passed
    .venv/Scripts/python scripts/checks/63_yoy.py --rankings=incumbent=outputs/season_ratings_gamesfree.parquet --tag=verify --splits=
                                                            # game_armse 8.5958, both directions together
    .venv/Scripts/python -m openrapm site --out-dir <scratch>/docs --built 2026-10-15
    cmp docs/data/ratings.json <scratch>/docs/data/ratings.json       # no output: byte-identical
