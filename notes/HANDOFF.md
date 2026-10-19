# Handoff

**What this file is.**  Where the work stands, what waits on the owner, and the commands a session runs.  Rewritten
whole on 2026-10-16; the earlier, longer file is in git history at feeb8e6.  `DECISIONS.md` is the permanent record and
carries every number quoted here; read its measurement traps before trusting a number.  `README.md` says what
OpenRAPM is, `PIPELINE.md` how each step works, `CONTRIBUTING.md` the writing rules.

## Where it stands, 2026-10-17

- **Live on the site:** experiment 78, published 2026-10-17: experiment 77 (experiment 76's booster settings and
  defence factor, three-point % off the offensive booster) with the prior averaged over 10 bootstraps of the players.
  Decider -0.086, z -3.13, the twenty untouched seasons z -2.40; guard z +0.60.  A product build now takes longer: a
  season's rating about 96 s against 24.
- **Tests:** `pytest -q` 663 passed, the Curry ceiling passing again, Curry 48th; `pytest -m slow` 7 passed; `ruff
  check .` clean.  Two consensus guards were loosened for experiment 76, the bigs rated above the consensus:
  bigness 0.30 to 0.40, archetype spread 0.25 to 0.30.
- **The input cull, done 2026-10-17:** check 152 `cull` / `cull-table`, every input shadowed and dropped at three
  seeds on the tune cache (DECISIONS.md "The input cull on the decider").  One cut, three-point % on offence, became
  experiment 77.  The cull also found the booster's seed alone moves the decider about 0.03; bagging, experiment
  78, replaced the single booster.  Tracking is on hold ("not yet").
- **Next, the owner's open threads:** tracking game by game (below); the arena scorekeeper checks; the big-man lean against the consensus.
- **Done 2026-10-18: check 153, what step 5 does** (`notes/PLAN_swap_step.md`; DECISIONS.md "Check 153"; sheet tabs
  "Swap step" and "Swap step types").  Step 5 helps on the decider, -0.361, z -6.0, and ties the guard.  Its gain is
  type content, not the within-team shrink: the shrink alone is +1.25 on the decider.  On offence its biggest terms,
  made and missed shots, are same-games luck.  Refitted on the folds' held-back games, `step5_other`, it beats shipped
  step 5 on the decider by 0.547, z -12.2, 30 of 30, and on the guard by 0.163, z -3.2.
- **The owner's ruling, 2026-10-19:** "fix step 5 but - yes it must ONLY come from the season in question.  since this
  is post hoc we should try it on the prior first.  so - commit everything, try fixing the prior, then work on fixes to
  step 5".  Order: (1) commit; (2) the same-games fix on the prior: each training row's label from games its box score
  did not come from, experiment 25's idea, judged on the decider; (3) then step 5, cross-fitted, its inputs from the
  rated season only, so `seasons_before` goes.

## What ships

| step | module | as shipped |
|---|---|---|
| 1, the label | `rapm.py` | one RAPM per player over every season but the rated one; penalties 40,000 / 40,000 / 0; the defensive label un-shrunk |
| 2, the box-score prior | `box_prior.py` | LightGBM, `params/booster_params_lgb1_tune2.json`, chosen on the decider by check 152's guarded search; feature set `hustle_no_three_pct`; five player folds, out of player; 10 bags, each on a bootstrap of the players |
| 3, the rating | `prior_rapm.py` | the season's RAPM pulled toward the prior, residual penalty 13,037, centred |
| 4, the shrink | `prior_shrink.py` | both parts multiplied by what holds up on held-back games, from the 8 nearest fold seasons, both defensive factors times 0.866: prior part 0.659-0.728 offence, 0.684-0.794 defence; games' part 1.186-1.393 and 1.252-1.351; folds `bags10_early,bags10_within` |
| 5, the swap step | `swap_adjust.py` | credit re-split inside each team from lineup swaps, team totals fixed, spread inside teams held to step 4's |

- `openrapm build` writes `outputs/season_ratings_product.parquet` with `_step3` and `_step4` beside it.  Previous
  product tables: `outputs/season_ratings_product*_before_bags10.parquet`, `*_before_nothreepct.parquet`.
- `openrapm site --bias --openrapm-decomposition --similarity` writes the site's files.  Commit, fast-forward `main`
  and push only on the owner's word, then check the Pages build:
  `gh api repos/bbstats/openrapm/pages/builds/latest`.
- The incumbent for scoring is `bags10`, the test-setting tables `outputs/season_ratings_bags10*.parquet`, its
  folds `bags10_early` and `bags10_within`; its noise control `noise_bags10`.  The decider applies the shipped defence factor, 0.866, to every system; a system
  built before experiment 76 is read with `--defence-factors=NAME:1`.

## How an experiment is judged

The owner, 2026-10-15: "within-season is truly the best metric, particularly WOWY".  Three readings, nothing else:

| reading | script | rule |
|---|---|---|
| the decider | `scripts/checks/151_wowy.py`, stint unit, step 4 as shipped | paired z of -2 or below against the incumbent; the noise control beside it |
| the guard | `scripts/checks/63_yoy.py` | year over year not worse at z +2 |
| the eye check | `scripts/checks/66_compare.py` | the 2026 top 20 |

- One build: `bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes "EXTRA RATE FLAGS"`.  About
  1 hour 40 minutes with 10 bags: steps 1-3 at the test setting, the candidate's own folds for all thirty seasons, steps 4 and 5, then the
  three readings.  `INC_NAME=tune2` names the incumbent's tables and `INC_SYSTEM=tune2` its folds.
- The noise control on the incumbent, `noise_bags10`, reads -0.003, se 0.011, z -0.33, 14 of 30 seasons; bagging cut
  the regrouping noise to about a third of experiment 76's, se 0.029.
- **Scale.**  On the decider, ratings of zero leave 3,648.6 a season and today's ratings 3,633.0: the ratings explain
  15.7, so a gain of 0.27 is 1.7% more of what ratings explain.  `scratch/2026-10-15_overnight/zero_baseline.py`.
- Every experiment gets a row in `experiments/runs.csv` through `scripts/checks/101_log_run.py`, mirrored to the
  sheet "OpenRAPM experiments", tab Experiments.
- Reports to the owner: four parts, what changed / result / 2026 top 20 / your call; one quantity per column, no
  parentheses; plain words; one question at the end and a one-sentence TL;DR.

## Waiting on the owner

- **Done 2026-10-16:** experiment 76 adopted and published (the owner: "Hell yeah. Publish").  It supersedes
  experiment 71 and the tune proposal; experiment 75 is not adopted.  Rerun a trial on the tune cache:
  `python scripts/checks/152_decider_tune.py trial --tag tune1 --params NAME --features hustle --guard`, about 45 s.
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
    .venv/Scripts/python scripts/checks/153_swap_audit.py folds --system=bags10 --first=1997 --last=2026
        check 153: every version of step 5 on every fold, cached per season in outputs/swap_audit/bags10/, about 10
        minutes; then `decider --systems=bags10,noise_bags10`, `whole --system=bags10`, `readings --tag=swap_audit`;
        the full order is scratch/2026-10-18_swap_audit/chain2.sh

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
