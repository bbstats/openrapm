# Within-season ledger: what each rating sees, and what it never sees

> **What this file is.** The ledger of the held-back games, the code's within-season folds: for every input a
> part-season rating reads, what it is built from, whether it is rebuilt from the fit games alone, and the check that
> proves it. The folds are built by `openrapm folds` (`src/openrapm/commands/within_season.py`, the checks in
> `leak_checks`, `design_checks` and `SeasonRun`), and Box Adj's line and the shrink are fitted on them. The table
> below is today's, as of 2026-11-16; the sections under "History" are the runs it was first written for, and their
> numbered-script paths refer to the code at the tag `archive/research-2026-10`.
>
> | in the History sections | today |
> |---|---|
> | `scripts/01_ingest.py`, `02_stints.py` | `openrapm data ingest`, `openrapm data stints` |
> | `scripts/49_role_panel.py` (with 65, 69, 86) | `openrapm data panel` |
> | `scripts/62_single_year_board.py` | `openrapm rate` (steps 1-3); `openrapm build` runs them all |
> | `scripts/97_within_season.py` | `openrapm folds` |
> | `scripts/98_calibrator.py` | retired; its fold reader is `src/openrapm/data/within_folds.py` |
> | `scripts/99_prior_shrink.py` | `openrapm shrink` |
> | `scripts/91_swap_adjust.py` | `openrapm swap` |
> | `scripts/52_site.py`, `76_bias_groups.py`, `83_decompose_site.py` | `openrapm site`; the pages of 76 and 83 are retired |
> | `scripts/63_yoy.py` and the other twelve checks | `scripts/checks/`, same number and name |
> | 62 `--exclude_neighbours=1` | `--setting test` |
> | 62 `--features=boruta_noonc` | `--features box_score_and_bio` |
> | targets `xpts_ft`, `x3def_w0.25` | `points_with_expected_free_throws`, `points_with_expected_opponent_threes` |
> | every other numbered script | retired |

The owner, 2026-10-03: "the leaving stuff out approach here is getting quite complicated but it's okay.  Just keep
really special good tabs on it."  This file is those tabs.  Every check named below runs on every fold and is written to
`outputs/within/<tag>/checks.parquet`, and a failed one stops the run.

## The set-up, in plain terms

- **Rated season**: one season, 1997 to 2026, regular season and playoffs together.
- **Fit games**: three quarters of the rated season's games, picked at random.  The rating is built from these alone.
- **Held-back games**: the other quarter.  Nothing about them reaches the rating.  The rating's misses are measured on
  them.
- **Fold**: one way of picking the held-back quarter.  Each season's games are assigned at random to four folds, so
  every game is held back exactly once per assignment.  **Repeat**: a fresh random assignment of the same season, three
  by default, seeded so a rerun assigns the same games.  `--folds-per-repeat 10` holds back a tenth instead.
- **The rating** here is the shipped steps 1-3 with Box Adj, on the saved boosters of the build named by `--models`:
  the box-score prior, Box Adj's line, the ridge, the centering.  The shrink is fitted afterwards on the folds' misses;
  the swap adjustment, off as shipped, is not applied.

What it is for: misses on the season's own games, so Box Adj and the shrink cannot learn "a peak season comes back
down next year" and shrink peak seasons.

## Every input, what it is built from, and the check

| input | read by | built from | rebuilt from the fit games only? | check, its name in `checks.parquet` |
|---|---|---|---|---|
| the possessions the ridge fits: who was on the court, points per 100 | the rating | the season's stints | **yes**: only the fit games' stints | `design <name>: rows only from fit games`, `design <name>: same rows as the points design` |
| box-score rates, the 13 counts per 100 and the ratios built from them, their padding and centering, each player's possessions | the box-score prior, Box Adj | the season's per-game box-score and possession tables | **yes**: both tables are cut to the fit games | `design <name>: box and possession tables only from fit games` |
| the defensive target's three-point repricing, each opposing shooter's three-point percentage | the rating, defense | the season's shot table | **yes**: the shooters' percentages come from the fit games only | `design points_with_expected_opponent_threes*: response against the whole-season design (max change)`: 0 for the whole season, above 0 for a part season |
| the free-throw adjustment in both targets, each shooter's season free-throw percentage leaving out the game itself | the rating | stored in the stint files when they were built | **no: the one accepted leak.** The held-back games' free throws move a shooter's season percentage by a quarter of their luck; through his rating that is about 0.03 points per 100 for the heaviest free-throw shooters and near zero for everyone else, under the 0.1 that counts | none possible without rebuilding the stints per fold |
| share of team possessions played, share of games started | the box-score prior, Box Adj | box scores, starters and games played, and stints, possessions | **yes**: rebuilt game by game, the team's possessions cut with the player's | `roles: box scores read only from fit games`, `roles: stints read only from fit games` |
| score state: share of his possessions in garbage time | Box Adj | the stints | **yes** | `context: stints read only from fit games` |
| the play-by-play parts: rebounds by the miss, steals by the turnover, points and threes by the assist and the spot, blocks, assists by zone | the season panel; no shipped input | the play-by-play counts per game, `openrapm data events` | **yes**: only the rows of the cut box-score table are read | `play-by-play parts: counts read only from fit games` |
| self-creation's counts: every shot's spot and situation | no shipped input; the shipped self-creation is built from the box rates | the self-creation counts per game, `openrapm data events` | **yes** | `self-creation: counts read only from fit games` |
| fouls, goaltends, shots blocked, three-shot trips | no shipped input; experiment 119 tried them | the foul counts per game | **yes** | `fouls and blocked shots: counts read only from fit games` |
| the hustle rates: screen assists, box-outs, loose balls, deflections, contested shots, charges | the box-score prior | the hustle box scores, one per game, from 2017, `openrapm data hustle-games` | **yes**: summed over the fit games alone | `hustle: per-game counts read only from fit games` |
| who he guarded and who guarded him | no shipped input; experiment 120 tried them | the per-game matchup box scores, `openrapm data matchup-games`, 2018 on | **yes** | `matchups: per-game pairs read only from fit games` |
| Box Adj's context numbers: teammates' and opponents' box-score priors with him on and off the court; his team's shots by zone, free throws, rebounds and turnovers on and off | Box Adj | the fit games' stint rows, every prior from the boosters that never saw him | **yes** | `second prior: stint rows only from fit games` |
| Box Adj's line itself, its weights | Box Adj | the sums `openrapm second-prior` wrote from an earlier build's held-back games, every finished season but the rated one and, at the test setting, its neighbors | the rated season's own held-back games never reach it | built into the line's fit, `second_prior.season_coefficients` |
| height, weight | the box-score prior, Box Adj | the biography table | not game data | -- |
| age | Box Adj | the roles table, from the birth date | not game data | -- |
| the boosters themselves and their labels | the box-score prior | every season except the rated one and the one either side of it, `--setting test` | never trained on the rated season at all | `prior models: never trained on this season or its neighbours` |
| the prior's free scale | the rating | fitted unpenalized with the ridge on the fit games | **yes**: from the fit games only.  Its inputs, the priors, also saw every fit game, so the scale is priced in sample (DECISIONS.md, "The Robustness pass").  Nothing from the held-back games reaches it | as above |
| the centering: the average possession is played by a zero player | the rating | the fit games' possessions | **yes** | `ridge: possessions are the fit games' possessions` |
| the season's game list: ids, dates, regular season or playoffs | bookkeeping | the schedule | no outcomes in it | -- |

**Held-back side.**  The held-back games are scored on points as actually scored, with no luck adjustment, so no
shooting percentage is estimated on that side at all, and they are scored the way the year-over-year test scores a
season: predict every possession from the ten players' ratings, refit only the league level and the home edge on the
held-back games, and take the error per team-game.  A player with no possessions in the fit games has no rating, and is
scored at the same replacement level the year-over-year test uses for an unrated player.  Checks: `test: rows only from
test games`, `test: no test game among the fit games`, `test: shares sum to five`, `folds: fit and test games
disjoint`, `folds: fit and test games cover the season`.

**Before any fold**, every season is run through the same code with all its games as the fit games, and the result
must equal the base table the boosters were saved with; `openrapm folds --reproduce-only` runs only this.  A
part-season rating is then the shipped rating code on fewer games, and nothing else.

## History

The sections below are the runs of 2026-10-03 to 2026-10-05 that this ledger was first written for, when the folds
covered 2017-2026, fed a calibrator that was not adopted, and the swap adjustment was still applied.  They are kept as
they were.

### The reproduction check

Before any fold, every season is run through the same code with ALL its games as the rating games, and the result
must equal the rankings table built by `scripts/62_single_year_board.py` in the same run that saved the prior models,
to 1e-6 points per 100.  The rebuilt inputs are also compared, column by column, with the season panel the
rankings read, and the rebuilt role table with the cached one.  A part-season rating is then the shipped rating code
on fewer games, and nothing else.  Results: `outputs/within/<tag>/reproduce_<season>.json`.

Run of 2026-10-03 (tag `within`).  First, the rankings script rebuilt with `--save_models` reproduces the incumbent's
own steps 1-3 table (`season_ratings_unshrinkdef.parquet`) exactly, difference 0.0 on every player of 2017-2026.

| season | rating difference (offence / defence) | prior difference (offence / defence) | largest input difference | status |
|---|---|---|---|---|
| 2017 | 0 / 0 | 0 / 0 | 3e-13 | passed |
| 2018 | 0 / 0 | 0 / 0 | 2e-11 | passed |
| 2019 | 0 / 0 | 0 / 0 | 2e-11 | passed |
| 2020 | 0 / 0 | 0 / 0 | 1e-11 | passed |
| 2021 | 0 / 0 | 0 / 0 | 5e-13 | passed |
| 2022 | 0 / 0 | 0 / 0 | 9e-12 | passed |
| 2023 | 0 / 0 | 0 / 0 | 8e-12 | passed |
| 2024 | 0 / 0 | 0 / 0 | 1e-11 | passed |
| 2025 | 0 / 0 | 0 / 0 | 2e-11 | passed |
| 2026 | 0 / 0 | 0 / 0 | 4e-12 | passed |

The largest input difference is always the season's plain adjusted plus-minus, which the prior does not read; every
input the prior reads is identical to the last digit.  The cached role table (games, starts, minutes, possessions,
team possessions, age) is rebuilt with difference 0.

**Folds: 120 (10 seasons x 3 repeats x 4 folds), 2,640 pass/fail checks, 0 failed.**  Evidence the part-season
rebuild bites: the defensive target's response on the rating games moves by up to 5.6 to 15.5 points per 100 on a
possession when the shooters' three-point percentages come from the rating games alone; possessions come out at
three quarters of the season's; on-court numbers move by a median 0.8 points per 100.

### Run of 2026-10-04: all thirty seasons, five sizes

The owner: split tests "at different sizes", and "think about ways to make it bigger".  Box-score models for
1997-2016 saved the same way (`62 --save_models=within_e1` for 1997-2006, `within_e2` for 2007-2016, both at
`--exclude_neighbours=1`); their tables equal the incumbent's steps 1-3 exactly (largest difference 0.0).  Then every
season rated at five sizes, the rest of its games held out (`97 --rate_from=one` rates from ONE fold and tests on the
rest; the default rates from every fold but one):

| folder | share of games rated from | held out | repeats x folds | ratings | pass/fail checks | failed |
|---|---|---|---|---|---|---|
| `q1of4` | 1/4 (one fold of four) | 3/4 | 3 x 4 | 360 | 7,920 | 0 |
| `q1of3` | 1/3 (one fold of three) | 2/3 | 3 x 3 | 270 | 5,940 | 0 |
| `within2` | 1/2 | 1/2 | 5 x 2 | 300 | 6,600 | 0 |
| `q2of3` | 2/3 | 1/3 | 3 x 3 | 270 | 5,940 | 0 |
| `within` | 3/4 | 1/4 | 3 x 4 | 360 | 7,920 | 0 |

The whole-season reproduction check runs once per season (in `within`; the other folders reuse its result, since
they read the same models and the same base tables).  A folder holds one `--rate_from` mode; 97 refuses to mix them.
Two ratings of one repeat share no game only at 1/4, 1/3 and 1/2, so agreement is measured at those sizes only.
Results: DECISIONS.md, experiment 33c.

### The calibrator (scripts/98_calibrator.py): what IT sees

| what | rule | check |
|---|---|---|
| the seasons a season's calibrator is trained on | every season of 2017-2026 except the rated season and the one either side of it, so it never sees the games the year-over-year test scores that season's rating on | built into the season loop (`abs(s - season) > 1`) |
| the number of boosting rounds | chosen inside the training seasons only: half of them against the other half, both ways round | -- |
| its target | the training seasons' held-out games, points as scored, the league level and home edge refit per fold | -- |
| its inputs | only the columns rebuilt from the rating games (the table above) and how the rating was built from them | -- |
| the within-season test | the season's own part-season ratings, corrected, scored on that fold's held-out games by the year-over-year test's scoring | the uncorrected score must equal the one 97 stored for the fold (asserted to 1e-8) |
| the yardstick | one multiplier per side, fitted the same way on the same seasons; the trees start from it | -- |

**The common level, found and removed (2026-10-03).**  Adding the same number to every player's offence AND defence
changes no team-game prediction (the offence scores it, the defence gives it back), so the first run let that level
drift to +1.4 to +2.1 points per 100 on both sides, read off the few players who appear only in the held-out games
(they carry the stand-in rating for unrated players).  That is a statement about the stand-in, not about the rated
players, and it put every regular in the 2026 top 20 up by 3 to 4.7.  Since then the corrections are centred within
each fold and each season by the rankings' own possession weights, in training and in use.

### Open items

1. **The swap adjustment.**  The shipped rankings carry it (scripts/91, within-team lineup swaps); the ratings here
   do not.  A calibrator trained on steps 1-3 is applied to steps 1-3, and the swap adjustment would then run after
   it.  Rebuilding the swap adjustment from the rating games is possible and not done yet.
2. **Less evidence than a full season.**  A rating from three quarters of a season is noisier than a full-season
   one, so the misses measured here are somewhat larger than a full-season rating's.  The calibrator sees each
   player's possessions, which lets it scale its corrections with the evidence; whether the scaling carries to a
   whole season is a question to test, for example with half-season folds beside the quarter-season ones.
3. **Not rebuilt because nothing here reads them:** the RAPM pieces of the decomposition (experiment 26), the
   same-team measure.  If a calibrator wants them, they must be rebuilt from the rating games first.

### Settings of the run

- prior models: `scripts/62_single_year_board.py --exclude_neighbours=1 --score=0 --boards=2017,...,2026
  --out=season_ratings_within_base --save_models=within` (the incumbent's steps 1-3 settings; the build checks the
  saved settings and refuses others)
- folds: four per repeat, three repeats per season, seeded by season and repeat
- held-out scoring: points as scored, the year-over-year test's scoring and replacement level

### Experiment 35 (2026-10-05): the 9/10 size, the baselines, OpenRAPM as shipped per fold

`within10`: every season rated from 9/10 of its games, one repeat of ten folds (300 folds, 6,600 checks, 0 failed).

| rating | what it is built from | what it never sees | checks |
|---|---|---|---|
| B1, linear box-score SPM (`102`) | the rating games' LeagueGameLog box rows (13 counts, minutes; never PLUS_MINUS); league rates and box-estimated possessions from the same games; padding constants averaged over the training seasons; coefficients fitted on player-seasons outside {H-1, H, H+1} against the leave-season-out RAPM with those seasons left out | the held-out games; season H and its neighbours in the label and the training rows | `baseline_b1_checks.parquet`: box rows only from fit games, fit games found in the game log (> 99%), label exclusion set; 3,750 checks, 0 failed |
| B2 vanilla RAPM, B3 RAPM on B1 (`103`) | the rating games' stints (actual points, context unpenalised); B3's prior is the fold's own B1 (asserted) | penalties and multipliers are chosen on blocks touching no season in {H-1, H, H+1} | test games never among the fit games (asserted per fold) |
| OpenRAPM as shipped (`106`) | the saved steps 1-3 fold rating; the pinned prior-shrink multipliers (2017-2026 folds outside {H-1, H, H+1}); the swap adjustment's type model from seasons more than one away, standardised on them; team possessions from the rating games' stints | the held-out games | whole-season version vs the shipped test table: shrink exact, swap adjustment within 0.006 pts/100 |

Scoring (`104`, `105`): the held-out team-games as 97 stored them, reproduced to 6e-14 on all 1,860 folds before any
new number was read; the held-out lineup swaps from the held-out games' stints only.
