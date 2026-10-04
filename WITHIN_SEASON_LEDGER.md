# Within-season ledger: what each rating sees, and what it never sees

The owner, 2026-10-03: "the leaving stuff out approach here is getting quite complicated but it's okay.  Just keep
really special good tabs on it."  This file is those tabs.  `scripts/97_within_season.py` is the code; every check
named below runs on every fold and is written to `outputs/within/<tag>/checks.parquet`, and a failed one stops the
run.

## The set-up, in plain terms

- **Rated season**: one season, 2017 to 2026, regular season and playoffs together.
- **Rating games**: three quarters of the rated season's games, picked at random.  The rating is built from these
  alone.
- **Held-out games**: the other quarter.  Nothing about them reaches the rating.  The rating's misses are measured
  on them.
- **Fold**: one way of picking the held-out quarter.  Each season is dealt into four folds, so every game is held out
  exactly once per deal.  **Repeat**: a fresh deal of the same season (three by default), seeded so a rerun deals the
  same games.
- **The rating** here is steps 1-3 of the rankings (the box-score prior, the plus-minus ridge, the centring), the
  table the swap adjustment starts from.  The swap adjustment is not applied (open item 1).

What it is for: a calibrator for the ratings' misses, trained on the held-out games of the season itself, so it
cannot learn "a peak season comes back down next year" and shrink peak seasons.

## Every input, what it is built from, and the check

| input | used by | built from | rebuilt from the rating games only? | check (name in `checks.parquet`) |
|---|---|---|---|---|
| the possessions the ridge fits (who was on the court, points per 100) | the rating | the season's stints | **yes**: only the rating games' stints | `design <name>: rows only from fit games`, for all four designs |
| box-score rates (13 counting stats per 100 and the shooting percentages built from them), their padding toward league average, their centring, each player's possessions | the prior, the calibrator | the season's per-game box-score and possession tables | **yes**: both tables are cut to the rating games (the season-design cut kept the whole season's tables; that leak was found and closed here, the same one FINDINGS 31 found in the in-season work) | `design <name>: box and possession tables only from fit games` |
| the defensive target's three-point repricing (each opposing shooter's three-point percentage) | the rating (defence), the on-court defensive number | the season's shot table | **yes**: the shooters' percentages come from the rating games only | `design x3def*: response against the whole-season design (max change)`: 0 for the whole season, above 0 for a part season |
| the free-throw adjustment in both targets (each shooter's season free-throw percentage, leaving out the game itself) | the rating | stored in the stint files when they were built | **no: the one accepted leak**, as in FINDINGS 31.  The held-out games' free throws move a shooter's season percentage by a quarter of their luck; through his rating that is about 0.03 points per 100 for the heaviest free-throw shooters and near zero for everyone else, under the 0.1 that counts | none possible without rebuilding the stints per fold |
| on-court points per 100, each side, and the possessions behind them | the prior, the calibrator | the stints | **yes** | the design checks above |
| off-court points per 100 and the on-court minus off-court difference | the calibrator only | the stints | **yes** | the design checks above |
| share of team possessions played, share of games started | the prior, the calibrator | box scores (starters, games played) and stints (possessions) | **yes**: rebuilt game by game, the team's possessions cut with the player's | `roles: box scores read only from fit games`, `roles: stints read only from fit games` |
| years with his current team, number of teams this season | the prior (offence: years with the team) | the role table: his main team is the one he played most possessions for | **yes**: the main team is decided by rating-game possessions | the same two checks |
| where his shots came from (shot difficulty, shot-making) | the calibrator only | the shot table | **yes** | built with the rating games' list |
| score state: share of his possessions in garbage time, how close the games were, average margin | the calibrator only | the stints | **yes** | `context: stints read only from fit games` |
| share of his possessions in the playoffs | the calibrator only | the stints | **yes** | the same check |
| age, height, weight, draft pick | the prior, the calibrator | the biography table | not game data | -- |
| career before the rated season: seasons played, possessions, age at entry | the prior, the calibrator | earlier seasons only | not this season's games | -- |
| the prior models themselves (the boosted trees) and their labels | the rating | every season except the rated one and the one either side of it (`--exclude_neighbours=1`, the incumbent's year-over-year setting) | never trained on the rated season at all | `prior models: never trained on this season or its neighbours` |
| the prior's free scale, priced on cross-fitted columns | the rating | five folds of the rating games, the on-court numbers rebuilt without each fold | **yes**: the cross-fitting runs inside the rating games | as above |
| the centring (the average possession is played by a zero player) | the rating | the rating games' possessions | **yes** | `ridge: possessions are the fit games' possessions` |
| the season's game list (ids, dates, regular season or playoffs) | bookkeeping | the schedule | no outcomes in it | -- |

**Held-out side.**  The held-out games are scored on points as actually scored, with no luck adjustment, so no
shooting percentage is estimated on that side at all, and they are scored the way the year-over-year test scores a
season: predict every possession from the ten players' ratings, refit only the league level and the home edge on
the held-out games, and take the error per team-game.  A player with no possessions in the rating games has no
rating, and is scored at the same replacement level the year-over-year test uses for an unrated player (a quarter of
the average rating of players under 500 possessions).  Checks: `test: rows only from test games`,
`test: no test game among the fit games`, `test: shares sum to five`, `folds: fit and test games disjoint`,
`folds: fit and test games cover the season`.

## The reproduction check

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

**Folds: 120 (10 seasons x 3 deals x 4 folds), 2,640 pass/fail checks, 0 failed.**  Evidence the part-season
rebuild bites: the defensive target's response on the rating games moves by up to 5.6 to 15.5 points per 100 on a
possession when the shooters' three-point percentages come from the rating games alone; possessions come out at
three quarters of the season's; on-court numbers move by a median 0.8 points per 100.

## The calibrator (scripts/98_calibrator.py): what IT sees

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

## Open items

1. **The swap adjustment.**  The shipped rankings carry it (scripts/91, within-team lineup swaps); the ratings here
   do not.  A calibrator trained on steps 1-3 is applied to steps 1-3, and the swap adjustment would then run after
   it.  Rebuilding the swap adjustment from the rating games is possible and not done yet.
2. **Less evidence than a full season.**  A rating from three quarters of a season is noisier than a full-season
   one, so the misses measured here are somewhat larger than a full-season rating's.  The calibrator sees each
   player's possessions, which lets it scale its corrections with the evidence; whether the scaling carries to a
   whole season is a question to test, for example with half-season folds beside the quarter-season ones.
3. **Not rebuilt because nothing here reads them:** the RAPM pieces of the decomposition (experiment 26), the
   same-team measure.  If a calibrator wants them, they must be rebuilt from the rating games first.

## Settings of the run

- prior models: `scripts/62_single_year_board.py --exclude_neighbours=1 --score=0 --boards=2017,...,2026
  --out=season_ratings_within_base --save_models=within` (the incumbent's steps 1-3 settings; the build checks the
  saved settings and refuses others)
- folds: four per deal, three deals per season, seeded by season and deal
- held-out scoring: points as scored, the year-over-year test's scoring and replacement level
