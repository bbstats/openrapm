# Handoff: the single-year player rankings, one experiment at a time

> **What this file is (header added 2026-10-11).** Where the work stands, what is waiting on the owner, and the
> commands a next session runs. Below this header the text stands as it did on 2026-10-09, at the end of the
> Robustness pass, except "What ships" (now in today's module names) and the command blocks (the year-over-year
> test, the trade loss, the site and "Verify you are where this file says"), which name today's commands, and a
> few words changed to follow the writing rules.
>
> Since then the 2026-10 cleanup (branch `cleanup-pass`) rewrote the code as five step modules
> (`src/openrapm/rapm.py`, `box_prior.py`, `prior_rapm.py`, `prior_shrink.py`, `swap_adjust.py`) behind one
> command, `openrapm`. Rebuilt with it, the published step-3 table is byte-identical and the published final
> table agrees to within 1.2e-12 per 100 (one linear-algebra thread instead of several); the year-over-year test
> still reads 8.5857 for the incumbent.
>
> Numbered-script paths below (`scripts/62_single_year_board.py` and so on) and `src/eracoef/` module paths refer
> to the code at the tag `archive/research-2026-10`: `git show archive/research-2026-10:scripts/62_single_year_board.py`.
> Column and target names are those before 2026-10-11; `src/openrapm/data/legacy_names.py` maps each to today's.
>
> | in this file | today |
> |---|---|
> | `scripts/01_ingest.py`, `02_stints.py` | `openrapm data ingest`, `openrapm data stints` |
> | `scripts/49_role_panel.py` (with 65, 69, 86) | `openrapm data panel` |
> | `scripts/62_single_year_board.py` | `openrapm rate` (steps 1-3); `openrapm build` runs all five steps |
> | `scripts/97_within_season.py` | `openrapm folds` |
> | `scripts/99_prior_shrink.py` | `openrapm shrink` |
> | `scripts/91_swap_adjust.py` | `openrapm swap` |
> | `scripts/52_site.py`, `76_bias_groups.py`, `83_decompose_site.py` | `openrapm site`, with `--bias`, `--decomposition` |
> | `scripts/63_yoy.py` and the other twelve checks | `scripts/checks/`, same number and name |
> | 62 `--exclude_neighbours=1` | `--setting test` |
> | 62 `--features=boruta_noonc` | `--features box_score_and_bio` |
> | targets `xpts_ft`, `x3def_w0.25` | `points_with_expected_free_throws`, `points_with_expected_opponent_threes` |
> | the noise control `noise_regroup2` | `--fold-assignment 1` |
> | every other numbered script | retired |
>
> `README.md` and `PIPELINE.md` describe the code as it is now; `CONTRIBUTING.md` has the writing rules.

**This file is transient.**  It starts the next session.  `DECISIONS.md` is the permanent record and carries
every number quoted here.  Do not let this grow into a lab notebook.

The work happens on feature branches off `main`; the live site is served from `docs/` on `main`.  `pytest -q`:
**648 passed, 1 failed, 7 deselected (the slow tests, `pytest -m slow`, 7 passed), 1 xfailed, ~75 s** with the scraped
data (2026-10-15).  The one failure is the star-guard rank ceiling on Curry, 100th against a ceiling of 80, failing
since experiment 59 and waiting on the owner.

## START HERE (2026-10-15): experiment 63 ADOPTED and PUBLISHED; the tests cleaned up to three readings

**What ships.**  The incumbent is `cols06`: experiment 59's prior with each tree of the LightGBM booster shown a random
60% of its inputs, the same on both sides (`params/booster_params_lgb1_cols06.json`, `box_prior.COLUMN_SHARES`), and
step 4 on its own folds `cols06_early,cols06_within`.  The product table, the site's ratings, bias page, OpenRAPM
decomposition and similarity files are rebuilt and live on `main`.  2026 top 5: Wembanyama, Dončić, Gilgeous-Alexander,
Jokić, Kawhi.  Previous product tables: `outputs/season_ratings_product*_before_cols06.parquet`.

**How experiments are judged now (the owner, 2026-10-15).**  Three readings, nothing else ("The test", below):

| reading | script | rule |
|---|---|---|
| decider | `scripts/checks/151_wowy.py`, stint unit, step 4 | paired z -2 or below against the incumbent; noise control printed beside it |
| guard | `scripts/checks/63_yoy.py` | year over year not worse at z +2 |
| eye check | `scripts/checks/66_compare.py` | the 2026 top 20 |

`bash scripts/experiment_chain.sh NAME TARGET "EXTRA RATE FLAGS"` builds a candidate with its own folds for all thirty
seasons and runs exactly these three; `FULL_READINGS=1` brings back the older readings.  Report every result as
columns, one quantity per column, in data-science names, never in parentheses.

**Results on the decider, against `nocareer`, 30 seasons:**

| model | WOWY ΔMSE | WOWY paired z | WOWY seasons improved | YoY ΔMSE | YoY paired z |
|---|---|---|---|---|---|
| noise control | +0.066 | +0.72 | 12 of 30 | -0.030 | -0.68 |
| exp 63 column subsampling | -0.248 | -7.53 | 28 of 30 | -0.026 | -1.15 |
| career features restored | +0.875 | +6.41 | 5 of 30 | -0.343 | -5.35 |

**Why the old test missed these.**  Step 3 overrules about 60% of the team-level part of any prior change, since a
team's own games pin its total, and keeps about 87% of the within-team part; a team-game margin is the sum of the team's
ratings, so the year-over-year team-game test barely sees who on a team gets the credit.  The decider fits each team's
level out and scores only lineup changes: substitutions and missed games.

**Waiting on the owner.**
1. The decider's bar: z -2, kept from the old rule; confirm or change.
2. The noise control on the new base, `noise_cols06`, is not built yet: run
   `bash scripts/experiment_chain.sh noise_cols06 points_with_expected_opponent_threes "--fold-assignment 1"`
   (about 50 minutes, one build at a time) before the next experiment, so the decider has its control beside it.
3. The swap step, step 5, is not in the within-season folds, so the decider scores steps 1-4 only.
4. Re-read the dashboard inputs and experiment 61 on the decider: the dashboards were a tie on the old test with a
   real defensive gain, and they need their own folds, which leak, because dashboards are season totals; experiment 61
   changed only the bench's weighting.
5. The Curry star-guard test, failing since experiment 59.

**Branches (cleaned 2026-10-15).**  Only `main` remains, locally and on GitHub.  Merged branches were deleted:
`cleanup`, `cleanup-pass`, `hybrid-and-xpts`, `similarity`, `stat-splits`, `era-prior`, and `cleanup-9c`, whose two
commits were already in `main` under other hashes.  The worktree `A:/code/spmm_9c` was removed after its `data` and
`outputs` junctions were unlinked.  The detached worktrees `A:/code/spmm_gate` and `A:/code/spmm_old` are left in place:
they hold no branch, and their junctions point at the real `data/` and `outputs/`, so remove them only with
`cmd /c rmdir` on each junction first.

## START HERE (2026-10-14, night): experiment 62, the tracking and hustle dashboards as prior inputs -- screen passed on defence, both builds a TIE, NOT adopted

The owner, after experiment 61: "go for it - don't stop! just pick your next steps."  I chose the dashboards (tracking
first, hustle second; Synergy and matchups wait).  DECISIONS.md "Experiment 62" has the idea, the first look and the screen's
gate, registered before any result.
- **Built (uncommitted, branch `era-prior`):** `openrapm data dashboards` (984 files fetched: tracking 2014-2026, hustle
  2016-2026, Synergy 2013-2026, matchups 2017-2026; `data/dashboards.py`); `data/dashboard_rates.py` (30 panel columns,
  league-relative, NaN before a dashboard starts); `features.TRACKING_RATES` / `HUSTLE_RATES` / `MISSING_OK`, feature sets
  `tracking`, `hustle`, `tracking_and_hustle`; `box_prior` averages a missing-OK column over the seasons that have it and keeps
  a row that lacks it; the season panel rebuilt (`openrapm data panel --force`, 35 s; the 90 old columns bit-identical to
  `outputs/season_panel_before_dashboards.parquet`); GLOSSARY regenerated; check 146 (`scripts/checks/146_dashboard_screen.py`)
  with a placebo arm (the same columns shuffled among the players of a season); tests `test_dashboard_rates.py`,
  `test_check_146.py`, additions to `test_features.py`, `test_box_prior.py`, `test_panel.py`.
- **Verified:** `pytest` 621+ passed, the known Curry failure; `pytest -m slow` 7 passed (2026's step 3 rebuilds bit for bit).
- **Read-only first look** (`scratch/2026-10-14_pieces/explore_dashboards.py`): nothing lines up with the offensive games'
  part; the big-man activity stats (box-outs, contested shots, rebound chances, blocks) line up with the defensive one, same
  sign in 9 of 9 seasons, r about 0.15 to 0.18.
- **The leak to remember:** the dashboards are season totals, so within-season folds cannot rebuild them from the fit games
  (WITHIN_SEASON_LEDGER.md, the second accepted leak); a build keeps the incumbent's step-4 multipliers.
- **The screen (check 146, 631 s; DECISIONS.md has the table):** hustle and tracking-and-hustle pass the gate on both sides,
  tracking alone fails.  **The placebo (the same columns shuffled among players, missing values kept) gained on offence**:
  its missing pattern is a modern-era flag, which is all the offensive gain is (against the placebo, offence t -0.4 to
  -1.2).  **On defence the real arms beat both today and the placebo:** career yardstick hustle -0.033 (t -2.7), both -0.039
  (t -2.3); against the placebo t -4.0 and -4.2; the neighbours yardstick -3.3% (hustle) and -7.0% (both) of its error, 9
  of 9 seasons.  (First read as an era flag; check 147 put the era in by name and it gained nothing: RETRACTED, the cause of
  the placebo's offensive gain is unknown.)
- **Builds (registered before either result), 21 and 24 minutes:** `dashhustle` (the registered choice) then `dashboth`, each by
  the chain at the test setting on the incumbent's step-4 multipliers (`NAME=dashhustle FEATURES=hustle FIRST=2017 bash
  scratch/2026-10-14_dashboards/exp62.sh`; logs `outputs/<name>_driver.log`, `outputs/<name>_chain.log`).  **Both are ties on
  the adoption number, NOT adopted** (experiment table rows 62 and 62b, sheet rows 50-51): the covered-seasons year over year
  hustle -0.092 (z -1.22), both -0.159 (z -1.55); all 28 seasons -0.032 (z -0.89) and -0.078 (z -1.45), the noise control
  -0.030.  The inputs do carry information: the prior alone z -3.6 for the combination, step 4 z -2.2, then step 3 and the swap
  step take most of it back; the trade loss is a tie (offence z -1.6 for both).  The stars move by about what the noise
  control moves them (0.5 on average, 2026 top 30), the bench by two to three times as much; part of the stint-level gain
  appears in 1998-2006 (no dashboards there), cause unknown.  DECISIONS.md "The builds' result" has the table.
- **Check 147, the era flag (572 s): nothing passes.**  A flag from 2005, from 2014, both, against the nonsense odd/even split:
  no gain on the career yardstick (the 2014 flag is worse on defence, t +3.7); only the neighbours yardstick on offence moves
  (-0.0014, t -2.2, about 1%).  The placebo's offensive gain is therefore not an era flag.
- **Check 148, what the placebo gained from (435 s):** plain random noise columns (nothing missing) gain on offence (-0.9% of
  the career-yardstick error, t -2.9) and hurt defence (+1.0%, t +2.7), as the shuffled dashboard columns do: extra columns of
  any kind move the prior, so the offensive booster looks under-regularised and the defensive one does not (the shipped
  LightGBM settings have no column subsampling).  A tuning lead, not data.
- **Synergy play types, read-only:** frequencies show nothing against the offensive games' part; the efficiency measures
  correlate 0.05-0.15 only because they are the team's own results on those possessions (outcome data, like the on-court
  inputs, off the prior since experiment 45).  Not built.
- **Matchups, read-only:** the scoring of the attackers a defender guarded (assignment, not results) correlates +0.12 with
  the defensive games' part after his own prior, height and possessions, 9 of 9 seasons (2018-2026; the 2017 file is a
  fragment); the mirror on offence is empty (-0.04).  A coach's trust, not an outcome.
- **Check 149, the matchup screen (388 s): closed, nothing built.**  Offence gains what its shuffled placebo gains (-0.0121
  against -0.0120: any added column moves the offensive booster, check 148); defence does not beat today's list (+0.0103,
  t +1.35) though it beats its own shuffled copy (-0.0214, t -2.71).  Panel is now 122 columns (`outputs/season_panel_before_matchups.parquet`
  is the 120-column one); `features.MATCHUP_RATES` and the feature set `matchups` stay in, off by default.  Playoff matchup
  tables are empty for every season and the 2017 file is a fragment, so the matchups cover 2018 on, regular season only.
- **Check 150 and experiment 63, column subsampling (2026-10-14):** each tree of the prior shown 60% of its inputs passed
  the screen on defence (-0.020, t -3.6) and built as `cols06` is a year-over-year TIE (-0.026, z -1.15; noise control
  -0.030, z -0.68), stint level z -4.8 in every era, step 4 z -2.45, defence alone z -2.77.  Not adopted under the rule;
  `params/booster_params_lgb1_cols06.json` and `box_prior.COLUMN_SHARES` change nothing unless named.
- **Why step 3 gives prior gains back (read-only, DECISIONS.md):** step 3 overrules about 60% of the part of a prior
  change that shifts a team's total (the team's games pin it) and keeps about 87% of the part that moves credit between
  teammates; the team-game year-over-year test is nearly blind to the second (`cols06` within-team part: team-game z
  -0.5, stint z -3.6).  The owner left experiment 63 off.
- **The tests cleaned up (2026-10-15):** the decider is now the with-or-without-you test (check 151, stint by stint on
  held-back games); year over year is a guard; the 2026 top 20 the eye check ("The test", below).  On it experiment 63 reads
  z -7.5 (28 of 30 seasons; noise control z +0.7) and the career counts back on z +6.4 worse.
- **Open, for the owner:** (0) experiment 63 under the new test; (1) per-game tracking (`BoxScorePlayerTrackV3`, about 17,000 requests) would let folds rebuild the
  inputs from their fit games and so refit step 4 for the new prior, where the gain is lost; (2) a column-subsampling screen for
  the offensive booster (the noise-column lead); (3) experiment 61's defence-only look (needs the same-method waiver).  All the dashboard code is off by default
  (`features.FEATURE_SETS` keeps `no_career_counts`), the shipped ratings unchanged and bit-identical.
- **pytest:** 628 passed, the one Curry failure; `pytest -m slow` 7 passed.  Uncommitted: `src/openrapm/{box_prior,decomposition,
  features}.py`, `commands/{cli,data,within_season}.py`, `data/{panel,dashboards,dashboard_rates}.py`, checks 145 and 146, tests,
  GLOSSARY, the notes, `experiments/runs.csv`; `scratch/` is ignored.  Commit and publish only on the owner's word.

## START HERE (2026-10-14, evening): check 145, how much each piece of a rating holds up -- DONE, two families pass the gate, nothing built

The owner: "RAPM is inherently noisy, has issues w/ single season. I think some of our post-hoc ideas have been good.
think through some possible post-hoc adjustments ... also any true REAL good data we may be missing", then "let's think
outside the box ... as creative as humanly possible", then: look for hustle, tracking, Synergy play types and matchups
data ("wingspan/standing reach I don't love right now").  Plan: `~/.claude/plans/rapm-is-inherently-noisy-lucky-floyd.md`;
DECISIONS.md, "How much each piece of a rating holds up" has the rule, registered before any result.
- **The idea:** step 4 holds the games' part u at 1 and has only ever freed it whole (33b, a tie).  Split u into pieces
  (on court / sat / missed / team SOS; twos / threes / free throws; by how sure u is; the box-score part by height
  and role) and fit each piece's multiplier on the folds' held-out games.  Changes no rating.
- **Built (uncommitted, branch `era-prior`):** `decomposition.split_pieces`, `outcome_responses`, `residual_scale`;
  `openrapm folds --pieces` (writes `pieces_<stem>.parquet` per fold; off by default, bit-identical when off; the
  fold ratings equal the shipped folds' to the last bit); `scripts/checks/145_piece_multipliers.py`; tests in
  `test_decomposition.py`, `test_check_145.py`, `test_cli.py`.
- **Run:** `openrapm folds --models nocareer --base season_ratings_nocareer_step3 --seasons 1997-2026 --tag
  nocareer_pieces --pieces` (360 folds, 31 minutes, log `outputs/nocareer_pieces_folds.log`), then
  `python scripts/checks/145_piece_multipliers.py --folds nocareer_pieces` (13 seconds; log
  `outputs/piece_multipliers_run.log`; csv `outputs/csv/piece_multipliers_nocareer_pieces*.csv`).
- **Result (DECISIONS.md has every number):** `source` (on court / sat / missed / team SOS) and `certainty` (u in thirds
  by its own sd) pass the registered gate on both sides; `outcome`, `size`, `height`, `role` fail.  `missed` holds up
  about 4x (on court 1.5-1.7); bench players' u about 2.0-2.3x (most-minutes third 1.1-1.3x).  u as a whole holds up
  1.34 / 1.48 (in-season gain -0.0236, t -10).  The gain beyond that is only -0.0036 (source) and -0.0049 (certainty),
  while the 2026 preview moves players 1-3 per 100 (Dončić -1.9 on source; Jalen Smith +2.9, Hugo González +2.2 on
  certainty), so weakly identified; all in-season.  The preview is a scratch script (`outputs/csv/piece_preview_2026.csv`).
- **Experiment 61 (the owner: "Go ahead"): the certainty split, year over year -- a TIE, NOT adopted** (DECISIONS.md
  "Experiment 61"; table row 61, sheet row 49; `scratch/2026-10-14_pieces/make_table.py` + `exp61.sh`, 4 minutes).
  `pcert` 8.5955 against 8.5951 (z +0.09); the trade loss better on both sides (offence z -3.7, defence z -5.3, bench
  tiers z -3.6 to -4.8); consensus 0.846 against 0.836; lineup-swap offence order worse (z -3.3); **defence alone z -4.9,
  offence alone z +3.2, so the sides cancel** (the same-reason-same-method rule keeps me from picking a side; the
  owner may waive it).  The as-first-registered spread-held arm lost (z +3.4); the amendment (anchor on the
  most-possessions third) was made before any result.  Code for a proper step does not exist (a scratch preview only).
- **Waiting on the owner:** whether to leave experiment 61 closed, or look at defence alone (or a lighter version);
  and which dashboard source to build into the prior first (my proposal: tracking, then hustle).
- **The dashboards are fetched in full** (`openrapm data dashboards`, about 25 minutes, log `outputs/dashboards_full.log`):
  tracking 2014-2026, hustle 2016-2026, Synergy 2013-2026, matchups 2017-2026, regular season and playoffs, under
  `data/raw/dashboards/`.
- **The data probe (also uncommitted):** `openrapm data dashboards --probe` (`src/openrapm/data/dashboards.py`, tests
  `test_dashboards.py`; `panel.panel_keys`): a few seasons of 34 pulls of the four sources, cached under
  `data/raw/dashboards/`, coverage of the rated players in `outputs/csv/dashboard_probe.csv`.  Tracking from 2014
  (about 100% of possessions), hustle from 2017 (2016 covers 31% of players), matchups from 2018 (2017 partial;
  regular season only), Synergy from 2013 (18 of 22 pulls; 40-97% by play type).  Nothing reads them yet; no input is
  added until the owner has seen the coverage and chosen.  Proposed order: tracking, then hustle.
- **pytest:** 606 passed, 1 expected failure, 1 failed (`test_star_guards_are_not_buried`: Curry 94th against 80, the
  experiment 59 item still waiting on the owner).
- The other session's check-143 work stays uncommitted; stage by explicit path and hunk.

## START HERE (2026-10-14, afternoon): experiment 60, team-season flags in step 1's labels -- STOPPED, left off

**Closed by the owner after check 144:** "This is kind of the opposite of what I was aiming for. So sounds like we
should leave it off."  The build was stopped (no python left running); the within-season audit it finished was a tie
(-0.167, z -1.5).  The code stays, off by default and bit-identical when off.  The rest of this entry is as it stood.


The owner's thread after experiment 59: "team-season level flags for offense & defense (in RAPM fits?)"; placed in
step 1 (the owner's pick), where players changing teams between seasons tell a flag from its players.  DECISIONS.md
"Experiment 60" has the rule (registered before the result) and every number; plan
`~/.claude/plans/now-that-we-ve-made-virtual-starfish.md`.
- **Built (uncommitted, branch `era-prior`):** `rapm.team_flags`, `SeasonSums.add(..., teams)`, `.penalty`,
  `.flag_columns`, `.solve`; `LabelSettings.team_offense_penalty / team_defense_penalty` (0 = off, shipped);
  `--team-season-penalties` on `rate` and `build`; `data/tables.design_row_teams`; check 144 (the penalties by EM,
  the label moves); tests in `test_rapm.py`, `test_settings.py`, `test_cli.py`.  Off = bit-identical.
- **Check 144:** EM with the players held at 40,000 gives flags of sd 2.5 / 2.2 per 100 (penalties 1,530 / 1,360) that
  take the stars' credit (Nash -4.5, one-team stars -1 to -2); EM on players and flags together, scaled back to
  40,000, gives 45,900 / 56,500 and labels that barely move (a certain tie, not built).
- **Was running, stopped:** `bash scratch/2026-10-14_team_season/exp60.sh` (name `teamseason`, penalties 1530,1360),
  after `pytest -m slow` (7 passed); logs `outputs/teamseason_slowtests.log`, `outputs/teamseason_chain.log`.
- The other session's check-143 work stays uncommitted; stage by explicit path and hunk.

## START HERE (2026-10-14): experiment 59, the career counts off both priors -- ADOPTED and PUBLISHED

The owner, on the decomposition page: the box prior looked over-stuffed with career possessions and the swap
adjustment knocked it back down (LeBron 2026 offence: +3.88 of his +6.77 box score from career counts, swap -2.33).
Experiment 59 took the career counts off both priors (feature set `no_career_counts`; DECISIONS.md "Experiment 59").
- **Result:** year over year 8.5951 against 8.5825 (z +5.3), all of it predicting the season BEFORE (z +7.9); the
  season after a tie (z +0.8, stint level better z -3.1).  Within-season held-out worse too (+0.258, z +3.3, 0 of 10
  seasons): veterans come out underrated in their own season.  Consensus 0.836 against 0.809.
- **ADOPTED anyway (the owner):** "single year purity wins. i want it to truly be 'single season' data."  New rule
  for later work: a player's other seasons may not reach his rating as inputs, even when they would describe this
  season better.  The owner will bring the next ideas ("no need to propose fixes, i'll do that in a later sesh").
- **Now shipped (settings):** `no_career_counts`; step 4's folds `nocareer_early,nocareer_within`.  The incumbent for
  scoring is `outputs/season_ratings_nocareer.parquet` (8.5951); `experiment_chain.sh` and 135 default to it; the
  noise control is `noise_nocareer` (`135 --controls=noise_nocareer`; 8.5939 against 8.5951, z -0.7: a tie).
- **PUBLISHED 2026-10-14** (the owner: "publish it when it's done, please"): the product table rebuilt with
  `openrapm build` (backups `outputs/season_ratings_product*_before_nocareer.parquet`), then `openrapm site --bias
  --openrapm-decomposition --similarity` (the vanilla decomposition does not read the ratings), committed on
  `era-prior`, `main` fast-forwarded.  The Similarity tab reads the prior's inputs, so it lost the career inputs.
- **Left, by the new rule:** step 5's type model still reads seasons played before (counted from 1997).
- The other session's check-143 work (the check, its test, the 142 edit, its notes hunks) stays uncommitted.

## START HERE (2026-10-13): the OpenRAPM decomposition tab -- built, NOT committed

The owner's request: "another tab for OpenRAPM decomposition".  Their choices: split each published rating, offense and
defense separately, into **box score + on court + off court + team SOS + swap adjustment** (off court is the vanilla
page's two off-court pieces summed; the swap adjustment its own column; no split of the box score by stat); a side
toggle Offense | Defense | Total; the nav stays four tabs, and the Decomposition tab opens on OpenRAPM with a toggle
[ OpenRAPM | Vanilla RAPM ] at the top of both pages.  Plan: `~/.claude/plans/vanilla-rapm-decomposition-is-precious-firefly.md`.
- **Built (uncommitted, branch `era-prior`):** `src/openrapm/decomposition.py` (the vanilla page's team and row-group
  helpers moved there, plus the refit and the split), `rapm.solved_ridge` and `prior_rapm.ridge_system` (the old code
  split out, bit for bit), `openrapm site --openrapm-decomposition` (132 seconds), `docs/decompose_openrapm.html`,
  `docs/data/decompose_openrapm.json` (14,578 rows, 1.9 MB), the toggle on `docs/decompose.html`, the nav link on every
  page, tests (564 pass, 1 expected failure; `pytest -m slow` as below).
- **How the three games' pieces are exact:** step 3's games' part is a weighted sum of row outcomes, so solving step 3
  again on the table's own priors and grouping the rows splits it (`decomposition.py` has the equation).  The run stops
  before writing unless the pieces add up to every rating, the refit reproduces every games' part and prior scale in the
  table (about 1e-12; 3e-8 in 2020, whose solve needs a jitter that follows the prior columns' scale), and the rows are
  `ratings.json`'s.  Vanilla files rebuilt after the move: byte-identical to the committed ones.
- **Example:** Victor Wembanyama 2026, offense 2.07 = 0.69 + 1.41 - 0.27 + 0.06 + 0.18; total 7.16 = 3.98 + 2.65 - 0.21
  + 0.14 + 0.61.  Gary Payton II 2022, total 3.62 = 1.53 + 0.49 - 0.01 + 0.08 + 1.53.
- **Waiting on the owner:** the piece names (provisional: Box score, On court signal, Off court adjustment, Team SOS
  adjustment, Swap adjustment, OpenRAPM), any wording for the page (it has labels and the equation only), and
  commit / publish.  `docs/decompose.html`'s footer still names the retired `scripts/83_decompose_site.py`: the owner's
  wording, left alone.
- **After any ratings build:** `openrapm site --openrapm-decomposition`, or `tests/test_decompose_openrapm_site.py` fails
  (the file's ratings must be `ratings.json`'s).  The vanilla files do not depend on the ratings.
- **To commit:** this tab's hunks only.  `tests/test_cli.py` and the notes are shared with other uncommitted work (the
  warm-start screen, check 143), so stage by explicit path and hunk, never `git add -A`.

## START HERE (2026-10-13): experiment 57, the career counted from each player's debut -- ADOPTED and PUBLISHED

The owner's pick after experiment 56; **adopted** (the owner: "Adopt (your data-fix rule)").  Uncommitted on the branch
`era-prior` (after e2a351b); DECISIONS.md "Experiment 57" has every number.
- **Now shipped (settings):** the feature set `career_from_debut`; step 4's folds `fromdebut_early,fromdebut_within`.
  **The incumbent for scoring is `outputs/season_ratings_fromdebut.parquet` (8.5825)**; `experiment_chain.sh` and
  135 default to it.  The data order is now ingest, stints, events, **careers**, panel.
- **Product table rebuilt** with `openrapm build` (previous: `outputs/season_ratings_product*_before_fromdebut`) and
  **PUBLISHED 2026-10-13** (the owner: "Publish now"): `openrapm site --bias --similarity` (the decomposition page is
  plain RAPM and was left as it was), commit 67b6441 on `era-prior`, `main` fast-forwarded and pushed, Pages built,
  the live ratings.json equal to the committed one.  The 2026 rebuild test now reads the frozen copy
  `outputs/season_ratings_product_fromdebut_step3.parquet`.
- **Noise control on this base: `noise_fromdebut`** (2026-10-13, 43 minutes; `OWN_FOLDS=1 EARLY_FOLDS=1 bash
  scripts/experiment_chain.sh noise_fromdebut points_with_expected_opponent_threes "--fold-assignment 1"`; its folds
  `noise_fromdebut_early` + `noise_fromdebut_within`).  Against `fromdebut`: year over year 8.5819 against 8.5825,
  -0.018 (z -0.5); lineup swaps within 0.01 (defensive order +0.007, z +0.6); trade loss within noise; consensus
  0.811.  Use it as `135 --controls=noise_fromdebut` and as the size of noise for any later experiment.
- **The warm-start screen (check 143, 2026-10-13): no arm passes** (defence worse in all four, t +3.3 to +4.4; offence
  a tie).  With experiment 56's weighting screen and ratios, the prior's prices following the era is closed: the pooled
  prior stands (DECISIONS.md, "The warm-start screen").
- **UPDATE, same day: that cost is noise.**  Five builds of the OLD setup that differ only in fold assignment score
  1.2287 / 1.2253 / 1.1784 / 1.2163 / 1.2301 on the prior's defensive teammate order (a standard deviation of 0.022);
  experiment 57's 1.1974 and 1.2109 are inside that (z about -0.6).  Everything below on its attribution is retracted
  (DECISIONS.md, "How much the prior's teammate order moves when only its training folds are regrouped").  What
  stands: the career inputs matter for the prior's year-over-year accuracy (8.848 without them against 8.826), and
  one build's swap-test order is not evidence at a 0.03 difference.
- **Experiment 57's lineup-swap cost, traced (DECISIONS.md, "Where experiment 57's lineup-swap cost comes from"; RETRACTED, see above):**
  defensive order about -0.03 against the old incumbent, already in the prior (step 2); real (the old base's
  regrouping loses 0.003); not the estimated pre-1997 possessions (the season counts alone reproduce it); the
  defensive booster leans a quarter more on the career inputs (0.068 -> 0.084 of its split gain).  Left as adopted.
  **The screen of taking the career inputs off the prior (experiment 58) FAILED**: defence order -0.046 (worse than
  57's -0.031) and the prior alone worse year over year (z +6.9 off defence, +2.7 off both), so the "leans on career
  length" mechanism is refuted and the cause stays unexplained (it may be that the old incumbent is a favourable draw
  on this measure).  Diagnostic builds: `abl_seasons`, `abl_possessions`, `scr_nocareer_d`, `scr_nocareer_both`
  (`scratch/2026-10-13_debut_ablation/ablate.py`), not logged as experiments.
- **Built:** `openrapm data careers` (`src/openrapm/data/careers.py`; 490 careers cached in `data/raw/careers/`, the
  rows before 1997 in `data/cache/careers_before.parquet`); four panel columns counted from the debut
  (`features.CAREER_FROM_DEBUT`), the feature set `career_from_debut`; the panel rebuilt with every old column
  identical (the previous one is `outputs/season_panel_before_career_from_debut.parquet`).  `pytest` 525, slow 7.
- **Build (`fromdebut`):** year over year 8.5825 against 8.5827, z -0.4: a tie (1998-2002 -0.18, z -1.6); consensus
  0.809 against 0.802; the 2026 top 20 the same players.  Against it: the lineup-swap test's defensive order is worse
  in every era (z -3.1; the noise control z -0.8).  Old veterans rise (Olajuwon 2000 +2.57).
- **The owner's call (2026-10-13): adopted** on the data-fix rule, over the swap test (recommended against).

## START HERE (2026-10-13): the Similarity tab -- nearest neighbours, published

The owner's tab: each player-season's nearest player-seasons of other players, by the distance between their 34
standardised inputs (everything the prior reads, the season's games' part, the final rating); opens on Gary Payton II
2022.  A classifier with one class per player-season (the owner's first design, Shepard-Luce on its scores) found a
player's own seasons less well (0.409 against 0.437) and took 21 minutes; the owner chose nearest neighbours ("Sounds
like nearest neighbors is way better!"; DECISIONS.md, "The similarity page: nearest neighbours, after a classifier").
Branch `similarity` (off 39553a3, experiment 56's edits riding uncommitted beside it); plan
`~/.claude/plans/build-another-tab-called-curried-boot.md`.
- **Built:** `src/openrapm/similarity.py`, `openrapm site --similarity` (33 seconds), `docs/similarity.html` with a
  Distance column (the owner's choice) and the tab on every page, tests; `docs/data/similarity/` (10.6 MB, 31 files).
- **Published 2026-10-13** (the owner: "yes please commit/publish"): committed on `similarity` with only the similarity
  hunks of `cli.py`, `tests/test_cli.py`, this file and DECISIONS.md (experiment 56 edits them too and stays
  uncommitted), `main` fast-forwarded and pushed.

## START HERE (2026-10-12, evening): experiment 56, era-level prior adjustments -- a tie, NOT ADOPTED (2026-10-13)

**The owner's rulings, 2026-10-13:** experiment 56 not adopted; experiments 52-54 closed, not adopted (55 marked
adopted in the experiment table, as it was); this work committed and pushed on the branch `era-prior` (moved up to
`main` 4a75027), `main` untouched; **next: experiment 57, the career counts' 1997 cut-off** (seasons played from each
player's draft year, pre-1997 minutes pulled from the NBA's stats site; a data fix, so adopted on a tie).

The owner asked for era-level prior adjustments ("'vs league average' ... only solves ... the 'bias', not the
'slope'").  DECISIONS.md "Experiment 56" has every number; plan `~/.claude/plans/any-way-to-get-keen-lagoon.md`.
- **Found:** the counting rates were already measured from their season's league level; only the six ratios (true
  shooting %, effective field-goal %, three-point %, share of shots that are threes, offensive-rebound share; true
  shooting % on defence) were not.  A season-wide constant on the prior cancels in step 3.
- **Built:** `--era-halflife H` (training rows weighted toward the rated season; 0 shipped), the `_vs_league` ratios
  (feature set `ratios_vs_league`), check 142 (the screen, `scripts/checks/142_era_screen.py`), the chain's
  `EARLY_FOLDS=1`, the driver `scratch/2026-10-12_era_prior/era56.sh`, `prices_by_era.py`, `drift_multipliers.py
  --folders`.  `pytest` 495 passed, slow 6 passed.
- **Screen:** weighting toward the rated season lost at every half-life (pooling stays best); the ratios against the
  league won on offence (-0.018, t -4.0) and flattened the offensive prices across eras.  Sheet tab "Era screen".
- **Build (`vsleague`):** year over year 8.5821 against 8.5827, z -0.5: a tie; robust review like the noise control;
  the offensive lean moves from three-point shooters to two-point scorers (Brunson -0.92 in 2026).  Not adopted.
  Experiment table row 45.
- **Noise control rebuilt on experiment 55's rule:** `noise_regroup2_near` (its folds `noise_regroup2_early` +
  `noise_regroup2_within`); use it, not `noise_regroup2`, against `nearshrink` (135 `--controls=noise_regroup2_near`).
- **Kept for later:** the next try at the defensive slope (a warm start: the pooled booster, then a few trees on rows
  near the rated season).

## START HERE (2026-10-12): stat splits tried, era drift found, experiment 55 adopted (not published, not committed)

All of it is on the local branch `stat-splits` (off `cleanup-pass`), uncommitted: commit only on the owner's word.
DECISIONS.md has every number: "The stat-splits screen", "Experiment 52", "Experiments 53-54", "Era drift",
"Experiment 55".
- **Play-by-play splits** (`openrapm data events`, `src/openrapm/data/event_counts.py`; 22 parts of box counts in the
  season panel, `features.EVENT_RATES`; check 141, forward stepwise selection; check 138 `--parts=1`): rebounds by the
  miss (52), and the cuts stepwise selection kept on each half of the seasons (53, 54), all tie year over year;
  verdicts pending the owner (recommended: not adopted).
- **Era drift:** the defensive prior part holds up less in older seasons (0.79 in 1997-2006 against 0.95 in
  2017-2026); offence and the games' penalty do not drift.  **Experiment 55 ADOPTED** (the owner: "yes. build."):
  step 4's multipliers from the 8 nearest fold seasons (folds `weightfix_early,weightfix_within`, now the defaults
  in `settings.ShrinkSettings`); year over year 8.5827 against 8.5857 (z -2.9).  The scoring incumbent is
  `outputs/season_ratings_nearshrink.parquet`; `experiment_chain.sh` and 135 default to it.
- **The product table is rebuilt** (`openrapm build`; the previous one is
  `outputs/season_ratings_product_before_nearshrink.parquet`): 2026 top 20 nearly unchanged; 1997-2006 defence moves
  a median 0.18 per 100.  **Not published**: `openrapm site --bias --decomposition`, commit, push, on the owner's
  word.
- Next proposed: a prior whose training rows are weighted toward the rated season (the prices drift on defence too).

## START HERE (2026-10-09): the Robustness pass -- steps 1-4 done, experiment 46 (the body-weight fix) adopted as the base, 47 (honest scale) not adopted, 48 (pad once) stopped, checkpoint done, 49 (own-side inputs) not adopted, 50 and 51 not adopted -- the pass is closed; experiment 46 published

The plan (`~/.claude/plans/read-handoff-md-here-s-the-atomic-frost.md`; DECISIONS.md "The Robustness pass" has every
number): measure every input's lean on each season's own held-out games, remove the causes inside the fit one build at
a time on the owner's go, correct after the fact only what survives.  Rulings: fixes that remove a counted lean (and bug
fixes) are adopted on a tie; career counts allowed as they are; height/weight listings kept.

- **Step 1, the audit -- built and frozen:** `scripts/138_heldout_audit.py` + `params/audit_groups.json`.  Baselines
  `audit_lgb_noonc_2017` (2017-2026, threshold 3.64, 15 leans count) and `audit_lgb_noonc` (thirty seasons, threshold
  3.63, 39 count: big men underrated on defence +1.79 and overrated on offence -0.88; steals overrated, -0.95 on the
  total; scorers overrated on defence; three-point volume underrated on offence).  A candidate is read with
  `--random=<its folds> --mult_tag=<its folds> --draws=0 --threshold_from=audit_lgb_noonc_2017
  --reference=lgb_noonc_within --control=noise_regroup1_within`.
- **Step 2:** docs corrected; 106 takes `--mult_tag/--whole_tag/--base/--shipped/--out`; 62 warns when the cross-fitted
  scale rebuilds nothing; `experiment_chain.sh` takes `OWN_FOLDS=1`, `FIRST_PASS=1`, `CONTROL_TAG=`.
- **Step 3:** folds for 1997-2016 (`outputs/within/lgb_noonc_early`, models `prior_models_lgb_noonc_early_{1997,2007}`),
  deadline folds (`lgb_noonc_deadline`), the trade set on actual points (`tradeset_lgb_noonc_rs_pts`), the noise control
  (`noise_regroup1`, both sides regrouped, its own folds `noise_regroup1_within`, dumps `prior_rows_noise_regroup1_*`).
  Driver logs: `outputs/robustness_instruments*.log`; drivers in `scratch/2026-10-08_robustness/`.
- **Step 4:** the honest scale is 4-12% below the shipped one (about 10% on defence, every season) -- step 6 is called
  for.  The body-weight preview: `scratch/2026-10-08_robustness/weight_check.py` on the noise control's exact dumps.
- **Experiment 46 (step 5), the body-weight fix: ADOPTED 2026-10-09 as the base, NOT published** (DECISIONS.md
  "Experiment 46").  The training weight lives in `row_weight` (singleyear, 62, stackprior, gbdt_prior, 93, 94, 71,
  72, 85 and tests; `singleyear._guard` refuses bookkeeping columns named like inputs; `pytest` 559 passed, 1
  xfailed).  Year over year 8.5857 (z -2.15, all of it looking back); in season a tie.  **The incumbent for scoring is
  now `outputs/season_ratings_weightfix.parquet`**, its folds `outputs/within/weightfix_within`;
  `experiment_chain.sh` (`INC_NAME`, `INC_FOLDS`) and 135 default to it.  The site still shows experiment 45.
- **Experiment 47 (step 6), the honest scale: NOT ADOPTED 2026-10-09** (DECISIONS.md "Experiment 47").  The prior's
  scale priced on cross-fitting folds whose every input is rebuilt from the fold's training games (`62
  --crossfit=honest`; 97's season rebuild moved to `src/eracoef/seasoninputs.py`, reproducing exactly).  Year over
  year +0.123 (z +3.4); better than the incumbent before the prior shrink, worse after it, in all 10 seasons: the
  shrink, fitted on truly held-out games, already corrects the in-sample scale.  The noise control for the new base
  is `noise_regroup2` (folds `noise_regroup2_within`).  Its code is uncommitted until the owner says.
- **Experiment 48 (step 7), pad once: STOPPED AT THE FIRST PASS 2026-10-09** (DECISIONS.md "Experiment 48").  Career
  and chunk rows padded once on their summed counts (`scripts/139_count_panel.py` -> `outputs/role_panel_counts.parquet`,
  `singleyear.aggregate(pad_once=True)`, `62 --pad_once=1`, default off).  In season a tie; no targeted lean moved
  (defensive steals -0.32 -> -0.31); the ratings reshuffle by half the noise control's amount, along no axis.  Its code
  is uncommitted until the owner says.
- **Step 9, the checkpoint: DONE 2026-10-09** (DECISIONS.md "The checkpoint").  Thirty seasons on the base: 35 leans
  count (experiment 45: 39).  The three big defensive leans -- big men underrated, scorers and steals overrated -- all
  start in the box-score prior (rebounds +3.35 per 100 on the prior alone, +1.72 after the shrink).  Folds:
  `weightfix_early` (1997-2016), `weightfix_deadline`; audit `audit_weightfix`.  Called for: 12(a) (offensive scoring
  inputs off the defensive list, steals off the offensive list), then 13 (the group correction); 8 weak; 10, 11 not.
- **Experiment 49 (step 12(a)), each prior on its own side's box score: NOT ADOPTED 2026-10-09** (DECISIONS.md
  "Experiment 49"; `--features=boruta_noonc_ownside`).  Year over year +0.054 (z +1.3), lineup-swap test worse: the
  defensive prior's scoring inputs are real signal (without them high-usage players are overrated on defence by ~1 per
  100 more, z -27).  Only the offensive half helped (offensive steal lean -0.40 -> -0.30); not proposed alone (the
  same-method rule).  Its code (one feature set + a test) and experiment 48's are uncommitted.
- **Experiment 50 (step 13), the group correction: NOT ADOPTED 2026-10-09** (the owner: "Sounds good") (DECISIONS.md "Experiment 50").
  `scripts/140_group_correction.py` in 99's place (`SKIP_BUILD=1 SHRINK_SCRIPT=scripts/140_group_correction.py
  SHRINK_TAG=weightfix_within bash scripts/experiment_chain.sh groupfix x3def_w0.25`): the shrink refitted with six
  lines (defence size and steals; offence size, three-point volume, turnovers, fouls) fitted on other seasons' held-out
  games.  Biases on thirty seasons 35 -> 10; in season -0.255 (z -3.2); lineup-swap better on defence; but year over
  year +0.051 (z +1.1) with every year-over-year slice family leaning worse than the noise control, and defensive
  consensus 0.759 vs 0.780.  Attribution (50b, multipliers held): +0.145 (z +3.0) -- the cost is in the lines.
  Side split: offence alone +0.003 (z +0.1), defence alone +0.046 (z +1.3).  Audit a correction with `138 --correction=season_ratings_groupfix_shrunk_raw`.
- **Experiment 51 (the owner's idea: "feature corrections to improve rank and then correct scaling on top"): NOT
  ADOPTED 2026-10-09** (the owner: "Ok sounds good! Publish away") (DECISIONS.md "Experiment 51").  Experiment 50's order with the base's spread
  (`scratch/2026-10-08_robustness/spread_restore.py`): year over year +0.031 (z +0.7; forward -0.064, back +0.126), the
  order alone z +0.8; within-team order better on each side (defence z +2.6, offence z +1.1) but the total's ties (z
  -0.9); defensive consensus 0.758; robust review's year-over-year slices 3 better / 8 worse (noise 7 / 4).
- **PUBLISHED 2026-10-09 (the owner: "Ok sounds good! Publish away"): experiment 46's product table** (the site's ratings
  and bias page; see "The pipeline" below for the build).  2026: Edey 40th -> 12th, Holmgren 6th -> 13th, Wembanyama
  -0.81 and still first; consensus 0.806 against 0.820 for experiment 45's product.

## Previously (2026-10-08): the LightGBM prior without the on-court inputs is live

> **QUESTION FOR THE OWNER -- answer here before the next build.**  The next build is a calibration step: correct each
> rating by one tree curve per input stat, so that no stat (steals first) is over- or underrated as a whole (the
> curves and the plan are under "Next" below).  **What should those curves be fitted to?**
>
> - **(a) Each season's own held-out games** -- the target your 2026-10-03 rule sets for any calibrator.  The folds
>   already exist for the current ratings (`outputs/within/lgb_noonc_within`, 2017-2026; earlier seasons would borrow
>   the pooled curves, as the prior shrink does).  First step: redraw the 137 curves on these misses (about 20
>   minutes), so you see whether steals still slope before anything is built.  **Recommended.**
> - **(b) The with/without correction** the 137 curves use now.  Simpler, but it reads the neighbouring seasons, so
>   it would also teach "peak seasons regress" -- the reason for the 2026-10-03 rule.
> - **(c) No calibration step:** fix it inside the prior instead (each stat's own noise, inputs as posterior means;
>   steals out of the defensive inputs).
>
> **Answer (2026-10-08):** none of the three alone -- **the Robustness pass**: "research the most robust way to deal w/
> these biases ... ideally these biases don't even exist in the first place."  Plan approved: measure every input's
> lean (and every group of correlated inputs) on each season's own held-out games, remove the causes inside the fit one
> build at a time, and correct after the fact only what survives.  Rulings, the two defects found (body weight
> overwritten by the training weight; the prior's scale not cross-fitted) and the plan: DECISIONS.md, "The Robustness
> pass"; the step-by-step plan is `~/.claude/plans/read-handoff-md-here-s-the-atomic-frost.md`.

**Published 2026-10-08 (the owner: "Publish").**  The site's ratings are experiment 45: the box-score prior fit by a
deterministic LightGBM (`params/booster_params_lgb1.json`, `src/eracoef/lgbprior.py`) on the box-score inputs WITHOUT
the on-court plus-minus and on-court possessions (`--features=boruta_noonc`), five player folds, the prior shrink
refitted on its own within-season folds (`outputs/within/lgb_noonc_within`).  Year over year 8.5894 against
chimeraboost's 8.5995 (z -2.5), stint z -8.9, 36 of 47 prediction slices better, lineup-swap test 11 of 12; offence
with/without and defensive consensus agreement worse.  The owner kept it on the 2026 top 20 ("keep the one that likes
butler/jokic/edey and drops hugo/ajay/oso down").  DECISIONS.md, experiments 43-45 (and 44's rematch).
**The Portable column is NOT on the site** (52's portable branch and `--portable` flag were deleted in the cleanup pass; `docs/index.html` restored from main): its
ratio was fitted on the chimeraboost ratings and the owner has not approved it.

**The incumbent for scoring** is `outputs/season_ratings_lgb_noonc_rs.parquet` (trade-loss alphas
`tradeset_lgb_noonc_rs_alpha.parquet`).  `scripts/experiment_chain.sh` defaults to it (`INC_NAME`), appends the
incumbent's 62 settings after an experiment's own flags (`BASE_FLAGS`, so the experiment's win), reads its multipliers
(`SHRINK_TAG=lgb_noonc_within`) and can start at the shrink (`SKIP_BUILD=1`).  `scripts/135_robust_review.py` (the
owner's not-brittle review: every test cut into slices, counted beside a noise control and a replicate) defaults to it.
A LightGBM test build takes ~20 minutes for thirty seasons; leave one player out (`--lopo=1`) ~70.

**Open, the owner's call:**
1. The attribution control: chimeraboost on the same inputs -- experiment 45's gain is not yet split between the
   learner and the inputs.
2. Leave one player out on experiment 45 (removes the remaining which-players-are-left-out noise; LightGBM has no
   seed noise).
3. The Portable column: refit `112_portable.py` on the new ratings (it needs 97 `--split=deadline` on the
   `lgb_noonc_within` models), then publish if approved.
4. Experiment 43 (the prior averaged over five groupings) was not adopted; with LightGBM, leave one out does its job.
5. The owner has a chimeraboost issue to file: "decide once, bag after" for bagged fits (the text is in the session).

### Next: the owner's stat-variance idea (2026-10-08 evening; not started)

The owner: "certain stats, archetypes, or +/- gotchas tend to have more variance ... a LOT of blocks + steals causing
issues ... it does look like we're overrating steals by some amount ... even just knowing what the variance of each
stat is might help us to an even more robust, bayesian, probabilistic input version", then: "I'm more confident that
steals are being at least slightly overvalued."

**First evidence, already measured** (`95_miss_by_group.py` on the new incumbent, `--tag=lgb_noonc_rs`; the lean is the
mean with/without correction, positive = the games say he is better than rated):

| defensive rating, fifths of the rated season's... | bottom | second | middle | fourth | top |
|---|---|---|---|---|---|
| steals | **+0.117 (z +4.3)** | +0.067 (z +2.7) | -0.017 | -0.018 | **-0.069 (z -2.3)** |
| blocks | +0.015 | -0.007 | -0.002 | +0.023 | +0.050 (z +1.6) |
| steals plus blocks | +0.042 | +0.039 | -0.002 | +0.006 | -0.006 |

So steals are overvalued on defence (monotone from the bottom fifth to the top: 0.19 in 95's units -- the lean is
alpha over the expected miss, `95:127` -- which is about **0.07 points per 100**, corrected 2026-10-08), blocks are not
(if anything slightly undervalued); the two cancel in `stocks`, which the defensive prior also reads.  The extra miss
(variance) by steals is unremarkable (top fifth 1.04 of expected, z +1.1): this is a lean, not noise.  Offence leans the
same way on steals (bottom fifth +0.069, z +2.6; `stl` is an offensive input too).  Player types: "most blocks and
personal fouls" lean -0.096 on defence (z -2.4), "most offensive rebounds and blocks" +0.096 (z +3.0).  The largest
leans anywhere are team context (team points with him off the court, z up to 11.5), not examined yet.  Full table:
`outputs/csv/miss_by_group_lgb_noonc_rs.csv` (and `.html`).

**Calibration curves, the owner's preferred view** ("a calibration plot style rather than binning ... we should see
y = 0 ... more like kbins or tree based model to get any non-monotonicity"): `scripts/137_stat_calibration.py`.  For
each input and side, the same with/without miss against the input: 20 equal-count bins (95% bars, clustered by player)
and a one-input LightGBM curve, its size measured out of player fold and its z against the input shuffled within
season.  `outputs/csv/stat_calibration_lgb_noonc_rs.csv`, `_bins.csv`, `outputs/stat_calibration_lgb_noonc_rs.png`.

| side | most mis-priced inputs (z; bottom-bin and top-bin miss, + = underrated) |
|---|---|
| defence | weight z +16.9 (heaviest +0.05); **steals z +12.8 (+0.06 -> -0.05)**; career possessions z +12.4 and seasons played z +11.3 (a hump mid-career, underrated); height z +7.0 (tallest +0.07); true shooting z +6.2.  Blocks z -0.2 (calibrated); steals plus blocks z +1.5 |
| offence | weight z +12.1 (heaviest -0.04); share of team possessions z +10.7 (a hump at 30-50%, underrated); career possessions z +7.0; true shooting z +5.9 (highest overrated); three-point rate z +5.6; **blocks per 100 z +5.3 and steals per 100 z +5.0 (high values overrated on offence)** |

The curves are small in points (spread 0.02-0.03 per 100, ends +-0.05 to 0.07) but the with/without correction is a
ridge estimate, so the true misses run larger.  Several bend in the middle (career, minutes, true shooting): a
straight-line fix would miss them.

**Proposed plan** (the owner's call; each build is one fast LightGBM chain, ~30 minutes):
1. **A calibration step** -- rating correction = the sum over inputs of one shallow-tree curve each (non-monotone
   allowed), per side, fitted on other seasons and shrunk by its out-of-fold fit; box score and body only, no team
   context (experiment 33's 78-input trees learned team context and hurt year over year).  **Its target needs the
   owner's ruling:** on 2026-10-03 the owner ruled a calibrator's target must be the season's OWN held-out games (97's
   folds; for the incumbent `outputs/within/lgb_noonc_within`, 2017-2026), not the with/without correction, which reads
   the neighbouring seasons and would teach "peaks regress".  So first redraw 137's curves on within-season misses
   (each player's held-out miss: a ridge of 97's test-game residuals on players' shares).
2. **Or fix it inside the fit:** each input's noise (`pad.mom_k` on per-possession counts: how many possessions until a
   rate is half signal; steals and blocks are rare events) and inputs as posterior means with their own constants;
   steals out of the defensive inputs (blocks kept).
3. Read: the year-over-year test, the robust review, and whether the 137 curves flatten.

**Built 2026-10-07/08:** `62 --prior_groupings` (43); `--booster_params=<file with "learner": "lightgbm">`, `--lopo=1
--prior_jobs`; `src/eracoef/lgbprior.py`; `scripts/135_robust_review.py`, `136_tune_lightgbm.py`; 97 `--booster_params
--features`; tests `test_prior_groupings.py`, `test_lgbprior.py`.  Drivers and diagnostics:
`scratch/2026-10-07_prior_groupings/`, `scratch/2026-10-08_lightgbm/`.

### What experiments 43-45 found (2026-10-07/08; DECISIONS.md has every number)

| finding | evidence |
|---|---|
| The prior's training groups add real noise, and with chimeraboost about half is the booster's own randomness | 2015: one grouping moves a defensive prior sd 0.40 per 100; the same rows refit with five seeds, 0.27; offence 0.13 / 0.07 (`scratch/2026-10-07_prior_groupings/seed_noise.py`) |
| Averaging chimeraboost over five groupings helps a little, broadly (experiment 43, not adopted) | 8.5970 (z -1.7), replicate 8.5990 (z -0.3); better in 42 / 34 of 47 prediction slices against 8 for one random regrouping |
| Judge a change by its pattern across slices, beside a noise control and a replicate (the owner's ask) | `135_robust_review.py`; a z between two noise draws is about N(0, 1) whatever their size, so compare sizes, not z |
| LightGBM: deterministic and 9-40x faster, but its screen win over chimeraboost was tuning | experiment 32's tuned chimeraboost scores as well on the same screen |
| With the on-court inputs, LightGBM loses inside teams | 44/44b: prior alone better (z -1.9), final worse (z +1.6 to +2.2); stint level, lineup swaps and team share of the rating all worse; refitting the prior shrink changes nothing (44r/44br) |
| Without them it wins, and the ratings stop tracking the team | experiment 45: z -2.5, stint z -8.9, team share of the defensive rating 0.08 (was 0.16); adopted and published |
| Linear leaves push low-possession players to the extremes | Edey's defensive prior 2.2 (chimeraboost) -> 4.1 (LightGBM, linear leaves) -> 2.7 (without); the owner preferred the linear-leaves numbers |
| The tuning screen cannot see the bench | it weights rows by possessions, so a 590-possession player barely counts there while the prior decides most of his rating |
| Steals are overvalued on defence; blocks are not | the table above |

### What the shot-quality build and experiment 42 found (2026-10-06/07; DECISIONS.md has every number)

- **Two leaks in the play-by-play, fixed:** a shot's own timestamp knows its result (makes and misses are logged with
  different delays; timing is now measured from the row before the shot), and inside 10 ft the scorer logs a made
  shot about 1 ft nearer the rim than a miss (the spot is coded as 0-6 / 6-10 ft zones there).  A third, unlocated
  (0, 0) twos in 1997-2010, is result-coded too and still in every model's twos (not fixed; threes are clean).
- **The shot model:** boosted trees on the current logistic predict makes clearly better (shooter-adjusted -0.64 per
  1000, z -12.8) and lose no team test; the tuned logistic's score-margin terms are what failed the team tests.
- **Experiment 42** (opponent threes priced at shot quality x the shooter's other-half skill): rejected.  As first
  run +0.128 (z +2.8), but two thirds of that was the training-group noise above; held fixed (42f) +0.043 (z +1.3),
  no gain.  It drops Ighodaro 18th to 46th in 2026 (the owner thinks he was too high).
- Pieces: scripts 114-134, `src/eracoef/{shotframe,shotclock,shotmodel,shotfeatures,shotlearners,shotsearch,tracking,
  movement,shottest}.py`, `data/shotq/` (tables `_table`, priced arms `c_*`, per-attempt qualities `q42_*` with their
  stint side tables), the sheet's "Shot quality" tab (rows 77-99) and Experiments rows 42, 42x, 42f, C1.  Memory
  "Shot-quality build", traps 24-27.

### Waiting on the owner (most recent first)

0. **Experiments 42 / 42x / 42f: rejected** (the owner, 2026-10-07: "Sounds good").
1. **Publish the Portable column?**  Built and checked locally, not committed: `docs/index.html` (a sortable
   Portable column, portable offence / defence / total in the CSV download) and `docs/data/ratings.json` (team
   ratings unchanged, max difference 0.0 over 14,578 rows).  **No explanatory text on the page**: the owner called
   the drafted footer "AI slop, do not include" (memory "No drafted prose on the site").  To publish: commit,
   fast-forward `main`, push.
2. **Experiment 41, the swap adjustment's give-back rule**: recommend keeping the shipped (minutes) rule; flat is
   not adoptable (below).
3. **Experiment 40, the portable rating**: the owner on the ranks: "Portable rank looks great"; the verdict rides
   on item 1.
4. **Experiment 34, teammates' shot mix**: a tie (8.5993 vs 8.5995); recommend not adopting.
5. **From experiments 35-38**: (a) the prior without on-court plus-minus -- the strongest open candidate now:
   experiment 39 finds on-court numbers are team context, and experiment 38 that they hurt out of season;
   (b) RAPM on the linear box prior (B3) as the base, with the luck-adjusted targets and the swap adjustment on top
   (B3 beats OpenRAPM year over year, 8.587 vs 8.600); (c) the swap adjustment at x0.75.  Experiment 37's stages 4-6
   (era drift, targets, lockbox freeze) not run.
6. **Proposed, not started**: the swap adjustment's TYPE model for White-type cases (its strength, now 0.5, or its
   features).  Boston's +0.75 net offensive type is what charges Derrick White, whatever the give-back rule.
7. Older and still open: experiment 32's tuned booster (options a/b/c, DECISIONS.md "Experiment 32"); experiment 29,
   the team-movement weight (the owner does not want one-team players halved); the stayed list (30b).
8. Back pocket: https://vorp.app/nba/about (JavaScript-rendered; a plain fetch returns only "vorp").

### What experiments 34-41 found (DECISIONS.md has every number)

| exp | the owner's question | answer |
|---|---|---|
| 34 | teammates' shot quality with him on the court | a tie once the shrink's folds are pinned (trap 22) |
| 35 | slope vs error; part-season OpenRAPM against vanilla baselines | slope reads scale (about twice as sensitive), error ranks systems; seasons are the unit (more deals add nothing); **B3, RAPM on a linear box-only SPM prior tuned on other seasons, beats OpenRAPM year over year (8.587 vs 8.600)** and orders teammates better; OpenRAPM leads within season, mostly 2017-2026 where its shrink was fitted |
| 36 | does the best penalty change with sample size? | vanilla RAPM's does not (Bayes); OpenRAPM's prior weight falls with more games while a linear box prior's rises; under identical tuning the linear prior wins (B4, RAPM on OpenRAPM's prior, 8.609) |
| 37 | a box prior fit directly on held-out games | every input group helps within season, none year over year; only fitting the shape year over year helps (t -3.4); probability of backtest overfitting 0.00 |
| 38 | what helps in-season vs other seasons | out of season wants 2-3x the penalty; role, same-season RAPM, on-court numbers, shot-making and assists help in-season and hurt out of season; age and career help out of season |
| 39 | traded players: real or regression to the mean? | the box score travels; defensive RAPM beyond the box score and on/off-court numbers stay with the team; eFG / TS / 2P% regress; the prior shrink and the swap adjustment are same-team corrections |
| 40 | a portable rating beside the team rating | half the plus-minus part travels (0.49 mid-season trades, 0.56 off-season moves, pooled 0.51); movers rate 0.8-1.0 worse but play to their rating once matched on minutes; not better on held-out error, so a second number, not a replacement |
| -- | how robust is the swap adjustment per player? | on average it holds up in every test; per player each piece is close to a coin flip (53-54% toward consensus, the whole 63%); the give-back is the weakest piece |
| 41 | the give-back rule (prompted by Derrick White) | flat give-back worse at the team-game level (z +2.2) and on the trade loss, better at the stint level (z -7.9); no give-back much worse (z +13); White 88th to 66th only |

### Where the pieces are

- Sheet "OpenRAPM experiments", https://docs.google.com/spreadsheets/d/1gPeRCFUtJNPtsBq_j2CNiTTdrarb98Yf3zXYx974jfI:
  tabs Definitions, Experiments (rows through 41), Scorecard, In vs out of season, Traded players, Portable rating.
- Chains and logs: `scratch/2026-10-05_scorecard/` (experiments 35-40) and `scratch/2026-10-06_giveback/` (41).
  Outputs: `outputs/within/<tag>/` (folds, incl. `deadline`), `outputs/scorecard/`, `outputs/heldout/`,
  `outputs/portable/`.
- `scripts/113_swap_explain.py --team=BOS --player="Derrick White" --also=2024,2025` rebuilds a team's swap
  adjustment step by step (type, give-back, centring, spread hold) and asserts it equals the shipped table.
- Emailed to the owner 2026-10-06, "OpenRAPM 2026: team vs portable ratings": a Drive CSV of all 582 players
  (team and portable rating, box part and plus-minus part per side).

## How the work is done (the owner, 2026-09-13/14)

- **Two jobs only: implement the owner's ideas, or propose new ones.**  Never run an idea the owner has
  not said "go" to.  One experiment at a time; never a queue of several overnight.
- **One test decides**: the year-over-year test below.  The consensus checks, the trade loss and the 2026
  top 20 (the eye test) are read every time and can veto; they are never fitted to.
- **Every experiment ends with the 2026 top 20** (`/experiment-comparison`; `scripts/checks/66_compare.py`).
- **Plain words.**  No invented labels, no single-letter names, no "board" (say the player rankings), no
  "floor" (say the check).  Define a term the first time it is used.  End every message with a
  one-sentence TL;DR unless the message is itself one sentence.  **Movement is reported in points per 100,
  never in rank places**, and under 0.1 is nothing.
- **On the site and in anything published the ratings are "OpenRAPM"** -- never "ours" or "we" -- and the
  published pages are American English.  The repo's own prose is British; leave it.

## The rulings that bind

1. **One rating per player per season, from that season's games only**, regular season and playoffs.
2. **Single year or bust**: a player's own other seasons may not reach his rating, not even through the
   booster's memory (hence the out-of-player priors).  Model coefficients may be learned from other seasons.
3. **Never train on the current season until its Finals are over.**  Loading is always allowed.
4. **The SPM is trained on one row per player** (his career average with the rated season out) PLUS chunk
   rows of his seasons; not one row per player-season.
5. **The consensus is a sanity check, never a fitting target**; a gross miss vetoes.  ("Keep chimeraboost" was
   lifted 2026-10-08: the owner chose LightGBM for speed, experiments 44-45.)
6. Each season is centred at possession-weighted zero per side (the RAPM convention), 2026-09-14.

## What ships (`openrapm build`; before the cleanup, `scripts/62_single_year_board.py` defaults plus `--booster_params=lgb1 --features=boruta_noonc`)

1. **Target** (step 1, `rapm.career_labels`): one RAPM per player over every season except the rated one
   (penalties 40,000 / 40,000 / 0, closed).
2. **SPM** (step 2, `box_prior.season_priors`; since 2026-10-08, experiment 45): a deterministic LightGBM (settings
   `params/booster_params_lgb1.json`: linear leaves, 250-300 rows a leaf, 31 bins, no subsampling) on
   `box_prior.training_rows` (the career row plus contiguous 1-, 2-, 3-season chunks, two features saying how much
   evidence a row rests on), features `box_score_and_bio` (called `boruta_noonc` until 2026-10-11; 17 offence / 14
   defence: the on-court plus-minus and on-court possessions OUT).  Until 2026-10-08 it was chimeraboost on
   `boruta` (21 / 17, on-court columns in).
   **Out-of-player**: five player folds balanced on the label, every player's prior from the fit that never saw his
   rows.  **The DEFENSIVE label is un-shrunk and the offensive one is not** (`settings.LabelSettings`,
   `unshrink_columns`, adopted 2026-09-18).
3. **Rating** (step 3, `prior_rapm.fit`): `scale x prior + residual`, the scale unpenalised on the full prior (the
   cross-fitting the old code ran gave identical values and was removed), the residual penalty fixed at
   **13,037** on both sides.  Then centred.
3b. **The prior shrink** (step 4; adopted by the owner 2026-10-03, "Good, adopt for now"; DECISIONS.md, experiment
   33b): the box-score prior part of each rating times a held-out multiplier per side (published: offence
   0.699-0.715, defence 0.929-0.976), the games' part unchanged, re-centred -- `openrapm shrink`, fitted on the
   within-season folds of 2017-2026 (`openrapm folds`, `WITHIN_SEASON_LEDGER.md`); earlier seasons borrow the
   pooled value.
4. **The swap adjustment, team version** (adopted by the owner 2026-10-01; DECISIONS.md, "The swap adjustment
   with the spread held"): credit inside each team is re-split by lineup swaps -- a type model at half strength,
   learned from other players and other seasons -- with each team's total fixed and the spread inside teams held
   to step 4's.  Step 5, `openrapm swap`; `openrapm build` runs it after steps 1-4.

`outputs/season_ratings_product.parquet` (every other season allowed in the prior) is what
`docs/data/ratings.json` and the site are built from.  Since 2026-10-09 it is experiment 46's product table (the
body-weight fix on experiment 45's prior), built then by `scratch/2026-10-08_robustness/publish46.sh` (62 in three
chunks, 99, 91).  Today the same table, and the site from it, are two commands; publish only on the owner's word:

    .venv/Scripts/python -m openrapm build
        (steps 1-5 at the product setting for all thirty seasons, ~15 min: writes outputs/season_ratings_product.parquet,
         the published table, with outputs/season_ratings_product_step3.parquet and _step4.parquet beside it; step 4
         reads the multipliers off outputs/within/weightfix_within, the default of --folds)
    .venv/Scripts/python -m openrapm site --bias --decomposition --openrapm-decomposition --similarity
        (docs/data/ratings.json, docs/bias.html with outputs/bias_groups.json and outputs/bgmm_proba.parquet,
         docs/data/decompose_1y.json and decompose_3y.json, ~5 min, docs/data/decompose_openrapm.json, ~2 min, and
         docs/data/similarity/, ~30 s; then commit and fast-forward main)

The previous product tables are kept as `outputs/season_ratings_product_before_weightfix.parquet` (experiment 45) and
`outputs/season_ratings_product_before_lgb_noonc.parquet` (chimeraboost).  The prior shrink's
multipliers come from `openrapm folds` on experiment 46's models (`weightfix_within`): product rule offence
0.699-0.715, defence 0.929-0.976 (experiment 45's, on `lgb_noonc_within`, were 0.706-0.719 and 0.927-0.970).
**The incumbent every candidate is scored against is now `outputs/season_ratings_weightfix.parquet`** (8.5857 on the
year-over-year test; experiment 46, adopted 2026-10-09); before it `outputs/season_ratings_lgb_noonc_rs.parquet`
(8.5894, experiment 45).  Before that, `outputs/season_ratings_priorshrink.parquet` (8.600;
adopted 2026-10-03): `season_ratings_unshrinkdef.parquet` (steps 1-3 at
`--exclude_neighbours=1`) through `99_prior_shrink.py --rule=test` (each season's multiplier fitted outside it and its
neighbours) into `season_ratings_priorshrink_raw.parquet`, then 91 with `--kappas=0.5 --taus= --hold_spread=within`
and the default `--exclude_near=1`, which keeps both scored seasons out of the type model.  The previous incumbent,
`outputs/season_ratings_swapadj_within.parquet` (8.666), is kept.  **A candidate that changes steps 1-3 must go
through the same 99 and 91 commands before it is scored**, or it is compared without the steps the incumbent has.
Caveat: 99 reads the multipliers off the incumbent's own within-season folds; a candidate that changes the prior
changes how far its prior part holds up, so a serious one needs its own folds (62 `--save_models`, then 97; today
`OWN_FOLDS=1` in `scripts/experiment_chain.sh`, which runs `openrapm rate --save-models` and `openrapm folds`).
The older system's `artifacts/season_ratings.parquet` was removed in the 2026-10 cleanup; player names now live
in `params/player_names.csv`.

Threads: every `openrapm` command pins the linear-algebra libraries to one thread before numpy loads (numba is no
longer used).  **Never run two
builds at once**, and kill a chain's python children when you stop it.

## The test (since 2026-10-15): with or without you, within the season

The owner: "too many tests, impossible to scrutinize ... 'within-season' is truly the best metric, particularly WOWY".
`bash scripts/experiment_chain.sh NAME TARGET "EXTRA RATE FLAGS"` builds the candidate, its own folds for all thirty
seasons (about 40 minutes), steps 4 and 5, and reads three things:

1. **The decider:** `scripts/checks/151_wowy.py` (stint unit, step 4): every stint of the held-back games predicted from
   the ten players on the court, each team's attacking and defending level fitted out, so only how a team's results move
   as its lineup changes is scored (substitutions and missed games).  Against the incumbent season by season, the noise
   control (`noise_nocareer`) beside it.  Below 0 is better; z -2 or below is better (the threshold is the owner's to confirm).
2. **The guard:** the year-over-year team-game test (63) must not be worse at z +2.
3. **The 2026 top 20** (66).

`FULL_READINGS=1` adds the older readings below.  The swap step (5) is not in the folds, so the decider scores steps 1-4.

## The older test: year-over-year

Rate a season from its own games.  Predict every stint of the season before and after from the ten players'
ratings alone, refitting only the intercept and home edge on the scored season.  28 scored seasons, each
predicted twice, 56 observations.  The prior must not have seen the two scored seasons:

    .venv/Scripts/python -m openrapm build --setting test --name <name> <the experiment's rate flags>
        (steps 1-5 at the test setting for all thirty seasons in one run, ~15 min, no stitching: the labels and the
         prior's training rows leave out both scored seasons, step 4 uses the test rule on outputs/within/weightfix_within
         with the fold seasons PINNED to 2017-2026 (settings.ShrinkSettings; an unpinned rerun shrank harder --
         experiment 34's first run), and step 5's type model trains only on seasons more than one away.  Writes
         outputs/season_ratings_<name>_step3.parquet, _step4.parquet and outputs/season_ratings_<name>.parquet)
    .venv/Scripts/python scripts/checks/63_yoy.py --rankings=<name>=outputs/season_ratings_<name>.parquet,incumbent=outputs/season_ratings_weightfix.parquet --ref=incumbent --tag=<name> --splits=
    .venv/Scripts/python scripts/checks/64_consensus_report.py outputs/season_ratings_<name>.parquet outputs/season_ratings_weightfix.parquet
    .venv/Scripts/python scripts/checks/66_compare.py incumbent=outputs/season_ratings_weightfix.parquet <name>=outputs/season_ratings_<name>.parquet --season=2026 --top=20 --yoy=outputs/yoy_<name>.parquet --ref=incumbent
    .venv/Scripts/python scripts/checks/90_swap_test.py --rankings=incumbent=outputs/season_ratings_weightfix.parquet,<name>=outputs/season_ratings_<name>.parquet --contexts=nofatigue --checks=0 --tag=<name>
        (the swap test: does it order teammates better; 40 s)
    .venv/Scripts/python scripts/checks/101_log_run.py --name=<name> --exp=<number> --idea=owner --verdict=pending --changed="<one plain sentence>"
        (the experiment table, the owner's 2026-10-04 ask: one row per experiment in experiments/runs.csv, every number read
         from the outputs above; then mirror the row into the Google Sheet "OpenRAPM experiments",
         https://docs.google.com/spreadsheets/d/1gPeRCFUtJNPtsBq_j2CNiTTdrarb98Yf3zXYx974jfI, tab Experiments.  Re-run with
         --verdict=adopted|rejected|"not adopted" once the owner rules; the same --exp replaces its row)

**A target experiment** (a new defensive or offensive target) once added `--deal_target_def=x3def_w0.25` to 62 so the
box-score prior was trained on the same groupings of players as the incumbent's; without it the experiment also pays a
random regrouping worth about 0.05 on this test (control C1, 2026-10-07).  That flag did not survive the cleanup:
`openrapm rate` assigns each side's player folds from that side's own labels, so read a target experiment beside the
noise control (`--fold-assignment 1`).  `bash scripts/experiment_chain.sh NAME TARGET "EXTRA RATE FLAGS"` runs this
whole chain and the trade loss below.

**Decision rule:** adopt only if `z` is -2 or below with no gross consensus miss and the 2026 top 20 not
worse; ties go to the simpler version.  Read the row "each side rescaled to the scored season" to see
whether the ORDER improved or only the spread.  `scale_*` below 1 on a neighbouring season is expected
(players change year to year) and is not a target.

**Saved priors**: `--save_priors=<name>` then `--priors_from=<name>` reran the ridge alone in a second a
season, which is how the penalty was swept.  Neither survived the cleanup (`openrapm rate --save-models` saves the
boosters, which `openrapm folds` reads).

## The second criterion: the trade loss (new, 2026-09-18)

The year-over-year test is an error per team-game, so a 200-possession man is a rounding error in it.  The
trade loss counts every player once.  **It has already overturned one reading and confirmed six others.**

    .venv/Scripts/python scripts/checks/70_tradeset.py --rankings=outputs/season_ratings_<name>.parquet --team_effects=team --out=tradeset_<name>   (85 s)
    .venv/Scripts/python scripts/checks/73_tradeloss.py --alphas=incumbent=outputs/tradeset_weightfix_alpha.parquet,<name>=outputs/tradeset_<name>_alpha.parquet --ref=incumbent

Paired by season on the exact intersection of eligible players; `--tier=` splits by exposure.  Read
`mean_diff` below zero as better and `z` against its own standard error.  It detects a defensive change of
0.0024 at z -2.2, so it has power at the size that matters.

**Every experiment is also split by player-quality tier: top 30, 31-90, 91-150, 151-300, 301+** (the owner,
2026-09-30: "in general we should do this").  Both tests, tiers set by the incumbent's rank in the rated season:

    .venv/Scripts/python scripts/checks/88_yoy_by_player.py --cands=<name>=outputs/season_ratings_<name>.parquet --ref=incumbent=outputs/season_ratings_weightfix.parquet --tag=<name>
        (the year-over-year difference shared among the players on the floor and summed by quality tier, by team
         change and by age; the groups add up to 63_yoy.py's number exactly; ~3 min)
    .venv/Scripts/python scripts/checks/73_tradeloss.py --alphas=<as above> --ref=incumbent --quality=outputs/season_ratings_weightfix.parquet --tier=each

Direction names in both year-over-year scripts: "prev" / "the PREVIOUS season's rankings predict this season"
is a rating looking FORWARD at the season after it; "next" is a rating looking BACK at the season before it.

## The record so far (year-over-year error per team-game, points per 100)

| rankings | error | verdict |
|---|---|---|
| shipped before this branch (`artifacts/`) | 8.587 | uses the banned `past_*` channel on offence |
| career row only, one per player | 8.804 | the starting point |
| + 1-3 season chunks (owner's design) | 8.736 | adopted |
| + scale cross-fitted, penalty not | 8.693 | adopted |
| + out-of-player priors | 8.705 | worse on the test, adopted by ruling 2 |
| + fixed penalty 13,037 | 8.697 | tie on the test; **the trade loss later read it z -2.19 on defence** |
| + the DEFENSIVE label un-shrunk (the incumbent until 2026-10-01) | **8.682** | z -5.24, 43 of 56; trade loss defence z -7.14, 27 of 30; adopted 2026-09-18 |
| 25: chunk rows labelled from the seasons OUTSIDE the chunk | 8.700 | z +5.0; trade loss z +4.5 / +5.0; rejected 2026-09-29 (25b, weighting those rows by their label's possessions: much worse) |
| 26: + RAPM pieces, same-team measure, game difficulty, rated as if every player changed teams | 8.682 | a tie (z +0.1); trade loss z +3.5 / +5.4; not adopted |
| 27: + a team-season random intercept | 8.679 | a tie (z -0.4); trade loss z +3.8 / +3.9; not adopted.  26 and 27 predict the season BEFORE better (z -2.5, -2.8) and the season after worse: an age tilt (veterans up, young players down), not a team-change effect |
| 28: + every chunk's label moved to its own age (ageing curve) | 8.721 | z +4.1; trade loss z +5.1 / +4.1; rejected: over-corrects (Curry 92nd, Durant 136th), though consensus agreement rises to 0.818 |
| 29: the incumbent + the team-movement weight, minimum 0.5, rebalanced within career bands | **8.669** | **z -4.1, 39 of 56**; trade loss a tie; consensus unchanged; 2026 top 20 close to the incumbent's.  Meets the adoption rule; awaiting the owner (it halves one-team players' weight, which they said they do not want) |
| 30: windows of two, four or six seasons split in the middle, each half labelled by the other; same-team measure as an input; rated as if every player changed teams | 8.706 | z +3.4, 16 of 56; trade loss z +2.8 / +7.6; rejected 2026-10-01.  Order alone leans better (z -1.6) but the ratings are too wide; best consensus agreement yet (0.829 total); loses as much on players who changed teams as on those who stayed.  LeBron James 126th in 2026 |
| 30b: experiment 30's models with the trade flag switched on only for the players who changed teams; and every player rated as if he stayed | 8.682 (matched) / **8.680** (stayed) | the flag does not help movers: matched against all-stayed a tie (z +0.8).  All-stayed against the incumbent: a tie (z -0.3) but stint level z -5.1 (mostly the bench), trade loss offence z -4.4 / defence z +4.1, best consensus yet (0.864).  Not adopted under the rule; owner's call.  Durant 72nd, LeBron 65th in 2026.  `62 --rate_same_team=both`, `89_stitch_by_move.py` |
| the swap adjustment, by player type at half strength (2026-10-01) | 8.698 | z +3.5, 16 of 56: worse at native scale; each side rescaled -1.02, z -10.0, 52 of 56 (the order is much better, the spread ~12% / 27% wider); swap test order +0.16, z +5.5; consensus 0.832 (from 0.794), top five 5 of 5; trade loss offence z -5.3.  Not adopted under the rule; next proposed: the same with each side's spread held to the incumbent's |
| the swap adjustment with the spread held to the incumbent's: teams_fixed / list_scaled (2026-10-01) | **8.666 / 8.652** | **z -5.2, 39 of 56 / z -7.7, 48 of 56**; better in every quality tier; consensus 0.836 / 0.834 (from 0.794); swap test order z +4.3 / +4.6 and gaps better; trade loss offence z -5.7 / -5.3, defence z +2.0 / a tie.  Both meet the rule.  **teams_fixed ADOPTED 2026-10-01 (the owner: "team version") -- the incumbent**, `outputs/season_ratings_swapadj_within.parquet`.  `91_swap_adjust.py --kappas=0.5 --taus= --hold_spread=within|whole` |
| all four on- and off-court ratings in both priors, on top of the incumbent (2026-10-01) | 8.667 | a tie (z +0.4); order-only worse (z +2.6); consensus 0.814 (from 0.836); Ajay Mitchell 9th to 7th, Caruso and Hugo González up ~0.9.  Not adopted.  `--features=boruta_onoff` |
| on/off alone in both priors, in place of the raw on-court ratings (2026-10-01) | 8.704 | z +6.9, 12 of 56: worse (order a tie, the offensive spread wider); trade loss offence z +5.6; Ajay Mitchell 9th to 6th.  Not adopted.  `--features=boruta_net` |
| the prior as a stack (2026-10-02): on- and off-court plus-minus into an elastic net, the other 50 columns (Boruta kept all, `92_stack_boruta.py`) into chimeraboost quality=3, blended on out-of-player-fold predictions | 8.677 | z +1.8, 23 of 56: worse; stint level z +8.4, order-only z +12.7; trade loss offence z +5.0 / defence z +7.8, worse in every tier but the top 30; swap test order worse (offence z -6.8, defence z -4.8); consensus 0.876 total / 0.859 defence (from 0.836 / 0.813); only the top 30 better (z -2.7).  2026: Zach Edey (590 poss) 8th, Keshad Johnson (621) 12th, Caruso 5th; Queta, Clingan, Harden out of the top 20.  Not adopted.  `62 --features=stack --stack=1 --stack_quality=3` (`stackprior.StackedSPM`).  Why (2026-10-03, `93_prior_oof.py`): trained on career labels its elastic net's one-season predictions are twice too wide, and even rescaled the stack orders players worse than the shipped booster (z +3.2 / +3.2) |
| 33b: the box-score prior part times a held-out multiplier per side (offence x0.72, defence x0.95), fitted on each season's own held-out games (within-season folds 2017-2026; 1997-2016 borrow the pooled value) (2026-10-03) | **8.600** | **z -12.2, 53 of 56**; every era better (1998-2006 18 of 18, 2007-2016 19 of 20, 2017-2025 16 of 18); swap test order z +3.1 (40 of 56), gaps z -6.2; trade loss offence z -3.1, defence z +5.5 (0.0007 on 0.248); stint level with each side rescaled z +2.1; consensus 0.834 (from 0.836), top five 5 of 5.  2026: Curry 21st, Harden 31st, Durant 41st (product).  **ADOPTED 2026-10-03 (the owner: "Good, adopt for now") -- the incumbent**, `outputs/season_ratings_priorshrink.parquet`.  `99_prior_shrink.py`; the trees of experiment 33 not a candidate |
| 34: teammates' shot mix with him on the court (2026-10-04) | 8.599 | a tie (z -0.2) once the shrink's folds are pinned; the owner's call pending, recommend not adopting |
| 35 / 36: baselines -- B3, RAPM on a linear box-only SPM prior; B4, RAPM on OpenRAPM's own prior; both tuned on other seasons (2026-10-05) | **8.587** / 8.609 | B3 z -2.6 (better, every era) and orders teammates better; B4 z +2.7.  Reference rows; B3's recipe as the base is an open option |
| 41: the swap adjustment's give-back taken flat instead of by minutes (2026-10-06) | 8.601 | z +2.2, 18 of 56: worse; stint level z -7.9 (mostly the bench); trade loss worse (defence z +4.3, top 30 offence z +3.9); consensus 0.839.  Not adoptable.  No give-back at all: z +13.2 |
| rejected | | every chunk size; booster settings; off-court features; one row per player-season; cross-fitting the penalty; the un-shrunk label on BOTH sides (the 2026 offensive spread collapses 1.60 to 1.14, Curry falls to 61st); `onc_d` off the defensive list (the trade loss finds nothing) |

## THE FINDING to carry forward: OpenRAPM is about a sixth too wide, and three sources agree

Full numbers in `DECISIONS.md`, "The amplitude finding".

| source | offence | defence | what it measures |
|---|---|---|---|
| the consensus, GLS scale | x0.85 | x0.85 | the same season, against public metrics |
| the trade set, three-season window | x0.749 | x0.919 | adjacent seasons' team-games |
| the year-over-year sweep, interior optimum | x0.65 | x0.85 | the neighbouring seasons' games |

**One sign, three independent readings.**  There is no offence/defence imbalance -- an earlier claim of one
was an artefact of the consensus being scaled to EPM, withdrawn the same day.

**A rescale is still not adoptable** (experiment 20): the two across-season sources cannot separate "too
wide" from "players regress", and a uniform rescale reorders nobody -- the order-only row is zero to 4e-14.
So this wants fixing INSIDE the fit, not bolted on afterwards.

**The lead that goes with it: the high-usage creators.**  Six of the eight highest-usage players sit below
their consensus offence after the spreads are matched -- LaMelo Ball -1.70, Ja Morant -1.61, Luka -1.59,
Jokic -1.50, Booker -1.06, Giannis -0.83 -- while the correlation with usage over all 391 players is
**-0.00**, so it is the extreme top and not a gradient.  The trade set independently found **shot creation,
three-point rate and offensive rebound share are under-credited by the box prior**, and shot creation is
what this group is.  Not the explanation, measured and dismissed: age, experience, team quality; the whole
per-36 box profile explains 21% of the player-by-player gap.

**Also named and unexploited: on defence the prior reads team offence as defensive credit** -- a player
whose team scored while he was on the floor has his defensive correction pushed down 0.062.

## What is closed.  Do not reopen without a new reason.

- **The exposure correction** (the prior is flat in exposure, the truth is steep).  Six attempts, none beat
  the incumbent; the owner: *"whatever our version is as last posted seems to be good enough."*
- **The trade set as a product.**  Alpha reads neighbouring seasons, so it can never be published under
  ruling 1, and a season's statistics see 3.7% of it on offence and 5.6% on defence.  It is an instrument.
- **`onc_d` off the defensive feature list** (experiment 19): the trade loss finds nothing (z -0.40) where
  it detects the penalty change at z -2.19, and agreement fell against every public defensive metric,
  furthest against the luck-adjusted ones, so it was not the `def3` pattern that excuses a drop.  Ruled:
  keep `onc_d`.  Note the 2026 top 20 was NOT the reason -- to the eye it was arguably better.
- **The amplitude as a rescale** (experiment 20).  See above.
- **The swap adjustment's give-back rule** (experiment 41): flat loses at the team-game level and on the trade
  loss; no give-back loses everything.  The minutes rule stays.
- **The portable rating as a replacement** (experiment 40): given to same-team players it is worse (t +5 to +8).
  It is a second number beside the team rating.
- **The trade flag as a rating-time switch** (experiments 26, 30, 30b).  Rating everyone as if traded loses;
  switching it on only for the players who really changed teams ties switching it off for everyone (z +0.8).
  The flag learns who gets traded (veterans, often declining), not what a trade does.

## The instruments, all read-only

Since the cleanup, 63, 64, 66, 70, 73, 88, 90, 95, 101, 113, 135, 137 and 138 live in `scripts/checks/` under the
same names; 91 is `openrapm swap`, 97 is `openrapm folds`, 76 is `openrapm site --bias`; the others in this table
are retired (tag `archive/research-2026-10`).

| script | what it answers |
|---|---|
| `63_yoy.py` | the criterion: team-game error, both directions, paired by season |
| `70_tradeset.py` + `73_tradeloss.py` | the per-player loss that counts a bench player once |
| `74_consensus_bars.py` | how far OpenRAPM sits from the consensus in units of its own uncertainty, each side scaled |
| `75_amplitude.py` | a per-side multiplier sweep on a finished table |
| `76_bias_groups.py` | the published page: bias by player type, `vs consensus` and `vs 2026 observed` |
| `64_consensus_report.py` | rank agreement and spreads for several tables side by side |
| `88_yoy_by_player.py` | the year-over-year difference split among the players on the court, by quality tier, team change and age; `tg_abs` gives each group its own typical miss |
| `89_stitch_by_move.py` | the trade-flag test's list: each player's "traded" or "stayed" rating (`62 --rate_same_team=both`) by what he did next; score it with `63_yoy.py` / `88_yoy_by_player.py` as `name=<fwd>|<bwd>` |
| `73_tradeloss.py --movers=1 --tier=each` | the trade loss for players who changed teams the season before or after, and for those who did not |
| `90_swap_test.py` | the swap test: does a ranking order teammates the way the games of the seasons either side do -- every pair of lineups that share four players, the two swapped players' rating gap against their swap difference (DECISIONS.md, "The swap test") |
| `91_swap_adjust.py` | the swap adjustment: re-split each team's credit by its lineup swaps (a type model, then each player's own swaps), team totals fixed; scores a grid on the swap test and writes the chosen arm (DECISIONS.md, "The swap adjustment") |
| `95_miss_by_group.py` | where the ratings miss most against expectation: each player's trade-set correction, adjusted for his rating level and the evidence behind it, by fifths of every statistic (plain names, `features.meaning`); the owner's variance question, DECISIONS.md "Where the ratings miss most" |
| `97_within_season.py` | within-season folds, every input rebuilt from the rating games (`WITHIN_SEASON_LEDGER.md`); `--split=deadline` rates before the date 60% of the regular season was played and scores after, and the reverse |
| `101_log_run.py` | the experiment table (`experiments/runs.csv`, mirrored to the sheet's Experiments tab); `--sheet` prints the rows |
| `102`-`106` | the scorecard baselines (B1 linear box SPM, B2 vanilla RAPM, B3/B4 RAPM on a prior), the scorecard itself (`104`, error and slope kept apart), held-out swaps (`105`), OpenRAPM as shipped on every fold (`106`) |
| `107` / `108` | the box prior fit on held-out games: exact quadratics per penalty pair (`107`; `--traded=1` splits traded players' contributions) and the selection (`108`) |
| `110` / `111` / `112` | in-season vs other seasons; the traded-player check; the portable rating (fit, cross-fitted tests, production ratio in `outputs/portable/production.json`) |
| `113_swap_explain.py` | one team's swap adjustment step by step, checked against the shipped table |
| `93_prior_oof.py` | a prior's out-of-player-fold error on one season's training rows (dumped by `62 --chunk_label=outside --dump_rows=`), scored on the one-season rows against labels that share no games with them; ~6 min for four priors.  A screen, not the test |

`outputs/bgmm_proba.parquet` carries `player_id`, `season`, the winning player type and **all eight mixture
probabilities** for 2026, from a mixture fitted on 2023-2025 (one row per player-season) and used to place
2026 out of sample, standardised by the training seasons' own constants.  **The owner wants those eight
columns available to the prior code; that work has not started.**

Whether the two deltas on that page are related was the test the owner set for building position-level
categories: across the eight types, `vs consensus` against `vs 2026 observed` is **r -0.04** (the
2026-fitted types read +0.03).  No relationship, so position categories are not warranted by it.

## The site

`docs/index.html` (rankings, newest season first), `docs/bias.html` (bias by player type) and
`docs/decompose_openrapm.html`, the "Decomposition" tab (2026-10-13: every published rating split, offense and defense,
into box score + on court + off court + team SOS + swap adjustment; PIPELINE.md, "The OpenRAPM decomposition page"; a
toggle at the top of both decomposition pages switches to `docs/decompose.html`, the vanilla page: vanilla RAPM split two exact ways: by player, on-court rtg + teammates + opponents + context +
ridge penalty; by possession, on court signal + off court adjustment (GP) + off court adjustment (DNP) + team SOS adjustment,
with the actual on and off court ratings beside them; single seasons
and the ten three-season windows, penalty 3,000), and `docs/similarity.html`, the "Similarity" tab (2026-10-12: each
player-season's nearest player-seasons of other players, by the distance between their standardised inputs; opens on
Gary Payton II 2022; PIPELINE.md, "The similarity page").  Rebuild with `.venv/Scripts/python -m openrapm site --bias
--decomposition --openrapm-decomposition --similarity` (once scripts 52, 76 and 83), commit, and fast-forward `main` to
publish, on the owner's word.  The similarity and OpenRAPM decomposition pages read the final ratings, so they are
rebuilt after every ratings build.
**Check the Pages build after pushing** (`gh api repos/bbstats/openrapm/pages/builds/latest`): on 2026-10-08 the push
to `main` did not start one, and `gh api -X POST repos/bbstats/openrapm/pages/builds` had to request it.  The
tabs at the top and the dark-mode button are `docs/site.css` and `docs/site.js`, shared by all four pages (`openrapm site
--bias` writes them into bias.html); a new page needs the same `<nav class="site">` block on every page
(`tests/test_site.py` checks they match) and its colours given again under `:root[data-theme="dark"]`.  bias.html's
nav can be rewritten without refitting the mixture: `site.bias_page(json.loads(outputs/bias_groups.json))` rebuilds
the page byte for byte.  The
bias page is deliberately a title, one table and two short paragraphs; the owner has trimmed it twice.
**Built, not published (2026-10-06):** a sortable Portable column on the rankings page, from
`outputs/portable/production.json` applied by `52_site.py` to the product table (`portable.portable_table`; both
deleted in the cleanup pass, kept at tag `archive/research-2026-10`).  **Never
add explanatory prose to a page** -- the owner rejected the drafted footer as "AI slop"; ask the owner for the wording.

## Traps that cost a day

1. **The consensus file was rescaled twice on 2026-09-18** -- once to make it a real consensus rather than
   one dominated by collinear raw-points metrics, once to turn off the scaling to EPM.  **No consensus
   spread or agreement figure is comparable across those changes.**  Both rescales moved `adj_*` for ~550 of
   582 players at correlation 0.99+, so a number can look like the same measurement and not be.
2. **`defense` in the rankings parquet is points allowed, negative-good.  `rating_def` is the positive-good
   column**, and `rating_total = rating_off + rating_def`.  Adding `offense + defense` gives a
   plausible-looking table with the defensive sign inverted.
3. **An uncentred table cannot be differenced against a centred one** without removing the level first.
   Anything built before 2026-09-14 predates the centring rule; 4b sits at +1.61 on offence in 2026.
4. **Stint level and team-game level disagree in this project**, and have given opposite signs on
   amplitude.  Say which one a number is.
5. `pip install -e` puts this working tree on `sys.path`: a fresh clone silently tests THIS repo's data.
6. Tests cannot reach the network (`tests/conftest.py`).
7. A score on a mask that cuts team-games is not a score; read stint-cutting splits on `mse`.
8. Two builds at once, or an unpinned numba pool, and a 15-second fit takes an hour.
9. `TaskStop` on a chain stops the bash wrapper only; its python children keep running.  Check
   `Get-Process python` before starting anything.
10. Seventeen more in `DECISIONS.md`, "The measurement traps".
11. chimeraboost 0.34 changed no rating but made `shap_values` exact interventional TreeSHAP (~200x slower):
    a Boruta trial on 33,000 rows took 30 minutes.  `gbdt_prior.run_boruta(explain_rows=1000)` fixes it.
12. Script 62 pins BLAS to one thread, so a 6,000-column label solve takes 5 s; outside labels need ~300 a
    season and are solved by Cholesky on `--label_threads` cores (0.6 s each) for that reason.
13. Twenty-three traps now: 22, a re-fitted constant read from a folder that later grew (pin it; 99's
    `--fold_seasons`), and 23, scoring traded players by weighting team-game rows (it scores the other nine;
    split the prediction by player).  Memory "era-coefs measurement traps".

## The LaMelo star-guard test: accepted as failing on 2026-09-18, passing since 2026-10-08 (experiment 45)

`tests/test_vs_consensus.py::test_star_guards_are_not_buried` puts LaMelo Ball at rank 187 against a
hand-set ceiling of 160 on the published table (162 before the prior shrink of 2026-10-03, which moves every
offence-first guard down).  The numbers below are from the 2026-09-18 ruling.  **In points that is 0.016 per 100** -- he sits
0.168, the player at rank 160 sits 0.184 -- six times smaller than the owner's threshold for nothing, and
the check is a rank ceiling, the instrument the owner has ruled does not measure size.  The owner: *"that
test can fail, no worries."*  The honest fix, not done, is to replace the rank ceiling with a points check.

## Where to start next

1. **Get the owner's calls** on the open list in START HERE; commit and publish only when the owner says so.
2. **If the owner says go: the stat-variance plan** (START HERE, "Next"): each input's noise constant, posterior-mean
   inputs, and steals out of the defensive inputs; read the 95 steal lean after each.
3. **Closed by the 2026-10 cleanup:** the attribution control (chimeraboost on `boruta_noonc`) and leave one player
   out on experiment 45 (`--lopo=1`).  `openrapm` fits only the shipped LightGBM prior (`params/booster_params_*.json`
   must name the LightGBM learner) and has no leave-one-player-out option; either needs the research code at the tag
   `archive/research-2026-10`, and only on the owner's go.
4. **The swap adjustment's type model** for White-type cases (item 6): its strength or its features, scored on the
   standard battery plus the per-player consensus pieces (`113_swap_explain.py`).
5. Older leads, unchanged: credit shot creation in the prior; the eight mixture probability columns into the prior; a
   standard error per player for alpha.

## Verify you are where this file says

    .venv/Scripts/python -m pytest -q                                        # 464 passed, 1 xfailed, 6 slow deselected (~1 min)
    .venv/Scripts/python scripts/checks/63_yoy.py --rankings=incumbent=outputs/season_ratings_weightfix.parquet --tag=verify --splits=
                                                                             # game_armse 8.5857, both directions together
    .venv/Scripts/python scripts/checks/64_consensus_report.py outputs/season_ratings_product.parquet
                                                                             # n 475; rho 0.85 / 0.781 / 0.806; spreads 0.816 / 1.155; top5 4
    .venv/Scripts/python scripts/checks/113_swap_explain.py --base=season_ratings_product_weightfix_priorshrink_pre_swap
                                                                             # "reproduces the shipped swap adjustment for every 2026 player: max difference 2.93e-14" (the default base, season_ratings_product_step4, is what
                                                                             # `openrapm build` writes)
    .venv/Scripts/python -m openrapm site --out-dir <scratch>/docs --built 2026-10-09
    cmp docs/data/ratings.json <scratch>/docs/data/ratings.json             # no output: byte-identical (14,578 rows over 30 seasons)
