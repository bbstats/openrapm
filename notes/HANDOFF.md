# Handoff

**What this file is.**  Where the work stands, what waits on the owner, and the commands a session runs.  Rewritten
on the afternoon of 2026-10-22, its status and next steps on 2026-10-23 (twice), 2026-10-24, 2026-10-26, 2026-10-27, 2026-10-28, 2026-10-29, 2026-10-30 and 2026-11-01; the earlier file is in git history.  `DECISIONS.md` is the permanent record and carries
every number quoted here; read its measurement traps before trusting a number.  `README.md` says what OpenRAPM is,
`PIPELINE.md` how each step works, `CONTRIBUTING.md` the writing rules, `CLAUDE.md` the owner's standing rules.

## Talking to the owner

- American English in everything shared: defense, offense, color.  Code names keep their spelling.
- No step numbers in messages.  The names: **box-score prior** (the LightGBM, also "preliminary prior"); **+/-** (what
  the season's games add on top of the prior); **swap adjustment** (the old after-the-fact correction from lineup
  swaps, off since experiment 82); **swap part of the prior** (experiment 82's line, learned from the same swaps,
  added before the games are weighed); **defense factor** (the shrink's extra factor on defense, 0.89 before 2017 and 0.84 from 2017; never "dial",
  the owner: "yuck").  Define each the first
  time it appears.
- Experiments by name in chat, numbers only in the record files ("the three-input prior", not "experiment 92").
- Answer exactly what is asked.  Results as tables, one quantity per column, no parentheses, one TL;DR sentence.
- Within-season WOWY is what counts, and a win must hold in every era: 1997-2006, 2007-2016, 2017-2026.
- **No jargon ever** (the owner, 2026-10-26: "'the players deal' is insane jargon.  No jargon ever").  In anything the
  owner reads: "how the box-score prior splits players into its five training groups", never "the deal" or "folds";
  "10 averaged versions, each trained on a random resample of players", never "bags"; "random draws", never "seeds";
  "full build", never "chain"; experiments by what they changed, never by number or code name; no formulas or letters.

## Where it stands, 2026-11-01: the fix pass ("Anything you'd fix?")

The owner asked what I would fix and answered four questions at 2 PM: merge main and push; the percentile pool of a
season in progress; the Power page's teams at 0; the swap part of the prior refitted as the next experiment.
DECISIONS.md, "The fix pass of 2026-11-01" and "Experiment 126", hold the record.

- **main is the working tree's code again** (51a20ad, pushed on the owner's word): main's src/ and tests/ had stopped
  at 2026-10-23, 148 commits behind this branch, which builds the site; CI's tests had failed on every push to main
  since the Power page (main's site test expected four tabs).  Merged with main's copies of the generated site files;
  CI green.  **At each publish, fast-forward main to this branch** (`git branch -f main HEAD`, push), never copy pages
  across through a temporary worktree again.  The site's pages and docs/index.html changes below reach the site only
  when main is pushed again: the owner's word.
- **The scheduled task fired at 5:00 AM since the clocks changed**: its trigger was a fixed instant (StartBoundary
  `2026-10-24T05:00:00-05:00`, 10:00 UTC).  Set to 6:00 AM local time (`schtasks /Change /TN "OpenRAPM daily update"
  /ST 06:00`); the next run 2026-11-02 at 6:00 AM.
- **The morning job** (8ef246c): a failed hustle dashboard pull now fails `openrapm data dashboards` (it returned 0,
  and the season in progress would have divided yesterday's totals by today's possessions); `hustle-games` is no
  longer a step, nothing the morning publishes reads the per-game files; `PUBLISH=0` writes the site's files to
  `outputs/daily/site_dry`, so a dry run catches what stopped the morning of 2026-10-31; a day with no new games
  publishes nothing (`scripts/daily_files.py` compares the three files with their built date masked; before, `git
  diff --quiet` never came back clean); the commit's date reads the playoff log too; a failed rebase is aborted.
- **The Power page** (9909e61): each player's rating on the page is shifted down by a fifth of the average team, so
  the teams average 0 and each player's row adds to his team's total (the rows missed it by 0.37); team totals
  unchanged to the hundredth; a player's numbers move -0.04 on offense and -0.03 on defense.  **The percentile pool**
  of a season in progress is 500 times the share of its regular season played, never under the 60 shown
  (`meta.percentile_min` in decompose_openrapm.json; 2027 today 60, where one player of 334 had qualified).
- **A full build holds the season in progress at the fresh scales of the finished seasons it has just rated**
  (66b06f5), not the previous build's table: after an adopted change 2027 moved twice.
- **Fold tools as production** (4101702): the season so far's share floored at 0.25 (the least the training rows
  cover), `--blend-from` shipped at 0.25, `--hold-prior-scale` drops the season in progress, a chain resumed with
  `REUSE=1` repeats the reproduction check when the base table is newer than its record.
- **Checked, not bugs**: the hustle dashboard's games and minutes fall short of the game log for a fifth of 2027's
  players (DECISIONS.md, the last measurement trap: the league lists a player in a game's hustle box score only when
  he had a hustle event, so the counts are complete and never to be scaled by minutes); 84 roles rows of 2027 without
  an age are roster players without a stint, and age is not a shipped input.
- **Experiment 126, the swap part of the prior refitted on today's boosters and folds, ADOPTED and PUBLISHED
  1:18 PM (e3405b4; the owner: "Adopt and publish")** (DECISIONS.md, "Experiment 126"): the shipped line `ctx82` came
  from experiment 78's (2026-10-17), before the parser fixes, the body listing, the part-season rows and
  self-creation.  The line `ctx110`: within-season WOWY -0.237 in 1997-2006 (z -3.06, 8 of 10), -0.263 in 2007-2016
  (z -3.03, 9 of 10), a tie in 2017-2026 (-0.014, z -0.23, 5 of 10), all seasons -0.171, z -3.61, the noise control
  -0.017; year over year a clear win, team-game -0.157, z -4.08, both directions.  Now shipped:
  `PriorSettings.second_prior = "ctx110"`, the shrink's folds `ctx110_early,ctx110_within`, tenth folds `ctx110_k10`;
  the full build of 1997-2027 with 2027 held at 1.085 / 0.985 from this run's own 2019-2026.  Against the morning's
  published ratings a player with 60+ possessions moves a median 0.24 per 100 in every era, the 90th percentile 0.6,
  the largest 1.9; the 2026 top 20 keeps its names (Jokić +0.7, Wembanyama +0.5, Kawhi +0.5, Cason Wallace -0.4).
  **The chain's incumbent is ctx110** (tables `season_ratings_ctx110*`, folds `ctx110_early`, `ctx110_within`); its
  noise control `noise_ctx110` (`--fold-assignment 1`, 1:18 to 2:15 PM) reads -0.013, z -1.63, 17 of 30, the same
  size as creation110's; the chain's `INC_NAME`, `INC_SYSTEM`, `INC_FOLDS` and `CONTROL_NAME` defaults and
  CONTRIBUTING.md name them now.  Delete `prior_models_creation110.pkl` only if its folds are never extended again.
- **The dry run of the morning job passed end to end at 9:31 AM** with every change of the day, 2027 held at
  1.116 / 1.000, the site's files in outputs/daily/site_dry (2027's percentile pool 60, every team's rows adding to
  its total within 0.011).  The next real run is 2026-11-02 at 6:00 AM.
- **Waiting on the owner, wording on the site** (I write no prose there): the About page says "Share of league games
  played" where the prior reads it floored at a quarter for the season in progress (`box_prior.rated_rows`); the box
  per-100 inputs are centered on the season's average player (`features.py`, `SHIPPED`) while the page says "vs
  league average" only for the hustle rows; its line "All stats use padding method" stands against ten "Not padded"
  rows; the Power page's intro lacks the latest-5 double weight and the average team; the changelog dates the blend
  entry 2026-10-31 though it first reached the site on 2026-11-01.
- **Open**: the settings check for the morning job (main now equals the working tree, which removes most of the
  risk).  **June 2027**: once the Finals end the job must not keep replacing 2027 alone, since every season's labels
  change; a full build, after `openrapm data hustle-games 2027 2027 --force` and `matchup-games ... --force`; a guard
  that refuses to run once `data/seasons.py` calls the season finished is proposed, not built.

## Where it stands, 2026-10-30 night: the bug-fix and make-better pass

The owner: "Bugfix/make better mode. Go."; four answers at 9 PM, all the plan's recommendations.  DECISIONS.md, "The
bug-fix and make-better pass of 2026-10-30", experiment 109 continued and experiment 123 hold every number.

- **2027's hustle inputs, a live bug since the daily job began, fixed** (6195755, 186424b): the 6 AM job cached the
  league's placeholder hustle box scores, 55 of 71; and each game's box score fills in over days, so the season in
  progress now reads the league's season-to-date dashboards.  Its box scores are never cached empty
  (`dashboards.NotProcessedYet`).  **When 2027's Finals are over: `openrapm data hustle-games 2027 2027 --force`
  (and `matchup-games`) before its panel.**
- **The held prior scale, adopted** (experiment 109, its continuation and, 2026-10-31, the line): the season in
  progress holds its prior scales at the 8 nearest finished seasons' mean through a quarter of its regular season, then
  a straight line to its own fitted ones at three quarters (`RidgeSettings.held_scale_blend`, `held_scale_blend_from`
  0.25, `held_scale_through` 0.75; the owner: "If it's just two different shrinkages we should probably just blend
  between them").  The line from the first game lost at a tenth and a quarter; from a quarter it ties holding at half a
  season and beats fitting.  2027 today: 1.116 / 1.000 held, against 1.72 / 1.83 fitted on 71 games.
- **What the 6 AM job of 2026-10-31 publishes**: 2027 with both; the first morning these reach the site.  On the
  same 71 games against the published 2027, players with 60+ possessions move a median 0.63 per 100 in total, the
  largest 5.20; the top 20 keeps 18 players, Anthony Edwards and Paul George in, Kyrie Irving and Kris Dunn out:

| player | offense | defense | total | published total | published rank |
|---|---|---|---|---|---|
| Cooper Flagg | +4.31 | +1.72 | +6.04 | +9.44 | 1 |
| Luka Dončić | +6.49 | -0.52 | +5.97 | +8.94 | 3 |
| Victor Wembanyama | +2.59 | +3.27 | +5.86 | +9.08 | 2 |
| Stephen Curry | +5.14 | -0.58 | +4.57 | +7.27 | 4 |
| Ty Jerome | +5.00 | -0.79 | +4.21 | +6.12 | 5 |
| Giannis Antetokounmpo | +3.97 | +0.11 | +4.08 | +6.05 | 6 |
| Deni Avdija | +3.14 | +0.27 | +3.41 | +5.30 | 7 |
| Tyrese Maxey | +3.33 | +0.03 | +3.36 | +5.16 | 8 |
| Ajay Mitchell | +2.72 | +0.59 | +3.31 | +4.80 | 12 |
| Jayson Tatum | +3.67 | -0.36 | +3.30 | +5.00 | 9 |
| Dyson Daniels | +0.89 | +2.24 | +3.13 | +4.99 | 10 |
| Moussa Diabaté | +1.36 | +1.65 | +3.01 | +4.96 | 11 |
| Alperen Sengun | +2.28 | +0.65 | +2.92 | +4.59 | 13 |
| Nikola Jokić | +2.03 | +0.80 | +2.82 | +4.22 | 15 |
| Tyrese Haliburton | +3.32 | -0.52 | +2.80 | +4.06 | 18 |
| Shai Gilgeous-Alexander | +2.76 | -0.03 | +2.73 | +4.20 | 16 |
| Zion Williamson | +1.30 | +1.43 | +2.73 | +4.41 | 14 |
| Scottie Barnes | +0.97 | +1.74 | +2.72 | +4.01 | 20 |
| Anthony Edwards | +2.88 | -0.21 | +2.67 | +3.92 | 21 |
| Paul George | +2.05 | +0.58 | +2.63 | +3.86 | 23 |

- **Test tools fixed** (ccb3d9b): the Newey-West z is floored at the plain one, so the plain z rules; check 151 scores
  each system with its own shrink (`--shrink-flags`) and prints the eras; the full-build script refuses a name already
  built unless `REUSE=1`; `openrapm folds` records its boosters per folder; 101 logs the within-season test by era and
  refuses partial or stale readings; the input screen's added arms keep the inputs per tree.
- **Experiment 123, the steadier within-team multiplier, not adopted** (the registered rule): the within layer fitted
  on every fold season is better before 2007, z -3.2, a tie in 2007-2016, worse since 2017, z +2.6; year over year
  a tie, z -1.6.  The window's jumps carry era drift, not only noise.  `--within-nearest` stays, off.  A wider window
  or a pull part way toward the pooled fit would be a new experiment.
- **The sheet** "OpenRAPM experiments": six within-season WOWY columns after Result, rows 109 Adopted and 123; the
  Definitions tab names the within-season test and today's adoption rule.
- **Checked tonight**: `PUBLISH=0 bash scripts/daily_update.sh` passes with all of it (2027 held at 1.116 / 1.000);
  the site's files build, the power page's average team at 0.00, 13 of 30 teams above it.
- **The morning of 2026-10-31**: the 6 AM job stopped at the site step: the decomposition solved 2027 again with its
  prior scales free and its pieces missed the ratings by 0.13.  Fixed (032228c, 58e0d54), the job run again by hand.
  A `PUBLISH=0` dry run stops before the site step, so after a change to step 3 also run `openrapm site
  --openrapm-decomposition --power --out-dir <a scratch folder>`.
- **The Changelog tab is live** (8d1ca8d, the owner: "We need to put this in the changelog"; wording the owner's pick):
  two rows dated 2026-10-31, 2027's hustle stats and its prior weight until the season is half over.  New entries go
  in docs/changelog.html, newest first, one short row per published change, in wording the owner approves; publish
  through a temporary worktree of origin/main, as the morning job's own checkout is reset each day.
- **Open, the owner's call**: a settings check for the morning job (DECISIONS.md, "The rest of the pass": the job runs
  the working tree's code, and the published tables come from three commands); experiments 111-122's verdicts.

## Where it stood, 2026-10-29

- **Experiments 121-122, 2026-10-30, verdict the owner's** (DECISIONS.md): the box-score prior as a blend of the booster
  and an elastic net on all 61 screened inputs and every product of two (`linear_prior.py`), on check 155's quick screen
  (tags `blend121`, `blend122`: 2001, 2010, 2019, 2025, four held-back game sets, about 1.5-2 minutes a trial).  No share
  of the line helps: with the booster on what Boruta keeps, a tenth ties, +0.021, larger shares are worse; the booster on
  the 13 passes alone loses 0.33.  The line explains more of the long-run label than the booster, 0.62 / 0.42 of the
  spread against 0.57 / 0.26, and splits credit within a season worse.  Today's lists less defensive loose balls
  recovered tie, -0.009.
- **Experiment 120, DONE 1:42 PM 2026-10-30, verdict the owner's** (DECISIONS.md, "Experiment 120"; the owner: "let's go to boruta
  town ... hyperparameter tuning + boruta should probably always happen in tandem (or - if it shouldn't we should prove
  it)").  One driver, `scratch/2026-10-29_matchup_screen/run.sh`, log `run.log` beside it, every step resumes:
  1. DONE 3:20 PM: the matchup screen at the shipped settings, check 155 tag `screen120`, seasons 2018-2025
     (`outputs/csv/screen_screen120_table.csv`, sheet tab "Input screen 120"): nothing passes; on defense the real
     values match their shuffled copies within 0.001, and any 21st defensive input costs +0.022 against the shipped
     lists because the column share 0.367 then shows each tree 8 inputs, not 7;
  2. the booster settings searched again with the matchup inputs in, `tune --study tune120`, 80 trials
     (`outputs/screen/screen120/tune120.db`, `params/booster_params_tune120.json`).  The first run stopped at 4:45 PM
     on a trial whose boosters were too large to pass between processes; restarted 5:54 PM with the boosters passed
     through files and trees capped at 24 leaves and 1,500 rounds, log `run2.log` (DECISIONS.md, "the tune's first run
     stopped");
  3. DONE 5:08 AM 2026-10-30: the tune's best, trial 53, read -0.110 on its own ten seasons; the matchup screen at its
     settings (`screen_screen120_tune120_tandem.csv`, sheet tab "Tandem 120"): no verdict differs, two readings move
     beyond z 2; today's lists at the retuned settings on 2018-2025 +0.011, z +0.27, and +0.060 on the five seasons the
     search never scored;
  4. DONE: experiment 119's ten counts at the tuned settings (`screen_screen119_tune120_tandem.csv`): no verdict
     differs.  The answer: across 18 inputs the retune changed no verdict; three readings moved beyond z 2, all toward
     their copies; screening at today's settings stands, and a retune is its own experiment, judged by a full build on
     the twenty seasons the search never read.  The tune's settings are not adopted.
  Do not edit `src/` or `scripts/checks/155_input_screen.py` until it ends: each step starts fresh processes.  To stop
  it, kill the bash driver and its python children.  When it ends: the tables to the sheet, the reading to DECISIONS.md,
  the experiment's row (`101_log_run.py --exp=120`), the report to the owner with the 2026 top 20 only if a candidate
  passes and is built.

- **Matchup inputs, game by game: BUILT, not screened, nothing published** (DECISIONS.md, "Matchup inputs, game by
  game"; the owner, 2026-10-28: "the goal of this session is merely to build, not to test in the model yet").
  - `openrapm data matchup-games` fetches `BoxScoreMatchupsV3`, one file a game, 2018-2026 all on disk (11,503 games,
    about 4 hours; `--limit N` runs a long fetch as fresh processes, since one stalled after about 600 requests).  2027
    since 2026-10-29 (the owner: "did you grab the 2027 stuff?"): the daily job fetches it each morning, not as a
    step, so a failure cannot stop the publish, and the season in progress leaves out the games the league has not
    processed yet, usually last night's (at 7 AM on 2026-10-29 the 12 games of 2026-10-28 answered empty rosters).
  - `data/matchups.py`, `features.MATCHUP_GAME_INPUTS`: eight panel columns from the fit games' pairs alone, padded,
    on both rows: the height of the players he guarded and of those who guarded him, the self-creation and total
    creation of the players he guarded, and his share of defense on the opposing top self-creator and top total
    creator, of all his defense and of the floor he shared with them.  Total creation adds the owner's estimated
    potential assists; the feed's tracked ones are all 0.
  - Both panels rebuilt (183 and 185 columns), the 175 older columns unchanged to the bit.  2026 values on the sheet's
    tab "Matchup inputs 2026".  No input list reads them; `SCREEN_CANDIDATES_MATCHUPS` waits for a screen.

## Where it stood, 2026-10-28

- **Power ratings page, PUBLISHED 2026-10-28 (8820ba0, the owner: "Publish")** (`docs/power.html`, `openrapm site --power`, the owner's design
  2026-10-28).  Each team's current-season offense, defense and total: every player's rating times his share of the
  team's possessions over its last 10 games, blowout possessions (garbage time, config `garbage_time`) counted half;
  shares add to 500%.  Click a team for its players.  A Power tab on every page.  The daily job writes
  docs/data/power.json every morning now that power.html is on main.  Since df01906 the latest 5 games count double
  the 5 before, and the Power tab stays hidden until every team has played 10 games (meta.ready; site.js).  Published without the Changelog tab, which
  went on main on 2026-10-31 (8d1ca8d).
- **Published ratings unchanged: creation110.**  Nothing below is published or changes a default.
- **Experiment 119, play-by-play fouls, goaltends, blocked shots and counts over expected through the Boruta screen;
  ALL TEN FAIL, verdict the owner's** (DECISIONS.md, "Experiment 119"; the owner, 2026-10-28: "go for it and remember
  to boruta").  Closest: blocks over those expected from fouls, z -1.18 against its copy, helping 1998-2004 only.
  Steals over those expected from fouls worse in all 10 seasons, z +6.32.  The two turnover inputs worse within the
  season but better on the guard, z -2.62 and -2.87.  No confirming build.  Self-creation shuffled in place reads
  +0.0233, z +2.62, as in experiment 112.  Table on the sheet's tab "Input screen 119"; row 119 in experiments/runs.csv and on the
  sheet.
  - New per-game counts, `data/foul_counts.py`, cached as `data/event_counts/{season}_fouls.parquet` by `openrapm data
    events`: goaltends, shooting fouls, loose-ball fouls, the shooter's blocked shots, three-shot trips.
  - Seven counts over expected, the owner's form for anything that would be a ratio (`features.OVER_EXPECTED`,
    `data/panel.add_over_expected`): the count minus a possession-weighted line on the others through the season's
    players on one side, refit on a fold's games.
  - Goaltends start in 2002 (`features.GOALTENDS_FROM`): before it most rows name only the team.
  - Both panels rebuilt with the 12 new columns; the 160 older columns unchanged to the bit
    (`scratch/2026-10-28_fouls/panel_check.py`).  Cache `outputs/screen/screen119/` (24 GB): the shipped boosters
    refitted give back creation110's priors to 6.7e-16; seed 0's folds give back creation110's to 3.9e-13.
  - The run: `scratch/2026-10-28_fouls/run.sh`, log `scratch/2026-10-28_fouls/run.log`, 68 trials, 12:56 to 8:10 PM;
    table `155_input_screen.py table --tag screen119`, eras `scratch/2026-10-28_fouls/eras.py`.  Check 155 now takes
    its pool from the tag (`SCREENS`).
  - `add_over_expected` builds its seven columns apart and joins them once (7c459dc), so pandas no longer warns.
- **Experiment 118, the box-score prior from NVIDIA's pretrained Kumo Tabular network; verdict the owner's**
  (DECISIONS.md, "Experiment 118").  `src/openrapm/in_context_prior.py`: a params file naming the learner
  `kumo_tabular` puts the network where each LightGBM booster stands, the same training rows and five player folds, no
  bags (`--booster-params kumo_small --bags 0`).  Library structured-data-models 0.2.1 in .venv, the optional group
  `kumo` in pyproject.toml; the weights download from the Hugging Face hub on first use.
  - Fits the RTX 3070 easily: under 1.1 GB of GPU memory per process; one season's prior 1.6 minutes small, 3.6
    medium.  Two processes running medium at once filled the 8 GB and paged: Kumo runs take one GPU process.
  - Check 155, one random draw each: the within-season WOWY test after the shrink +2.89 small, +3.07 medium, +2.99
    with the rows drawn equally, z +11 to +13, worse in all 10 seasons and every era; the next-season check z +4.
    It rates the bench near average on offense and well below on defense, and the defense twice as wide.
  - No full build.  `openrapm folds` would need every rebuilt season asked in one pass, as check 155's
    `prime_in_context` does, or each held-back game set costs twelve passes of the network.
  - Tables `season_ratings_kumo118_2026*`, 2026 alone on today's shrink, for the eye check only.

## Where it stood, 2026-10-27

- **Published ratings unchanged: creation110.**  Nothing below is published or changes a default.
- **Experiment 114, the box-score prior game by game, built in full; verdict the owner's** (DECISIONS.md, "Experiment
  114"; report https://claude.ai/artifact/5G3TBE5FcyP7vCJDhc9M7Y).  The owner's multi-stat network handoff
  (scratch/2026-10-27_multistat/) on the real seasons: `openrapm data player-games`, `openrapm game-rates train/score`
  (30 networks under outputs/game_rates/net1/, trained on the RTX 3070; PyTorch's CUDA build is in .venv),
  `openrapm rate --game-prior net1 --part-season-rows 0.02,0.05,0.1,0.25,0.5,0.75`, the folds rebuilt game by game.
  - The network forecasts every stat better than the padding in every held-out season, -283.7 log-loss per 1,000
    player-games; the opponent adjustment is -38.4 of it.
  - As ratings: held-back team-games better in all 30 seasons (-0.64 at step 3), the within-season WOWY test worse in
    all 30 (+0.74 at step 4, +0.28 at step 3), year over year a tie.  Width is not the reason.
  - Tables `season_ratings_net1*`, folds `net1_early`, `net1_within`, boosters `prior_models_net1.pkl` (about 2 GB,
    delete when the experiment closes).
- **Experiment 115, the same both ways, the owner's idea; verdict the owner's** (DECISIONS.md, "Experiment 115"; same
  report, version 2).  30 backward networks under outputs/game_rates/net1_back/ (`--reverse`); each player's first
  half of games takes the backward prior, the rest the forward one (`--game-prior-backward net1_back`).
  - The slow start is gone.  Before the shrink it wins the within-season WOWY test in every era (-0.34, 26 of 30), but
    at a 10% narrower offense; after the shrink, at matched width, it loses in every era (+0.20, 5 of 30).  Held-back
    team-games -0.66, 30 of 30; year over year a tie, the season after worse at z +3.5.
  - Tables `season_ratings_net1b*`, folds `net1b_early`, `net1b_within`, boosters `prior_models_net1b.pkl`.
- **Experiment 116, the shrink's four fixed settings refitted; verdict the owner's** (DECISIONS.md, "Experiment 116";
  same report, version 3).  Check 156 (`scripts/checks/156_shrink_refit.py search|judge`) searched them for both ways
  and for today's ratings alike, on check 152's 10 tune seasons with the next-season check as a limit.
  - Both ways at its best still loses in every era of the 20 untouched seasons, +0.26 against today at its best, while
    wider than today's ratings: the game-by-game prior splits credit inside teams worse.
  - Today's ratings at their best gain on the within-season test in every era, -0.11, 18 of 20, but lose year over year
    at z +4.2: the search sat on its limit's edge on the tune seasons.  The shipped settings stay.
  - Tables `season_ratings_*_refit156_step4`, studies `outputs/tune/refit156_*`.
  - Waiting on the owner: the cheap check of the network's inner numbers against the prior's target, and whether to
    close the game-by-game prior; next proposed, the network's whole-season state as one prior per season.
- **Experiment 117, the box-score prior from the networks once a season; verdict the owner's** (DECISIONS.md,
  "Experiment 117"; same report, version 4).  `openrapm rate --game-prior net1 --game-prior-whole-season`: the
  network's state after each player's last game as the box-score prior's input, one prior a season.
  - Within-season WOWY +0.088 after the shrink, every era worse, a tie before it; held-back team-games -0.32, 28 of
    30; year over year z +2.1, just past the rule.  About half of both ways' loss and half its team-game gain.
  - Tables `season_ratings_net1s*`, folds `net1s_early`, `net1s_within`, boosters `prior_models_net1s.pkl`.
  - Open on the owner's word: the season in progress from the network's states; stats as if teammates were average.

## Where it stood, 2026-10-26

- **Published ratings: unchanged since experiment 110 (creation110), 2026-10-25.**  The 2027 daily job runs every
  morning on it (last: 2026-10-26, 2f65f94).
- **New since 2026-10-25, all committed, none published** (DECISIONS.md, "Experiment 111", "Experiment 112", "The
  noise control read closely"):
  - `src/openrapm/data/creation_counts.py`: a play-by-play reader that follows the ball (possession starts, offensive
    rebounds), every shot's spot and possession situation, free-throw trips and and-ones, and per-game counts.
    Built by `openrapm data events` (the daily job builds 2027's).  Its constants were fitted once,
    `scratch/2026-10-25_creation111/fit_constants.py`.
  - The panel has 163 columns: two uncentred unassisted parts, 21 uncentred self-creation counts and 8 candidate
    inputs measured from the average player (`features.UNCENTRED_EVENT_RATES`, `UNCENTRED_CREATION_RATES`,
    `CREATION_INPUTS`); every older column unchanged.  Backup before any of it: `outputs/season_panel_before_creation111.parquet`.
  - Experiment 111, self-creation with the passes estimated (`self_creation_estimated_per100`): a better measure,
    a worse input, not adopted.  Experiment 112: check 155 screened 13 play-by-play candidates (the owner's potential
    assists among them, `potential_assists_estimated_per100`); one passed, unassisted makes at the rim and in the
    paint; its full build gains in 1997-2006 only.  Both verdicts are the owner's; my recommendation is not to adopt.
  - **Check 155** (`scripts/checks/155_input_screen.py`): Boruta on the within-season WOWY test, rated through the
    real fold code; caches in `outputs/screen/screen112/` (25 GB, reusable while the panel and shipped settings stand).
  - **The first noise control on today's ratings, `noise_creation110`**, is now the chain's default control.
  - **The noise finding** (write-up https://claude.ai/artifact/RqJCDQZfMtZbbcAvAL8oGt): how the box-score prior splits
    players into its five training groups moves a regular's total by a median 0.05 per 100 and the within-season test
    by -0.017; averaging two splits beats today's by -0.013, z -2.70, and beats the two builds' average score by 0.004
    in all 30 seasons.  One build carries about 0.008 of avoidable error.  This is the next piece of work (Next).

## Where it stood, 2026-10-23

- **The bug-fix pass of 2026-10-22** (DECISIONS.md, "The bug-fix pass"): the owner asked for a hunt for bugs behind the
  artifacts; the findings, the code-only fixes, experiments 94-99, the width test and the season widths are there.
- **The chain's incumbent and the published ratings are experiment 110 (creation110), ADOPTED and PUBLISHED
  2026-10-25** (6cb55cd; the owner: "the rating changes are good - we can adopt and publish"): experiment 107 with the
  owner's self-creation on the offensive box-score prior (`--features hustle_no_three_pct_self_creation`), its own folds
  `creation110_early`, `creation110_within` and tenth folds `creation110_k10`; tables `season_ratings_creation110*`;
  DECISIONS.md, "Experiment 110 ADOPTED".
- **Before it, experiment 107 (parts107), PUBLISHED 2026-10-24** (42311b4): the box-score prior trained on part-season
  rows too, folds `parts107_early`, `parts107_within`, `parts107_k10`.
- **Before it, experiment 106 (layers106), PUBLISHED 2026-10-23** (the owner: "please adopt, yes - D. you may
  publish if 2026 our rankings correlate more nicely w/ consensus"; DECISIONS.md, "Experiment 106 ADOPTED and
  PUBLISHED"): inputs99 with the shrink in two layers, each player's deviation from his team's mean multiplied on its
  own, fitted on held-back stints within games, times 0.75 on offense and 1.0 on defense.  2026 defense over offense
  1.09 to 0.91; consensus agreement of the total 0.811 to 0.831 in 2026.  The chain's incumbent tables are
  `season_ratings_layers106*`, its folds still inputs99's.
- **Before it, experiment 99 (inputs99), PUBLISHED 2026-10-22** (the owner: "Publish").  On top of experiment
  94's parser fixes, published the same morning: the heave fix (95,
  threes of a period's last two seconds counted as they happened); the defense factor by era (98, 0.89 before 2017 and
  0.84 from 2017 in place of 0.866); each season's own body listing and the pad k around the bins (99).  The 2001
  Lakers moved most, back toward their usual split (DECISIONS.md, "Experiment 99 ADOPTED").
- **Experiment 101 (fullseason101), PUBLISHED 2026-10-22** (the owner: "we are going to do it", then "Publish"): each
  season's multipliers fitted on folds holding back a quarter and a tenth of its games (`inputs99_k10`, ten folds a
  repeat, `openrapm folds --folds-per-repeat 10`) and read to a full season off the line through the two: the
  games' part 5-10% smaller, the prior part 0-2% larger, the same in every era.  The year-over-year test +0.053,
  z +2.5, past the guard.  The chain judges candidates without it (`--extrapolate-folds=`) against inputs99's
  tables built the same way; an adopted candidate needs its own tenth folds, about 50 minutes.
- **The widths** (DECISIONS.md, "The widths"): the within-season WOWY test wants every rating wider, the
  year-over-year test narrower; a candidate that comes out wider wins one and loses the other for its width alone.
  Proposed, not built: every chain also reads the candidate at the incumbent's width.
- **The data on this machine**: the rebuilt stints, caches and season panel at the default paths; the old ones are
  `data/stints_before_parser94`, `data/cache_before_parser94`, `outputs/season_panel_before_parser94.parquet` and
  `outputs/season_panel_before_inputs99.parquet`.
- **New in the reports**: `z_clustered` in the year-over-year test and `z_newey_west` in check 151, beside the plain
  z; which rules is the owner's.
- **The season in progress, 2027, rated and published every morning since 2026-10-24** (Next, "The daily job").
- **The defense is wider than before experiment 82**: standard deviation 1.73 / 1.61 / 1.60 by era, 2026 1.64, against
  1.38 / 1.33 / 1.36 and 1.53 for experiment 80.  Against the consensus in 2026 the slope of our defense on theirs is
  about 1; the extra spread is players rated differently, carried by big-man stats: blocks, height, defensive
  rebounds, fouls (DECISIONS.md, "Experiment 82's defensive prior read through").
- **The defense read of 2026-10-23** (DECISIONS.md, the four sections from "Why the defense is wide relative to the
  offense" to "Should defense take a heavier ridge penalty"; scripts `scratch/2026-10-23_defense/`, tables
  `outputs/csv/defense_2026_*.csv`, the per-player table on the sheet tab "Defense 2026"):
  - Where the width is made: nothing before the shrink is defense-heavy, 0.92 in 2026; the shrink keeps 0.97 of the
    defensive prior part and 0.68 of the offensive since 2017, 1.30, and the defense factor brings it to 1.09.
  - At team level the defense is not too wide: held-back team-games want it wider, every metric's defense adds up to
    too little team defense in 2026, and our defensive order is LEBRON's at 0.90.
  - Within teams it is: from each season's own games, with a free level per playing-time band, ours keeps 0.89 of the
    true defensive spread among regulars and 0.69 of the offensive since 2017, where the less-known side, defense,
    should keep less.  The WOWY test's wish for a wider offense, 1.17 in 2026, points the same way.
  - The textbook heavier defensive ridge penalty is the bench's offensive gap passing for offensive spread; around
    each playing-time level the two sides are equally spread.
  - Experiment 102, both labels un-shrunk, taken apart with its control, experiment 103: the un-shrink made the
    offensive box-score prior a measure of playing time and cost the scorers, worse within season in every era; the
    line refit on today's data gave the big men their defense and the whole year-over-year gain.  The high-usage lean
    on offense does not come from the shrunk label.
- **The remaining known flaw: cross-side inputs.**  The defensive line reads his offensive box-score prior with weight
  0.91 beside 1.03 on his defensive one (2026, raw scale), copied from the swap adjustment's inputs.
- **Faster runs**: `openrapm rate --jobs N` and `openrapm folds --jobs N`, same numbers; the chain asks for 4, cut to
  what free memory holds.  A chain takes about 50-75 minutes.
- **The season so far, experiment 107, adopted** (the owner, 2026-10-23: "we need to CLEANLY support
  mid-season ... want to support 'at any point in the season' really"; DECISIONS.md, "Experiment 107"; code not
  committed).  New test: `openrapm folds --cut SHARE` rates every past season from its games through the day SHARE of
  its regular season was played and scores it on the rest; check 151 reads those folds, `--step4-from` gives a system
  another system's step 4.  New candidate: `openrapm rate --part-season-rows 0.25,0.5,0.75` with the panel of
  `openrapm data panel --so-far 0.25,0.5,0.75`.  Results, ratings of zero 0 and today's prior with step 4 refit at the
  cut 100: step 4 of whole seasons on a quarter season 90.1, the refit is the big lever; the part-season rows 100.4 /
  100.2 / 100.3 at a quarter / half / three quarters, better in every era, and on whole seasons WOWY z -3.91 and the
  year-over-year test z -2.20; 2026 top 20 the same players.  Folds `outputs/within/inputs99_cut25` (50, 75) and
  `parts107_cut25` (50, 75), `parts107_early`, `parts107_within`; boosters `prior_models_parts107.pkl`.

## Next

### Closed 2026-10-26: making the box-score prior robust to how players are split

The owner's verdict on experiment 113, 2026-10-26: "Not adopted".  The five-step plan ran through step 4; step 5 does
not run.  DECISIONS.md, "Three noise controls, and the prior made robust to the deal in code", has every reading.

- Three random splits of today's prior differ from each other as much as from today's: today's fixed split is not worse.
- Out-of-bag prediction with 60 bags won the quick screen, then tied in the full build: four builds against four,
  +0.0009, z +0.09; better 1997-2006 in all four, worse 2017-2026 in all four.  The prior alone is better and
  steadier before the swap step, which takes the gain back.
- The settings stay in the code, off by default: `--prediction out_of_bag`, `--split-per-bag`, `--bag-draw D`; check 155
  `run --variant`.  Noise controls for today's build: noise_creation110, noise2_creation110, noise3_creation110.
- The Changelog tab stayed empty then: this change did not ship.  Published 2026-10-31 (8d1ca8d) with its first
  entries.
- Next: the owner's call.  Open from before: experiments 111 and 112, the foul-type question.

### The daily job: 2027 is rated and published every morning (2026-10-24)

The owner, 2026-10-24: "You can publish. Yea median player in 2027, off poss. 6am is great."  Done and running;
DECISIONS.md, "The season in progress, 2027"; README.md, "The season in progress, every morning".

- **The daily job**: Task Scheduler "OpenRAPM daily update" runs `scripts/daily_update.sh` at 6:00 AM (this PC keeps
  Eastern time): ingest `--refresh`, stints, events, hustle games, matchup games (not a step), dashboards, the panel, `openrapm build --seasons
  2027 --replace-seasons`, `openrapm site --openrapm-decomposition` into `A:/code/spmm_site`, a worktree of
  origin/main only the job writes, then the two data files committed and pushed to main.  Log
  `outputs/daily/<date>.log`, the last result `outputs/daily/last.txt`.  First published 2026-10-24 at 1:36 AM
  (861c30c), 16 games, 345 players.
- **A 60-possession minimum on the site** since 2026-10-24 (5ff2daf, `site.MIN_POSSESSIONS_SHOWN`): fewer offensive
  possessions and a player-season is rated but not shown.
- **The page** opens on `meta.default_season` (8f025fd): the latest season whose median player, of everyone who
  played, has 250 offensive possessions (`site.default_season`).  2026 today, the 2027 median is 46.
- **The job runs the working tree's code**, committed as it is written (the owner, 2026-10-24: "always commit"): a
  stash or a branch switch in A:\code\spmm makes the next morning fail and leaves yesterday's ratings up.  Keep
  `config.yaml` `last_season: 2027`.  Local `main` lags origin/main by the job's commits: fetch before site work.
- **Waiting on the owner**:
  1. Done 2026-10-24: experiment 107 published for every season.
  2. The first days are extreme (four 2-possession San Antonio players at +7 to +8 on 16 games: the prior scale
     2.24 against 1.17, and the swap part of the prior on their context).  Proposed, not built: score ratings built
     from each past season's first 2%, 5% and 10% of games on the rest of it, with the prior scale fitted on the
     games against held at the finished seasons' value.

The offense/defense balance is fixed and published for 2017-2026 (experiment 106, version D).

**The season so far: the shrink by share was built and FAILED (experiment 108, DECISIONS.md).**  On held-back games
inside a season so far, today's whole-season step 4 beats multipliers fitted on the season so far at every share from
0.40 and ties at a quarter season, on the within-season WOWY test; the refit narrows the ratings within teams, which
that test punishes.  So a season so far is described with today's step 4 and the part-season rows; the step 4 fitted on
date folds is only for a forecast of the rest of a season.  What is left for mid-season support is the season in
progress itself (below).  Earlier plan, kept for the record: for any day of a season: step 4 fitted by the
share of the season played (a refit at a quarter season is worth 10 on the scale above; one date fold a season is
noisy, so fit on several cut days or a smooth line in the share); and the season in progress itself, which cannot be
rated yet (the bug-fix pass's list: the game log cached forever, both phases always built, the panel's rated rows
dropped with the season, the roles cache, the bio ages).

**Must fix, lower priority (the owner, 2026-10-23: "we REALLY need to fix pre-2017, but that's a lower priority"):
the pre-2017 defense.**  Version D widened it instead of narrowing it: defense standard deviation 1.73 to 1.92 in
1997-2006 and 1.63 to 1.78 in 2007-2016, defense over offense 0.94 to 1.03 and 0.98 to 0.99; 2017-2026 went 1.55 to
1.44.  The within-season WOWY test liked D in every era, so the fix must hold that test there too.  Leads: the within
layer's defensive games' part is about 2.0 in every era (DECISIONS.md, "Experiment 106"), and the defense factor is
0.89 before 2017 against 0.84 after.

What else the work of 2026-10-23 left open:
- Your two tests use different units and disagree on how far apart teammates should be: the within-season WOWY test
  scores stints, the year-over-year test whole team-games, and minutes that react to the game pull the team-game
  reading down (DECISIONS.md, "Why the shrink cuts the offense: the unit it is fitted on").  Version D cost the
  year-over-year test +0.16, z +4.4, adopted by the owner anyway.  A stint-level year-over-year reading already exists
  in `63_yoy.py`'s output; which unit rules the guard is the owner's.
- Before 2017 version D widens the defense, 1.73 to 1.92 and 1.63 to 1.78 by era: the balance moved only in 2017-2026.
- Parked, not next: experiment 104, the band-difference un-shrink; experiment 103, the line refit alone;
  width-matched readings in the chain; a noise control on the incumbent; the EM ridge idea.

## Waiting on the owner

- **Self-creation in lineups, check 154** (2026-10-24; DECISIONS.md, "Self-creation in lineups"): on held-back
  stints, lineups short of the owner's self-creation score 0.4-0.7 per 100 below OpenRAPM's expectation in every era,
  concentrated self-creation beats evenly spread, the top fifth falls below expectation since 2017, and none of it is
  larger after a trade.  With the owner's lineup shape held, good lineups multiplying and concentrated talent, both
  terms keep their size in every era; offense times defense quality multiplies, talent concentration does not.
  Every held-back stint of 2007-2026 is saved (`outputs/selfcreation_rows`, `--save-rows`), and a smooth curve drawn
  (`scratch/2026-10-24_creation/smooth.py`; page https://claude.ai/artifact/584UMVRduxWxRJYHnuTYGA).
- **Experiment 110 (creation110), ADOPTED and PUBLISHED 2026-10-25**: self-creation as an offensive box-score prior input.  Within-season
  WOWY -0.012, z -1.83, Newey-West -2.24, 7 of 10 in every era; year over year a tie; ratings nearly unchanged; check
  154's lineup pattern unchanged.  DECISIONS.md, "Experiment 110".
- **Experiment 111 (creation111), not adopted yet, verdict the owner's** (2026-10-25; DECISIONS.md, "Experiment 111"):
  self-creation with actual unassisted makes and the passes on misses and free-throw trips estimated
  (`data/creation_counts.py`, built by `openrapm data events`; `--features hustle_no_three_pct_self_creation_estimated`).
  A better measure of self-creation against 2016 tracking, a worse prior input: within-season WOWY +0.017, z +2.58,
  1997-2006 z +3.05, since 2017 a tie; year over year a tie.  Check 154 with it: amount +0.38 per sd against +0.67,
  concentration -0.49 against -0.43.  Open: the foul's type for shooting against bonus trips.
- **Experiment 112 (screen112), not adopted yet, verdict the owner's** (2026-10-26; DECISIONS.md, "Experiment 112"):
  check 155, Boruta on the within-season test on today's pipeline, screened 13 play-by-play candidates, the owner's
  estimated potential assists among them; only unassisted makes at the rim and in the paint passed.  Its build: WOWY
  -0.0016, z -1.24; 1997-2006 z -4.86, 10 of 10, nothing since; year over year a tie; ratings moved a median 0.004.
  The first noise control on creation110, `noise_creation110`, reads -0.017, z -1.83: use it beside creation110 from
  now on (`CONTROL_NAME=noise_creation110`).  Screen table on the sheet's tab "Input screen 112".
- Deleting the saved boosters of experiments that were not adopted, about 2 GB each in `outputs/prior_models_*.pkl`
  (creation111, screen112; noise_creation110 stays while its folds may be extended; luckall88, luckall88b, luckhalf89, luckhalf89b, near83, same84, ctx85, body86, spread90, three92, unshrinkbase102,
  unshrink102, shrunkbase103 and older ones; refit103's stays while experiment 104 is open).
- Filed earlier: tracking game by game; arena scorekeeper checks; Stockton over Jordan in 1997-98.  The Curry
  star-guard ceiling test passes again (2026-11-01: Curry 20th in 2026).

## What ships

| step | plain name | module | as shipped |
|---|---|---|---|
| 1 | the label | `rapm.py` | one RAPM per player over every season but the rated one; penalties 40,000 / 40,000 / 0 |
| 2 | box-score prior | `box_prior.py` | LightGBM `lgb1_tune2`, linear leaves; feature set `hustle_no_three_pct_self_creation`, self-creation on offense since experiment 110; five player folds; 10 bags; part-season rows at a quarter, a half and three quarters of each training season since experiment 107 |
| 2b | swap part of the prior | `second_prior.py` | line `ctx110` on `context_and_types_centred`, team totals held; refitted on creation110's boosters and folds since experiment 126, adopted 2026-11-01 (ctx82, experiment 78's, until then) |
| 3 | the rating | `prior_rapm.py` | RAPM pulled toward the prior, penalty 13,037, centred; the season in progress holds its prior scales at the 8 nearest finished seasons' mean through a quarter of its season, then a straight line to its own fitted ones at three quarters (experiment 109, adopted 2026-10-30; the line 2026-10-31) |
| 4 | the shrink, with the defense factor | `prior_shrink.py` | both parts multiplied by what holds up on held-back games, 8 nearest fold seasons, defense times 0.89 before 2017 and 0.84 from 2017; folds `ctx110_early,ctx110_within` since experiment 126, each multiplier read to a full season with the tenth folds `ctx110_k10` (experiment 101); two layers since experiment 106: each part's team mean on held-back team-games, each player's deviation from it on held-back stints within games, the within layer times 0.75 on offense and 1.0 on defense |
| 5 | swap adjustment | `swap_adjust.py` | off: applied share 0, the build skips it |

- `openrapm build` writes `outputs/season_ratings_product.parquet` with `_step3` and `_step4` beside it.
- `openrapm site --bias --openrapm-decomposition --similarity` writes the site's files.  Commit, fast-forward `main`
  and push only on the owner's word, then check the Pages build: `gh api repos/bbstats/openrapm/pages/builds/latest`.
  With the second prior, the decomposition's "box score" piece carries the whole prior; its split is the owner's.
- The incumbent's test-setting tables `outputs/season_ratings_ctx110*.parquet` (experiment 126), folds `ctx110_early`, `ctx110_within`;
  its noise control `noise_creation110` (2026-10-26), the chain's default `CONTROL_NAME`.

## How an experiment is judged

| reading | script | rule |
|---|---|---|
| within-season WOWY test | `scripts/checks/151_wowy.py`, stint unit, each system's own step 4 (`--shrink-flags`, the chain passes them) | better in every era, against the incumbent; the noise control beside it; the era table is printed and written (`wowy_<tag>_eras.csv`) |
| year-over-year test | `scripts/checks/63_yoy.py` | not worse at z +2 |
| the eye check | `scripts/checks/66_compare.py` | the 2026 top 20 |

- One build: `bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes_late_as_scored "EXTRA RATE
  FLAGS"`, about 50 minutes with 4 processes.  `INC_NAME` and `INC_SYSTEM` name the incumbent's tables and folds,
  `ctx110` for both since 2026-11-01 (experiment 126); `CONTROL_NAME` the noise control, `noise_ctx110`.
- **The season-page score**: each season, ratings of zero score 0 and the reference 100.  The zero line on today's
  stints is `outputs/csv/wowy_zero_baseline_inputs99_seasons.csv`, column zero, 3,681.6 a season: ratings of zero do
  not depend on the system, and every system is scored on the same held-back stints.  creation110 explains 19.7 a
  season, 14.1 to 26.2.  The older file, `wowy_zero_baseline_seasons.csv`, which
  `scratch/2026-10-20_charts/season_scores.py` reads, predates the parser fixes of 2026-10-22 (3,648.6).
- **Which z rules**: the plain z.  `z_newey_west` is floored at it since 2026-10-30: unfloored, a noise control
  read -2.44 against plain -1.83, so noise passed -2 (DECISIONS.md, "The bug-fix and make-better pass of
  2026-10-30").  Two builds that differ only in the box-score prior's random split differ by about 0.010 on the
  within-season test.
- The noise control `noise_ctx110` moves the within-season test by -0.013, z -1.63, on its own (`noise_creation110`
  on creation110 read -0.017, z -1.83); a candidate is read against today's ratings on the same split, so that noise
  does not enter its change, but any one build's standing carries 0.01-0.02 of it.
- Every experiment gets a row in `experiments/runs.csv` (`scripts/checks/101_log_run.py`), mirrored to the sheet
  "OpenRAPM experiments", tab Experiments; the next free row is 127.  Since 2026-10-30 it reads the within-season
  test by era too, and refuses a partial or stale reading.  Input screens get their own tab ("Input screen 112").

## Commands a session runs

    bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes_late_as_scored "EXTRA RATE FLAGS"
        one experiment against the incumbent, about 50-75 minutes; log outputs/NAME_chain.log
    .venv/Scripts/python scripts/checks/151_wowy.py --systems=ctx110,noise_ctx110,NAME --ref=ctx110 --tag=NAME
        the within-season WOWY test on folds already built, about 3 minutes
    .venv/Scripts/python scripts/checks/155_input_screen.py run --tag TAG --workers 2 --fitters 6
        the input screen: candidates against their shuffled copies, three random draws, ten test seasons, about 8 hours
        for 13 candidates (cache once: `cache --tag TAG`, check it: `dry --tag TAG --seasons 2025`, read it: `table`)
    .venv/Scripts/python -m openrapm data matchup-games 2018 2026 --limit 400
        the per-game matchups, 400 games a process; rerun until nothing is missing (scratch/2026-10-28_matchups/fetch_driver.py)
    .venv/Scripts/python -m openrapm second-prior --models bags10 --folds bags10_early,bags10_within --tag TAG --inputs SET
        a second prior's line from held-back swaps, about 13 minutes
    .venv/Scripts/python scripts/checks/101_log_run.py --name=NAME --exp=N --idea=owner --verdict=pending --changed="..." --notes="..."
        the experiment table's row
    .venv/Scripts/python scratch/2026-10-21_charts/defense_gap_stats.py
        every 2026 defense rating split by stat against the consensus, about 8 minutes

## The rulings that bind

1. One rating per player per season, from that season's games only, regular season and playoffs.
2. Single year or bust: a player's own other seasons may not reach his rating.  Model coefficients may be learned from
   other seasons.
3. Never train on the current season until its Finals are over.
4. The consensus is a sanity check, never a fitting target.
5. Each season is centred at possession-weighted zero per side.
6. The second prior is fitted to points only through swaps (the owner, 2026-10-19).

## How the work is done

- Implement the owner's ideas or propose new ones; one experiment at a time unless the owner says otherwise.
- Never thrash the machine: several processes at once are fine while they fit its 12 threads and 32 GB.  When a chain
  is stopped, stop its python children too; `TaskStop` can leave the processes running, so check `ps` after.  Never
  edit `scripts/experiment_chain.sh` or `src/` while a chain runs: bash reads its script as it goes, and each new
  python process imports the code as it is then.
- Commit every piece of work; never push or publish without the owner's word, but for the daily job.  The site is
  `docs/`, served from `main`.
- Movement in points per 100, never rank places; under 0.1 is nothing.

## Traps that cost a day

1. `defense` in the ratings tables is points allowed, negative good; `rating_def` is positive good.
2. A season-total input leaks into the within-season folds: whole-season inputs read z -31.
3. A long bash heredoc fails in this shell: write the script to a file first.
4. A background job is stopped after 2 hours: launch long queues with `nohup ... &`.
5. LightGBM's own Shapley values refuse linear leaves: sample orders instead (`scratch/2026-10-21_charts/prior_shap.py`).
6. A line fitted on swaps has no intercept: centre its inputs, or each cross-fitted copy carries its own level.
7. More in DECISIONS.md, "The measurement traps".

## Verify you are where this file says

    .venv/Scripts/python -m pytest -q                       # 885 passed, 2 skipped, 7 deselected, 1 xfailed: the backup bigs
    uvx ruff check .                                         # clean
    git log --oneline -1                                     # the handoff of 2026-11-01
