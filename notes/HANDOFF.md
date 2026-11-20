# Handoff

**What this file is.**  Where the work stands, the next piece of work in full, what waits on the owner, and the
commands a session runs.  Its top was rewritten on 2026-11-19 for the next research path; the file before it is in git
at a77b019.  `DECISIONS.md` is the permanent record and carries every number quoted here; read its measurement traps
before trusting a number.  `README.md` says what OpenRAPM is, `PIPELINE.md` how each part works, `GLOSSARY.md` what
each word means, `CONTRIBUTING.md` the writing rules, `CLAUDE.md` the owner's standing rules.

## Talking to the owner

- American English in everything shared: defense, offense, color.  Code names keep their spelling.
- No step numbers in messages.  The names: **box-score prior** (the LightGBM); **Box Adj** (the line on his teammates'
  and opponents' priors and his team's shooting with him on and off the court, added before the games are weighed);
  **+/-** (what the season's games add on top of the prior); **the shrink** (both parts multiplied by what holds up on
  held-back games); **defense factor** (the shrink's extra factor on defense, 0.8682 before 2017 and 0.8704 from 2017;
  never "dial").  Define each the first time it appears.
- Experiments by what they changed, never by number or code name; numbers only in the record files.
- Answer exactly what is asked.  Results as tables, one quantity per column, no parentheses, one TL;DR sentence.
- Within-season WOWY is what counts, and a win must hold in every era: 1997-2006, 2007-2016, 2017-2026.
- **No jargon ever.**  "how the box-score prior splits players into its five training groups", never "folds";
  "10 averaged versions, each trained on a random resample of players", never "bags"; "random draws", never "seeds";
  "full build", never "chain"; no formulas or letters.
- Decisions go through the question tool, with the numbers in the question and one outcome per option.  The owner
  does not read every message: a question stands alone.
- I write no prose for the site.  Wording is the owner's; a page change of mine adds data, a column, or a line the
  owner has already approved elsewhere.
- **Age is not an input to any rating** (the owner, 2026-11-18: "I want age removed!").  Never offer age, an age
  curve or age bands to the prior, to Box Adj, to a screen's candidates or to a tree stage.
- The swap adjustment is off and the owner does not care about it (2026-11-19: "i dont care about the swap adjustment
  if we're not using it").  Do not raise it.  Box Adj is the swap adjustment's lesson moved into the prior.
- Before asking for a verdict, compare every season's top-rated player and WOSA leader in both tables, not 2026's
  alone: on 2026-11-19 two old seasons' WOSA leaders changed and the owner had answered on 2026.

## Done 2026-11-20: experiment 144, three-point luck out of the offense's target

DECISIONS.md, "Experiment 144" (registered before any result).  The offense fitted on the defense's own target,
`--offensive-target points_with_expected_opponent_threes_late_as_scored`, by README.md's passes with Box Adj's line
refitted: `scratch/2026-11-20_offense_threes/build144.sh`, started 9:40 AM, log `outputs/off3_144/driver.log`, markers
in `outputs/off3_144/done/` (a rerun skips finished steps).  Names: `off3_144_base`, `off3_144_line`, `off3_144`,
`matched144`.  **Result: fails the rule**, within-season -0.022, z -0.70, eras +0.003, -0.091, +0.021; before the shrink -0.083, z -2.85; year over year a tie.  Asked the owner for the verdict 2026-11-20.

## Done 2026-11-19: experiment 143, two defense inputs in Box Adj, ADOPTED and PUBLISHED (71436f5)

DECISIONS.md, "Experiment 143" (registration, the four reads, the full build, the product build, adoption, publish).

- **Shipped now:** Box Adj's line `rd143_line` on `context_types_no_age_team_zones_defence_only_centred`: today's 56
  inputs an end plus, on the defense end alone, his offensive rebound share and his close-game share (his share of his
  team's possessions in the last five minutes of the fourth quarter or overtime within 5, minus his share of all of
  them, padded by 50; `second_prior.DEFENCE_ONLY`).  The shrink reads `rd143_early,rd143_within` and `rd143_k10`.  The
  published tables are `product143`'s; the ones before are `*_before_rd143*`.
- **What it did:** within-season -0.099, z -3.62, every era better (1997-2006 at noise-control size); year over year a
  tie as built.  The line now weighs defensive rebounds about four times as heavily on defense and offensive rebounds
  against it; mostly-offensive-rebounding centers fall (Robinson 2.40 to 1.42).  WOSA's 2025 and 2026 leader became
  Jokić over Gilgeous-Alexander; WOSA leaders also changed in 1999, 2006 and 2017.  The owner saw all of it and chose
  "Adopt and publish".
- **Done:** the noise control `noise_rd143`, -0.012, z -1.37, by era -0.031, -0.010, +0.004; the chain names
  `rd143` and it (`scratch/2026-11-19_box_adj_reads/incumbent143.py`).  The Changelog row is live ("Box-adj now reads two more defensive inputs.", 9e43827).  **Left to do:** the Power table's weights, not refitted (check 166 cannot run on today's build).
- **Read 4's candidate, not built:** one ChimeraBoost step on what the line leaves read -0.085 and -0.054 on two draws;
  a build would average several draws.  The owner chose the two inputs first.
- **Leak caught:** 13 tracking columns are season totals a fold only rescales; they leak and were left out of the
  screen (DECISIONS.md, measurement traps).

## The next research path, the reads on Box Adj (done 2026-11-19, above)

**Approved by the owner on 2026-11-18** ("box adj is good ... the rest is fine"), not started.  The plan with the
whole ranked list: `C:\Users\Nathan\.claude\plans\what-are-some-possible-zippy-quill.md`; the record:
DECISIONS.md, "The research paths of 2026-11-18".  Four reads, all from saved held-back games, no rating built, 3 to 4
hours.  **Report and stop**: a read that passes goes to the owner as a candidate for a full build, through the
question tool; nothing is built on a read alone.

**Why Box Adj.**  It is the only part of a rating fitted on the within-season test, about 45% of a regular's rating,
and the last three gains came from it: its weights fitted on that test, 0.488; the team's shots by five zones, 0.183;
its training seasons faded by distance, 0.133.  The season's games take back 57% to 60% of a prior change that moves a
team's total and 12% to 14% of one that moves credit between teammates, so only the second kind is worth trying, and
Box Adj holds team totals by construction.

**The rule, registered before any read** (experiment 140's, used for the zones and the fade): a read goes forward when
it is better than today's line in every era, 1997-2006, 2007-2016 and 2017-2026, and by more than 0.042 over the 30
seasons, with its shuffled control at nothing.  Between two that tie, the simpler one.  A read from saved sums gives the
sign in each era and overstates the size: the fade read 0.207 and built 0.133; age out read 0.031 and built 0.027.
Register each read in DECISIONS.md before its result, as "Experiment 142, read 1" was.

**What is on disk for it.**

| what | where |
|---|---|
| the held-back games the line is fitted on, rated without Box Adj | `outputs/within/rebuilt_base_early`, `rebuilt_base_within`; saved models `outputs/prior_models_rebuilt_base.pkl` |
| each of those sets' players, 239 columns rebuilt from its own games | `players_<season>_r<repeat>_f<fold>.parquet` inside those folders |
| today's line: its sums and each set's inputs | `outputs/second_prior/noage142_line/sums_<season>.npz`, `inputs_<season>.parquet`, `run.json` |
| the same with age, the line before it | `outputs/second_prior/rebuilt_line` |
| the incumbent's test tables, held-back games and noise control | `noage142`, `noage142_early`, `noage142_within`; `noise_noage142` |

A season's sums hold 114 columns: each end's fitted part, then 56 inputs for offense and 56 for defense, in the order
of `second_prior.INPUT_SETS["context_types_no_age_team_zones_centred"]`.  `rating_line._solve(total, columns, 1e-4)`
gives the weights on any subset of columns and `RatingSums.error(weights)` the within-season test's error at them.
Today's line for a rated season is solved on the seasons more than one from it, each counted
`second_prior.season_weight(other, rated, 8)`.  `scratch/2026-11-18_age_out/age_out_read.py` is the shortest working
example of such a read: 30 seasons, paired by season, by era, in seconds.

**The four reads.**

1. **More inputs** (`scratch/2026-11-03_direct/direct_screen.py`, `build` then `read`, about 8 minutes a pass).  On
   2026-11-03 every candidate added to the line at once read 0.830 better, 28 of 30 seasons, eras z -7.24, -4.73,
   -1.25; the team rates that were adopted were 0.218 of it; 172 shuffled candidates cost 0.244.  The add-one and
   drop-one tables were lost with the data.  To run it today:
   - `FOLDS_TAG` to `rebuilt_base_early,rebuilt_base_within`; `inputs_dir` to `outputs/second_prior/noage142_line`.
   - Its within-layer multipliers come from a csv of a full build, `season_ratings_ctx110_step4_multipliers.csv`,
     gone.  The line's own fit computes them from the same held-back games with
     `commands.build.within_layer_multipliers`, as `commands/second_prior.run_second_prior` does; use that.
   - It solves on every training season alike.  Add the fade: each training season's sums `.scaled(season_weight(...))`.
   - Take `age`, `age_at_first_season`, `age_at_debut` and every career column out of the candidates; `NOT_INPUTS`
     already holds the career ones.  `playoff_share` stays out: a coach's choice made after the held-back games.
   - The tracking columns are season totals and leak the held-back games; the screen drops any column equal across two
     sets of one season (`panel_candidates`).  Keep that check.
   - Two candidates to build, from each set's own games only: his share of his team's possessions in the fourth
     quarter and overtime with the margin within 5, set beside his share of all of them; and minutes per game.  The
     stint rows carry `period`, `margin_h` and `frac_rem`; `garbage_time_share` in `data/panel.py` is the pattern.
   - Then forward steps: the best candidate by the rule, added, and again, each beside its shuffled copy, until none
     passes.  Report the whole table, every candidate's verdict, never a prose list.
2. **What the LightGBM is worth beside the line.**  On 2026-11-03 dropping the box-score prior from one end of the
   line cost 0.060 on offense and 0.082 on defense; it is 92% to 94% a straight line on its own inputs.  Not read
   since the zones, the fade or age out, and never year over year.  The read: today's line without its `box_prior`
   and `box_prior_other_end` columns, with the LightGBM's own inputs offered to the line directly,
   `features.FEATURE_SETS["hustle_no_three_pct_self_creation"]`, 18 a side; those not already among the line's inputs
   are candidates of read 1's screen, so this is two more rows of the same pass.  The hustle inputs exist from 2017
   only: read 2017-2026 with them and every era without.  **A small cost is not a reason to drop the LightGBM by
   itself**: it alone sets each team's total in the prior, which the within-season test cannot see and the
   year-over-year test can.  A result here goes to the owner as a question, with what a build without it would have
   to show year over year.
3. **The box-score prior's weight by playing time.**  The line gives the box-score prior one weight for a starter and
   a tenth man alike.  The read: one more column an end, the box-score prior times his centered share of team
   possessions, scored like any candidate.  It is the backup-bigs question the tests keep visible,
   `tests/test_vs_consensus.py::test_backup_bigs_are_not_in_the_top_twenty`.  A product of two inputs is not in the
   saved sums: it needs the per-player sums of read 4, or a column in read 1's screen.
4. **Trees on what the line leaves** (`scratch/2026-11-10_boost_vs_line/player_sums.py`, 22 minutes, then
   `boost_vs_line.py`).  One step at ChimeraBoost's defaults read 0.100 better, 21 of 30, eras -0.111, -0.066, -0.124;
   a second random draw 0.115; shuffled targets +0.004.  DECISIONS.md, check 167: "A fit that repeats the step, or
   weighs teammates, could find more or the same; none of that was tried."  To run it today:
   - `player_sums.py`: `FOLDS` to `rebuilt_base_early,rebuilt_base_within`; `SCREEN` and `LINE` to `noage142_line`,
     whose folder holds both the inputs and the sums; `OUT` to `noage142_players`.  Its built-in check must pass: each
     season's per-player sums, projected on the line's columns, give back the line's own sums.
   - `boost_vs_line.py`: `NAMES` to the list without age, the folder to `noage142_players`, and the fade in `scores`
     and `block_columns`, where every training season counts alike today.
   - Then repeat the step: targets at the line plus the trees so far, a new column, weights solved again, up to five
     steps, with `BOOST_SHUFFLE=1` and a second `BOOST_SEED` at each.  The per-player sums hold the whole quadratic,
     so no ridge is solved again.  chimeraboost is installed in the environment and is not a dependency of the package.
   - Three seasons to a block, a block's models fitted on the seasons two or more away: keep it, and write down that
     a later step's targets come from columns whose models saw other blocks.  The shuffled control is what says
     whether that matters.

**Traps for these reads.**

- A candidate built from lineup results restates what the ridge already reads from the same games.  Fitted through
  the ridge it loses its weight by itself, which is why the direct screen exists; a candidate scored any other way
  must first pass `scratch/2026-11-03_team_rates/why_it_lost.py`'s check.
- 2017-2026 is the weak era for added inputs: more columns, more noise, and before the fade it wanted a stronger
  penalty on the line.  Read the line's penalty, 0.0001, again at 0.001 to 0.1 once inputs are added.
- In 1997-2006 the noise control alone reads 0.031 better, z -3.2.  A gain there of that size is not a gain.
- 2007-2016 is the era that fails most often.  Say which era failed and by how much; a tie is not noise.
- Every one of these is a read.  A full build is `bash scripts/experiment_chain.sh NAME
  points_with_expected_opponent_threes_late_as_scored "EXTRA RATE FLAGS"`, 70 to 120 minutes on this machine, and
  needs the owner's word.

**After these reads, the ranked paths none has started** (the plan has what the record says of each): three-point
luck out of the offense's target, as the defense's target has it; early-season defense, the hustle rates' padding
first; the ridge penalty, 13,037, never read on the within-season test, and the owner's EM idea; lineup terms from the
self-creation check; lineups against the league's official minutes.

## Where it stands, 2026-11-19

DECISIONS.md holds every number, from "The data deleted and rebuilt" to "The morning of 2026-11-19".

- **The data was deleted on 2026-11-16 and rebuilt on 2026-11-17.**  `git worktree remove --force` on old side
  checkouts followed their junctions into `data` and `outputs`.  **Never remove a side checkout before unlinking its
  junctions.**  Every earlier experiment's saved files are gone; what is on disk is the rebuild's and this week's.
  Backup of the rebuild: `D:\openrapm_backup\2026-11-17`.
- **Age is out of Box Adj: adopted and published 2026-11-19** (experiment 142; site 102e307).  The shipped line is
  `noage142_line`, 56 inputs an end; the shrink reads `noage142_early,noage142_within` and `noage142_k10`; the
  published tables are the full build `product142`'s, the ones before kept as `*_before_noage142*`.  It cost 0.027 on
  the within-season test, none of it since 2017, and its year-over-year cost is all a season's ratings asked about the
  season after it.  No 2026 total moved 0.1; two WOSA leaders changed by a hair, 1999 and 2003.
- **The incumbent is `rd143`, its noise control `noise_rd143`**, -0.012 on the within-season test; the chain
  names both since 2026-11-19.  Before it `noage142` and `noise_noage142`, -0.013.  Check 151's errors on the incumbent before it, `rebuilt`: 3,455.273, 3,595.240 and 3,932.011 by era.
- **The morning job waits for one game.**  0022600265, Hawks at Thunder of 2026-11-18, came from the league with every
  play-by-play description blank, so it builds no stints.  `config.yaml`'s `no_stints_accepted` names it (the owner:
  "skip the game for now but we should keep checking for it"): the season is rated without it, every morning asks the
  league again, and the day a morning's log says it has stints, take it off the list.  Basketball Reference's page for
  the game was not used: Sports Reference's terms bar feeding its pages to a model.
- **Main holds everything**, c29342d, GitHub's tests green.

## Waiting on the owner

1. **Your word on any read of the next research path that passes its rule**, before a full build.
2. **Google's AI answers.**  The robots file cannot keep the site out of the AI answers inside Google search; that is
   the Search Console setting under Settings, "Search generative AI", in the owner's account, if it is live there.
3. **The side checkout `A:/code/spmm_cleanup`** and its branch `cleanup-2026-11-16`, merged: remove them or keep them.
   Unlink its junctions first.
4. **The phone notice.**  Install the ntfy app and subscribe to the topic in `outputs/daily/notify.conf`, given to the
   owner in chat on 2026-11-09.  A test line: `bash scripts/daily_notify.sh "test"`.
5. **The link preview's one sentence**, for the six pages' `description` and `og:description`.
6. **No Changelog row for age out of Box Adj for now** (the owner, 2026-11-19: "since we haven't explained box adj
   just leave it for now").
7. **Changelog rows**, in the owner's words: the robots file and the credit line; the rookies button, WOSA, the page
   opening on the season in progress, the Detail view, Detail's Leverage and WOSA columns, Similarity's search box, the
   Power page's one rating, the Similarity and WOSA corrections and the Power table's new weights.
8. **Page text that no longer matches the code.**  About: "Share of league games played", held at a quarter while a
   season is in progress; "vs league average" on the hustle rows only, though the box inputs are centered too; "All
   stats use padding method" beside ten "Not padded" rows.  Power: the column tips leave out the Last 10 row, the
   multipliers and blowout possessions counting half.  Ratings: the Leverage tip says close games where the code
   compares moments inside each game.
9. **Verdicts still open in the experiments table**: defense's boosters at 330 trees, a tie, recommended off;
   self-creation with estimated passes; the play-by-play and matchup input screens; the team rates without the direct
   plus-minus.  The three removed priors' rows read not adopted.
10. **Sheet "OpenRAPM experiments", row 66**: a stray copy of the early-season padding row; one click.

## What ships

| part | plain name | module | as shipped |
|---|---|---|---|
| step 1 | the label | `rapm.py` | one RAPM per player over every finished season but the rated one; penalties 40,000 / 40,000 / 0; the defensive label un-shrunk |
| step 2 | box-score prior | `box_prior.py` | LightGBM `lgb1_tune2`, linear leaves; inputs `hustle_no_three_pct_self_creation`, 18 a side; five player folds; 10 bags; part-season rows at a quarter, a half and three quarters |
| between 2 and 3 | Box Adj | `second_prior.py`, `rating_line.py` | line `noage142_line` on `context_types_no_age_team_zones_centred`, 56 inputs an end, none of them his age, weights fitted on the within-season test through the ridge, team totals held, penalty 0.0001, each training season counted half for every 8 seasons from the rated one; the Detail view's Box score column learned first (`box_first`) |
| step 3 | the rating | `prior_rapm.py` | RAPM pulled toward the prior, penalty 13,037; the season in progress holds its prior scales at the 8 nearest finished seasons' mean through a quarter of its season, then a straight line to its own at three quarters |
| step 4 | the shrink | `prior_shrink.py` | both parts multiplied, 8 nearest fold seasons, folds `noage142_early,noage142_within`, read to a full season with `noage142_k10`; two layers, the within layer times 0.7663 on offense and 0.9968 on defense; defense factor 0.8682 before 2017, 0.8704 from 2017 |
| step 5 | swap adjustment | `swap_adjust.py` | off: applied share 0 |
| | WOSA | `wins.py` | wins over the 150 most-started players' average, from the regular-season rating alone, 30 rating points a win |
| | Power | `power_forecast.py`, `data/power_files.py` | Power at its multipliers plus the last-10 stats; weights at five points of the season in `params/power_forecast.json`, refitted 2026-11-09 by check 166 |

`settings.py` is the source for every number above and names the experiment that set each.

## The morning job

`scripts/daily_update.sh`, Task Scheduler "OpenRAPM daily update", 6:00 AM, from the working tree's code; about 13
minutes.  PIPELINE.md, "The morning job", lists its commands and what protects it.

- **It runs the working tree.**  No edit under `src/` or `scripts/` of `A:/code/spmm` between 5:50 and 7:30 AM.  Work
  that touches code goes in a second checkout, as the cleanup did, and reaches `A:/code/spmm` by one fast-forward once
  proven.  After any change to the job, the stints, the build or the site step: `PUBLISH=0 bash
  scripts/daily_update.sh`, 12 minutes, which writes the site's files to `outputs/daily/site_dry` and says how it
  ended in `outputs/daily/last_dry.txt`.
- **`outputs/daily/last.txt`** says how the last real morning ended; `outputs/daily/<date>.log` is its log.
- **When it fails** the phone gets one line, and yesterday's ratings stay up.  A game that failed the games check has
  had its cached files removed and is fetched again by the next run.  A game whose points differ from its box score
  for good goes in `config.yaml`, `points_mismatch_accepted`.  A game the league posted without usable play-by-play
  goes in `no_stints_accepted`: the season is rated without it, every morning asks for it again, and the check says
  when it can come off.  Since 2026-11-19 it holds 0022600265, Hawks at Thunder of 2026-11-18, every description
  blank (the owner: "skip the game for now but we should keep checking for it").
- **A second task, "OpenRAPM morning check", 8:00 AM**, sends the notice when `last.txt` lacks the day's date.
- **A command that writes the name `product`** is refused here unless `OPENRAPM_WRITE_PUBLISHED=1` is typed before it.
  A long one takes the lock first:
  `( . scripts/daily_lock.sh; take_lock "what it is" || exit 1; trap release_lock EXIT; COMMAND )`.
- **June 2027.**  The morning after the Finals the job stops after its ingest, `last.txt` reads "stopped", one notice.
  That day: `openrapm data hustle-games 2027 2027 --force`, `openrapm data matchup-games 2027 2027 --force`,
  `openrapm data panel --force`, `openrapm data panel --so-far 0.25,0.5,0.75 --force`, then a full build of every
  season, the site's files, a refit of `params/power_forecast.json`, and the publish on the owner's word.  In October
  2027 raise `last_season` in `config.yaml` and `SEASON` in the job.
- **The first playoff morning of 2027** is the WOSA lock's first live use: 2027's WOSA should stop moving.

## Publishing

The site is `docs/`, served from `main`.  Nothing is pushed without the owner's word, but for the morning job's data.

    git fetch origin main && git merge --no-ff origin/main      # the job's data commits; the merge has been clean
    bash scripts/publish_check.sh                               # the lint check, every test, the tutorial, on a clone
    git push origin HEAD:main && git fetch origin main && git branch -f main origin/main
    gh run watch --repo bbstats/openrapm                        # GitHub's tests, then the Pages build

- A page that reads a new data file goes to main in the same push as the first copy of that file, and the code that
  writes it: the morning job writes from the working tree into a checkout of main.
- A new build of the ratings: `openrapm site --openrapm-decomposition --similarity --power --power-forecast`, then
  refit `params/power_forecast.json` (`scripts/checks/166_power_forecast_refit.py --write`, after its season-so-far
  ratings are rebuilt on the new build).
- A page is checked in a browser before it goes: `playwright-core` with the installed Chrome, at 1,280 and at 390
  pixels, and its tables against the live site's, row for row.
- **The live site bars AI fetchers.**  A tool that fetches pages for an AI, the WebFetch tool among them, is refused by
  openrapm.com's robots file; read the live site with `curl` or a browser.  To refresh the crawler list:
  `scratch/2026-11-16_cleanup/make_robots.py <commit of ai.robots.txt>`, read the diff, run `tests/test_site_pages.py`.

## How an experiment is judged

| reading | script | rule |
|---|---|---|
| within-season WOWY test | `scripts/checks/151_wowy.py` | better in every era against the incumbent, the noise control beside it |
| year-over-year test | `scripts/checks/63_yoy.py` | not worse at a paired z of +2, read at each side's width matched to the incumbent's |
| the eye check | `scripts/checks/66_compare.py` | the 2026 top 20 |

- The incumbent is `rd143`: tables `outputs/season_ratings_rd143*`, folds `rd143_early`, `rd143_within`,
  year-over-year score 8.6900.  Its noise control is `noise_rd143`: -0.012 on the within-season test, paired z -1.37,
  17 of 30, and -0.031, z -3.18 in 1997-2006 alone, so that era is read beside it.  The incumbent before it,
  `noage142`, and `noise_noage142` are still on disk.  Both are the defaults at the top of `scripts/experiment_chain.sh`.  The plain z is what adoptions have
  been read on.
- One full build: `bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes_late_as_scored "EXTRA
  RATE FLAGS"`, 50 to 75 minutes, log `outputs/NAME_chain.log`.  It waits while the morning job runs.
- A Box Adj input or setting is read first from the saved sums, in minutes
  (`scratch/2026-11-09_improvements/box_adj_weights.py`, `scratch/2026-11-05_second_stage/zones_screen.py`), then
  confirmed by a full build.  Any number per player is scored on the within-season test in minutes from
  `outputs/second_prior/zones139_players/` (`scratch/2026-11-10_boost_vs_line/player_sums.py`, `boost_vs_line.py`).
- Register a reading in DECISIONS.md before any result.  Every experiment gets a row in `experiments/runs.csv`
  (`scripts/checks/101_log_run.py`), mirrored to the sheet "OpenRAPM experiments", tab Experiments, written at the first
  free row after the last, never appended.
- Report in four parts: what changed, the result, the 2026 top 20, the owner's call.

## Leads, none started

- **The Power table from half a season to three quarters.**  The straight line between those two points passed its
  rule by a hundredth of a z and loses 2017-2026 to a fit at the day.  Read it again before the season in progress
  gets there, about the middle of January 2027, with weights fitted at 62.5% as the candidate
  (`scripts/checks/166_power_forecast_refit.py`).
- **Early-season defense.**  2027's defense is 0.60 as wide as its offense among players with 250 or more possessions;
  finished seasons read 0.87, 0.85 and 0.82 by era.  The hustle rates are padded toward the raw rate of players with
  similar possessions, as the box rates were before the early-season fix (`data/dashboard_rates.py`).
- **The ridge penalty**, 13,037, set on the year-over-year test in September and never read on the within-season
  test; the owner's EM idea.
- **Numbers fitted once and since overtaken**: the width settings; WOSA's 30 points a win; the 0.25 of opponents'
  three-point luck kept; `lgb1_tune2`, tuned before Box Adj existed.
- **2027's play-by-play** forces 0.60 possession switches a game against 0.30 over 2026's first 146.
- **Three neutral-site games of 2024** carry a home team in the game log: 0022301229, 0022301230 and 0022300527.
- **The other ranked paths of 2026-11-18, none started**: three-point luck out of the offense's target; early-season
  defense, the hustle rates' padding first; the ridge penalty on the within-season test and the owner's EM idea;
  lineup terms; lineups against the league's official minutes.  The plan file has what the record says of each.
- **The published Box Adj line** is fitted on `rebuilt_base`'s held-back games, pass 1 of README.md's recipe, on the
  default list of inputs, the one without age, so a fresh build repeats it.
- **Check scripts that cannot run on today's build as they stand**: `155_input_screen.py`, whose reference is
  creation110; `152_decider_tune.py`, whose pipeline has no Box Adj; `156_shrink_refit.py`, `159_power_next10.py` and
  `166_power_forecast_refit.py`, which name `fade141`'s tables and season-so-far folders, deleted and not rebuilt.

## Commands a session runs

    .venv/Scripts/python -m pytest -q                        # every test, about 2 minutes
    uvx ruff@0.17.0 check .                                   # the lint check, at the version GitHub pins
    PUBLISH=0 bash scripts/daily_update.sh                    # the morning job, nothing published, 12 minutes
    bash scripts/publish_check.sh                             # before any push to main
    bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes_late_as_scored "EXTRA RATE FLAGS"
    .venv/Scripts/python scripts/checks/151_wowy.py --systems=rd143,noise_rd143,NAME --ref=rd143 --tag=NAME
    .venv/Scripts/python scripts/checks/101_log_run.py --name=NAME --exp=N --idea=owner --verdict=pending --changed="..."
    OPENRAPM_WRITE_PUBLISHED=1 .venv/Scripts/python -m openrapm build      # a full build of the published tables

## The rulings that bind

1. One rating per player per season, from that season's games only, regular season and playoffs.
2. Single year or bust: a player's own other seasons may not reach his rating.  Model coefficients may be learned from
   other seasons.
3. Never train on the current season until its Finals are over.
4. The consensus is a sanity check, never a fitting target.
5. Each season is centered at possession-weighted zero per side.
6. WOSA reads the regular season alone.

## How the work is done

- Implement the owner's ideas or propose new ones; one experiment at a time unless the owner says otherwise.
- Never thrash the machine: several processes at once are fine while they fit its 12 threads and 32 GB.  When a full
  build is stopped, stop its python children too.  Never edit `scripts/experiment_chain.sh` or `src/` while a build
  runs from the same checkout: each new python process imports the code as it is then.
- Commit every piece of work; never push or publish without the owner's word, but for the morning job.
- Movement in points per 100, never rank places; under 0.1 is nothing.
- A proof that a change moves no number is a byte comparison with both sides built the same way.
- Code that lost its test is removed and kept at an archive tag; the decisions record keeps naming it.

## Traps that cost a day

1. `defense` in the ratings tables is points allowed, negative good; `rating_def` is positive good.
2. A season-total input leaks into the within-season folds: whole-season inputs read z -31.
3. A long bash heredoc fails in this shell: write the script to a file first.  And a doubled backslash typed in a
   heredoc arrives as one, so a regular expression written there loses its escapes.
4. A background job is stopped after its limit: long runs save as they go and resume.
5. A year-over-year gain can be width: read it with each side's width matched to the incumbent's.
6. `--config` and `OPENRAPM_CONFIG` do not move the root, which comes from where the code sits: a scratch config needs
   absolute paths.  A product build's WOSA processes start without `--config`: use `OPENRAPM_CONFIG`.
7. A finished season's stints on disk differ from a rebuild in the expected-makes columns by 8e-11, the shot curve
   having been fitted in memory through another config: compare two builds, not a build with the disk.
8. The job's log is written in the console's code page: a print of a name with ć or č from a plain script stops it.
9. Python's `urllib.robotparser` matches a crawler's name anywhere inside a longer one, so it reads LinkedInBot as the
   AI crawler "nbot" and bars it; crawlers match the whole name (RFC 9309), and `tests/test_site_pages.py` reads the
   file that way.
10. More in DECISIONS.md, "The measurement traps".

## Verify you are where this file says

    .venv/Scripts/python -m pytest -q        # every test passes or skips; 1 expected failure
    uvx ruff@0.17.0 check .                  # clean
    cat outputs/daily/last.txt               # the last morning: published, nothing new, or FAILED
    gh run list --repo bbstats/openrapm --branch main --limit 3      # GitHub's tests on main: success
