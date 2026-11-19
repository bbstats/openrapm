# Handoff

**What this file is.**  Where the work stands, what waits on the owner, and the commands a session runs.  Rewritten on
2026-11-16, at the end of the cleanup pass; the file before it is in git at 2c26966.  `DECISIONS.md` is the permanent
record and carries every number quoted here; read its measurement traps before trusting a number.  `README.md` says
what OpenRAPM is, `PIPELINE.md` how each part works, `GLOSSARY.md` what each word means, `CONTRIBUTING.md` the
writing rules, `CLAUDE.md` the owner's standing rules.

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
- Decisions go through the question tool, with the numbers in the question and one outcome per option.
- I write no prose for the site.  Wording is the owner's; a page change of mine adds data, a column, or a line the
  owner has already approved elsewhere.

## Where it stands, 2026-11-17: the data deleted, fetched again, rebuilt

**What happened.**  On 2026-11-16 at about 9:30 PM, `git worktree remove --force` on five old side checkouts deleted
`A:/code/spmm/data` and `outputs` through the checkouts' junctions.  Drive A: is an NVMe SSD with TRIM, so nothing
could be recovered.  The data was fetched again on 2026-11-17 (`scratch/2026-11-16_cleanup/refetch.sh`) and the ratings
rebuilt by README.md's recipe on every season (`rebuild.sh`), the owner's choice.  DECISIONS.md, "The cleanup of
2026-11-16", has the account.  **Never remove a side checkout before unlinking its junctions.**

- **The shipped settings read the rebuild until 2026-11-19:** Box Adj's line `rebuilt_line`, the shrink's held-back
  games `rebuilt_early,rebuilt_within` and `rebuilt_k10`; since then experiment 142's, below.  The rebuild's test-setting tables and saved models are `rebuilt`;
  pass 1, without Box Adj, `rebuilt_base`.
- **Checked against the site before publishing** (the owner: "No publishing till we do testing on the data"): finished
  seasons within 0.05 points per 100 typical, 0.29 at most; stints equal to the surviving copy in 60 of 61
  season-phases; every game present.  `outputs/rebuild/against_site.txt`, `data_tests.txt`, `pieces_vs_site.txt`.
- **Backup:** `D:\openrapm_backup\2026-11-17\data` and `outputs`, verified byte for byte.
- **Lost and not rebuilt:** every old experiment's saved files, the earlier incumbent `fade141` and its noise control
  `noise_fade141` among them; `outputs/daily/notify.conf`, the phone notice's topic.

## Where it stands, 2026-11-19: age is out of Box Adj, adopted and published

DECISIONS.md holds every number, from "The data deleted and rebuilt" to "Experiment 142: adopted".

- **The research paths, ranked** (the owner, 2026-11-18): the plan
  `C:\Users\Nathan\.claude\plans\what-are-some-possible-zippy-quill.md`, and DECISIONS.md, "The research paths of
  2026-11-18".  The owner's answer on the first: "box adj is good but guess what - I want age removed! :) the rest is
  fine."  **Age is not an input to any rating, and is offered to no screen or tree stage.**
- **Experiment 142, age out of Box Adj's line: adopted and published on the owner's word, "Adopt and publish".**  The
  shipped line is `noage142_line` on `context_types_no_age_team_zones_centred`, 56 inputs an end; the shrink reads
  `noage142_early,noage142_within` and `noage142_k10`; the published tables are the full build `product142`'s, the
  ones before them kept as `*_before_noage142*`.  Within-season +0.027, z +1.99, 10 of 30, by era +0.051, +0.041,
  -0.011; year over year at matched width z +2.18, all of it a season's ratings asked about the season after it.
  Published against the site before it: regulars' totals move a median 0.02 and 0.25 at most; no season's top-rated
  player changes; two WOSA leaders change by a hair, 1999 Duncan to Mutombo and 2003 Nowitzki to Garnett.  The Power
  table's weights stay.
- **The rebuild of 2026-11-17 read on the tests** before any of it: year over year 8.6891 against the earlier
  incumbent's 8.6908; check 151's errors on it 3,455.273, 3,595.240 and 3,932.011 by era; its noise control
  `noise_rebuilt` -0.012.
- **Next, approved by the owner:** the reads on Box Adj without age, from saved held-back games, no rating built:
  more inputs through the direct screen, close fourth-quarter share and minutes per game among them; what the LightGBM
  is worth beside the line; the box-score prior's weight by share of team possessions; trees on what the line leaves,
  several steps.  The tools must be pointed at `rebuilt_base_early`, `rebuilt_base_within` and the shipped line's
  folder, and given the fade's season weights: `scratch/2026-11-03_direct/direct_screen.py`,
  `scratch/2026-11-10_boost_vs_line/player_sums.py` and `boost_vs_line.py`.  Report and stop.
- **The noise control on the new incumbent, `noise_noage142`:** -0.013 on the within-season test, by era -0.032,
  -0.010, +0.004.  The chain names `noage142` and it since 2026-11-19.
- **Left from the adoption:** the swap adjustment's unused model, `swap_adjust.PANEL_FEATURES`, still lists age: the
  owner's one-line choice.  Row 142 is in the experiments sheet, row 139 of its Experiments tab.  Two commits after
  the publish, the record and the chain's incumbent, are local and not pushed.

## Where it stood, 2026-11-16

**The cleanup pass** (the owner: "clean up the documentation / readme / and do a general cleanup pass", then "a strict
robots ... + please give Nathan some credit").  DECISIONS.md, "The cleanup of 2026-11-16", holds every number.

- **Live since 2026-11-16, main 9effe2d:** `docs/robots.txt` bars the 207 AI crawlers of the ai.robots.txt list at
  be22abe from the whole site; search engines, Meta's link previews and Apple's search crawler stay allowed (the
  owner's choice); Content-Signal search=yes, ai-input=no, ai-train=no.  Every page's footer carries the owner's line,
  "Ratings by Nathan Walker. Use them with credit and a link to openrapm.com. No bulk copying or scraping.", and every
  head the text-and-data-mining reservation.  `LICENSE-RATINGS.md` gives the ratings their own terms; the code stays
  MIT.  `CITATION.cff`.  `data/external/impact_metrics_2526.csv`, other makers' ratings, untracked; `docs/progress.png`
  and `progress.csv` deleted.
- **Docs rewritten to what ships**: README, PIPELINE, GLOSSARY, CONTRIBUTING, this file, the held-back-games ledger.
  Box Adj has its own section everywhere; the swap adjustment is written as off; the build recipe builds Box Adj's line.
  `tests/test_docs.py` fails when a doc leaves out a check or a module, or names a stale setting or incumbent.
- **Code that lost its test removed** (the owner: "Take all three out"): the game-by-game network prior (experiments
  114-117), the Kumo prior (118), the elastic-net blend (121-122), and the site command's writer for the retired
  vanilla RAPM page.  About 4,300 lines.  All of it is readable at the tag `archive/research-2026-11`.  A saved model file
  of a game-by-game build is refused by name.  The proof that no number moved is in DECISIONS.md.
- **Comments, docstrings and help text** to what the code does, in American English; proven by comparing every touched
  file's syntax tree with docstrings and help text blanked (`scratch/2026-11-16_cleanup/same_code.py`).
- **The noise control on experiment 141** was read on 2026-11-11: `noise_fade141`, -0.013 on the within-season test,
  paired z -1.41, 16 of 30.  The chain names it.

## Waiting on the owner

1. **Push the rest of the cleanup.**  The docs, the removals and the tests are on the branch `cleanup-2026-11-16`;
   `A:/code/spmm` runs them.  The site part went live on the owner's word.
2. **Google's AI answers.**  The robots file cannot keep the site out of the AI answers inside Google search; that is
   the Search Console setting under Settings, "Search generative AI", in the owner's account, if it is live there.
3. **Delete the old experiments' saved files?**  About 280 GB under `outputs/`: saved models 143 GB in 105 files, of
   which `fade141`, `noise_fade141` and `padbins127` are today's (Box Adj's line was fitted on `padbins127`'s
   held-back games); held-back games 61 GB in 255 folders, of which the `fade141_*` ones are today's; the input screens
   87 GB.  All of that is gone since the deletion of 2026-11-16; what is on disk is the rebuild's and this week's.
   And the old side checkouts `A:/code/spmm_gate`, `spmm_old`, `spmm_split`, `spmm_wosa`, `spmm_verify`, with
   the merged branches `split-robust`, `ui-tweaks`, `wosa-detail`.
4. **The phone notice.**  Install the ntfy app and subscribe to the topic in `outputs/daily/notify.conf`, given to the
   owner in chat on 2026-11-09.  A test line: `bash scripts/daily_notify.sh "test"`.
5. **The link preview's one sentence**, for the six pages' `description` and `og:description`.
6. **A Changelog row for age out of Box Adj**, in the owner's words.
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
  for good goes in `config.yaml`, `points_mismatch_accepted`.
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

- The incumbent is `noage142`: tables `outputs/season_ratings_noage142*`, folds `noage142_early`,
  `noage142_within`, year-over-year score 8.6901.  Its noise control is `noise_noage142`: -0.013 on the within-season
  test, paired z -1.39, 16 of 30, and -0.032, z -3.26 in 1997-2006 alone, so that era is read beside it.  The
  incumbent before it, `rebuilt`, and `noise_rebuilt` are still on disk.  Both are the defaults at the top of `scripts/experiment_chain.sh`.  The plain z is what adoptions have
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
    .venv/Scripts/python scripts/checks/151_wowy.py --systems=noage142,noise_noage142,NAME --ref=noage142 --tag=NAME
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
