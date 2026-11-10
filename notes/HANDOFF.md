# Handoff

**What this file is.**  Where the work stands, what waits on the owner, and the commands a session runs.  Rewritten on
2026-11-09, at the end of the improvements round; the file before it, a thousand lines of dated history, is in git at
f4a5cba.  `DECISIONS.md` is the permanent record and carries every number quoted here; read its measurement traps
before trusting a number.  `README.md` says what OpenRAPM is, `PIPELINE.md` how each step works, `CONTRIBUTING.md` the
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

## Where it stands, 2026-11-09 night: the improvements round

The owner: "improvements, go."  DECISIONS.md, "The improvements round of 2026-11-09", holds every number.  The owner's
four answers: the first experiment "Box Adj's weights"; what may go live without asking again, "Every fix that passes";
a failed morning reaches the owner by "Phone push"; the site additions "Date on Ratings", "Faster first load" and "Link
preview and icon".

**Live since tonight, no rating moved.**

- **Similarity shows the published ratings again.**  It had shown the ratings of 2026-11-04 for three days: the
  five-zone change went live through the morning job, which did not write Similarity's files.  The job now rebuilds them
  on any morning the finished seasons' ratings changed.
- **main is the working tree's code, and GitHub's tests are green**, for the first time since 2026-11-06.  15 pushes
  in a row had failed.  `bash scripts/publish_check.sh` before every push runs what GitHub runs.
- **The site.**  Find forgives a trailing space and a missing hyphen, apostrophe or period.  The Power tab no longer
  comes in late and pushes the other tabs.  "Download CSV: this view" follows the screen.  The Ratings page opens on a
  list file a quarter the size, 0.29 MB against 1.18 compressed, and says "through Nov 8, 2026" on the season in
  progress.  On a phone the buttons and headers are 40 pixels.  The Similarity page says so when a file fails to load.
- **The morning job** checks that no game is missing, is the only thing that may write the published tables, holds a
  lock, sends a phone notice when it fails, and stops after the Finals.  Its log is 85 lines a stints step, not 3,300.

**In the tables since that night, on the site with the morning job of 2026-11-10.**  Read that morning's log first:
it is the first with the games check, the lock and the notice, and it carries both changes below.

- **WOSA reads the regular season alone** (2b71432).  The rating behind WOSA had read Box Adj's lineup inputs from
  playoff stints too.  Every finished season rebuilt; no rating of the published table moved.  WOSA moves a median
  0.026 wins and at most 0.85, Roy Hibbert 2014.  One season's leader changes: 2026, Gilgeous-Alexander 14.26 over
  Jokić 14.10, where Jokić had led 14.227 to 14.225.  To take it back: the three backups named `*_before_wosafix*`
  under `outputs/`, and revert 2b71432.
- **The Power table's weights refitted on the ratings the page multiplies** (`params/power_forecast.json`).  The
  weights of 2026-11-09 were fitted on ratings shrunk on their own early games; the page multiplies ratings under the
  whole season's shrink.  Five points of the season now carry weights, the fifth at three quarters.  The registered
  rule holds at both test days, at 62.5% of a season by a hundredth of a z; in 2017-2026 the line there is worse than
  a fit at the day itself, z +2.30.  The defense multiplier through a tenth of a season is 1.75 to 1.88 where it was
  1.20 to 1.25: the table's spread widens by a third, San Antonio +6.57 to +8.91, and its order holds, no team moving
  more than 2 places.  The weights file before it is in git at 143e24d.

**The experiment, Box Adj's weights: read, nothing built.**  Two reads from the saved sums of the shipped line.  No
penalty and no count of nearest seasons passes the registered rule.  A stronger penalty helps 2017-2026 at every
strength up to 0.1 and hurts the two earlier eras; the 16 nearest seasons gain 0.130 over all 30 seasons and lose
2007-2016 by 0.117.  Both ends of the 30 seasons want weights learned nearer themselves and the middle does not, which
is what drift in the weights would give.  Proposed, not run: each training season weighted by how near it is to the
rated one, read from the same sums in minutes.  The owner's call.

## Waiting on the owner

1. **The phone notice.**  Install the ntfy app and subscribe to the topic in `outputs/daily/notify.conf`, given to the
   owner in chat on 2026-11-09.  Until then a failure posts to a topic nobody reads.  A test line:
   `bash scripts/daily_notify.sh "test"`.
2. **The link preview and tab icon.**  The candidate picture and icon are drawn, not on the site: a blue square with a
   white plus-minus mark, and "OpenRAPM" with the address.  They need a yes, and the preview needs one sentence of the
   owner's wording.
3. **Box Adj's weights**: whether to read the weights with each training season weighted by its distance.
4. **Changelog rows**, in the owner's words, for: the rookies button, WOSA, the page opening on the season in progress,
   the Detail view, Detail's Leverage and WOSA columns, Similarity's search box, the Power page's one rating; and from
   this round the Similarity correction, the WOSA correction and the Power table's new weights.
5. **Page text that no longer matches the code.**  About: "Share of league games played", held at a quarter while a
   season is in progress; "vs league average" on the hustle rows only, though the box inputs are centered too; "All
   stats use padding method" beside ten "Not padded" rows; a Changelog date of 2026-10-31 for an entry first live
   2026-11-01.  Power: the column tips leave out the Last 10 row, the multipliers and blowout possessions counting
   half.  Ratings: the Leverage tip says close games where the code compares moments inside each game.
6. **`docs/progress.png` and `docs/progress.csv`**: linked from nowhere, reachable at openrapm.com, full of internal
   labels.  Delete them?
7. **Verdicts still open in the experiments table**: defense's boosters at 330 trees, a tie, recommended off; the
   network and Kumo priors; self-creation with estimated passes; the play-by-play and matchup input screens; the team
   rates without the direct plus-minus.  And whether to delete the saved boosters and builds of those not adopted,
   about 15 GB under `outputs/second_prior/` and `outputs/prior_models_*.pkl`.
8. **Sheet "OpenRAPM experiments", row 66**: a stray copy of the early-season padding row.  Clearing it was refused to
   me as a write to an outside system; it takes one click.
9. Older, still the owner's to ask for: EPM or xPM on the Bias page through a key; regular-season possessions beside
   WOSA; a win-probability rule for garbage time; the ridge penalty read on the within-season test.

## What ships

| part | plain name | module | as shipped |
|---|---|---|---|
| step 1 | the label | `rapm.py` | one RAPM per player over every season but the rated one; penalties 40,000 / 40,000 / 0; the defensive label un-shrunk |
| step 2 | box-score prior | `box_prior.py` | LightGBM `lgb1_tune2`, linear leaves; inputs `hustle_no_three_pct_self_creation`; five player folds; 10 bags; part-season rows at a quarter, a half and three quarters |
| step 2b | Box Adj | `second_prior.py`, `rating_line.py` | line `zones139_line` on `context_types_team_zones_centred`, 57 inputs an end, weights fitted on the within-season test through the ridge, team totals held, penalty 0.0001, every season; the Detail view's Box score column learned first (`box_first`) |
| step 3 | the rating | `prior_rapm.py` | RAPM pulled toward the prior, penalty 13,037; the season in progress holds its prior scales at the 8 nearest finished seasons' mean through a quarter of its season, then a straight line to its own at three quarters |
| step 4 | the shrink | `prior_shrink.py` | both parts multiplied, 8 nearest fold seasons, folds `zones139_early,zones139_within`, read to a full season with `zones139_k10`; two layers, the within layer times 0.7663 on offense and 0.9968 on defense; defense factor 0.8682 before 2017, 0.8704 from 2017 |
| step 5 | swap adjustment | `swap_adjust.py` | off: applied share 0 |
| | WOSA | `wins.py` | wins over the 150 most-started players' average, from the regular-season rating alone, 30 rating points a win |
| | Power | `power_forecast.py`, `data/power_files.py` | Power at its multipliers plus the last-10 stats; weights at five points of the season in `params/power_forecast.json`, refitted 2026-11-09 by check 166 |

`settings.py` is the source for every number above and names the experiment that set each.

## The morning job

`scripts/daily_update.sh`, Task Scheduler "OpenRAPM daily update", 6:00 AM, from the working tree's code; about 13
minutes.  README.md, "The season in progress, every morning", lists its commands and what protects it.

- **It runs the working tree.**  No edit under `src/` or `scripts/` between 5:50 and 7:30 AM.  After any change to
  the job, the stints, the build or the site step: `PUBLISH=0 bash scripts/daily_update.sh`, 12 minutes, which writes
  the site's files to `outputs/daily/site_dry` and says how it ended in `outputs/daily/last_dry.txt`.
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
  season, the site's files, a refit of `params/power_forecast.json`, and the publish on the owner's word.  The last
  Finals game reaches the site only through that build.  In October 2027 raise `last_season` in `config.yaml` and
  `SEASON` in the job.
- **The first playoff morning of 2027** is the WOSA lock's first live use: 2027's WOSA should stop moving.

## Publishing

The site is `docs/`, served from `main`.  Nothing is pushed without the owner's word, but for the morning job's data
and, in the round of 2026-11-09 only, the fixes the owner allowed.

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

## How an experiment is judged

| reading | script | rule |
|---|---|---|
| within-season WOWY test | `scripts/checks/151_wowy.py` | better in every era against the incumbent, the noise control beside it |
| year-over-year test | `scripts/checks/63_yoy.py` | not worse at a paired z of +2, read at each side's width matched to the incumbent's |
| the eye check | `scripts/checks/66_compare.py` | the 2026 top 20 |

- The incumbent is `zones139`: tables `outputs/season_ratings_zones139*`, folds `zones139_early`, `zones139_within`,
  year-over-year score 8.6901.  Its noise control `noise_zones139` reads -0.009 on the within-season test, paired z
  -1.03, 16 of 30.  The plain z rules.
- One full build: `bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes_late_as_scored "EXTRA
  RATE FLAGS"`, 50 to 75 minutes, log `outputs/NAME_chain.log`.  It waits while the morning job runs.
- A Box Adj input or setting is read first from the saved sums, in minutes
  (`scratch/2026-11-09_improvements/box_adj_weights.py`, `scratch/2026-11-05_second_stage/zones_screen.py`), then
  confirmed by a full build.
- Register a reading in DECISIONS.md before any result.  Every experiment gets a row in `experiments/runs.csv`
  (`scripts/checks/101_log_run.py`), mirrored to the sheet "OpenRAPM experiments", tab Experiments, written at the first
  free row after the last, never appended.  The next free row is 141.
- Report in four parts: what changed, the result, the 2026 top 20, the owner's call.

## Leads, none started

- **The Power table from half a season to three quarters.**  The straight line between those two points passed its
  rule by a hundredth of a z and loses 2017-2026 to a fit at the day.  Read it again before the season in progress
  gets there, about the middle of January 2027, with weights fitted at 62.5% as the candidate
  (`scripts/checks/166_power_forecast_refit.py`).
- **Early-season defense.**  2027's defense is 0.60 as wide as its offense among players with 250 or more possessions,
  1.05 against 1.76; finished seasons read 0.87, 0.85 and 0.82 by era.  And the hustle rates are padded as the box
  rates were before the early-season fix: toward the raw rate of players with similar possessions, the pad measured
  around the league mean (`data/dashboard_rates.py`).  The Power refit met it from the team side: a team's defense
  wants a multiplier of 1.75 to 1.88 through a tenth of a season.  First a read on finished seasons rated from their
  first tenth.
- **The ridge penalty**, 13,037, set on the year-over-year test in September and never read on the within-season
  test; the owner's EM idea.  It has no flag, and the Box Adj sums carry it.
- **Numbers fitted once and since overtaken**: the width settings, set before the zones narrowed the defense; WOSA's
  30 points a win; the 0.25 of opponents' three-point luck kept; `lgb1_tune2`, tuned before Box Adj existed.
- **2027's play-by-play** forces 0.60 possession switches a game against 0.30 over 2026's first 146: look for a change
  in the league's feed.
- **Three neutral-site games of 2024** carry a home team in the game log: 0022301229, 0022301230 and 0022300527.
- **The morning job** rebuilds all 31 seasons' panel rows and site pieces each morning, about 6 of its 13 minutes.
  Its scheduled task runs only while the owner is logged on.  A record of what built the stored tables is not written.
- **Check scripts that cannot run on today's build as they stand**: `155_input_screen.py`, whose reference is
  creation110, and `152_decider_tune.py`, whose pipeline has no Box Adj.

## Commands a session runs

    .venv/Scripts/python -m pytest -q                        # every test, about 2 minutes
    uvx ruff@0.17.0 check .                                   # the lint check, at the version GitHub pins
    PUBLISH=0 bash scripts/daily_update.sh                    # the morning job, nothing published, 12 minutes
    bash scripts/publish_check.sh                             # before any push to main
    bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes_late_as_scored "EXTRA RATE FLAGS"
    .venv/Scripts/python scripts/checks/151_wowy.py --systems=zones139,noise_zones139,NAME --ref=zones139 --tag=NAME
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
  runs: each new python process imports the code as it is then.
- Commit every piece of work; never push or publish without the owner's word, but for the morning job.
- Movement in points per 100, never rank places; under 0.1 is nothing.
- A proof that a change moves no number is a byte comparison with both sides built the same way.

## Traps that cost a day

1. `defense` in the ratings tables is points allowed, negative good; `rating_def` is positive good.
2. A season-total input leaks into the within-season folds: whole-season inputs read z -31.
3. A long bash heredoc fails in this shell: write the script to a file first.  And a doubled backslash typed in a
   heredoc arrives as one, so a regular expression written there loses its escapes.
4. A background job is stopped after its limit: long runs save as they go and resume.
5. A year-over-year gain can be width: read it with each side's width matched to the incumbent's.
6. `--config` and `OPENRAPM_CONFIG` do not move the root, which comes from where the code sits: a scratch config needs
   absolute paths.
7. A finished season's stints on disk differ from a rebuild in the expected-makes columns by 8e-11, the shot curve
   having been fitted in memory through another config: compare two builds, not a build with the disk.
8. The job's log is written in the console's code page: a print of a name with ć or č from a plain script stops it.
9. More in DECISIONS.md, "The measurement traps".

## Verify you are where this file says

    .venv/Scripts/python -m pytest -q        # 1,132 pass, 2 skipped, 7 deselected, 1 expected failure, about 2 minutes
    uvx ruff@0.17.0 check .                  # clean
    cat outputs/daily/last.txt               # the last morning: published, nothing new, or FAILED
    gh run list --repo bbstats/openrapm --branch main --limit 3      # GitHub's tests on main: success
