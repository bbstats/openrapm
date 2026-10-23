# Handoff

**What this file is.**  Where the work stands, what waits on the owner, and the commands a session runs.  Rewritten
on the afternoon of 2026-10-22, its status and next steps on 2026-10-23 (twice); the earlier file is in git history.  `DECISIONS.md` is the permanent record and carries
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

## Where it stands, 2026-10-23

- **The bug-fix pass of 2026-10-22** (DECISIONS.md, "The bug-fix pass"): the owner asked for a hunt for bugs behind the
  artifacts; the findings, the code-only fixes, experiments 94-99, the width test and the season widths are there.
- **The incumbent is experiment 106 (layers106), PUBLISHED 2026-10-23** (the owner: "please adopt, yes - D. you may
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
- **Deferred by the owner**: rating the season in progress (DECISIONS.md, "The bug-fix pass").
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

## Next

The offense/defense balance is fixed and published (experiment 106, version D).  Nothing is queued; the owner brings
the next idea.  What the work of 2026-10-23 left open:
- Your two tests use different units and disagree on how far apart teammates should be: the within-season WOWY test
  scores stints, the year-over-year test whole team-games, and minutes that react to the game pull the team-game
  reading down (DECISIONS.md, "Why the shrink cuts the offense: the unit it is fitted on").  Version D cost the
  year-over-year test +0.16, z +4.4, adopted by the owner anyway.  A stint-level year-over-year reading already exists
  in `63_yoy.py`'s output; which unit rules the guard is the owner's.
- Before 2017 version D widens the defense, 1.73 to 1.92 and 1.63 to 1.78 by era: the balance moved only in 2017-2026.
- Parked, not next: experiment 104, the band-difference un-shrink; experiment 103, the line refit alone;
  width-matched readings in the chain; a noise control on the incumbent; the EM ridge idea.

## Waiting on the owner

- Deleting the saved boosters of experiments that were not adopted, about 2 GB each in `outputs/prior_models_*.pkl`
  (luckall88, luckall88b, luckhalf89, luckhalf89b, near83, same84, ctx85, body86, spread90, three92, unshrinkbase102,
  unshrink102, shrunkbase103 and older ones; refit103's stays while experiment 104 is open).
- Filed earlier: tracking game by game; arena scorekeeper checks; Stockton over Jordan in 1997-98; the Curry star-guard
  ceiling test fails.

## What ships

| step | plain name | module | as shipped |
|---|---|---|---|
| 1 | the label | `rapm.py` | one RAPM per player over every season but the rated one; penalties 40,000 / 40,000 / 0 |
| 2 | box-score prior | `box_prior.py` | LightGBM `lgb1_tune2`, linear leaves; feature set `hustle_no_three_pct`; five player folds; 10 bags |
| 2b | swap part of the prior | `second_prior.py` | line `ctx82` on `context_and_types_centred`, team totals held |
| 3 | the rating | `prior_rapm.py` | RAPM pulled toward the prior, penalty 13,037, centred |
| 4 | the shrink, with the defense factor | `prior_shrink.py` | both parts multiplied by what holds up on held-back games, 8 nearest fold seasons, defense times 0.89 before 2017 and 0.84 from 2017; folds `inputs99_early,inputs99_within`, each multiplier read to a full season with the tenth folds `inputs99_k10` (experiment 101); two layers since experiment 106: each part's team mean on held-back team-games, each player's deviation from it on held-back stints within games, the within layer times 0.75 on offense and 1.0 on defense |
| 5 | swap adjustment | `swap_adjust.py` | off: applied share 0, the build skips it |

- `openrapm build` writes `outputs/season_ratings_product.parquet` with `_step3` and `_step4` beside it.
- `openrapm site --bias --openrapm-decomposition --similarity` writes the site's files.  Commit, fast-forward `main`
  and push only on the owner's word, then check the Pages build: `gh api repos/bbstats/openrapm/pages/builds/latest`.
  With the second prior, the decomposition's "box score" piece carries the whole prior; its split is the owner's.
- The incumbent's test-setting tables `outputs/season_ratings_layers106*.parquet`, folds `inputs99_early`, `inputs99_within`;
  no noise control on it yet, `noise_bags10` stands in.

## How an experiment is judged

| reading | script | rule |
|---|---|---|
| within-season WOWY test | `scripts/checks/151_wowy.py`, stint unit, step 4 as shipped | better in every era, against the incumbent; the noise control beside it |
| year-over-year test | `scripts/checks/63_yoy.py` | not worse at z +2 |
| the eye check | `scripts/checks/66_compare.py` | the 2026 top 20 |

- One build: `bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes_late_as_scored "EXTRA RATE
  FLAGS"`, about 50 minutes with 4 processes.  `INC_NAME` and `INC_SYSTEM` name the incumbent's tables and folds, `layers106` and `inputs99` by default.
- **The season-page score**: each season, ratings of zero score 0 and the reference 100; the ratings before
  experiment 82 explain 18.2 a season on this test (the 15.7 in older notes is experiment 63's step 4).
  `scratch/2026-10-20_charts/season_scores.py` and `season_page.py`; page https://claude.ai/artifact/L7vwSTAvCCKQnj9jnnhYrq.
- The noise control `noise_bags10` moves a season at most 0.7 on that score.
- Every experiment gets a row in `experiments/runs.csv` (`scripts/checks/101_log_run.py`), mirrored to the sheet
  "OpenRAPM experiments", tab Experiments; the next free row is 100.

## Commands a session runs

    bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes_late_as_scored "EXTRA RATE FLAGS"
        one experiment against the incumbent, about 50-75 minutes; log outputs/NAME_chain.log
    .venv/Scripts/python scripts/checks/151_wowy.py --systems=inputs99,noise_bags10,NAME --ref=inputs99 --tag=NAME
        the within-season WOWY test on folds already built, about 3 minutes
    .venv/Scripts/python -m openrapm second-prior --models bags10 --folds bags10_early,bags10_within --tag TAG --inputs SET
        a second prior's line from held-back swaps, about 13 minutes
    .venv/Scripts/python scripts/checks/101_log_run.py --name=NAME --exp=N --incumbent=inputs99 --idea=owner --verdict=pending --changed="..." --notes="..."
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
- Never commit, push or publish without the owner's word.  The site is `docs/`, served from `main`.
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

    .venv/Scripts/python -m pytest -q                       # 755 passed, 2 skipped, 1 xfailed: the backup bigs
    uvx ruff check .                                         # clean
    git log --oneline -1                                     # the publish of experiment 106
