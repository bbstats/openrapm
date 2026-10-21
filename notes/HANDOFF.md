# Handoff

**What this file is.**  Where the work stands, what waits on the owner, and the commands a session runs.  Rewritten
whole on 2026-10-21; the earlier file is in git history.  `DECISIONS.md` is the permanent record and carries every
number quoted here; read its measurement traps before trusting a number.  `README.md` says what OpenRAPM is,
`PIPELINE.md` how each step works, `CONTRIBUTING.md` the writing rules, `CLAUDE.md` the owner's standing rules.

## Talking to the owner

- American English in everything shared: defense, offense, color.  Code names keep their spelling.
- No step numbers in messages.  The names: **box-score prior** (the LightGBM, also "preliminary prior"); **+/-** (what
  the season's games add on top of the prior); **swap adjustment** (the old after-the-fact correction from lineup
  swaps, off since experiment 82); **swap part of the prior** (experiment 82's line, learned from the same swaps,
  added before the games are weighed); **defense dial** (step 4's extra 0.866 on defense).  Define each the first
  time it appears.
- Experiments by name in chat, numbers only in the record files ("the three-input prior", not "experiment 92").
- Answer exactly what is asked.  Results as tables, one quantity per column, no parentheses, one TL;DR sentence.
- Within-season WOWY is what counts, and a win must hold in every era: 1997-2006, 2007-2016, 2017-2026.

## Where it stands, 2026-10-21

- **The incumbent is experiment 82**, adopted 2026-10-20 ("adopt for now"): step 3 pulls toward a second prior, the
  box-score prior plus a line fitted on held-back lineup swaps that reads teammates' and opponents' box-score
  priors, the player's other-side box-score prior and the swap adjustment's 19 box-score, role and body numbers, team
  totals held; the swap adjustment off; step 4 on 82's own folds.  These are the shipped defaults
  (`settings.py`); the chain's incumbent is `ctx82`.
- **The product table is built with experiment 82** (`outputs/season_ratings_product.parquet`, 2026-10-20; the old
  one kept as `season_ratings_product_before_ctx82*`).  **Not published**: the site still shows experiment 80.  The
  owner, on seeing 82's ratings: "we can't publish an even stronger defense".  `docs/` was restored to the live
  files; rebuild it with `openrapm site` when publishing.
- **Experiment 82's defense is wider**: defense spread 1.69 / 1.59 / 1.49 by era against 1.38 / 1.32 / 1.36 before.
  Against the consensus in 2026 the slope of our defense on theirs is about 1; the extra spread is players rated
  differently, carried by big-man stats: blocks, height, defensive rebounds, fouls (DECISIONS.md, "Experiment 82's
  defensive prior read through").
- **Two flaws found in experiment 82's line** (same section):
  1. **The ID-group bug.**  The line is fitted five times, each without one player fold (player_id % 5), and has no
     intercept on uncentred inputs, so each fold's line carries its own level.  2026 defense: ids ending 2 or 7
     average +0.50, ending 4 or 9 -0.31.  Fix: centre the inputs on the season's possession-weighted average.
  2. **Cross-side inputs.**  The defensive line reads his offensive box-score prior with weight 0.91 beside 1.03 on
     his defensive one (2026, raw scale), copied from the swap adjustment's inputs.
- **Tried since and not adopted** (DECISIONS.md has each): 86 padding toward height, weight and possessions; 87 the
  swap adjustment reads age as a curve; 88/88b none of the opponents' three-point luck kept; 89b half kept; 90 the
  spread inside teams held; 91 the defense dial at 0.70-0.95 (0.80 narrows defense at a within-season cost); 92 the
  three-input prior, own prior plus teammates' same-side and opponents' other-side sums, which scored like no swap
  correction at all: 82's gain comes from the box-score stats in its line, not from teammates or opponents.
- **Boruta on both stages**: the box-score prior's inputs on 2026-10-16; the second stage on 2026-10-21, a table of
  ten candidates per side by era (DECISIONS.md, "Boruta on the second stage").
- **Faster runs**: `openrapm rate --jobs N` and `openrapm folds --jobs N`, same numbers; the chain asks for 4, cut to
  what free memory holds.  A chain takes about 50 minutes instead of 1 hour 40.

## Next: experiment 82 with the ID-group fix (the owner, 2026-10-21: "next handoff we will begin experiment 82")

1. Add an input set that is 82's inputs, centred: in `second_prior.py`, `INPUT_SETS["context_and_types_centred"]`
   equal to the `context_and_types` list and the name in `CENTRED_SETS`; in `settings.py` the name in
   `SECOND_PRIOR_INPUTS`.  `side_inputs` must treat it as `context_and_types` (it reads `types`), and
   `season_second_priors` already centres any set in `CENTRED_SETS`.  The grams of `ctx82` fit it unchanged: the line
   is fitted on swap differences, which centring does not change, and `line_setup` compares the input lists.
2. A test: on uncentred inputs two fold lines that differ by a constant give the average player different values,
   centred they do not.
3. `bash scripts/experiment_chain.sh ctx82c points_with_expected_opponent_threes "--second-prior-inputs context_and_types_centred"`
   about 50 minutes, against ctx82.
4. Read by era, with each side's width (`scratch/2026-10-20_defence/defence_width_table.py ctx82c ctx82`), the fold
   means of the prior part (`scratch/2026-10-21_charts/fold_level.py`, pointed at ctx82c), and the season-page score
   (`scratch/2026-10-20_charts/season_scores.py`).
5. Then, if the owner wants it: the same without the other side's own box-score prior.

## Waiting on the owner

- What to publish: experiment 82 as it is, 82 with the defense dial at 0.80, experiment 90, or hold; after the
  ID-group fix is read.
- The EM ridge idea, a larger thread "not now": the ridge penalty by EM per season and side instead of the fixed
  13,037.
- Deleting the saved boosters of experiments that were not adopted, about 2 GB each in `outputs/prior_models_*.pkl`
  (luckall88, luckall88b, luckhalf89, luckhalf89b, near83, same84, ctx85, body86, spread90, three92 and older ones).
- Filed earlier: tracking game by game; arena scorekeeper checks; Stockton over Jordan in 1997-98; the Curry star-guard
  ceiling test fails.

## What ships

| step | plain name | module | as shipped |
|---|---|---|---|
| 1 | the label | `rapm.py` | one RAPM per player over every season but the rated one; penalties 40,000 / 40,000 / 0 |
| 2 | box-score prior | `box_prior.py` | LightGBM `lgb1_tune2`, linear leaves; feature set `hustle_no_three_pct`; five player folds; 10 bags |
| 2b | swap part of the prior | `second_prior.py` | line `ctx82` on `context_and_types`, team totals held |
| 3 | the rating | `prior_rapm.py` | RAPM pulled toward the prior, penalty 13,037, centred |
| 4 | the shrink, with the defense dial | `prior_shrink.py` | both parts multiplied by what holds up on held-back games, 8 nearest fold seasons, defense times 0.866; folds `ctx82_early,ctx82_within` |
| 5 | swap adjustment | `swap_adjust.py` | off: applied share 0, the build skips it |

- `openrapm build` writes `outputs/season_ratings_product.parquet` with `_step3` and `_step4` beside it.
- `openrapm site --bias --openrapm-decomposition --similarity` writes the site's files.  Commit, fast-forward `main`
  and push only on the owner's word, then check the Pages build: `gh api repos/bbstats/openrapm/pages/builds/latest`.
  With the second prior, the decomposition's "box score" piece carries the whole prior; its split is the owner's.
- The incumbent's test-setting tables `outputs/season_ratings_ctx82*.parquet`, folds `ctx82_early`, `ctx82_within`;
  no noise control on it yet, `noise_bags10` stands in.

## How an experiment is judged

| reading | script | rule |
|---|---|---|
| within-season WOWY test | `scripts/checks/151_wowy.py`, stint unit, step 4 as shipped | better in every era, against the incumbent; the noise control beside it |
| year-over-year test | `scripts/checks/63_yoy.py` | not worse at z +2 |
| the eye check | `scripts/checks/66_compare.py` | the 2026 top 20 |

- One build: `bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes "EXTRA RATE FLAGS"`, about
  50 minutes with 4 processes.  `INC_NAME` and `INC_SYSTEM` name the incumbent, `ctx82` by default.
- **The season-page score**: each season, ratings of zero score 0 and the reference 100; the ratings before
  experiment 82 explain 18.2 a season on this test (the 15.7 in older notes is experiment 63's step 4).
  `scratch/2026-10-20_charts/season_scores.py` and `season_page.py`; page https://claude.ai/artifact/L7vwSTAvCCKQnj9jnnhYrq.
- The noise control `noise_bags10` moves a season at most 0.7 on that score.
- Every experiment gets a row in `experiments/runs.csv` (`scripts/checks/101_log_run.py`), mirrored to the sheet
  "OpenRAPM experiments", tab Experiments; the next free row is 87.

## Commands a session runs

    bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes "EXTRA RATE FLAGS"
        one experiment against ctx82, about 50 minutes; log outputs/NAME_chain.log
    .venv/Scripts/python scripts/checks/151_wowy.py --systems=ctx82,noise_bags10,NAME --ref=ctx82 --tag=NAME
        the within-season WOWY test on folds already built, about 3 minutes
    .venv/Scripts/python -m openrapm second-prior --models bags10 --folds bags10_early,bags10_within --tag TAG --inputs SET
        a second prior's line from held-back swaps, about 13 minutes
    .venv/Scripts/python scripts/checks/101_log_run.py --name=NAME --exp=N --incumbent=ctx82 --idea=owner --verdict=pending --changed="..." --notes="..."
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

    .venv/Scripts/python -m pytest -q                       # 723 passed, 2 skipped, 1 xfailed: the backup bigs
    uvx ruff check .                                         # clean
    git log --oneline -1                                     # the commit of 2026-10-21's cleanup
