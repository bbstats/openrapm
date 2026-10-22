# Handoff

**What this file is.**  Where the work stands, what waits on the owner, and the commands a session runs.  Rewritten
on the afternoon of 2026-10-22; the earlier file is in git history.  `DECISIONS.md` is the permanent record and carries
every number quoted here; read its measurement traps before trusting a number.  `README.md` says what OpenRAPM is,
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

## Where it stands, 2026-10-22, evening

- **The bug-fix pass of 2026-10-22** (DECISIONS.md, "The bug-fix pass"): the owner asked for a hunt for bugs behind the
  artifacts; the findings, the code-only fixes, experiments 94-96 and the width test are there.
- **The incumbent is experiment 95 (heave95), adopted 2026-10-22, in the code, NOT published** (the owner: "might as
  well do the heave fix i guess but - we can push for later publish"): experiment 94 with the threes of a period's
  last two seconds counted as they happened in the defensive target.  The product is built at the next publish.
- **The site shows experiment 94 (parser94), PUBLISHED 2026-10-22** (63b2acc): experiment 93 on stints rebuilt with
  the parser's four fixes (free-throw trips credited to the five on the court at the foul; no possession from a
  defensive rebound with nothing after it; playoff free throws priced on the whole season; the 2020 bubble's regular
  season neutral).  Within-season WOWY test against 93: better in every era, -0.230, z -9.4; at 93's width -0.101,
  z -4.4, still every era: about half the published gain is width (DECISIONS.md, "The widths").
- **Experiment 96, verdict the owner's**: each season's own listed body and the pad k around the bins.  Within-season
  WOWY better in every era, -0.017, z -3.3; the year-over-year test a tie; the ratings barely move, a median 0.02 per
  100.  Built on parser94; beside 95 it needs one more chain, about 55 minutes.
- **The widths, read**: the within-season WOWY test wants every rating wider, the year-over-year test narrower; no
  width passes both, so the shipped width stays (DECISIONS.md, "The widths").
- **The data on this machine**: the rebuilt stints, caches and season panel at the default paths; the old ones are
  `data/stints_before_parser94`, `data/cache_before_parser94`, `outputs/season_panel_before_parser94.parquet`.
  Experiment 96's panel is `outputs/season_panel_b3.parquet` (`scratch/2026-10-22_bugfix/config_inputs.yaml`).
- **New in the reports**: `z_clustered` in the year-over-year test and `z_newey_west` in check 151, beside the plain
  z; which rules is the owner's.
- **Deferred by the owner**: rating the season in progress (DECISIONS.md, "The bug-fix pass").
- **The defense is wider than before experiment 82**: standard deviation 1.73 / 1.61 / 1.60 by era, 2026 1.64, against
  1.38 / 1.33 / 1.36 and 1.53 for experiment 80.  Against the consensus in 2026 the slope of our defense on theirs is
  about 1; the extra spread is players rated differently, carried by big-man stats: blocks, height, defensive
  rebounds, fouls (DECISIONS.md, "Experiment 82's defensive prior read through").
- **The remaining known flaw: cross-side inputs.**  The defensive line reads his offensive box-score prior with weight
  0.91 beside 1.03 on his defensive one (2026, raw scale), copied from the swap adjustment's inputs.
- **Faster runs**: `openrapm rate --jobs N` and `openrapm folds --jobs N`, same numbers; the chain asks for 4, cut to
  what free memory holds.  A chain takes about 50-75 minutes.

## Next: the owner's to pick

- Experiment 96 beside 95: one chain, about 55 minutes, then experiments 95 and 96 published together.
- Width-matched readings in the chain (proposed, DECISIONS.md, "The widths"): every candidate also read rescaled to
  the incumbent's width, so a win is told apart from a width.
- The incumbent without the other side's own box-score prior in the line, about 50 minutes: the first suspect for the
  wider defense.
- A noise control on the incumbent (`--fold-assignment 1`), so its readings have their own.
- The EM ridge idea, a larger thread "not now": the ridge penalty by EM per season and side instead of the fixed
  13,037.

## Waiting on the owner

- Deleting the saved boosters of experiments that were not adopted, about 2 GB each in `outputs/prior_models_*.pkl`
  (luckall88, luckall88b, luckhalf89, luckhalf89b, near83, same84, ctx85, body86, spread90, three92 and older ones).
- Filed earlier: tracking game by game; arena scorekeeper checks; Stockton over Jordan in 1997-98; the Curry star-guard
  ceiling test fails.

## What ships

| step | plain name | module | as shipped |
|---|---|---|---|
| 1 | the label | `rapm.py` | one RAPM per player over every season but the rated one; penalties 40,000 / 40,000 / 0 |
| 2 | box-score prior | `box_prior.py` | LightGBM `lgb1_tune2`, linear leaves; feature set `hustle_no_three_pct`; five player folds; 10 bags |
| 2b | swap part of the prior | `second_prior.py` | line `ctx82` on `context_and_types_centred`, team totals held |
| 3 | the rating | `prior_rapm.py` | RAPM pulled toward the prior, penalty 13,037, centred |
| 4 | the shrink, with the defense dial | `prior_shrink.py` | both parts multiplied by what holds up on held-back games, 8 nearest fold seasons, defense times 0.866; folds `heave95_early,heave95_within` |
| 5 | swap adjustment | `swap_adjust.py` | off: applied share 0, the build skips it |

- `openrapm build` writes `outputs/season_ratings_product.parquet` with `_step3` and `_step4` beside it.
- `openrapm site --bias --openrapm-decomposition --similarity` writes the site's files.  Commit, fast-forward `main`
  and push only on the owner's word, then check the Pages build: `gh api repos/bbstats/openrapm/pages/builds/latest`.
  With the second prior, the decomposition's "box score" piece carries the whole prior; its split is the owner's.
- The incumbent's test-setting tables `outputs/season_ratings_heave95*.parquet`, folds `heave95_early`, `heave95_within`;
  no noise control on it yet, `noise_bags10` stands in.

## How an experiment is judged

| reading | script | rule |
|---|---|---|
| within-season WOWY test | `scripts/checks/151_wowy.py`, stint unit, step 4 as shipped | better in every era, against the incumbent; the noise control beside it |
| year-over-year test | `scripts/checks/63_yoy.py` | not worse at z +2 |
| the eye check | `scripts/checks/66_compare.py` | the 2026 top 20 |

- One build: `bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes_late_as_scored "EXTRA RATE
  FLAGS"`, about 50 minutes with 4 processes.  `INC_NAME` and `INC_SYSTEM` name the incumbent, `heave95` by default.
- **The season-page score**: each season, ratings of zero score 0 and the reference 100; the ratings before
  experiment 82 explain 18.2 a season on this test (the 15.7 in older notes is experiment 63's step 4).
  `scratch/2026-10-20_charts/season_scores.py` and `season_page.py`; page https://claude.ai/artifact/L7vwSTAvCCKQnj9jnnhYrq.
- The noise control `noise_bags10` moves a season at most 0.7 on that score.
- Every experiment gets a row in `experiments/runs.csv` (`scripts/checks/101_log_run.py`), mirrored to the sheet
  "OpenRAPM experiments", tab Experiments; the next free row is 92.

## Commands a session runs

    bash scripts/experiment_chain.sh NAME points_with_expected_opponent_threes_late_as_scored "EXTRA RATE FLAGS"
        one experiment against the incumbent, about 50-75 minutes; log outputs/NAME_chain.log
    .venv/Scripts/python scripts/checks/151_wowy.py --systems=heave95,noise_bags10,NAME --ref=heave95 --tag=NAME
        the within-season WOWY test on folds already built, about 3 minutes
    .venv/Scripts/python -m openrapm second-prior --models bags10 --folds bags10_early,bags10_within --tag TAG --inputs SET
        a second prior's line from held-back swaps, about 13 minutes
    .venv/Scripts/python scripts/checks/101_log_run.py --name=NAME --exp=N --incumbent=heave95 --idea=owner --verdict=pending --changed="..." --notes="..."
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

    .venv/Scripts/python -m pytest -q                       # 724 passed, 2 skipped, 1 xfailed: the backup bigs
    uvx ruff check .                                         # clean
    git log --oneline -1                                     # the commit of 2026-10-21's adoption of experiment 93
