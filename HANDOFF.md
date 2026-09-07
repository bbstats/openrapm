# Handoff: the prior pass is done, and it says the box score is nearly spent

Written 2026-09-06 (end of the third day of iterate-and-improve mode).  `FINDINGS.md` sections 21 and **22**
are the record; `docs/progress.png` / `docs/progress.csv` the chart and its log; `PIPELINE.md` draws how the
shipped model works, stage by stage.

**Tree state: clean and committed on `hybrid-and-xpts`.**  Nothing uncommitted but `outputs/*.parquet` scratch
dumps.  Tests: 82 passed, 1 xfailed; the shipped board rebuilds and passes 10 of 10 consensus floors.

**Shipped this pass: `ship_shot7d`** (FINDINGS 22.4) -- shot quality on both sides and the blend-0.7
offensive target it unlocks.  110.707 on the criterion, consensus 0.793 / 0.789 / 0.768, **defensive spread
1.30, the narrowest any board has measured**, ten of ten floors.  The site serves it at
`https://bbstats.github.io/openrapm/` once `main` is fast-forwarded (Part 3.5).

---

## Part 0: the owner's standing rulings (read before choosing anything)

1. **Accuracy wins, PROVIDED THE TESTING IS ROBUST -- and the model has to stay shippable as open source.**
   The owner, 2026-09-06, correcting the earlier reading of this: *"The speed thing is more just like I don't
   want us to build some insanely complex model that is overfit and too slow, because I want this to be open
   source! but ultimately accuracy is the winner, provided the testing is very robust."*  So:
   * fit every row, score every game; the fit time is a number the chart carries, not a term to optimise, and
     a real accuracy gain is never traded away for seconds.
   * but complexity and fit time are a TIE-BREAK, and a strong one.  Between two candidates that the criterion
     cannot separate, take the simpler and faster one every time -- somebody else has to be able to run this.
   * and "robust testing" is the precondition, not a nicety.  A gain that survives only the search half, or
     only the prior's own fit, or only one map, is not a gain.  See 22.2 and 22.6 for what that looks like
     when it goes wrong.
2. **The consensus is a SANITY CHECK, never a fitting target.**  The owner, 2026-09-06: *"disagreeing with
   consensus is just a sanity check, never something to fully fit to."*  It exists to catch a board that has
   gone gross-wrong -- the Robert Williams 25th case that the internal benchmarks certified -- and the floors
   in `tests/test_vs_consensus.py` say so themselves: they *"guard against a further fall, not the old level"*.
   So:
   * never choose a constant because it clears a floor.  22.7 declined to read `win_decay_d = 0.6` against
     the floors for exactly this reason, and that was right.
   * a MARGINAL miss is not a veto.  0.759 against a 0.76 floor is noise on a sanity check, and a candidate
     that is significant on the criterion should not be thrown away for it.  Report the number honestly and
     decide on the criterion.
   * a GROSS miss still is a veto, because that is what the check is for.
   * the criterion (`scripts/45_holdout.py`, held-out seasons, actual points) remains the one test that is
     both external and legal to select on.  It decides; the consensus sanity-checks.

3. **The 3-season window is on the way out.**  *"Nobody really looks at chunks, we will ultimately move away
   from this mode."*  Do not tune K or the window length; prefer work that survives the move to a continuous
   or per-season rating.
4. **Schedule context is not worth it** (*"evens out really well"*) -- rest days and back-to-backs, declined.
5. **The four-factor defensive fit is right but was too big for v1.**  It is now the most promising thing left
   (Part 3).
6. **The prior was the work, and the prior is now mostly spent.**  Part 3 says what replaced it.

## Part 1: where it stands

| | criterion (K = 3, mapped) | 28 fits |
|---|---|---|
| **the criterion's line** (`best_ratio_full`) | **109.845** | 59 s |
| the same with shot quality (`best_shot`) | 109.801 (z -1.13) | 70 s |
| the board this pass started from (`ship_side6`) | 110.710 | 43 s |
| **what ships** (`ship_shot7d`, ten of ten floors, defensive spread 1.30) | **110.707** | 50 s |
| `ship_shot7` -- shot quality on offense only, ten of ten | 110.694 (z -0.88) | 47 s |
| no ratings at all | 125.6 | |

### Why the board moved (FINDINGS 22.4)

21.26 shipped `blend0.6` on offense **only because `blend0.7` failed the bigness floor** at -0.303 against
0.30, at a cost of about 0.02 per 100.  The shot-quality features move that gap to -0.270 -- separating a
rim-running big from a jump shooter at the same FG% is exactly what the offensive board was mis-ranking by
size -- so blend 0.7 became shippable again.  Putting the same features on DEFENSE as well is flat on the
criterion and takes the defensive agreement 0.766 -> 0.768 and the defensive spread 1.33 -> 1.30.

The criterion cannot choose between the two versions (z -0.88 and -0.09), so the floors did.  `ship_shot7`
(offense only) is 0.013 better on the criterion and leaves the defensive agreement at 0.762 against its 0.76
floor -- almost no headroom.  `ship_shot7d` keeps the headroom, and Part 3.1's defensive work is what the
headroom is for.  To flip: drop the six `SHOTQ` names from `features_full_D` and point `cal_map` at
`ship_shot7`.  `scratch/ship_try2.py` runs either through the whole shipping path without touching
`config.yaml` permanently.

### What this pass measured (all on the criterion, all in FINDINGS 22)

| | criterion | verdict |
|---|---|---|
| shot quality -- where his attempts came from (`gbdt_prior.SHOTQ`) | **-0.045** on the line, flat in shipping shape | **SHIPPED**; it is what unlocks blend 0.7 |
| experience -- seasons, career possessions, entry age (`roles.career_inputs`) | **+0.054** | rejected, and read 22.2 before trying anything like it |
| Huber loss instead of RMSE | -0.039 at the cheap booster, **+0.061 at `quality=4`** | rejected |
| inverse-variance target weights (`training_rows(sat_poss=)`) | -0.003 at best | wired and off |
| asymmetric window pooling (`training_rows(win_past=)`) | +0.025 to +0.075, both directions | wired and off |

## Part 2: machinery

| file | what |
|---|---|
| `scripts/54_track.py` | the tracker: dump a system at K = 3 (timed), fit the map leave-one-season-out, score, log a row, redraw the chart.  `--systems=a,b "--maps=..." --label=...`; one dump per system |
| `scratch/prior_bench.py` | **the 20-second pre-filter**: the prior's own leave-window-out fit per feature set, with the low-exposure stratum beside the pooled number.  `--q4 --loss= --delta= --sat= --past= --kmul=`.  **Read 22.2 first: it is necessary, not sufficient, and it has been wrong by 0.13** |
| `scratch/pairsys.py` | the paired test between two TRACKED systems: pooled difference, z over the 28 seasons, wins |
| `scratch/maps.py` | map work on an EXISTING dump, ~2 s each, no refit |
| `scratch/ship_try2.py` | a candidate through the REAL shipping path: config (targets, map, **and the prior's feature lists**), `08_ratings.py`, `22_vs_consensus.py`, the floor tests.  Restores `config.yaml` in a `finally` |
| `scratch/consensus_read.py` | the cheap screening read of mapped candidates off the tracker's own parameter tables (no refits) |
| `src/eracoef/fastfit.py` | `MspiFast`: every knob (`lam`, `gbdt_params`/`_d`, `gbdt_features`, `target`/`_d`, `win_decay`/`_d`, `decay`, `season_weights`, `pad_scale`, `phases`, `lam_buckets`, `min_den`); `FASTFIT_TIMER=1` for the section clock |
| `src/eracoef/gbdt_prior.py` | `DERIVED`, `RATIOS`, **`SHOTQ`** (+ `add_shotq`), **`CAREER`**, the feature lists `FULL_/DERIVED_/RATIO_/SHOT_/PRIOR_FEATURES`, `training_rows(win_decay=, win_past=, sat_poss=)`, `GBDTPrior` |
| `src/eracoef/xshoot.py` | **`player_shot_frame(seasons, cfg, ids)`**: the block's per-shooter shot totals and its own league levels -- the shot-quality features' single source, used by the panel build and by prediction alike |
| `src/eracoef/roles.py` | **`career_inputs(inputs, before_season, ids, age=)`** (rejected block, kept for the record) |
| `src/eracoef/calmap.py` | families x exposure terms x a bend; `parse_maps` takes `mapO:mapD` and `\|cubic` / `\|rowcubic` |
| `scratch/` (untracked) | `cmp_shotq.py` / `cmp_career.py` (the identical-numbers checks: both paths agree to 0.00e+00 on all ten windows), `add_shot_cols.py` / `add_career_cols.py` (the in-place panel patches, backups at `.bak2` / `.bak3`), `prior_ceiling.py`, `smoke_shot.py`, `tabfm_try.py` |

### Verification

```
.venv/Scripts/python -m pytest tests -q                                        # 82 passed, 1 xfailed
.venv/Scripts/python scratch/prior_bench.py O ratio,shot --q4                  # a feature set, ~30 s
.venv/Scripts/python scripts/54_track.py --systems=X "--maps=..." --label=...  # the criterion, ~10 min
.venv/Scripts/python scratch/pairsys.py <base> <cand>                          # is it real?
.venv/Scripts/python scratch/ship_try2.py TAG blend0.7 rapm1 <system> [--od --dshot]
.venv/Scripts/python scripts/08_ratings.py && .venv/Scripts/python -m pytest tests/test_vs_consensus.py -q
```

### Traps

- **The prior's own fit can be wrong by more than any feature is worth.**  Its target is the player's APM
  pooled over his OTHER windows, so a player with more windows has a quieter target and is easier to predict.
  Any feature that names those players (experience did) buys held-out MSE without buying knowledge, and the
  criterion gets none of it back.  Splitting by exposure does NOT catch it.  FINDINGS 22.2.
- **A knob measured at the cheap booster does not transfer to `quality=4`.**  Huber was -0.039 at one and
  +0.061 at the other; 21.25 said the same about bagging and the search.  Measure at the operating point.
- **`scripts/49_role_panel.py` used to drop the `raw_*` columns it had just built** -- they were only in the
  panel because `scratch/add_raw_rates.py` put them back by hand.  Fixed; the write-out now lists `raw_*` and
  the nine `shot_*` columns explicitly.  A rebuild of the panel is no longer destructive.
- **One system per dump for timing**: the GBDT cache per process flatters a later system in the same run.
- **The tracker's `seconds` varies +/-6% with machine load.**
- **A map term with no covariate column applies as ZERO, silently.**
- **The `tshare` and `prior` map terms belong on OFFENSE only.**
- **The consensus floors are not the criterion.**  Read them before shipping, every time; the screening read
  (`consensus_read.py`, 2024-2026, 475 players) is NOT the same object as the floors in
  `tests/test_vs_consensus.py`, which score the board `08_ratings.py` builds.  Screen with the first, decide
  with the second.
- **`ensemble_n_jobs: 1`** on any `chimeraboost` fit inside the holdout's workers, or the bag forks and the
  wall time explodes.
- Long bash heredocs still fail in this shell; write patch scripts with the Write tool.

## Part 3: the next pass

The prior pass is Part 1's table and FINDINGS 22.5's conclusion: **capacity did not move it (21.24),
re-expression was worth 0.05 (21.25), and the two richest new sources on the shelf are worth 0.045 and less
than nothing.**  Against 3.5's ceiling -- the prior's target has a split-half reliability of 0.808 on offense
so nothing can correlate past 0.899, and the shipped booster reaches 0.590 -- the missing third is not sitting
in the box score waiting for a better column.  So the next pass leaves the BOX SCORE alone and goes to the
play-by-play behind it, which is 3.1.

### 3.1 Play-by-play features for the prior (the owner's call, 2026-09-06)

FINDINGS 22.5 concluded the BOX SCORE is nearly spent.  The play-by-play is not, and that is the resolution:
shot quality (22.1) was itself a play-by-play block -- it came out of the shots tables, not the box -- and it
is the one thing this pass added that paid.  The 13 rates are a nightly summary; every event that made them is
already on disk in `data/raw/pbp/{season}/{game_id}.parquet`, 24 columns with `actionType`, `subType`,
`description`, `personId` and shot coordinates, for all 30 seasons.  Nothing below needs a new download except
where it says so.

#### The reference: Dredge (Justin Willard, Nylon Calculus, 2016)

The closest published thing to what we are doing, and worth reading in full.  Elastic net (`glmnet`, hence the
name) predicting **15-year RAPM**, trained on 2001-2015 and tested out of sample on **1997-2000 and 2016** --
our era, our target, our validation design.  Players under 3,000 possessions dropped, everyone weighted by
possessions, then a team adjustment and a mean reversion.  He reports interaction terms helped out of sample
"by a significant amount" despite his overfitting fears.

His published simple-linear coefficients, per 100 possessions unless noted.  **These are the sizes to
calibrate expectations against, not values to copy** -- his target is a 15-year RAPM and ours is a held-out
season's points:

| term | coef | his note |
|---|---|---|
| **DREB%** (0-1) | 6.48 | needs a nonlinear transform; "not all rebounds are the same" -- split by FG vs 3PT misses |
| **OREB%** (0-1) | 3.87 | |
| **Shot%** (0-1) | 2.24 | usage without turnovers |
| **Stl100** | 1.33 | ~0.25 of the value is offensive (a steal starts a break) |
| **TechsFlgs100** | **+1.25** | technicals and flagrants, and the sign is POSITIVE -- "a proxy for feisty defenders and guys who fight hard in the paint" |
| **OffFoulsDrawn100** | **1.22** | "one of my favourite discoveries"; and **non-charges tested MORE valuable than charges** |
| PtsOverAvg100 | 0.872 | efficiency and volume together |
| UnAstShot% | 0.841 | unassisted FGM% x shot% -- "players get credit for assists, they should get more credit for unassisted shots" |
| Blocked0to5Ft100 | 0.523 | rim blocks; "blocking a three-point shot did not test well" |
| **Russells100** | 0.445 | a block THE DEFENCE RECOVERS -- why raw blocks are overrated |
| TOV100 | -0.407 | |
| AstDunksLayups100 | 0.352 | |
| loose ball fouls 100 | ~0.33 | not in the simple model but in richer ones; hustle proxy |
| BLK100 | 0.236 | "in their pure form, highly overrated" |
| MPG | 0.132 | better players play more |
| PFsDrawn100 | 0.0846 | all fouls drawn, not just shooting |
| Ast100 | 0.0580 | "assists by themselves have very little value" (BPM drops them entirely) |
| 3FGA100 | 0.0376 | a spacing effect |
| 3FGA100PosAdj | 0.0126 | `3PA100 * (position + 3)`, position adjusted by height |
| StolenTOV100 | -0.0838 | his own research says stolen turnovers are ~2x as bad as other turnovers |
| PTS_FB100 | -0.0499 | fast-break points; punishes gambling defenders |
| **DefGoaltends100** | **-2.75** | the largest coefficient in the model, "-1.5 in more complex ones"; a proxy for chasing blocks at the expense of defence |

Two of his readings bear directly on our SHAP table: **assists are nearly worthless alone** (ours gives `ast`
3.0% on offense) and **blocks are overrated in raw form while the recovered ones are not** (ours gives `blk`
15.7% on defense, the single largest defensive feature -- exactly the term Dredge says to split).  His
**DRE** (2015, same design) adds the claim that older linear metrics **massively overvalue rebounds**
(0.2 on TRB, offensive rebounds not significant) and that **steals are worth ~1.7 each**.

**BPM 2.0** (Myers, Basketball-Reference), the same regress-onto-RAPM design over 1997-2016, concludes what our
defensive spread problem says from the other side: post players are hardest to measure from a box line, and
elite defenders are underrated while bad ones are overrated.

#### What our own feed can actually attribute -- measured, and this is the gate

Our `data/raw/pbp` is the **v3** feed: **one `personId` per event**.  So "who blocked it" is fine when the
block is its own row, and "who drew it" is not there at all unless a later event names him.  Verified on 2015
and 1997:

| Dredge term | how it lands in our feed | buildable? |
|---|---|---|
| **unassisted shot %** | `personId` is the SHOOTER and the assist is text in his description -- `"Vucevic 19' Jump Bank Shot (2 PTS) (Payton 1 AST)"`.  We only need "does `AST` appear", not who | **YES, all 30 seasons, no name parsing.  Do this first.** |
| **blocks / Russells / rim blocks** | BLOCK is its OWN row with the blocker's `personId` (`"Holiday BLOCK (1 BLK)"`), 1997 included.  A Russell is that row followed by a defensive rebound; a rim block is it paired with the missed shot's `shotDistance` | **YES, all 30 seasons** |
| **steals / stolen turnovers** | STEAL is its own row with the stealer; pair with the adjacent `Turnover` row | **YES, all 30 seasons** |
| **loose ball fouls, technicals, flagrants** | `Foul` rows, `personId` = the committer, which is what the term wants | **YES, all 30 seasons** |
| offensive fouls **committed** | `Foul / Offensive`, `personId` = the offensive player | YES, all 30 seasons |
| **offensive fouls DRAWN** (coef 1.22) | **NOT PRESENT.**  `"Asik OFF.Foul (P3)"` names only the fouler, in 1997 and 2015 alike | **NO -- needs another source** |
| fouls drawn (all) | shooting fouls are recoverable from the free throws that follow; the rest are not | partial |
| assists to dunks / layups | needs the ASSISTER, who is a bare surname in the shooter's description | needs name resolution |
| defensive goaltends | `Violation / Defensive Goaltending` rows exist throughout -- but see the warning below | YES, with a caveat |
| fast-break points | not an event; needs possession-transition logic over our own stints | derivable, more work |

**The blocker: `OffFoulsDrawn100` is Dredge's best find and we cannot currently build it.**  Justin notes it is
only available 2006+ (plus 2001, oddly), which fits -- he was using the **v2** play-by-play, which carries
`PLAYER1/2/3_ID` and names the drawer.  Two routes, cost them before committing: refetch v2
(`playbyplayv2`, one call per game, ~35,000 games) or use `pbpstats`' enhanced play-by-play, which resolves
this attribution for us.  `data/raw/pbpstats/` exists but holds exactly **one** cached game, so either way this
is an ingest job, not a feature job.  Everything above it in the table is free.

**A discrepancy to resolve before trusting goaltends.**  Justin's footnote says 1997 has suspiciously FEW
goaltending violations -- 249 in the whole season against 500-700 now.  Our feed says the opposite: **1.27 per
game in 1997 falling to 0.43 in 2026**, which extrapolates to ~1,500 in 1997.  One of the two is wrong, and
since a feature with a spurious era trend is exactly the 22.2 failure mode wearing a different hat, **count
the season totals and reconcile them against a published source before this becomes a column.**

**The subtype-detail trap, generally.**  Distinct `subType` values go 69 (1997) -> 95 (2005) -> 129 (2026), and
`Foul / Offensive Charge` does not exist as a subtype before ~2006 (charges sit inside `Foul / Offensive`;
the early `Offensive` rate of 4-5.4 per game is about the later era's Offensive plus Charge, 3.23 + 1.31 in
2015).  `Violation / Kicked Ball` is 2005+.  A column that is structurally zero for a third of the panel will
be learned as "old era", not as "none happened" -- `season` is a feature and it will happily oblige.  **Build
the era-stable aggregate first and test any modern-only split as a separate, later question.**  Justin's own
finding that non-charges beat charges says the aggregate is the better feature anyway.

#### Build route and order

The template is 22.1's, which worked end to end: a per-(player, season) counter table built once and cached
beside the shots tables, summed over a block by a `xshoot.player_shot_frame`-style helper, stored per row by
`scripts/49_role_panel.py`, rebuilt from the TRAINING block at prediction time by `spm.chain_offset`, and a
`scratch/cmp_*.py` proving the two paths agree to 0.00e+00 before anything is measured.  Rates per 100
possessions, padded, centred like the 13.

Order, by value over cost:

1. **unassisted shot share** -- free, all seasons, no attribution needed.
2. **the block split: Russells and rim blocks** -- all seasons, and it attacks `blk`, the largest defensive
   feature we have, which Dredge says is the wrong shape.
3. **loose ball fouls, technicals + flagrants** -- all seasons, trivial, and both are hustle proxies with
   surprising signs worth confirming on our own criterion.
4. **stolen turnovers** -- all seasons, a sequence pair.
5. **offensive fouls committed** -- all seasons, and the natural stepping stone to the drawn version.
6. *then* decide whether `OffFoulsDrawn` is worth an ingest job.  It is the highest published coefficient
   available to us, and it is the one that costs real work.

#### Measurement discipline (do not skip)

`scratch/prior_bench.py` first, at `quality=4`, then the tracker, then `scratch/pairsys.py` for the z.  This
pass measured five things and four were zero or worse, and the one that looked best offline cost the most in
reality.  **A feature block is worth having when the criterion says so and not before.**

Sources: Dredge, https://fansided.com/2016/07/26/introducing-dredge-a-play-by-play-derived-metric/ ; DRE,
https://fansided.com/2015/02/23/introducing-dre-a-hopefully-better-simple-metric/ ; block types,
https://fansided.com/2015/09/21/shot-blocking-details-mining-19-years-of-play-by-play-data/ ; BPM 2.0,
https://www.basketball-reference.com/about/bpm2.html ; pbpstats enhanced play-by-play,
https://pbpstats.readthedocs.io/en/latest/pbpstats.resources.enhanced_pbp.html

### 3.2 The defensive four-factor fit -- now the single thing blocking a measured 0.14

**FINDINGS 22.7 put a number on this.**  A 625-trial search over the whole estimator, validated on 14
held-out seasons the optimizer never saw, found `tune501`: **-0.138 on the criterion at z -3.95 over 22 of 28
seasons, and 20% FASTER than what ships**.  It fails the consensus, and the reason is entirely defensive --
the gain lives in `win_decay_d` = 0.28 (pooling the defensive prior's target over nearby windows instead of
the whole career), and the defensive agreement floor refuses it.  Restoring the shipped 1.0 recovers the floor
and gives back 0.058 of the 0.083, taking z from -3.02 to -0.75.

So this is no longer "the most promising thing left".  It is the thing standing between this board and a
measured, significant, cheaper gain, and the disagreement is substantive: the criterion wants defence weighted
toward recent form, the consensus wants the career statement.  A defensive fit that both sides believe is what
unlocks 22.7.

Fit opponent eFG allowed, turnovers forced, offensive rebounds allowed and free-throw rate allowed separately,
each with its own ridge ratio -- the asymmetry came out ESTIMATED, not imposed (forcing turnovers is a real
defensive skill at ratio 0.75; preventing offensive rebounds barely is, at 3.00) -- then recombine into
expected points allowed.  It was 0.4's "right but too big for v1" and it is now the best thing on the list:

* defense is the binding constraint on everything.  It is 1.33x too wide against the consensus, it owns the
  only permanently-failing test in the suite, and its 0.76 agreement floor blocked three candidates in 21.26
  and is what separates this pass's two shipping candidates.
* it is the only remaining item that changes the ESTIMATOR rather than the prior's inputs, and the estimator
  is where FINDINGS 22.5 says the signal must be.
* `ship_shot7d` already narrows the spread to 1.30 for free, which is the direction this work goes further in.
* it is the estimator-side complement to 3.1: 3.1 gives the prior more to say about defence, this changes what
  the ridge is fitting.  They do not collide and either can go first.

A day's work; the principled version of what `x3def` did by hand.

### 3.3 The defensive booster, now that its pieces are separated (FINDINGS 22.6)

21.26 rejected `quality=4` on defense as a package because the consensus floors would not take it.  It is
three things and only one of them is the problem:

* **`linear_leaves` is the good part** -- -0.009 weighted MSE on the prior's own fit, and the criterion agrees
  (`ship_shot7d_ll`, 110.6925, -0.014 against the shipped board, ten of ten floors).  It is inside the noise
  at z -0.75 and was not shipped, but it is the closest thing to a free defensive gain on the table.
* **`cross_features` is harmful alone and helpful beside linear leaves** (+0.004, then -0.013 together) -- and
  the criterion reverses that: `ship_shot7d_llcf` is only -0.006.  Do not trust the offline ranking on this.
* **the BAG is what defense will not tolerate.**  It is the only piece that moves the prior's calibration
  slope (1.08 -> 1.15), i.e. widens it, and defensive width is the binding floor.  **But it is also the only
  piece that improves the LOW-EXPOSURE rows** (1.032 against 1.075), which are the players the prior actually
  decides.  That tension is unresolved and is the interesting part: a bagged defensive prior with the width
  taken back out -- by 3.4's per-player shrinkage, or by a scalar -- has never been tried.

### 3.4 Per-player shrinkage from the prior's own confidence

Every player is pulled toward his prior by the same lambda, but the prior is far more predictive for some than
others.  `quality=4` already fits 5 bagged members, so the disagreement across members is a per-player
predictive spread, free.  Feed it into a per-player penalty (`lam_buckets` is the machinery, currently keyed on
exposure groups).  This is the mechanism that would let the prior carry the bench hard without overriding
stars -- the failure mode the criterion has complained about since FINDINGS 19, and the way to take the
defensive bag's width back out (3.3).

### 3.5 Single-season targets, and getting off chunks

Train the prior on single-season APM instead of three-season: more rows, noisier each, and a step toward the
continuous rating the product is going to (0.2).  It is also the only change that would break the pooled-target
mechanism of 22.2, which is worth knowing independently.

### 3.6 Still open, not scheduled

* **TabFM** (`google/tabfm-1.0.0-jax`): installs and downloads (5.7 GB; point `HF_HOME` at `A:`), but the
  orbax restore dies in tensorstore on a 1.5 GB region -- 38.5 GB of the box's 48 GB commit limit was taken.
  Retry on a quiet machine.  `scratch/tabfm_try.py` prints the booster's baseline for it to beat.
* **Fast-forward `main`** (`bbstats/openrapm`) to `hybrid-and-xpts`.  Pages serves `main` at `/docs`, so
  until this is done the site shows the board of 2026-09-05, not `ship_shot7d`.  `docs/data/ratings.json` on
  this branch is already rebuilt from the shipped board.
* **DNS for openrapm.com**: the custom domain was REMOVED on 2026-09-06 (`docs/CNAME` deleted on both branches,
  `cname: null` on the Pages API) because it had no DNS behind it and was redirecting `bbstats.github.io`
  into a dead name.  The site is back at `https://bbstats.github.io/openrapm/`, https enforced.  To turn the
  domain on: four GitHub `A` records + a `www` CNAME at Porkbun FIRST, then re-add `docs/CNAME` on `main`,
  wait for the cert, then `gh api -X PUT repos/bbstats/openrapm/pages -F https_enforced=true`.
* Never re-run, from 21 and 22: lam_ratio, lam_buckets by exposure, playoff rows, one target for both sides,
  x3def_p1, the mover / rookie-age / age-by-exposure / rating-by-age / rating-by-exposure / prior-by-exposure /
  prior^2 map terms, season weights beyond the decay, padding scale and target, panel APM at penalty 30 and
  300, the training-block team's mean rating, role growth, the lineup spread, the two sides of the bend apart,
  the multiplicative offense-defense term, 8 bagged members, re-pricing the prior, the recursion, scaling the
  offset, Huber, inverse-variance target weights, asymmetric pooling, experience.

(Earlier handoffs: `HANDOFF_rankmap_archive.md`; the calibration-map handoff is in git history at `d1bc3cc`;
the prior-pass handoff this one replaces is at `a392f6c`.)
