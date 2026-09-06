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

1. **Accuracy beats the clock.**  Fit every row, score every game.  The fit time is a number the chart
   carries, not a term to optimise.  Do not trade accuracy for time again.
2. **The 3-season window is on the way out.**  *"Nobody really looks at chunks, we will ultimately move away
   from this mode."*  Do not tune K or the window length; prefer work that survives the move to a continuous
   or per-season rating.
3. **Schedule context is not worth it** (*"evens out really well"*) -- rest days and back-to-backs, declined.
4. **The four-factor defensive fit is right but was too big for v1.**  It is now the most promising thing left
   (Part 3).
5. **The prior was the work, and the prior is now mostly spent.**  Part 3 says what replaced it.

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
in the box score waiting for a better column.  So the next pass should leave the panel alone.

### 3.1 The defensive four-factor fit (do this first)

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

A day's work; the principled version of what `x3def` did by hand.

### 3.2 Per-player shrinkage from the prior's own confidence

Every player is pulled toward his prior by the same lambda, but the prior is far more predictive for some than
others.  `quality=4` already fits 5 bagged members, so the disagreement across members is a per-player
predictive spread, free.  Feed it into a per-player penalty (`lam_buckets` is the machinery, currently keyed on
exposure groups).  This is the mechanism that would let the prior carry the bench hard without overriding
stars -- the failure mode the criterion has complained about since FINDINGS 19.

### 3.3 Single-season targets, and getting off chunks

Train the prior on single-season APM instead of three-season: more rows, noisier each, and a step toward the
continuous rating the product is going to (0.2).  It is also the only change that would break the pooled-target
mechanism of 22.2, which is worth knowing independently.

### 3.4 Still open, not scheduled

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
