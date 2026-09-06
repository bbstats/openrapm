# Handoff: the prior is the work

Written 2026-09-06 (end of the second day of iterate-and-improve mode).  `FINDINGS.md` section 21 (items 1-26)
is the record; `docs/progress.png` / `docs/progress.csv` the chart and its log; `PIPELINE.md` draws how the
shipped model works, stage by stage.

**Tree state: clean, committed and pushed on `hybrid-and-xpts` at `d9d99b7`.**  Nothing uncommitted but
`outputs/*.parquet` scratch dumps.  Tests: 82 passed, 1 xfailed.

---

## Part 0: the owner's standing rulings (read before choosing anything)

1. **Accuracy beats the clock.**  True loss = (error / 111.2955) x (28-fit seconds / 134) has a hole in it:
   dropping short stints improves it monotonically down to a board worse than the one previously shipped
   (21.22).  The owner: *"not an apples to apples comparison anyway... a universal solution is preferred which
   requires testing on every game that we can"*, and *"best one looks fine speed wise, we don't need to be too
   brittle"*.  **Choose the line on the criterion alone.  Fit every row.  Score every game.**  The fit time is
   a number the chart still carries, not a term to optimise.  Do not trade accuracy for time again.
2. **The 3-season window is on the way out.**  *"Nobody really looks at chunks, we will ultimately move away
   from this mode."*  So: do not spend effort tuning K or the window length (a K sweep was proposed and
   declined for this reason), and prefer work that survives the move to a continuous or per-season rating.
3. **Schedule context is not worth it** (*"evens out really well"*) -- rest days and back-to-backs, declined.
4. **The four-factor defensive fit is right but too big for v1** -- keep it on the shelf, not in the next pass.
5. **The prior is the work.**  That is the owner's call for what comes next, and Part 3 is the list.

## Part 1: where it stands

| | criterion (K = 3, mapped) | 28 fits |
|---|---|---|
| the board this phase started from (`mspi_linear+sat`) | 111.30 | 134 s |
| **the criterion's line** (`best_ratio_full`, map `linear+log2&age2&xlog&prior&tshare\|rowcubic`) | **109.85** | 59 s |
| **what ships** (`ship_side6`, no held-out season, ten of ten consensus floors) | **110.71** | 43 s |
| no ratings at all | 125.6 | |

The line carries three things that **cannot ship**: the player's age at H (a window has no single "age at H"),
and the two bends (a rating carries no team).  The shipped board is the same prior in shipping shape.

### What ships, exactly (`config.yaml`)

| | offense | defense |
|---|---|---|
| GBDT features | the 37-name list (13 rates + season + role + 10 linear aggregations + 10 efficiency ratios) | the 15-name list |
| booster | `quality: 4` (5 bagged members, audition fits, cross features) | `params_def`: no auditions |
| target | 0.6 APM + 0.4 RAPM_1 over his OTHER windows, 0.3 discount per window of distance | RAPM_1, every window alike |
| map | `linear+log2&xlog&prior&tshare` | `linear+log2&xlog` |

Consensus 0.792 / 0.792 / 0.766 (floors 0.75 / 0.75 / 0.76), defensive spread 1.33 (floor 1.4), offensive
bigness gap -0.283 (floor 0.30).

### What moved the score this phase

1. The age term in the map (age at H, quadratic): -0.15.  Prediction-time only.
2. H-2 at half weight in the ridge rows and behind the padded rates: -0.10 and -0.04.
3. The map's terms (log-exposure level, rating-by-log-exposure slope, the prior part as its own column): -0.22.
4. The GBDT prior trained on unshrunk APM instead of RAPM_1: **-0.29, z -3.1**, the largest single gain.
5. A cubic in the team-game total after the map: -0.07 to -0.08 on every board tried (21.20).
6. The same cubic at STINT level beside it: **-0.205, z -4.8** (21.22).  The pair separates "this team-game's
   total is extreme" from "the lineups inside it were extreme"; the coefficients have opposite signs.
7. The training block's role in the map (`tshare`): -0.058.  **Offense only** -- on defense it takes the
   consensus agreement 0.766 -> 0.754.
8. The prior's features and the booster's budget TOGETHER: -0.135 at z -3.3 (21.25).  Each part alone is under
   0.07 and none is significant; they interact.

### The leak control -- reuse this pattern

Two candidates measured big and died on it.  A map term in the player's minutes AT H is worth -0.19 (z -3.7);
the same share measured from HALF of H's games is worth -0.006, so the whole of it was within-season feedback.
`u x the game's mean stint length` is worth -0.15 and dies the same way against the team's other games
(substitutions follow the score).  **Any covariate measured on the held-out season gets a control that measures
the same thing from data the outcome could not have touched** -- the other half of the season, the team's other
games, the training block.  21.21 and 21.23.

### Flat or negative -- do not re-run

lam_ratio, lam_buckets by exposure, GBDT shape at the plain feature set, playoff rows, one target for both
sides, x3def_p1, mover / rookie-age / age-by-exposure / rating-by-age / rating-by-exposure / prior-by-exposure /
prior^2 map terms, season weights beyond the decay, padding scale and target, panel APM at penalty 30 and 300,
the training-block team's mean rating, role growth, the lineup spread and the best/worst man on the floor, the
two sides of the bend apart, the multiplicative offense-defense term, 8 bagged members instead of 5, re-pricing
the prior, the recursion, scaling the offset.

## Part 2: machinery

| file | what |
|---|---|
| `scripts/54_track.py` | the tracker: dump a system at K = 3 (timed), fit the map leave-one-season-out, score, log a row, redraw the chart.  `--systems=a,b "--maps=..." --label=...`; one dump per system |
| `scratch/maps.py` | **the loop for map work**: score maps on an EXISTING dump, ~2 s each, no refit.  Use this before ever re-running the tracker |
| `src/eracoef/fastfit.py` | `MspiFast`: every knob (`lam`, `lam_ratio`, `gbdt_params`, `gbdt_params_d`, `gbdt_features`, `target`, `target_d`, `win_decay`, `win_decay_d`, `panel`, `decay`, `decay_exposure`, `season_weights`, `pad_scale`, `pad_target`, `phases`, `lam_buckets`, `counter_columns`, `min_den`); `FASTFIT_TIMER=1` for the section clock |
| `src/eracoef/gbdt_prior.py` | the prior: `DERIVED` (linear aggregations), `RATIOS` (efficiency, from the panel's `raw_*` columns), `add_derived`, `training_rows(win_decay=)`, `GBDTPrior` |
| `src/eracoef/calmap.py` | families x exposure terms (`&`-combinable: `sat`, `log2`, `age2`, `xlog`, `prior`, `tshare`, `hshare`, ...) x a bend after the map (`\|cubic`, `\|rowcubic`), `parse_maps`, `row_columns`, `bent_prediction` |
| `src/eracoef/designcache.py` | per-season pieces (disk + LRU), `build_window_cached(counter_cols=, min_den=)` |
| `scratch/` (untracked) | `prior_ceiling.py`, `timing_run.py`, `loss_vs_true.py`, the `cmp_*.py` identical-numbers checks, `teamnl2.py` / `rowbend2.py` / `spread.py` second-stage prototypes, `add_raw_rates.py`, `tabfm_try.py` |

### Verification

```
.venv/Scripts/python -m pytest tests -q                                        # 82 passed, 1 xfailed
.venv/Scripts/python scratch/maps.py best_ratio_full "<map>" "<map>"           # map work, ~2 s each
.venv/Scripts/python scripts/54_track.py --systems=X "--maps=linear+log2&age2&xlog&prior&tshare|rowcubic" --label=...
.venv/Scripts/python scripts/08_ratings.py && .venv/Scripts/python -m pytest tests/test_vs_consensus.py -q
.venv/Scripts/python scripts/52_site.py && .venv/Scripts/python scripts/22_vs_consensus.py
```

### Traps

- **One system per dump for timing**: the GBDT cache per process flatters a later system in the same run.
- **The tracker's `seconds` varies +/-6% with machine load.**  Differences smaller than that are not real.
- **A map term with no covariate column applies as ZERO, silently.**  `apply_params` needs `prior_o=`/`prior_d=`
  for a `prior` term and `extra=` for `tshare`; `08_ratings.py` builds the `tshare` column per window.
- **The `tshare` and `prior` map terms belong on OFFENSE only** -- either one on defense breaks the 0.76
  consensus floor.
- **The consensus floors are not the criterion.**  Anything that widens or sharpens the DEFENSIVE prior trips
  them; that is why the two sides now take separate priors.  Read the consensus before shipping, every time.
- **Adding a `chimeraboost` fit inside the holdout's workers**: pass `ensemble_n_jobs: 1` or the bag forks
  inside each worker and the wall time explodes (323 s once).
- **The first chimeraboost fit in a process pays a JIT cost** (0.49 s against 0.14 s warm).  Warm it before
  timing anything.
- Long bash heredocs still fail in this shell; write patch scripts with the Write tool.

## Part 3: the next pass -- the prior

The owner's call: *"improving priors is the real most important key."*  The through-line from 21.24/21.25 is
that **capacity is not the constraint, information is** -- bagging and the model-selection search moved nothing
at the plain feature set and only started paying once the features got richer.  So lead with information.

### 3.1 The prior has never seen shot quality (do this first)

The 13 rates know how many shots a player made and nothing about where from.
`data/stints/{season}_RS_shots.parquet` already carries, **per player per game**:

```
fg2a, fg2m, xl2, fg3a, fg3m, xl3        # xl2 / xl3 = the league's expected makes from HIS locations
```

It is built, cached, and already read on every fit (`xshoot.season_totals`).  Two features fall straight out:

* **shot difficulty**: `xl2/fg2a`, `xl3/fg3a` -- the league make probability of his average attempt.  Separates
  a rim-runner from a mid-range shooter at identical FG%.
* **shot-making over location**: `(fg2m - xl2)/fg2a` -- how much he beats a league shooter from his own spots.
  The padded version already exists as `ShooterRates.ratio2` / `ratio3`.

Route: add the per-player season totals to the panel the way `scratch/add_raw_rates.py` added the uncentred
rates (it patches `outputs/role_panel.parquet` in place and checks every existing column identical, backup at
`.parquet.bak`), then fold the same into `scripts/49_role_panel.py`, then extend `gbdt_prior.RATIOS`.  At
prediction time the numbers must come from the same place -- see the `raw=` argument `chain_offset` already
threads through to `gbdt_offset`.

**This is the only item on the list that adds information rather than re-expressing what is there**, which is
why it leads.  Everything measured on 2026-09-06 was re-expression, and re-expression was worth 0.05.

### 3.2 Two nearly-free corrections

* **Out-of-fold affine recalibration of the prior.**  Measured: its slope against held-out truth is **1.10**
  (`scratch/tabfm_try.py` prints it), so it is ~10% over-dispersed, and the ridge treats the offset as truth.
* **Huber loss instead of RMSE.**  The target is APM; the tails are heavy.

### 3.3 Then, in rough order

* Era-standardise the rates (z within season).  They are centred on the league mean but not scaled, and the sd
  of three-point volume across 28 seasons is enormous.  The tree gets `season`, but not the scale.
* Experience: career possessions to date, seasons in the league.  Distinct from age -- a 25-year-old rookie and
  a 25-year-old in year seven are different players.  `role_inputs` has what is needed.
* Trajectory: the previous window's rates and the delta (+0.008 for the *linear* prior years ago; never tried
  on the tree, where an interaction with age is available).
* Minutes per game and games played (durability); the playoff box line as its own feature block.
* Inverse-variance weights on the target rows instead of possessions.
* Asymmetric pooling: past windows weighted differently from future ones.  Aging is directional and the 0.3
  distance discount is symmetric.
* **Single-season APM targets instead of three-season** -- more rows, noisier each, and it is a step toward
  getting off chunks, which is where the product is going (Part 0.2).
* One model predicting both sides and sharing structure, instead of two.
* Monotone constraints on the obvious features; a role interaction or per-exposure-band models.
* Blend the tree with the old linear prior (FINDINGS 19's open item: "the GBDT's middle with the linear top").
* Free-throw trips and and-1s per player -- needs the slot counters summed in the panel build, a step more work
  than the rest.

### 3.4 The structural item, for after v1

**Per-player shrinkage from the prior's own confidence.**  Every player is pulled toward his prior by the same
lambda, but the prior is far more predictive for some than others.  `quality=4` already fits 5 bagged members,
so the disagreement across members is a per-player predictive spread, free.  Feed it into a per-player penalty
(`lam_buckets` is the existing machinery, currently keyed on exposure groups).  This is the mechanism that
would let the prior carry the bench hard without overriding stars -- the failure mode the criterion has
complained about since FINDINGS 19 ("too timid for starters, half-spread for deep bench").

### 3.5 The ceiling, so nobody chases the last third

`scratch/prior_ceiling.py`: the prior's target (a player's APM pooled over his other windows) has a split-half
reliability of 0.808 on offense and 0.889 on defense, so **no predictor can correlate with it beyond 0.899 /
0.943**.  The shipped booster reaches 0.590 / 0.629 -- two thirds of the ceiling.  The missing third is not
all reachable from a box line; some of it is only in the play-by-play, which is what the ridge is for.

### 3.6 Still open, not scheduled

* **The defensive four-factor fit** (Part 0.4).  Fit opponent eFG allowed, turnovers forced, offensive rebounds
  allowed and free-throw rate allowed separately, each with its own ridge ratio -- the asymmetry came out
  estimated, not imposed (forcing turnovers is a real defensive skill at ratio 0.75; preventing offensive
  rebounds barely is at 3.00) -- then recombine into expected points allowed.  Defence is 1.33x too wide
  against the consensus, owns the only permanently-failing test in the suite, and is the floor that blocked
  three shipping candidates today.  A day's work; the principled version of what `x3def` did by hand.
* **TabFM** (`google/tabfm-1.0.0-jax`): installs and downloads (5.7 GB; point `HF_HOME` at `A:`), but the orbax
  restore dies in tensorstore on a 1.5 GB region -- 38.5 GB of the box's 48 GB commit limit was already taken.
  Retry on a quiet machine.  `scratch/tabfm_try.py` prints the booster's baseline on the same task for it to
  beat (weighted MSE 2.69 against 5.88 for the training mean).  It has to beat 0.14 s per leave-window-out pair.
* **Fast-forward `main`** (`bbstats/openrapm`) to `hybrid-and-xpts` so the site picks up the new board.
* **DNS for openrapm.com**: four GitHub `A` records + `www` CNAME at Porkbun, wait for the cert, then
  `gh api -X PUT repos/bbstats/openrapm/pages -F https_enforced=true`.

(Earlier handoffs: `HANDOFF_rankmap_archive.md`; the calibration-map handoff is in git history at `d1bc3cc`.)
