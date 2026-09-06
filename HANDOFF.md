# Handoff: iterate-and-improve mode

Written 2026-09-06. The owner's instruction for this phase: keep improving the K = 3 out-of-season team-game
error, charge every fit for its time, post the chart, write nothing but FINDINGS.  `FINDINGS.md` section 21
(items 1-22) is the record; `docs/progress.png` / `docs/progress.csv` the chart and its log.

**Tree state: committed and pushed on `hybrid-and-xpts`** except the rebuilt board artefacts of the last step
if the final chain had not finished (`git status`).  Tests: 82 passed, 1 xfailed on the last full run.

---

## Part 0: do first

1. **Fast-forward `main`** (`bbstats/openrapm`) to `hybrid-and-xpts` so the site picks up the new board.
2. **DNS for openrapm.com** (unchanged, still not done): four GitHub `A` records + `www` CNAME at Porkbun,
   wait for the cert, then `gh api -X PUT repos/bbstats/openrapm/pages -F https_enforced=true`.

## Part 1: where it stands

| | criterion (K = 3, mapped) | 28 fits | true loss |
|---|---|---|---|
| shipped before this phase (`mspi_linear+sat`, section 20) | 111.30 | 134 s | 1.00 |
| the criterion's best now (`best`, map `linear+log2&age2&xlog&prior&tshare|rowcubic`) | **109.98** | **23.2 s** | **0.171** |
| what ships (`ship_blend07`, no held-out season, floors green) | 110.80 | 32 s | |

**True loss** = (score / 111.30) x (28-fit seconds / 134).  Every time cut was checked to reproduce the
ratings bit for bit (`scratch/cmp_design.py`, `cmp_counters.py`, `cmp_expo.py`, `cmp_x3def.py`).

### What moved the score (all leave-one-season-out; `calmap.py` unless said otherwise)

1. **The age term** in the map (age at H, quadratic): -0.15, 28 of 28.  Prediction-time only.
2. **H-2 at half weight** in the ridge rows and behind the padded rates (`decay`, `decay_exposure`): -0.10, -0.04.
3. **The map's terms**: log-exposure level, rating-by-log-exposure slope, the prior part of the rating as its
   own column (`linear+log2&age2&xlog&prior`): -0.11 and -0.11.
4. **The GBDT prior trained on the panel's unshrunk APM** (`target="apm"`): -0.29, z -3.1; with it the ridge
   goes back to the shipped value.  The less the prior's target is shrunk, the better the mapped board.
5. **A cubic in the team-game total after the map** (`|cubic`, section 21.20): -0.07 to -0.08 on every board
   tried, z -3.0 to -3.3.  Not reachable per player -- the prediction is the SUM of the five on the floor.
6. **The same cubic at STINT level beside it** (`|rowcubic`, 21.22): -0.205, z -4.8, 24/28.  (mean c)^3 and
   mean(c^3) differ by the spread of the lineups inside the team-game; the criterion wants both, opposite signs.
7. **The training block's role** (`tshare`, his share of his teams' possessions over the block): -0.058, z -1.8.
   Marginal, and it IS a rating -- unlike the held-out season's minutes, which are a leak (21.21).

### What moved the clock (identical numbers)

One-pass fit (`fastfit.MspiFast`), per-season pieces on disk (`designcache.py`), X in CSR, the exposure from
the design's parts, no edf, box / shots / shooter-totals caches, the GBDT without its audition fits, and then
this phase: **the design carries only the counters its targets read** (`counter_cols`, 118 columns -> 6), **X
built on demand** (`WindowData.X_src`), **the exposure's lineup sums as one sparse product**, and the shooter
rates by binary search.  134 s -> 23.2 s for the 28 fits.  `FASTFIT_TIMER=1` prints the split.

### The metric's hole (21.22, decision open)

Dropping short stints (`MspiFast.min_den`) improves the TRUE loss monotonically down to keeping only 12+
possession rows, where the board is worse than section 20's and the metric calls it 40% better.  Nothing on
that curve is taken.  The owner may want a floor on the loss instead of the bare product.

### What ships, and why not the best

Every board with the APM prior as is fails one of the owner's consensus floors (`tests/test_vs_consensus.py`):
the defensive agreement drops under 0.76, or the offensive rank gap against bigness passes 0.30.  **What
ships** (`ship_blend07`): the GBDT prior trained on 0.7 APM + 0.3 RAPM_1 on OFFENSE, RAPM_1 on defense, the
map `linear+log2&xlog&prior` on offense and `linear+log2&xlog` on defense: 110.80, ten of ten floors green.
The decay, the age term and the bends need a held-out season and do not ship.  Section 21.18-19 has the table.

### Flat or negative (do not re-run): section 21 items 4, 7, 12, 14, 20, 21, 22 and the reads inside 1, 2, 9, 17

lam_ratio, lam_buckets, GBDT shape and regularisation (both targets), playoff rows, one target for both sides,
x3def_p1, mover / rookie-age / age-by-exposure / rating-by-age / rating-by-exposure / prior-by-exposure /
prior^2 map terms, season weights beyond the decay, padding scale and target, panel APM at penalty 30, the
training-block team's mean rating (`team`), role growth, the lineup spread and the best/worst man on the floor,
the two sides of the bend apart, and the multiplicative offense-defense term.

## Part 2: machinery

| file | what |
|---|---|
| `scripts/54_track.py` | the tracker: dump a system at K = 3 (timed), fit the map leave-one-season-out, score, log a row, draw the chart.  `--systems=a,b --maps=... --label=...`; one dump per system |
| `src/eracoef/fastfit.py` | `MspiFast`: the one-pass board fit with every knob (`lam`, `lam_ratio`, `gbdt_params`, `target`, `target_d`, `panel`, `decay`, `decay_exposure`, `season_weights`, `pad_scale`, `pad_target`, `phases`, `lam_buckets`, `counter_columns`, `min_den`); `direct_layout`; `FASTFIT_TIMER=1` for the section clock |
| `src/eracoef/designcache.py` | per-season pieces (disk + LRU) and `build_window_cached`; `windows.build_window` routes to it unless `margin_bins` |
| `src/eracoef/calmap.py` | families x exposure terms (`&`-combinable: `sat`, `log2`, `age2`, `xlog`, `prior`, `tshare`, ...) x a team-level bend after the map (`\|cubic`, `\|rowcubic`: `TeamBend`, `fit_bend`, `row_columns`, `bent_prediction`), `parse_maps`, `SeasonFrame.covariates`, `apply_params(prior_o=, prior_d=)` |
| `src/eracoef/systems.py` | `best`, `ship`, `ship_mix`, `ship_rapm1`, `ship_blend07`, the `mspi1_*` variants |
| `scratch/` (untracked) | `maps.py` (score maps on an existing dump, ~2 s each: THE loop for map work), `timing_run.py` (a timed dump with the section clock), `cmp_counters.py` / `cmp_expo.py` / `cmp_x3def.py` / `cmp_design.py` (identical-numbers checks), `teamnl2.py` / `rowbend2.py` / `spread.py` (second-stage prototypes), `remap.py`, `consensus_read.py`, `panel_lam.py` |

### Verification

```
.venv/Scripts/python -m pytest tests -q                                                # 82 passed, 1 xfailed
.venv/Scripts/python scripts/54_track.py --systems=best --maps=linear+log2\&age2\&xlog\&prior\&tshare\|rowcubic --label=...   # ~60 s
.venv/Scripts/python scripts/08_ratings.py && .venv/Scripts/python scripts/52_site.py && .venv/Scripts/python scripts/22_vs_consensus.py
```

### Traps (new; the earlier ones stand)

- **One system per dump for timing**: the GBDT cache per process flatters a later system in the same run.
- **The tracker's `seconds` is the per-fit wall in a 4-worker run**, about 1.5x a single warm fit (contention);
  compare like with like.
- **`53_calmap.py fit` overwrites `outputs/holdout_calmap_<tag>.parquet`** with only the maps of that run.
- **`apply_params` needs the prior parts** for any map with a `prior` term; without them the term is silently 0.
- **The panel's APM target vs the consensus**: anything that widens the defensive prior trips
  `test_defensive_spread_is_calibrated` (1.4x), the 0.76 defensive-agreement floor, or the 0.30 offensive bigness gap.  Read the consensus before shipping.
- Long bash heredocs still fail in this shell; the patch scripts were written with the Write tool.

(Earlier handoffs: `HANDOFF_rankmap_archive.md`; the calibration-map handoff is in git history at d1bc3cc.)
