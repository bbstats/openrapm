# Handoff: iterate-and-improve mode

Written 2026-09-06. The owner's instruction for this phase: keep improving the K = 3 out-of-season team-game
error, charge every fit for its time, post the chart, write nothing but FINDINGS.  `FINDINGS.md` section 21
(items 1-18) is the record; `docs/progress.png` / `docs/progress.csv` the chart and its log.

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
| the criterion's best now (`best`) | 110.32 | 32 s | 0.24 |
| what ships now (`ship_rapm1`, no held-out season, floors green) | 111.15 | 32 s | |

**True loss** = (score / 111.30) x (28-fit seconds / 134).  Every time cut was checked to reproduce the
ratings to 1e-13 (`scratch/cmp_design.py` for the design, the ratings-vs-dump check in the scratch scripts).

### What moved the score (all leave-one-season-out, all in `src/eracoef/calmap.py` or `fastfit.py`)

1. **The age term** in the map (age at H, quadratic): -0.15, 28 of 28.  Prediction-time only.
2. **The ridge x0.5** under the map: -0.16 (later moot, see 5).
3. **H-2 at half weight** in the ridge rows and behind the padded rates (`decay`, `decay_exposure`): -0.10 and -0.04.
4. **The map's terms**: log-exposure level, rating-by-log-exposure slope, the prior part of the rating as its own
   column (`linear+log2&age2&xlog&prior`): -0.11 and -0.11.
5. **The GBDT prior trained on the panel's unshrunk APM instead of RAPM_1** (`target="apm"`): -0.29, z -3.1,
   the largest single gain, and with it the ridge goes back to the shipped value.  A sweep of the panel's
   ridge (x0.7 ... x0.15 ... APM) is monotone: the less the prior's target is shrunk, the better the mapped
   board.  Unmapped it is worse -- the map does the shrinking.

### What moved the clock (identical numbers)

One-pass fit (`fastfit.MspiFast`: one design, one exposure, one offset, cross-products once, two solves), the
design assembled from cached per-season pieces kept on disk (`designcache.py`, `data/cache/pieces/`), X written
straight in CSR, the exposure fitted from the design's parts, no edf trace, sparse accumulators, box / shots /
shooter-totals caches, the GBDT without its audition fits (`gbdt.params`).  134 s -> 32 s for the 28 fits.

### What ships, and why not the best

Every board with the APM prior fails one of the owner's consensus floors (`tests/test_vs_consensus.py`): on
defense the agreement drops under 0.76 (0.68 with it on both sides, 0.751 with the prior term on defense
only), on offense the rank gap against bigness passes 0.30 (-0.30 to -0.35 for every map tried), and the
half-ridge panel prior fails defense too (0.74-0.75).  **What ships** (`ship_rapm1`): the RAPM_1 prior on both
sides, the GBDT without its audition fits, the map `linear+log2&xlog` (no prior term: that one trips the
defensive floor): criterion 111.15, consensus 0.768 / 0.769 / 0.765, spread 1.33, ten of ten floors.
`config.yaml -> gbdt.params`, `ratings_prior.cal_map -> outputs/calmap_ship.parquet`; `gbdt_target`,
`gbdt_target_def`, `gbdt_panel` are wired into `08_ratings.py` and off.  The decay and the age term need a
held-out season and do not ship.  Section 21.18 has the candidate table.

### Flat or negative (do not re-run): section 21 items 4, 7, 12, 14 and the reads inside 1, 2, 9, 17

lam_ratio, lam_buckets, GBDT shape and regularisation (both targets), playoff rows, one target for both sides,
x3def_p1, mover / rookie-age / age-by-exposure / rating-by-age / rating-by-exposure / prior-by-exposure /
prior^2 map terms, season weights beyond the decay, padding scale and target, panel APM at penalty 30.

## Part 2: machinery

| file | what |
|---|---|
| `scripts/54_track.py` | the tracker: dump a system at K = 3 (timed), fit the map leave-one-season-out, score, log a row, draw the chart.  `--systems=a,b --maps=... --label=...`; one dump per system |
| `src/eracoef/fastfit.py` | `MspiFast`: the one-pass board fit with every knob (`lam`, `lam_ratio`, `gbdt_params`, `target`, `target_d`, `panel`, `decay`, `decay_exposure`, `season_weights`, `pad_scale`, `pad_target`, `phases`, `lam_buckets`); `direct_layout` |
| `src/eracoef/designcache.py` | per-season pieces (disk + LRU) and `build_window_cached`; `windows.build_window` routes to it unless `margin_bins` |
| `src/eracoef/calmap.py` | families x exposure terms (`&`-combinable: `sat`, `log2`, `age2`, `xlog`, `prior`, ...), `SeasonFrame.covariates`, `apply_params(prior_o=, prior_d=)` |
| `src/eracoef/systems.py` | `best`, `ship`, `ship_mix`, `ship_rapm1`, the `mspi1_*` variants |
| `scratch/` (untracked) | `cmp_design.py` (design equality vs git HEAD), `remap.py` (re-score dumps with a map), `consensus_read.py` (mapped candidates against the consensus), `panel_lam.py` (role panel at a scaled ridge / APM penalty), the patch scripts |

### Verification

```
.venv/Scripts/python -m pytest tests -q                                                # 82 passed, 1 xfailed
.venv/Scripts/python scripts/54_track.py --systems=best --maps=linear+log2\&age2\&xlog\&prior --label=...   # ~45 s
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
