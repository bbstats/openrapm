# Handoff: the calibration map ships

Written 2026-09-05. The owner asked for a smooth function of the offensive and defensive ratings that is
best calibrated out of season. Answer, measured at team-game level: **no function of the rating alone helps
the multi-stage board**, it is already calibrated at game level. A **level term in the player's block
exposure** is worth 0.8-1.0 per 100 and ships. `FINDINGS.md` section 20 is the full record.

**Tree state: everything below is UNCOMMITTED on `hybrid-and-xpts`.** Tests: 82 passed, 1 xfailed.

---

## Part 0: do first

1. **Commit and push.** `git add -A && git commit`, push `hybrid-and-xpts`, fast-forward `main`
   (`bbstats/openrapm`). `docs/data/ratings.json` is already regenerated with the map.
2. **DNS for openrapm.com** (unchanged, still not done): four GitHub `A` records + `www` CNAME at Porkbun,
   wait for the cert, then `gh api -X PUT repos/bbstats/openrapm/pages -F https_enforced=true`.

## Part 1: what was found

1. **A map of the rating alone is the identity.** Games want a scalar of 0.96-1.03 per side at every K;
   quadratic, cubic, sinh, exp and two-tail hinge are all within 0.03 per 100 of the unmapped board.
   The stint-level x1.27 of section 19 is a **starter-vs-bench LEVEL error absorbed by the game-level
   intercept**, not top-end timidity. Fit corrections on the objective you score.
2. **The exposure term is the lever.** `linear+sat` = per side, a scalar on the rating plus
   `c * poss/(poss+1000)` in block possessions, fitted leave-one-season-out on team-game residuals:
   **-0.79 / -0.91 / -1.01 per 100 vs `mspi` at K=2/3/4, z -5.7 / -6.3 / -7.1, 24-25 of 28. WINS.**
   Half is the unseen player, half the gradient among players the block did see. Richer shapes
   (`log2`, `bins`, `poly2+sat`, `hinge+sat`, sat scales 250-4000) are all within 0.06. Not wins.
3. **The old board ties this one once both are mapped.** `def3_p0`'s offense is 22% too wide at game
   level; a scalar alone takes -0.58 off it, and `def3_p0_linear+sat` is within 0.05 of `mspi_linear+sat`
   at every K. The chain's section-19 edge WAS the old board's amplitude. What still separates them is
   the Stockton problem and the consensus read, not the criterion.
4. **The top-end question is unanswered.** With exposure in the fit the good tails are calibrated on both
   sides (~1.0); the BAD tails want compressing. The criterion cannot adjudicate "defenders too high" at
   this sample size. If the owner wants it moved it is a modelling change, not a map.
5. **Consensus** 2024-26, read once: 0.785 / 0.779 / 0.768 with the map, from 0.772 / 0.778 / 0.785.
   Archetype bias 0.12 from 0.21. Same top ten. Test floors (0.75/0.76/0.75) hold.
6. **Unseen players** keep the owner's rule of 0 (the exposure term is 0 at poss=0), but every rated
   player moves up, so after re-centring an unseen player sits ~5 per 100 below a regular. The
   replacement level came back as the poss->0 end of one fitted function, not as a rule.

### Shipped

`config.yaml -> ratings_prior.cal_map: {table: outputs/calmap_chain.parquet, system: mspi_linear+sat,
base: mspi, k: 3}`. `08_ratings.py` applies the all-seasons K=3 row (offense `0.849x + 2.883 sat`,
defense `0.885x - 3.148 sat`, raw sign), keeps `rating_*_raw`, and re-centres each side possession-weighted
per window. **K=3 because the shipped block is 3 seasons and the exposure coefficient scales with block
length** (2.883 at K=3, 3.405 at K=4).

## Part 2: machinery

| file | what |
|---|---|
| `src/eracoef/calmap.py` (new) | `dump_systems`/`dump_ratings`, `SeasonFrame` (held-out season reduced to lineup matrices + level projection + team-game aggregation), `FAMILIES` (rating shapes) x `EXPOSURES` (level terms) via `SideMap` ("family+exposure"), `build_design`/`fit_theta`/`evaluate` (LOO through the criterion's own scorer), `CalMappedSystem` |
| `scripts/53_calmap.py` (new) | `dump` -> `outputs/ratings_<tag>.parquet`; `fit` -> `outputs/holdout_calmap_<tag>.parquet` + `outputs/calmap_<tag>.parquet` (params; `held_out=-1` is the all-seasons fit) |
| `systems.py`, `holdout.py`, `45_holdout.py` | `registry(cfg, rankmap, calmap)`; `--calmap=` threads through the parallel runner |
| `08_ratings.py`, `config.yaml` | the `cal_map` block |
| `tests/test_calmap.py` (new, 5) | incl. the pooled WLS == a direct grid search on the criterion's team-game score |

### Verification

```
.venv/Scripts/python -m pytest tests -q                                                    # 82 passed, 1 xfailed
.venv/Scripts/python scripts/53_calmap.py dump --systems=mspi,def3_p0 --k=2,3,4 --workers=4 --tag=chain   # ~3.5 min
.venv/Scripts/python scripts/53_calmap.py fit  --tag=chain --systems=mspi,def3_p0 --k=2,3,4               # ~1.5 min
.venv/Scripts/python scripts/08_ratings.py && .venv/Scripts/python scripts/52_site.py && .venv/Scripts/python scripts/22_vs_consensus.py
```

### Traps (new this session; earlier ones all still stand)

- **Long bash heredocs fail in this shell.** Write the text with the Write tool and `cat >>` it.
- **The exposure term needs exposure to vary**, or the column is collinear with the intercept and
  `fit_theta` raises "Singular matrix". The tests give their table varying possessions for this reason.
- **`level: full` on a one-season frame has a duplicate intercept column**; `SeasonFrame` uses `pinv`.
- **`neighbourhood()` is asymmetric at odd K**: K=3 is {H-2, H-1, H+1}, because `H-d` is taken before
  `H+d`. Flips at the era edges.
- **The LOO wall is not perfectly clean.** The map for H is fitted on rows from H's neighbours, whose
  ratings were trained on blocks that include H. 4 params on ~66k rows so the influence is tiny, but the
  channel exists and the old rank map had it too. Worth one sentence if anyone re-derives this.

### Standing warning, kept

Added this session: "offense is too timid" was true at stint level and false at game level, and "the chain
beats the old board" was really "the old board's offense is too wide". Measure first, on the objective
you score.

(Earlier handoffs: `HANDOFF_rankmap_archive.md`; FINDINGS 19 has the chain's own build and diagnostics.)
