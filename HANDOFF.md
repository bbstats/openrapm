# Handoff: the cleanup landed, Phase 1 has not started

**This file is transient.** It exists to start the next session and should be deleted when Phase 1
ships. `DECISIONS.md` is the permanent record; do not turn this back into a lab notebook — the last
one reached 7,800 lines and was deleted on purpose (tag `archive/research-2026-09` has it).

Branch `cleanup`, 14 commits, working tree clean, `main` untouched.

---

## What the owner decided, 2026-09-10

These are rulings, not suggestions. They came from direct questions this session.

1. **One rating per player per season, from that season's games only** — regular season and playoffs
   together, nothing else. Not the current kernel (2026 full + 2025 half + 2024 quarter). The owner,
   on low-minute players: *"low minute players should have REALLY GOOD REASONABLE PRIORS now. if they
   dont we should fix that eventually."*
2. **The 3-year block is gone**, as a product and as a training substrate.
3. **The playoff delta is gone.** Playoff games just join the fit like any other games, with playoff
   box scores padded like any other box scores.
4. **Two layers.** A data-loading layer where anyone may add a source (*"if they have tracking data
   they can/want to load in, that is fine!"*), and a modelling layer. *"anything that is ever trained
   on/fit on other than just for inference CANNOT USE the 'current NBA season' until the finals are
   over."* All data may always be **loaded**; the current season may never be **trained on**.
5. **Keep chimeraboost.** *"chimeraboost is great :) keep it for now. trying to get more users!"*
6. **PR gate: PBO, plus a runtime test that fails above 5x the baseline.**
7. **Delete hard**, tag first.

Standing rules from before, still binding: accuracy wins provided the testing is robust and the
thing stays open-source-shippable; the external consensus is a sanity check and never a fitting
target; between two candidates the criterion cannot separate, take the simpler and faster.

---

## What is done

| | where |
|---|---|
| 51 of 64 scripts, all tracked `scratch/`, 5 modules, 252 artifacts deleted | `73f0ab9` |
| `seasons.py` — the trainable/current-season boundary | `ca4b15e`, `6a4187a` |
| `test_layer_boundary.py` — model code may not load data, directly or via a sibling | `27044a7`, `f0d7454` |
| `pbo.py` — probability of backtest overfitting, thresholds measured | `f855c26` |
| `DECISIONS.md` — 7,800 lines of notebook down to 350 | `0f7c0b2` |
| Site: one table, season picker; block and playoff views gone | `8248102` |
| Consensus floors repointed at the season board | `911932e` |
| README, CONTRIBUTING, CI | `d89d668`, `71edf8d` |
| Network blocked in tests; offline suite made fast | `49babdf`, `f8649a7` |

455 tracked files → 90. 85 scripts → 12. `.gitignore` 288 lines → 31.

`pytest -q` on a fresh clone: **194 passed, 16 skipped, 90 s, nothing downloaded.**

---

## Phase 1 — the work, in order

Ruling 3 is done (`9b281b3`). Rulings 1 and 2 are not.

### 1. Fold playoffs into the fit — DONE, 2026-09-10

`design.FIT_PHASES = ("RS", "PO")` is the one training default, shared by `windows.build_window`,
`designcache.build_window_cached`, `holdout.default_loader`, `Context.design`, `exposure.BoxExposure`
and `fastfit.MspiFast`. `roles.ROLE_PHASES` and `xshoot.SHOT_PHASES` carry the role inputs and the
shooter totals. `tests/test_playoffs_in_fit.py` pins all of it.

The owner then ruled: *"RS + playoffs should be considered a single entity in our new version."* So
there is **one** constant, `design.SEASON_PHASES`, and the criterion scores playoff games too. There
is no second constant and `tests/test_playoffs_in_fit.py` fails if one appears.

That move exposed a bug that had hidden the playoffs on both sides at once. `season_frac` gave a
playoff game **−1**, which is below every cut — so it was excluded from training by a special case in
`kernel_game_mult` and excluded from *scoring* by the same comparison. It was in neither half. The
first attempt to move the estimand came back identical to four decimal places, which is how it was
found. A playoff game now scores 1.0.

**Every criterion number below this line is on the new estimand and does not compare to one measured
before 2026-09-10.** The shipped board reads 113.11 per 100 at team-game level at q75, against 109.36
on regular-season rows alone. On the estimand that includes them the fold is worth +0.057 per 100
(z 2.28); on the old one it read +0.022 (z 0.83) and looked like nothing.

Two things this file expected to be broken were correct by construction, and the notes are kept
because the reasoning is not obvious: `roles.cut_role_inputs` reads regular-season stints because
`inseason.keep_games` names regular-season ids only, and `kernel_game_mult` already zeroes the anchor
season's playoff games under a cut. A fit that has seen the first q of a season cannot see its
playoffs. `exposure.py`'s split-half `pad_k` needed no change either: the halves are built from
whatever `BoxExposure.phases` names, so flipping that default was the whole fix.

The one caveat on the measurement: at q75 the anchor season's playoffs are excluded by the cut, so
the fold only reaches H-1 and H-2's playoff rows plus the role and shooter inputs. On the BOARD,
where there is no cut, it reaches the rated season's own playoffs — which is the case that matters
for the product and which the criterion, by construction, cannot score.

### 2. Move to one row per player per season

`scripts/49_role_panel.py --season` already builds a per-season panel and
`tests/test_season_panel.py` covers the label handling (`label_step` returns 1.0 not 3.0).

Target: features = that season's padded box rates + role inputs + bio; target = his APM/RAPM_1
pooled over his **other seasons**, discounted by distance. Roughly eight other seasons instead of
three other blocks, so the target is no noisier.

**The +0.52/100 measurement in DECISIONS.md does not apply to this.** It was for training on
one-season rows and *scoring* three-season rows. Train on seasons and score seasons and there is no
mismatch. Verify that claim rather than trusting this paragraph.

### 3. Re-pick every constant, on trainable seasons only

All of these were calibrated at three-season scale and none of them transport:

`lam_plugin` (18351.8), `lam_ratio_plugin`, `lam_scale`, `gbdt_win_decay` (0.514 *per window*),
`gbdt_win_decay_def`, `PAST_DECAY` (`gbdt_prior.py`, 0.5 per window), `low_poss_threshold` (1500),
`boost_min_poss` (4500), `gbdt.params` / `params_def`, `features_full_O` / `_D`,
`k3 = 450` (`xshoot.py:431`), `FACTOR_LAMS` (`fastfit.py:65`), and the shipped targets
`xpts_ft` / `x3def_w0.25`.

`FACTOR_LAMS` and `xpts.FIXED_LAMBDA` carry comments saying they were selected by REML **on
2024-2026** — the current block — and then held fixed everywhere including inside the criterion.

### 4. Make the prior carry low-minute players

This is ruling 1's second sentence and it matters more than usual now. A single-season rating is
**20% on-court evidence on offense and 44% on defense** (DECISIONS.md); the rest is the prior. For a
bench player the prior essentially *is* the rating.

**Measured first, 2026-09-10: the bottom of the board is the calibration map, not the prior.** On a
single-season fit of 2026 (`ks00_lam05_ow_w0.25`), all 52 players under 250 possessions land between
−6.18 and −3.60. Not one is average and the spread among them is 0.70. The prior does not say that —
it says −1.86, and `u` adds −0.04. The `log2` + `xlog` exposure terms in `calmap_insea_ship2_q75`
add the other −3.10, and they were fitted on `ks52` at q75, where the same player carried 1.75× the
kernel-weighted possessions. The shipped board has the same defect, milder (−2.74 at <250):

| his own possessions | n | prior + u | shipped rating | the map |
|---|---|---|---|---|
| <250 | 52 | −1.90 | −5.00 | −3.10 |
| 250–500 | 35 | −1.28 | −4.08 | −2.80 |
| 500–1k | 43 | −1.66 | −3.85 | −2.19 |
| 4k+ | 288 | +0.25 | +0.31 | +0.06 |

**That reading was wrong, and the re-fit disproved it, 2026-09-10.** The map DOES transport. Fitted
on the single-season kernel's own dump (`outputs/calmap_insea_ks00_q75.parquet`, family `linear+sat`,
scored after the cut) the bottom bucket takes −2.85 where the `ks52`-fitted map took −2.77 on the same
51 players — refitting made it very slightly *stronger*, not weaker. The exposure term is what the
criterion asks for at 200 possessions, not a stale constant, so item 4 is NOT item 3. Anything done
here has to beat the map on the criterion, not merely argue that the bottom looks too low.

| his own possessions | n | prior + u | `ks52` map | its own map |
|---|---|---|---|---|
| <250 | 51 | −1.83 | −4.72 | −4.68 |
| 250–500 | 35 | −1.46 | −3.82 | −3.96 |
| 500–1k | 43 | −1.68 | −3.35 | −3.63 |
| 4k+ | 293 | +0.23 | +0.24 | +0.18 |

Height and weight in the SPM are still the obvious lever, and the note below still stands.

**The demographic-only prior already exists and the owner had forgotten.** `spm.fit_spm`
(`spm.py:81`) is a possession-weighted ridge of APM on seven role inputs — possession share, its
square, starts share, its square, age, age², age³ (`roles.design7`). No box score at all. It is
registered as the system `spm` and ships as the offset the box prior builds on.

What it is missing is what the owner named: **height and weight**. Those live in `bio.py`
(`PLAYER_INPUTS`) and reach the GBDT as `height2` / `weight15` bins (`gbdt_prior.BIO_BINS`) but
never reach the SPM. Adding them to `design7` is small and well-scoped, and is the obvious first
lever. Note DECISIONS.md's trap: fine height and weight together name a player almost uniquely and
were **+0.20 on the criterion at z 4.7** while being −0.27 on the prior's own fit. Binned is the
form that survived. Do not un-bin them.

Report, as a first-class number, predicted-vs-actual error for players under 500 possessions.

### 5. Measure before switching

The owner asked to measure both. Do not replace the live board until the single-season board has
been scored against the current kernel board on the criterion, broken out by possession bucket.

---

## Baselines to beat

The season board as it stands today (kernel `ks52_lam05_ow_w0.25`, RS only), pooled over 2024-2026,
484 matched players, against `data/external/consensus.csv`:

| | season board | the block board it replaced |
|---|---|---|
| total | **0.810** | 0.793 |
| offense | **0.822** | 0.789 |
| defense | **0.758** | 0.768 |
| defensive spread | 1.39 | 1.30 |

Ten of ten floors in `tests/test_vs_consensus.py` pass. Those floors are now computed on the season
board *pooled over the three seasons the consensus covers* — deliberately, so the estimand matches
what they were calibrated against. When the board becomes single-season, decide explicitly whether
to keep pooling for this test or re-base the floors, and write the reason into the test.

---

## Things that will cost you a day if you meet them fresh

1. **The editable install breaks fresh-clone testing.** `pip install -e` puts `A:\code\spmm\src` on
   `sys.path`, so a clone imports the *working tree's* `eracoef` and `config.ROOT` resolves to the
   main repo — it reads the main repo's data and every skip turns into a pass. Two verification
   attempts were silently wrong before this was caught. Use:

       PYTHONPATH='<clone>\src' A:/code/spmm/.venv/Scripts/python -m pytest tests -q

2. **Tests cannot reach the network** (`tests/conftest.py`, session-scoped). The block is
   session-scoped because a function-scoped one runs *after* module-scoped fixtures, which is where
   the scraping was. `test_the_network_block_works` is the regression. If you need the network in a
   test, take the `allow_network` fixture.

3. **`grep check_trainable`** is the list of call sites where the season boundary is a runtime check
   rather than a type. Converting them is how the boundary gets real.

4. **`NEEDS_SPLIT` in `tests/test_layer_boundary.py`** is the honest list of model code that still
   reaches its own data, with the specific import named. Only seven modules are genuinely clean.
   Shrinking that list is real work and nothing may be added to it.

5. **`systems.py` builds 1,326 systems** through nested loops and needs a rewrite to a small
   registry. Do it as part of Phase 1, not before — the season rewrite touches it anyway.

6. **Deferred deletions**, left because Phase 1 rewrites them: `windows.py` (still block-based and
   widely imported), `systems.py`, `teamloo.py`, `xpts.py` + `factors.py`, `checks.py`. `xshoot.py`
   imports `league_constants` and `lineup_rates` from `xpts.py` — lift those two functions before
   deleting it. `xpts_ft`, the shipped offensive target, lives in `design.TARGETS` and does **not**
   depend on `xpts.py`.

7. **`scripts/49_role_panel.py` does not rebuild the shipped panel.** This is the biggest open hole
   and it blocks Phase 1 item 2, which needs `--season` to be trustworthy.
   - It computes `onc_o` / `onc_d` / `onc_poss_*` in pass 1 and then **dropped them at write time**.
     `gbdt_prior` takes them with `if c in p.columns`, so a panel without them trains a prior quietly
     missing the luck-adjusted on-court features *and* the whole `past_onc_*` family, and nothing
     fails. Fixed 2026-09-10, with an assert on the written column list.
   - `outputs/role_panel_season.parquet` — what every `sp_*` system reads — **has never had them**
     (103 columns against the block panel's 107). Some of "a season-granularity prior loses half its
     defensive spread" may be this bug rather than granularity. Rebuild it before believing that.
   - A rebuild lands defensive shrinkage at **0.219** where the shipped `artifacts/role_panel.parquet`
     has **0.314**, and an isolated probe of the same `plugin_fit` on regular-season rows alone
     reproduces 0.209 — so the gap is not the playoff fold. Something built the shipped panel that is
     not in the repo. Deciding whether to accept the rebuilt panel needs the criterion, not an
     argument; it is a different prior, not a refresh.
   - The 49 `dr_*` columns the shipped panel carries have no reader in `src/` and can go.

8. **Read the traps in `DECISIONS.md` before trusting any number.** Especially: an unmapped gain is
   not a gain (the calibration map absorbed 99% of one candidate); a feature can predict how
   well-*measured* a row is rather than how good the player is; an argmax on a grid boundary has
   chosen nothing.

---

## Not started at all

- **The poison test** (planned as `tests/test_no_current_season.py`): rebuild every fitted artifact
  twice, once with the current season's parquets replaced by noise, and assert every artifact hashes
  identical. This is the real guarantee; `seasons.py` is only the courtesy. Note that today
  `in_progress()` is **empty** — 2026's Finals are over — so the test needs a synthetic in-progress
  season, the way `tests/test_seasons.py` already does.
- **The merge gate as a command.** `pbo.py` exists and is tested; nothing calls it yet. Wire
  `openrapm evaluate --against=ship` to print the four numbers CONTRIBUTING.md promises: delta,
  seasons won, PBO, runtime ratio. The runtime one is a failing test above 5x.
- **The sealed confirm seasons.** A fixed subset whose per-season scores are never printed, CI
  returns one bit. Blum & Hardt's Ladder applied.
- **The CLI.** `openrapm ingest | stints | train | rate | evaluate | site` replacing the 12 numbered
  scripts, and with it the 62 copies of `sys.path.insert(...)  # noqa: E402`.

---

## Verify you are where this file says

    .venv/Scripts/python -m pytest tests -q          # 209 passed, 1 xfailed, ~100 s
    .venv/Scripts/python scripts/52_site.py          # 14,568 rows, 30 seasons, 1.0 MB
    git log --oneline -14                            # ends at 73f0ab9 "Phase 0: delete the research sprawl"

That first step is DONE, 2026-09-10, and it answered three questions and found a leak. See DECISIONS.md.

  * `53_calmap.py fit` was scoring an in-season fit on the WHOLE held-out season, including the games it
    trained on. It flipped the headline: leaked, the single-season kernel beat the three-season one 28 of
    28 at z 9.3; scored after its own cut it loses 19 of 28. `calmap.frames_for_dump` + `tests/test_calmap_cut.py`.
  * **Ruling 1 costs +0.606 per 100, z +2.91, 9 of 28 seasons** (`ks00` 112.41 against `ks52` 111.80, K=3,
    q75, each on its own map). The map does not recover it.
  * **Item 4's premise is wrong.** Refitting the map on the single-season kernel leaves the bottom of the
    board where it was: the 51 players under 250 possessions in 2026 take −2.85 from their own kernel's map
    against −2.77 from the shipped one. The exposure term is not a stale constant; the games ask for it.
  * The shipped map FAMILY no longer earns its parameters on the moved estimand: `linear+sat` ties
    `linear+log2&xlog&prior&tshare : linear+log2&xlog` at z −0.12. Take the simpler one.
  * The rebuilt season panel is a tie on the criterion (`sp_ks00` vs `ks00`: +0.008, z +0.10), so item 2's
    granularity question is a product question, not a prediction one.

Artifacts: `outputs/calmap_insea_ks00_q75.parquet` (both kernels' maps), `outputs/season_ratings_ks00.parquet`
(the single-season board, built with `--map-system=`/`--map-base=`, new flags on `60_season_board.py`).

Item 5 is also done. The single-season board is **better against the consensus in every bucket** (0.839
total against the shipped 0.809, and 0.675 against 0.643 among players with under 5,000 of their own
possessions) while being 0.606 per 100 worse on the criterion. Nine floors pass;
`test_offense_has_no_big_man_bias` reads −0.437 against its 0.35 floor and was LEFT FAILING — it has been
re-based twice already and its own comment says a third time wants the owner. Score a candidate board with

    OPENRAPM_BOARD=outputs/season_ratings_ks00.parquet .venv/Scripts/python -m pytest tests/test_vs_consensus.py -q

Next concrete step: put the bigness number in front of the owner — the single-season board lifts guards over
bigs relative to the consensus, further than the shipped board does, and that is the one thing standing between
`season_ratings_ks00.parquet` and being the board. Everything else about ruling 1 is measured.

`60_season_board.py` takes `--out=<stem>` now. A one-season or candidate run used to overwrite
`outputs/season_ratings.parquet`, which `tests/test_vs_consensus.py` reads as the shipped board; it
then fails as a big-man-bias assertion, which looks nothing like a clobbered artifact.
