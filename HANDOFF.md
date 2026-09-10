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

Everything below is unstarted. It is the whole of ruling 1, 2 and 3.

### 1. Fold playoffs into the fit

`MspiFast.phases` already exists (`fastfit.py:253`) and `("RS","PO")` already works —
`systems.py` has `mspi1_lam05_po` using it. The design carries a `playoff` fixed-effect column and
`config.yaml:18 neutral_site_seasons_po: [2020]` already zeroes home court for the bubble.

What is **not** done, and each is a place playoffs are silently dropped:

- `roles.py:110` and `roles.py:237` — `load_gamelog(season, "RS", cfg)`. Role inputs (minutes,
  starts, possessions) are regular-season only.
- `xshoot.py:66`, `:141` onward — shooter totals that price the luck-adjusted targets are RS-only.
- `holdout.py:208` — `season_box([season], ["RS"], ...)`, and the `phases=("RS",)` defaults at
  `holdout.py:80` and `:260`. Those defaults are what every caller silently inherits.
- `exposure.py:283-296` — `pad_k: auto` split-half constants. Confirm the halves are built from
  pooled RS+PO, not RS alone.

DECISIONS.md records that two of the three fixes needed to make the playoff delta work were bugs in
shared code, one of them *"the exposure always filtered to RS"*. Expect more of that shape.

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

7. **Read the traps in `DECISIONS.md` before trusting any number.** Especially: an unmapped gain is
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

First concrete step: `python scripts/60_season_board.py --first=2026 --last=2026` and look at what a
single-season fit does to the bottom of the board.
