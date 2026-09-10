# Handoff: the board is one season now; rulings 1, 2 and 3 have landed

**This file is transient.** It exists to start the next session and should be deleted when Phase 1
ships. `DECISIONS.md` is the permanent record and carries every number quoted here; do not turn this
back into a lab notebook — the last one reached 7,800 lines and was deleted on purpose (tag
`archive/research-2026-09` has it).

Branch `cleanup`, 26 commits ahead of `main`, working tree clean, `main` untouched.
`pytest -q`: **229 passed, 1 xfailed, ~110 s.**

---

## What the owner decided, and what it now means

Rulings, not suggestions.

1. **One rating per player per season, from that season's games only** — regular season and playoffs
   together, nothing else. **Shipped 2026-09-10.** On low-minute players: *"low minute players should
   have REALLY GOOD REASONABLE PRIORS now. if they dont we should fix that eventually."* → item 2 below.
2. **The 3-year block is gone**, as a product and as a training substrate. Gone from the product; the
   *prior's panel* is still block-granular, which is the open half (item 1 below).
3. **The playoff delta is gone.** Playoff games join the fit like any other games. Done.
4. **Two layers.** Anyone may add a data source; the model layer may not open a file. *"anything that
   is ever trained on/fit on other than just for inference CANNOT USE the 'current NBA season' until
   the finals are over."* All data may always be **loaded**; the current season may never be **trained on**.
5. **Keep chimeraboost.** *"chimeraboost is great :) keep it for now. trying to get more users!"*
6. **PR gate: PBO, plus a runtime test that fails above 5x the baseline.**
7. **Delete hard**, tag first.
8. **The archetype guard is unsupervised** (2026-09-10): *"the guardrail is arbitrary ... would rather
   use a bayesian gaussian mixture (legit unsupervised clusters rather than center/big/guard)"*. Done.
   The open half is whether an archetype penalty belongs in the FIT, not just in a guard.

Standing rules: accuracy wins provided the testing is robust and the thing stays open-source-shippable;
the external consensus is a sanity check and never a fitting target; between two candidates the
criterion cannot separate, take the simpler and faster. **The criterion is the tiebreak between
candidates — which product to build is not a tiebreak, and a ruling names it.**

---

## What the board is today

`config.yaml` → `ratings_prior.season_board`: system `ks00_lam05_ow_w0.25` (kernel `{0: 1.0}`), map
`linear+sat` per side from `artifacts/calmap_insea_ks00_q75.parquet`. `artifacts/season_ratings.parquet`
and `docs/data/ratings.json` are built from it: 14,578 rows, 1997–2026, 1.0 MB.

Against the three-season board it replaced, on the 475 players both match in the consensus:

| | shipped (`ks00`) | the `ks52` board it replaced |
|---|---|---|
| criterion, mapped, K=3 q75 | 112.36 | **111.80** (+0.557, z +2.57, 10 of 28 seasons) |
| consensus total / offense / defense | **0.835** / **0.835** / 0.756 | 0.809 / 0.824 / 0.755 |
| defensive spread | 1.33 | 1.38 |
| archetype spread (`58_archetype.py`) | **0.145** | 0.213 |
| consensus total, under 5k own possessions | **0.651** | 0.643 |

Ten of ten floors in `tests/test_vs_consensus.py` pass. **The pooling question is settled**: the floors
stay pooled over the three seasons the consensus covers even though the board is single-season, because
the consensus snapshot is itself a multi-season blend and every floor was calibrated on that estimand.
All ten passed unchanged, so nothing needed re-basing except the archetype floor (0.30 → 0.25, as its own
docstring instructed). Score a candidate board without touching the shipped one:

    OPENRAPM_BOARD=outputs/season_ratings_cand.parquet .venv/Scripts/python -m pytest tests/test_vs_consensus.py -q

---

## Phase 1, what is left

### 1. The prior panel stays block-granular — DECIDED 2026-09-10, with one thing to fix

The board is one season; `artifacts/role_panel.parquet`, the prior's training substrate, is one row per
player-window and **stays that way for now**. The season twin was built and scored end to end
(`outputs/season_ratings_spanel.parquet`, from `sp_ks00_lam05_ow_w0.25`):

| | block panel (ships) | season panel |
|---|---|---|
| criterion | 112.36 | 112.42 (+0.063, z +0.70 — a tie) |
| consensus total / offense | 0.835 / 0.835 | **0.846** / **0.865** |
| consensus defense | **0.756** | **0.739 — below the 0.75 floor** |
| rating defensive sd | 1.33 | 1.22 |
| PRIOR defensive sd | **0.682** | **0.476** |
| archetype spread | 0.145 | **0.108** |

Nine floors pass; `test_defense_agrees_with_the_consensus` fails. The old claim that a season-granularity
prior "loses half its defensive spread" is **real, and it is not the missing `onc_*` columns** — this panel
has them. The defensive prior is 30% narrower and the finished defensive rating follows it down. Offense
gains from the finer substrate, so this is a trade, not a worse panel: **fix the defensive attenuation and
the season panel probably wins.** `spy` (cube-rooting `win_decay` to the same decay per year) is +0.002,
z +0.06, so it is not the fix by itself.

Reproduce with the commands at the bottom of this file. `--tag=spanel2` is the current dump; the earlier
`insea_spanel` one predates the zero-weight-season fix and should not be read.

### 2. Make the prior carry low-minute players

Ruling 1's second sentence. A single-season rating is **20% on-court evidence on offense and 44% on
defense**; for a bench player the prior essentially *is* the rating.

**Two things already measured, so do not redo them.** The bottom of the board is the calibration map's
exposure term, not a defect in the prior: players under 250 possessions take −2.85 from the map, and
re-fitting that map on the single-season kernel changed it by 0.08 (the `ks52` map gave −2.77 on the
same players). It is what the games ask for at 200 possessions. And the offensive prior's archetype
tilt (−0.469 against bigness, on every kernel and every map) is an offense/defense **attribution**
disagreement with the consensus, not a bias: the total gap is +0.068.

**The lever the owner named:** height and weight are in `bio.py` (`PLAYER_INPUTS`) and reach the GBDT
as `height2` / `weight15` bins (`gbdt_prior.BIO_BINS`), but never reach the SPM. `spm.fit_spm`
(`spm.py:81`) is a possession-weighted ridge of APM on seven role inputs — possession share and its
square, starts share and its square, age, age², age³ (`roles.design7`). No box score, no body. Adding
binned height and weight to `design7` is small and well-scoped. **The trap:** fine height and weight
together name a player almost uniquely and were +0.20 on the criterion at z 4.7 while being −0.27 on
the prior's own fit. Binned is the form that survived. Do not un-bin them.

Report, as a first-class number, predicted-versus-actual error for players under 500 possessions.

### 3. Finish re-picking the constants

Four are done and the old headline ("none of them transport") is already wrong:

| constant | verdict, 2026-09-10 |
|---|---|
| `lam_plugin` (the board's effective 5,726) | **unchanged.** Interior minimum, z +2.34 above and +1.16 below; ×0.71 tied at z −0.11 |
| `lam_ratio_plugin` (0.6245) | **unchanged.** Flat: ×0.5 and ×2 are +0.060 and +0.061, z +1.05 and +1.19 |
| `low_poss_threshold`, `starter_poss_threshold` | **1500/4500 → 500/1500.** Design possessions are one season now. Touches no rating (`lam_buckets` is empty; the 2026 fit is bit-identical) — it names the diagnostic groups |
| `boost_min_poss` (4500) | **dead.** No reader in `src/` or `scripts/`. Delete it with the booster |

`board_lam<m>` and `board_lr<m>` in `systems.py` register those sweeps on the exact system that ships:
re-run with `53_calmap.py dump --tag=lamsweep` then `fit --tag=lamsweep --maps=linear+sat`. **Every
sweep must bracket the current value on both sides** — the old lambda grid was {0.125, 0.25, 0.5, 2.0}
with no 1.0 in it, so 0.5 had won a boundary it was never asked to beat.

The three decays are **unblocked** now that the panel stays block-granular: `gbdt_win_decay` (0.514),
`gbdt_win_decay_def` (0.280) and `PAST_DECAY` (`gbdt_prior.py`, 0.5) are per WINDOW of that panel, and the
panel is the one that ships. Sweep them the same way, bracketing on both sides.

Left: `lam_scale`, the three decays, `gbdt.params` / `params_def`,
`features_full_O` / `_D`, `k3 = 450` (`xshoot.py:431`), `FACTOR_LAMS` (`fastfit.py:65`), and the shipped
targets `xpts_ft` / `x3def_w0.25`. `FACTOR_LAMS` and `xpts.FIXED_LAMBDA` carry comments saying they were
selected by REML **on 2024-2026** — the current block — and then held fixed everywhere, inside the
criterion included.

### 4. Not started at all

- **The poison test** (`tests/test_no_current_season.py`): rebuild every fitted artifact twice, once
  with the current season's parquets replaced by noise, and assert every artifact hashes identical.
  This is the real guarantee; `seasons.py` is only the courtesy. `in_progress()` is **empty** today
  (2026's Finals are over), so the test needs a synthetic in-progress season the way
  `tests/test_seasons.py` already does.
- **The merge gate as a command.** `pbo.py` exists and is tested; nothing calls it. Wire
  `openrapm evaluate --against=ship` to print the four numbers CONTRIBUTING.md promises: delta, seasons
  won, PBO, runtime ratio. The runtime one is a failing test above 5×.
- **The sealed confirm seasons.** A fixed subset whose per-season scores are never printed; CI returns
  one bit. Blum & Hardt's Ladder applied.
- **The CLI.** `openrapm ingest | stints | train | rate | evaluate | site` replacing the 13 numbered
  scripts, and with it the copies of `sys.path.insert(...)  # noqa: E402`.
- **An archetype penalty in the fit** (ruling 8's open half). `archetype.py` only reports today.

---

## Things that will cost you a day if you meet them fresh

1. **The editable install breaks fresh-clone testing.** `pip install -e` puts `A:\code\spmm\src` on
   `sys.path`, so a clone imports the *working tree's* `eracoef` and `config.ROOT` resolves to the main
   repo — it reads the main repo's data and every skip turns into a pass. Two verification attempts were
   silently wrong before this was caught. Use:

       PYTHONPATH='<clone>\src' A:/code/spmm/.venv/Scripts/python -m pytest tests -q

2. **Tests cannot reach the network** (`tests/conftest.py`, session-scoped — a function-scoped block
   runs *after* module-scoped fixtures, which is where the scraping was). `test_the_network_block_works`
   is the regression. If a test needs the network, take the `allow_network` fixture.

3. **`scripts/49_role_panel.py` does not reproduce the shipped panel.** The biggest open hole, and it
   blocks item 1.
   - It computed `onc_o` / `onc_d` / `onc_poss_*` in pass 1 and **dropped them at write time**.
     `gbdt_prior` takes them with `if c in p.columns`, so a panel without them trains a prior quietly
     missing the luck-adjusted on-court features *and* the whole `past_onc_*` family, and nothing fails.
     Fixed 2026-09-10 with an assert on the written column list.
   - A rebuild lands defensive shrinkage at **0.219** where `artifacts/role_panel.parquet` has **0.314**,
     and an isolated probe of the same `plugin_fit` on regular-season rows reproduces 0.209 — so the gap
     is not the playoff fold. Something built the shipped panel that is not in the repo.
   - The 49 `dr_*` columns the shipped panel carries have no reader in `src/` and can go.

4. **`grep check_trainable`** lists the call sites where the season boundary is a runtime check rather
   than a type. Converting them is how the boundary gets real.

5. **`NEEDS_SPLIT` in `tests/test_layer_boundary.py`** is the honest list of model code that still
   reaches its own data, with the specific import named. Eight modules are genuinely clean
   (`MODEL_LAYER`). Shrinking `NEEDS_SPLIT` is real work and nothing may be added to it.

6. **`systems.py` builds 1,446 systems** through nested loops and wants a rewrite to a small registry.

7. **Deferred deletions**, left because Phase 1 rewrites them: `windows.py` (still block-based and widely
   imported), `systems.py`, `teamloo.py`, `xpts.py` + `factors.py`, `checks.py`. `xshoot.py` imports
   `league_constants` and `lineup_rates` from `xpts.py` — lift those two before deleting it. `xpts_ft`,
   the shipped offensive target, lives in `design.TARGETS` and does **not** depend on `xpts.py`.

8. **Read the traps in `DECISIONS.md` before trusting any number.** Fifteen of them, each having produced
   a confident wrong number here. The four that cost the most recently:
   - **an in-season fit scored on the whole season is scored on its own training games.** `53_calmap.py
     fit` did this until 2026-09-10 and it *reversed* which kernel won. Any comparison involving a `_q*`
     system must show `cut` populated in its holdout rows;
   - **a weight of zero is not zero on the feature path.** `kernel_game_mult` zeroed the games of a
     zero-weight season, but the season stayed in the training LIST, and the inputs built from season
     tables (shot-quality features, role inputs) are built over whatever the list names — three seasons
     of shot data reaching a "one season only" rating, worth up to 0.61 per 100;
   - **an unmapped gain is not a gain** — the calibration map absorbed 99% of one candidate;
   - **an argmax on a grid boundary has chosen nothing.**

---

## Verify you are where this file says

    .venv/Scripts/python -m pytest tests -q          # 229 passed, 1 xfailed, ~110 s
    .venv/Scripts/python scripts/60_season_board.py  # ks00_lam05_ow_w0.25, kernel {0: 1.0}, ~70 s
    .venv/Scripts/python scripts/52_site.py          # 14,578 rows, 30 seasons, 1.0 MB
    .venv/Scripts/python scripts/58_archetype.py     # 0.145 spread, 8 clusters, centres at +0.11
    git log --oneline -1                             # 49a4f59 "The board's two ridge constants..."

**First concrete step:** re-pick `gbdt_win_decay_def` (0.280) on the block panel. It is the constant that
sets how much of a player's other windows the DEFENSIVE prior pools, defence is where both open problems
live (the panel's attenuation and the 0.756 floor), and it is per-window on a substrate that is now settled.
Register the sweep beside `board_lam<m>` in `systems.py`, bracket 0.280 on both sides, then:

    .venv/Scripts/python scripts/53_calmap.py dump --systems=<the sweep>_q75 --k=3 --workers=4 --tag=decaysweep
    .venv/Scripts/python scripts/53_calmap.py fit  --systems=<the sweep>_q75 --k=3 --maps=linear+sat --tag=decaysweep

Reproducing the panel decision, if it is questioned (`SP=sp_ks00_lam05_ow_w0.25`):

    .venv/Scripts/python scripts/53_calmap.py dump --systems=${SP}_q75,spy_ks00_lam05_ow_w0.25_q75 --k=3 --workers=4 --tag=spanel2
    .venv/Scripts/python scripts/53_calmap.py fit --systems=${SP}_q75,spy_ks00_lam05_ow_w0.25_q75 --k=3 --maps=linear+sat --tag=spanel2
    .venv/Scripts/python scripts/60_season_board.py --system=$SP --map=outputs/calmap_spanel2.parquet --map-system="${SP}_q75_linear+sat" --map-base=${SP}_q75 --out=season_ratings_spanel
    OPENRAPM_BOARD=outputs/season_ratings_spanel.parquet .venv/Scripts/python -m pytest tests/test_vs_consensus.py -q

The dump is ~6 minutes and the board ~8 (the season-panel prior is heavier than the block one).
