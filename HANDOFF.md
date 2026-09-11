# Handoff: the bottom of the board is measurable now, and one ruling is waiting

**This file is transient.** It exists to start the next session and should be deleted when Phase 1
ships. `DECISIONS.md` is the permanent record and carries every number quoted here; do not turn this
back into a lab notebook — the last one reached 7,800 lines and was deleted on purpose (tag
`archive/research-2026-09` has it).

Branch `cleanup`, 27 commits ahead of `main`, working tree clean, `main` untouched.
`pytest -q`: **231 passed, 1 xfailed, ~110 s.**

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

### 2. Make the prior carry low-minute players -- MEASURED, and one ruling is waiting

**The criterion cannot see this question.** It is scored at team-game level, where a 200-possession player is
a rounding error, so it calls every change to the bench a tie. `scripts/61_lowposs.py` is the instrument that
can see it: refit the prior with one panel window excluded, predict that window's own rows, compare to the
row's training target, and report by POSSESSIONS PER SEASON. On the shipped board, skill by bucket:

| poss per season | <250 | 250-500 | 500-1500 | 1500-4500 | 4500+ |
|---|---|---|---|---|---|
| offense | 0.193 | 0.173 | 0.179 | 0.255 | **0.480** |
| defense | 0.096 | 0.160 | 0.171 | 0.265 | **0.327** |

A starter's prior is two to five times as good as a bench player's. That is the problem, quantified.

**The handoff's lever was the wrong one, and the right one is measured.** `roles.design7` has no body -- but
the board runs `mode="full"` on both sides and `spm.offset` returns `np.zeros(2 * m)` on exactly that
condition, so the Simple SPM offset is identically zero in the shipped board. `design7` reaches the rating
only through the `rapm1` column `49_role_panel.py` writes into the panel. The live lever is `gbdt_features`,
and the finding is in `DECISIONS.md`: **height belongs nowhere, weight belongs on defense.** `board_D_weight`
(`weight15` added to the 11-feature defensive prior) is a criterion tie at z -0.32, passes ten of ten floors,
and improves the prior at <250 (z -1.39) and 250-500 (z -1.90) possessions and nowhere else. `board_D_height`
does the reverse -- helps the middle and top, makes the deepest bench worse, and fails the defensive floor at
0.7485.

**Ruling 9, 2026-09-10:** *"Bench players need to be in the accuracy test."* Done -- `53_calmap.py` takes
`--splits=` now and `evaluate` / `unmapped_rows` score each held-out season inside groups. The pooled row is
bit-identical either way (`test_splits_reach_the_calmap_scorer`).

**Use `--splits=game_bench_share`, not `--splits=exposure`.** This cost a round trip and is trap 16 in `DECISIONS.md`.
`by_exposure` labels a STINT, so its groups cut team-games in half, and the criterion sums a team's points
over its rows in a game -- on a partial mask that is a partial point total against a level fitted on complete
games. Its groups recombine to 337.6 where the pooled score is 113.6. It produced two confident wrong numbers
before the recombination check caught it (a z of -2.16 for `board_D_weight` that is not there, and a story about
the calibration map taxing the bench that is backwards). `score` returns NaN for `tg` on any mask that cuts a
team-game now, so it cannot happen again; a stint-cutting split is read on `mse`.

`by_game_bench_share` (`game_bench_share`) bins each TEAM-GAME by the share of its possessions played by players the training
block saw fewer than 500 of. Constant within a team-game, recombines to the pooled score exactly.

**What it says.** The calibration map's exposure term is *not* taxing the bench -- against no map at all it is
worth -1.14 in the least bench-heavy games and **-4.47 (z -3.33)** in the 15-30% group, its largest gain by
far, where plain `linear` manages -0.87. And `board_D_weight` is not a candidate: under `game_bench_share` it is -0.022,
-0.003, +0.097 and +0.351 across the four groups, nothing significant, and the two bench-heavy groups mildly
favour the shipped board. It is -0.396 at z -2.73 on zero-exposure rows at STINT level, which is a real
measurement of a different unit; the criterion's unit is the team-game and it says no.

**Ruling 10, 2026-09-10, and it is the one that worked:** *"games started% and minutes played can 100%
help us here"* -> *"gs% * feature and poss played % x feature, for all available features, boruta test adding
in these interaction features"*. Built (`gbdt_prior.interaction_features`, `50_boruta.py --modes=interactions`) and
measured end to end. **`board_D_interactions` is a finished candidate that passes every gate**, and the case for it
is in `DECISIONS.md`:

| | shipped | `board_D_interactions` |
|---|---|---|
| criterion, mapped, K=3 q75 | -- | **-0.090 per 100, z -1.68, 19 of 28 seasons** |
| consensus total / offense / defense | 0.8354 / 0.8350 / 0.7565 | 0.8335 / 0.8354 / **0.7583** |
| floors | 10/10 | **10/10** |
| prior error, <250 poss per season | -- | **-0.138 (z -3.94, 9 of 10 windows)** |
| prior error, 250-500 | -- | **-0.095 (z -4.45, 9 of 10)** |

It is the shipped defensive list plus eight interaction features; **nothing else this session moved a single possession
bucket.** Boruta rejected `gs_pct` and `poss_pct` outright on both sides while accepting eight of their
interaction features on defense -- gs% and poss played % carry nothing as columns of their own.

**Run the ablation before you believe any of it, because it changes the story.** `board_D_past_only` (the raw
PAST block on defense, no interaction features) is **-0.133 at z -2.68 over 20 of 28** -- better on the criterion, and
the first thing since the single-season board to clear |z| = 2 -- but it **fails the defensive consensus
floor at 0.7468**. `board_D_interactions_nopast` (interaction features, no past) is +0.005 on the criterion: nothing. So the criterion
gain is the past block arriving on a defensive prior that had no `past_*` column at all, and the interaction features'
job is different: they cost 0.043 of that gain, buy back 0.0115 of consensus defensive agreement, and carry
the whole bench improvement (z -4.39 at 250-500 with no past block present). **Only the pair is shippable.**

**The owner's call:** ship `board_D_interactions_stats_possplayed` -- the shipped defensive list, the eight accepted
interaction features, the six original stats they are built from, and `poss_pct`. -0.147 per 100 at z -2.74
over 20 of 28 seasons, ten of ten floors, the defensive prior better in all five buckets and all ten panel
windows in the two lowest. `board_D_interactions` (interaction features with no original stats) was the earlier,
weaker version of the same idea at z -1.68. The criterion is -0.090 at z -1.68, which the standing tie rule
would not act on alone -- but every floor passes, defensive agreement goes UP, and the prior improves
significantly in all five buckets. Offense is rejected (`board_OD_interactions` drops offensive consensus to 0.8257).

**What is still open on item 2 after this.** `game_bench_share` says the criterion's team-game gain sits in the games
with the FEWEST barely-seen players (-0.105 at z -2.29 in the 0-5% group) and is slightly negative, not
significant, in the bench-heavy ones. So the bench improvement is real in the prior and still invisible to
the criterion's own unit. That gap is the thing ruling 9 was aimed at and it is not closed.

**Two things already measured, so do not redo them.** The bottom of the board is the calibration map's
exposure term, not a defect in the prior: players under 250 possessions take -2.85 from the map, and
re-fitting that map on the single-season kernel changed it by 0.08 (the `ks52` map gave -2.77 on the
same players). It is what the games ask for at 200 possessions. And the offensive prior's archetype
tilt (-0.469 against bigness, on every kernel and every map) is an offense/defense **attribution**
disagreement with the consensus, not a bias: the total gap is +0.068.

### 3. Finish re-picking the constants

Four are done and the old headline ("none of them transport") is already wrong:

| constant | verdict, 2026-09-10 |
|---|---|
| `lam_plugin` (the board's effective 5,726) | **unchanged.** Interior minimum, z +2.34 above and +1.16 below; ×0.71 tied at z −0.11 |
| `lam_ratio_plugin` (0.6245) | **unchanged.** Flat: ×0.5 and ×2 are +0.060 and +0.061, z +1.05 and +1.19 |
| `low_poss_threshold`, `starter_poss_threshold` | **1500/4500 → 500/1500.** Design possessions are one season now. Touches no rating (`lam_buckets` is empty; the 2026 fit is bit-identical) — it names the diagnostic groups |
| `boost_min_poss` (4500) | **dead.** No reader in `src/` or `scripts/`. Delete it with the booster |
| `gbdt_win_decay_def` (0.280024) | **unchanged, and it transports.** Seven points; nothing separates at \|z\| >= 2, but the sign is monotone -- every value below 0.280 is better, every value above it worse, and 1.0 ("pool every window alike") is the worst point on the grid. The low end is a plateau: 0.035 / 0.070 / 0.140 are within 0.006 per 100 of each other |

`board_lam<m>` and `board_lr<m>` in `systems.py` register those sweeps on the exact system that ships:
re-run with `53_calmap.py dump --tag=lamsweep` then `fit --tag=lamsweep --maps=linear+sat`. **Every
sweep must bracket the current value on both sides** — the old lambda grid was {0.125, 0.25, 0.5, 2.0}
with no 1.0 in it, so 0.5 had won a boundary it was never asked to beat.

Two decays are left and both are unblocked: `gbdt_win_decay` (0.514) and `PAST_DECAY`
(`gbdt_prior.py`, 0.5). Sweep them the way `board_defdecay_<value>` did -- and note what that sweep cost to get right:
the first grid's low end won, which is an argmax on a boundary and has chosen nothing, so it needed a second
pass beneath it. Bracket on both sides FIRST. Decay 0 is not a value this dial has:
`_pooled_by_distance` weights every other window by `0 ** |i - j|`, `training_rows` keeps only `other_w > 0`,
and the booster gets an empty frame.

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
    .venv/Scripts/python scripts/61_lowposs.py       # defensive skill 0.096 at <250, 0.327 at 4500+, ~25 s
    git log --oneline -1                             # 71969bf "A team-game score on a mask that cuts team-games..."

**First concrete step:** get the owner's ruling on item 2 above -- it is one question and it unblocks a
finished, measured candidate. Then sweep `gbdt_win_decay` (0.514) the way `board_defdecay_<value>` swept its defensive
twin, registering the grid beside it in `systems.py`:

    .venv/Scripts/python scripts/53_calmap.py dump --systems=<the sweep>_q75 --k=3 --workers=4 --tag=wdosweep
    .venv/Scripts/python scripts/53_calmap.py fit  --systems=<the sweep>_q75 --k=3 --maps=linear+sat --tag=wdosweep

The dump is about 20 seconds per system and the fit about 5. A board is ~8 minutes; only build one for a
candidate that already passed the criterion.

Reproducing the item-2 decision:

    .venv/Scripts/python scripts/61_lowposs.py --systems=ks00_lam05_ow_w0.25,board_D_height_weight,board_D_height,board_D_weight
    .venv/Scripts/python scripts/53_calmap.py dump --systems=ks00_lam05_ow_w0.25_q75,board_D_height_q75,board_D_weight_q75 --k=3 --workers=4 --tag=biosweep2
    .venv/Scripts/python scripts/53_calmap.py fit --systems=ks00_lam05_ow_w0.25_q75,board_D_height_q75,board_D_weight_q75 --k=3 --maps=linear+sat --tag=biosweep2
    .venv/Scripts/python scripts/60_season_board.py --system=board_D_weight --map=outputs/calmap_biosweep2.parquet --map-system=board_D_weight_q75_linear+sat --map-base=board_D_weight_q75 --out=season_ratings_bioDw
    OPENRAPM_BOARD=outputs/season_ratings_bioDw.parquet .venv/Scripts/python -m pytest tests/test_vs_consensus.py -q

Reproducing the panel decision, if it is questioned (`SP=sp_ks00_lam05_ow_w0.25`):

    .venv/Scripts/python scripts/53_calmap.py dump --systems=${SP}_q75,spy_ks00_lam05_ow_w0.25_q75 --k=3 --workers=4 --tag=spanel2
    .venv/Scripts/python scripts/53_calmap.py fit --systems=${SP}_q75,spy_ks00_lam05_ow_w0.25_q75 --k=3 --maps=linear+sat --tag=spanel2
    .venv/Scripts/python scripts/60_season_board.py --system=$SP --map=outputs/calmap_spanel2.parquet --map-system="${SP}_q75_linear+sat" --map-base=${SP}_q75 --out=season_ratings_spanel
    OPENRAPM_BOARD=outputs/season_ratings_spanel.parquet .venv/Scripts/python -m pytest tests/test_vs_consensus.py -q

The dump is ~6 minutes and the board ~8 (the season-panel prior is heavier than the block one).
