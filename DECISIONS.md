# OpenRAPM: what the out-of-season test decided, and what it could not

> **Where `FINDINGS N.N` points.**  Sixty-one comments in `src/` and a handful here cite section
> numbers in `FINDINGS.md`, the 5,292-line research log that was removed from the tree in the
> open-source cleanup (commit `0f7c0b2`).  It is preserved whole under the annotated tag
> `archive/research-2026-09`; read a section with
>
>     git show archive/research-2026-09:FINDINGS.md
>
> Nothing in the tree named that tag until 2026-09-17, so every one of those citations looked
> like a reference to a deleted file.  They are not: the sections are all still there.

## What the test is

Hold out one NBA season H. Fit the whole system on a symmetric neighbourhood of seasons around it — K = 2 is
{H-1, H+1}, K = 3 is {H-2, H-1, H+1}, K = 4 is {H-2..H+2} minus H — so the held-out season is identical across
every method and every K, comparisons are exactly paired, and aging cancels. Then predict every stint of H
from the ten players on the floor, refitting only an intercept and a home term on H (level "home", never
"full": context that is a consequence of the score is not an input). Score possession-weighted squared error
in points per 100, at stint level and again after summing each team's points within a game. The owner ruled on
2026-09-05 that **game level is the whole test** — ratings remove about 10% of team-game error and 0.7% of
stint error, and stint error is mostly binomial noise — but the stint column is always reported, because the
two have disagreed. Baseline with no player ratings: 125.6 per 100 at game level. This is the only criterion
the project has that is both external to the model and legal to select on: it scores actual points in a season
the model never saw, and uses no outside data. `src/eracoef/holdout.py`, `scripts/45_holdout.py`, stop rules
in `scripts/48_ladder.py`, the mapped and timed version in `scripts/54_track.py`. A second instrument,
`investigate.attributable` (`scripts/57_investigate.py`), reads the same residual for what a player ridge can
still put on named players: the criterion decides forecasting, the investigator decides attribution, and a
candidate should lose neither.

## What was settled, and how

**Three-season windows, not five.** Ten three-season windows against six five-season ones over the same thirty
seasons, game-grouped folds inside each. Held-out weighted MSE 3624.5 against 3624.9, the box prior worth 30.1
bp against 24.5, rating stability between adjacent windows 0.714 against 0.676. Five years buys 15% tighter
coefficient standard errors, which is mechanical, and blends too much career change into one player effect.
Later superseded in season by the rolling kernel; the owner's standing note is that chunks are on the way out,
so do not tune K.

**No defensive box prior (the hybrid).** The box term is an offset `Xbox @ beta` with separate offensive and
defensive columns, so zeroing beta's defensive half IS "no defensive prior" — one fit, one penalty. Consensus
agreement went 0.784 to 0.896 total, 0.755 to 0.888 on defense, defensive spread 2.13 to 1.23, archetype bias
+0.63 to +0.20. The mechanism: weighted R-squared of a player's 13 box rates on his own on-court RAPM is 0.53
on offense and 0.26 on defense, and the stats the defensive prior was built from are the most conserved into
the lineup sum (defensive rebounds 0.59, blocks 0.69, against steals 0.90 and missed twos 0.88). **A
lineup-level coefficient transfers to individuals exactly to the extent the stat is not conserved.**

**The free-throw luck adjustment (shipped).** Each free throw is priced at the shooter's padded leave-one-out
FT%. Worth **+0.26 per 100 against raw points, mapped** (unmapped it read 0.459; see the map-absorption trap).
It is the only luck adjustment that has ever paid here, and the reason is the padding constant: FT% pads at k
= 24 attempts, so a shooter's own rate is trusted almost entirely on 30 attempts, the outcome belongs to one
identified player, and the defence is absent.

**The opponent-three-point adjustment, `x3def` (shipped).** For the DEFENSIVE fit only, every opponent
three-point make becomes 3 x the shooter's 3P% from the OTHER half of the block, padded toward the league with
k = 450 attempts. Team opponent 3P% stabilises only at 4,000-7,000 attempts against about 2,000 in a season.
Worth -0.39 / -0.47 per 100 at z -4.4 / -5.1; removing it altogether costs +0.248 at z 3.25 mapped, and
handing the defence back up to a quarter of its realised threes is free (-0.007, z -0.34). Never
leave-one-game-out — a leave-one-out mean is negatively correlated with the left-out outcome by 1/(N-1). It
cost consensus defensive agreement 0.888 -> 0.814, and the component split explains that: every raw-points
metric in the blend agrees less, every luck-adjusted one more.

**Shot quality in the prior.** Six features from the per-shooter shot tables: difficulty (`xl / a`, the
league's make probability from his spots) and shot-making (`(m - xl) / a`) on twos, threes and in points, each
padded in attempts toward the BLOCK's own league level, never a constant (the league two-point rate runs
0.4648 in 2000-02 to 0.5468 in 2024-26). Worth -0.045 on the accuracy line (z -1.13) and nothing in shipping
shape — it shipped for the floors, moving the offensive bigness gap -0.303 to -0.270 (which unlocked the 0.7
offensive blend) and the defensive spread to 1.30, the narrowest any candidate had measured. Later pruned by
Boruta at no cost.

**The calibration map and its exposure term** (`src/eracoef/calmap.py`, `scripts/53_calmap.py`). A per-side
map fitted leave-one-season-out on the pooled TEAM-GAME residuals. A map of the RATING ALONE is the identity
at game level — the scalar the games want is 1.03 / 0.98 / 0.96 on offense at K = 2/3/4 and every shape family
lands within 0.03. **The exposure term is what pays**: `sat(poss) = poss / (poss + 1000)`, worth **-0.79 to
-1.01 per 100 (z -5.7 to -7.1, 25 of 28 seasons)**. At K = 3 the fit is offense `0.85 x + 2.88 sat`, defense
`0.885 x - 3.15 sat`: a rotation player is about 5 points per 100 above one the block never saw, and the term
is flat among regulars, so it reorders the low-minute end and leaves the top alone. The shipped family is
`linear+log2&xlog` plus a prior re-weighting term on OFFENSE only (on defense it takes consensus agreement to
0.751). Prediction-time only, so unable to ship: a quadratic in the player's age at H (-0.14) and a cubic bend
on the team-game total and on the stint rows (-0.07 and -0.21). The earlier stint-fitted rank map (-0.41, z
-6.2) is superseded and off.

**The two ridge constants were re-picked at season scale and did not move** (2026-09-10, Phase 1 item 3).
The handoff's blanket claim was that none of the three-season constants transport. For the two that decide how
much of a season's on-court evidence reaches a rating, that is false. Sweeping the SHIPPED board's own lambda
(5,726 = `lam_plugin` x tune501's 0.624 x 0.5), mapped, K = 3, q75, paired against it:

| x lambda | 0.25 | 0.5 | 0.71 | **1.0** | 1.41 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|---|
| per 100 | +0.387 | +0.060 | -0.003 | **0** | +0.064 | +0.189 | +0.647 |
| z | +3.85 | +1.16 | -0.11 | | +2.34 | +3.28 | +4.97 |

A clean interior minimum, bracketed and significant on both sides, with 0.71 tied (z -0.11). The old grid
{0.125, 0.25, 0.5, 2.0} had **no 1.0 in it**, which is why this needed re-running rather than re-reading:
0.5 had won a boundary it was never asked to beat. `lam_ratio` (0.6245) is flatter still -- x0.5 and x2 are
+0.060 and +0.061 at z +1.05 and +1.19, nothing separable within a factor of two either way. Neither
constant changes.

**`low_poss_threshold` and `starter_poss_threshold` did move, and touch no rating.** They are per-player
possessions in the DESIGN, which is now one season, so 1500 / 4500 named 160 of 582 players "low" and left
270 "high". They are 500 / 1500 now, which keeps the meaning they had per season. The 2026 fit is
bit-identical either way because `lam_buckets` is empty -- they name the diagnostic groups, and the bucket
penalties if one is ever used. `boost_min_poss` has no reader anywhere in `src/` or `scripts/`: it is dead
with the booster and is marked as such rather than silently kept.

**The board is one season, shipped 2026-09-10.** `ratings_prior.season_board.system` is
`ks00_lam05_ow_w0.25` with the `linear+sat` map fitted on its own dump
(`artifacts/calmap_insea_ks00_q75.parquet`), and `artifacts/season_ratings.parquet` and the site are rebuilt
from it: 14,578 rows over 1997-2026. Owner's ruling 1. Against the three-season board it replaced: **+0.557
per 100 on the criterion (z +2.57, 10 of 28 seasons)** and **0.835 against 0.809 on consensus agreement**,
with all ten floors passing and the archetype spread at 0.145 against 0.213. The criterion is the tiebreak
between candidates; which product to build is not a tiebreak, and the ruling names it.

**A zero-weight season was still a training season.** `KernelSystem.train_for` and the board both built the
list as "every offset the kernel names", so the single-season kernel trained on `[a-2, a-1, a]` with the first
two weighted zero. `kernel_game_mult` zeroes those games, so the design, the padded box rates and
`Ratings.poss` were all correct -- but the inputs built from SEASON TABLES rather than from the design (the
shot-quality features from `xshoot`, the role inputs) are built over whatever the list names, and they pooled
three seasons for a rating the kernel said was one. It moved a 2026 offensive rating by up to **0.61 per 100**
(mean 0.056) and left the defensive prior and the possessions identical, which is the signature: only the
offensive features come from those tables. `inseason.kernel_seasons` is now the one place the list is built,
and `tests/test_inseason.py::test_a_zero_weight_season_is_not_a_training_season` pins it. Fixing it improved
the criterion (112.410 -> 112.362 mapped) and cost 0.004 of consensus agreement. **The check: a weight of
zero in a model has to be checked against the FEATURE path as well as the design path.**

**What ruling 1 costs on the criterion, and what the map needs afterwards** (2026-09-10, post-cut estimand).
One rating per player per season from that season's games only is the owner's ruling, not a candidate, but the
price is now measured. At K = 3, cut q75, mapped on each kernel's own leave-one-season-out map, the
single-season kernel `ks00_lam05_ow_w0.25` reads **112.41 against the three-season kernel's 111.80 -- +0.606
per 100, z +2.91, winning 9 of 28 seasons**. Unmapped it is 113.76 against 113.24. That is the cost of the
ruling and it is not recovered by the map.

The map itself does transport. Refitted on the single-season dump the exposure term barely moves: at the
bottom of the 2026 board (51 players under 250 possessions, shared with the shipped board) it takes -2.85,
where the shipped `ks52`-fitted map took -2.77 on the same players. **Refitting the map on the right kernel
removes none of the -2.8 the bottom of the board gets** -- it is what the criterion asks for, not a stale
constant, and the handoff's item 4 premise is wrong. Their prior is -1.80 and their ratings have an sd of 0.70:
the bottom is compressed and uniformly negative because the games say a player with 200 possessions is worth
about that much less than the ones with 4,000, not because the map was fitted at the wrong possession scale.

On the moved estimand the shipped map FAMILY is also no longer earning its parameters: `linear+sat` (2 per
side) ties `linear+log2&xlog&prior&tshare : linear+log2&xlog` (6 and 4) at -0.007 per 100, z -0.12 on the
single-season kernel and -0.002, z -0.08 on the three-season one. The standing rule takes the simpler and
faster. The four-term family was chosen before the playoff fold moved what the criterion scores.

**The archetype guard is now unsupervised, and calibrated in points** (2026-09-10, the owner's call:
*"the guardrail is arbitrary ... would rather use a bayesian gaussian mixture (legit unsupervised clusters
rather than center/big/guard)"*). `src/eracoef/archetype.py` fits a Bayesian Gaussian mixture on the per-36 box
profile and reports the board-minus-consensus gap per cluster; the statistic is the sd, across clusters of at
least ten players, of the mean TOTAL gap. `tests/test_vs_consensus.py::test_no_archetype_bias_by_cluster`
replaces `test_offense_has_no_big_man_bias`, with the floor at 0.30.

Three things had to hold before it could replace a test, and were measured (`58_archetype.py stability`):

  * **stable under the seed.** Ten seeds at k = 8: the three-season board reads 0.182-0.246 and the
    single-season board 0.107-0.174, sd 0.02 each and **the ranges do not overlap**. The test averages
    seeds 0-2 and costs 2.5 s;
  * **stable under k.** Over k in {6, 8, 10, 12}: 0.190-0.229 and 0.081-0.137. Same ordering. Per single
    season it is noisier (0.206-0.320 against 0.175-0.267) and the ranges do overlap, so the statistic is
    pooled over the three seasons the consensus covers, not read per season;
  * **calibrated.** Adding a flat bump to every player in the top bigness tercile moves it by **0.16 per
    point per 100**, the same slope on both boards. So the floor converts: 0.30 permits about 0.6 per 100 of
    archetype distortion on today's board and about 0.95 on the single-season one. When the board becomes
    the single-season one, re-base to 0.25 -- still four seed-sds of headroom, and half the tolerance.

`tests/test_archetype.py` pins the instrument itself against a synthetic world of three known types: the
mixture recovers them at 95%+ purity from k = 6, an unbiased board reads under 0.10, a one-point type-shaped
bonus takes it over the shipped floor, and a fabricated four-player cluster with an absurd gap cannot move it.

**The big-man floor is measuring an offense/defense ATTRIBUTION disagreement, not a bias** (2026-09-10,
asked for by the owner after the single-season board failed it at -0.437). Decomposed by stage, on the same
475 players:

| | offense gap vs bigness | defense gap | TOTAL gap |
|---|---|---|---|
| shipped `ks52` board | -0.326 | +0.279 | +0.169 |
| single-season board | **-0.437** | +0.166 | **+0.068** |

The two boards agree with the consensus about how good bigs ARE; they disagree about which side of the ball it
comes from, and the single-season board's total is the closer of the two. Within the offense, the source is the
**box-and-role prior and nothing else**: its gap-vs-bigness is **-0.469 on every kernel and every map tried**.
Two things then move it, and both were measured by building the same board three ways
(`season_ratings_ks00_shipmap.parquet` is the middle row):

  * the on-court residual `u` pulls back toward bigs (+0.12 with bigness on its own), and a single-season fit
    gives it a third of the evidence, so less of the pull survives: -0.390 becomes -0.425;
  * the shipped map's `prior` re-weighting term shrinks the prior and so incidentally corrects another 0.064
    (-0.390 -> -0.326). `linear+sat` has no such term, so the new board keeps the whole tilt (-0.426 with the
    shipped map, -0.437 with its own).

In points the disagreement is small: per +1 sd of bigness the consensus offense falls 0.291 per 100, the board
0.420 and the prior 0.554. It reads as a 0.44 correlation because both sides are standardised and the board's
offensive spread is 1.19 against the consensus's 1.96.

**Unsupervised archetypes say the new board is the less biased one** (`scripts/58_archetype.py`, a Bayesian
Gaussian mixture on the per-36 box profile -- the owner's suggestion, and no hand-picked axis). Per-cluster
mean total gap, sd across clusters: **0.229 on the shipped board against 0.133 on the single-season one**. The
shipped board's worst clusters are the rim-running centres at +0.21 / +0.27 (Gobert, Allen, Mobley; Okongwu,
Looney) and the offensive engines at -0.38 (Jokic, Gilgeous-Alexander, Harden); on the single-season board
those are +0.11 and -0.10. The centre cluster is where offense and defense are large and opposite (-0.42 /
+0.39), which is the attribution story again. So `test_offense_has_no_big_man_bias` and the cluster spread
disagree about which board is worse, and the cluster spread is the one without five hand-chosen weights in it.

**The two instruments disagree about ruling 1, and the disagreement is stable across the board.** The
single-season board (`ks00_lam05_ow_w0.25` + its own `linear+sat` map) is **worse on the criterion by 0.606 per
100** and **better against the external consensus everywhere**: pooled over 2024-2026 on the 475 players both
boards match, rank agreement is 0.839 total / 0.848 offense / 0.757 defense against the shipped board's 0.809 /
0.824 / 0.755, and it is ahead in every possession bucket including the smallest (players with under 5,000 of
their own possessions over the three seasons: 0.675 against 0.643). Defensive spread 1.34 against 1.38. The
consensus is a sanity check and never a fitting target, so this does not overturn the criterion -- but ruling 1
is the owner's and the sanity check does not object to it.

Nine of the ten consensus floors pass on it. The tenth, `test_offense_has_no_big_man_bias`, reads **-0.437
against a floor of 0.35** (the shipped board is -0.344). The single-season board pushes further along the axis
that floor has already been re-based twice for: relative to the consensus its offense lifts guards over bigs.
Its own comment says twice re-based is a pattern worth the owner's eye, so it was left failing rather than
re-based a third time.

**The board keeps the BLOCK panel, and the reason is defensive attenuation** (2026-09-10, the measurement
the criterion could not make). A board built on `sp_ks00_lam05_ow_w0.25` -- same kernel, same map family,
only the prior's training substrate swapped to one row per player-season -- against the shipped one:

| | block panel (ships) | season panel |
|---|---|---|
| criterion | 112.36 | 112.42 (+0.063, z +0.70, a tie) |
| consensus total / offense | 0.835 / 0.835 | **0.846** / **0.865** |
| consensus defense | **0.756** | **0.739 -- below the 0.75 floor** |
| rating defensive sd | 1.33 | 1.22 |
| PRIOR defensive sd | **0.682** | **0.476** |
| archetype spread | 0.145 | **0.108** |

Nine floors pass on it and `test_defense_agrees_with_the_consensus` fails. The old claim that a
season-granularity prior "loses half its defensive spread" is **real and it is not the missing `onc_*`
columns** -- this panel has them. The defensive prior is 30% narrower (0.476 against 0.682) and the finished
defensive rating follows it down. Offense gains from the finer substrate (0.835 -> 0.865, and the archetype
spread improves) which is what makes this a genuine trade rather than a worse panel.

So: the panel stays block-granular until the defensive attenuation has a fix, and **the decay constants are
therefore unblocked** -- they can be re-picked on the block panel, which is the substrate that ships. The
obvious candidate for the fix is that `win_decay` is per WINDOW and a window is one season on this panel:
`spy` cube-roots it to the same decay per year and is worth +0.002 (z +0.06) on the criterion, so it is not
the answer by itself.

**The season-granularity role panel is criterion-neutral** (2026-09-10, on the rebuilt panel that finally
carries the `onc_*` columns, re-measured after zero-weight seasons left the training list). Same kernel, same
map, panel swapped: `sp_ks00` against `ks00` is **+0.063 per 100, z +0.70, 14 of 28 seasons** -- a tie.
Cube-rooting `win_decay` to the same decay per year (`spy`) is +0.002, z +0.06: nothing. On the three-season kernel the season panel is +0.106, z +1.32, also not separable.
So the panel granularity is free to follow the product rather than the criterion, and the earlier "a season
panel loses half its defensive spread" is a question about the prior's spread, not about prediction.

**The boosted GBDT prior beats the linear one.** `mspi` scored 112.06 at K = 4 against 112.33 for the
linear-prior board, 20 of 28 seasons. Its prior is a third narrower and the on-court residual carries half
again as much of the rating (1997-99 offense: prior sd 1.15 against a residual of 0.67, where the linear prior
was 1.69 against 0.40 and correlated 0.98 with the final rating). That is the Stockton problem answered:
Jordan first by 0.5 and Stockton seventh, where the linear board had Stockton 5.18 over Jordan 5.27. The era
term is in the model and does almost nothing — moving an assist from its 10th to its 90th percentile is worth
0.14 per 100 in 1998 and 0.16 in 2025.

**The role prior chain: APM -> Simple SPM -> RAPM_1 -> GBDT** (`scripts/49_role_panel.py`). APM is the plug-in
ridge with beta 0 at penalty 100 (under 1% of the shipped 18,352); the Simple SPM is a possession-weighted
ridge of APM on possession share, starts share and age with squares, fitted leave-window-out (role prior sd
1.4-1.9 per 100 on offense, 0.7-0.9 on defense); RAPM_1 is the shipped ridge with the SPM as its offset; the
GBDT is trained on that. **What the prior is trained ON is the only thing about it that has ever mattered**:
RAPM_1 -> unshrunk APM is -0.29 at z -3.1, while capacity, twenty-three engineered features and a
distance-weighted target all move the criterion by less than 0.07. Pure APM fails the floors (defense 0.678,
spread 1.56), so the shipped offensive target is **0.7 APM + 0.3 RAPM_1**, with RAPM_1 on defense.

**The estimator search.** `scripts/55_tune.py`, 625 TPE trials in 97 minutes, scored on 14 alternating
held-out seasons with the other 14 never shown to the optimizer. `tune501_b7` ships: 110.624 against 110.707,
**z -3.02 over 18 of 28, and 37 s for the 28 fits against 46**. Four independent candidates converged on the
same structure — linear leaves on both sides, cross features on offense only, no defensive bag, 64-128 bins,
subsample 0.65-0.83 — an independent rediscovery of what a hand decomposition had found. The best search trial
(491, -0.194) regressed to -0.060 on the confirm half while the eventual winner ranked 8th on search: the
region is real, the ranking inside it is noise.

**The turnover-aware prior, evaluated at a settled context.** Teammate turnover is measured for everyone
(`src/eracoef/turnover.py`; movers sit at a season-to-season median of 1.0, stayers at 0.36). The prior is
trained on ordered window pairs with the target window's turnover as a feature and then evaluated at a settled
0.35 for everyone, fixed a priori. **-0.054 at z -3.39 over 22 of 28, offense only.** Why: the pooled target
is a player's value over windows three years away, where turnover averages 0.70, so the context-change penalty
is baked in; the held-out season sits inside its own block at 0.40.

**Binned height and weight.** Zero on the criterion (-0.006, z -0.32) and shipped on the owner's call, because
the prior's own fit puts height at -0.14 on defense and the archetype table shows what it corrects: **the
defensive prior under-rates 6-10-and-taller players by about a third of a point per 100 in nine windows of
ten, with no era trend at all.** The criterion's resolution for a correction shared by an archetype is about
0.015, so it cannot adjudicate this. Binned, never fine (trap five).

**His own past on-court record in the prior.** `past_apm` (his possession-weighted APM over the panel windows
BEFORE this one, discounted 0.5 per window), `past_poss`, `past_rapm`, on the offensive list, on PAIR rows
because a pooled row's target contains the past windows (`training_rows` raises on any PAST name). **-0.085 at
z -2.46 on the criterion and -0.204 at z -4.32 on the investigator: the first candidate ever significant on
both instruments.** The gain is in the body of the distribution, not the top — the top offensive decile is
still under-rated by +0.64 against +0.70 before.

**The Boruta prune.** 111 candidates on pair rows, 50 trials, 5.7 hours (`scripts/50_boruta.py
--modes=sink,sinknoagg`). Boruta's own accepted lists LOSE badly (+0.199 on the criterion at z 4.8 and +0.574
on the investigator at z 9.1 for the offensive one). But removing its REJECTS from the shipped lists costs
nothing on either instrument: 48 offensive names to 31 and 25 defensive to 11, at -0.024 (z -0.54) and -0.044
(z -0.89), 58 s for the 28 fits against 65. Shipped on the parsimony tie-break. The defensive prior is back to
eleven box columns, the season and two role inputs.

**The in-season kernel.** A rating is ANCHORED at a season and fit on that season and the two before it,
`kernel = {0: 1, -1: 0.5, -2: 0.25}`, nothing after the anchor in it, at half the block board's ridge. A CUT
lets the fit see only the first q of the anchor season and scores it on the rest. Chosen on the search half,
replicated on the confirm half: **-0.225 per 100 (z -3.0) at a quarter of a season and -0.205 (z -2.4) at
half**, attribution -0.363 (z -2.9). The optimum is a plateau with three neighbours inside 0.11, so the
simpler member wins. The chunk board is not close in season — +2.7 to +6.2 on prediction and +4.2 to +7.3 on
attribution — and most of that is coverage, 0.716 against 0.988. Against a stratified permutation null the
season board has captured **84% of the player credit a lineup model can see**, the flat rolling window 80%,
the chunk board 45%.

**The season board ships beside the block board, which is untouched.** `scripts/60_season_board.py` fits the
kernel once per anchor season 1997-2026 (66 s for all thirty) and `scripts/52_site.py` writes both with a
Block/Season switch. It is the same board, slightly narrower: Spearman 0.969 on 580 shared players, sd 2.52
against 2.93.

**Two knobs measured and left alone.** The offensive and defensive penalties were already separate and already
optimal (the best cell of a 4x5 sweep is 110.047 against 110.049 for the single multiplier), and on a single
season the ridge does NOT collapse onto the box score: every target's optimum is interior at x0.5, where a
rating is 20% on-court evidence on offense and 44% on defense. Between x0.25 and x1 both instruments are flat
while the offensive evidence share moves 10% to 33% — a product decision with a measurement attached, and the
owner's to make.

**A season is its regular season and its playoffs, one entity.** Owner, 2026-09-10: the playoff delta is
gone and "playoff games just join the fit like any other games", then "RS + playoffs should be considered a
single entity in our new version". `phases=("RS",)` had been the default in six places -- the training design,
the design cache, the two holdout entry points, the box exposure's padded rates, and `MspiFast` itself -- and
each was somewhere the playoffs were silently dropped; so were the role inputs (minutes, starts, possessions)
and the shooter totals that price the luck-adjusted targets. There is now one constant, `design.SEASON_PHASES`,
and `tests/test_playoffs_in_fit.py` fails if a second one appears.

**This moved the criterion's estimand, and the move exposed a bug that had hidden the playoffs on both sides
at once.** A cut q trains on the anchor season's games with `season_frac < q` and scores the ones with
`frac >= q`. `season_frac` gave a playoff game **-1**, which is below every cut -- so a playoff game was
excluded from training by a special case in `kernel_game_mult` AND excluded from scoring by the same
comparison. It was in neither half, and the first attempt to move the estimand came back identical to four
decimal places, which is how it was found. A playoff game now scores 1.0: it comes after every regular-season
game, so a cut trains on none of them and scores all of them, and the special case is gone.

The criterion is therefore higher than every number recorded above this entry: the shipped board is 113.11
per 100 at team-game level over 28 held-out seasons at q75, against **109.36 on regular-season rows alone**.
Playoff games are harder to predict. Numbers measured before 2026-09-10 are on the old estimand and do not
compare.

On the estimand that includes them, the fold is worth something real. Regular-season-only training against
the same board with nothing but `phases` differing, K = 3, q75, 28 seasons: **+0.057 per 100 at team-game
level (z 2.28) and +0.128 at stint level (z 3.75)**. On the old RS-only estimand the same comparison read
+0.022 (z 0.83) -- not separable. You predict playoff games better if you trained on playoff games, and the
yardstick that could not see that was the one that scored none of them. Against the consensus, pooled over
2024-2026 and 485 matched players, the board moves 0.810 -> 0.809 total, 0.822 -> **0.824** offense, 0.758 ->
0.755 defense, defensive spread 1.39 -> 1.38.

The design keeps its `is_po` and `po_home` fixed columns, and that is not a contradiction. They are level
controls on the environment, the same kind of thing as the per-season intercepts `int_<s>`, which nobody
would call treating each season as a separate entity. Dropping them would make the playoffs' scoring level
something the fit has to explain with the players on the floor, and the players on the floor in the playoffs
are disproportionately the good ones. A level control is what keeps the two phases one entity rather than an
advantage for whoever got there.

One thing the handoff expected to be broken was correct by construction: `roles.cut_role_inputs` reads
regular-season stints because `inseason.keep_games` names regular-season ids only, and a cut excludes the
anchor's playoffs anyway.

**The single-season board is worse, and much worse early in the season.** The owner's item 5 gate, measured
with only the kernel differing, paired within cut over 28 held-out seasons on the whole-season estimand:
`ks00` (one season) against the shipped `ks52` ({1, 0.5, 0.25}) is **+0.50 per 100 at team-game level at q75
(z 2.43, 11 of 28 seasons) and +2.12 at q25 (z 11.2, 1 of 28)**, and coverage falls 0.987 -> 0.972 and
0.955 -> 0.886. At q0 the single-season kernel is degenerate: no games at all, `covered` 0.00, so it has
nothing to say about a season in progress until it is played. No calibration map is involved in any of these
-- `45_holdout.py` applies none unless asked -- so this is the kernel, not the map. The caveat that keeps it
from being the final answer: `ks00` reads the block role panel, so its prior is attenuated at season
granularity; `sp_`/`spy_` twins on the season panel are registered and unmeasured.

**`gbdt_win_decay_def` (0.280024) is unchanged, and it does transport.** Phase 1 item 3, the first constant
re-picked with the panel settled as block-granular. The dial weights another window of the SAME player by
`decay ** |i - j|` when the defensive prior pools his training target; it was tuned by `tune501` against a
three-season product. `board_defdecay_<value>` in `systems.py` is the shipped board with only that number replaced.
Seven points, K=3, q75, mapped `linear+sat`, paired against the shipped board's OWN mapped rows over 28
held-out seasons (`cut` populated on every row, so no fit is scored on its training games):

| decay | 0.035 | 0.070 | **0.280** | 0.198 | 0.396 | 0.560 | 1.000 |
|---|---|---|---|---|---|---|---|
| per 100 vs 0.280 | -0.029 | -0.026 | -- | -0.011 | -0.008 | +0.014 | +0.045 |
| z | -1.39 | -1.79 | -- | -0.72 | -0.40 | +0.82 | +1.25 |

(0.140, the first grid's low end, is -0.032 at z -1.76 over 16 of 28 seasons.) **Nothing separates at
|z| >= 2**, so the standing tie rule keeps the incumbent. What the sweep does buy is the shape: the sign is
monotone across the grid -- every value below 0.280 is better and every value above it is worse -- so the
direction is real even though no point is significant, and "pool every other window alike" (1.0), the value
the dial had before `tune501`, is the single worst point on the grid. The low end is a plateau, not a peak:
0.035, 0.070 and 0.140 are within 0.006 per 100 of each other, which is why moving to any of them would be
choosing noise. Built out to a full board, 0.140 changes nothing that is measurable elsewhere either --
consensus total 0.8354 -> 0.8358, offense 0.8350 -> 0.8351, defense 0.7565 -> **0.7582**, defensive spread
1.3314 -> 1.3305, ten of ten floors pass on both. The 0.280 rebuild through the same map path reproduces the
shipped board to four decimals on every one of those, which is the check that the comparison was fair.

Decay 0 is not available and would not mean what it looks like: `_pooled_by_distance` weights every other
window by `0 ** |i - j|`, `training_rows` keeps only `other_w > 0`, and the booster is handed an empty frame
(`ValueError: X has 0 sample(s)`). 0.035 is the floor this dial has.

**The prior's body lever is the feature list, not `design7`.** The handoff pointed Phase 1 item 2 at
`roles.design7` -- the Simple SPM's seven role-and-age inputs, with no box score and no body. That is true of
`design7` and beside the point for the board: `ks00_lam05_ow_w0.25` runs `mode="full"` with `sides=("O","D")`,
and `spm.offset` returns `np.zeros(2 * m)` on exactly that condition. The SPM offset is identically zero in
the shipped board and `design7` reaches the rating only indirectly, through the `rapm1` column
`49_role_panel.py` writes into the panel the GBDT trains on. The live lever is `gbdt_features`: the offensive
prior carries `weight15` and **no height at all**, and the defensive prior carries **no body of any kind** --
11 features, box counters plus `season`, `gs_pct`, `age` -- on the side where one season is only 44% evidence.
`board_{O,D,OD}_height_weight` adds the binned pair from `gbdt_prior.BIO_BINS`.

**And the criterion cannot see the question, so `scripts/61_lowposs.py` was built to.** The out-of-season
criterion is scored at TEAM-GAME level, where a 200-possession player is a rounding error; it will call any
change to the bench a tie forever. The new script scores the prior itself, leave-one-panel-window-out: refit
with window w excluded, predict w's own rows, and compare to the row's training target -- the player's value
pooled over his other windows, or the exact other-window value of the pair when the feature list carries a
PAST block. Buckets are POSSESSIONS PER SEASON (the window's possessions over its length), so "under
500" means what `low_poss_threshold` now means. On the shipped board, skill (1 - mse/null_mse, the null being
the panel mean for everyone) by bucket:

| poss per season | <250 | 250-500 | 500-1500 | 1500-4500 | 4500+ |
|---|---|---|---|---|---|
| offense | 0.193 | 0.173 | 0.179 | 0.255 | **0.480** |
| defense | 0.096 | 0.160 | 0.171 | 0.265 | **0.327** |

That is the shape of the problem in one table: the prior a starter gets is two to five times as good as the
prior a bench player gets, on both sides.

**Height and weight in the prior: weight belongs on defense, height does not, and neither belongs on
offense.** Five variants, all mapped `linear+sat`, K=3, q75, `cut` populated. The criterion is a tie for every
one of them (`board_O_height_weight` -0.020 at z -1.00, `bioD` -0.003 at z -0.14, `bioDh` -0.012 at z -0.50, `bioDw`
-0.007 at z -0.32), so it arbitrates nothing and the other two measurements do:

| defensive prior | criterion z | prior d_mse, <250 | 250-500 | consensus def | floors |
|---|---|---|---|---|---|
| shipped (no body) | -- | -- | -- | 0.7565 | 10/10 |
| + height2 + weight15 | -0.14 | +0.015 (z +0.57) | **-0.072 (z -2.72)** | 0.7489 | **9/10** |
| + height2 only | -0.50 | +0.015 (z +0.57) | **-0.072 (z -2.62)** | 0.7485 | **9/10** |
| + weight15 only | -0.32 | -0.028 (z -1.39) | -0.028 (z -1.90) | 0.7519 | 10/10 |

**Height is what breaks the floor.** It also helps the wrong players: it improves the prior in the middle and
at the top (1500-4500 z -2.37, 4500+ z -1.87) and makes the deepest bench slightly WORSE. Weight does the
opposite -- it helps only the two lowest buckets, which is the population ruling 1's second sentence names,
and is neutral everywhere else. Consensus defensive agreement falls under all three (0.7565 -> 0.7519 for
weight alone) and the defensive spread widens (1.331 -> 1.341); FINDINGS 21.26 predicted exactly this, that
the consensus floors do not tolerate a richer defensive feature list, and it is again the offense/defense
attribution disagreement rather than a total one (total 0.8354 -> 0.8334).

On offense, adding `height2` beside the `weight15` already there is worse at the bottom (<250 z +1.27) and
worse in the middle (1500-4500 z +1.73) and better nowhere. Rejected.

**The owner's ruling, 2026-09-10: put the bench players in the accuracy test.** Asked whether a measurement
the criterion cannot see may move the board, the answer was neither yes nor no: *"Bench players need to be in
the accuracy test."* Not a side diagnostic with its own estimand -- the criterion itself, split.

The machinery already existed and had never reached the script that picks a board. `holdout.SPLITS` scores a
held-out season again inside groups of rows, and `45_holdout.py` has always used it; `53_calmap.py` wrote
`split="all", group="all"` and nothing else. `evaluate` and `unmapped_rows` now take `splits` and `ctx`, and
the script takes `--splits=exposure,bench`. The pooled row is bit-identical either way
(`test_splits_reach_the_calmap_scorer` asserts that, and the re-scored `biosweep2` reproduced -0.011979 and
-0.006555 exactly), so no number ever read off this script moved.

**Then the split itself turned out to be the trap.** The first reading of it -- that weight on defense is
-0.202 per 100 at z -2.16 on the zero-exposure rows, and that the calibration map "buys its pooled gain by
taxing the bench" -- was **wrong, and both halves of it were wrong for the same reason.** `by_exposure`
labels a STINT. Its groups therefore cut a team-game in half, and `tg` -- the criterion -- sums a team's
points over its rows in a game. On a partial mask that sum is a partial point total scored against a level
fitted on complete games. It is not the criterion restricted to those rows; it is not the criterion.
Measured: the `exposure` groups recombine to **337.6** where the pooled score is **113.6**, a factor of three.

`by_game_bench_share` (`--splits=game_bench_share`) is the sound one. It bins each TEAM-GAME by the share of its possessions
played by players the training block saw fewer than 500 of, so it is constant within a team-game, the groups
partition the criterion's own unit, and they recombine to the pooled score **exactly** (112.362 either way).
`score` now returns NaN for `tg` on any mask that cuts a team-game (`_keeps_whole_team_games`,
`test_team_game_score_is_nan_when_the_mask_cuts_a_team_game`), so this cannot be read wrong again. A
stint-cutting split must be read on `mse`, at stint level, where a subset is perfectly well defined.

**What the sound split says about the map: the opposite.** Each map against no map at all, by how bench-heavy
the team-game is:

| share of the team-game played by barely-seen players | 0-5% | 5-15% | 15-30% | 30%+ |
|---|---|---|---|---|
| `linear` (rescale only, no exposure term) | -0.777 | -0.424 | -0.874 | +2.64 |
| `linear+sat` (what ships) | **-1.143** | **-0.636** | **-4.472 (z -3.33)** | +1.71 |

The exposure term does not tax the bench. It earns its largest gain by far exactly where the bench is, and
the 30%+ group -- where every map loses to no map -- is 22 seasons with a standard error of 2 to 5 and says
nothing at |z| < 1.2. The pooled -1.24 is not bought at the bottom of the board's expense.

**And what it says about the bio candidate: nothing.** `board_D_weight` under `game_bench_share`, mapped against the
shipped board's mapped rows: -0.022 (z -0.93) at 0-5%, -0.003 at 5-15%, **+0.097** at 15-30%, **+0.351** at
30%+. No group is significant and the two bench-heavy groups mildly favour the shipped board. At STINT level
inside the `exposure` split -- the valid metric there -- it is -0.396 at z -2.73 over 19 of 28 seasons on the
zero-exposure rows, which is a real measurement of a different thing: how well the margin of the individual
stints a barely-seen player is on the floor for is predicted, before nine other players and a level refit
dilute him. **The two do not agree, and the criterion's own unit is the team-game.** So the standing tie rule
stands and the shipped board keeps its feature list; `board_D_weight` is not a candidate any more.

The instrument survives the correction and is the lasting result: `--splits=game_bench_share` is how a candidate gets
asked about the bench from here, and it is a decomposition, so the answer adds up.

**Playing time only works multiplied by a stat -- and the defensive prior was missing its own past.** The
owner, 2026-09-10, after height and weight failed: *"games started% and minutes played can 100% help us
here"*, then *"gs% * feature and poss played % x feature, for all available features, boruta test adding in
these interaction features"*. `gbdt_prior.interaction_features` builds `gs_pct_x_<f>` = `gs_pct * f` and `poss_pct_x_<f>` =
`poss_pct * f`; `50_boruta.py --modes=interactions` ran the kitchen sink plus all 118 interaction features, 179 candidates, 50
trials, pair rows, both sides, the shipped target and window decay per side.

**The result is in the rejections.** `gs_pct` (-0.32 D, -0.27 O) and `poss_pct` (-0.30 D, -0.25 O) are
rejected on BOTH sides, near the shadow floor -- and the interaction features built from them are at the top of defense: `poss_pct_x_stocks`
3.77, second of 179 behind `past_rapm`; `gs_pct_x_entry_age` 2.68; `gs_pct_x_stocks` 1.65; `poss_pct_x_past_apm` 1.37, against
a `Max_Shadow` bar of 0.59. Eight accepted on defense, ten on offense; Boruta also rejects 8 of the
11 names the defensive prior ships and 16 of the 31 on offense. Two blocks per 100 possessions in 200
possessions and two per 100 in 5,000 are the same number and nothing like the same evidence, and a depth-4
oblivious tree can only say so by splitting on the rate and again on the exposure inside every leaf.

**The criterion then says the interaction features are not what moves it -- the PAST block is.** The ablation, because
`poss_pct_x_past_apm` puts the player's own plus-minus record on the defensive side for the first time (the shipped
defensive list has no `past_*` column at all), so a gain could be the past block arriving:

| defensive prior | criterion | z | seasons | consensus def | floors | prior, <250 poss | 250-500 |
|---|---|---|---|---|---|---|---|
| shipped | -- | -- | -- | 0.7565 | 10/10 | -- | -- |
| + `past_apm/poss/rapm`, no interaction features | **-0.133** | **-2.68** | 20/28 | **0.7468** | **9/10** | -0.140 (z -3.42) | -0.037 (z -2.21) |
| + 7 interaction features, no past | +0.005 | +0.15 | 13/28 | -- | -- | -0.109 (z -3.69) | -0.083 (z -4.39) |
| + past + 8 interaction features (`board_D_interactions`) | -0.090 | -1.68 | 19/28 | **0.7583** | **10/10** | -0.138 (z -3.94) | **-0.095 (z -4.45)** |

Interaction features alone move the criterion by nothing at all. The past block alone is the first thing since the
single-season board to clear |z| = 2 on it -- and it **fails the defensive consensus floor at 0.7468**, with
the defensive spread blown out to 1.383.

**So the two are complementary, and only together are they shippable.** The interaction features cost 0.043 of the past
block's criterion gain and buy back 0.0115 of consensus defensive agreement -- 0.7468 to 0.7583, from below
the floor to ABOVE the shipped board. And the prior diagnostic says why the interaction features are worth having on
their own terms: `board_D_interactions` improves the defensive prior in **all five possession buckets**, by z -3.94
at under 250 possessions per season and **z -4.45 at 250-500**, where neither ingredient alone is as
good as the pair. Nothing else tried this session moved a single bucket.

**Adding the ORIGINAL STATS back is what makes it clear the bar.** The owner, on reading the above: *"I
told you to do products PLUS the original stats."* Right -- the Boruta run did include all 61 originals
beside the 118 interaction features, but the systems built from its verdict did not. `board_D_interactions`
carries `poss_pct_x_stocks` and `gs_pct_x_stocks` with no `stocks`, `gs_pct_x_astr` with no `astr`,
`gs_pct_x_entry_age` with no `entry_age`. Boruta accepted those original stats too and they were dropped
between its verdict and the system. An interaction feature cannot stand in for the stat inside it:
`poss_pct * stocks` is near zero for a man who barely plays whatever his rate is, so without `stocks` beside
it the booster cannot tell "does not play" from "is not good at this". **Defense was missing six of its
seven original stats; offense was missing one (`p3r`)**, which is most of why the offensive interaction
features looked worthless -- they already had theirs.

| what is added to the shipped defensive list | criterion | z | seasons | floors | consensus def | prior <250 | 250-500 |
|---|---|---|---|---|---|---|---|
| the 8 interaction features only (`board_D_interactions`) | -0.090 | -1.68 | 19/28 | 10/10 | 0.7583 | -0.138 (z -3.94) | -0.095 (z -4.45) |
| ...and the 6 original stats (`board_D_interactions_stats`) | -0.110 | **-2.19** | 19/28 | 10/10 | **0.7588** | -0.176 (z -5.66, 10/10) | -0.075 (z -5.85, 10/10) |
| ...and `poss_pct` (`board_D_interactions_stats_possplayed`) | **-0.147** | **-2.74** | 20/28 | 10/10 | 0.7563 | **-0.177 (z -5.95, 10/10)** | -0.088 (z -4.09, 10/10) |
| the same on both sides (`board_OD_interactions_stats_possplayed`) | **-0.155** | -2.38 | 21/28 | 10/10 | 0.7567 | as above | as above |

Every one of them clears |z| = 2 on the criterion where the interaction features alone did not, and every
one passes ten of ten floors. `board_D_interactions_stats_possplayed` is the recommendation: the best criterion z of the
defence-only candidates, the defensive prior better in all five buckets and **all ten panel windows** in the
two lowest, and offense untouched. The both-sides version has the larger raw gain and a worse z, and it drops offensive
consensus agreement 0.8350 -> 0.8306, so offense stays rejected either way.

**Offense: rejected.** `board_OD_interactions` drops offensive consensus agreement 0.8350 -> 0.8257 and makes the
offensive prior worse in four of five buckets (z +2.03 at 4500+). Defense only.

**Boruta's full replacement lists lose.** Swapping in everything it accepted and dropping everything it
rejected is +0.056 on the criterion, worse than the shipped board and far worse than keeping the shipped
names and only ADDING. "Boruta prunes, it does not decide" held exactly as FINDINGS 22.2 says.

Three bugs stood between the idea and the measurement, all the same bug: an interaction is a name no gate
recognises. `poss_pct_x_past_apm` did not force pair rows the way `past_apm` does; `spm.offset`'s `wants` set gated
the career, bio and PAST blocks on names that never matched, so the prediction frame lost ingredients the
training rows had; and `pair_rows` selects only feature columns, so a deferred interaction feature's playing-time column was
absent from the pair frame while present on the panel. `interaction_inputs` is the fix in all three places, and
`test_interaction_features_reach_training_and_prediction_alike` is the regression.

### The board is a list of PLAYERS, so it is scored on players (2026-09-11)

*"players is the only thing that matters here."* The criterion is a possession-weighted MSE over team-games,
so a player enters it in proportion to how much he played and a 200-possession player is a rounding error:
the defensive prior for players under 250 possessions per season more than doubled in skill (0.096 -> 0.233)
and the team-game criterion moved 0.130% of its MSE. Two player-level losses now sit beside `score()` in
`holdout.py`, reached by `45_holdout.py --players` and `53_calmap.py --players`, one row per player and every
player counted once:

  * **rank, WITHIN a roster** -- Kendall tau-b over pairs of TEAMMATES. Owner, 2026-09-11: *"rank should only
    apply to within that team."* That is the decision an ordering supports (who plays, who is extended), and
    a league-wide tau is partly scoring "is this a good team": the board and the truth both inherit
    team-level effects and that easy part inflates the number. Within a roster the team mean is gone and
    what is left is separating players who share their possessions. `tau_league` is kept beside it, and
    `top_k` (top-50 concordance) stays league-wide, because "who are the best 50 players" is a league
    question and a full-list tau is dominated by the easy middle;
  * **dollars, ACROSS rosters** -- *"money should only apply to trades (really mid season here)."* For every
    CROSS-TEAM pair the board orders wrong, the money misallocated taking one for the other. `money_skill`
    is the share of that a coin-flip board would lose which this board avoids: 1 is perfect, 0 is saying
    nothing. The shipped board is at **0.299** against a near-unbiased truth. With 30 teams ~97% of
    league-wide pairs are already cross-team, so the restriction moves the number very little -- it is there
    to name what the loss is. The mid-season part needs no separate machinery: on a `_q75` frame the scored
    possessions ARE the season's last quarter, so the dollars are already deadline-onward, rest-of-season
    dollars.

**What "actual" is, and it is one thing.** `player_truth` is the prior-free ridge fit (`beta_none`, no box
term, no role offset) of EXACTLY the rows the criterion scores -- season H, or H's post-cut games for an
in-season system, so a `_q75` candidate is never scored on games it trained on. It contains no box prior, so
it cannot favour the board whose prior it shares, and it shrinks toward the average player, not toward
anything under test. One truth is built per held-out frame and shared by every candidate in the run.

**Three design decisions that each reversed a number, and all three are pinned by tests.**

*The losses must ignore where a board puts its zero.* `predict_season` refits the intercept on the held-out
season, so the team-game criterion cannot see a constant shift in a board at all. The dollars loss could:
dollars are a rating TIMES possessions, so a board whose zero sits at replacement instead of average makes
every player positive and orders them by minutes. On the simulator, an uncentred true-talent board (mean
+6.7) lost MORE dollars than a near-useless shrunken RAPM while beating it on every rank measure. Both sides
are now re-centred on the average possession before the conversion (`test_the_losses_ignore_where_a_board_puts_its_zero`).

*Rank and dollars order players by different quantities on purpose.* Rank is the board's own claim, points
per 100. A trade compares totals, so both sides of the dollars loss are the rating times the player's own
possessions in the window -- minutes held at what actually happened, not something either board is asked to
predict. Two tests pin the split: giving every player on a team the same wrong bonus leaves `tau` at 1.0 and
wrecks `tau_league`, and reversing a roster's internal order costs `tau` while the cross-team dollars do not
see it.

*A pair the board is indifferent about costs half the gap, not nothing.* Charging only the strictly
discordant pairs put a board with nothing to say at ZERO dollars lost -- the best possible score for the
worst possible board. Indifference means picking at random, so it is charged half
(`test_a_zero_board_sits_at_the_no_information_point`).

**What it decided.** `board_D_interactions_stats_possplayed` -- the owner's call on the criterion -- also wins
at player level, and it is the first evidence for it from a loss that can see the bench. Against the shipped
`ks00_lam05_ow_w0.25`, K=3, q75 frames, 28 held-out seasons, paired:

| truth (`--truth-lam`) | tau (within roster) | tau_league | dollars per trade | level: shipped tau / money_skill |
|---|---|---|---|---|
| `lam_plugin` (default) | +0.0035, z +1.75 | +0.0034, z +3.84 | -1,075, z -1.68 | 0.1722 / 0.4811 |
| `spm.apm_lam` = 100 (near-unbiased) | **+0.0074, z +3.17** | +0.0041, z +3.83 | **-29,821, z -3.22** | 0.1598 / 0.2986 |

Same sign on every measure at both truths. **The within-roster question separates the two boards almost twice
as hard as the league-wide one** at the honest truth (+0.0074 against +0.0041), which is the dilution the
owner's restriction was aimed at. It is also a harder question: the shipped board scores 0.160 within a
roster against 0.182 league-wide.

**Read `tau_pairs` before reading a bucket.** Within-roster tau on the whole board rests on ~2,286 teammate
pairs per season, but a possession bucket cuts it to ~37 pairs in 100-250 and ~61 in 250-500 -- the bucket
z's there are ±0.4 and decide nothing. The league-wide tau, which keeps ~1,200 to ~21,000 pairs per bucket,
is the one to read at the bottom of the board.

**Always read a player-level verdict at both truths.** The
calibration map is why: `linear+sat` against no map at all is tau **-0.0269 at z -6.82** on the shrunk truth
and **+0.0060 at z +1.35** on the near-unbiased one. The player loss has decided nothing about the map. The
default truth is low-variance but shrinks a low-possession player harder than a starter, so it rewards a
board that does the same; `apm_lam` is nearly unbiased and very noisy, and noise in an unbiased target costs
power without choosing a winner.

**Still open.** The year-over-year form of the question -- score a season-H board against the seasons AFTER
H, the owner's reliability criterion -- needs a training set that stops before H, because the criterion's
symmetric neighbourhood already contains H+1 and H+2. That is a different run, not a different loss.

### Single year or bust (2026-09-11)

*"i dont want to do that. single year or bust."* A player's season-H rating uses H's games for the evidence
and no games of his own from any other season. The trigger was noticing what the prior actually does: the
board rates H from H's games, but the prior reaches `past_apm` / `past_poss` / `past_rapm` -- that specific
player's own pre-H on-court record -- so "from that season's games only" was true of the evidence and not
of the prior. A prior's COEFFICIENTS may still be learned from history; that is what a prior is. The
per-player channel is what is out.

**It unships the candidate approved the day before.** `board_D_interactions_stats_possplayed`'s defensive
list carries `poss_pct_x_past_apm` and, through `_originals_for`, `past_apm` itself. The cost is known and
is not small: `board_D_interactions_nopast` -- the same interaction features with the past block removed --
was **+0.005 per 100 on the criterion, i.e. nothing**, so the whole defensive criterion gain WAS the past
block. What is open is whether single-year defense also loses on the player losses, which are the losses
that can see the population the past block was bought for.

`notebooks/single_year.ipynb` is the owner's own instrument for answering that: plain scikit-learn on both
sides (`HistGradientBoostingRegressor` for the prior, `Ridge` for PI-RAPM), `eracoef` only to load and to
score, one row per player-season, no `past_*`, no career pooling, and therefore no `gbdt_win_decay`. It
fits the first 75% of a season and scores the last 25% -- the shipped `_q75` estimand -- on the team-game
criterion and the player losses together. `python notebooks/build_single_year.py` regenerates it.

Smoke test on 2015, prior against no prior at all: team-game 114.58 -> 114.66, a tie and slightly worse,
while within-roster tau goes 0.133 -> 0.226 and money_skill 0.286 -> 0.316. One season decides nothing,
but it is the session's finding in miniature -- the criterion cannot see what the prior is for.

### De-biasing the prior's cross-validation: measured, and it is not where the money is (2026-09-11)

*"i don't care WHAT we do, provided it makes for better models. empirical data wins / scoreboard wins."*
Five ways of choosing the single-year prior's training rows, 27 held-out seasons (1999-2025), each fit on
the first 75% of its season and scored on the last 25%, on the team-game criterion AND the player losses.
`scratch/loo_bakeoff.py` (a local one-off; its source is gone and `scratch/` is gitignored, so the
table below is the record, not the script), results in `outputs/loo_bakeoff.parquet`.

| | tg | within-roster tau | money_skill | train rows |
|---|---|---|---|---|
| no prior at all | 114.576 | 0.1156 | 0.2630 | -- |
| `plain` (train on every season but H) | 112.847 | 0.1878 | 0.3116 | 14,092 |
| `reb_season` (+ drop the partner season) | 112.835 | 0.1886 | 0.3122 | 13,632 |
| `tilt_season` (+ reweight to the full mean) | 112.848 | 0.1889 | 0.3117 | 14,092 |
| `drop_player` (+ drop H's players) | 112.792 | 0.1880 | 0.3127 | 10,196 |
| `drop_player_tilt` (both) | 112.761 | 0.1876 | 0.3123 | 10,196 |

**The answer is that none of them matter, and the table says why in its first two rows.** The prior itself
is worth **+62% of within-roster tau** (0.1156 -> 0.1878) and 1.73 per 100 of team-game error. Every
de-biasing variant then moves tau by at most 0.0011 -- half a percent of what the prior already bought.
The place to spend effort is the prior, not the cross-validation around it.

Paired against `plain`, the only three results past |z| = 2, and they do not agree:

* `drop_player_tilt` is **-0.085 per 100 on the criterion, z -2.49, 21 of 27 seasons** -- and **-0.0081
  on money_skill at 100-250 possessions, z -2.65**. It wins the team-game number and loses the deepest
  bench's money. Under ruling "players is the only thing that matters here" the criterion win does not
  carry, and the prior it produces is 1.0-1.3% NARROWER on offense, so part of that criterion gain is
  plain shrinkage rather than information.
* `tilt_season` is **+0.0011 on within-roster tau, z +2.21**, costs no data at all, and is flat on
  everything else.

**And three hits is what 28 comparisons produce by chance.** Four variants x seven metrics, so ~1.3
results past |z| = 2 are expected with nothing going on. This is at the edge of noise and is reported as
such. `plain` stays the default; `tilt_season` is free and harmless if you want it; `drop_player` throws
away 28% of the training rows and buys nothing.

### Choose a training target's penalty on the end-to-end score, never on its own accuracy (2026-09-12)

The prior's target is now `LeaveSeasonOutRAPM` -- one rating per PLAYER from every season except H
(`src/eracoef/looseason.py`).  Its three penalties were swept twice, against two different objectives, and
the two disagree in a way that is worth stating as a rule.

**The wrong objective.** `LeaveSeasonOutRAPM.sweep` scores a penalty by how well the resulting RAPM
predicts an unseen season's games directly.  Over 1997-2026 it chose offense 28,690, defense 69,711,
context 0.  Taking that answer cost **0.017 game ARMSE on 2015**, consistently across three different board
configurations: the defensive prior narrowed from sd 0.78 to 0.62 and `PriorRidgeCV` answered by driving
its own defensive penalty to the grid ceiling -- the ridge finding nothing useful to do with the data
rather than confidence in the prior.

**Why, in one line:** a ridge penalty trades bias for variance, direct prediction rewards the variance
reduction, and a training LABEL is punished by the bias.  The booster averages unbiased noise away across
2,500 players; it cannot average away shrinkage.  And nothing downstream repairs it -- `PriorRidgeCV`
shrinks the residual TOWARD the prior, so a compressed prior is a compressed board with no mechanism to
re-inflate.  This is the same principle as the truth-lambda rule the player losses needed: noisy but
unbiased beats quiet but shrunk, whenever something else is going to average over it.

**The right objective**, pooled over five seasons, building the prior and the board each time
(`scratch/sweep_end_to_end.py`, `outputs/end_to_end_sweep_wide.parquet`).  All figures are **game ARMSE in
points per 100 possessions**; the baseline with no player ratings at all is **8.9261 points per 100
possessions**.

> **Every number in this table is inflated, and the table is kept for the reasoning, not the values.**  The
> 36-name feature list it was run on carried the four `onc_*` columns, and season H's `onc_*` is computed
> over the quarter the score holds out -- worth +0.375 per 100 at z +7.47 (next section).  The rule the
> section states survives; the surface does not, and it was re-swept on the honest setup, where the
> penalty turns out not to be identified at all.

| target defence lambda -> | 10,000 | 20,000 | **40,000** | 80,000 | 160,000 |
|---|---|---|---|---|---|
| offence 10,000 | 8.4675 | 8.4440 | 8.4311 | 8.4588 | 8.5235 |
| **offence 40,000** | 8.3854 | 8.3662 | **8.3503** | 8.3699 | 8.4358 |
| offence 160,000 | 8.4052 | 8.3825 | **8.3483** | 8.3751 | 8.4321 |
| offence 640,000 | 8.4523 | 8.4359 | 8.4118 | 8.4348 | 8.4901 |
| offence 2,560,000 | 8.4527 | 8.4284 | 8.4053 | 8.4267 | 8.4807 |
| offence 10,240,000 | 8.4583 | 8.4305 | 8.4153 | 8.4269 | 8.4884 |

**Both axes are interior** -- the surface rises in every direction from the middle, so neither sits on a
grid edge.  An earlier, narrower run had offence pinned at its ceiling and had therefore chosen nothing on
that axis; four more decades of grid settled it.

**Defence is 40,000 and decisive**: one step either way costs 0.016 to 0.086 points per 100 possessions.

**Offence is bracketed but not identified.**  40,000 and 160,000 are 0.0020 points per 100 possessions
apart, split the five seasons 2-3, and swing by +/- 0.1 per season, so the gap is noise.  **40,000 is
taken**, because 160,000 halves the board's own offensive spread (sd 0.87 -> 0.45 points per 100
possessions) for no measurable accuracy, and a board that compressed says less about players.

**Context 0 wins on all three sweeps**: 8.3483 at 0, 8.3647 at 1,000, worse at 100,000.

**Still open: the board's own penalties.**  `PriorRidgeCV` pinned its offensive penalty at the grid top in
24% of fits and its defensive penalty in 25%.  With a multi-season prior this good and only three quarters
of one season of games, the board often wants to barely update the prior at all.  `DEFAULT_PLAYER_LAMBDAS`
runs to 1e9 now, where the residual is numerically nil so the top of the grid IS "keep the prior
unchanged", but how often that gets chosen has not been re-measured.

**Context 0 wins on every row of both sweeps.**  Including the context columns UNPENALISED is exactly
Frisch-Waugh -- the owner's point that a linear regression already strips a covariate out by carrying it --
and the equivalence only holds when that coefficient is free.  A ridge that penalises `home` and `margin`
along with the players is not controlling for them, it is partly ignoring them.

**A Frisch-Waugh bug the rewrite exposed.**  The old ridge residualised `y` on the context and then fit raw
`Z`, which is neither of the two correct routes.  FWL residualises BOTH sides; residualising only the target
leaves the gram at `Z'WZ` instead of `Z'WMZ`, so every player is shrunk harder than the penalty says and
unevenly -- a player whose minutes correlate with context (lopsided home/road, garbage time, always on with
a lead) more than one whose do not.  Both classes assemble `[Z | A]` and solve once with a block-diagonal
penalty now, which is exact FWL at context lambda 0.  *The check:*
`test_a_zero_context_penalty_is_the_frisch_waugh_fit` pins it against a two-sided residualiser.

**Not the board's 3-D CV.**  Sweeping three penalties per season instead of one is worth ~0.008 ARMSE,
noise, and it is kept because it removes two assumptions rather than because it pays.

### The single-year board: an amplitude instrument, a Boruta selection, and a leak (2026-09-13)

Three things were wrong with the single-year board and they compounded: its prior's features were
hand-picked rather than selected, everything downstream had been tuned on top of that, and the board came
out compressed with the plus-minus stage contributing nothing.  What follows is what each turned out to be.

**The objective could not see compression, and now it can.**  `calmap.py:808-811` already said why: a
team-game total is linear in a per-player linear map, so a uniform rescale of the board trades almost
nothing at game level.  The demonstration: target offence penalties 40,000 and 160,000 scored **8.3503
against 8.3483 game ARMSE** -- a tie, 0.0020 apart -- while 160,000 **halved** the board's offensive spread,
sd 0.866 to 0.448.  `holdout.score` was already returning the instrument that sees it and the callers were
throwing it away.  `scale_off` / `scale_def` are what the held-out quarter wants each side multiplied by;
`priorridge.calibration_miss` is |log(scale_off)| + |log(scale_def)|, zero when calibrated, symmetric in
the factor.  The same pair read **0.239 against 0.576**.  **The rule: choose on `game_armse`, and where
candidates sit within noise of each other -- these axes are flat, so that is most of the time -- break the
tie on `miss`.**  It is a tie-break and never a score: a board that predicts worse cannot win on it.

**`onc_*` leaks into the DIAGNOSTIC, and stays in the BOARD.**  `scripts/50_boruta.py` grew a `--panel=`
flag and a `single_year` mode (the target is the `LeaveSeasonOutRAPM` joined on `player_id`, not pooled --
it already IS the leave-one-out quantity and `training_rows` would pool it a second time).  Over 56
candidates and 50 trials it ranks **`onc_o` first on offence at importance 7.26 against 1.18 for the next
name**, and `onc_d` first on defence at 6.61.  Dropping the four columns costs 0.169 game ARMSE.

The 75/25 diagnostic cannot be believed on that number.  `outputs/role_panel_season.parquet` is built from
each season's FULL design (`scripts/49_role_panel.py`: `onc = oncourt_rates(wd_o, wd_d)`), so season H's
`onc_o` -- his points per 100 while on the floor -- is averaged over every game of H, the scored quarter
included.  Rebuild it from the first 75% on the same luck-adjusted targets and change nothing else
(`scratch/onc_leak.py`; the whole-season rebuild arm reproduces the panel to **0.0000**, so the arms differ
in the window and in nothing else):

| | 2015 | 2024 | 2025 | 2026 | pooled |
|---|---|---|---|---|---|
| panel `onc_*` (whole season) | 8.4732 | 8.8065 | 9.1021 | 9.0196 | 8.8503 |
| honest `onc_*` (first 75%) | 8.7766 | 9.3273 | 9.4179 | 9.3780 | 9.2249 |
| the gap | +0.303 | +0.521 | +0.316 | +0.358 | **+0.375, z +7.47, 4/4** |

**So the 75/25 score cannot compare two boards that differ in `onc_*`** -- including the 8.3872 the
single-year board was being benchmarked at, and both earlier end-to-end penalty sweeps, whose 36-name list
carried `onc_*` too.  It remains valid for every comparison that holds them fixed.

**It does not follow that the columns should go, and taking them out first was the wrong turn.**  The leak
is in the evaluation, not the product.  The board rates a COMPLETED season, ruling 12 allows H's own games
as the evidence, and at ship time H's on-court record is legitimately known -- as it is to every public
metric in the consensus, all of which use the season's own plus-minus.  "Would `onc_*` help if we only had
75% of the season" is a question the product never asks.  With the criterion silent the consensus decides,
as a sanity check: **with `onc_*`, rho 0.753 / 0.811 / 0.766 and nine of ten floors; without, 0.721 / 0.667
/ 0.663 and all three agreement floors fail, defence by 11%** -- a gross miss, which the standing rule does
treat as a veto.  `singleyear.FEATURE_SETS` keeps `boruta_noonc` so the comparison can be re-run.

The cost is real and is recorded rather than hidden: `onc_*` makes the PRIOR and the EVIDENCE the same
games, so the ridge is no longer combining two independent sources.  It shows up as the plus-minus stage
doing less -- the season's own games move the board by sd 0.19 on offence with them, 0.32 without.  An
honest 75/25 diagnostic needs a panel built from a 75% design, and that is not built.

**What Boruta actually keeps, and what it does not.**  The hypothesis being tested was that the
hand-picked list was missing the efficiency ratios and the shot-quality pair -- the things an axis-aligned
tree cannot construct, since a ratio of two columns is not a function of either one.  **It is not
supported on this target.**  Boruta rejects *every* `RATIOS` and *every* `SHOTQ` feature on defence, and
all but `fg3p` and `p3r` (both essentially on the bar) on offence: 33 of 54 rejected on offence, 37 on
defence.  The board agreed -- the full 54-name list left prediction flat on 2015 (8.3853 against 8.3872)
and made the prior *narrower*, offensive spread 0.717 against the shipped board's 1.056.  What survives is
small: 21 names on offence and 17 on defence, and dropping to them costs nothing (8.8503 against 8.8397,
z well inside noise, on a third of the features).  `outputs/csv/boruta_single_year_*.csv` has every
candidate and its importance; read the rejections, never a prose list of winners.

**`free_prior_scale` is the lever on amplitude, and it is the one that mattered.**  A ridge already shrinks
each player by n_i / (n_i + lambda), so possession-dependent shrinkage is free and per-player shrinkage was
never the missing piece.  The missing piece was that the fit was *told* how far to trust the prior instead
of deciding.  `PriorRidgeCV(free_prior_scale=True)` enters each side's prior as one free UNPENALISED
column, so the rating is `scale * prior + residual`; pinning the scale at its own optimum is the same
least-squares problem, which is the sense in which it contains the old fit exactly
(`test_a_free_prior_scale_is_the_fixed_fit_at_the_scale_it_chose`).  Over four seasons:

| | game ARMSE | miss | rating_off sd | u_off sd |
|---|---|---|---|---|
| fixed centre | 8.7896 | 1.038 | 0.705 | 0.009 |
| free prior scale | 8.8397 | **0.274** | **1.454** | **0.191** |
| paired difference | +0.050, **z +0.92** | -0.764, **z -4.07** | | |

A tie on the score and decisive on calibration, which is exactly the case the tie-break rule was written
for.  The fit asks for **2 to 3 times** the prior it is handed.  It also un-sticks the plus-minus stage: the
board's own penalty had pinned at the 1e9 ceiling in 7 of 8 fits -- "keep the prior exactly as it came" --
and the season's own games moved the offensive board by sd 0.009.  With the scale free the penalty is
interior in **4 of 4** seasons and the games move it by 0.191.

**REML: not adopted, and the reason is the loss.**  The motivation was that the board's CV curve is flat
and its argmin wandered to the grid ceiling.  Both halves of that dissolved.  `MixedModelRAPMCV` with the
box block zeroed and the GBDT prior as `prior_offset` puts the REML lambda at **2,516, identical in all
four seasons and interior once the grid runs down to 20** -- against the board's own team-game CV at 12,608
to 316,569.  REML maximises a stint Gaussian likelihood; the board is selected on a closeness-weighted
team-game error, and this project has already been burned by choosing a constant on stint MSE
(`lam_plugin`, trap 3).  Meanwhile the ceiling-pinning that REML was meant to cure is gone, cured by
`free_prior_scale` instead.  `scratch/reml_lambda.py`.

**`lam_buckets`: a null, and a clean one.**  The claim was that the under-500-possession group wants its
penalty multiplied.  Swept over ratios 0.25 to 10 -- a 40-fold range -- the score moves **0.004 per 100**,
monotonically, and `miss` does not move at all (0.195 to 0.195).  For scale, every other effect in this
section is one to two orders of magnitude larger.  The best point is the *lightest* bench penalty tested,
which is the opposite of the hypothesis, and it is on a grid edge, which means it has chosen nothing.
`lam_buckets` stays `{}`.  `scratch/bucket_grid.py`.

**The luck-adjusted targets are flat here, and are kept anyway.**  The board now explains `xpts_ft` on
offence and `x3def_w0.25` on defence, one fit per side, scored always on the points actually scored.
Within-season on the 75/25 diagnostic that is worth **-0.0013 per 100, z -0.10** -- nothing.  It is kept
because the evidence for these targets is out-of-season (+0.26 and -0.39 to -0.47 per 100, above in this
file) and a within-season split cannot see an attribution gain, not because this measurement supports it.

**The target's penalty is no longer identified, so the incumbent stays.**  Re-swept coarsely on the honest
setup (`outputs/end_to_end_sweep_honest*.parquet`), 19 triples over five seasons: the surface spans 0.048
per 100 in total, it is **non-monotone** in both axes (offence 10,000 / defence 1,250 beats offence 2,500 /
defence 1,250), and the argmin slides to whatever edge is lowest on every widening.  Best-of-19 against the
incumbent 40,000 / 40,000 is z -1.80 over five seasons, which is not significant before accounting for
having selected it out of 19.  **40,000 / 40,000 / 0 is kept**, on the `lam_plugin` precedent: the incumbent
is the one value not chosen with knowledge of the answer.  The mechanism behind the flatness is visible in
the sweep's own columns -- as the offensive penalty goes 10,000 to 160,000 the prior's spread collapses
1.168 to 0.288 and `prior_scale_` rises 1.59 to 5.44 to compensate, leaving the finished board at 1.85
against 1.44.  **The free scale absorbs what the penalty used to control**, which is why a dial that was
worth re-sweeping for six hours is now worth 0.048.

**One engineering note that is worth more than it sounds.**  `solve_penalised` fell back to `np.linalg.lstsq`
on *half* of every penalty grid -- every triple with `context_lambda = 0`, because inside a CV fold an
unpenalised fixed column can be identically zero (a fold with no garbage-time rows) and `solve` then raises.
`lstsq` costs five to ten times a solve on a thousand columns.  `solve_diag` now retries with a jitter on
the unpenalised diagonal only, which is the same fit to nine figures.  A season went from about twenty
minutes to about thirty seconds, which is the difference between a sweep being affordable and not.

### The year-over-year test, and what it says about the single-year rankings (2026-09-13)

**The test.**  Rate a season from that season's games only.  Use those ratings, and nothing else, to predict
every stint of the season BEFORE it and the season AFTER it, with only the level (intercept and home edge)
refit on the scored season.  Score against the points actually scored, per stint and summed to team-games.
Twenty-eight scored seasons (1998-2025), each predicted twice, every table paired by scored season.
`scripts/63_yoy.py`, on top of `holdout.py`'s runner; nothing is fit inside it, so what is measured is
exactly the table handed in.

Why it replaces the within-season 75/25 split as the test for this pipeline: the 75/25 split holds a quarter
of the games out of the ridge but not out of the prior's on-court features (the leak above); here the scored
games are a different season from the rated one, so nothing in the rated season's prior or evidence has seen
them.  It is the owner's own reliability test (2026-09-04), run for the first time on one-season ratings.

One requirement, and it turns out to cost almost nothing: the prior's population-level fits for the rated
season must not have seen the two seasons being scored.  `scripts/62_single_year_board.py
--exclude_neighbours=1` leaves the rated season and its two neighbours out of the leave-season-out target and
out of the prior's training rows (`outputs/season_ratings_sy_yoy.parquet`).  Against the same pipeline with
the neighbours in, that is worth **+0.21 team-game MSE, 15 of 56 better** -- the leak through population
coefficients is small.  The 75/25 on-court leak (+0.375 per 100 at z +7.47) was the one that mattered, and
this test does not have it.

**What it says.**  Pooled over both directions, 56 scored-season observations, team-game error in points per
100 (`game_armse`); `scale_off` / `scale_def` are what the scored season wants each side multiplied by:

| | game_armse | scale_off | scale_def | paired vs single-year, team-game MSE |
|---|---|---|---|---|
| single-year rankings, neighbours excluded | 8.804 | 0.73 | 0.71 | -- |
| single-year rankings, neighbours in (`season_ratings_sy`) | 8.812 | 0.72 | 0.71 | +0.21, z +2.2, 15 of 56 |
| shipped rankings (`artifacts/season_ratings.parquet`) | **8.587** | 1.18 | 0.78 | **-5.93, z -16.9, 56 of 56** |
| single-year PRIOR alone (`prior_*` columns) | 8.840 | 0.71 | 0.70 | |
| shipped PRIOR alone | 8.681 | 1.32 | 1.51 | -4.32 vs the single-year prior, 50 of 56 |
| single-year offence + shipped defence | 8.730 | 0.76 | 0.73 | -2.04, z -12.7, 52 of 56 |
| shipped offence + single-year defence | 8.659 | 1.13 | 0.74 | -3.97, z -14.3, 55 of 56 |

Three findings, in the order they matter.

1. **The single-year rankings predict the neighbouring seasons worse than the shipped rankings, in every one
   of 56 comparisons, and the gap is ranking, not amplitude.**  After each side is rescaled to what the scored
   season wants, the shipped rankings are still better by 6.2 stint MSE (z -19.5, 56 of 56), against 7.5
   before rescaling.  The `movers` split shows the gap in every group, including lineups where none of the
   ten changed team, so it is not specifically a team-to-team attribution failure.
2. **The prior is the weak stage.**  The shipped prior on its own (8.68) beats the finished single-year
   rankings (8.80); the ridge stage adds about 0.04 per 100 on top of the single-year prior and about 0.09 on
   top of the shipped one.  Both sides lose.  Swapping in the shipped offence is worth -3.97 and the shipped
   defence -2.04, and **the defensive comparison is the legal one**: `features_full_D` is eleven box rates
   with no on-court and no `past_*` column, so it is a single-year-legal prior that beats ours -- whereas
   the shipped offensive list carries `past_apm`, `past_poss`, `past_rapm`, the player's own other seasons,
   which ruling 12 bans, so its offensive edge is not a target this pipeline may chase.
3. **Amplitude, both tables, in numbers.**  The single-year rankings are too wide for the neighbouring
   season on both sides -- 0.73 offence, 0.71 defence, below 1 in 28 of 28 seasons in both directions.  The
   free prior scale asks for 2.8x the prior within the season; the neighbouring season wants about 2.0x.  The
   shipped rankings are too NARROW on offence (1.18) and too WIDE on defence (0.78), which is the owner's
   "defence is counted too much" as a measurement.  A diagnostic: the score barely moves under a uniform
   rescale, and finding 1 says the gap is elsewhere.

**Decision rule from here.**  A change to the single-year pipeline is adopted only if it improves the pooled
year-over-year team-game error with a paired mean difference at least twice its standard error over the 56
observations, with no gross miss on the consensus checks; ties go to the simpler version.  One change per
run, named after the change.  The consensus report (`scratch/consensus_report.py`) now also prints
`def_share` (defence's share of rating variance) and `team_r2_*` (how much of a player's rating his team
alone predicts); the neighbours-excluded table reads 0.758 / 0.788 / 0.758 agreement, top-five overlap 3,
defensive team R-squared 0.241 against the consensus's 0.195.  Reported every time, chosen on never.

### Experiment 2: the prior trained on one row per player-season, not one per player (2026-09-13)

**The change, and whose idea it was.**  NOT the owner's design -- it was the assistant's proposed first step
toward it and was wrongly reported as the owner's intent.  The owner wants the SPM trained on ONE row per
player (his career average with the rated season out, what `prior_rows` builds) PLUS extra rows for the same
player built from chunks of his seasons -- leave two out, any three, any one -- as an artificial increase of
the sample that teaches the booster how noise changes the map (experiment 3 below).  What this experiment
ran instead REPLACED the career row with single-season rows: `--rows=season` trains on one row per
player-season (12,248 rows against 2,860), each labelled with his RAPM over his seasons OTHER than that one
and the rated one.  Everything else held: features, targets, penalties, the ridge.  It is kept as a
measurement of one ingredient of the owner's design, no more.

**Result, year-over-year, both directions, 56 observations** (`outputs/yoy_exp2_rows.log`):

| | game_armse | scale_off | scale_def | paired vs one-row-per-player, team-game MSE |
|---|---|---|---|---|
| one row per player (`season_ratings_sy_yoy`) | 8.804 | 0.73 | 0.71 | reference |
| one row per player-season (`season_ratings_sy_rows`) | **8.715** | 0.85 | 0.77 | **-2.48, z -7.0, 47 of 56** |
| after each side is rescaled to the scored season | | | | -1.63, z -4.7, 45 of 56 |
| the PRIOR alone, player-season rows vs player rows | 8.825 vs 8.840 | | | -0.37, z -0.57, 21 of 56 |
| shipped rankings, for scale | 8.587 | 1.18 | 0.78 | -5.93, 56 of 56 |

It passes the decision rule on the test, and the gap to the shipped rankings closes from 5.93 to 3.45.
**The gain is in the ridge stage, not the prior.**  The prior on its own is a tie.  What changed is that the
season's own games now get in: the defensive ridge penalty, pinned at the 1e9 ceiling under the old prior
("keep the prior exactly"), is interior in every season, and what the games add on defence goes from sd
0.000 to sd 0.31 per 100.  The reading: a booster trained on single seasons learns how far to trust one
season's on-court number, so the prior no longer pre-empts the games, and the defensive rating stops being
a copy of the team's -- the team R-squared on defence falls from 0.241 to **0.190**, below the consensus's
0.195, the first version here to get there.

**And it fails the consensus sanity check grossly, which is a veto as it stands.**  Agreement 0.570 offence /
0.656 defence / 0.651 total against 0.758 / 0.788 / 0.758 before, top-five overlap 2.  Per season the prior
alone agrees less too (2024 offence 0.404 against 0.483).  Two defects in the specification as run, both
found after the fact:

1. **No per-player weight cap.**  Each row is weighted by the possessions behind its label, so a
   fifteen-season player has fifteen rows each carrying his whole career, and the top tenth of players
   hold 53% of the training weight against 43% under one row per player.  Capping (divide by his row
   count) brings it to 39%.  `season_rows(cap_per_player=True)`, `--rows=season_capped`, the next run.
2. **One-season players lose their label.**  A player with a single season outside the excluded ones has
   nothing left to label it from, so 1,917 players train against 2,494.  The missing ones are exactly the
   short-career, low-sample players the consensus cut (1,000+ possessions over 2024-26) is full of.

The free amplitude also reads 3.4x to 9.4x on offence across seasons, against 1.6x to 2.2x before: the
booster's output is much narrower (noisier labels, sd 0.815 against 0.567) and the amplitude is weakly
identified.  Not chosen on, but it is the symptom to watch when the capped run is read.  (The capped rerun
was stopped unread once the owner made clear this was not the design; experiment 3 is.)

### Experiment 3: the owner's design -- the career row per player PLUS chunks of his seasons (2026-09-13)

**The change.**  `prior_rows` kept exactly: one row per player, his box score averaged over every season
but the rated one (and its neighbours, for the test), labelled with his RAPM over those seasons.  ADDED,
for the same player and with the same label: one row per contiguous run of 1, 2 and 3 of his seasons, his
inputs averaged over the chunk, plus two features saying how much evidence the row rests on
(`chunk_poss`, `chunk_seasons`; the rated season's own row reads its possessions and 1).  Weights: a
player's chunk rows together weigh what his career row weighs, split by chunk possessions, so no one's
weight grows with career length.  35,647 training rows against 2,860; 70 seconds a season.
`singleyear.chunk_rows`, `--rows=chunks --chunk_sizes=1,2,3`.  The owner's reasoning: an SPM suffers from a
small sample, and the same player at several noise levels teaches the booster how the map degrades with
less evidence -- padding learned from data instead of set per stat.

**Result, year-over-year, both directions, 56 observations** (`outputs/yoy_exp3_chunks.log`):

| | game_armse | scale_off | scale_def | paired vs career row only, team-game MSE |
|---|---|---|---|---|
| career row only (`season_ratings_sy_yoy`) | 8.804 | 0.73 | 0.71 | reference |
| career row + chunks (`season_ratings_sy_chunks`) | **8.736** | 0.75 | 0.77 | **-1.90, z -5.7, 42 of 56** |
| stint level, each side rescaled to the scored season | | | | **-5.42, z -18.5, 56 of 56** |
| the PRIOR alone, chunks vs career row only | 8.798 vs 8.840 | | | -1.12, z -2.0, 31 of 56 |
| shipped rankings, for scale | 8.587 | 1.18 | 0.78 | -5.93; rescaled -6.21 |

Passes the decision rule.  Two things distinguish it from experiment 2.  **The prior itself improves** (z
-2.0, on the line), where the single-season-only prior was a tie.  And **once amplitude is taken out, this
design is nearly the shipped rankings**: after each side is rescaled to what the scored season wants, it
beats the career-row prior by 5.42 in every one of 56 comparisons, against the shipped rankings' 6.21.
The gap that remains at native scale is amplitude -- the rankings are too wide for the neighbouring season
by a quarter on both sides (0.75 / 0.77), the free prior scale asking for 2.2x within the season.  That is
a different problem from the one experiments 1-3 were about, and it is the next one.

**Consensus (`outputs/consensus_exp3_chunks.log`):** agreement 0.730 offence / 0.745 defence / 0.729 total
against 0.758 / 0.788 / 0.758 for the career row alone; three checks below 0.75 by 0.005 to 0.021; top-five
overlap 2.  Offensive spread relative to the consensus 0.764 (from 0.544, closer to 1), defensive 1.085.
Team R-squared on defence 0.179, below the consensus's 0.195.  A moderate miss, not the gross one
experiment 2 produced (0.570 / 0.656 / 0.651); between the "marginal is not a veto" and "gross is" lines
of the standing rule, and left to the owner.

What the season's own games add fell (offence sd 0.23 against 0.25, defence 0.19 against 0.09 -- the
defensive ridge is off its ceiling here too), and the rating spread rose (offence sd 1.83 against 1.38).

### Experiment 4: amplitude -- cross-fitting the free prior scale (2026-09-13)

**The problem.**  The rating is `scale * prior + residual`, the scale a free least-squares coefficient on
the prior summed over the five on the floor, fitted on the season's games.  The prior carries the season's
own on-court columns (`onc_*`, averaged over every game), so that column contains the outcome of every row
it is regressed on and the coefficient reads them back: 2.2x within the season where the neighbouring
seasons want about 0.75 of that on both sides (experiment 3).

**The change.**  `PriorRidgeCV.fit(fold_prior=...)`: the prior's on-court columns are rebuilt inside each
of the five whole-game CV folds from the training games alone (`oncourt_rates` on the two targets the
panel used, which reproduces the panel's columns to four decimals on the full season), the boosters are
re-asked, and each row's prior column takes the value from the prior that never saw its fold.  The scale
is then priced on games the prior has not seen.  The final rating still takes `scale * (full prior) +
residual`.  `--crossfit=1`, on top of experiment 3's rows.

**Result, year-over-year, both directions, 56 observations** (`outputs/yoy_exp4_crossfit.log`):

| | game_armse | scale_off | scale_def | paired vs experiment 3, team-game MSE |
|---|---|---|---|---|
| experiment 3 (`season_ratings_sy_chunks`) | 8.736 | 0.75 | 0.77 | reference |
| + cross-fitted scale and penalty (`season_ratings_sy_chunks_cf`) | **8.707** | 0.80 | **0.89** | **-0.75, z -6.5, 46 of 56** |
| stint level, each side rescaled to the scored season | | | | **+0.81, z +10.3, 0 of 56** |
| shipped rankings | 8.587 | 1.18 | 0.78 | -4.04 |

**It passes the rule at native scale and it is a trade, and the diagnostic row says which.**  The amplitude
moved the right way -- defence from 0.77 to 0.89, offence 0.75 to 0.80, the within-season scale from 2.2x
to about 1.9x -- and that is the whole of the gain: with each side rescaled to the scored season, the
cross-fitted rankings are WORSE in 56 of 56.  The mechanism is visible: with the cross-fitted columns in
the penalty CV too, the residual penalty pins at the 1e9 ceiling in 19 of 30 seasons on defence and 17
on offence (6 and 2 before), so the season's own games stop contributing (what they add: sd 0.04 offence,
0.06 defence, against 0.23 / 0.19).  An honest within-season CV cannot validate a per-player residual --
the same five appear together in every fold, trap 2 -- so once the prior column stops leaking, the CV
sees nothing left for the residual to do and switches it off.  The year-over-year test, which can see
the residual, says it was worth keeping.  Consensus: defence 0.705 against 0.745, offence 0.730 unchanged,
total 0.723; defensive share of variance 0.313 (from 0.386; the consensus 0.238).

**Experiment 4b, cross-fit the scale only -- ADOPTED as the single-year pipeline's incumbent (2026-09-14).**
`--crossfit=scale` (`crossfit_penalty=False`): the penalty grid scored with the full prior columns as in
experiment 3, the cross-fitted columns entering for the final fit alone, so the scale is priced honestly
and the residual is not switched off.  Same 56 observations (`outputs/yoy_exp4b_scaleonly.log`):

| | game_armse | scale_off | scale_def | games add, sd off / def | paired vs experiment 3 |
|---|---|---|---|---|---|
| experiment 3 | 8.736 | 0.75 | 0.77 | 0.23 / 0.19 | reference |
| experiment 4, scale and penalty cross-fitted | 8.707 | 0.80 | 0.89 | 0.04 / 0.06 | -0.75, z -6.5, 46 of 56 |
| **experiment 4b, scale only** (`season_ratings_sy_chunks_cfs`) | **8.693** | 0.80 | 0.90 | 0.22 / 0.18 | **-1.14, z -13.9, 54 of 56** |
| 4b, each side rescaled to the scored season | | | | | +0.13, z +5.6, 10 of 56 |
| shipped rankings | 8.587 | 1.18 | 0.78 | | -4.04 |

The amplitude gain of experiment 4 with the residual kept: the best single-year rankings on the test so
far, 54 of 56 against experiment 3, at a rescaled cost of 0.13 (against experiment 4's 0.81).  Consensus
0.730 / **0.753** / 0.737 -- defence back above 0.75, total 0.737 against 0.729; defensive share of rating
variance 0.338 (experiment 3: 0.386; the consensus 0.238); team R-squared on defence 0.191, below the
consensus's 0.195.  `scripts/62_single_year_board.py` now defaults to `--rows=chunks --crossfit=scale`.
The gap to the shipped rankings is 4.04 team-game MSE (8.693 against 8.587), from 5.93 when the test was
first run; the shipped offensive prior's `past_*` channel, banned by ruling 12, is part of what is left.

### The overnight queue: four single changes on the incumbent (2026-09-14, 00:34 to 02:37)

Each built on experiment 4b (`--rows=chunks --chunk_sizes=1,2,3 --crossfit=scale`) with one thing changed,
scored on the year-over-year test against it, both directions, 56 observations.  `outputs/yoy_<name>.log`,
`outputs/consensus_<name>.log`.

| change | game_armse | native scale, paired vs incumbent | each side rescaled | scale off / def | games add, off / def | consensus off / def / total | verdict |
|---|---|---|---|---|---|---|---|
| incumbent (4b) | 8.693 | -- | -- | 0.80 / 0.90 | 0.22 / 0.18 | 0.730 / 0.753 / 0.737 | |
| every contiguous chunk size, 1 to the full career (`sy_chunks_all`, 59,514 rows) | 8.682 | -0.30, z -2.1, 32 of 56 | **+1.44, z +10.0, 3 of 56** | 0.85 / 0.92 | 0.31 / 0.14 | 0.707 / 0.765 / 0.704 | **rejected** |
| `onc_d` off the defensive list (`sy_noonc_d`) | 8.684 | **-0.24, z -2.6, 36 of 56** | **-0.36, z -3.8, 40 of 56** | 0.80 / 0.88 | 0.22 / **0.40** | 0.731 / **0.689** / 0.705 | passes the test; consensus defence 8% under 0.75; **the owner's call** |
| booster `l2_leaf_reg` and `min_child_weight` x5, both sides (`sy_reg5`) | 8.692 | -0.04, z -0.4, 34 of 56 | +0.08, z +1.0 | 0.81 / 0.90 | 0.32 / 0.17 | 0.730 / 0.763 / 0.727 | tie, rejected |
| booster depth 3, both sides (`sy_depth3`) | 8.697 | +0.10, z +0.8, 28 of 56 | +0.14, z +1.6 | 0.81 / 0.90 | 0.31 / 0.15 | 0.750 / 0.757 / 0.738 | rejected |

**Every chunk size: rejected, and the reason is the rescaled row.**  The native-scale gain (z -2.1, 32 of
56, just on the line) is amplitude -- both scales move toward 1 -- and with amplitude removed the rankings
are worse in 53 of 56.  Offensive and total consensus agreement fall 0.02 to 0.03.  The extra rows are
long chunks, nearly the career row again with nearly the same label, and they dilute the short chunks
that carry the noise information.  Sizes 1, 2 and 3 stay.

**`onc_d` off the defensive list: the one that passed, and it is left to the owner.**  Better at native
scale and better with amplitude removed, and what the season's own games add on defence goes from sd 0.18
to 0.40 -- the plus-minus stage doing on defence what it was supposed to do once the prior stops handing a
player his lineup's points allowed.  The defensive rating's team R-squared falls to 0.101, half the
consensus's 0.195.  The price is consensus defensive agreement 0.689 against 0.753, 8% under the 0.75
check -- more than the 0.4% the standing rule called marginal, less than the 11% it called gross.  The
prior spread on defence rises (sd 1.38 against 1.16).  Not adopted without a ruling; the consensus is the
attribution sanity check and this is an attribution change.

**The booster's settings: closed.**  Five times the regularisation is a tie (z -0.4) and depth 3 is
slightly worse; the incumbent settings stay on the standing tie rule.  One more caution for the amplitude
question: a rating calibrated within its own season is EXPECTED to read below 1 on the neighbouring
season, because true impact changes from year to year, so `scale_*` at 1.0 is not the target; what the
diagnostic is for is comparing two tables and the two sides.

The incumbent's prior alone reads 8.722 against the shipped prior's 8.681 (`outputs/yoy_incumbent_prior.log`).

### The owner's idea: the off-court record beside the on-court one (2026-09-14)

*"On court rating is a useful but flawed metric. Usually we also include off-court-rating."*
`investigate.offcourt_rates`: for each player-season, his team's luck-adjusted points scored and allowed
per 100 over the rows of games he played in where his team was on the floor WITHOUT him, centred and
padded exactly as the on-court columns are (`scripts/65_offcourt_panel.py` writes them into the season
panel; the recomputed on-court columns match the panel's to six decimals).  Feature set `boruta_offc`:
the incumbent's lists plus `offc_o`, `offc_d`, their possessions, and the net `onc - offc` per side, on
both sides.  Cross-fitting rebuilds all of them per fold.  One change, on the incumbent.

| | game_armse | paired vs incumbent | each side rescaled | consensus off / def / total | games add, off / def |
|---|---|---|---|---|---|
| incumbent (4b) | 8.693 | -- | -- | 0.730 / 0.753 / 0.737 | 0.22 / 0.18 |
| + off-court and net (`sy_offc`) | 8.688 | -0.15, z -1.6, 32 of 56 | -0.02, z -0.3, 29 of 56 | 0.726 / 0.755 / 0.731 | 0.22 / 0.22 |

**A tie on the test, and the eye test on the 2026 top 20 says worse.**  The order of the top seven is
nearly unchanged (Wembanyama, Kawhi, Jokic, Giannis, Curry, LeBron, Harden) but Shai Gilgeous-Alexander
falls from 8th to 32nd, Donovan Mitchell 26th to 47th, Jalen Brunson 28th to 51st, while Ajay Mitchell
and Cason Wallace rise to 9th and 10th and Al Horford enters at 11th on 1,992 possessions.  The
mechanism: an off-court column on a deep team is the player's TEAMMATES' quality, so a role player on
Oklahoma City reads as good because the Thunder stay good without him, and the star on the same team
reads as ordinary because the team does not collapse without him.  The on/off net is the honest
version of that comparison, but it enters beside the raw off-court column and the booster used both.
Rejected on the standing tie rule (the simpler wins a tie) with the eye test agreeing; not a ruling,
the owner reads the top 20.

### The SPM memorises long-career stars, and the out-of-player prior removes it at a price (2026-09-14)

The owner: *"I think but am not sure that the model is memorizing the players ... LeBron should not by
any metric imaginable be ranked this highly."*  Test (`scratch/memorisation_2026.py`): the 2026 SPM
fitted five times on player folds, every player's prior from the fit that never saw his rows, against the
prior from the fit that did.  Across 391 players with 1,000+ possessions the two agree (correlation 0.98
offence, 0.95 defence; typical gap 0.12 per 100); for a few long-career stars they do not: Curry's prior
falls 1.8 per 100 (4.7 to 2.9), LeBron's 1.4 (4.4 to 3.0), Brunson's 1.1, Klay's 0.9, Kawhi's 0.6;
Harden's and Shai's do not move.  Mechanism: every one of a player's chunk rows carries his career label,
and career possessions (LeBron 276,000) and years played identify him, so the booster returns his career
number for his 2026 row.  The panel rows are current (LeBron 2026: age 41, 4,797 possessions).

**The fix, the owner's design: LOSO x five player folds, folds balanced on the label.**  `--player_folds=5`:
the SPM fitted once per fold, each player's prior from the fit without his rows (`OutOfPlayerSPM`).  The
folds are built by `rloocv.BalancedGroupKFold`: players sorted by weighted mean label, dealt in snake
order, so every fold's weighted mean equals the full mean and the leave-out shift of Austin, Pe'er and
Korem (2025) is zero by construction (measured 0.005 per 100 against a label sd of 0.82; no partner fold
dropped).  Cross-fitting re-asks the right fold model per player.  Five fits a season, about 3 minutes.

| | game_armse | paired vs incumbent | each side rescaled | consensus off / def / total, top five | 2026 top 20 |
|---|---|---|---|---|---|
| incumbent (4b) | 8.693 | -- | -- | 0.730 / 0.753 / 0.737, 2 | |
| out-of-player priors (`sy_oop`) | 8.705 | **+0.34, z +2.6, 24 of 56** | +0.71, z +5.4, 11 of 56 | 0.721 / 0.750 / 0.733, **3** | LeBron 5th to 13th, Curry 4th to 6th; Shai 8th to 5th, Luka 27th to 10th, Butler 39th to 16th; Draymond 14th to 43rd, Jamal Murray 16th to 64th, Derrick White 17th to 34th |

**Worse on the test, and that is the point.**  The memorised channel carried real information -- a
player's own long-run level -- and the year-over-year test rewards it, because a career-level number
does predict his next season.  It is exactly the per-player channel ruling 12 bans ("no games of his own
from any other season"), reaching the rating through the booster's memory instead of a `past_*`
column.  So the test and the ruling disagree here and the ruling outranks the test.  Not adopted by the
assistant; it is a ruling, and the owner reads the top 20 (which now has Shai 5th, Luka 10th, LeBron
13th, but also drops Draymond, Murray and White a long way).

### Experiments 1 and 2 of 2026-09-14 evening: the ridge penalty, and the un-shrunk label

Both on the incumbent (out-of-player priors, `season_ratings_sy_oop`), one change each, 56 observations.
The priors were rebuilt once with `--save_priors` so the ridge alone could be swept in a second a season.
A thread-pinning bug cost the evening: with numba at twelve threads and a second process on the machine
the same booster fit went from 15 s to 45+ minutes; `62_single_year_board.py` now pins BLAS to one thread
and numba to four before importing anything (30 s for the offensive fit, 9 s for the defensive one).

**Experiment 2: one fixed ridge penalty for every season instead of the per-season whole-game CV.**  The
CV had switched the season's games OFF on offence in 2024-2026 (penalty at the 1e9 ceiling, offensive
rating = SPM x 1.9), and an honest within-season CV cannot validate a per-player residual (experiment 4).

| penalty, both sides, all seasons | game_armse | paired vs incumbent (8.706) | each side rescaled | consensus off / def / total, top five |
|---|---|---|---|---|
| 2,000 | 8.781 | +2.08, z +8.6, 7 of 56 | +1.71, 9 of 56 | |
| **13,037** | **8.697** | **-0.22, z -1.6, 29 of 56** (a tie) | +0.49, 15 of 56 | **0.777 / 0.757 / 0.771, 4** |
| 84,978 | 8.718 | +0.37, z +2.5, 16 of 56 | +1.74, 0 of 56 | |
| 553,918 to 1e9 | 8.732 to 8.736 | +0.76 to +0.86, 8 to 12 of 56 | +2.26 to +2.39, 0 of 56 | |
| incumbent, chosen per season by CV | 8.706 | reference | | 0.721 / 0.750 / 0.733, 3 |

No single penalty beats the per-season choice on the test; 13,037 ties it.  What 13,037 changes: the games
are back on for offence in every season (2026: what they add, sd 0.39 against 0.00), every consensus check
rises (offence 0.72 to 0.78, top five 4 of 5), and the 2026 list moves the way the owner's eye test asked --
Jokic 2nd, Shai 4th, Curry 7th, Harden 14th, LeBron out of the top 20, Jamal Murray off the offensive top.
Against: the rescaled row is worse by 0.49.  Recommended for adoption on the standing tie rule (one fixed
number is simpler than a CV that cannot see what it chooses).  **ADOPTED by the owner, 2026-09-15**: the
rankings script defaults to `--lambda_player=13037`; the product table was rebuilt and the site republished.

**Offence and defence separately (2026-09-15, the owner: "same penalty on both sides though?").**  The
1-D sweep had tied the two sides.  A 4 x 4 grid over 2,000 / 13,037 / 84,978 / 553,918 for each side, the
ridge alone on the saved priors, paired against 13,037 / 13,037 over the same 56 observations: **every one
of the fifteen other pairings is worse.**  The nearest are offence 84,978 / defence 13,037 (+0.18 team-game
MSE, z +2.7, 21 of 56) and offence 13,037 / defence 2,000 (+0.52, z +5.7, 12 of 56); offence at 2,000 is
+1.8 to +2.4 whatever the defence.  The equal pair stays.  `outputs/yoy_sy_lo<off>_ld<def>.log`.

**The fine grid (2026-09-15, the owner: "defense really should be penalized more"):** offence 6,519 to
26,074 and defence 13,037 to 104,296 in root-2 steps, 35 pairings, same saved priors, paired against
13,037 / 13,037.  Paired team-game MSE (negative = better), rows offence, columns defence:

| off \ def | 13,037 | 18,437 | 26,074 | 36,875 | 52,148 | 73,750 | 104,296 |
|---|---|---|---|---|---|---|---|
| 6,519 | +0.23 | +0.30 | +0.38 | +0.46 | +0.54 | +0.62 | +0.69 |
| 9,219 | +0.07 | +0.14 | +0.22 | +0.31 | +0.39 | +0.46 | +0.53 |
| 13,037 | 0 | +0.07 | +0.15 | +0.23 | +0.31 | +0.39 | +0.45 |
| 18,437 | -0.02 | +0.05 | +0.13 | +0.21 | +0.29 | +0.37 | +0.43 |
| 26,074 | 0.00 | +0.07 | +0.15 | +0.23 | +0.31 | +0.39 | +0.45 |

**Every step up in the defence penalty is worse, monotonically**, +0.07 per root-2 step at first and
z above 5 by 26,074, and consensus defensive agreement falls with it (0.757 at 13,037, 0.741, 0.727,
0.715, 0.705, 0.697, 0.691).  With the coarse grid's defence 2,000 at +0.52, 13,037 is an interior
optimum on defence.  Offence is a plateau from 13,037 to 26,074 (18,437 reads -0.02, z -1.0, a tie) and
worse below; the consensus's OFFENSIVE agreement rises as the offence penalty falls (0.797 at 6,519)
while the test gets worse, which is the forecasting-versus-attribution split again.  13,037 / 13,037
stays; a heavier defence penalty is the one direction the test rules out.

**Experiment 1: the un-shrunk label, with players with few possessions shrunk toward their possession tier.**
`--unshrink_label=1`: label_i = beta_i / max(s_i, s_floor) + (1 - s_i / max(s_i, s_floor)) x m(tier_i),
s_i = n_i / (n_i + 40,000), s_floor at 4,444 possessions, m(tier) the tier's mean coefficient un-shrunk by
the tier's mean s.  Tier levels learned (offence, per 100): -5.4 under 500 label possessions, -3.9 to
1,500, -2.5 to 4,444, +0.3 above; defence the mirror.  The free prior scale reads 0.84 / 0.92 instead of
1.9 / 2.0 -- the prior arrives on the right scale by itself.

| | game_armse | paired vs incumbent | each side rescaled | consensus off / def / total, top five |
|---|---|---|---|---|
| un-shrunk label (`sy_unshrink`) | **8.667** | **-1.06, z -4.3, 41 of 56** | **+2.28, z +10.9, 3 of 56** | **0.677 / 0.689 / 0.680, 2** |

**Passes the test and is REJECTED, and this is the clearest case yet of what the test cannot see.**  The
native-scale gain is real and it comes from the bottom of the roster: players under 200 possessions now
sit at -5.2 per 100 instead of 0, which is the truth about replacement-level minutes and predicts the
neighbouring season's games where those minutes are played.  But the top of the list is scrambled --
2026: Wembanyama, Giannis, Donovan Clingan, Holmgren, Neemias Queta, Ausar Thompson, Shai, Ajay Mitchell,
Hartenstein, Kalkbrenner, Cooper Flagg, Jokic 12th; Curry 61st, Luka 72nd -- young bigs and rookies
everywhere, the consensus checks fall to 0.68 (a gross miss, 9% under), the rescaled row is the worst
recorded, and the games add nothing (penalty at the ceiling, sd 0.00 on offence).  The mechanism: the tier
level enters the label of every short-career player, and "few career possessions" is what a rookie or a
young big looks like at inference, so the SPM hands them the tier's optimism.  The bench level is right and
the way it was put in is wrong.  Keep the idea (players with few possessions at replacement level) and find another route
-- a replacement-level FILL for players with too few possessions to rate, outside the SPM, is the obvious one.

**Experiment 11: a replacement level for the players a season's rankings have no row for.**  A season's
table covers only the men who played that season, so when it predicts a neighbouring season the rookies and
the returnees of THAT season have no rating and the criterion scores each of them as the average player.
They are not a rounding error: **10.0% of the scored season's possessions going forward, 6.0% going back,
94 to 98 players a season.**  The fill is `holdout.ReplacementSystem`, now with a `shrink` fraction, wired
into the year-over-year test as `63_yoy.py --fill=<name>:500x<fraction>`: the absent player scores that
fraction of the possession-weighted mean rating of the rating season's OWN players under 500 possessions.
It reads nothing but the rating season, it changes no rating in the table, and so the 2026 top 20, the
consensus checks and the site are untouched by construction -- the two vetoes cannot fire on it.

The bench level itself is stable: **-1.73 offence / -0.47 defence, -2.20 total** under 500 possessions,
-2.33 under 1,000, -2.22 under 2,000.  The depth that wins is a quarter of it.

| fill, as a fraction of the bench level | total | game_armse | paired vs incumbent | prev (rookies) | next (departures) |
|---|---|---|---|---|---|
| none (the incumbent) | 0 | 8.6971 | -- | -- | -- |
| **0.25** | **-0.55** | **8.6919** | **-0.144, z -5.24, 42 of 56** | **-0.084, z -1.89** | **-0.204, z -6.99** |
| 0.50 | -1.11 | 8.6898 | -0.200, z -3.55, 39 of 56 | -0.048, z -0.53 | -0.352, z -6.20 |
| 0.75 | -1.66 | 8.6910 | -0.169, z -1.92, 36 of 56 | +0.107, z +0.78 | -0.445, z -5.34 |
| 1.00 | -2.20 | 8.6954 | -0.050, z -0.41, 34 of 56 | | |
| 1.82 (-3.0 / -1.0 by hand) | -4.00 | 8.7209 | +0.638, z +2.83, 28 of 56 | | |

**The two directions are different populations and that is what fixes the depth.**  Going backwards -- a
season rated by the NEXT season's rankings -- the absent men are the ones who left the league, and deeper
is monotonically better out to the full bench level.  Going forwards they are the rookies, and the fill
stops paying past a quarter and is worse than nothing by three quarters: a first pick plays starter minutes
and is not a bench player.  **0.25 is the only depth that is better in both directions**, it has the best
z of the five, and the forward direction is the one a rating is actually used in.  The pooled error is flat
from 0.25 to 0.75 (8.6919 / 8.6898 / 8.6910), so the choice is made on the direction split, not on the
pooled number.

**The in-table players with few possessions needed nothing.**  The idea that started this (experiment 1: the bench sits at
0 and should sit at replacement) is not true INSIDE the table.  Under 100 possessions a player already
reads -1.61 offence / -0.47 defence with sd 0.42, and his own season's games move him by 0.06; under 500,
-1.71 / -0.46.  Centring at possession-weighted zero plus a box prior on a box score with few possessions already puts him
at the bench level.  Only the players with NO row were at 0, and that is the whole of the effect.

**Experiment 12: the prior blended toward a replacement level, the blend fitted on APM (the owner's
design, 2026-09-15).**  Between the box prior and the final ridge:

    new_prior = old_prior * w(n) + x * (1 - w(n)),     w(n) = n^a / (n^a + k^a)

per side, `n` the offensive possessions on offence and the defensive ones on defence.  Fitted against APM
and never RAPM -- a ridge penalty pulls every player toward zero, so the RAPM of a player with few possessions is near zero by
construction and a blend fitted on it would return `x` ~ 0, which is the error being corrected.
`scripts/67_blend_apm.py`: the season's normal equations solved with NO penalty on either player block,
and the blend chosen to minimise `(b - APM)' G (b - APM)` with `G` the player gram after the context block
is profiled out -- the same thing as least squares on the season's own stints with the context refit free.

**Three things had to be right before the fit meant anything, and each was wrong first.**

1. *The blend must carry its own free prior scale.*  Fitted as written, `k` pinned at the bottom of the
   grid in every season with `|x|` at 40 to 64.  That corner is `w` ~ constant, where `x (1 - w)` is a free
   constant and `w * prior` is a plain RESCALE -- so the blend was being used as a prior scale, which
   `PriorRidgeCV.free_prior_scale` already provides and would have absorbed downstream.  The fit now reads
   `b = s * w * prior + x * (1 - w)` and the reference it must beat is `s * prior` alone.
2. *`G` is singular by construction.*  Profiling the context out takes the season intercept with it, and
   "add a constant to every offensive rating" puts five times that constant on every row, which IS the
   intercept column.  Both all-ones directions are exactly null, so **APM cannot see a constant added to
   every player: only how a correction VARIES with possessions is identified.**  `np.linalg.solve` does not
   raise on that -- it returns a huge vector -- so `solve_diag`'s lstsq fallback never fires.  2018 has two
   further null directions and came back with `|apm| = 4e20`, and that one season swamped every pooled fit
   (held-out objectives of 2e10 against a 1.2e7 reference).  APM is now a minimum-norm solve through an
   eigendecomposition; every direction it drops is one the objective is flat along.
3. *The 4 x 4 needs a normalised basis and a guarded argmin.*  Unnormalised, the degenerate corner returned
   `|theta| = 1e5` of pure rounding error and the argmin picked it.  A least-squares minimum cannot fall
   outside [0, quad], so any grid point that does is dropped.

**Validation before any number was read:** the fitted scale comes out at 1.6 to 2.4 per side, and the
ridge's own `free_prior_scale` independently reads 1.9 / 2.0.  Two estimators, one number.

| | offence | defence |
|---|---|---|
| held-out APM distance vs a free scale alone, `a` free | 0.9883, better in 25 of 30 | 0.9926, 21 of 30 |
| the same with **`a` = 1**, the owner's plain `n / (n + k)` | **0.9895, better in 26 of 30** | **0.9936, 22 of 30** |
| pooled `k` (identical in all 30 leave-one-out fits, `a` = 1) | **64.2** | **12.1** |
| pooled `x`, 1997 -> 2026 | -15.4 -> -10.8 | +22.7 -> +16.3 |

**The steepness exponent is not needed.**  Freeing `a` wins 25 of 30 where `a` = 1 wins 26 of 30, and the
pooled ratio is 0.9883 against 0.9895 -- a tenth of the gain, in the wrong direction on the season count.
Once the fit carries `s`, the absorption the exponent was meant to dodge is handled by the scale instead.
The owner's original curve stands.

**`x` and `k` do NOT hold still season by season, and the pooled fit does.**  On one season alone
`x_off` has sd 11.95 on a mean of -16.30 and `k_off` runs 5.3 to 1,791.  Pooled over 29 seasons, `k` is
64.16 on offence and 12.14 on defence in **every one of the 30 leave-one-out fits**, and `x` drifts
smoothly with the era rather than jumping.  One season cannot price this blend; thirty can.

What the pooled curve does, points per 100, `a` = 1:

| possessions | 200 | 500 | 1,000 | 2,000 | 4,000 | 6,000 |
|---|---|---|---|---|---|---|
| offence, added | -3.60 | -1.69 | -0.89 | -0.46 | -0.23 | -0.16 |
| offence, multiple kept on his own prior | 1.52 | 1.78 | 1.89 | 1.95 | 1.98 | 1.99 |
| defence, added (positive = allows more) | +1.27 | +0.53 | +0.27 | +0.13 | +0.07 | +0.04 |

Only the VARIATION down those rows is identified, not the level: the gap between a 200-possession man and
a 6,000-possession one is **-3.44 on offence and +1.23 on defence**, and that much APM can see.  1999, the
50-game season, is the worst held-out season on both sides (1.016 and 1.012) -- the least data, as expected.

**Not yet a rankings result.**  Everything above is APM distance, the blend's own fitting criterion.  It
says the blend generalises to a season it was not fitted on; it does not say the rankings improve.  The
year-over-year test has not been run on it.

**Experiment 13: the replacement level as a MODEL on covariates instead of one constant (the owner's
design, 2026-09-15).**  Experiment 12's scalar `x` produced twenty 2026 players below -10 per 100 and one
at -22.41 on four possessions.  The diagnosis was that a constant target sits entirely in the direction
the loss cannot see, so `x` floated.  The fix: `x` becomes `g(z) = z . beta` over `singleyear.
LEVEL_COVARIATES` -- age, exp_yrs, exp_poss, entry_age, height, weight, draft_pick, poss_pct, gs_pct,
tenure, n_teams -- every one measured EXACTLY however few minutes a man played, with `beta` pinned by
players who DO have minutes so a man with few possessions borrows strength from players with heavy minutes who resemble him.
`scripts/67_blend_apm.py --covs=level`; the basis went from 2 columns a side to `1 + p` and the closed
form is unchanged.  **`--covs=scalar` reproduces experiment 12 to a maximum relative objective difference
of 0.0 on six season-sides, with identical k, scale and level** -- the scalar is a point in this family,
not an approximation of one.

| held-out APM distance vs a free prior scale alone | offence | defence |
|---|---|---|
| the scalar (experiment 12) | 0.9895, 26 of 30 | 0.9936, 22 of 30 |
| **the covariate model** | **0.9799, 26 of 30** | **0.9801, 25 of 30** |
| the same without `height` | 0.9802, 26 of 30 | 0.9807, 25 of 30 |
| without `poss_pct` and `gs_pct` | 0.9817, 25 of 30 | 0.9829, 25 of 30 |

**The fit roughly doubles on offence and triples on defence, and the level is still not a level.**

1. *`height` buys nothing*: 0.9802 against 0.9799.  `DECISIONS.md` above already found height belongs
   nowhere on offence and makes the deepest bench worse; nothing here overturns it.
2. *With `poss_pct` and `gs_pct` in, the fitted level RISES with possessions*: -5.33 under 100 possessions,
   +0.59 at 1-2k, **+15.39 above 4,000**.  A replacement level cannot do that.  `poss_pct` was the largest
   offensive coefficient (+11.52, z 7.8), so the model had learned "the coach played him a lot, therefore
   he is good" -- the very covariate this file already sized as 97% hindsight.  `g` had quietly become a
   SECOND PRIOR, not a bench level.
3. *Dropping those two fixes the shape and costs almost nothing* (0.9817 / 0.9829, level now flat at -6.6
   to -12.7 across every tier).  The hindsight pair was carrying nothing real: the control passes.
4. *The magnitude is still wrong by a factor of six.*  The intercept reads **-8.82 offence** for an average
   player against a measured bench of **-1.55**, and `exp_poss` at -6.84 (z -12.5) against `exp_yrs` at
   +5.02 (z 7.8) nearly cancel for a normal player and extrapolate violently for a veteran.  Applied, the
   correction `(1 - w) g` is:

| possessions | under 100 | 100-200 | 200-500 | 500-1k | 1-2k | 2-4k | over 4k |
|---|---|---|---|---|---|---|---|
| offence, mean (worst) | **-4.82 (-9.83)** | -3.37 (-8.19) | -2.70 (-7.88) | -1.25 | -0.82 | -0.48 | -0.36 (-1.30) |
| defence, mean (worst) | **+3.48 (+12.44)** | +1.47 | +0.82 | +0.35 | +0.19 | +0.09 | +0.06 |

**The covariates were never going to fix this, and the reason is the loss, not the parameterisation.**
Each player's weight in `(b - apm)' G (b - apm)` is his own information `G_ii`, so the level is set by the
mid-range players who want a -0.4 correction, and the players with the fewest possessions -- who carry no weight at all --
inherit the WHOLE of `g`.  Any parameterisation of a level runs away there; more flexibility just changes
the shape of the runaway.  The real content of the experiment is a genuine, generalising, possession-
dependent correction to the box prior worth about **-0.4 per 100 on a full-time player**, confirmed on 25
to 26 of 30 unseen seasons.  That is an exposure calibration of the prior, and it must not be expressed as
a replacement level, because a replacement level is precisely the quantity this loss cannot measure.

**What is already known and keeps being re-derived**: the box prior ALREADY places players with few possessions at the
measured bench (-1.55 offence / -0.45 defence under 100 possessions, sd 0.42, their own season's games
moving them 0.06).  There is no gap at the bottom to close.  Experiments 12 and 13 both found their real
gain in the middle and then damaged the bottom on the way out.

**Experiment 14: the exposure slope, and what it accidentally uncovered (2026-09-15).**  The parsimonious
end of experiments 12 and 13: one slope per side on log possessions added to the RATING, replacing the
blend, `k`, `a` and the eleven covariates with two numbers.  `scripts/68_exposure_slope.py`.

The motivating measurement is sound and stands on its own.  Centred at possession-weighted zero, one row
per player, 30 seasons, `scratch/bucket_apm.py` and `scratch/bucket_persist.py`:

| possessions | 0-50 | 100-200 | 500-1k | 1-2k | 4k+ |
|---|---|---|---|---|---|
| box prior, offence | **-0.76** | -0.80 | -0.83 | -0.80 | +0.45 |
| APM, offence | **-9.94** | -5.37 | -3.35 | -2.15 | +1.31 |

**The prior is FLAT in exposure where the truth is steep** -- it hands -0.76 to a man with 0-50
possessions and -0.81 to one with 1,000-2,000, a difference of 0.05 on a standard error of 0.03, while APM
runs across eleven points.  And the deficit is not a rough few possessions: those same men read -5.15 the
season before and -3.64 the season after, playing about 700 possessions in each, so **48% of the
deficit from the season with few possessions persists** (86% by 200 possessions, essentially all of it by 1,000).  Both directions
were measured because forward alone cannot separate persistence from ageing; it is symmetric, so it is a
property of the man.  Fitted against the rating, the slope is **+1.170 offence / -0.381 defence per decade**,
leave-one-neighbour-out sd 0.042 and 0.040 -- about as stable as a coefficient gets.

**And it does not buy a single point of forecasting accuracy.**

| year-over-year, both directions, 56 paired | game_armse | vs incumbent | order row (each side rescaled) |
|---|---|---|---|
| incumbent | 8.6971 | -- | -- |
| the slope, full dose | 8.7450 | **+1.310, z +13.3, 2 of 56** | -0.009, z -0.1 |
| quarter dose | 8.7057 | +0.235, z +9.6, 5 of 56 | -0.077, z -3.6 |

Every dose is worse, monotonically in the dose, while the ORDER row is mildly better -- the signature of a
candidate that is all amplitude.  `scale_off` went 0.796 to 0.707: the board was already 26% too wide and
the slope made it wider.

**Holding the spread fixed reversed it, and the control showed why.**  Renormalised to the incumbent's
spread, the full dose read 8.6533 at z -12.9 and every dose passed.  But a pure uniform shrink -- the
incumbent multiplied by 0.85, no exposure term at all -- reads **8.6430 at z -16.6, 56 of 56, with an order
row of 4.9e-14, i.e. no reordering whatsoever by construction.**  The shrink alone beats the exposure
correction that carried it.  The apparent win was the renormalisation smuggling in an amplitude fix.

Tested fairly, with amplitude equalised on both sides (`--rankings=season_ratings_ps7080 --mult=0.5
--match_spread=1`), the exposure correction is **-0.102 at z -1.77, 33 of 56, with an order row of z -0.69**:
short of the -2 bar and no order gain. **Rejected.**

**What was actually found, and it is the largest single term on this test.**  A per-side constant
multiplier, changing no player's rank at all:

| | game_armse | vs incumbent | wins | order row |
|---|---|---|---|---|
| incumbent | 8.6971 | -- | -- | -- |
| x0.90 both sides | 8.6573 | -1.085, z -17.7 | 56 of 56 | 0 |
| x0.80 both sides | 8.6323 | -1.764, z -15.5 | 56 of 56 | 0 |
| x0.70 both sides | 8.6222 | -2.038, z -12.9 | 54 of 56 | 0 |
| **offence x0.70, defence x0.80** | **8.6187** | **-2.139, z -13.7** | **54 of 56** | **1.6e-14** |
| offence x0.60, defence x0.70 | 8.6188 | +0.005 vs ps7080, z +0.1 | 27 of 56 | 0 |

A plateau from about x0.60/0.70 to x0.70/0.80.  The stint-level per-side regression says the MSE-optimal
factors are 0.796 and 0.889 (`scale_off` / `scale_def` at the incumbent), and the closeness-weighted
team-game objective wants rather more shrinkage than that; the two are allowed to disagree and
`game_armse` decides.

***The check this forces on everything else.*** **The board's amplitude is worth 0.078 points per 100 on
this test, and every experiment scored on `game_armse` that moves the board's spread has been partly
measuring that.**  For scale: the absent-player fill (experiment 11) was 0.008, the whole fine penalty grid
0.02.  The amplitude term is five to fifteen times larger than the differences those experiments were
deciding.  *The check:* pair candidates at MATCHED calibration -- read the "each side rescaled to the
scored season" row, or equalise the spread first -- before quoting any `game_armse` difference as a
ranking gain.  A uniform per-side rescale reorders nobody, so it is a calibration fix and never a ranking
result; the single-year pipeline has no amplitude-calibration stage at all (the ridge's `free_prior_scale`
prices the PRIOR, not the finished board), which is why this was sitting unclaimed.

**Also true and separately useful:** a same-season attribution gap can be real, large, monotone, z 7 to 15,
persistent across neighbouring seasons, and still worth nothing for forecasting.  Experiments 12, 13 and 14
all found their signal in the middle of the exposure range and then either damaged the bottom or moved only
the amplitude.

**Experiment 15: tell the prior what SCORE STATE a player's statistics were compiled in (the owner's
idea, 2026-09-15).**  The owner's question after experiment 14: "can't we deal with it by using the current
margin of victory -- eg it's a garbage time possession?"  The asymmetry is real and it is in the code.  The
ratings side already knows the score state: `design.py` carries a garbage-time column plus `margin` and
`margin x time-remaining`, so APM is already net of the blowout effect.  **The prior's inputs know nothing
about it** -- zero garbage-time references anywhere in the box-score feature build -- so a player's counting
statistics are credited at full value whether the game was tied or a rout.

And the premise holds.  From the stints (`scratch/gt_share.py`, all seasons, per side):

| possessions | 0-50 | 100-200 | 500-1k | 2-4k | 4k+ |
|---|---|---|---|---|---|
| share of his possessions in garbage time | **61%** | 40% | 19% | 5.6% | **2.5%** |
| mean absolute score margin while on court | 18.0 | 15.2 | 11.0 | 7.8 | 6.7 |

Monotone across every group: a man with 0-50 possessions takes three fifths of them in blowouts, a
4,000-possession man one fortieth.  His per-possession rates describe a different game.

`scripts/69_closeness_panel.py` adds three columns per player-season **per side** (his offensive and his
defensive possessions need not sit at the same score state): `gt_share`, `closeness` (the possession-
weighted mean of 1 / max(|margin|, 1) -- deliberately the same form `priorridge.team_game_weights` uses on
the ratings objective, so "close" means one thing in both halves of the pipeline) and `abs_margin`.
Feature set `boruta_close`.  The owner's call was to hand the prior the exposure and let the booster learn
the discount rather than reweighting the statistics -- and, explicitly, **not to pad these columns**.

*They are not padded, and that was the trap.*  None of the three is in `RATIOS`, `DERIVED` or `BIO_BINS`
and none has a `raw_` twin, so `add_derived` skips them: 166 player-seasons sit at exactly 1.000 and 192
at exactly 0.000, which padding can never produce.  Padded, a 61% share and a 2.5% share would both
collapse toward the league mean of 0.052 and the column would still exist, still look sensible and carry
nothing.  `CLOSENESS` is in `INPUT_COLUMNS` so `aggregate` carries the columns into the career and chunk
training rows; that is possession-weighted averaging, not shrinkage.

| year-over-year, both directions, 56 paired | game_armse | vs incumbent | order row (each side rescaled) |
|---|---|---|---|
| incumbent | 8.6971 | -- | -- |
| `boruta_close` | 8.6925 | -0.134, z -1.73, 29 of 56 | -0.150, **z -2.09**, 32 of 56 |

**A near miss on the score that decides, and the mechanism did not fire.**  The consensus checks all pass
and defensive rank agreement IMPROVES, 0.757 to 0.770, with `spread_off` 0.707 and the 2026 top of the
list intact (Kawhi, Wembanyama, Jokic, Shai, Giannis, Towns).  Unlike experiment 14 this is not amplitude:
`scale_off` barely moves, 0.7958 to 0.8015.  So the small gain is real ranking movement.  But it is not the
gain that was aimed at, because **the prior is no less flat in exposure than before**: its spread from the
fewest possessions to the most reads **+2.424 before and +2.303 after** -- slightly flatter -- against the
+11.2 that APM says is really there.

***Why no feature could have fixed it, and this is the finding worth keeping.***  `singleyear.chunk_rows`
gives every training row of a player **the SAME label as his career row** -- its own docstring says so.
The features vary across his chunks, including `chunk_poss` and now the three closeness columns; the target
does not.  So the booster cannot learn "fewer possessions means genuinely worse" from within a player, only
"short careers are worse" across players -- and the career leave-season-out RAPM label is itself shrunk at
penalty 40,000, which flattens even that.

That separates two things this project has been conflating:

  (a) *noisier inputs for the same true value* -> regress the estimate harder.  This is what the chunk
      design teaches, it is what its docstring means by "how the map degrades with less evidence", and it
      works.
  (b) *fewer possessions correlating with genuinely lower true value* -> shift the estimate down.
      **Nothing in the pipeline teaches this, and (b) is what experiment 14 measured.**

Experiments 12, 13 and 14 all tried to patch (b) onto the OUTPUT and failed, because the fitting loss
weights each player by his own information and is therefore blind to the men being corrected.  Experiment
15 tried to teach it through a FEATURE and failed, because the target has no exposure gradient to learn
from.  The only place (b) can enter is the label -- which is what experiment 1 did, and it scrambled the
top of the list.  *The check:* before adding a feature meant to teach a gradient, confirm the TARGET
varies along that gradient.  A constant-per-player label cannot teach a within-player effect however many
columns are added beside it.

### Experiment 16: the owner's blend with the level MEASURED, not fitted (2026-09-15)

`new_prior = prior * w(n) + x * (1 - w(n))`, `w = n / (n + k)`, applied to the PRIOR before the ridge and
inside the cross-fitted fold priors, so the ridge re-prices the amplitude afterwards.  `x` was NOT fitted:
it was read off the neighbouring seasons' APM for the lowest possession group (-4.40 offence, +1.69
defence) and divided by the ridge's prior scale of about 1.9, giving **-2.3 on offence and +0.9 on
defence**.  Only `k` was swept, over 50 / 150 / 400 / 1,000 / 2,500, on the saved priors (`priors_sy_base`,
built at `--exclude_neighbours=1`), one second a season.

| year-over-year, both directions, 56 paired | game_armse | vs incumbent | order row (each side rescaled) |
|---|---|---|---|
| incumbent | 8.6971 | -- | -- |
| k = 50 | 8.7040 | +0.188, **z +2.90**, 18 of 56 | -0.067, z -1.00, 37 of 56 |
| k = 150 | 8.7137 | +0.456, z +4.61, 14 of 56 | +0.081, z +0.83, 30 of 56 |
| k = 400 | 8.7247 | +0.756, z +5.84, 15 of 56 | +0.385, z +2.97, 24 of 56 |
| k = 1,000 | 8.7322 | +0.963, z +6.35, 16 of 56 | +0.761, z +4.89, 16 of 56 |
| k = 2,500 | 8.7358 | +1.058, z +6.42, 15 of 56 | +1.136, z +6.61, 12 of 56 |

Monotone in `k`: every larger blend is worse.  Nothing clears the rule (z -2 or below), so **nothing is
adopted**.  The consensus does not veto -- k = 50 is marginally BETTER than the incumbent on every
agreement number (rho 0.781 / 0.760 / 0.779 against 0.777 / 0.757 / 0.771, top five 4 of 5 both) -- and the
2026 top 20 is untouched except two adjacent swaps (Doncic up one past Mitchell, Barnes up one past
Butler).  The bottom of the 2026 list moves as designed and stays sane: 6 players below -6 per 100 at
k = 50 and 17 at k = 150, against 0 for the incumbent and 57 for experiment 12.

***The mechanism fired, for the first time in five attempts.***  Experiments 12-15 never changed the
prior's exposure gradient.  This one does, and lands close to the measured target.  Offence, mean rating
by possessions, 0-50 against 1-2k, over 30 seasons:

| | 0-50 | 200-500 | 1-2k | 4k+ | gradient, 0-50 minus 1-2k |
|---|---|---|---|---|---|
| incumbent | -1.59 | -1.71 | -1.74 | +0.94 | **+0.15** (flat, slightly backwards) |
| k = 50 | -4.70 | -2.22 | -1.78 | +0.97 | **-2.92** |
| k = 150 | -5.27 | -2.84 | -1.85 | +1.00 | -3.42 |
| the measurement (neighbours' APM) | -4.40 | -3.44 | -2.07 | +0.83 | **-2.33** |

So the defect named on 2026-09-15 is now fixable, and the ridge did re-price it rather than letting it run
away.  The rankings are still worse.

***Where it lost, and it is amplitude and not order.***  The exposure split puts the whole cost on the men
being corrected -- stint level, k = 50, possessions 1-499 in the rated season: **+1.197, z +3.65**, against
+0.171 at 1,500-3,999 and +0.039 at 500-1,499.  But on the same split with each side rescaled to the
scored season the cost is **-0.002, z -0.01**: dead neutral.  The order of those men is neither better nor
worse; their LEVEL is further from what the neighbouring season pays than the flat prior was.  The
realised gradient overshoots the measurement by about 25% (-2.92 against -2.33) because the ridge's free
prior scale rose with the blend (1.65 / 2.06 to 1.65 / 2.08 at k = 50, and 1.87 / 2.70 at k = 2,500)
instead of absorbing it.  *The check:* when a correction is applied through the prior, read the realised
gradient in the finished rankings, not the `x` that was handed in -- a free scale downstream can amplify it.

### Experiment 17: the owner's linear ramp -- no prior at zero possessions, the whole prior at N (2026-09-15)

The owner's design, replacing the rational weight of experiment 16: `new_prior = prior * w + x * (1 - w)`,
**`w = min(n / N, 1)`**, `n` the player's possessions on that side.  At zero possessions a man gets no
prior at all and exactly the replacement level; at `N` possessions and above he gets exactly the prior;
linear in between.  The replacement level was fixed by the owner at **-1.5 on each side** (entered as
`+1.5` on defence, whose prior is in the raw sign where positive means allows more), and **only `N` was
swept**.  Implemented as `--blend_off=x/N/lin`, `--blend_def=x/N/lin`; the rational weight is unchanged.

| year-over-year, both directions, 56 paired | game_armse | vs incumbent | order row (each side rescaled) |
|---|---|---|---|
| incumbent | 8.6971 | -- | -- |
| **N = 100** | **8.6972** | **-0.0002, z -0.003, 33 of 56** | -0.020, z -0.46, 32 of 56 |
| N = 250 | 8.7006 | +0.093, z +1.24, 27 of 56 | +0.039, z +0.58, 24 of 56 |
| N = 500 | 8.7092 | +0.328, z +3.19, 20 of 56 | +0.254, z +2.76, 19 of 56 |
| N = 1,000 | 8.7213 | +0.658, z +4.84, 14 of 56 | +0.670, z +4.94, 14 of 56 |
| N = 2,000 | 8.7303 | +0.906, z +5.44, 17 of 56 | +1.194, z +6.84, 8 of 56 |
| N = 4,000 | 8.7341 | +1.020, z +5.74, 15 of 56 | +1.784, z +9.75, 4 of 56 |

Monotone in `N` again, and **N = 100 is an exact tie** -- one ten-thousandth of a point per 100, z -0.003.
Not adopted: the rule needs z -2 or below, and a tie goes to the simpler version, which is the incumbent
with no blend at all.  The consensus does not veto anywhere (N = 100 reads 0.777 / 0.757 / 0.772 against
the incumbent's 0.777 / 0.757 / 0.771; N = 500 is the best of the lot at 0.780 / 0.765 / 0.781 and is z
+3.19 on the test, which is the usual reminder that the consensus is a sanity check and not a target).
The 2026 top 20 is untouched at both N = 100 and N = 250 -- the single move is Doncic and Mitchell swapping
9th and 10th.  The bottom of the 2026 list: 8 players below -6 per 100 at N = 100 and 28 at N = 250,
against 0 for the incumbent.

***The tie is made of a small loss where it acts and nothing anywhere else.***  The exposure split, stint
level, N = 100: possessions 1-499 costs **+0.387, z +1.63**, and every other group is a hair worse too
(1,500-3,999 at +0.033, 500-1,499 at +0.013).  On the rescaled row the 1-499 group is +0.013, z +0.08.  So
the ramp buys no ranking gain anywhere; it is merely cheap enough at N = 100 that the total washes out.

***What the ramp cannot do, and this is the finding worth keeping.***  It does move the exposure gradient,
offence, mean rating by possessions over 30 seasons:

| | 0-50 | 50-100 | 100-200 | 200-500 | 500-1k | 1-2k | 2-4k | 4k+ |
|---|---|---|---|---|---|---|---|---|
| incumbent | -1.59 | -1.64 | -1.70 | -1.71 | -1.83 | -1.74 | -0.89 | +0.94 |
| N = 100 | **-3.82** | -2.35 | -1.70 | -1.71 | -1.83 | -1.74 | -0.89 | +0.94 |
| N = 250 | -4.26 | -3.66 | -2.87 | -1.77 | -1.83 | -1.74 | -0.89 | +0.95 |
| N = 1,000 | -4.36 | -4.20 | -4.04 | -3.51 | -2.45 | -1.67 | -0.84 | +0.98 |
| **the measurement** | **-4.40** | **-3.44** | **-3.29** | **-3.44** | **-2.85** | **-2.07** | **-1.21** | **+0.83** |

But the measurement is not a ramp.  It is a **cliff below about fifty possessions and then a plateau**:
-3.44, -3.29, -3.44 across 50 to 500, flat, before it rises.  A straight line from `x` to the prior cannot
make that shape.  The `N` that fits the cliff (100) leaves 100-500 untouched; the `N` that fits 200-500
(1,000) has already driven 50-200 half a point past the truth and costs z +4.84.  Every version of this
correction so far -- rational weight, linear ramp, log slope -- has been a monotone function of
possessions, and the thing being corrected is a step.

### The exposure correction is CLOSED: the shipped level is good enough (the owner, 2026-09-15)

After experiments 12 to 17 the owner looked at the live site, saw a man with four possessions rated -1.96
per 100, and ruled: *"whatever our version is as last posted seems to be good enough."*  **No exposure
correction ships, and the line is closed.**  The shipped rankings are unchanged.

The case for closing it, written down so it is not reopened by accident:

- **The group is 7% of the players and 0.13% of the possessions.**  At or below 100 possessions in the
  rated season: 1,023 of 14,579 player-seasons over 30 seasons, carrying 0.13% of all offensive
  possessions (0-50 possessions alone is 3.9% of players and **0.04%** of possessions).  In 2026 it is 46
  of 582 players.  The criterion is a possession-weighted team-game error, so it can barely see them --
  and a tie on it is the arithmetically expected result, not evidence the correction works.
- **The shipped number for a man with no evidence is already defensible.**  Darius Brown II, 4
  possessions, 2026: -1.45 offence and -0.51 defence, **-1.96 total**, which is about where the
  conventional replacement level sits.  The correction was never needed to keep the bottom of the list
  sane; the incumbent has nobody below -6 per 100 in 2026 without it.
- **Six attempts, nothing reached the bar.**  Rational-weight blend on the output (12, 13), log slope on
  the output (14), a feature (15), rational-weight blend on the prior (16, best z +2.90), the owner's
  linear ramp on the prior (17, best an exact tie at z -0.003).  Not one was better; the best case
  available was "free of harm".

***A trap this cost, worth carrying.***  A replacement level handed to `--blend_off` / `--blend_def` is
**not** the level the player ends up with: the ridge's free prior scale multiplies the prior it is given,
by 2.07 on offence and 1.82 on defence averaged over 30 seasons.  Experiment 17 was run with `x = -1.5`
per the owner's specification and delivered **-4.12 / -3.19, a total of -7.31**, for the four-possession
man -- about double what was asked for, and past even the measured -6.09.  *The check:* after any change
routed through the prior, read the realised number on a real player, not the constant that was handed in.

### Experiment 18: the trade set -- what a player's absence says that his rating did not (2026-09-16)

The owner's design.  A season's rating is fitted on that season's stints, where the same five men keep
appearing together, so how the credit for a good lineup is SPLIT among them is barely identified.  What
identifies it is a player being gone.  So: take the rated season and the two either side, pool every
team-game, give each player a column whose entry is his share of that team's possessions in that game --
zero when he did not play and zero for his old team's games once he is traded -- subtract the rated
season's rating from every row, and ridge what is left back onto the same columns.  That coefficient is
**alpha**.  `src/eracoef/tradeset.py`, `scripts/70_tradeset.py`, `scripts/71_tradeset_features.py`,
`tests/test_tradeset.py`; `scripts/63_yoy.py` gained `--offset=`.

A modern three-season block is **7,880 team-games against about 1,550 player columns**, which fits in
seconds; the whole 30 seasons with a six-point penalty grid runs in about nine minutes.

Every teammate has a column, which is the entire reason this is a regression and not a difference of
averages.  The straight off-court record was tried on 2026-09-14 and rejected because a role player on a
deep team read as good when his team stayed good without him -- the off-court number is his teammates'
quality.  Here the other four are estimated alongside him.

**The rating enters as two free unpenalised columns, one per side (`tradeset.rating_columns`), and that
is not a detail.**  Without them every strong player got a negative alpha and every weak one a positive
alpha, because the rating's amplitude is wrong and a rescale in a per-player costume is what a ridge
will produce.  With them the correlation between alpha and the rating is **-0.001 on both sides**: the
tilt is entirely in the scale and alpha is what is left over.

***What the three seasons say about amplitude.***  The multiplier each side asks for, 1.00 meaning the
rating as it stands:

| | mean | min | max |
|---|---|---|---|
| offence | **0.749** | 0.589 | 0.967 |
| defence | **0.919** | 0.762 | 1.074 |

The offensive rating is about a quarter too wide for the neighbouring seasons to bear out.  Note the
sign flips when the block is the rated season alone (1.03 to 1.12 offence, 1.38 to 1.51 defence): inside
its own games the ridge has over-shrunk, across seasons it has not shrunk enough.

***The penalty, scored on the games TWO seasons away*** (outside the window alpha was fitted on), 56
paired observations, `game_armse` in points per 100 per team-game.  Three arms: the rating as it stands,
the rating with only its amplitude corrected, and the rating with the per-player correction on top.

| penalty | rescale only | rescale + alpha |
|---|---|---|
| the rating alone | 8.8046 | -- |
| 300 | 8.7763 | 9.0222 |
| 1,000 | 8.7564 | 8.7554 |
| 3,000 | 8.7416 | 8.6489 |
| **10,000** | 8.7305 | **8.6328** |
| 30,000 | 8.7250 | 8.6549 |
| 100,000 | 8.7223 | 8.6861 |

10,000 is interior.  Split in two, in game_armse removed: **0.0740 by the rescale alone** and **0.0977
by the per-player correction on top of it**, the latter -2.659 mse at a paired statistic of -20.4,
**56 of 56 observations better**.  **That 0.0977 is UNCONTROLLED: experiment 18b puts one free level per
team into the same fit and it falls to 0.0530.  Quote 18b's number, not this one.**

***The control that makes the number mean something (`--block=0`).***  The corrected rating carries
three seasons of games and the plain one carries a single season, so part of any gap is just the larger
sample rather than anything an absence revealed.  Run the identical machinery on the rated season ALONE
-- no neighbouring seasons, therefore no with-and-without contrast at all, and alpha can only be a
team-game refit of the games the rating already saw:

| | rescale removes | the per-player part removes |
|---|---|---|
| three seasons (the trade set) | 0.0740 | **0.0977** |
| the rated season alone (control) | 0.0566 | **0.0091** |

**Nine tenths of the per-player gain needs the neighbouring seasons.**  Refitting on the season's own
games buys 0.0091 and the absences buy the rest.  That is the claim the trade set makes, and it is the
one number in this entry that would have been wrong without the control.

***The trade loss.***  What the absences say a season's ratings are out by, in points per 100, every
player counting once -- which is the point, since the year-over-year test is a team-game error in which
a 200-possession man is a rounding error.  Pooled over 30 seasons, 403 eligible players a season (one
team all year, 100+ possessions, and games his team played without him):

| side | all | under 500 poss | 500 to 1,500 | over 1,500 |
|---|---|---|---|---|
| offence | **0.328** | 0.171 | 0.229 | 0.354 |
| defence | **0.281** | 0.146 | 0.200 | 0.302 |

Weighting each player by the effective sample of his own contrast raises those to 0.364 and 0.310.
Dropping the two one-sided seasons (1997 and 2026 have one neighbour) moves them to 0.332 and 0.283.
Alpha's spread is 0.412 offence and 0.351 defence against the rating's own 1.669 and 1.122, so this is a
correction of roughly a quarter of the rating's spread, not a rewrite.

***Which statistics see it (`scripts/71_tradeset_features.py`).***  Alpha regressed on the season panel's
Boruta features plus the rankings' own rating and the panel's `rapm1`, chimeraboost, five
out-of-player folds, 12,103 player-seasons a side.  (The second of those two was called "prior"
in scripts 71/72 and written up here as the rankings' prior.  It is `rapm1` -- the role prior plus
the season's own residual -- not the box prior the rankings are centred on.  Name corrected
2026-09-17; the numbers below are unchanged, they were always measured on `rapm1`.)  Weighted out-of-fold share of alpha accounted for:

| side | every feature | without the how-much-he-played features | the rankings' own two numbers |
|---|---|---|---|
| offence | 0.0686 | **0.0501** | 0.0115 |
| defence | 0.0770 | **0.0607** | 0.0217 |

**The middle column is the honest one.**  Career possessions is the top feature on both sides, and
on-court possessions carries the largest ridge coefficient on offence (-0.43 per standard deviation).
That is measurement trap 8 firing again: alpha is noisier and more shrunk for a player with less
exposure, so a feature naming those players predicts alpha's NOISE without knowing anything about
basketball.  Removing the whole family costs about a quarter of the fit and leaves 5.0% and 6.1%.

Against the rankings' own rating and prior at 1.2% and 2.2%, **a season's statistics see four to five
times as much of the correction as the rankings themselves do**.  That is the result the trade set was
built to produce: there is signal in the box score and the play-by-play that the current prior is not
using, and this is the first instrument in the project that can point at it per player.

Ranked feature tables for both sides are in `outputs/tradeset_features.parquet` and printed in full by
the script; every feature is labelled with whether it measures how much a player played, how he played,
or is one of the rankings' own numbers.

***2026.***  The correction is small at the top and the order barely moves: Wembanyama, Jokic and Kawhi
stay the top three.  The largest corrections go to players whose teams played a great deal without them
-- Chris Paul -2.02 on 456 possessions played against 8,688 his team played without him, Nick Smith Jr.
-1.66, Myles Turner +1.32, Ty Jerome +1.34.  Pascal Siakam is the largest positive at +1.71 on only 448
possessions of absence, which is the shape to be careful of: a short contrast is a loud one.

***What this is not.***  Alpha reads the games of the seasons either side, so it can never enter a
published rating (ruling 1).  It is a diagnostic and a training target.  When it becomes the prior's
target, the prior for season H must be trained on other players' rows only (the existing out-of-player
folds) and, for the strict check, on alphas from seasons at least three away from H so their windows
never touch H-1 or H+1; run the candidate both ways and if the gain disappears with the buffer it was
the leak.  Nothing here rewrites any shipped artefact: every output is a new `tradeset_*` name.

### Experiment 18b: team terms, and how much of the trade set was really team quality (2026-09-16)

The owner asked for this before anything else, and it was the right call: **about half of the
per-player gain reported in experiment 18 was the team, not the player.**

A team's output is the sum of its players' columns, so with nothing standing for the team a player's
alpha can absorb "his team was good that year."  That is exactly the channel that sank the off-court
record on 2026-09-14.  `tradeset.team_columns` frees one column per team on offence and one per team
on defence, unpenalised, in two strengths (`--team_effects=team|team_season`):

* **per team, pooled over the three seasons** -- removes persistent team quality: the franchise, the
  coaching, the building.  A team's change from one season to the next still identifies the players
  who moved.
* **per team and per season** -- the strict bound.  Each team's level in each season is free, so a
  player is identified only by variation in who was on the floor inside that team-season.  The season
  intercepts are dropped under this one, since each team-season column already carries its season's
  level.

Two sets of columns and not one, because a team's scoring and its defending are separate facts.

***The penalty grid, all arms at the same six penalties, 56 paired observations, scored two seasons
out.***  Every arm chose 10,000 and every choice is interior.  In points per 100 of `game_armse`
removed, split into the part a single multiplier per side buys and the part the per-player correction
buys on top of it:

| arm | the rescale | the per-player part | paired statistic | seasons better |
|---|---|---|---|---|
| no team terms (experiment 18's headline) | 0.0740 | **0.0977** | -20.4 | 56 of 56 |
| one level per team | 0.0685 | **0.0530** | -10.9 | 53 of 56 |
| one level per team per season | 0.0647 | **0.0234** | -6.0 | 44 of 56 |
| the rated season alone, no team terms | 0.0566 | 0.0091 | -5.8 | 43 of 56 |

**The correction survives both controls and it is half the size.**  46% of it was team quality under
the pooled-team term and 76% under the strict one.  Experiment 18's 0.0977 should be read as an
uncontrolled number; **0.0530 is the defensible one.**

***Which arm to believe.***  The pooled-team term.  The strict version also absorbs a player's own
standing relative to his teammates within a season -- that is partly a team-season fact, so it
removes real signal along with the confound -- and it should be read as a lower bound, not the
answer.  The pooled term removes the thing the off-court record actually died of and leaves the
cross-season contrast intact, which is the contrast the trade set is built on.

***The finding that MATTERS survives, and barely moves on defence.***  What a season's statistics can
see of the correction, out of fold, with the how-much-he-played family removed:

| side | no team terms | one level per team | the rankings' own two numbers, team terms on |
|---|---|---|---|
| offence | 0.0501 | **0.0367** | 0.0081 |
| defence | 0.0607 | **0.0556** | 0.0172 |

So with team quality taken out, the box score and the play-by-play still see **four and a half times**
what the rating and prior see on offence and **three times** on defence.  Defence loses almost nothing
(0.0607 to 0.0556); offence takes the larger cut, which is consistent with offensive team quality
being the more persistent of the two.  **The reason to have built the trade set stands.**

The trade loss shrinks in step, as it must, since alpha is smaller: offence 0.328 to 0.300 and defence
0.281 to 0.256 pooled, same tier ordering.

`outputs/tradeset_team_*` is the pooled-team arm and `outputs/tradeset_ts_*` the strict one;
`outputs/tradeset_*` without a suffix remains the uncontrolled arm and should not be quoted on its own.

***A test premise that was wrong, recorded because it is the natural guess.***  I first asserted that
under per-team-season terms a player who never leaves the floor would keep no correction, being
collinear with his own team's column.  He keeps one: his possession SHARE still varies game to game,
and more appearances means less shrinkage, which dominates.  The property that does hold, and is what
`tests/test_tradeset.py` now checks, is that pooling each team-season's lineup correction over its own
games moves closer to zero once the team has a column of its own -- the team-level part of alpha goes
to the team.

### Experiment 18c: which statistics move a correction, and by how much (2026-09-16)

`scripts/72_tradeset_shap.py`, on the team-controlled alpha (`outputs/tradeset_team_alpha.parquet`).
Exact interventional TreeSHAP from chimeraboost, computed OUT OF FOLD -- each player's contributions
come from the fold model that never saw a row of his -- in the model's additive space, which is
alpha's own units.  **Every number here is points per 100 possessions of correction: how far a
statistic moves a player's rating away from what his own season's games said.**

Three columns per statistic.  `moves_typical` is the mean absolute contribution: how much it moves a
player at all.  `low` and `high` are the mean SIGNED contribution among the bottom and top fifth of
the statistic, which is the only readable form of direction -- an importance rank cannot say which way.

***Offence, the eight that move a player most:***

| statistic | moves_typical | low fifth | high fifth | measures |
|---|---|---|---|---|
| `onc_d` (opponent points while on court) | 0.027 | -0.035 | +0.050 | how he played |
| `exp_poss` (career possessions) | 0.022 | 0.000 | -0.042 | **how much he played** |
| `onc_o` (own points while on court) | 0.019 | -0.036 | +0.016 | how he played |
| `p3r` (three-point rate) | 0.019 | -0.032 | +0.019 | how he played |
| `poss_pct` (possession share) | 0.018 | +0.019 | -0.028 | **how much he played** |
| `weight` | 0.017 | -0.013 | +0.024 | how he played |
| `creation` | 0.013 | -0.019 | +0.025 | how he played |
| `orbsh` (offensive rebound share) | 0.012 | -0.014 | +0.023 | how he played |

***Defence, the six that move a player most:***

| statistic | moves_typical | low fifth | high fifth | measures |
|---|---|---|---|---|
| `exp_poss` | 0.030 | -0.026 | +0.031 | **how much he played** |
| `onc_o` | 0.029 | +0.041 | -0.062 | how he played |
| `gs_pct` (games-started share) | 0.013 | -0.007 | +0.024 | **how much he played** |
| `height` | 0.011 | -0.014 | +0.018 | how he played |
| `stocks` (steals plus blocks) | 0.010 | -0.015 | +0.018 | how he played |
| `drb` | 0.007 | -0.010 | +0.013 | how he played |

`rating` and `prior` -- the rankings' own two numbers, handed in as features on purpose -- move a
player 0.013 and 0.012 on offence and 0.025 and 0.029 on defence, and both push DOWN at their high end
(offence `rating` high -0.034; defence `prior` high -0.046).  That is the amplitude story showing up
per player: the higher the rating, the more the absences take back.

***Three findings.***

1. **Shooting threes, creating shots and offensive rebounding all push a correction UP**, each about
   0.02 per 100 at the high end.  The rating under-credits the players who do those three things and
   the trade set adds it back.  This is the most directly actionable thing the trade set has produced.
2. **On-court plus-minus is the loudest real statistic on both sides, and on DEFENCE it points the
   wrong way.**  A player whose team scored most while he was on the floor has his defensive
   correction pushed DOWN 0.062.  The prior is reading team offence as defensive credit.
3. **`exp_poss` is near the top on both sides and it is the warning, not the finding** (measurement
   trap 8).  A longer career means a better-MEASURED correction, so the feature predicts alpha's noise.
   Totals, summing `moves_typical` by family: the how-he-played statistics move a typical player
   **0.158 on offence and 0.118 on defence**, the how-much-he-played family **0.060 and 0.073**.  So
   two thirds of the real movement on offence is basketball and only three fifths of it on defence.

***Per player, 2026, and this is the number that sets the ceiling.***  The largest corrections beside
how much of each the statistics account for: Westbrook -1.52 against -0.06, Chris Paul -1.42 / -0.29,
Amen Thompson -1.29 / -0.23, Dort -1.25 / -0.05, Markkanen +1.16 / +0.44, Durant -1.15 / -0.10, Trey
Murphy III +1.04 / +0.47.  **Most of what an absence reveals is not in the box score at all**, which is
the same thing the 3.7% and 5.6% shares say, read one player at a time.

***A bug caught before the numbers were quoted.***  Melting the per-player frame repeated each
player's correction once per statistic, so summing it multiplied every correction by the number of
features -- 42 across the two sides.  The first run read Westbrook at -34.39 instead of -1.52 and the
error was invisible in the ranking, which was correct throughout.  The script now carries the
correction and the prediction as one row per player per side, never melted.

### The trade set is a diagnostic, not a product: the owner's verdict (2026-09-16)

Shown the corrected ratings, the owner: ***"basically no movement."***  **Not shipped.**

**The owner's measurement rule, given in the same breath and adopted:** *"anything that's within 0.1
is 'nothing'."*  Report how far a change moves players in **points per 100**, not in rank places -- a
rank move has no size, and the middle of a list can be packed tightly enough that a large one is a
rounding error.  Applied here it **overturns the dismissal it was offered to support.**

Absolute change against the incumbent, 12,103 eligible player-seasons with absence evidence, 30 seasons:

| | median | ninth decile | largest | over 0.1 | over 0.25 |
|---|---|---|---|---|---|
| the per-player correction alone | **0.273** | 0.784 | 2.43 | **79.1%** | 53.2% |
| the uniform rescale's part | 0.381 | 0.909 | 3.88 | 83.3% | 64.2% |
| both together | 0.512 | 1.244 | 4.61 | 89.7% | 74.4% |

Per-player part by exposure: under 500 possessions median 0.130, 58.7% over the threshold; 500 to
1,500, 0.244 and 79.1%; 1,500 to 3,000, 0.306 and 82.8%; over 3,000, 0.356 and 85.0%.  **Four fifths of
players move more than the threshold and the movement grows with exposure**, so it is not a bench
artefact.

The packing density is an empirical question and the guess was wrong: near the middle of a 2026 season
**0.84 points per 100 spans 20% of the players**, so the men who moved 18 to 26 places moved a median of
**0.243** points per 100.  What IS true is that the top does not move -- the 2026 top five is the same
five men reordered, the top 20 keeps 15.7 of its 20 names over 30 seasons, rank agreement 0.942,
Kendall 0.802, and 0.947 among players with 1,000+ possessions.  Nothing happens where the eye test
looks; the middle of the list is rearranged by amounts that clear the owner's own threshold.

**Why it does not ship, and the reason is not the size of the movement.**  Alpha reads the seasons
either side of the rated one, so it can never be a published rating (ruling 1).  The route to shipping
was to teach the prior what alpha knows, and a season's statistics see **3.7% of alpha on offence and
5.6% on defence** out of fold.  The correction is real and it is mostly invisible to the box score.
Per player: Westbrook -1.52 with the statistics seeing -0.06, Dort -1.25 against -0.05, Durant -1.15
against -0.10; the best are Trey Murphy III +1.04 against +0.47 and Markkanen +1.16 against +0.44.

**What the line produced that outlives it**, in order of what it should change:

1. The offensive rating is about **a quarter too wide** across adjacent seasons (multiplier 0.749
   offence, 0.919 defence), worth 0.069 of game error on its own, and the sign REVERSES inside a single
   season (1.03-1.12 and 1.38-1.51).  Within its own games the ridge over-shrinks; across seasons it
   does not shrink enough.  That is a live, unexploited finding about the incumbent.
2. Three statistics the rating under-credits, each about 0.02 per 100 at its high end: **three-point
   rate, shot creation, offensive rebound share**.
3. A named defect: on defence the prior **reads team offence as defensive credit** (on-court plus-minus
   pushes the defensive correction down 0.062 at its high end).
4. The first per-player loss in this project that counts a bench player the same as a starter (the trade
   loss: offence 0.300, defence 0.256, against a rating spread of 1.67 and 1.12).

**How to use it from here:** as an instrument, on candidates the team-game criterion calls ties.  Do not
adopt alpha as a rating, and do not train the prior on it until the 3.7% and 5.6% are raised.
Experiments 18, 18b and 18c above have the full record; `HANDOFF.md` has the commands and the traps.

## What was tried and rejected

**The LRBoost branch (a boosted correction on a frozen linear prior).** Five things had to be right before it
did anything: pooling across all ten windows (shrinkage 0.26/0.03 within one window, 0.87/ 0.59 pooled),
weighting by the shrinkage diagonal, stripping the 8% of the linear span the plug-in fit leaves in the
residual, freezing playing time the way the scorer does (this alone dropped the offensive permutation null
from 0.42 to 0.00), and keeping playing time on defense and off offense. It then won held-out stint MSE by
+0.278 ± 0.090 (3.1 SE) and the next-window criterion by +0.0129. On the ladder against the consensus it is 0
on the total and -0.020 on offense. Off — but the offset machinery stays, because a correction must be carried
INTO the fit, never added afterwards.

**The xPTS stage-4 shot term, and the shooter-level version after it.** The full geometric closure (condition
on the possession's first attempt, marginalise everything downstream with the lineup's four cross-fitted
factor rates) was built and it CLOSES: expected attempts within 0.4% of observed on all 112 training blocks,
points ratio 0.994-0.999, no clipping. It loses at every `c_def`, both K, both variants: +0.32 at game level
(z +1.8, 9 of 28) and +1.02 at stint level. A diagnostic with league make rates instead of lineup ones loses
by +1.96, so shot-making is about 15% of everything the ratings know and the lineup eFG fit recovers only 80%
of it. The shooter-level rebuild (his own padded other-half 2P%/3P% times a per-season shot-location curve)
loses by more, +1.5 to +2.4, and the padding constants say why: k = 175 attempts for 2P%, 226 for 3P%, against
24 for FT% — a heavily padded shooter rate IS mostly the league rate. The location curve itself is worth 0.7
per 100 and is kept.

**The four-factor defensive fit.** Opponent eFG%, turnovers forced, offensive rebounds allowed and free-throw
rate allowed, each solved on the points fit's own layout with its own ridge, the points prior shared out by
zero-prior slopes and recombined with the row-level gradients (eFG 1.55, TOV -1.04, OREB 0.63, FTR 0.32 points
per 100 per point of rate, R-squared 0.88). Twelve forms; the best is -0.040 at z -0.94 and 7x slower, and
re-run on the current board it is +0.005 / -0.010, exactly nothing. Where the value actually is: **per-factor
shrinkage WITHOUT a prior is worth -0.11 (z -1.44); the fixed-share prior split costs more than that**,
because a share puts a big's prior into rebounding-points and a guard's into foul-points the same way for
everyone. Per-factor PRIORS were never built. Splitting eFG by zone was killed before it was built: REML's
four ratios match an independent split-half estimate (eFG 1.50 against 1.50), so the constants were already
right.

**The team-game leave-one-out luck adjustment — it works at team level and fails at player level.** Every rate
a possession's points depend on has a measured skill share on each side, nothing is a chosen constant, and the
adjustment was tested directly: first half of a team's season predicts its second, 1,784 team-seasons, no
ratings pipeline in the way, scored so that generic shrinkage cannot win. **It removes 6.6% of the forward
error on offense and 10.4% on defense**, three-point shooting being almost the whole of it (-0.357 and
-0.568), while shrinking a STYLE rate (rebound chance or field goals per attempt) COSTS 0.09-0.17 at z 3-4. In
the ratings none of it survives: team-level free throws recover 0% of the shipped shooter-level gain (k = 297
attempts against 24, so a 90% and a 60% shooter price identically), the two-point rung costs +1.63 mapped
because a defence controls 0.85 of two-point percentage, and the three-point rung wins the search half at
-0.170 (z -2.8) and gives back two thirds on the confirm half (-0.055, z -0.90) while being significantly
WORSE at stint level there. `src/eracoef/teamloo.py` is kept as a self-contained single-season instrument.

**The Dredge play-by-play counter block.** Twenty-one event types per player-season, twenty-six padded
features, validated to 0.0000-0.0024 against the box score and to 0.00e+00 between the panel and prediction
paths. **Seven candidates, best z -0.20, worst z 2.33.** The mechanism is reliability: year-over-year, `blk`
per 100 scores 0.918 and the Russell SHARE — Dredge's headline decomposition — scores **0.126**. It is 0.575
for everybody and 98% of its spread is sampling noise. Dredge's strong claim, that the split should REPLACE
`blk`, reverses: dropping `blk` costs the defensive prior +0.328 weighted MSE, the largest effect of any
single column here, and its decomposition recovers 87%. A decomposition is worth having when the model cannot
make it, and a depth-4-to-7 booster over 23 to 43 crossed features can. The assist-by-zone extension
(`pot_ast`, year-over-year 0.922, -0.018 on the prior's own offensive fit) reads +0.0005 at z 0.12.
`OffFoulsDrawn100`, Dredge's best published coefficient, is not buildable from the v3 feed and is untested.

**Schedule context.** Rest days and back-to-backs, declined by the owner — *"evens out really well."*
Independently, removing the shooting team's own leave-one-out three-point rate from every team-game moves the
measured defensive three-point spread only from 0.59 to 0.58 points.

**The playoff delta, as anything other than a separate view.** A rating can be the prior for another rating:
the regular-season number becomes the OFFSET for a fit that sees only playoff possessions, so that fit's
residual IS the delta and no playoff box score is needed. Penalty chosen on five blocks and read on the other
five: x2 the regular-season penalty, **-0.459 per 100 at z -2.16 on prediction and -0.508 at z -2.27 on
attribution**, surviving a per-side slope refit, so it is ranking and not amplitude. But the delta's spread is
0.21 against 2.43 for the ratings — a playoff run moves a player a tenth of the distance between players — so
it changes no board and ships only as its own view. Two bugs in shared code had to be fixed first: the
exposure filtered to `phase == "RS"` always, so a playoff-only design had zero exposure; and the unpenalized
fixed block goes singular on a one-phase design (`is_po` constant, `po_home` a copy of `home`), returning
solves around 1e12.

**Career-experience features.** Seasons played, career possessions and entry age, counted before the block's
first season. **-0.077 weighted MSE on the prior's own leave-window-out fit — the largest gain any block ever
posted there, -0.247 on the low-exposure rows — and +0.054 on the criterion at z +1.65, 11 of 28.** See the
fifth trap; this is its purest instance.

**Also rejected, with the number.** Squared lineup-sum terms (diminishing returns): unanimous in sample,
0.3-0.4 bp worse out of sample, rank correlation 0.998. Single-season training panels for the prior: +0.52 at
z 4.4 — trained on one-season feature lines and applied to three-season ones, the defensive prior loses a
third to a HALF of its spread. Destination/"traded-to" features as the prior: -0.072 on the criterion and
+0.125 on the investigator, so kept as the trade-to instrument. The per-player trade delta applied to H:
+0.069. On-court ORTG/DRTG in the prior: +0.05 at z 2.22. A recursive prior-informed RAPM converges worse
(0.786 -> 0.765): refitting beta on `Rbeta + u` launders shrunk residual into unshrunk prior.

## The measurement traps

**A player-level loss can be gamed by the zero point and by the truth's own shrinkage.** Two separate ways
the same new instrument produced a confidently wrong number on the day it was built. (1) Dollars are a rating
times possessions, so an additive shift the team-game criterion cannot even see -- it refits the intercept --
made a true-talent board score WORSE than a near-useless one, because every player being positive orders them
by minutes. (2) The truth is itself a ridge fit, and a heavily shrunk truth pulls low-possession players
harder than starters, which rewards a board that does the same -- and it bites the WITHIN-ROSTER rank hardest,
because a coach plays his better players more and minutes are most of the within-team signal to begin with.
The calibration map reads z -6.82 against the default truth and z +1.35 against a nearly unbiased one. *The checks:* both sides are re-centred before the
conversion, and a shift-invariance test pins it; and every player-level verdict is read at two truth lambdas
(`53_calmap.py --players --truth-lam=100`), with anything that changes sign between them treated as undecided.

**`rapm1` is 97.6% `spm`, so r-squared against it is not skill.** The role panel's shipped target is
`rapm1 = spm + u`, and `spm` is a leave-window-out LINEAR ridge of APM on the role inputs -- which are
sitting in the booster's own feature list. Correlation of `rapm1` with `spm` is **0.976** (offense, season
panel); `apm` has sd 4.19 against `rapm1`'s 1.62 and `u`'s 0.35. A GBDT handed those features "predicts"
`rapm1` at r = 0.99 and has relearned an equation, not the game. This is not a reason to stop using
`rapm1` -- the booster's job is to beat `spm`'s linear FORM, and it does -- it is a reason never to read a
fit statistic against it. *The check:* report `corr(pred, apm)` beside it (~0.53 on the same fit), or skip
to the criterion and the player losses, which are the only things that decide anything.

**A team-game score on a mask that cuts team-games is not a score.** The criterion sums a team's points over
its rows in a game. Restrict it to a subset of stints and that sum becomes a partial point total compared
against a level fitted on complete games -- three times the pooled error on the shipped board (337.6 against
113.6), and enough to flip the sign of a map comparison and manufacture a z of -2.16 that is not there. Every
split in `holdout.SPLITS` except `game_bench_share` labels a STINT, so `tg` is NaN for all of them now
(`_keeps_whole_team_games`) and they must be read on `mse`. The check: **do the group scores recombine to the
pooled score?** If they do not, the group is not a piece of the thing you are pooling.


**An in-season fit scored on the whole season is scored on its own training games.** `scripts/53_calmap.py
fit` built its scoring frames with `load_frames(...)` and no cut, while `54_track.py` and `57_investigate.py`
had always passed `cut_of(dump, system)`. A system fit on the first 75% of H was then scored on 100% of H. The
leak is proportional to how much of the rating comes from H, so it favours the SHORTEST kernel: the
single-season kernel read 105.31 against the three-season kernel's 106.06 and won 28 of 28 seasons at z 9.3.
Scored on the games after the cut, the same fits read 113.76 against 113.24 and the single-season kernel LOSES
19 of 28. The sign of the headline comparison was the leak. *The check:* `calmap.frames_for_dump` now takes the
cut from the dump and refuses a run that mixes cuts; `tests/test_calmap_cut.py` pins it, including that the
script never calls the uncut loader.

**Benchmarking a model against another model built from the same data.** The board's tilt toward big men was
certified as "evidence, not bias" by regressing the shipped rating on the same player's NEXT window pure
on-court RAPM: the bigness term came out **+0.045 at z +4.7**, and the conclusion was published. The benchmark
was built from the same margin data with the same blind spot and could not catch a shared error. Against an
external blend the same board had Robert Williams 25th to the consensus's 134th, Nurkic 21st to 166th, Trae
Young 313th to 79th. *The check:* score against something that does not share the estimate's construction —
actual points in a season the model never saw.

**Calibration on a lineup sum is blind to attribution.** A stint row observes only the lineup SUM of player
effects, so reallocating credit between teammates barely moves it. Swept over the defensive prior weight
`c_def`, held-out stint MSE **prefers the wrong end at 3.4 standard errors while moving 0.065% in absolute
terms**, choosing 1.0 over the 0.0 that raises player rank agreement from 0.78 to 0.89. Letting the stint
regression estimate its own trust in the prior returns 1.12 on offense and 1.06 on defense — trust the box
score MORE — and moves Robert Williams 25th to 12th. *The check:* ask which subspace a tunable moves; in the
design's null space no in-sample loss can select it.

**An argmax on a grid boundary has chosen nothing.** The four-factor OREB fit first selected `lam_ratio =
2.0`, the top of the configured grid; widening it moved the answer to 3.0, interior. The same failure produced
a whole ladder row's defensive knobs, where the internal criterion was flat across the four weakest penalties
(0.4789 to 0.4813) and then declined monotonically — no interior optimum, so it picked whichever weak-end grid
point won a coin flip. *The check:* verify the argmax is interior with both neighbours worse, and say so. The
in-season kernel and the single-season penalty were both accepted on exactly this evidence.

**A target's gates cannot tell you whether it should be the target.** The xPTS shot term closed to within 0.4%
on all 112 training blocks with a points ratio of 0.994-0.999 and no clipping, and lost by up to +1.7 per 100.
The team leave-one-out target at zero shrinkage returns actual points per team-game **to 1e-13** while
changing every stint of that game, because it has spread the game's makes over its attempts — its gates pass
everywhere and it has erased which LINEUP did the shooting. *The check:* a gate that aggregates cannot see
what the target did to the rows the ridge reads. Only out-of-season prediction against ACTUAL points decides a
target.

**A feature can predict how well-MEASURED a row is rather than how good the player is.** The prior's target is
a player's value pooled over his OTHER windows, so a player with more windows has a lower-noise target and is
intrinsically easier to predict. Career experience names exactly those players: -0.077 offline, the best block
ever measured there, and +0.054 on the criterion. Fine height and weight are the purest form — the pair names
a player almost uniquely, is -0.27 on the offensive prior's own fit and +0.20 on the criterion at z 4.7;
binned, it keeps 0.124 of 0.145 on defense (physiology) and 0.031 of 0.272 on offense (identification).
Boruta, selecting against that same target, ACCEPTED the career block. *The check:* bin any static player
attribute and read the binned-versus-fine gap as the identification share. Splitting by exposure does NOT
catch it.

**A knob measured at a cheap operating point does not transfer to the shipped one.** A Huber loss is -0.039
weighted MSE at the cheap booster and **+0.061 at `quality=4`, the booster that ships** — the bag and the
model-selection search already buy what the robust loss was buying. The same applies to the base list: the
play-by-play block read -0.050 on the 43-name offensive line at `quality=4`, which would have been the largest
feature gain ever measured here, and -0.012 on the shipped 23-name defensive list with the shipped defensive
booster. Same features, opposite conclusion. Parts interact in both directions: cross features are harmful
alone (+0.004) and the best thing available beside linear leaves (-0.013 together). *The check:* bench at the
operating point it would join, booster and base list both.

**An unmapped criterion gain can be entirely absorbed by the shipped calibration map.** The offensive
three-point target read **-0.21 per 100 unmapped and -0.024 at z -0.63 once the map was fitted**, because the
whole of it was offensive amplitude (`scale_off` moved 1.06 to 1.15) and a per-side calibration curve is
exactly what corrects amplitude. The audit quantified the absorption: 99% of the offensive three-point
target's effect, 44% of the free-throw target's, 29% of the two-point disaster's, 33% of the on-court prior's.
*The check:* run `54_track.py`, which fits the shipping map leave-one-season-out, and pair the systems before
quoting any number as a gain — never `45_holdout.py` alone. Near-total absorption is itself diagnostic: the
candidate was only rescaling a side.

**The team-level and player-level luck tests are allowed to disagree, and must both be run.** Shrinking the
outcome rates removes 6.6% and 10.4% of a team's forward error; handing a defence back its real three-point
signal in the ratings is worse, up to +0.248 at z 3.25 at full replacement. A defence's 0.59 points of true
three-point spread is a TEAM property, and the ridge's job is to split it among five players against 1.05
points of season noise. *The check:* a luck adjustment needs the team-level forward test AND the criterion;
neither substitutes, and only the second decides a rating.

**An identical-paths check proves the two paths agree, never that either is right.** The play-by-play feature
builder read the block's league totals from the frame's FIRST row, so with all ten windows handed to it at
once every row was padded toward 1997-1999 — the exact era contamination the block was designed against. The
comparison reported 0.00e+00 throughout because it calls the same function on both sides; every bench number
moved 0.005 to 0.02 when it was fixed. *The check:* anything with a per-block constant needs a test that puts
two eras in ONE frame.

**A covariate measured on the held-out season needs a control from data the outcome could not have touched.**
A player's share of his team's possessions AT H was the largest single map gain ever measured (-0.19, z -3.7);
the same share on HALF of H's games is worth -0.006, keeping 3% of its value, so the whole of it was knowledge
of who was good that season. The team-game's mean log stint length reads -0.149 and its leave-one-game-out
cousin +0.008: substitutions follow the score. Teammate turnover survived the same control (-0.084 against an
identical -0.084) and is real. *The check:* the other half of the season, the team's other games, or the
training block.

**A search-half gain at z -2.8 can be two thirds noise.** The team leave-one-out three-point target read
-0.170 (z -2.8, 11 of 14) on the search half and -0.055 (z -0.90) on the confirm half, while being
significantly WORSE at stint level there. *The check:* the 14/14 alternating-seasons protocol — choose on the
search half, read on the confirm half, and read the stint column even when game level decides.

**Two more worth carrying.** A player-level split-half correlation rejects a defensive residual and never
selects its ridge: tightening the factor ridges 16-32x brought the factor sum level with the points residual
player by player and cost +0.50 and +0.83 on the criterion, because the criterion values team-coherent content
a correlation across players cannot see. And Boruta cannot gate a collinear list: given `stocks` = `stl +
blk`, `blk` fails against its own shadow — accepted on offense, rejected on defense, and swapping the two
costs +0.050.

## Standing rules

**Accuracy wins, provided the testing is robust — and the model has to stay shippable as open source.** The
owner, 2026-09-06: *"The speed thing is more just like I don't want us to build some insanely complex model
that is overfit and too slow, because I want this to be open source! but ultimately accuracy is the winner,
provided the testing is very robust."* So fit every row and score every game; fit time is a number the chart
carries, not a term to optimise, and a real accuracy gain is never traded away for seconds. "Robust" is the
precondition, not a nicety: a gain that survives only the search half, or only the prior's own fit, or only
one map, is not a gain.

**Between two candidates the criterion cannot separate, take the simpler and faster one, every time.**
Complexity and fit time are a tie-break, and a strong one — somebody else has to be able to run this. It
shipped the Boruta prune, the simpler kernel from a plateau of four, offense-only PAST, and the incumbent
board over every candidate inside noise.

**The consensus is a sanity check, never a fitting target.** The owner, 2026-09-06: *"disagreeing with
consensus is just a sanity check, never something to fully fit to."* The floors in
`tests/test_vs_consensus.py` say the same of themselves — they *"guard against a further fall, not the old
level."* So never choose a model constant because it clears a floor: a defensive pooling constant worth -0.067
was deliberately NOT read against them, because choosing the value that just clears a gate is fitting the
gate. **A marginal miss is not a veto** — 0.7592 against a 0.76 floor is noise on a sanity check and does not
overturn a z -3.02 result. **A gross miss still is**, because that is what the check is for: an 8% overshoot
on the archetype gap is exactly what the consensus exists to catch. A floor is re-based once, deliberately,
with the reason written into the test, and the model constant is not touched. The criterion decides; the
consensus sanity-checks.
