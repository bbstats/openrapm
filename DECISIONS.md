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

### Experiment 19: the trade loss re-judges `onc_d` off the defensive list (2026-09-18)

`onc_d` off the defensive feature list (`season_ratings_sy_noonc_d.parquet`) has been the one candidate
left unresolved since the overnight queue of 2026-09-14: better on the year-over-year test in both
directions (8.684 against experiment 4b's 8.693; -0.24 at z -2.6 at native scale, -0.36 at z -3.8 with
the spread removed) and 8% under the consensus defensive check (0.689 against 0.753), so it was left for
a ruling rather than adopted, because the consensus is the attribution check and this is an attribution
change.  Two things arrived afterwards that bear on it: the trade set's finding that the defensive prior
reads team offence as defensive credit, which is the same defect with a named mechanism, and the trade
loss itself -- the first per-player loss in this project that counts a bench player the same as a
starter.  Nothing new was fitted; three trade set arms and a component split, about 20 minutes.

**The trade loss declines to support it.**  `scripts/73_tradeloss.py` compares two rankings' corrections
on the exact intersection of eligible player-seasons (24,206 player-season-sides, 12,103 per side, 30
seasons, no rows dropped from either arm), paired by season, against the baseline it was built on:

| against 4b | side | trade loss, candidate | 4b | mean difference | z | seasons won of 30 |
|---|---|---|---|---|---|---|
| `sy_noonc_d` | offence | 0.3002 | 0.3003 | -0.0000 | -0.06 | 17 |
| `sy_noonc_d` | **defence** | **0.2576** | 0.2582 | **-0.0006** | **-0.40** | 16 |
| the shipped penalty 13,037 | offence | 0.3003 | 0.3003 | -0.0000 | -0.00 | 13 |
| the shipped penalty 13,037 | **defence** | **0.2558** | 0.2582 | **-0.0024** | **-2.19** | 21 |

The offensive row is a tie by construction -- the change is defensive -- and the defensive row is a tie
by measurement.

**It is not a power problem, and the bottom two rows are a finding of their own.**  The same instrument,
on the same players and the same pairing, reads the fixed residual penalty of 13,037 as a real defensive
gain over 4b at z -2.19, 21 of 30 seasons.  **That change was adopted on 2026-09-15 as a TIE on the
team-game test** (8.697 against 8.693).  So the trade loss can see a defensive improvement of 0.0024
points per 100 at z -2.2 and sees 0.0006 at z -0.4 here; and the first candidate it has been pointed at
after the fact turns a team-game tie into a measured gain.  That is the case for priority item 3 -- 
re-judging the candidates already on disk -- made by example.

**The consensus drop is NOT the `def3` pattern, and this is what decides it.**  The standing way to read a
fall in consensus defensive agreement is to split it by component first, because the blend is 90%
raw-points metrics: when the opponent-three-point adjustment cost 0.888 -> 0.814, every raw-points
component agreed less and every luck-adjusted one agreed MORE, which is what a luck adjustment should
do.  Here, 2026, players over 1,000 possessions, 391 matched, Spearman against `rating_def`:

| | xDRAPM | pred_depm | DRAPM | td_drapm | td_ladrapm | LA_DRAPM | DLEBRON |
|---|---|---|---|---|---|---|---|
| shipped | 0.747 | 0.807 | 0.701 | 0.720 | 0.685 | 0.681 | 0.814 |
| 4b | 0.745 | 0.809 | 0.709 | 0.731 | 0.692 | 0.685 | 0.831 |
| `sy_noonc_d` | 0.673 | 0.739 | 0.573 | 0.652 | **0.594** | **0.543** | 0.774 |

Agreement falls against every component, and it falls FURTHEST against the luck-adjusted ones
(`LA_DRAPM` 0.685 -> 0.543, `td_ladrapm` 0.692 -> 0.594) -- the opposite of the `def3` signature, and
against exactly the metrics the owner keeps the file for.  (`xDLEBRON` is empty in
`data/external/impact_metrics_2526.csv` and is not readable.)

**The 2026 top 20 is not worse to the eye; it is arguably better.**  Defensive specialists rise --
Draymond Green 77th to 10th, Derrick White 32nd to 13th, Rudy Gobert 23rd to 15th, Donovan Clingan 41st
to 20th, LeBron James 33rd to 6th -- and the top three (Wembanyama, Jokic, Kawhi) do not move.  What
leaves is Shai Gilgeous-Alexander 4th to 12th, Cade Cunningham 20th to 48th and Ajay Mitchell 9th to
38th.  The defensive spread over players with 2,000+ possessions rises from 1.36 to 1.49.  So the eye
test does not veto, and the two attribution instruments disagree with each other about the same table.

**The movement is real in size**, by the owner's own threshold.  Neither 4b nor `sy_noonc_d` is centred
-- both were built on 2026-09-14, before the possession-weighted centring rule, and 4b's 2026 table sits
at +1.61 on offence and +0.33 on defence -- so a raw difference between them carries a constant, and the
level shift (+0.206 on defence in 2026) has to come out before the movement is read.  Level removed, the
defensive rating moves a median of **0.365** points per 100 in 2026, ninth decile 0.937, largest 2.108,
with 82.8% of players over the 0.1 threshold; over all 30 seasons, shift removed season by season,
median **0.300**, ninth decile 0.823, 82.0% over the threshold.  (The raw figures, 0.391 and 88.0%, were
quoted first and are 7% high.)  This is not a cosmetic change either way.

**Recommendation: keep `onc_d` on the defensive list; do not adopt.**  The only criterion that prefers it
is the one that cannot see the bench.  The loss built to see the bench is silent, and agreement with
every public defensive metric -- including the luck-adjusted ones that are the closest proxy to the
target, and which a defensible change should move TOWARD -- falls by 0.04 to 0.14.  Ties go to the
incumbent.

**RULED (the owner, 2026-09-18): keep `onc_d`.**  `onc_d` stays on the defensive feature list, the
shipped rankings are unchanged, and nothing was rebuilt or republished.  The candidate is closed; do not
reopen it on the year-over-year number alone.

The arms are `outputs/tradeset_noonc_d_*` and `outputs/tradeset_cfs_*` (both `--team_effects=team`),
against the existing `outputs/tradeset_team_*`; the page is
`outputs/compare_base4b_noonc_d_shipped_2026.html`.

### The amplitude finding: three independent sources agree OpenRAPM is about a sixth too wide (2026-09-18)

The consensus file was rescaled twice on 2026-09-18.  The second change is the one that matters here: the
owner turned OFF the scaling to EPM and used the scale the GLS weighting implies.  On offence that is a
pure rescale -- correlation with the previous column **0.9994** -- taking the consensus's offensive spread
from 1.89 to 1.40, with defence barely moving (1.03 to 1.06) and the per-player uncertainty on offence
falling with it (0.76 to 0.56).  **Every consensus spread figure measured before that file is against a
different scale and must not be compared with one measured after it.**

**It killed a finding and replaced it with a better one.**  Against the EPM-scaled file, OpenRAPM's
offence read 18% too NARROW and its defence 17% too wide -- an offence/defence imbalance, written up
earlier the same day and now withdrawn.  Against the GLS scale both sides want the same multiplier:

| | OpenRAPM sd | consensus sd | asks for | correlation |
|---|---|---|---|---|
| offence | 1.63 | 1.40 | **x0.85** | 0.829 |
| defence | 1.41 | 1.19 | **x0.85** | 0.780 |
| total | 2.42 | 1.95 | x0.81 | 0.806 |

So there is no per-side imbalance against the consensus, only one uniform amplitude difference.  And that
now agrees with everything else that has looked at it:

| source | offence | defence | what it measures |
|---|---|---|---|
| the consensus, GLS scale | x0.85 | x0.85 | the same season, against public metrics |
| the trade set, three-season window | x0.749 | x0.919 | adjacent seasons' team-games |
| the year-over-year sweep, interior optimum | x0.65 | x0.85 | the neighbouring seasons' games |

**Three sources, one sign: the published rating is wider than every other reading of the same players.**
Experiment 20's conclusion stands unchanged -- a rescale is not adoptable, because the year-over-year and
trade-set versions are measured across seasons and cannot separate "too wide" from "players regress", and
because a uniform rescale reorders nobody (the order-only row is zero to 4e-14).  What changed is that the
within-season instrument no longer contradicts them, so the case that the SPREAD itself is too wide is now
consistent rather than split, and it wants fixing inside the fit rather than after it.

**What also survives the rescale, at half its former size: the high-usage creators.**  Six of the eight
highest-usage players sit below their consensus offence after the spreads are matched -- LaMelo Ball
-1.70, Ja Morant -1.61, Luka Doncic -1.59, Jokic -1.50, Booker -1.06, Giannis -0.83 -- against -2.30,
-2.27, -2.14, -2.06, -1.52 and -1.18 on the EPM-scaled file.  Luka's total gap is 1.2 rather than 2.7
(OpenRAPM +4.47, consensus +5.70; the offence is +4.13 against +5.50) and his defence is not in dispute
(+0.34 against +0.20).  The correlation with usage over all 391 players is **-0.00**, so this is the
extreme top of the league and not a usage gradient.

**The mechanism with the best claim, because it was measured independently:** the trade set found shot
creation, three-point rate and offensive rebound share to be UNDER-credited by the box prior.  Shot
creation is the defining trait of exactly this group, and a single-season rating shrinks toward that
prior.  What is NOT the explanation, measured and dismissed: age (no correlation with the gap),
experience (none), and team quality (team R-squared 0.166 against the consensus's 0.160).  The whole
per-36 box profile explains 21% of the player-by-player gap, so most of it is not a box-score story.

### Experiment 22: un-shrink the DEFENSIVE label only -- it passes every standing check (2026-09-18)

Experiment 21 found the un-shrunk label's whole measured gain on defence and its whole visible damage on
offence, so `scripts/62_single_year_board.py --unshrink_label=def` now does one side (the flag takes
`0|1|off|def`; `unshrink_label` takes a `sides` argument).  Built on the incumbent's settings -- fixed
penalty 13,037, `--exclude_neighbours=1`, three chunks of ten seasons, 93 minutes -- and stitched to
`outputs/season_ratings_unshrinkdef.parquet`, 14,579 rows over 30 seasons.

**It moves what it was aimed at and nothing else.**  Against the incumbent, absolute change per
player-season: defence median **0.441** points per 100, ninth decile 1.381, largest 3.648; offence median
**0.022**, ninth decile 0.051 -- below the owner's 0.1 threshold, so offence is untouched in the sense
that matters (the residue is the shared context penalty, not the label).

| | the incumbent | un-shrunk BOTH sides | **un-shrunk defence only** |
|---|---|---|---|
| year-over-year, `game_armse` | 8.6971 | 8.6672 | **8.6819** |
| paired vs incumbent, team-game | -- | -0.84, z -3.77, 39 of 56 | **-0.42, z -5.24, 43 of 56** |
| the same, stint level | -- | +1.57, z +5.31, 17 of 56 | +0.09, z +0.89, 29 of 56 |
| the same, each side rescaled (order only) | -- | **+1.79, z +7.75, 8 of 56** | +0.10, z +1.20, 28 of 56 |
| the trade loss, defence | -- | -0.0092, z -6.3 (vs 4b) | **-0.0093, z -7.14, 27 of 30** |
| the trade loss, offence | -- | +0.0017, z +0.8 | -0.0005, z -2.42, 20 of 30 |
| consensus rho off / def / total | 0.777 / 0.757 / 0.771 | 0.677 / 0.689 / 0.680 | **0.777 / 0.763 / 0.764** |
| consensus top five of 5 | 4 | 2 | **5** |
| 2026 offensive spread (2,000+ poss) | 1.60 | 1.14 | **1.64** |
| 2026 top five | Wemby, Jokic, Kawhi, SGA, Giannis | Wemby, Giannis, Clingan, Holmgren, Queta | **the incumbent's five, one swap of order** |

**The standing adoption rule is met on all three counts**: z is -5.24, well past -2; there is no consensus
miss at all -- defensive agreement RISES 0.757 to 0.763 and the top five goes 4 of 5 to **5 of 5**, the
best any candidate has scored; and the 2026 top 20 is not worse -- Curry 11th and Luka 12th (they were
61st and 72nd with both sides un-shrunk), with Ausar Thompson 11th to 6th, Queta 16th to 10th and Clingan
41st to 20th.

**The reservation, and it is real.**  The year-over-year gain does NOT survive removing amplitude: with
each side rescaled to the scored season the paired difference is +0.10 at z +1.20, an exact tie.  At stint
level it is also a tie (+0.09, z +0.89).  So on the criterion's own objective this reads as better
DEFENSIVE CALIBRATION -- `scale_def` moves 0.8890 to 0.8979, toward 1 -- rather than a better order.  What
says it is more than calibration is the trade loss, which prices the rating with two free unpenalised
scale columns and is therefore amplitude-blind by construction: defence -0.0093 at **z -7.14, 27 of 30
seasons**, the largest per-player gain any candidate has posted here.  Two objectives, and the one that
counts a bench player the same as a starter is the one that sees it.

Against the both-sides version this is the whole point: there, the order was clearly WORSE (rescaled
z +7.75, 8 of 56) and the team-game gain was amplitude.  Here the order is neutral and the per-player
loss improves.

**The one countervailing reading**: in consensus bar units the defensive disagreement grows after matching
spread, 1.741 to 1.974, while defensive RANK agreement improves.  The two disagree because the bars are
2026 only over 391 players and `rho_def` pools 2024-26 over 475; it is not resolved and is recorded as
unresolved.

**ADOPTED (the owner, 2026-09-18).**  `--unshrink_label=def` is the default in
`scripts/62_single_year_board.py` and `season_ratings_unshrinkdef.parquet` is the incumbent the
year-over-year test compares against, replacing `season_ratings_sy_lam13037.parquet`.  The shipped product
table was rebuilt at `--exclude_neighbours=0` in the same session, which also clears the `poss_def` column
that had been a copy of `poss_off` since before 2026-09-18.

### Experiment 21: the trade loss re-judges eight candidates on disk, and splits the verdict on the
### un-shrunk label (2026-09-18)

Eight distinct candidates already built were re-scored with the trade loss -- the per-player loss that
counts a bench player the same as a starter -- against the baseline they were built on, experiment 4b.
`scripts/70_tradeset.py --team_effects=team` on each (85 s each, not the 9 minutes `HANDOFF.md` quotes;
that figure included the penalty grid) then `scripts/73_tradeloss.py`.  30 seasons, 12,103 players a
side, paired by season, on the exact intersection of eligible player-seasons.

| candidate | offence | z | defence | z | won of 30 | the team-game verdict it had |
|---|---|---|---|---|---|---|
| `sy_unshrink`, the un-shrunk label | +0.0017 | +0.8 | **-0.0092** | **-6.3** | 25 | passed, rejected on the eye test |
| `sy_noonc`, on-court off BOTH sides | +0.0003 | +0.2 | **-0.0049** | **-2.5** | 22 | never ruled on |
| the incumbent, fixed penalty 13,037 | -0.0000 | -0.0 | -0.0024 | -2.2 | 21 | adopted as a TIE |
| `sy_offc`, the off-court record | +0.0002 | +0.2 | -0.0012 | -1.5 | 17 | rejected |
| `sy_reg5`, regularisation x5 | +0.0001 | +0.2 | -0.0004 | -0.6 | 17 | rejected as a tie |
| `sy_depth3`, booster depth 3 | +0.0002 | +0.4 | +0.0008 | +1.2 | 13 | rejected |
| `sy_oop`, out-of-player priors | +0.0016 | +2.0 | +0.0008 | +0.8 | 13 | worse; kept by ruling 2 |
| `sy_chunks_all`, every chunk size | +0.0033 | +2.5 | +0.0024 | +2.5 | 10 | rejected |
| `sy_rows`, one row per player-season | +0.0105 | +4.9 | +0.0086 | +4.7 | 5 | rejected |

**Six of the eight confirm the ruling the team-game test already made**, which is the first evidence that
this loss and that test mostly agree.  Two do not, and `sy_noonc` -- on-court off BOTH sides at z -2.5 on
defence, against z -0.40 for `sy_noonc_d`, off the defensive list only -- has no explanation yet and is
noted as an open oddity rather than a finding.

**The un-shrunk label: every prediction instrument prefers it and every attribution instrument rejects
it.**  It was rejected on 2026-09-14 on the 2026 top 20 alone, having passed the test at 8.667 against
8.697.  What is known now:

| instrument | reading | verdict |
|---|---|---|
| year-over-year, team-game | 8.667 against 8.697 | better |
| the trade loss, defence | -0.0092, z -6.3, 25 of 30 | much better |
| ... **over 1,500 possessions** | **-0.0097, z -6.1, 25 of 30** | **not an artefact of thin evidence** |
| the trade loss, offence | +0.0017, z +0.8 | a tie |
| consensus rank agreement | 0.677 / 0.689 / 0.680, top five 2 of 5 | worse on all three, ~10% under |
| the consensus bars, spread matched | rms 2.91 offence, 2.24 defence (2.03 / 1.74 shipped) | worse on both sides |
| the 2026 top 20 | Curry 7th to 61st, Luka 10th to 72nd, Jokic 2nd to 12th, Kawhi 3rd to 24th | the veto |

The obvious suspicion -- that the trade loss gain hides among players the trade set cannot see, since a
first-year player has no absence contrast -- was checked and is **wrong**: by tier the defensive gain is
-0.0033 (z -1.3) under 500 possessions, -0.0089 (z -4.2) from 500 to 1,500, and -0.0097 (z -6.1) above
1,500.  It is strongest where the evidence is thickest.

**The mechanism of the eye-test damage is visible and it is OFFENSIVE.**  The offensive spread over 2026
players with 2,000+ possessions collapses from 1.60 to **1.14** while defence holds at 1.25 against 1.36,
and the 2026 list fills with defensive bigs and rookies -- Kalkbrenner 175th to 10th, Jakucionis 145th to
15th, Flagg 75th to 11th, Ighodaro 60th to 13th, Murray-Boyles 71st to 18th -- while the offensive stars
fall.  **So the whole measured gain is defensive and the whole visible damage is offensive**, which the
current flag cannot separate: `unshrink_label` in `scripts/62_single_year_board.py` loops over both
columns.  A per-side version is a two-line change and one 90-minute build, and it is the experiment this
re-judging has earned.  Not run, and nothing adopted.

### Experiment 20: the amplitude swept per side -- the contradiction is within-season vs across-season,
### and nothing ships (2026-09-18)

Two instruments disagreed about which side of the published rating is mis-scaled, so the criterion was
asked directly.  `scripts/75_amplitude.py` multiplies a finished table by one number per side and changes
nothing else; `scripts/63_yoy.py` then scores each arm, refitting only the scored season's intercept and
home edge.  51 arms over two sweeps on `season_ratings_sy_lam13037.parquet` (the `--exclude_neighbours=1`
table, so no prior saw the scored seasons), 28 seasons, both directions, 56 observations.  Four minutes.

**What the test says, pooled `game_armse`, points per 100 per team-game:**

| arm | offence x | defence x | game_armse | vs unchanged | z | won of 56 |
|---|---|---|---|---|---|---|
| the consensus bars | 1.196 | 0.867 | 8.7886 | **+2.53** | +17.8 | 1 |
| unchanged | 1.00 | 1.00 | 8.6971 | -- | -- | -- |
| the trade set | 0.749 | 0.919 | 8.6261 | -1.94 | -14.5 | 55 |
| the grid's own best | **0.65** | **0.85** | **8.6160** | -2.21 | -12.6 | 54 |

The second sweep was needed because the first grid's argmin sat on its edge at offence 0.70 (trap 6: an
argmin on a boundary has chosen nothing).  Extended to 0.45, the minimum is interior at 0.65 on offence
and 0.85 on defence, and the surface is a plateau -- 8.616 to 8.629 anywhere between 0.55 and 0.75 on
offence.  So the test wants the offensive rating about a third narrower and the defensive about a sixth.

**It moves nobody.**  The order-only row -- stint error with each side rescaled to the scored season --
is **zero to machine precision (4e-14) for all 27 arms of the first sweep**.  A uniform rescale is
entirely a units change: the whole list breathes in or out together and no player passes another.  That
is the measured version of the standing warning that the test punishes spread whatever the order did.

**Why this does NOT license shrinking the published rating, and this is the finding.**  `HANDOFF.md`
already says it: `scale_*` below 1 on a NEIGHBOURING season is expected, because a player's true impact
changes from year to year, and a shrunk rating is the better forecast of a different season even when it
is perfectly calibrated for its own.  This sweep is measured entirely across seasons, so it cannot tell
"the rating is too wide" from "players regress".  Sort the instruments by what they measure and the
contradiction largely dissolves:

| | offence | defence | measured |
|---|---|---|---|
| the year-over-year sweep | x0.65 | x0.85 | across seasons |
| the trade set, pooled over the three-season window | x0.749 | x0.919 | mostly across |
| **the trade set, inside a single season** | **x1.03 to 1.12** | x1.38 to 1.51 | within |
| **the consensus error bars, spread matched** | **x1.196** | x0.848 | within (2026) |

**On offence the two within-season instruments agree that the rating is too NARROW** (x1.03-1.12 and
x1.196) while both across-season instruments want it narrower, which is exactly the signature of
regression to the mean and not of a miscalibrated rating.  **On defence they still conflict**: the trade
set's within-season reading wants defence much wider (x1.38-1.51) and the bars want it 15% narrower
(x0.848).  That one is unresolved.

**Conclusion: the amplitude item is closed as a rescale.**  The 0.069 of game error the trade set
attributed to "the offensive rating is a quarter too wide" is a forecasting gain, available to anyone
willing to publish a shrunk number, and it buys nothing for a rating of the season it describes -- it
cannot, since it reorders nobody.  Nothing was rebuilt and no published number changed.  What remains
live is the defensive disagreement above, which is a question about the rated season itself and is
therefore answerable without the neighbours.

### The team-movement weight: the measure is Gini-Simpson, and the rule now has tests (2026-09-18)

The owner's idea of 2026-09-16 -- weight the prior's training rows by how much team variation sits behind
a player's label, since the label is one career-pooled leave-season-out RAPM and it is team variation that
identifies it.  `scripts/62_single_year_board.py --trade_weight=<floor>`, off by default and not in the
shipped run.  Raised again on 2026-09-18 by the owner as *"a little brittle / not robust"*, and it was:

**The measure was `1 - the share of his possessions on his most-played team`**, which ignores the shape of
the tail.  A player at 50/10/10/10/10/10 read 0.50, identical to one at 50/50, though the first has five
independent contrasts and the second has one.  The owner found that defect the same day he proposed the
measure and the fix was never implemented; it is implemented now.

**`team_movement` is `1 - sum(share ** 2)`, the Gini-Simpson index**: the chance that two possessions of
his career came from different teams.  It is a strict generalisation and never below the old form --
`sum(share ** 2) <= max(share)` always, with equality exactly when every team he played for got the same
share of him, so one team, 50/50 and three-at-a-third all read the same as before -- and the six-team
case reads 0.70 against 0.50.  On the 2,584 players with 100+ possessions before 2026: correlation 0.985,
rank 0.992, 70% of players gain, largest gap 0.211, and the 767 one-team players sit at exactly zero under
both.  Its reciprocal is the effective number of teams.  Rejected alternatives: a geometric or harmonic
mean of the shares (one tiny share crushes a mean, which is backwards -- a cameo on a seventh team is a
little more contrast, not almost none) and entropy perplexity (too generous to the tail: 4.47 effective
teams for the 50/10-times-five case against this index's 3.33).  Also rejected, and tempting: the label's
own standard error, which is dominated by total possessions -- and possessions are already the row weight,
so it would count exposure twice.  Movement is the part of precision that exposure does not carry.

**Seven tests, where there were none** (`tests/test_singleyear.py`): the three cases that pin the shape
down, the cameo, invariance to how many possessions the shares are made of, the `min_poss` drop, the
strict-generalisation property over 200 random splits, that the reweighting preserves total weight and
moves only its distribution across players, that a positive floor keeps everybody, and that an all-zero
training set raises instead of quietly fitting a constant.

**What the rule still loses on, unchanged by this.**  At `floor=0` it zeroes the one-team players -- 29% of
those behind a 2026 label, every single-franchise star among them -- and the booster then extrapolates at
the top; the one run scored a tie at game level and lost on the amplitude-free row.  **Nothing was rebuilt
and no rating moved**: the flag is off in the shipped run and `--rows=chunks` is its only effect.  If the
idea is revived, sweep the floor above zero, and judge it on the trade loss rather than the team-game
test, because better-identified bench players is the point and the team-game test cannot see them.

**What it is still blind to:** teams, not teammate sets.  Two seasons on one team with a rebuilt roster
give real contrast and score zero.  The finest version is the effective number of distinct teammate sets,
or the coefficient's variance directly.

### The consensus has error bars now, and the check is read in bar units (the owner, 2026-09-18)

`data/external/consensus.csv` gained `var_offense` and `var_defense`: a **standard deviation** per player
per side (the column name says variance; the numbers are standard deviations).  The owner built them and
the method is his, recorded here because how they were made decides how they may be read:

1. **The votes are de-duplicated before they are counted.**  Each metric does not get one vote;
   the weights are `(R + lambda I)^-1 1` on the metrics' own correlation matrix, so five collinear
   metrics split a single vote and an independent metric keeps a whole one.
2. **The bar is the correlation-weighted spread around that consensus.**  Because the collinear five can
   no longer outvote one independent dissenter, a lone disagreeing metric widens the bar instead of being
   averaged away.  A wide bar means the public metrics genuinely do not know where a player belongs.

`adj_offense`, `adj_defense` and `adj_overall` are unchanged to the last decimal, so every consensus
number quoted anywhere above still holds; nothing in `src/` or `scripts/` read the old `var` or
`Disagreement Among Metrics` columns, both of which changed meaning (`var` was a percentage and is now
`var_offense + var_defense`, a sum of standard deviations rather than a quadrature sum, which is the
conservative choice -- it assumes a player's offensive and defensive errors are perfectly correlated).

**The blend changed on 2026-09-18 and every consensus number above it is against the OLD one.**  The
owner rebuilt `adj_offense` / `adj_defense` / `adj_overall` so the blend is a real consensus rather than
one dominated by a few collinear raw-points metrics.  Same 582 players, `var_offense` and `var_defense`
untouched to the decimal; the central values moved, defence most (correlation with the previous blend
0.975 on offence, **0.911 on defence**).  What that does to the readings, 2026, 475 player-seasons for the
rank agreement and 391 players over 1,000 possessions for the bars:

| | rho off / def / total, OLD blend | rho off / def / total, REAL consensus | top five |
|---|---|---|---|
| the incumbent before adoption (`sy_lam13037`) | 0.777 / 0.757 / 0.771 | **0.792 / 0.834 / 0.808** | 4 -> 5 |
| the adopted defence-only un-shrink | 0.777 / 0.763 / 0.764 | **0.790 / 0.801 / 0.797** | 5 -> 4 |
| un-shrunk BOTH sides (rejected) | 0.677 / 0.689 / 0.680 | 0.675 / 0.730 / 0.699 | 2 -> 2 |
| `onc_d` off the defensive list (rejected) | 0.731 / 0.689 / 0.705 | 0.736 / **0.797** / 0.742 | -> 3 |

**Two of the session's rulings lose a line of evidence and neither reverses.**

* **Experiment 22 (adopted).**  Under the old blend the defence-only un-shrink RAISED defensive agreement
  0.757 to 0.763; under the real consensus it LOWERS it, 0.834 to 0.801, and the top five goes 5 to 4.
  That line of the adoption case is withdrawn.  It is still far from a veto -- 0.801 against a 0.75 check
  -- and the two instruments the adoption actually rested on are untouched: the year-over-year test at
  z -5.24 and the trade loss at z -7.14.  The owner should know the consensus now mildly disagrees.
* **Experiment 19 (kept `onc_d`).**  Its defensive agreement was 0.689 against the old blend, which read
  as a broad miss; against the real consensus it is 0.797, no miss at all.  The component split that
  carried the argument was measured on `impact_metrics_2526.csv` directly and is unaffected, but the
  blend-level number no longer supports it.  The ruling stands on the trade loss finding nothing
  (z -0.40), which was always the primary reason.

**The bars, re-read against the real consensus** (2026, 391 players, each side scaled first, medians):

| | our typical gap | their typical uncertainty | inside 1 | inside 2 |
|---|---|---|---|---|
| offence, adopted | 0.72 | 0.70 | 46% | 78% |
| defence, adopted | 0.50 | 0.47 | 49% | 75% |
| defence, previous incumbent | 0.42 | 0.47 | 55% | 81% |

**This overturns what the old blend appeared to say.**  Against it our typical distance was about twice
the consensus's own uncertainty and -- the part that looked diagnostic -- the same size in points whether
the metrics agreed with each other or not, which read as "we are measuring something else".  Against the
real consensus our typical gap is the same size as their own disagreement (0.72 against 0.70; 0.50 against
0.47) and it GROWS where they are less sure (offence, narrowest fifth of their uncertainty 0.81 points,
widest fifth 1.05, correlation +0.13).  That is the pattern of two measurements of the same thing, one
noisier.  The earlier reading was an artefact of a blend whose defence was 90% two collinear metrics.

**The standing way to read the check, from here:** `scripts/74_consensus_bars.py`, **each side scaled on
its own** (the owner, 2026-09-18: *"feel free to scale first when comparing to the consensus, esp by O vs
D"*).  Spearman against the blend treats a disagreement about Jokic's defence (bar 0.39) and one about
Wembanyama's offence (bar 2.05) as the same size of miss.  The bars say whether we are outside what the
public metrics can themselves resolve.  2026, 391 players over 1,000 possessions, the shipped rankings:

| side | bar median | our sd | consensus sd | rms as published | rms, level and spread matched | over 1 bar | over 2 bars | correlation | rms on the best-fit line |
|---|---|---|---|---|---|---|---|---|---|
| offence | 0.70 | 1.59 | 1.91 | 2.02 | **2.03** | 52.4% | 25.3% | 0.826 | 1.93 |
| defence | 0.47 | 1.29 | 1.12 | 2.27 | **1.93** | 47.6% | 20.2% | 0.805 | 1.74 |

Three readings, and the third is the one that matters:

* **The level is close**: -0.32 bars on offence and -0.57 on defence over these players, which is a
  level offset and not a disagreement about players.
* **The spreads are mismatched in OPPOSITE directions by side.**  Matching each side's standard deviation
  to the consensus's asks for **x1.196 on offence** -- our published offensive spread is 20% narrower
  than theirs -- and **x0.867 on defence**, ours 15% wider.  So the O/D balance differs by about 1.38
  between the two, which is a bigger statement than either side's amplitude.  Read it beside the trade
  set: that instrument wants offence NARROWER (x0.749 across adjacent seasons) and defence about right
  (x0.919).  The two agree in direction on defence and **contradict each other on offence**, which is
  worth an experiment before any amplitude is changed.
* **Player by player we are well outside the bars, and units are not the reason.**  After matching level
  and spread per side the root mean square disagreement is 2.03 bars on offence and 1.93 on defence --
  no better than as published, because the mismatch is not mostly a scale.  The best any straight line
  can do is 1.93 and 1.74, with half the players more than one bar out and a fifth more than two.  The
  correlations are 0.826 and 0.805.

**Do not read the least-squares slope as a spread comparison.**  It is `correlation x sd_ratio`, so a
low correlation drags it down on its own: on defence it is 0.698 while the spreads differ by 0.867, and
quoting it as a scale is how our defensive spread was first written up here as 1.4 times the consensus's
when the honest figure is 1.15.  Corrected 2026-09-18, same session.

**What this does not say.**  The bar is the public metrics' disagreement with each other, not the
uncertainty of our rating.  Five metrics that share a defect agree tightly and produce a narrow bar, and
being two bars away from them is then a claim about them as much as about us -- which is exactly what
`onc_d` off the defensive list was doing, in the wrong direction, in experiment 19.  Use the bars to
size a disagreement, never as a target to shrink.

### Experiment 23: the pieces of vanilla RAPM predict next season's RAPM, not next season's points (2026-09-28)

**The owner's idea** ("99% certain"): the pieces of a season's vanilla RAPM -- the Decomposition page's two
exact splits -- predict a player's RAPM in the season before and after better than their sum does.

**How it was run.**
- `scripts/84_piece_panel.py` splits every piece into its offensive and defensive half: 14,578
  player-seasons, 1997-2026, penalty 3,000 per side.  Every per-side identity holds to 1e-13, and the two
  halves add up to the page's net pieces exactly.
- `scripts/85_piece_rating.py` does the rest.  A pair is a season t (the inputs) and a neighbouring season
  the player also played (the target: his vanilla RAPM there, at 3,000, per side).  Both directions are
  pooled, and the weight is the harmonic possessions of the two seasons.
- The rating for season t comes from fits on pairs touching neither t-1 nor t+1, with five player folds:
  the incumbent's `--exclude_neighbours=1` plus the out-of-player rule.
- Every candidate has a like-for-like control, the same model given RAPM and possessions but no pieces.
- Each split adds up to RAPM exactly, so no linear fit ever sees a redundant set.  The eight-piece basis
  is the five by-player pieces plus the three off-court possession groups, and every linear fit asserts
  full rank first.
- The whole chain took 55 minutes: GBDT early stopping keeps 50-60 trees on a target this noisy, so a fit
  takes about 0.2 s.

**The pieces do deserve different weights.**  Players with 1,000+ possessions in both seasons, OLS per
split, per point of the piece.  RAPM itself gives every piece 0.47 on offence and 0.40 on defence:

| piece | offence | defence |
|---|---|---|
| on-court rtg | 0.64 (0.55-0.69) | 0.44 (0.38-0.50) |
| teammates | 0.59 (0.50-0.65) | 0.39 (0.33-0.45) |
| opponents | 0.84 (0.73-0.93) | 0.54 (0.47-0.62) |
| context | 0.64 (0.55-0.72) | 0.46 (0.40-0.53) |
| ridge penalty | 0.92 (0.78-1.02) | 0.54 (0.43-0.64) |
| on court signal | 0.45 (0.41-0.49) | 0.40 (0.36-0.43) |
| off court adjustment (GP) | 0.40 (0.34-0.44) | 0.28 (0.24-0.33) |
| off court adjustment (DNP) | 0.25 (0.15-0.34) | 0.23 (0.15-0.28) |
| team SOS adjustment | 0.90 (0.73-1.05) | 0.57 (0.47-0.66) |

- On offence every by-player piece earns MORE than the total's weight.  That is possible because the
  pieces cancel one another: on-court rtg correlates -0.80 with the ridge penalty and -0.78 with teammates.
- Out of sample, on the same players, the by-player weights beat RAPM x slope on offence against both
  truths: -0.051 at z -10.2 (30 of 30 seasons), and -0.052 at penalty 100, z -4.0.
- On defence the sign flips between the two truths (-0.025 at z -7.0, then +0.030 at z +3.6), so it is
  undecided.
- The elastic net on the eight-piece basis kept nearly everything: pure L1, a tiny alpha.  Tables:
  `outputs/csv/piece_coef_table1.csv`, `_table2.csv`, `_oos.csv`.

**Every player, against the like-for-like control** (net; a negative difference is better):

| test | pieces, linear | pieces, GBDT |
|---|---|---|
| next/previous RAPM at 3,000 | -0.065, z -9.3, 29 of 30 | -0.084, z -7.2, 28 of 30 |
| -- players who changed team | z +1.5 | z +1.5 |
| -- players who stayed | z -11.4 | z -7.8 |
| next/previous RAPM at 100, spread removed | +0.185, z +5.4, 4 of 30 | +0.166, z +3.4, 7 of 30 |
| year-over-year, team-game, as built | -0.55, z -10.1, 52 of 56 | -0.99, z -15.0, 56 of 56 |
| -- each side rescaled (order only) | +0.60, z +5.8, 11 of 56 | +0.27, z +2.0, 25 of 56 |
| -- every table at vanilla's spread | +2.23, z +14.7, 1 of 56 | +1.52, z +8.5, 5 of 56 |
| trade loss, offence / defence | +0.0029 z +3.9 / +0.0034 z +4.0 | +0.0006 z +0.7 / +0.0011 z +1.2 |

Year-over-year `game_armse` as built:

| table | error |
|---|---|
| incumbent | 8.6819 |
| pieces GBDT | 8.6735 |
| pieces GBDT on Boruta's inputs | 8.6744 |
| pieces linear | 8.6901 |
| RAPM + possessions, GBDT | 8.7095 |
| RAPM + possessions, linear | 8.7099 |
| RAPM x slope | 8.7197 |
| vanilla RAPM | 8.7184 |

That vanilla row is the first time single-season vanilla RAPM has been scored on this test.

**Reading: what the pieces predict is RAPM's own habit, not the player.**
- **Against next season's RAPM, the whole gain belongs to players who stayed on their team**, and it
  reverses against the lightly penalised truth once spread is removed.  That is the planned trap:
  - the same teammates make the same shared-credit split again;
  - a penalty-3,000 truth rewards a model that shrinks the way it does.
- **On actual points, the team-game win as built is spread:**
  - the piece models come out wider than their controls (the scored season wants them x1.59, the control
    x1.83), and every model trained on a shrunk target is too narrow;
  - at one common spread they lose;
  - with each side rescaled they lose;
  - the trade loss, which is spread-blind by construction and counts every player once, finds the linear
    version worse on both sides and the GBDT a tie.
- **Three independent order readings agree.**
- **Split by how many of the ten on the court changed team** (stint level, each side rescaled):
  - the GBDT ties where nobody moved (-0.05, z -0.10), ties with 1-2 movers (z +1.0) and loses with 3+
    (z +2.5);
  - the linear version loses in every group (z +1.5, +6.1, +4.7).
  - The pieces' extra signal lives with unchanged teammates, and even there it only ties on actual points.

**Boruta** (50 trials, per side, with and without the RAPM total; full table in
`outputs/csv/piece_boruta_table.csv`):
- **Offence:** the top two inputs are not pieces.  Team possessions without him in games he played scores
  3.54 and his own possessions 2.53, both above RAPM itself (2.36).  Pieces accepted beyond the total:
  on-court rtg 0.88, teammates 0.66, on court signal 0.39.
- **Defence:** RAPM itself 4.05, on court signal 1.67, opponents 1.27.
- Boruta dropped 12 and 17 inputs, so a GBDT on its kept inputs was run.  It matches the full GBDT (8.6744,
  the same verdicts) and carries a small selection leak, because Boruta saw every season.

**Movement**, the pieces against their own control at vanilla's spread:
- GBDT: median 0.41 points per 100, ninth decile 1.04, 87% of players past 0.1.
- Linear: median 0.28, ninth decile 0.71, 81% past 0.1.
- The pieces reorder players a good deal, and the tests that see order say the new order is worse.

**2026 top 20, pieces GBDT:**
- The top 10: Wembanyama, Gilgeous-Alexander, Jokic, Vassell (the incumbent's 58th), Ausar Thompson,
  Leonard, Amen Thompson (43rd), Ajay Mitchell, Holmgren, Caruso (51st).
- Out of the incumbent's top 20: Antetokounmpo to 64th, Curry 85th, LeBron James 119th, Clingan 182nd.
- Page: `outputs/compare_incumbent_vanilla_pieces_gbdt_rapm_gbdt_pieces_linear_2026.html`.

**Also measured, and recorded here only:** OpenRAPM x slope, rescaled season by season to vanilla's spread,
scored 8.6170 against the incumbent's 8.6819.  It narrows offence (the scored season wants x1.05 against
x0.79), in line with "The amplitude finding"; a rescale is still not adoptable (experiment 20).

**Verdict: not adopted as a rating.**  The idea is right that the pieces are not worth equal weights.  It is
wrong that re-weighting them predicts the next season's points better.  Offence for players with heavy
minutes is the one corner where the weights survive both truths, and it was never scored on points alone.
Awaiting the owner.

### Experiment 24: the same models told whether the player stayed on his team (2026-09-28)

**The owner's follow-up**: pieces x "stayed on the same main team", linear then GBDT.
- The flag belongs to the PAIR: same main team in the rated season and in the target season, in either
  direction.
- The linear models get the flag plus the flag x every input; the GBDT gets it as one more input.
- The controls get the flag too.
- Every player-season is rated twice, as if he stays and as if he moves.  The direct test takes each pair's
  real flag.  The year-over-year test uses `_fwd` tables (same team the next season) for the rows where the
  rated season predicts the one after, and `_bwd` for the rows where it predicts the one before, stitched
  into one system (`outputs/yoy_pieces_stay_stitched.parquet`).
- 48.1% of player-seasons keep their main team into the next season.

| test, against the like-for-like control (also told the flag) | pieces x stayed, linear | pieces + stayed, GBDT |
|---|---|---|
| next/previous RAPM at 3,000 | -0.074, z -9.5, 28 of 30 | -0.088, z -8.7, 28 of 30 |
| -- players who changed team | z +0.3 | z -0.1 |
| -- players who stayed | z -10.4 | z -8.4 |
| next/previous RAPM at 100, spread removed | +0.256, z +6.2, 4 of 30 | +0.211, z +4.6, 7 of 30 |
| year-over-year, team-game, as built | -0.55, z -8.5, 49 of 56 | -0.97, z -13.4, 55 of 56 |
| -- each side rescaled (order only) | +0.87, z +8.7, 7 of 56 | +0.58, z +4.6, 16 of 56 |
| -- every table at vanilla's spread | +3.59, z +19.7, 0 of 56 | +2.44, z +12.6, 2 of 56 |
| trade loss, rated as if he moves (off / def) | z +5.4 / +1.6 | z +0.7 / -1.3 |
| trade loss, rated as if he stays (off / def) | z +3.8 / +4.8 | z +1.4 / +1.6 |

Year-over-year `game_armse` as built:

| table | error |
|---|---|
| pieces GBDT with the flag | 8.6661 (lowest yet) |
| pieces linear with the flag | 8.6821 |
| RAPM + possessions GBDT with the flag | 8.7011 |
| RAPM + possessions linear with the flag | 8.7019 |
| incumbent (experiment 23) | 8.6819 |
| vanilla | 8.7184 |

**Reading: the same verdict as experiment 23.**
- The flag helps every model a little as built.  The pieces with it beat the pieces without it: z -4.4
  linear, z -5.4 GBDT.  Rescaled, those are ties (z -0.4, z 0.0).
- It stops the pieces hurting players who changed team: z +1.5 before, a tie now.
- But the pieces' gain over the control is still all players who stayed, and still reverses against the
  lightly penalised truth.
- On real points, order only or at one spread, the pieces lose to the control, and by more than without the
  flag.
- The spread-blind trade loss finds the GBDT a tie and the linear version worse.

**2026 rankings against vanilla** (rank agreement, players with 1,000+ possessions):
- Linear 0.90 as if he stays, 0.85 as if he moves; GBDT 0.88 / 0.84.
- Players with heavy minutes rise: Brunson 88th to 11-24th, Murray 133rd to 15-57th, Doncic 47th to
  13-38th, Vassell 20th to 4-5th, Cunningham 26th to 6-16th.
- Players with few possessions fall: Josh Green 17th (1,809 possessions) to 63-132nd, Hugo Gonzalez 19th to
  72-109th, Butler 11th to 23-83rd.
- That is the possessions signal Boruta ranked first in experiment 23, not the pieces.
- Page: `outputs/compare_vanilla_linear_if_stays_linear_if_moves_gbdt_if_stays_gbdt_if_moves_2026.html`.

**Verdict: not adopted.**  Awaiting the owner.

### Experiment 25: each chunk row labelled with the player's RAPM OUTSIDE the chunk (2026-09-29)

**The change** (the owner's plan of 2026-09-28, the first of three: team and game context in the prior).
- Until now every training row of a player -- his career row and all his 1-, 2- and 3-season chunks --
  carried the SAME label: his RAPM over every training season.  A label that is constant across a
  player's rows cannot teach anything that differs between his chunks (experiment 15), and the next
  experiment's same-team input is exactly such a thing.
- Now a chunk's label is his RAPM over the training seasons OUTSIDE the chunk, so its box score and its
  label share no game.  The career row keeps its label.  A chunk with no season of his outside it is
  dropped (about 4% of rows: 34,222 against 35,647 for 1997).
- Nothing else changed: features, weights (a player's chunk rows together still weigh what his career
  row weighs), booster, out-of-player folds, ridge, penalty.  `--chunk_label=outside`;
  `singleyear.chunk_rows(labels=...)`.

**How it was run.**
- chimeraboost went from 0.32 to 0.34 first.  2015 and 2024 rebuilt under 0.34 match the incumbent to
  0.0 on every column, so the incumbent was not rebuilt; it reproduces 8.682 below.
- The default setting under the new code reproduces 2015's 492 players exactly, before and after the
  speed fix.
- Outside labels need one RAPM per distinct set of chunk seasons: **278 to 308 per season and side**, not
  the 84 the plan guessed, because a player who skipped seasons makes chunk sets nobody else has.
- At one BLAS thread (the build's pinning) a solve took 5.0 s, so a season took 52 minutes; the first
  run was stopped at three seasons.  The outside labels are now solved by Cholesky with BLAS allowed
  twelve cores for the duration: 0.62 s a solve, 11 minutes a season, 5.5 hours for thirty.  They agree
  with the shipped solver to 2.3e-7 per 100.  The career label keeps the shipped solver on one thread.
- Unit tests: a 1-season chunk of season s carries exactly the label `season_rows` gives s; a chunk
  that straddles the excluded seasons (2011 + 2013) is labelled without both; weights unchanged.

**Year-over-year, both directions, 56 observations** (`outputs/yoy_outside_labels.log`):

| | game_armse | scale_off | scale_def | paired vs incumbent, team-game MSE |
|---|---|---|---|---|
| incumbent | **8.6819** | 0.795 | 0.898 | -- |
| outside labels | 8.6999 | 0.781 | 0.868 | +0.488, z +4.98, 14 of 56 |
| -- stint level | | | | +0.305, z +2.48, 22 of 56 |
| -- each side rescaled to the scored season | | | | **+0.008, z +0.08, 31 of 56** |

**Trade loss** (`outputs/tradeloss_outside_labels.log`, against `tradeset_unshrinkdef_alpha`, 24,206
player-season-sides in both): offence +0.0033, z +4.5, 6 of 30 seasons; defence +0.0055, z +5.0, 6 of 30.

**Consensus** (2024-26, 1,000+ possessions): rank agreement 0.775 / 0.806 / 0.779 (offence / defence /
total) against the incumbent's 0.785 / 0.803 / 0.794; spreads 1.027 / 1.010 against 1.029 / 1.007; top
five 4 against 4.  No gross miss.

**Movement against the incumbent** (`outputs/movement_outside_labels.log`), all 14,579 player-seasons:
- median 0.24 per 100 on offence, 0.34 on defence, 0.43 total; ninth decile 0.65 / 0.89 / 1.10; 87% of
  players move more than 0.1 in total, in every possession tier.
- Almost all of it is order, not spread: a per-season rescale of each side accounts for a median 0.02
  on offence and 0.04 on defence.
- The ridge stretches the new offensive prior more (prior scale 2.32 against 2.10, larger in 29 of 30
  seasons), and the finished offence ends a little wider (sd 1.86 against 1.79 among 1,000+
  possessions), which is what `scale_off` reads.

**2026 top 20** (`outputs/compare_incumbent_outside_labels_2026.html`; the phone table was emailed):
- The top six are the same six: Wembanyama, Leonard, Antetokounmpo, Jokic, Gilgeous-Alexander, Ausar
  Thompson.  Wembanyama's defence rises 1.3 (5.9 to 7.3).
- In: Caruso 51st to 18th (defence +2.0), Butler 36th to 8th (offence +1.1), Donovan Mitchell 23rd to
  7th, Cason Wallace 32nd to 16th (defence +1.2), Brunson 33rd to 20th (offence +1.0), Anunoby 24th to
  19th.
- Out: Champagnie 18th to 54th (offence -1.0), Cunningham 14th to 28th, Diabaté 16th to 29th, Durant
  15th to 26th, LeBron James 19th to 27th, Clingan 20th to 22nd.

**Reading.**
- The label now varies within a player, which is what experiments 26 and 27 need, and it is honest in
  the same way `--rows=season` was: no row is scored against its own games.
- On its own it loses every test that sees level and spread: the year-over-year test at z +5.0, the trade
  loss at z +4.5 and +5.0.  With each side rescaled it is a tie (z +0.08), so it does not order players
  worse on the neighbouring seasons' games; the trade loss, which is blind to spread, still says worse.
- It moves players a lot (median 0.43 in total), mostly by reordering them.

**Verdict: not adopted on its own.**  By the plan, experiments 26 and 27 use outside labels regardless,
because a same-team input needs a label that changes with the team.  The plan also says: if this loses
badly on its own, ask the owner before going on.  It does; awaiting the owner.

### Experiment 25b: the outside labels, each chunk weighted by the possessions its own label rests on (2026-09-29)

**Why.**  Experiment 25's movement looked like fitted noise, not a pattern: age explains 0.3% of who
moved, and the tilts by possessions (-0.14 to +0.09) are small beside a median move of 0.43.  An outside
label rests on his career minus the chunk, so it is noisier than the career label, yet each chunk row kept
the career label's weight.  The owner's go (2026-09-29): weight each chunk row by `share x the possessions
its own label rests on` instead.  Career rows untouched.  `--chunk_weight=label`,
`singleyear.chunk_rows(weight_by="label")`; everything else as experiment 25.
- The weight moved as intended: in 1997 the career rows carry 33.8 million possessions of weight as
  before, the chunk rows 23.7 million, where they had carried as much as the career rows.

**Year-over-year, both directions, 56 observations** (`outputs/yoy_outside_reweighted.log`,
`_vs25.log`):

| | game_armse | scale_off | scale_def | vs incumbent, team-game MSE | vs experiment 25 |
|---|---|---|---|---|---|
| incumbent | **8.6819** | 0.795 | 0.898 | -- | |
| experiment 25 | 8.6999 | 0.781 | 0.868 | +0.488, z +5.0, 14 of 56 | -- |
| reweighted | 8.6926 | 0.859 | 0.985 | +0.310, z +1.6, 25 of 56 | -0.178, z -0.8, 28 of 56 |
| -- stint level | | | | +1.41, z +5.4, 12 of 56 | |
| -- each side rescaled (order only) | | | | **+1.90, z +8.1, 8 of 56** | |

- The previous season's rankings predict worse (8.716, z +3.1) and the next season's slightly better
  (8.670, z -0.8, 17 of 28).
- The team-game near-tie is amplitude: the rankings came out narrower (offence sd 1.53 against 1.79
  among 1,000+ possessions), which that score rewards, while the order got much worse.

**Trade loss** against the incumbent: offence +0.0104, z +8.2, 2 of 30; defence +0.0152, z +9.5, 2 of 30.
Against experiment 25: +0.0071, z +5.8 and +0.0097, z +5.0, 4 of 30 each.

**Consensus:** rank agreement 0.721 / 0.768 / **0.702** against the incumbent's 0.785 / 0.803 / 0.794;
spreads 0.832 / 0.906; top five 4.  Offence and total fall below 0.75: a gross miss.

**Movement** (`outputs/movement_outside_reweighted.log`): median 1.33 per 100 in total, 96% of players past
0.1.  It has a direction, by possessions (mean signed change in total, rating then prior):

| possessions | rating | prior |
|---|---|---|
| under 500 | +2.06 | +2.08 |
| 500-1,500 | +1.97 | +2.08 |
| 1,500-4,444 | +0.55 | +0.61 |
| 4,444+ | -0.86 | -1.15 |

**2026 top 20** (`outputs/compare_incumbent_outside_labels_outside_reweighted_2026.html`, emailed):
- In, nearly all injury-shortened seasons of established players: Trae Young 128th to 15th (849
  possessions), Tatum 85th to 13th (1,467), Dejounte Murray 75th to 14th (805), Jalen Williams 57th to
  17th, Porzingis 53rd to 19th, Horford 39th to 10th; also Jarrett Allen 35th to 8th, Curry 11th to 3rd.
- Out: Diabaté 16th to 89th, Champagnie 18th to 88th, Clingan 20th to 76th, Queta 10th to 47th, Ajay
  Mitchell 7th to 39th, Cunningham 14th to 50th; Gilgeous-Alexander 5th to 12th, Ausar Thompson 6th to
  20th.

**Reading: weighting by a label's possessions is weighting by career length, and career length is
quality.**
- The rows that carry the "few possessions" part of the map are now mostly long-career players' short
  seasons -- injury years, rookie years -- whose labels are their good levels elsewhere.  The short
  seasons of short careers, which are mostly worse players with worse labels, weigh little.
- So the booster learned that a season with few possessions belongs to a good player.  Every player
  under 1,500 possessions rises about 2 points per 100 before centring pushes the heavy-minute players
  down, and injured stars with a few hundred 2026 possessions jump into the top 20.
- The same family as measurement trap 8 (a feature that says how well-measured a row is) and the
  team-movement weight: a weight that follows career length changes which players the map is learned
  on, not just how noisy they are.

**Verdict: rejected**, on every test that sees order, with a gross consensus miss.  The noise explanation
for experiment 25's loss may still be right, but label-possession weights are not the way to act on it.
Awaiting the owner.

### Experiment 26: team and game context in the prior, rated as if every player changed teams (2026-09-29)

**The change** (the owner's plan of 2026-09-28, on experiment 25's outside labels and career weights):
- **The pieces** of each season's vanilla RAPM as inputs, per side, on the prior's own luck-adjusted designs:
  the Decomposition page's by-player split (on-court rtg, teammates, opponents, context, ridge penalty), its
  by-possession split (on court signal, off court adjustment GP and DNP, team SOS adjustment) and the actual
  off-court rating, `_o` from the offensive design and `_d` from the defensive one, each padded toward 0
  over its possessions at 3,000.  `src/eracoef/pieces.py` (moved out of script 84, whose raw-points panel
  is bit-identical through it); `scripts/86_context_panel.py` writes them.  Every identity under 1e-11; a
  fold of every row reproduces the panel's pieces exactly; the panel's on-court columns come from the same
  designs (gap 0.0).
- **Game difficulty**: the pieces' opponents, context and team SOS, the closeness columns already in the
  panel, and `po_share`, the share of his possessions in the playoffs.
- **The soft same-team measure** on every training row (the owner's harmonic-mean idea): the sum over
  franchises of HM(the row's possession share with the team, the label seasons' share).  Charlotte's
  1997-2002 id is New Orleans's franchise.  Exactly 1 on every career row; across 31,772 chunk rows in 2026,
  16.2% exactly 0, 7.4% exactly 1, median 0.47.
- **Rated as if every player changed teams**: `same_team = 0` for everyone at rating time, which is also the
  only value ruling 2 allows.  The cross-fit rebuilds the pieces from each fold's training games, and the
  fold priors carry `same_team = 0` too.
- The incumbent's settings reproduce 2015 exactly on the new panel and code.

**Boruta** (`scripts/87_context_boruta.py`; the rows 62 trained on for 2026, 33,617 a side; 50 trials; the
cheap booster; each trial's mean |SHAP| over a fresh random 1,000 rows, because chimeraboost 0.34's exact
interventional SHAP made an all-rows trial take 30 minutes).  Full table:
`outputs/csv/boruta_context_table.csv`.
- Offence keeps 47 of 48 candidates (rejects `onc_d`), defence 39 of 44 (rejects pc_on_rtg_d,
  pc_on_signal_o, pc_on_rtg_o, pc_teammates_o and `chunk_seasons`, which the booster keeps by design).
- `same_team` is accepted on both sides: importance +0.53 on offence, +1.18 on defence (the best shadow
  sits at -0.35 and -0.39).
- The strongest new input on both sides is the ridge-penalty piece: +2.84 on offence (third, after career
  possessions and years), +4.93 on defence (first).  `gt_share` is second on defence (+3.70).
- Kept = accepted + tentative: feature set `boruta_context`, 45 offensive and 38 defensive names.

**Year-over-year, both directions, 56 observations** (`outputs/yoy_team_context.log`, `_vs25.log`):

| | game_armse | scale_off | scale_def | vs incumbent, team-game MSE | vs experiment 25 |
|---|---|---|---|---|---|
| incumbent | 8.6819 | 0.795 | 0.898 | -- | |
| experiment 25 | 8.6999 | 0.781 | 0.868 | +0.488, z +5.0, 14 of 56 | -- |
| **team context** | **8.6820** | 0.820 | 0.864 | +0.018, z +0.1, 28 of 56 | -0.388, z -1.9, 35 of 56 |
| -- stint level | | | | -0.083, z -0.4, 33 of 56 | |
| -- each side rescaled (order only) | | | | +0.032, z +0.2, 28 of 56 | |

- By direction it splits: a season's ratings predict the season BEFORE it better than the incumbent's do
  (8.658 against 8.677, -0.49, z -2.5, 18 of 28) and the season AFTER it worse (8.706 against 8.687,
  +0.53, z +2.0, 10 of 28).  **Corrected 2026-09-30**: first written the other way round.  63_yoy.py's block
  "the NEXT season's rankings predict this season" is a rating looking back at the season before it.  The
  age tilt fits the corrected direction: against the incumbent, players 34 and over rise 0.20 and players
  under 25 fall 0.08-0.09, which is right for the season before and wrong for the season after (the split
  by player below).

**Trade loss** against the incumbent: offence +0.0045, z +3.5, 6 of 30; defence +0.0060, z +5.4, 4 of 30.
Against experiment 25: +0.0011, z +1.3 and +0.0005, z +0.5 -- ties.

**Consensus:** rank agreement 0.769 / 0.777 / 0.757 against 0.785 / 0.803 / 0.794; spreads 0.953 / 0.989;
top five 3 against 4.  Team R-squared on defence 0.123 against 0.164 (the consensus's own is 0.160): the
defensive ratings lean less on the team.  A drop, not a gross miss.

**What "changed teams" does** (`outputs/season_ratings_team_context_same_team_diag.parquet`, the SPM prior
before the ridge at `same_team = 0` against the player's real value):
- Defence: median move 0.02, ninth decile 0.60, 40.5% of players past 0.1; the real value would make
  the defensive prior better by 0.115 on average, and the move follows how much of his team he kept
  (correlation -0.42).  Players who stayed lose the team's defensive credit.
- Offence: median 0.005, ninth decile 0.15 -- nearly nothing.

**Movement against the incumbent**: median 0.30 per 100 on offence, 0.45 on defence, 0.59 total; 91% of
players past 0.1; almost all of it order, not spread.

**2026 top 20** (`outputs/compare_incumbent_outside_labels_team_context_2026.html`, emailed):
- Jokic 1st, Gilgeous-Alexander 2nd, Antetokounmpo 3rd, Wembanyama 4th -- his defence falls from 5.9 to 3.5.
- In: Caruso 51st to 5th (defence +2.4), Cason Wallace 32nd to 8th (defence +2.3), Butler 36th to 9th,
  Dyson Daniels 22nd to 12th, Anunoby 24th to 14th, Paul George 74th to 17th, Fox 49th to 18th, Adebayo
  29th to 20th.
- Out: Doncic 12th to 56th (both sides), Holmgren 8th to 43rd (defence 3.5 to 1.4), Queta 10th to 38th,
  Towns 9th to 30th, Diabaté 16th to 74th, Champagnie 18th to 55th, Jamal Murray 17th to 28th, Ajay
  Mitchell 7th to 21st.
- Oklahoma City's defenders rise while its young centre falls: partly the pattern that sank the off-court
  record in 2026-09-14's eye test.

**Reading.**
- The context wins back all of experiment 25's loss on the year-over-year test and ties the incumbent;
  against its own base it is better (z -1.9).  It does not beat the incumbent on any test that decides.
- "Rated as if he changed teams" acts almost only on defence, and in the direction the idea predicted:
  one-team players lose defensive credit that belongs to their team.
- The trade loss still says worse (z +3.5 and +5.4, the same size as experiment 25's), and the
  consensus drops without a gross miss.

**Verdict: not adopted** -- a tie on the test that decides goes to the simpler version, the incumbent, and
the trade loss and the consensus lean against.  Awaiting the owner; experiment 27 (a team-season
intercept) is next in the plan.

### Experiment 27: experiment 26 plus one random intercept per team-season (2026-09-30)

**The change** (the owner's plan; go given 2026-09-29).  Every booster fit of experiment 26 -- the full fit
and the five player folds, both sides -- becomes F(inputs) + b[team-season]:
- A training row's group is its main team-season, the (franchise, season) holding most of its possessions;
  the career row's is the biggest of his label span.  832 team-seasons in 2026's training rows.
- chimeraboost 0.34's own `random_effects=True` refuses a bagged model and the offensive booster is a bag of
  five, so the algorithm runs by hand with chimeraboost's own solver (`fit_with_team_season` in script 62):
  fit F; `estimate_ratio_reml` and `solve_intercepts` on its residuals; refit F from scratch on y - b.
- The weights are normalised to mean 1 for the solve only: read as row counts, possession weights switch
  the shrinkage off (chimeraboost's own fit with them gives intercepts of sd 0.27 against 0.17 normalised).
- Rated with the trees alone.  The rated season is never a training season, so its team-seasons carry no
  intercept: every player is rated as if on an average team, consistent with "changed teams".
- **The check against chimeraboost's own version** (2026 defensive rows, both on normalised weights):
  intercepts correlate 0.980 (sd 0.166 against 0.174), variance ratio 8.2 against the helper's 7.6 after its
  refit, trees-only predictions correlate 0.988 (median gap 0.13 per 100).  The trees differ more than the
  intercepts because chimeraboost replays its first fit's tree structures on y - b and the helper refits.

**The intercepts are small and never absent**: the variance ratio (noise over team-season variance, in rows)
runs 17-25 on offence and 12-22 on defence across the 30 seasons, never infinite; the intercepts' sd is 0.04
per 100 on offence and 0.16 on defence (0.14-0.23).

**Year-over-year, both directions, 56 observations** (`outputs/yoy_team_intercept.log`, `_vs26.log`):

| | game_armse | scale_off | scale_def | vs incumbent, team-game MSE | vs experiment 26 |
|---|---|---|---|---|---|
| incumbent | 8.6819 | 0.795 | 0.898 | -- | |
| experiment 26 | 8.6820 | 0.820 | 0.864 | +0.018, z +0.1, 28 of 56 | -- |
| **team-season intercept** | **8.6788** | 0.824 | 0.866 | -0.072, z -0.4, 27 of 56 | -0.090, z -1.3, 33 of 56 |
| -- stint level | | | | -0.211, z -1.0, 34 of 56 | -0.128, z -1.3 |
| -- each side rescaled (order only) | | | | -0.057, z -0.3, 26 of 56 | -0.089, z -1.1 |

- The same split by direction as experiment 26, a little stronger: a season's ratings predict the season
  before it better than the incumbent's do (8.656 against 8.677, -0.54, z -2.8, 17 of 28) and the season
  after it worse (8.701 against 8.687, +0.39, z +1.6, 10 of 28).  (Corrected 2026-09-30; see experiment 26.)

**Trade loss** against the incumbent: offence +0.0045, z +3.8, 6 of 30; defence +0.0051, z +3.9, 7 of 30.
Against experiment 26: offence 0.0000, z 0.0; defence -0.0009, z -1.2, 22 of 30.

**Consensus:** 0.765 / 0.782 / 0.758 (incumbent 0.785 / 0.803 / 0.794; experiment 26 0.769 / 0.777 /
0.757); spreads 0.951 / 0.982; top five 4.

**Movement against experiment 26**: median 0.11 per 100 on offence, 0.26 on defence; against the incumbent
it keeps experiment 26's size.

**2026 top 20** (`outputs/compare_incumbent_team_context_team_intercept_2026.html`, emailed):
- Jokic 1st, Wembanyama back to 2nd (defence 5.3, from experiment 26's 3.5), Antetokounmpo, Curry,
  Gilgeous-Alexander, Leonard, Harden 7th, Dyson Daniels 8th, Ausar Thompson, Cunningham.
- The intercept undoes part of experiment 26's worst moves: Holmgren 43rd to 17th, Doncic 56th to 22nd,
  Caruso 5th to 11th, Cason Wallace 8th to 16th.
- Still out of the incumbent's top 20: Ajay Mitchell 24th, Towns 30th, Queta 40th, Diabaté 44th,
  Champagnie 54th, LeBron James 33rd, Jamal Murray 25th; Dejounte Murray is 18th on 805 possessions.

**Reading.**
- The best year-over-year score of the three (8.6788) and the best ordering against experiment 26, but a tie
  with the incumbent on every row that decides, and the trade loss still says worse at z +3.8 / +3.9.
- The team-season effects the booster was carrying were small (sd 0.16 per 100 on defence), and taking them
  out repairs part of the 2026 list's damage at the top.
- **The one result that holds across experiments 26 and 27: their ratings predict the season BEFORE the
  rated one better than the incumbent's do (z -2.5, z -2.8) and the season after it worse.**  Pooled over
  both directions it is a tie.  (Corrected 2026-09-30: first reported the other way round.)  The split by
  player below traces it to age.

**Verdict: not adopted** -- a tie on the test that decides, the trade loss against.  The plan's three
experiments are done: outside labels cost the year-over-year test z +5.0 on their own; team context and
then the team-season intercept win all of it back and a little more, but never beat the incumbent.

### Where experiments 26 and 27 win and lose: by player quality, by team change and by age (2026-09-30)

**The owner's follow-ups** (2026-09-30): check the direction split for players who changed teams, look at aging,
and **split every result by player-quality tier: top 30, 31-90, 91-150, 151-300, 301+ -- "in general we should
do this."**

**The tool.** `scripts/88_yoy_by_player.py` shares each team-game's squared-error difference (candidate minus
the incumbent) among the players on the floor by their possessions in it, and adds the shares up by player group.
The groups add up to the year-over-year test's own paired difference in every season and direction (largest gap
5e-15), so each group's number is its part of the result.  Tiers are the incumbent's rank by total rating in the
rated season; team change is the main team in the scored season against the rated one; age is in the scored
season.  `scripts/73_tradeloss.py --quality=<rankings> --tier=each` splits the trade loss the same way.
Logs: `outputs/yoy_by_player_exp26_27.log`, `outputs/tradeloss_quality_exp26_27.log`.

**First, a correction.**  Experiments 26 and 27 predict the season BEFORE the rated one better and the season
after worse -- the reverse of what was first reported.

**Experiment 27 against the incumbent, team-game level, each group's part of the paired difference** (below zero
= experiment 27 better; "looking back" = a season's ratings predicting the season before it):

| group | share of possessions | looking back | z | looking forward | z | both | z |
|---|---|---|---|---|---|---|---|
| top 30 | 0.12 | -0.060 | -2.0 | +0.061 | +1.8 | +0.001 | +0.0 |
| 31-90 | 0.19 | -0.115 | -2.6 | +0.102 | +1.8 | -0.007 | -0.2 |
| 91-150 | 0.16 | -0.064 | -1.7 | +0.080 | +2.4 | +0.008 | +0.3 |
| 151-300 | 0.29 | -0.169 | -2.8 | +0.085 | +1.2 | -0.042 | -0.9 |
| 301+ | 0.16 | -0.112 | -3.1 | +0.023 | +0.5 | -0.045 | -1.4 |
| no rating that season | 0.08 | -0.017 | -1.1 | +0.042 | +1.9 | +0.013 | +0.9 |
| stayed on his team | 0.64 | -0.359 | -2.7 | +0.217 | +1.4 | -0.071 | -0.7 |
| changed teams | 0.28 | -0.161 | -2.8 | +0.134 | +1.8 | -0.014 | -0.3 |
| 23 and under | 0.21 | -0.142 | -3.0 | +0.080 | +1.5 | -0.031 | -0.8 |
| 24-26 | 0.27 | -0.150 | -2.7 | +0.047 | +0.7 | -0.052 | -1.1 |
| 27-29 | 0.24 | -0.139 | -3.0 | +0.071 | +1.2 | -0.034 | -0.8 |
| 30-32 | 0.16 | -0.084 | -2.3 | +0.079 | +2.0 | -0.002 | -0.1 |
| 33+ | 0.12 | -0.023 | -0.8 | +0.116 | +2.7 | +0.047 | +1.7 |
| all | 1.00 | -0.537 | -2.8 | +0.393 | +1.6 | -0.072 | -0.4 |

- **Not a team-change effect.**  Per unit of possession share, players who changed teams gain looking back and
  lose looking forward exactly as much as players who stayed (-0.58 against -0.56 back, +0.48 against +0.34
  forward).  "Rated as if he changed teams" helps movers no more than stayers.
- **Age, as the owner suspected.**  Experiments 26 and 27 rate older players higher and younger players lower
  than the incumbent (34 and over +0.20 per 100, under 25 -0.08 to -0.09).  That is right looking back -- a year
  earlier the veterans were better and the young players worse -- and wrong looking forward.  The loss looking
  forward is biggest for players 33 and over (+0.93 per unit of share, z +2.7; experiment 26 z +3.4), and over
  both directions they are the one group experiments 26 and 27 do worse on (z +1.7 and +2.6).  The gain looking
  back is biggest, per unit of share, for players 23 and under.  Stint level says the same (under 24, looking
  back: z -5.7; 30 and over, looking forward: z +2.4).
- **Quality:** the top 150 are a wash over both directions; what little experiment 27 gains comes from players
  ranked 151st and below.
- **The trade loss by tier disagrees about the lower tiers**: experiment 27 ties the incumbent on the top 30
  (z +0.5 / +0.9) and is worse in every tier below it -- 31-90 z +2.7 / +1.0, 91-150 +2.6 / +2.1, 151-300
  +3.4 / +3.5, 301+ +2.9 / +2.4 (offence / defence).

**Reading.**  The direction split is an age tilt the new labels brought in, and it is not neutral: it costs the
veterans' forward predictions.  Aging is the lead the owner named, and it is a candidate for its own experiment.

### Experiment 28: experiment 27 with every chunk's label moved to the chunk's own age (2026-09-30)

**The change** (the owner's go, 2026-09-30).  An outside label is fit on the player's other seasons, which are at
other ages, so it describes him at those ages.  Each chunk's label now gains `k x (curve at the chunk's ages -
curve over the label's seasons)`, both possession-weighted, where the curve is an aging curve and `k` puts it on
the label's own scale (a ridge label moves by n / (n + penalty) of it; the un-shrunk defensive label by all of it
above its floor).  Career rows never move.  Everything else is experiment 27.  `--age_adjust_labels=1`,
`singleyear.AgingCurve`, `chunk_rows(age_curve=...)`.
- **The curve**: the delta method on single-season RAPM at penalty 100 (`outputs/piece_panel.parquet`), pairs of
  consecutive seasons weighted by harmonic possessions, the change regressed on age with a quadratic and
  integrated from 27; fit per rated season without the seasons it is scored on (10,575 pairs for 1997).
- For 1997, against age 27, in points per 100: offence 20 -2.2, 23 -0.4, 25 0.0, 30 -0.9, 33 -2.5, 36 -4.7;
  defence (points allowed) 20 +1.2, 23 +0.3, 30 +0.3, 33 +0.8, 36 +1.4.
- The labels moved as intended: chunks at 33 and over lost 1.1 on offence and 0.6 on defence on average, chunks
  at 24-29 gained about 0.2; median move 0.16-0.22 per 100.

**Results** (`outputs/yoy_age_adjusted.log`, `_vs27.log`, `outputs/tradeloss*_age_adjusted*.log`,
`outputs/yoy_by_player_age_adjusted.log`):

| test | against the incumbent | against experiment 27 |
|---|---|---|
| year-over-year error | **8.721** against 8.682: +1.09, z +4.1, 17 of 56 | +1.16, z +4.0 |
| -- each side rescaled (order only) | +0.27, z +1.0 | +0.33, z +1.0 |
| -- looking back (a season's ratings predicting the one before) | +1.58, z +4.0, 6 of 28 | |
| -- looking forward | +0.60, z +1.8, 11 of 28 | |
| trade loss, offence / defence | +0.0093 z +5.1 / +0.0060 z +4.1 | z +3.0 / +0.8 |
| consensus (offence / defence / total) | **0.814 / 0.778 / 0.818** against 0.785 / 0.803 / 0.794; top five 3 | |

- Worse in every quality tier (year-over-year, both directions: z +3.0 to +4.6) and in every age group (z +3.4 to
  +4.6); the trade loss is worse in every tier as well.
- Wider: the scored seasons want offence x0.76 and defence x0.85 (incumbent x0.79 / x0.90).
- Players moved a median of 0.72 per 100 in total against the incumbent.
- The one test it improves is agreement with the public metrics, the best in this series (offence 0.814, total
  0.818).  The consensus is a sanity check and never a target.

**2026 top 20** (`outputs/compare_incumbent_team_intercept_age_adjusted_2026.html`, emailed): the veterans collapse
-- Curry 11th to 92nd, Durant 15th to 136th, LeBron James 19th to 80th, Harden 13th to 35th, Butler 36th to 85th,
Giannis Antetokounmpo 4th to 58th (on 2,129 possessions) -- and young players rise: Ausar Thompson 4th, Cunningham
5th, Dyson Daniels 6th, Evan Mobley 15th, VJ Edgecombe (20) 16th, Amen Thompson 18th.

**Reading.**
- The adjustment over-corrects.  Experiments 26-27 tilted veterans up; this tilts them far down.  It applies the
  AVERAGE decline to every veteran, but a player still in the league at 37 is one who has aged better than average
  (the pairs behind the curve do not include the ones who fell out), and the booster, which has no age input of
  its own, carries the correction through years of experience -- so every long career is marked down.
- It hurts looking back most (z +4.0): a year earlier the veterans were better than their new ratings say.
- Ratings that match each player's current age better (the consensus agrees more) predict the neighbouring
  seasons worse.  The year-over-year test rewards a player's level around the season more than his level on the
  day.

**Verdict: rejected** -- clearly worse on the test that decides, on the trade loss and on the eye test.  With it,
the line that began at experiment 25 ends: every version built on outside labels ties or loses to the incumbent.

### Experiment 29: the owner's team-movement weight, with a floor and rebalanced within career bands (2026-09-30)

**The change** (the owner, 2026-09-30: *"I just can't abide that there's no way to trade weight"*).  The
incumbent, with every training row's weight multiplied by the player's team movement (`1 - sum(share^2)`, the
chance two of his possessions came from different teams, franchises mapped) **plus a floor of 0.5**, and then
**rebalanced within career-length bands** (label possessions 0 / 2,000 / 5,000 / 15,000 / 40,000+), each band
keeping the total weight it had.  The one earlier run (floor 0, no bands) dropped every one-team player and let
the weight drift toward long careers.  Weight factors: Curry and Jokic 0.50, LeBron James, Harden and Durant 1.08,
Chris Paul 1.25.  In 1997 nobody sits at zero weight, one-team players' share of the weight goes 10.4% to 5.6%,
and every band's share is unchanged.  `--trade_weight=0.5 --trade_bands=1`, `reweight_by_movement(bands=)`.

**Results** (`outputs/yoy_trade_weight.log`, `outputs/tradeloss*_trade_weight.log`,
`outputs/yoy_by_player_trade_weight.log`, `outputs/consensus_trade_weight.log`):

| test | against the incumbent |
|---|---|
| year-over-year error | **8.6694 against 8.6819: -0.343, z -4.1, 39 of 56** |
| -- stint level | -0.287, z -2.8, 36 of 56 |
| -- each side rescaled (order only) | -0.144, z -1.7, 32 of 56 |
| -- looking back (a season's ratings predicting the one before) | -0.590, z -5.1, 22 of 28 |
| -- looking forward | -0.095, z -0.9, 17 of 28 |
| trade loss, offence / defence | -0.0006 z -0.9 / -0.0004 z -0.5 (ties) |
| consensus (offence / defence / total) | 0.782 / 0.809 / 0.792 against 0.785 / 0.803 / 0.794; top five 4 (same) |

- **Better in every group over both directions**: every quality tier (z -3.2 to -4.2), every age group (-3.1 to
  -4.8), players who stayed (-4.1) and players who moved (-3.6).  Looking back it is better in every group at
  z -4 to -6; looking forward every group is a slight, insignificant gain.
- The trade loss by tier is a wash: defence 31-90 worse (z +2.1), defence 91-150 and 301+ better (z -2.3, -2.5),
  offence 151-300 better (z -2.0), the rest ties.
- Movement against the incumbent: median 0.15 per 100 on offence, 0.26 on defence, 0.32 total.
- **2026 top 20** (`outputs/compare_incumbent_trade_weight_2026.html`, emailed): the top 13 are the incumbent's
  players in a slightly different order (Wembanyama, Leonard, Gilgeous-Alexander, Jokic, Antetokounmpo, Towns,
  Ausar Thompson, Curry, Ajay Mitchell, Durant, Harden, Doncic, Holmgren); in: Butler 36th to 14th, Adebayo,
  Anunoby, Tobias Harris, Hartenstein, Jarrett Allen; out: Jamal Murray to 22nd, Cunningham 29th, Champagnie,
  Diabaté, LeBron James 36th, Clingan 42nd.  Wembanyama's defence 5.9 to 5.0.

**Reading.**  The first candidate since 2026-09-18 to clear the decision rule: z -4.1 on the test that decides, no
consensus miss, the trade loss a tie, and the 2026 list not worse to the eye.  The gain sits almost entirely in
predicting the season before; why a movement weight should help that direction and not the other is not known.
**It down-weights one-team players, which the owner said the same day they do not want** (they want them to count
equally and the model to adapt at rating time; experiment 30 is that design).

**Verdict: meets the adoption rule.  Awaiting the owner.**

### Experiment 30: every stretch labelled by the seasons right next to it, rated as if traded (2026-10-01)

**The change** (the owner, 2026-09-30: one-team players should count equally, and at rating time the model should
adapt to being traded; their idea of chunks with trades in the middle).  The chunks are replaced by windows of two,
four or six consecutive seasons split in the middle, each half predicting the other: its box score is the
features, its label a RAPM fit on the other half's seasons alone (penalty 3,000, the defensive side un-shrunk as
the career label is).  The career row stays.  Weights are the incumbent's (a player's window rows together weigh
his career row's weight), so one-team players count in full.  The same-team measure is an input; everyone is rated
with it at 0, "as if traded".  `--chunk_label=adjacent --features=boruta_same_team`, `singleyear.adjacent_rows`.
- In 2026's training rows: 241 window labels a side, 44,122 window rows, **42% of them traded examples** (same
  team under 0.5).  The window labels are far wider than the career labels on offence (sd 1.56 against 0.59: the
  career label on offence is fit at penalty 40,000) and about as wide on defence (1.92 against 1.76).
- The same-team measure does move one-team players.  For players whose rated season's teams overlap their other
  seasons' teams by 0.8 or more (20% of possessions), rating them at that real value instead of 0 raises the prior
  by +0.26 on offence and +0.17 on defence on average (possession-weighted; median move 0.2-0.3 a side).  For the
  40% of possessions whose real value is 0.2 or less it moves nothing (median 0.000).
  (`outputs/season_ratings_adjacent_labels_same_team_diag.parquet`.)

**Results** (`outputs/yoy_adjacent_labels.log`, `outputs/tradeloss*_adjacent_labels.log`,
`outputs/yoy_by_player_adjacent_labels.log`):

| test | against the incumbent |
|---|---|
| year-over-year error | 8.7055 against 8.6819: +0.650, z +3.4, 16 of 56 |
| -- stint level | -0.001, z 0.0 |
| -- each side rescaled (order only) | **-0.333, z -1.6, 34 of 56** |
| -- looking back / looking forward | +1.04, z +4.1 / +0.26, z +0.9 |
| trade loss, offence / defence | +0.0034 z +2.8 / **+0.0086 z +7.6** |
| consensus (offence / defence / total) | **0.834 / 0.788 / 0.829**, the best yet; top five 4 |

- The ratings are too wide for the neighbouring seasons (they want offence x0.77 and defence x0.85, the
  incumbent x0.79 / x0.90): the team-game loss is spread, while the order alone leans better (z -1.6).
- Worse in every quality tier, and **equally for players who changed teams and players who stayed** (per unit
  of possession share +0.67 and +0.66): it loses as much on the players who changed teams, for whom "as if
  traded" was the right setting, as on those who stayed.  That compares it with the incumbent, not with itself
  rated as if every player stayed, so it does not isolate the same-team measure.
- 2026: Wembanyama's defence 7.6; Towns 3rd, Doncic 6th, Donovan Mitchell 8th, Jamal Murray 10th, Gobert 15th;
  LeBron James 126th, Tobias Harris 106th, Ajay Mitchell 55th, Durant 42nd.

**Verdict: rejected** on the test that decides and the trade loss, though it agrees with the public metrics best of
any version.  The owner asked the same morning whether the trade flag is tested by switching it on only for players
who were actually traded: it is not -- everyone gets it -- and that is the right test of whether the flag works.
Experiment 30b runs it.

### Experiment 30b: the trade flag switched on only for the players who moved (2026-10-01)

**The question** (the owner, 2026-10-01): "are you testing the trade flag by turning it on for players who are
traded?"  No: experiment 30 rated every player as if traded, and the year-over-year test scored that one rating for
everyone, the 64% of possessions whose player kept his main team included.

**The test.**  Experiment 30 rebuilt with `--rate_same_team=both`: the same models write two lists, every player
rated as if traded (same_team 0; bit-identical to experiment 30's table in all 30 seasons) and as if he stayed
(same_team 1, `outputs/season_ratings_flagtest_stayed.parquet`).  `scripts/89_stitch_by_move.py` then gives each
player the "traded" rating if his main team in the scored season differs from the rated season's and the "stayed"
one if not (62% of possessions take "stayed"), one list per direction, which `63_yoy.py` and `88_yoy_by_player.py`
read as `name=<forward>|<backward>`.  This matched list reads the neighbouring season's teams, so it tests the
flag and can never be a ranking.  Switching a player from "traded" to "stayed" moves his 2026 total by a median
0.55 per 100 (one player in ten 1.4 or more), and the season's own games stretch the stayed prior less (offence
about x2.1 against x3.0).

**Results** (`outputs/yoy_flagtest.log`, `outputs/yoy_flagtest_vs_incumbent.log`, `outputs/yoy_by_player_flagtest*.log`,
`outputs/tradeloss_flagtest*.log`, `outputs/consensus_flagtest.log`; the first row is paired from
`outputs/yoy_flagtest.parquet`):

| comparison | team-game (decides) | stint | order only (each side rescaled) |
|---|---|---|---|
| matched against all "stayed": only the movers differ, so this is the flag | +0.048, z +0.8, 30 of 56 | -0.009, z -0.1 | -0.176, z -2.9, 38 of 56 |
| all "stayed" against all "traded" | **-0.697, z -5.7, 45 of 56** | -0.998, z -6.0 | -0.540, z -3.7 |
| all "stayed" against the incumbent | -0.046, z -0.3, 28 of 56 | **-0.999, z -5.1, 43 of 56** | -0.873, z -5.3 |
| matched against the incumbent | +0.002, z 0.0 | -1.008, z -5.4 | -1.049, z -6.8 |

- **The flag does not help the players it describes.**  Giving the players who changed teams their "traded" rating
  instead of their "stayed" one is a tie on the test that decides (looking forward, the case a trade is about:
  +0.085, z +1.0, 15 of 28), and a small gain only once each side's spread is matched.
- **Rating everyone as if traded is what sank experiment 30.**  The same models rating everyone as if they stayed
  go from clearly worse than the incumbent to a tie on the test that decides, and clearly better at stint level:
  in every quality tier, most in 151-300 (z -5.3) and 301+ (z -6.3), i.e. mostly the bench, which the team-game
  level cannot see.  At team-game level every tier, mover group and age group is a tie.
- Trade loss, stayed against the incumbent: offence -0.0036, z -4.4 (24 of 30), better; defence +0.0044, z +4.1
  (5 of 30), worse -- worse in 31-90 and 151-300, a tie in the top 30.  Against "traded": better on both ends
  (z -6.4 / -5.1).
- Consensus, stayed: **0.855 / 0.843 / 0.864**, the best of any version (incumbent 0.785 / 0.803 / 0.794); top
  five 4.
- 2026, stayed: Wembanyama 9.3, Kawhi Leonard 8.3, Jokic 7.9, Giannis 6.4, Gilgeous-Alexander 6.3.  Durant 72nd
  (the incumbent 15th), LeBron James 65th (19th), Cade Cunningham 41st (14th), Clingan 44th (20th); Gobert 20th
  (47th), Hartenstein 11th (26th).

**Verdict.**  The flag: no help, measured where it applies.  The stayed list: a tie on the test that decides, so
not adopted under the rule; its stint-level and consensus gains against a worse defensive trade loss are the
owner's call.

### Every list split by traded and not-traded players (2026-10-01)

The owner: "can you split all our results into traded/not traded and get the accuracy?"  `88_yoy_by_player.py`
now also shares each team-game's own squared error among the players on the court (`tg_abs`), so a group gets its
own typical miss; `73_tradeloss.py --movers=1` splits the trade loss by whether the player's main team changed the
season before or after.  Logs: `outputs/yoy_by_player_split_by_trade.log`, `outputs/tradeloss_split_by_trade.log`.

- No list does better on traded players in particular.  The one clear winner, 29 (trade weight), wins by the same
  amount on both groups (-0.013 each; 37 and 41 of 56).  Rating as if traded (30) loses as much on traded players
  as on the rest (+0.024 each); stayed and matched tie the incumbent on both.
- Trade loss: the stayed list is clearly better than the incumbent on offence for traded players (0.322 against
  0.329, 24 of 30) and worse on defence; 29 ties everywhere; every other list is worse.
- Traded players are barely harder to predict than the rest: the incumbent misses by 8.672 with them on the court
  and 8.655 without a team change.

### What drives the gap between the traded and stayed ratings: a lasso (2026-10-01)

The owner read the gap (traded minus stayed total) player by player in 2026 and asked for its causes, "lasso
first".  Lasso on the standardised inputs the prior models read, plus age, teams that season and the team's
ratings with him off the court; the penalty is the sparsest within one standard error of the cross-validated best
(`scratch/2026-10-01_experiments_25_to_30b/why_traded_differs.py`; every coefficient in
`outputs/lasso_traded_minus_stayed.csv`).

- It explains about half: R2 0.53 cross-validated, in 2026 (582 players) and over all 30 seasons (14,579).
- Toward "better off traded", per standard deviation (2026 / all seasons): on-court possessions +0.42 / +0.33,
  team points allowed with him on the court +0.35 / +0.26, steals plus blocks +0.10 / +0.12; over all seasons
  also three-point rate +0.27 and points +0.20.
- Toward "worse off traded": team points scored with him on the court -0.27 / -0.33, seasons in the league
  -0.20 / -0.27, seasons with his current team -0.15 / -0.21.
- Reading: the traded rating gives less credit for the team's results with him on the court and more for heavy
  minutes and box-score production; veterans and long-tenured players lose most.  For the veterans, one tilt is
  mechanical -- window rows weigh by the feature half's possessions, so for players 33 and older who changed teams
  57% of the weight is on "season 1 predicts season 2" rows (47-49% for 23 and under) -- but the larger suspect is
  selection: the veterans who did move were often declining.

### The swap test: does a ranking order teammates the way the games do? (2026-10-01)

**The owner's goal** (2026-10-01): rank players among their own team, as an adjustment on top of PI-RAPM, judged
by "player vs replacement in lineups" -- *"every 5-man group that Brandin Podziemski is in, find the times those 4
players played with someone else, and find his delta vs that player, adjusting for game context."*  This first
step builds the test; the adjustment comes next.

`src/eracoef/swaptest.py`, `scripts/90_swap_test.py`, `tests/test_swaptest.py` (seven tests on a synthetic league
whose ratings are known).  For every pair of lineups on the same team that share four players, the difference in
their results in the scored season is set against what a NEIGHBOURING season's ranking says about the two swapped
players -- never its own season, which the rating was fitted on.  1998-2025, scored by the season before and the
season after: 56 observations.  Only pairs where both swapped players carry a rating.  A pair's weight is its
information, `n_a * n_b / (n_a + n_b)` possessions.  Two scores:

- **order** (the owner's pick of the five questions): the swap difference with the opponents faced and the context
  taken out -- once per scored season, by that season's own plain RAPM, identically for every ranking -- signed by
  the ranking's gap between the two swapped players.  Higher is better, 0 is no idea, and it ignores spread.
- **gaps**: the squared error of each ranking's own prediction of the swap difference.  Lower is better.

The context refit on the scored season: intercept, home, playoffs, garbage time, margin, margin x time left,
minutes into the period, its last two minutes, minutes each five had been on the court, and the clutch.

**Size.**  Per observation about 70,000 swap pairs and 970,000 possessions of information (net).  95% of a
regular's possessions have a same-four match, but one player's pooled swap difference is noisy (median regular
+/-4.3 per 100; Podziemski 2026 -2.9 +/-3.6 against a rating gap of -1.3 for the same comparisons): large samples
for a test, small for one player.

**Results** (net, full context, against the incumbent; "better in" counts observations of 56):

| ranking | order | order vs incumbent | gaps vs incumbent |
|---|---|---|---|
| the incumbent (PI-RAPM) | 1.685 | -- | -- |
| the box prior alone (`prior_off` / `prior_def`) | 1.577 | -0.108, z -6.4, incumbent better in 44 | +1.70, z +6.7 |
| plain single-season RAPM, penalty 3,000 | 1.580 | -0.105, z -2.3, incumbent better in 32 | -0.76, z -1.3 (a tie) |
| the incumbent shuffled inside each team | 0.098 | -1.587, z -20.7 | +31.0, z +22.6 |

- **Pairs who were teammates in the rated season too** (44% of the information): the prior loses (-0.127, z -5.2);
  plain RAPM ties on order (-0.033, z -0.5) and predicts the gaps slightly better (-1.46, z -2.5, 35 of 56).
- **Pairs who were not** (56%): the prior loses (-0.089, z -3.5), and so does plain RAPM (-0.164, z -2.8).
- Offense and defense apart, plain RAPM loses clearly (-0.262, z -8.4; -0.220, z -6.9): its net is better than its
  split.  Home-and-intercept context instead of the full one changes no conclusion (prior -0.100, z -6.1; plain
  RAPM -0.163, z -3.6).

**The checks.**  Shuffled inside each team: loses at z -20.7, so the test sees the order inside a team.  A random
constant per team of the SCORED season: -0.056 (z -3.5), and -0.014 (z -0.9) on pairs who were teammates then --
what is left is players on two teams that season.  The same per team of the RATED season: exactly 0 on pairs who
were teammates then and -0.531 (z -7.0) on the rest, so more than half the test is whether a player's credit
travels with him to new teammates.

**Spread inside a team.**  The slope the swaps ask of the incumbent's within-team gaps is 0.69 net -- 0.76 on
offense and 0.97 on defense.  A third reading, after the trade set (offense at 0.73 of defense, 29 of 30 seasons)
and the year-over-year sweep (x0.65 / x0.85), that offensive gaps are too wide against defensive ones.  All three
are across seasons, so they mix "too wide" with "players change".

**The context terms.**  The minutes a five had been on the court came out the opposite of tiredness, with the
time into the period controlled: +1.75 points per 100 per minute for the offense, -0.79 for the defense (a five
that has been on longer allows less).  It behaves like an artifact of how stretches begin at substitutions, or of
coaches leaving a lineup in while it goes well, not like fatigue.  The clutch term averages -0.06.  Neither changes
a conclusion.

**Verdict.**  PI-RAPM orders teammates better than either of its parts alone.  For pairs who were already teammates,
plain RAPM orders them as well and sizes their gaps slightly better -- that is the room the within-team adjustment
has to work in.  Logs: `outputs/swaptest_first.log`, `outputs/swaptest_first.parquet`.

### The swap adjustment: re-split each team's credit by its lineup swaps (2026-10-01)

**The owner's plan** (all four recommendations, "go"): an adjustment on top of PI-RAPM, by player type first and each
player's own swaps on top, each team's total held fixed, judged on the swap test.  `src/eracoef/swapadjust.py`,
`scripts/91_swap_adjust.py`, `tests/test_swapadjust.py` (four tests on the synthetic league: team totals stay put, a
planted split between two teammates is found and undone, a planted feature is recovered, a fold's model is exactly
the model of the other folds' pairs).

- Every swap of the rated season leaves a residual: the swap difference PI-RAPM's own prediction (context without the
  time-on-court term) did not account for.
- **By type**: a linear model of each swap's residual on the two swapped players' feature difference -- the season
  panel's padded box rates, share of team possessions, starts share, garbage-time share, age, experience, height,
  weight, and PI-RAPM's box prior.  Five player folds (a player's correction from the fold that never saw his swaps)
  and seasons at least two away from the rated one, so neither scored season reaches it (ruling 2).
- **By player**: his own swaps, pulled toward the type prediction by a penalty.
- **Each team's total fixed exactly**: corrections sum to zero weighted by each player's possession share of that
  team (a constrained least squares, so players on two teams are handled).

**A trap, caught on the first run.**  With PI-RAPM's own residual (`u_off` / `u_def`) among the features, the type
model put +2.0 and +1.9 on it: the swap residuals are read in the same season, so the evidence the ridge held back is
still in them, and the model learned to multiply it.  That is a lighter ridge penalty labelled as a player type --
median move 0.9 per 100 -- and the penalty is the year-over-year test's to choose.  Removed.

**The grid** (net order score against the incumbent, swap test, 56 observations):

| arm | order | gaps | within-team slope (incumbent 0.72) |
|---|---|---|---|
| type x0.5 | **+0.162, z +5.5, 42 of 56** | +0.23, z +0.7 (a tie) | 0.67 |
| type x1 | +0.191, z +4.9 | +5.16, z +7.9 (worse) | 0.56 |
| type x1.5 / x2 / x3 | +0.147 / +0.062 / -0.023 | worse | 0.46 / 0.38 / 0.27 |
| own swaps, penalty 30,000 | +0.059, z +2.7 | a tie | 0.70 |
| own swaps, penalty 1,000 | -0.205, z -3.6 | worse | 0.28 |
| type x1 + own, penalty 30,000 (best order) | +0.210, z +5.0 | +5.47, z +8.4 (worse) | 0.56 |

The type strength peaks inside the grid; a player's own swaps add almost nothing on top of the type model and hurt
when lightly shrunk.  **Type x0.5 is the candidate**: most of the order gain and no worse on gaps -- better on
offense (order z +7.8, gaps z -3.7) and on defense (order z +9.7, gaps z -6.7), a tie on net gaps.

**What the type model says** (fitted on every season; per standard deviation of the difference from the teammate he
swaps with, points per 100 beyond what PI-RAPM credited; `outputs/swapadj_types.csv`).  Offense: made threes +1.62,
made twos +1.31, missed twos -1.05, missed threes -0.99, assists +0.77, turnovers -0.74, made free throws +0.66,
offensive rebounds +0.62.  Defense: defensive rebounds +0.54, garbage-time share -0.52, fouls +0.42, steals +0.31,
offensive rebounds -0.31, made threes -0.29, the box prior's defense +0.35.

**The battery on type x0.5** (`outputs/season_ratings_swapadj_x05.parquet`, "type_half" on the comparison page):

| test | result |
|---|---|
| year-over-year, team-game level | **+0.429, z +3.5, 16 of 56: worse** (8.698 against 8.682) |
| year-over-year, stint level | -0.194, z -1.3: a tie |
| year-over-year, each side rescaled to the scored season (order only) | **-1.021, z -10.0, 52 of 56: much better** |
| what the scored season wants each side multiplied by | offense 0.760 (incumbent 0.795), defense 0.796 (0.898) |
| consensus, 1,000+ possessions | rank agreement 0.849 / 0.808 / 0.832 (incumbent 0.785 / 0.803 / 0.794), top five 5 of 5; spread 1.155 / 1.279 (1.029 / 1.007) |
| trade loss | offense -0.0043, z -5.3, 24 of 30, better in every quality tier; defense a tie |
| year-over-year by quality tier | worse in every tier, team-game level (z +2.4 to +4.6) |

- **2026 top 20** (`outputs/compare_incumbent_type_half_type_plus_own_2026.html`, emailed): the top five stay, with
  Giannis 4th to 2nd.  In: Clingan 20th to 11th, Butler 36th to 14th, Hartenstein 26th to 16th, Allen 35th to 17th,
  Ighodaro 37th to 18th, Anunoby, Harper.  Out: Harden 13th to 21st, Cade 14th to 23rd, Murray, Champagnie, Durant
  15th to 36th, LeBron 19th to 47th.  Type x1 + own swaps is far more extreme (LeBron 158th, Durant 88th).
- **Movement**: median 0.54 per 100 in 2026, 90th percentile 1.43, largest 2.97; mostly players with few possessions
  going down and the stars of their teams going up, since each team's total is fixed.

**Verdict: not adoptable under the rule** (team-game z +3.5).  Every test that reads ORDER says it is better -- the
swap test, the year-over-year order-only row (z -10.0), the consensus and the offensive trade loss -- and the test that
also reads SPREAD says worse: the adjusted list is about 12% wider on offense and 27% wider on defense, and the ratings
were already too wide.  Proposed next, not started: the same adjustment with each season's spread on each side held to
the incumbent's, which keeps the new order and nothing else.  Logs: `outputs/swapadj.log` (the grid),
`outputs/swapadj_x05.log`, `outputs/yoy_swapadj.log`, `outputs/consensus_swapadj.log`,
`outputs/yoy_by_player_swapadj.log`, `outputs/tradeloss_swapadj.log`, `outputs/tradeloss_quality_swapadj.log`.

### The swap adjustment with the spread held: it passes (2026-10-01)

The owner said go.  The same type model at half strength, then one factor per season and side gives back the width
it added (`swapadjust.hold_spread`, `91_swap_adjust.py --hold_spread=within|whole`, two more tests):

- **teams_fixed** (`within`): each player's distance from his team's possession-weighted mean is shrunk until the
  spread INSIDE teams equals the incumbent's.  Team means -- team totals -- do not move, and the order inside every
  team is the adjustment's.  Factors: offense 0.88 (0.83-0.92 over seasons), defense 0.73 (0.69-0.77).
- **list_scaled** (`whole`): each side's whole list scaled back to the incumbent's spread.  The order on each side is
  the adjustment's; team totals shrink with everything else.  Factors 0.90 and 0.77.

| test | teams_fixed | list_scaled |
|---|---|---|
| year-over-year, team-game level | **-0.432, z -5.2, 39 of 56** (8.666) | **-0.814, z -7.7, 48 of 56** (8.652) |
| year-over-year, stint level | -1.078, z -9.1, 49 of 56 | -1.513, z -11.1, 53 of 56 |
| year-over-year, each side rescaled | -0.823, z -9.1 | -1.021, z -10.0 |
| what the scored season wants each side multiplied by (incumbent 0.795 / 0.898) | 0.814 / 0.960 | 0.846 / 1.040 |
| year-over-year by quality tier | better in every tier, z -3.5 to -5.1 | better in every tier, z -3.8 to -8.7 |
| swap test, net order / gaps | +0.123, z +4.3 / -1.71, z -6.9 | +0.125, z +4.6 / -2.25, z -8.6 |
| consensus agreement, off / def / total (incumbent 0.785 / 0.803 / 0.794) | 0.848 / 0.813 / 0.836; spread 1.035 / 1.018; top five 4 | 0.850 / 0.808 / 0.834; spread 1.033 / 1.019; top five 4 |
| trade loss, offense / defense | -0.0046, z -5.7 / +0.0015, z +2.0 (151-300: z +3.6) | -0.0043, z -5.3 / a tie |
| 2026 movement, median | 0.42 per 100 | 0.44 per 100 |

The trade loss for list_scaled is exactly the unheld version's, as it must be: the trade set frees one multiplier
per side and season, and list_scaled differs from it by exactly that.

**2026 top 20, teams_fixed** (`outputs/compare_incumbent_teams_fixed_list_scaled_2026.html`, emailed): the top five
stay, Giannis 4th to 2nd.  In: Clingan 20th to 13th (+0.6), Butler 36th to 14th (+1.0), Hartenstein 26th to 16th
(+0.4), Anunoby 24th to 17th, Harper 30th to 18th, Allen 35th to 20th.  Out: Cade 14th to 21st (-0.7), Murray 17th
to 23rd (-0.3), Champagnie 18th to 24th (-0.4), Durant 15th to 25th (-0.6), LeBron 19th to 49th (-1.2).  list_scaled
is close to it.

**Verdict: both meet the adoption rule** (z -2 or below, the consensus up, not down).  list_scaled is the best legal
score in the record (8.652; experiment 29 was 8.669) and ties the defensive trade loss, but it shrinks team
strength with the rest of the list.  teams_fixed is the owner's design -- each team's total held exactly -- passes at
z -5.2, and carries a small defensive trade loss flag.  **ADOPTED by the owner, 2026-10-01: "Team version."**  The
published table is `91_swap_adjust.py --base=<the pre-swap product table> --kappas=0.5 --taus= --hold_spread=within
--exclude_near=0 --score=0`: every other season trains the type model (ruling 2), the player folds keep a player's
own swaps out.  Spread factors 0.886 / 0.744; 2026 median move 0.43 per 100; team totals move a median 0.007 per
100 of the team's possessions (largest 0.12, from players on two teams).  Consensus on the published table 0.848 /
0.813 / 0.833 (from 0.787 / 0.803 / 0.795).  The old table is kept as `season_ratings_product_pre_swap.parquet`; the
site was rebuilt (`52_site.py`, `76_bias_groups.py`) and its footer names the step.  To ship either, the adjustment must also
be run on the product table (`season_ratings_product.parquet`), with the type model trained on every season, which
ruling 2 allows.  Logs: `outputs/swapadj_within.log`, `outputs/swapadj_whole.log`, `outputs/yoy_swapheld.log`,
`outputs/consensus_swapheld.log`, `outputs/yoy_by_player_swapheld.log`, `outputs/tradeloss_swapheld.log`,
`outputs/tradeloss_quality_swapheld.log`.

### All four on- and off-court ratings in both priors (2026-10-01)

**The question** (the owner): "anything we can do to tame the Ajay Mitchells of the world? High on court rtg in his
prior, not a ton else."  Ajay Mitchell, 2026: OKC +16.8 per 100 with him on the court and +11.7 with him off it
(on/off +5.1); the largest gap between on-court net and box score of any regular with 2,000+ possessions, ahead of
Caruso, Hugo González, Dort, Champagnie, Cason Wallace and McBride.  The 2026-09-13 Boruta list predates the
off-court columns (65_offcourt_panel.py, 2026-09-14), so they were never its candidates; the 2026-09-14 run that
added them beside on-court (with their possessions and the net) tied and was dropped.  **The owner: "let's put
them all in."**  `--features=boruta_onoff`: `offc_o` and `offc_d` added to both priors beside `onc_o` / `onc_d`,
nothing else, on the current pipeline; then the swap adjustment; scored against the incumbent.

| test | result |
|---|---|
| year-over-year, team-game level | +0.026, z +0.4, 24 of 56: a tie (8.667 against 8.666) |
| year-over-year, stint level | +0.196, z +2.4: worse |
| each side rescaled (order only) | +0.168, z +2.6: worse |
| by quality tier | a tie in every tier (z -0.2 to +0.7) |
| swap test, order: net / offense / defense | +0.039, z +2.0 / -0.017, z -1.4 / -0.032, z -1.9 |
| consensus | 0.840 / 0.800 / 0.814 (incumbent 0.848 / 0.813 / 0.836); top five 5 of 5 |
| trade loss | offense +0.0012, z +1.9; defense a tie |

**2026:** Ajay Mitchell 9th to 7th (-0.19).  Role players on great teams go UP: Caruso 52nd to 30th (+0.95), Hugo
González 30th to 17th (+0.81); LeBron 49th to 108th (-1.17), McBride 54th to 101st (-0.81).  The same mechanism as
2026-09-14: a raw off-court rating is the teammates' quality, and the booster reads "the team stays good without
him" as a plus.  **Not adopted**: a tie on the test that decides, the order and the consensus worse, and the case it
was for moves the wrong way.  (On/off alone, tried next, below.)  Logs: `outputs/onoff_chain.log`,
`outputs/yoy_onoff.log`, `outputs/swaptest_onoff.log`, `outputs/consensus_onoff.log`, `outputs/tradeloss_onoff.log`,
`outputs/yoy_by_player_onoff.log`; page `outputs/compare_incumbent_all_four_2026.html`.

**A wrong explanation, corrected by the owner and then by the data.**  I said the prior reads "team good without
him" as a plus because its label (the player's RAPM over his other seasons) over-credits role players on good
teams.  The owner: "That makes no sense."  The 2026 training rows say so too: with the on-court net held fixed, the
off-court net pushes the label DOWN, -0.20 per standard deviation on offence and -0.23 on defence (-0.06 and -0.10
with the box and role inputs beside it) -- on/off logic.  The jumps in the all-four build came from the prior itself
(2026: Caruso's prior +1.18, Hugo González's +0.80, Ajay Mitchell's -0.34), not from the games or the swap step;
the unverified guess is extrapolation by the booster's sloped leaves (`linear_leaves_selected_` is True on both
sides) at the edge of the off-court range.

### On/off alone in both priors (2026-10-01)

The owner: "try it."  `--features=boruta_net`: `net_o` / `net_d` (on-court minus off-court) in place of `onc_o` /
`onc_d` in both priors, possession counts kept; then the swap adjustment; scored against the incumbent.

| test | result |
|---|---|
| year-over-year, team-game level | **+1.074, z +6.9, 12 of 56: worse** (8.704 against 8.666) |
| year-over-year, stint level | +0.739, z +4.4: worse |
| each side rescaled (order only) | -0.092, z -0.7: a tie -- the loss is spread (`scale_off` 0.761 against 0.814) |
| swap test, net order / gaps | -0.017, z -0.8 / +1.78, z +6.7 |
| consensus | 0.856 / 0.798 / 0.827 (incumbent 0.848 / 0.813 / 0.836); how much of a rating the team alone predicts falls to 0.085 / 0.109 (from 0.128 / 0.161; the consensus itself 0.081 / 0.160) |
| trade loss | offence +0.0042, z +5.6: worse |

**2026:** Ajay Mitchell 9th to 6th (+1.10) -- his on/off is itself +5.1 per 100; Butler 14th to 7th (+1.90), SGA 5th
to 4th; McBride 54th to 141st (-1.71), LeBron 49th to 75th; median move 0.66.  **Not adopted.**  Neither on/off-court
version tames him: every reading -- raw on-court, on/off, the swaps -- says Oklahoma City played better with him than
its own level.  Logs `outputs/net_chain.log`, `outputs/yoy_net.log`, `outputs/swaptest_net.log`,
`outputs/consensus_net.log`, `outputs/tradeloss_net.log`; page `outputs/compare_incumbent_on_off_2026.html`.

### Experiment 31: the prior as a stack, and why it lost (2026-10-02, measured 2026-10-03)

**The owner** (2026-10-02): "let's focus on improving the prior, this time the model algorithm", and first a
stacking regressor -- the plus-minus columns into an elastic net, everything else through Boruta into chimeraboost.
`stackprior.StackedSPM`: `ElasticNetCV` on on- and off-court plus-minus and their possessions (`PLUS_MINUS`, eight
columns), chimeraboost `quality=3` on the other 50 `PRIOR_FEATURES` plus the chunk features (`scripts/92_stack_boruta.py`
kept all 50), and a non-negative linear blend fitted on the out-of-player-fold predictions.  `62 --features=stack
--stack=1 --stack_quality=3`.  Against the incumbent it changed four things at once: the stack, the booster's list
(17-21 columns to 50), the offence bag of five (dropped whenever `quality` is set) and the off-court columns.

| test | result |
|---|---|
| year-over-year, team-game level | 8.677 against 8.666, z +1.8, 23 of 56: worse |
| year-over-year, stint level / each side rescaled (order only) | z +8.4 / z +12.7: worse |
| by quality tier | only the top 30 better (z -2.7) |
| swap test, order | offence z -6.8, defence z -4.8: worse |
| consensus | 0.876 total / 0.859 defence (incumbent 0.836 / 0.813) |
| trade loss | offence z +5.0, defence z +7.8: worse in every tier but the top 30 |

**2026:** Zach Edey (590 possessions) 8th, Keshad Johnson (621) 12th, Caruso 5th; Queta, Clingan and Harden out of
the top 20.  Fitted on 2026: elastic net offence `onc_o` +0.247, `offc_o` -0.072 per point; defence (points
allowed) `onc_d` +0.692, `offc_d` -0.213; blend offence -0.064 + 0.571 linear + 0.655 booster, defence +0.019 +
0.834 + 0.378; out-of-fold error offence linear 0.637 / booster 0.636 / stack 0.545, defence 1.035 / 1.407 / 0.953.
**Not adopted.**  Log `outputs/stack_q3_chain.log`.

**Why it lost (2026-10-03, `scripts/93_prior_oof.py`).**  Those out-of-fold errors were measured on career labels,
where a row's plus-minus and its label come from the same games, and never against the shipped booster.  So
2026's training rows were dumped with outside labels (`62 --chunk_label=outside --features=stack
--exclude_neighbours=1 --boards=2026 --dump_rows=oofcheck`, 33,617 rows) and the four priors refitted on the same
five player folds, every row predicted by the fit that never saw its player, on two training sets cut from that one
dump: **as built** (every row on its career label, as the build trains; it reproduces the logged numbers to 0.006)
and **clean** (the chunk rows alone on their outside labels -- the career row keeps the career label even under
`--chunk_label=outside`, so it goes).  The score that matters is on the one-season rows, the shape of the rated
season's own row, against their OUTSIDE label, which shares no game with the row's inputs.  Weighted rmse per 100,
offence / defence:

| prior | trained as built | trained clean |
|---|---|---|
| elastic net on plus-minus | 0.938 / 1.872 | 0.779 / 1.589 |
| booster, 50 columns, no plus-minus | 0.668 / 1.556 | 0.679 / 1.563 |
| the stack | **0.754 / 1.696** | 0.662 / **1.443** |
| the shipped booster | 0.667 / 1.509 | 0.661 / 1.512 |
| everyone at the average | 0.923 / 1.793 | 0.923 / 1.793 |

z against the shipped booster as built, player by player: the stack as built +11.7 / +12.3; the stack trained clean
-0.4 / **-5.1**; the 50-column booster as built +0.1 / +2.6.

- **The logged 0.545 never beat the shipped prior.**  On the same rows and career labels the shipped booster scored
  0.506 on offence (the stack 0.551, z +3.5) and tied on defence (0.943 against 0.958, z +1.0).
- **Trained as built, the elastic net's one-season predictions are about twice too wide**: the slope of the outside
  label on its prediction is 0.49 / 0.46 (1 is right; the shipped booster 0.85 / 0.74, the stack 0.67 / 0.55), and
  the blend gives it 0.57 / 0.83 of the weight.  At that width it does worse than predicting everyone at the average
  (0.938 against 0.923, 1.872 against 1.793).  It sees only the eight plus-minus columns, so one slope serves a career
  row and a one-season row alike; the booster sees the chunk features.  Trained clean, its slope is 0.86 / 0.80.
- **Rescaled, which is the reading that counts** -- the build fits the prior's scale on the season's games, so the
  order is what the prior must get right.  Each prior mapped onto the outside label by its own best line (weighted
  least squares, intercept and slope), one-season rows: the shipped booster 0.657 / 1.464; the stack as built 0.688 /
  1.495, **z +3.2 / +3.2, still worse**; its elastic net 0.787 / 1.578 (z +5.5 / +10.0: signal, correlation 0.52 /
  0.47, but less than the booster's 0.70 / 0.58); the stack trained clean 0.660 / **1.438** (z +0.4 / **-2.3**).
  So most of the raw gap was width, and what is left is order -- the build's order-only reading lost at z +12.7.
  (I first told the owner the elastic net "did worse than predicting the average"; that holds only at its own width.)
- **Trained on outside labels the same elastic net helps**, and the stack ties the shipped booster on offence and
  beats it on defence: by 0.066 raw (z -5.1), by 0.026 rescaled (z -2.3).
- **The 50-column booster without plus-minus ties the shipped booster on offence** and is slightly worse on defence.
- **A screen, not the test.**  By this score training on outside labels costs the shipped booster nothing (0.667 /
  0.661, 1.509 / 1.512), yet experiment 25 -- outside labels with the career row kept -- lost the year-over-year
  test at z +5.0.  The shipped booster's own slopes (0.85 / 0.74) are not a width reading either: an outside label
  rests on fewer possessions than the career label it was trained on and is shrunk harder, which lowers a slope by
  itself.

Outputs: `outputs/oofcheck_chain.log`, `outputs/csv/prior_oof_oofcheck_2026.csv`, every out-of-fold prediction in
`outputs/prior_oof_oofcheck_2026_preds.parquet`.

### Experiment 32: the shipped booster's settings tuned on the screen (2026-10-03)

**The owner** ("a", 2026-10-03): item 2 of their list, chimeraboost hyperparameter tuning, then item 4, bagging --
every setting screened first, and only what beats the shipped booster built.  `scripts/94_tune_booster.py`: the
booster on the SHIPPED lists, trained as the build trains it (career labels), five player folds, scored on 2026's
one-season rows against their outside labels (experiment 31's dump), the score the error after each prediction is
rescaled by its own best line (the build fits the prior's scale, so the order is what counts).  Optuna TPE, 60
trials a side, the shipped settings enqueued first.  Searched: depth 3-8, learning rate 0.02-0.2, `l2_leaf_reg` and
`min_child_weight` 0.3-100 (chimeraboost normalises the weights to mean 1, so these are in rows), subsample 0.5-1,
colsample 0.3-1, bins 32/64/128, linear leaves and their penalty 0.1-100, cross features, and the early-stopping
split: random ROWS (the shipped fit passes no `groups`, so a player's career row and chunks -- one label -- sit on
both sides of the 20% chimeraboost holds out to pick its tree count) or whole PLAYERS (`groups=player_id`).

Then each candidate on two player splits -- the search's, and a second one balanced the same way but dealt
differently -- against the shipped booster as shipped (offence a bag of five, defence single, rows split).
Rescaled error per 100, z per player against the shipped booster:

| candidate | offence, search / second split | defence, search / second split |
|---|---|---|
| shipped | 0.6568 / 0.6537 | 1.4641 / 1.4510 |
| shipped, players split (one change) | 0.6577 / 0.6568, z +0.2 / +0.7 | 1.4335 / 1.4289, **z -4.5 / -4.0** |
| tuned, single | 0.6424 / 0.6432, **z -2.8 / -2.9** | 1.4213 / 1.4172, **z -7.3 / -6.0** |
| tuned, bag of 5 | 0.6434 / 0.6387, z -3.1 / -3.8 | 1.4214 / 1.4172, z -6.8 / -6.3 |
| tuned, bag of 8 | 0.6425 / 0.6383, z -3.2 / -3.9 | 1.4207 / 1.4159, z -6.9 / -6.4 |

Correlation with the outside label: offence 0.702 shipped to 0.718 tuned, defence 0.577 to 0.609.  The tuned
settings: offence (trial 44) depth 7, learning rate 0.021, `l2_leaf_reg` 6.4, `min_child_weight` 99, subsample 0.59,
colsample 0.82, 128 bins, linear leaves (penalty 1.7), cross features, rows split; defence (trial 47) depth 6,
learning rate 0.038, `l2_leaf_reg` 1.75, `min_child_weight` 3.5, subsample 0.83, colsample 0.35, 128 bins, linear
leaves (penalty 6.1), no cross features, PLAYERS split.  What moved the score (fANOVA): offence the learning rate
0.31, the bins 0.28, the split 0.20; defence the split 0.70, the learning rate 0.15.  The ten best offensive trials
sit within 0.003 of each other with `min_child_weight` anywhere from 53 to 100, so the edge of the range is a plateau,
not a cut-off optimum.

- **Defence's early-stopping split is a real fix**: the shipped settings with only the split changed already gain
  z -4.5 / -4.0.  On offence the same change does nothing (z +0.2 / +0.7).
- **Bagging (item 4) adds little on top**: offence 0.001-0.005, defence under 0.002.  The build keeps the shipped
  bag structure, offence five and defence single (`outputs/booster_params_tune1b.json`).
- **Caveat**: the ten settings were chosen on 2026's training rows, which hold every season but 2025-2027 -- the
  seasons the year-over-year test scores on included.  Population-level settings chosen on the same data have
  precedent (the penalty 13,037 was chosen on the test itself), and the test still decides.

**The owner's ruling (2026-10-03), and the build stopped for it:** "we are not going to use different holdout
methods for statistical reasons that are 100% identical."  tune1's offence came out on the row split and its
defence on the player split; the reason for holding out whole players is the same on both sides, so both hold out
whole players, whatever a screen says.  The tune1 build (`tuned_chain.sh`, offence rows / defence players) was
stopped in its last ten seasons and is not a candidate.  **tune2** re-tunes both sides with the split fixed to
players (`94 --split=players`, the patience `early_stopping_rounds` 50-400 added to the search and the learning
rate down to 0.01, since a whole-player holdout is noisier and may stop the fit early), gates on the screen
(`scratch/2026-10-03_tune/gate_tune2.py`: both splits, z -2 or below, offence bag of five / defence single), and
builds on a pass: chain `scratch/2026-10-03_tune/tuned2_chain.sh`, log `outputs/tuned2_chain.log`, NAME=tuned2.

**tune2 result (2026-10-03 11:42 AM): the gate failed on offence, nothing was built.**  Rescaled error, search /
second split, z against the shipped booster as shipped:

| candidate | offence | defence |
|---|---|---|
| shipped | 0.6568 / 0.6537 | 1.4641 / 1.4510 |
| shipped settings, players split | 0.6577 / 0.6568, z +0.2 / +0.7 | 1.4335 / 1.4289, z -4.5 / -4.0 |
| tuned, single | 0.6461 / 0.6490, z -2.1 / -1.1 | 1.4229 / 1.4174, **z -7.1 / -6.5** |
| tuned, bag of 5 | 0.6520 / 0.6493, **z -1.0 / -1.0** | 1.4199 / 1.4178, z -7.0 / -6.2 |
| tuned, bag of 8 | 0.6514 / 0.6504, z -1.2 / -0.8 | 1.4198 / 1.4165, z -7.0 / -6.3 |

With whole players held out, the best offence (trial 51: depth 8, learning rate 0.034, patience 100, `l2_leaf_reg`
3.2, `min_child_weight` 73, subsample 0.57, colsample 0.85, 128 bins, linear leaves with penalty 37, cross features)
gains only about a third of tune1's row-split offence (0.6493 against tune1's 0.6387 on the second split, both bags
of five); defence is as good as tune1 (trial 31: depth 7, learning rate 0.010, patience 100, `l2_leaf_reg` 9.9,
`min_child_weight` 1.7, subsample 0.88, colsample 0.36, 64 bins, linear leaves with penalty 5.9, no cross features).
Every one of the ten best trials on both sides took patience 100.  Awaiting the owner.

### Where the ratings miss most, against expectation (2026-10-03)

**The owner:** "find the players for whom we struggle with the most ... not bias like the bias chart, this is like
variance ... Something like Ajay Mitchell (+/- God) ... or height might over or under shoot lots on avg", with the
miss adjusted for magnitude.  `scripts/95_miss_by_group.py`, on the incumbent: a player's miss is the trade set's
correction (alpha, `outputs/tradeset_swapadj_within_alpha.parquet`, the trade loss's own rows: one-team players
whose team played neighbouring games without them, 12,103 player-seasons a side); each squared miss is divided by
its expectation from the rating level, his on-floor possessions over the three seasons, his team's neighbouring
games without him and the season (cross-fitted gradient-boosted Poisson model); groups are fifths of every
GLOSSARY.md statistic within each season; z with each player's seasons as one cluster.

- **Magnitude, the owner's guess, holds on offence**: the typical correction grows from 0.30 per 100 for the
  worst-rated sixth to 0.44 for the best (defence flatter, 0.26 to 0.33).  It grows faster with the evidence
  behind the correction (0.21 to 0.52), because the correction is a ridge estimate and shrinks where it sees little.
- **Height and the plus-minus extremes do not stand out**: both ends of height, on-court points scored and allowed,
  and on/off on both sides all read within -4% to +7% of expectation, no |z| above 2.1.
- **What does stand out, modestly**: offence, players whose team scores poorly with them OFF the court (+18%, z
  4.9, leaning rated too high; 5 of the 18 points are the lean) and the top draft picks (+11%, z 3.0); defence,
  players whose team rarely played without them in the rated season (+15%, z 4.0, no lean) -- the classic case of
  a player the season's games cannot separate from his teammates -- and players with little garbage time (+13%,
  z 3.2).  +18% in squared miss is about 9% in the typical size of a miss.
- **A first pass misled** (see `adjusted_misses`): with one combined evidence number, starters and heavy-minute
  players read 15-40% above expectation; with the two possession counts separate they vanish.
- **Ajay Mitchell** is among the largest DEFENSIVE misses since 2022, in the other direction from the owner's
  worry: the neighbouring games say his defence is under-rated, +0.94 (2025) and +0.73 (2026) per 100, 3.3 times
  the expected miss.  The largest offensive misses since 2022: Ty Jerome 2025 (+1.40 better than rated), Anfernee
  Simons 2023 (+1.90), Deni Avdija 2024 (+1.57), Paul George 2024 (-1.62 worse).

**Then every cut** (the owner: "did you test a whole bunch of cuts and stats?? We should do the latter"): 371 a
side -- fifths of all 64 statistics, eight player types (a Bayesian mixture on the 13 box rates standardised within
season, fitted on 500+ possession seasons), context (age, years in the league, era, team quality, team changes
before and after), how the rating was built (prior, how far the games moved him, the swap move, the other side's
rating) -- plus a three-level tree fitted on half the players and measured on the other half.

- **Few cuts pass, and none is large**: 4 of 371 on offence and 9 of 371 on defence pass |z| 3, against 1.0
  expected by chance on each side.
- **Nothing in player types** (16 cuts, largest |z| 2.5), **age, experience, era or team changes** (largest |z|
  1.6), or **how the rating was built** (50 cuts, largest |z| 2.5).
- **What passes is team context**: bad teams (bottom fifth of team quality, +12% offence, +14% defence); on
  offence, players whose team scores poorly WITHOUT them (+18%, z 4.9, rated too high) -- and the tree's held-out
  subset, team off-court offence of -5.8 per 100 or worse, reads +36% (z 4.1, 521 player-seasons, rated too
  high); on defence, players whose team rarely played without them (+15%), and the tree's held-out subset -- few
  possessions without him, little garbage time, his team scoring poorly with him on -- reads +33% (z 3.8, 538
  player-seasons, rated too LOW on defence).
- **The misses are close to unpredictable**: a gradient-boosted model on every variable above predicts 0.0% of
  the adjusted miss on offence and 0.8% on defence for players it never saw.
- Page with every cut: `outputs/miss_by_group_incumbent.html`; tree leaves `outputs/csv/miss_subsets_incumbent.csv`.

**The universe of overrated players, by magnitude** (the owner: "there will be outliers ... more prominently seen in
players with a higher magnitude ... see the whole universe of players that we are overrating").  Measured against the
FINAL rating (prior + his own games + swap step, which add up exactly).  By quality tier (rank by total rating that
season), share of player-seasons the neighbouring games say are overrated by 0.5+ per 100 against underrated by 0.5+:
offence top 30 **17% against 8%** (average correction -0.13), 31-90 11% / 8%, 91-150 7% / 11%, 151-300 7% / 11%,
301+ 5% / 6%; defence top 30 **10% against 5%** (-0.09), falling to 3% / 4% at 301+.  So the best-ranked are
overrated twice as often as underrated, on top of the season-wide rescale (offence x0.755 median).  Among the
top-30 overrated the rating is almost all prior: offence 2.62 = prior 2.58 + own games 0.10 + swap -0.06 (games say
-0.74); defence 1.05 = 1.07 + 0.04 - 0.06 (-0.69).  1,542 player-seasons overrated by 0.5+; 368 player-sides
overrated in two or more seasons (Lowry and Duncan 10 of 18 on offence, Durant 9 of 16 on defence) -- neighbouring
seasons share games, so one stretch can count twice.  All of it is in the owner's Google Sheet "Where OpenRAPM
misses most (Oct 3, 2026)" (`scratch/2026-10-03_tune/overrated_payload.py`).

**It is mostly regression to the mean** (the owner: "I think this might just be regression to the mean ... Lets test
it"; `scripts/96_rtm_test.py`).  Rank the top 30 by a season that shares no games with the three-season window (H-2
or H+2) instead of by the tested season H, on the same rows: offence mean miss -0.157 -> -0.056 (z -2.0; over/under
20%/7% -> 15%/10%) and -0.128 -> -0.041 (z -1.4; 18%/7% -> 14%/11%); defence -0.092 -> -0.001 (z 0.0) and -0.093 ->
-0.027 (z -1.3).  So most of the top-30 overrating comes from picking the top by the season being tested -- a season
ranked near the top is partly a peak year or a lucky rating.  What remains on offence (about 0.05 per 100) is under
the owner's 0.1 bar.  Not separable with this window: a real peak in H from a lucky H rating; that needs a
within-season test (rate on part of the season, score on the rest), which needs half-season box-score inputs.

**The team-context patterns are not luck** (the owner: "Team context next").  Each group split at the median of the
possessions behind the number that defines it; luck would live in the noisy half.
- Offence, team scores badly WITH HIM OFF (bottom fifth): few off-court possessions -0.008 (z -0.6; over/under 8%/9%);
  many off-court possessions **-0.075 (z -5.7; 14%/7%)**.  Everyone else +0.026.
- Defence, heavy minutes and little garbage time with his team scoring badly WITH HIM ON (the held-out tree's subset):
  noisy half +0.117 (z 7.6; 3%/13%), precise half **+0.160 (z 8.6; 5%/19%)** -- defence rated too LOW.
Both are largest where the number is most precise, so both are structural.  Defence matches the mechanism named on
2026-09-18 (the defensive prior reads team offence: `onc_o` is on `BORUTA_D`); offence fits the owner's
bad-replacements hypothesis but is not yet separated from "bad team overall" (bottom-fifth team quality also leans
too high, +12%).

Outputs: `outputs/csv/miss_by_group_incumbent.csv`, `outputs/csv/miss_players_incumbent.csv`.

### Experiment 33: a calibrator trained on each season's own held-out games (2026-10-03)

The owner: "build some sort of calibrator, that takes lots of our inputs and tries to solve for the calibration misses
that we are seeing", then "how to not shrink peak seasons ... bootstrapping the heck out of the single seasons to get
lots of holdout data?", then "let's do the 10 most recent for now" and "keep really special good tabs on it".  The
trade set measures misses on the neighbouring seasons, so a calibrator trained on it would learn that peaks come
back down; this one is trained on the season's OWN games, held out.

**The machinery** (`scripts/97_within_season.py`, `WITHIN_SEASON_LEDGER.md`).  Each season 2017-2026, three random
deals of its games into four folds: the rating is built from three folds by the rankings' own code (steps 1-3: the
saved prior models of `62 --save_models`, the ridge, the centring), with every input the prior reads rebuilt from
those games -- box rates, on-court numbers, playing time, starts, tenure, the defensive target's three-point
repricing -- and the fourth fold is held out.  One leak was found and closed on the way: `WindowData.subset` keeps the
whole season's box-score and possession tables, so a subset fit still read every game's box score.  Accepted, as in
FINDINGS 31: the free-throw adjustment's season percentages (stored in the stints), about 0.03 per 100.  Checks: the
rebuilt 62 table equals `season_ratings_unshrinkdef.parquet` exactly; the whole season through the part-season code
equals the rankings exactly (rating difference 0.0, inputs within 2e-11); 120 folds, 2,640 leak checks, 0 failed.

**Before any calibrator**, the held-out games say the offence is too wide: what they want each side multiplied by
averages 0.89 / 1.00 (part-season ratings, stint scoring).

**The calibrator** (`scripts/98_calibrator.py`): per season, trained on the other seasons outside the season and its
neighbours; two arms.  (a) One multiplier per side, weighted least squares on the held-out team-games: offence 0.735
to 0.756, defence 0.954 to 0.999, every season.  (b) The multiplier plus boosted trees on 78 inputs per player and side,
fitted to the held-out team-games through the possession shares (leaf values by exact team-game least squares),
corrections centred per fold and season.  The first run of (b) let a common level drift to +1.4 to +2.1 on both
sides -- adding one number to every player's offence AND defence moves no prediction; it was read off the players who
appear only in the held-out games -- and put every regular in the 2026 top 20 up 3 to 4.7; centring fixed it.

| within-season test (120 folds, the year-over-year scoring on each fold's held-out games) | error per team-game |
|---|---|
| the rating | 8.636 |
| one multiplier per side | 8.587 (better in 10 of 10 seasons, z -15.4) |
| the multiplier + trees | 8.527 (better than the multiplier in 10 of 10, z -9.5) |

| year-over-year test (the incumbent's swap adjustment applied; 2017-2026 replaced) | error | vs incumbent | order (each side rescaled) |
|---|---|---|---|
| incumbent | 8.666 | -- | -- |
| one multiplier per side | **8.648** | z -4.1, 40 of 56; on the 18 pairs whose rating changed, 17 of 18 (z -6.3) | unchanged (z +0.9) |
| the multiplier + trees | 8.645 | z -3.5, 28 of 56 | **worse (z +3.3)** |

Paired, trees against the multiplier: team-game a tie (z -1.2, 21 of 56), order worse (z +3.3, 12 of 56).  **The trees'
per-player corrections fit their own season's held-out games and do not carry to the neighbouring seasons.**  What
they lean on says why: team quality with him on and off the court, playoff share, garbage-time share -- the
team-season's context, real inside the season, not the player's to take with him.

The multiplier alone: better at team-game level in every quality tier (top 30 z -4.3, 31-90 -4.2, 91-150 -4.1,
151-300 -4.0, 301+ -3.6), for players who stayed (z -4.2) and who moved (z -3.7); consensus 0.832 / 0.847 / 0.814
total / off / def (incumbent 0.836 / 0.848 / 0.813), top five 5 of 5 (from 4); offensive spread 1.66 -> 1.24.  2026:
the top four unchanged; Curry 12th -> 21st, Harden 19th -> 42nd, Doncic 8th -> 12th, Javonte Green 31st -> 18th, Paul
Reed 29th -> 19th.  It meets the adoption rule on the numbers; the 2026 top 20 is the owner's call (experiment 21's
offensive collapse, 1.60 -> 1.14, was rejected on the eye test).  Caveat: the multiplier was fitted on three-quarter-
season ratings, which are noisier than whole-season ones, so 0.75 likely over-shrinks a whole season somewhat.

**33b: which part comes back to the mean** (the owner: "Are you sure that offense just doesn't regress to the mean
more", then "Sure try it").  Split each rating into the box-score prior part (the ridge's scale times the prior) and
the games' own adjustment, and score each on the held-out games (1 = holds up; 10 seasons, standard error from
season to season): offence prior part **0.70 (+-0.02)**, games' part 1.21 (+-0.08); defence 0.91 / 1.30.  By the
player's possessions: offence 2,500+ **0.64**, 1,000-2,500 0.83, under 1,000 1.44 (defence 0.90 / 0.99 / 1.49).  So it
is regression to the mean, but not the small-sample kind (that would hit the bench and the games' part hardest): it
is the stretched box-score prior of heavy-minute players that the season's other games do not back up.  Two shapes,
fitted per season outside the season and its neighbours (`scratch/2026-10-03_within/two_part_arms.py`): (a) the
prior part alone, offence x0.70-0.73, defence x0.92-0.97; (b) both parts, offence prior x0.70-0.72 and games
x1.11-1.25, defence x0.88-0.94 and x1.19-1.35.

| | year-over-year | vs incumbent | order | within-season |
|---|---|---|---|---|
| one multiplier | 8.648 | z -4.1, 40 of 56 | tie | 8.587 |
| (a) prior part only | **8.648** | z -4.1, 40 of 56 | tie (z +0.6) | **8.582** (vs one multiplier z -3.4, 9 of 10) |
| (b) both parts | 8.648 | z -4.0, 40 of 56 | tie | 8.580 (vs (a) z -1.1) |

Across seasons the three tie (every pairing within 0.01, |z| under 1): the gain is the offensive amplitude, whatever
its shape.  Within the season, shrinking only the prior part is the better shape; boosting the games' part adds
nothing further.  (a): consensus 0.840 / 0.857 / 0.816 total / off / def (incumbent 0.836 / 0.848 / 0.813), top five 5
of 5.  2026 (swap adjustment applied): top four unchanged; Curry 12th -> 24th (4.35 -> 2.78), Harden 19th -> 50th
(3.35 -> 1.90), Doncic 8th -> 12th, Dylan Harper 18th -> 25th; Javonte Green 31st -> 17th, Paul Reed 29th -> 18th, Hugo
Gonzalez 30th -> 21st, Jarrett Allen 20th -> 15th.

**Adopted 2026-10-03 (the owner: "Good, adopt for now"), for all thirty seasons** (`scripts/99_prior_shrink.py`).
The within-season folds cover 2017-2026, so 1997-2016 borrow the multipliers fitted on all ten (offence 0.718, defence
0.949; 2016 drops 2017).  The thirty-season table against the previous incumbent: year-over-year **8.600 against
8.666, z -12.2, 53 of 56**, every era better (1998-2006 18 of 18, 2007-2016 19 of 20, 2017-2025 16 of 18 -- the older
seasons gain most, their offensive prior was stretched hardest, 2.0-2.95x against 1.7-2.1x lately); stint level z -4.1;
stint level with each side rescaled z +2.1 (the order inside a side slightly worse); swap test net order z +3.1 (40 of
56), gaps z -6.2, the within-team slope 0.79 -> 0.99; trade loss offence z -3.1, defence z +5.5 (0.0007 on a 0.248
scale, 3 of 30 seasons better); consensus 0.834 / 0.851 / 0.816 (from 0.836 / 0.848 / 0.813), top five 5 of 5.  The
year-over-year stint-level scale on offence moves 0.81 -> 1.11 (the stint/team-game disagreement of 2026-09-05).  The
new incumbent is `outputs/season_ratings_priorshrink.parquet`; the product (`--rule=product`, a season's multiplier
fitted on every fold season but itself) is `outputs/season_ratings_product.parquet` (previous kept as
`season_ratings_product_before_priorshrink.parquet`).  Product 2026: Wembanyama, Kawhi, Giannis, Jokic, Gilgeous-
Alexander; Curry 13th -> 21st, Murray 14th -> 27th, Harden 15th -> 31st, Durant 20th -> 41st; Hugo Gonzalez 21st -> 15th,
Caruso 23rd -> 16th, Paul Reed 25th -> 19th, Javonte Green 29th -> 20th.  Open: the defensive trade loss; multipliers
for 1997-2016 from their own folds; a candidate that changes the prior needs its own folds.

**33c: the within-season test on all thirty seasons, at five sizes** (2026-10-04; the owner: "I like your split half
idea ... best done at different sizes", "Ok go ahead").  Every season rated from 1/4, 1/3, 1/2, 2/3 and 3/4 of its
games and scored on the rest (1,560 ratings, 34,320 leak checks, 0 failed; `scratch/2026-10-03_within/sizes.py`).
Error per team-game on the games each rating never saw, the published recipe against the Oct 1 one:

| rated from | no ratings | Oct 1 | now | now better |
|---|---|---|---|---|
| 1/4 | 8.977 | 8.606 | 8.536 | 30 of 30 seasons, z -19.3 |
| 1/3 | 8.977 | 8.564 | 8.494 | 30 of 30, z -16.6 |
| 1/2 | 8.975 | 8.505 | 8.432 | 30 of 30, z -18.2 |
| 2/3 | 8.972 | 8.466 | 8.396 | 30 of 30, z -16.6 |
| 3/4 | 8.967 | 8.451 | 8.377 | 30 of 30, z -18.5 |

The multiplier the held-out games ask for on the box-score part, all thirty seasons pooled: offence 0.638 / 0.651 /
0.666 / 0.684 / 0.684 at 1/4 ... 3/4, a straight line through them 0.712 at a whole season (adopted: 0.704-0.728);
defence 0.889 / 0.887 / 0.896 / 0.895 / 0.900, the line 0.905 at a whole season (adopted: 0.920-0.968, so defence
is shrunk less than the evidence asks).  Agreement of two ratings from separate parts of a season (about 370 players
a season): total 0.768 / 0.784 / 0.810 at 1/4 / 1/3 / 1/2 now, against 0.788 / 0.803 / 0.829 for the Oct 1 recipe;
offence 0.786 / 0.799 / 0.825, defence 0.665 / 0.697 / 0.743.  The shrink lowers agreement by about 0.02 at every
size and lowers the error on unseen games at every size: the box-score part repeats from one part of a season to
another partly because its career and body inputs are the same in both parts, which does not make it right.  The
full-season reliability these imply falls as the parts grow (0.930 from quarters, 0.916 from thirds, 0.895 from
halves): agreement grows more slowly with games than the Spearman-Brown formula assumes, for the same reason, so the
half-season figure, 0.895, is the least extrapolated.

Outputs: `outputs/within/within/` (folds), `outputs/calibc_within.parquet`, `outputs/calibc_corrections.parquet`,
`outputs/season_ratings_calib{c,res}_full.parquet` (swap-adjusted), `outputs/yoy_calibc.parquet`,
`outputs/yoy_by_player_calibres.parquet`; chains and logs in `scratch/2026-10-03_within/`.

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

## Experiment 34: teammates' shot mix with him on the floor (2026-10-04, the owner's idea; awaiting the owner's call)

**The owner:** "what about 'teammate shot quality while on court' for dudes?"  For every pair of teammates who
shared the floor in a season, the teammate's expected points per shot -- the league's make rate at that distance
that season times the shot's value, so the shot's location and nothing about who took it -- with the player on
the floor minus without him, averaged over his teammates weighted 1 / (1/on + 1/off) in attempts, padded toward the
season mean with k = 404 (chosen so his first-half number best predicts his second half and the reverse, pooled
over all thirty seasons).  `scripts/100_shot_mix_panel.py` writes `mix_lift` into the season panel; feature set
`boruta_mix` adds it to the OFFENSIVE list only.  Half-season agreement 0.22 over everyone, 0.60 for the heavier
half; padding keeps 0.78 of a regular's own season.  2025 top: Jokic, Trae Young, Morant, Wembanyama, Vucevic,
Towns.  Its "makes" twin (teammates' makes over their own season rates, him on) read -0.37 between halves on 2025
and was not built.  Earlier, assists priced by the league value of their zone put Haliburton 106th of 144 in 2025:
location cannot see how open a shot was.

**Result** (`scratch/2026-10-04_shotmix/mix_chain.sh`, through 99 `--rule=test` and 91 like the incumbent):
year-over-year **8.597 vs 8.600** per team-game (-0.070, z -2.56, 32 of 56), predicting the season after z -0.61, the season before z -2.81;
better in every quality tier (top 30 z -5.5).  But stint level is worse (z +4.9) and the order-only row (each side
rescaled) is a tie (+0.005, z +0.14); the offence comes out narrower (consensus spread 0.725 vs 0.764; scale_off
1.17 vs 1.11).  Swap test: net order +0.011, z +1.2 (tie).  Consensus rank agreement 0.838 vs 0.834.  Trade loss:
offence tie, defence +0.0002 (z +4.3, tiny).  2026 top 20 nearly unchanged; offensive movers among 2,000+
possession players: Brunson -0.57, Edwards -0.53, Murray -0.53, Doncic -0.37; Draymond Green +0.46, Paul George
+0.44.  ~~Caveat: the shrink multipliers are the incumbent's own~~ -- wrong, see the correction below.

**Correction (2026-10-05): the result above is confounded, and the corrected result is a tie.**  `99_prior_shrink.py`
read every fold in `outputs/within/within`, which grew from 2017-2026 to 1997-2026 on 2026-10-04 (experiment 33c).  So
the shot-mix table was shrunk with multipliers fitted on 27-28 fold seasons (offense 0.677-0.690, defense
0.888-0.910) while the incumbent's were fitted on 2017-2026 (0.704-0.728 / 0.920-0.968).  99 now pins
`--fold_seasons=2017-2026` by default and has `--check=<table>`; the pinned rerun rebuilds both shipped tables
(`season_ratings_priorshrink_raw`, `season_ratings_product_priorshrink_pre_swap`) with largest difference 0.0
(`tests/test_prior_shrink.py`).  Re-scored on the same 62 build with the pinned shrink (tag `shotmix_pin`,
`scratch/2026-10-05_scorecard/mix_repin.sh`): year-over-year **8.5993 vs 8.5995** (-0.006, z -0.22, 29 of 56;
predicting the season after z +1.96, the season before z -1.46); order only z -0.21; top 30 z -0.70; swap test net
order +0.015, z +1.6; consensus 0.837 vs 0.834, offensive spread 0.753 vs 0.764; trade loss offense z -0.24, defense
z +1.34.  A tie on every test.  The 2026 movement is the same as before (Edwards -0.50, Brunson -0.46, Murray -0.40,
Paul George and Draymond Green +0.49, Jokic +0.33).  So the first run's "win" was the stronger shrink, not shot mix
-- which itself says the 1997-2026 multipliers may beat the adopted ones (an open question, not tested).

## Experiment 35: slope vs. error, a within-season scorecard, and three vanilla baselines (2026-10-05)

**The owner's questions.**  "When should we care about slope vs error?  Lots of these tests must by design be small
sample size."  Score carefully non-leaked OpenRAPM ratings built from part of a season on the games left out, and
compare with (1) a linear, box-score-only SPM and (2) a vanilla tuned RAPM; also a middle rung (RAPM with the linear
SPM as its prior), both tests, a 9/10 fold size, the swap adjustment rebuilt per fold.  Plan:
`~/.claude/plans/one-thing-i-want-joyful-book.md`.

**What was built.**
- `src/eracoef/scorecard.py` (model layer): per scored block, the six sums that give the error at any per-side
  multiplier (raw error, slope, error after a cross-fitted rescale), full-level side and part/tier equations, blend
  weights, jackknife SEs by season, the controls.  `scripts/104_scorecard.py` reproduces every stored fold error
  (1,860 folds, largest difference 6e-14) and the year-over-year file (56 season-directions, 6e-14) before scoring.
- `src/eracoef/boxspm.py` + `scripts/102_box_spm.py`, **B1**: 13 box counts per 100 box-estimated possessions plus
  minutes share, padded, weighted least squares onto the shipped prior's label family (leave-season-out RAPM, {H-1,
  H, H+1} out, defense un-shrunk).  3,750 leak checks, 0 failed.
- `src/eracoef/vanilla.py` + `scripts/103_vanilla_rapm.py`, **B2** (vanilla RAPM) and **B3** (RAPM on B1's prior with a
  multiplier per side): one factorization per penalty pair, the held-out error an exact quadratic in the
  multipliers, penalties and multipliers chosen per rated season on blocks that touch no season in {H-1, H, H+1}.
  Chosen: B2 penalty 4,420 both sides at EVERY fold size (7,620 year over year); B3 7,620-13,130 (defense 0.5-0.75 of
  offense) with B1 multiplied 1.5-1.7 on offense and about 1.1 on defense (22,640 / x1.78 / x1.01 year over year).
  The non-nested B2 picked the same penalties: tuning on the test does not flatter it here.
- `scripts/106_fold_swapadj.py`: OpenRAPM as shipped on every fold (pinned prior shrink, then the swap adjustment at
  0.25/0.5/0.75/1).  The whole-season version reproduces the shipped test table: shrink exactly (0.0), swap
  adjustment within 0.006 pts/100 (median 0.0015; the type model's standardizing now excludes the rated season).
- 9/10 fold size: `97 --folds=10 --repeats=1 --tag=within10`, 300 folds, 6,600 checks, 0 failed.
- `scripts/105_held_out_swaps.py`: the lineup-swap test on each fold's HELD-OUT games.
- `scratch/2026-10-05_scorecard/semisynth.py`: power check on the real held-out designs with a known truth.

**Results** (error = typical team-game miss, pts/100; within season at 3/4 unless stated; full tables in the sheet
"OpenRAPM experiments", tab Scorecard; `outputs/scorecard/`):

| system | within 3/4 | slope O / D | year over year | slope O / D | held-out teammate order (net) |
|---|---|---|---|---|---|
| no ratings | 8.970 | | 8.962 | | |
| B1 linear box SPM | 8.692 | 2.50 / 1.45 | 8.735 | 2.18 / 1.19 | 2.016 |
| B2 vanilla RAPM | 8.426 | 0.93 / 0.92 | 8.677 | 0.99 / 0.96 | 1.677 |
| B3 RAPM on B1 | 8.384 | 1.00 / 0.95 | **8.587** | 0.99 / 0.97 | **2.128** |
| OpenRAPM steps 1-3 | 8.455 | **0.70** / 0.97 | 8.682 | 0.62 / 0.89 | 1.840 |
| OpenRAPM + prior shrink | 8.381 | 0.96 / 1.01 | 8.608 | 0.86 / 0.93 | 1.900 |
| OpenRAPM as shipped | **8.370** | 0.97 / 1.03 | 8.600 | 0.88 / 0.97 | 2.085 |

1. **OpenRAPM as shipped and B3 are close, and B3 wins where attribution matters.**  Year over year B3 beats the
   shipped ratings: 8.587 vs 8.600, z -2.54, 20 of 28 seasons, and in each era (z -1.3 to -1.6).  Within season the
   shipped ratings beat B3 (z +3.8 at 3/4) -- but almost all of it is 2017-2026 (z +4.6, 0 of 10), the seasons whose
   folds the prior shrink was fitted on; 1997-2006 and 2007-2016 read z +1.4 and +1.3.  On held-out lineup swaps B3
   orders teammates better than OpenRAPM before its swap adjustment (+0.23, z +4.8) and slightly higher than after
   it (2.128 vs 2.085, not tested).  Blend weights: OpenRAPM's offensive differences from B3 are mostly noise (0.25 year over year, 0.34-0.41
   within), its defensive ones a coin flip (0.52-0.58).  Consensus: B3 0.831 vs 0.834 total (offense 0.820 vs 0.851,
   defense 0.852 vs 0.816); its ratings depend less on the team (team R-squared 0.10 / 0.07 vs 0.16 / 0.13).
   Swap test across seasons: order a tie (z -0.2), gaps better (z -2.7).  2026: rank correlation with OpenRAPM 0.81;
   top 10 Wembanyama, Jokic, Gilgeous-Alexander, Doncic (OpenRAPM 12th), Leonard, Holmgren, Antetokounmpo, Towns,
   Queta, Tatum (85th); Robert Williams 18th (OpenRAPM 183rd).
2. **OpenRAPM's boosted prior knows more about lineup sums and less about the split.**  Its prior alone beats B1 on
   error (blend weight 0.72-0.95) but orders teammates worse on held-out swaps (1.807 vs 2.016 at 3/4); B1, with no
   plus-minus input at all, orders teammates better than OpenRAPM before the swap adjustment.  The prior's on-court
   plus-minus inputs are the leading suspect (they carry team information into each player's prior) -- not tested.
   The swap adjustment, a linear model of box-score types, moves OpenRAPM toward that split: error z -7, teammate
   order z +6 vs the shrunk ratings.  Strength 0.5 vs 0.75: a tie on error, 0.75 slightly ahead on held-out swaps.
3. **Slopes.**  Before the shrink, OpenRAPM's offense is about 30% too wide at every fold size (0.67-0.70); the part
   that overreaches is the box prior (part slope 0.62-0.69) while the games' part is too timid (1.25-1.57).  The
   shrink brings it to 0.92-0.97 (rising with evidence).  B3 sits at 0.96-1.00 everywhere; B2 drifts from 1.05 (1/4)
   to 0.92 (9/10) at its one penalty.  Year over year the shipped offense still reads 0.88 (mixing "too wide" with
   "players change", experiment 20).
4. **What the tests can see.**  Setting every player to his team's average costs 16% of OpenRAPM's error gain within
   season and 41% year over year: team-game error sees the split partly, more across seasons.  Held-out lineup swaps
   see only the split (the team-average control scores 0.08, about nothing).
5. **Power and sample size.**  Truth-known check on the real designs: a calibrated rating's slope is 1.003; a rating
   10% too wide on offense is flagged by its slope (|z| > 2) in 66% of replications and by paired error in 32% --
   slope is about twice as sensitive to scale.  More deals do not help: SE of OpenRAPM minus B2 0.140 now vs 0.138
   with unlimited deals; seasons are the unit.  OpenRAPM rated from 1/2 of a season predicts about as well as
   vanilla RAPM from 9/10.

**The rule (also in the sheet's Definitions tab).**  Error ranks systems (paired, same games, SE by season).  Slope
sets scale: shrinkage, prior weight, published magnitudes; slope 1 is necessary, never sufficient.  Read slopes
pooled at team-game level as a profile over fold sizes.  The split among teammates needs the held-out lineup swaps.

**Not done / open:** none of this changes what ships.  Candidates the results point at, for the owner: (a) the
prior without its on-court plus-minus inputs; (b) B3's recipe (linear box prior + tuned RAPM) as the base, with the
luck-adjusted targets and the swap adjustment on top; (c) swap strength 0.75.

## Experiment 36: RAPM on OpenRAPM's own box prior, tuned like B3 -- the penalty and the prior's weight by sample size (2026-10-05)

**The owner:** "the best penalty for RAPM (and especially the penalty used on the PI RAPM) will change as the sample size
of the season changes."  The reasoning given back: for vanilla RAPM the ridge penalty is noise variance over talent
variance, both per possession, so it should not move with games (the data term grows with possessions instead), and it
did not -- 4,420 at every fold size; for prior-informed RAPM it should, because the prior is built from the same games.
**B4** puts OpenRAPM's own boosted prior (`prior_raw_*`, the booster's prediction before its free scale, from the same
saved models and rating-game inputs as the fold's OpenRAPM rating) through exactly B3's tuning (`103 --prior=openrapm`):
penalty pair and prior multipliers per fold size and per rated season, from blocks touching no season in {H-1, H, H+1}.

**How the tuning moves with sample size** (median over rated seasons; `outputs/csv/baseline_penalties{,_b4}.csv`):

| fold size | B3 (linear box prior): penalty, D/O, multiplier O / D | B4 (OpenRAPM's prior): penalty, D/O, multiplier O / D |
|---|---|---|
| 1/4 | 7,620, 0.75, 1.50 / 1.06 | 7,620, 1.5, 1.08 / 0.67 |
| 1/2 | 7,620, 0.75, 1.46 / 1.13 | 7,620, 1.5, 0.91 / 0.68 |
| 3/4 | 13,130, 0.5, 1.73 / 1.16 | 7,620, 1.5, 0.79 / 0.72 |
| 9/10 | 13,130, 0.5, 1.70 / 1.18 | 13,130, 1.0, 0.96 / 0.76 |
| year over year | 22,640, 0.5, 1.78 / 1.01 | 22,640, 0.75, 1.05 / 0.73 |

The linear box prior earns MORE weight as games accumulate (its inputs get less noisy); OpenRAPM's prior earns LESS on
offense up to 3/4 of a season (1.08 to 0.79) -- consistent with a prior that already carries the same games'
plus-minus (`onc_*`), so more games means more double counting.  The grid steps about 70% per penalty value and the
penalty and multiplier trade off (the 9/10 row jumps), so read the direction, not the decimals.

**Which prior is better under identical tuning: the linear box prior.**  B4 minus B3: within season z +2.1 (3/4), +2.6
(9/10), -0.5 (1/4, a tie); year over year +0.597 squared, z +3.6, B4 better in 8 of 28.  Blend weights B4 vs B3: offense
0.15-0.40 (OpenRAPM's offensive prior differences are mostly noise), defense 0.58-0.61 within season (slightly better),
0.47 year over year.  Held-out teammate order: B4 1.975 vs B3 2.128 at 3/4.  Battery: consensus 0.801 (offense 0.783)
vs the incumbent's 0.834 and B3's 0.831; swap test order z -3.6 (worse), trade loss offense z +6.1 (worse).  OpenRAPM's
own pipeline (fixed penalty, free prior scale, shrink, swap adjustment) gets more out of the same prior than plain
tuning does (year over year the shipped ratings beat B4, z +2.7 for B4) and still loses to B3 (z -2.5).
Year over year: B3 8.587 < shipped 8.600 < B4 8.609 < vanilla 8.677.

**Reading.**  The prior, not the ridge tuning, is where OpenRAPM gives up ground to the plain recipe, and it is the
offensive prior.  The falling weight with more games points at the on-court plus-minus inputs -- option (a) of
experiment 35 (the prior without them) is the direct test.  Nothing here changes what ships.

## Experiment 37: a box-score prior fit directly on held-out games (2026-10-05, in progress)

**The owner:** "build a model that will crush this testing ... a wide array of experiments to see what's actually going
on ... is there some sort of feature selection/model selection we can do with this as a target?  Be super honest" --
and then: "by crush I just mean like robustly look at."  Plan: `~/.claude/plans/one-thing-i-want-joyful-book.md`.

**Stage 0: a stray column in the baselines' ridge, fixed.**  A cached design (`designcache.make_X`) carries the game
index as its last column; `vanilla.Ridge` left it in the unpenalized block, so B2/B3/B4 each had a free within-season
time trend the shipped ridge does not.  `Ridge(..., n_fixed=)` now keeps only the named fixed effects (asserted in
103's `ridge_for`; `tests/test_baselines.py`).  Re-run: within season the baselines get slightly better (B3 8.384 ->
8.377 at 3/4, B2 8.426 -> 8.414), year over year they barely move (B3 8.5874 -> 8.5868).  **B3 still beats the shipped
ratings year over year: -0.348 squared, z -2.75, 20 of 28 seasons**; within season the shipped ratings lead at z +1.8
(was +3.8).  B4 vs B3: year over year z +3.7, within z +2.3 (unchanged in substance).  Vanilla RAPM's best penalty is
now 2,560 at 1/4 of a season and 4,420 from 1/3 up (not exactly constant).  The trend versions are kept as `*_trend`.

**Stages 1-3 (2026-10-05 evening).**  `src/eracoef/heldoutprior.py` (tests/test_heldoutprior.py: the quadratic in the
weights equals a direct solve to 1e-8 at both levels; one input per side is B3's six numbers; constants move
nothing; drifting weights need no new solve), `scripts/107_heldout_prior_build.py` (65 inputs per side; every
season's first fold reproduces the rerun B3 grid exactly), `scripts/108_heldout_prior_select.py`.  Grids widened twice
until no choice sat on an edge (penalty 2,560-67,290 x defense ratio 0.5-1.5; corrections' penalty 1e-4 to 100).
Development seasons only (20; the lockbox 1999, 2002, ..., 2026 untouched); everything nested by season.

Each input group added to B3 (change in typical miss, pts/100; negative = better; `outputs/heldout/ablation_table.csv`):

| added to B3 | within season (2/3, 3/4, 9/10) | z | year over year | z | eras agree |
|---|---|---|---|---|---|
| B1's 14 inputs refit on held-out games | -0.024 | -7.7 | +0.005 | +0.7 | yes |
| box rates per play-by-play possession | -0.020 | -8.2 | +0.012 | +1.3 | yes |
| shooting efficiency | -0.008 | -4.9 | +0.007 | +1.5 | yes |
| shot location | -0.006 | -4.1 | +0.009 | +1.8 | yes |
| role (possession share, starts) | -0.015 | -4.7 | +0.019 | +2.3 | yes |
| age, body, career | -0.017 | -5.4 | -0.011 | -1.4 | no |
| score context | -0.007 | -2.8 | +0.001 | +0.1 | no |
| on-court plus-minus | -0.007 | -3.8 | +0.007 | +2.1 | yes |
| off-court plus-minus | -0.010 | -5.7 | +0.002 | +0.4 | yes |
| same-season plain RAPM | -0.009 | -4.6 | +0.014 | +2.7 | yes |
| OpenRAPM's boosted prior | -0.006 | -3.2 | -0.002 | -0.6 | yes |
| ALL inputs | -0.085 | -13.6 | -0.004 | -0.4 | no |
| all but every plus-minus input | -0.055 | -10.8 | +0.010 | +1.0 | no |
| all inputs, shape fit year over year (diagnostic) | -0.014 | -3.9 | **-0.035** | **-3.4** | yes |

**Reading.**  Every input group improves within-season prediction; none improves the next season's.  The groups that
carry the same season's context -- role, on-court plus-minus, same-season RAPM -- make the next season WORSE (z +2.1
to +2.7).  The year-over-year scale the shapes get (offense x0.65, defense x0.75 for all inputs) says a prior fit on
same-season held-out games has to be shrunk about a third to predict next season.  The only thing that helps year
over year is fitting the shape to year over year itself (z -3.4, both eras): a forecast, which learns what carries
across seasons (and which the owner has not wanted as the rating, experiment 33).  Probability of backtest
overfitting over the 29 configurations: 0.00 -- the year-over-year-fit diagnostic wins consistently, and nothing else
does.  Not run yet: stage 4 (by fold size; era drift), stage 5 (targets), the lockbox.

## Experiment 38: what makes a rating better in-season vs for other seasons (2026-10-05)

**The owner:** "I would like to find out what stats / model decisions make one model better at in-season vs better at
'outside seasons'."  `scripts/110_in_vs_out.py`; development seasons only (lockbox untouched); paired t over seasons
(in-season n = 20, out of season n = 19).  Full per-stat table: sheet "OpenRAPM experiments", tab "In vs out of season".

**Model decisions.**
- **Shrinkage strength is the main lever.**  Out-of-season error wants a larger penalty: vanilla RAPM 7,620 vs
  4,420 in-season; RAPM on the linear box prior 22,640 (defense half of it) vs 7,620 (defense 0.75 of it).  At a
  fixed penalty the best prior multipliers are close (e.g. 13,130/0.5: offense 1.74 in-season, 1.60 out); the
  difference is how far the same season's games are trusted against the box prior.  Using one target's choice on the
  other costs 0.011-0.027 pts/100.
- **Help both:** adding a linear box prior to vanilla RAPM (t -9.7 in, -7.9 out), OpenRAPM's prior shrink (-14.7 /
  -8.9), the swap adjustment (-4.4 / -2.9).
- **Splits them:** OpenRAPM's boosted prior instead of the linear one, in-season t +1.4 (tie), out of season t +3.3.

**Stats (each added alone to RAPM on the linear box prior).**
- **Help in-season, hurt out of season (out-of-season t >= +2.4):** share of team possessions played (in -3.7, out
  +4.4), shot-making above expectation (two-point m2 +3.3, points per shot above expected +3.1, three-point +2.8),
  same-season plain RAPM (+2.7), team points scored with him on court (+2.6), assists (+2.6 to +3.2).
- **Help out of season more than in-season:** age (in -5.2, out -2.5), seasons played before (-2.6 / -3.0), career
  possessions before (-1.8 / -3.1) -- aging, which a forecast uses and a description of the season does not need.
- **Help in-season, neutral out of season:** most box rates (threes made and missed, blocks, steals, rebounds), body
  size, score context, off-court numbers.
- **Stability does not explain it.**  Across the 64 stats, the transfer gap (out-of-season minus in-season change) is
  nearly unrelated to within-season split-half reliability minus year-to-year correlation (correlation +0.10).
  Possession share fits the story (reliability 0.97 within a season, 0.62 across seasons); assists do not (0.95 and
  0.92) yet still hurt out of season -- the in-season fit gives them a weight that does not hold across seasons.
- 64 stats x 2 targets: about 3 per column cross |t| = 2 by chance; the coherent pattern (role, same-season
  plus-minus, shot-making luck) is the finding, not any single row.

## Experiment 39: what travels with a traded player -- player, team, or regression to the mean (2026-10-05)

**The owner:** "an in-season check that boosts the weight of traded players in the scoring ... I want to know which
pieces are regression to the mean, and which pieces are 'real things about a player'."  `scripts/111_traded_check.py`
(folds: `97 --split=deadline`; split quadratics: `107 --tags=deadline --traded=1`); development seasons only; sheet
"OpenRAPM experiments", tab "Traded players".

**The test.**  Each season is rated on its games before the date by which 60% of the regular season was played and
scored on the rest (playoffs included), and the reverse.  On the scored games a player is TRADED when he plays for a
team other than his team in the rating games (8.1% of scored player-possessions, 2.8-12.9% by season; 53 of 454 players
per fold), STAYED otherwise.  Every prediction splits exactly into the two groups' parts (`scorecard.split_traded`), so
a change is given to one group at a time with the other held at the reference.  Up-weighting team-game rows that
contain a traded player was the first design and was dropped: in those rows he is about 1 of 10 players, so the score
mostly measured the other nine.  Statistics: the calibration slope per group; the paired t over seasons of the change
in team-game MSE; the blend weight (forecast encompassing, season-clustered jackknife).

**Rating systems (calibration slope, stayed vs traded; 1 = calibrated, 0 = no signal).**
- **The box score travels.**  Linear box SPM alone: offense 2.31 vs 2.62, defense 1.23 vs 1.36.  The box-prior part
  of every RAPM keeps (or raises) its slope on a new team: RAPM on the linear box prior, offense 0.87 vs 1.36,
  defense 0.83 vs 1.00.
- **Defense beyond the box score stays with the team.**  The games part of RAPM on the linear box prior (the rating
  minus its prior): defense 1.08 for players who stayed vs 0.18 for traded players (difference -0.90, SE 0.36);
  plain RAPM's defense 0.97 vs 0.34 (SE 0.22).  Low in both directions (rated before the cut 0.33, after 0.03 -- so not
  selection on the trade) and both eras (1997-2011 0.02, 2012-2025 0.41).
- **Offense beyond the box score: era-dependent.**  Games part 1.27 vs 0.73 (SE 0.55) overall; 1997-2011 it travels
  (2.34), 2012-2025 it does not (-0.57).  Plain RAPM's offense: 1.73 traded in 1997-2011, 0.36 in 2012-2025 (stayed
  about 1.0 in both).
- **OpenRAPM as shipped is portable in the modern era:** 2012-2025 traded vs stayed, offense 0.97 vs 0.95, defense
  1.08 vs 0.99.  1997-2011: offense too narrow for traded players (1.46 vs 0.86), defense too wide (0.69 vs 0.97).
- Controls (every player at his team's average rating): stayed 0.91-0.95, traded -0.26 to 0.24 -- the traded test
  sees the split inside a team that the same-team team-game test cannot.

**Model decisions (given to one group at a time; t paired over 20 seasons; next season from experiment 38).**
- Adding a linear box prior to plain RAPM: same team t -5.9, traded -4.5, next season -7.9; blend weight 1.28 for
  traded vs 0.88 for players who stayed (the box prior is worth more on a new team).  Holds in every direction/era split.
- Boosted prior instead of the linear one: worse for both (+2.4 / +2.1) and next season (+3.3).
- OpenRAPM's prior shrink: same team -13.1, traded -1.6 (blend weight 0.28, SE 0.35; -3.0 in 2012-2025, -0.1 in
  1997-2011), next season -8.9.  Fit on, and mostly a correction for, players who stayed.
- Swap adjustment (x0.5): same team -3.7, traded -0.1 (+1.8 / -1.5 by direction), next season -2.9.  Same team only.

**Inputs (each added alone to RAPM on the linear box prior; 77 rows per column, about 4 cross |t| = 2 by chance).**
- **Travel with the player:** coach's usage -- share of games started (traded t -2.9), share of available minutes
  (-3.1), garbage-time share (-3.4), how close his games were (-3.1), average score gap (-2.8); age, body and career
  as a group (blend weight 0.91 traded vs 0.80 stayed).
- **Team context:** on-court plus-minus (traded t +2.8; next season +2.1), off-court plus-minus (+2.3), same-season
  plain RAPM (+2.4; blend weight 0.02 traded vs 0.59 stayed; next season +2.7).
- **Regression to the mean:** effective FG %, true shooting % and two-point % hurt the same team's later games (t
  +2.0 to +2.4) and traded players (+2.4 to +2.8): the first 60% of a season's shooting overstates the rest.
  Shot-making above expectation and assists have no in-season signal across the cut and hurt next season (+2.6 to
  +3.3).
- **Real at the moment, drifts:** share of team possessions played helps when rating and scored games are interleaved
  (random folds, experiment 38, t -3.7), not across the cut (+0.8), and hurts next season (+4.4) -- a role that
  changes over time.
- Single-row hits with no basketball reason (made free throws +2.6 / +3.0, age entering the league) are read as chance.

**Caveats.**  Traded players are not a random sample (often role players moving from sellers to buyers); their role
on the new team can change, which this test counts as "did not travel".  Power is set by 8% of possessions: system
slopes carry SE 0.1-0.3, games-part slopes 0.4-0.6.

## Experiment 40: a portable rating next to the team rating (2026-10-05, awaiting the owner's call on the site)

**The owner:** "Yes portable rating. It would be good to deal w/ hey these are often role players as well and ideally
openrapm can surface a traded rating and non (provided we adjust for the fact that traded players are worse often)."
`scripts/112_portable.py`, `src/eracoef/portable.py` (tests/test_portable.py); development seasons only; sheet
"OpenRAPM experiments", tab "Portable rating"; `outputs/season_ratings_portable.parquet`, `outputs/portable/`.

**The model.**  OpenRAPM as shipped = box-prior part + the part beyond the box score (games part + swap adjustment).
On held-out games each part's contribution is split by new team / same team and by playing-time tier
(season-equivalent possessions <1,000 / 1,000-2,500 / 2,500+).  Same-team players set each tier's slope; players on a
new team get the same tier's slope times a ratio, with tier and new-team level terms -- traded players are compared
with same-tier players who stayed.  Two samples: mid-season trades (experiment 39's deadline folds) and off-season moves
(season rated, previous and next season scored; a player on a team other than his rated-season team).

**Traded players are worse, and their ratings already know it.**  Players on a new team rate 0.83 pts/100 below
same-team players at the deadline (paired t -19.1) and 0.97 below in the off-season (t -20.7), on fewer possessions
(median 1,764 vs 2,826; 2,096 vs 3,350).  Matched on playing time, they play to their rating on the new team: the
new-team level is -0.07 (SE 0.12) on offense and +0.23 more allowed (SE 0.13) on defense, mid-season.

**About half of the plus-minus part travels.**  One ratio for the part beyond the box score, both sides: 0.49 (SE
0.12) from mid-season trades, 0.56 (SE 0.17) from off-season moves; pooled by inverse variance 0.51 (SE 0.10).  The
box part's ratio is not stable (offense 1.32 mid-season, 0.92 off-season), so it is held at 1.  By playing time the
plus-minus part travels about 0.6-0.8 for 2,500+ possession players and not at all below 1,000 (mid-season -0.26 to
-1.0, large SEs) -- the owner's role-player point, and the reason the comparison is made inside tiers.

**Held-out error does not separate them.**  Each version cross-fitted (season H's ratios without H-1, H, H+1) and given
to players on a new team only: no version lowers the error measurably.  Best, one ratio fit on off-season moves:
mid-season t -0.4, off-season t -0.9, blend weights 0.56 (SE 0.36) and 0.80 (SE 0.34).  The four-ratio and per-tier
versions overfit (off-season t +1.9).  Given to same-team players the portable rating is worse (t +5 to +8): it is a
second number, not a replacement.

**2026.**  The portable rating (box part + 0.51 x the part beyond it, re-centered) keeps Wembanyama, Giannis, Kawhi,
Jokic and SGA on top; the top falls 0.2-0.8 because stars' plus-minus parts are positive.  Rises: Draymond Green +1.32,
Westbrook +1.21, LeBron +1.01 (70th to 25th), Harden +0.95 (46th to 15th), Durant +0.71.  Falls: Paul Reed -1.13, Luka
Garza -1.09, Alex Caruso -0.93, Neemias Queta -0.80, Ajay Mitchell -0.77.  224 of 293 players with 2,000+ possessions
move 0.1 pts/100 or more.

**The site column (built locally, not published).**  `scripts/52_site.py` applies the published ratio
(`outputs/portable/production.json`, 0.508) to the PRODUCT table with `portable.portable_table`, so the column cannot
fall out of step with the ratings beside it; `docs/index.html` gains a sortable Portable column and portable
offense / defense / total in the CSV.  No explanatory text on the page (the owner: the drafted footer was "AI slop,
do not include").  The team ratings in `docs/data/ratings.json` are unchanged
(max difference 0.0 over 14,578 rows).  2026 on the product table: Wembanyama 7.57 -> 6.75 stays 1st; Jokic 4th -> 2nd;
Harden 31st -> 10th (+1.03); Jamal Murray 27th -> 16th; Curry 21st -> 17th; Queta 6th -> 11th (-0.84).

## Experiment 41: the swap adjustment's give-back rule (2026-10-06, awaiting the owner's call)

**The owner:** "Yes, run it" -- after experiment 40's read of Derrick White (2026: 2.60 before the swap step, 1.25 after,
the league's largest swap charge, -1.35, of which -0.99 is give-back; consensus 10th-11th).  Each team's total is held
fixed, so the team's net type prediction must be taken back from its players.  Shipped ("minutes"): the nearest point in
plain squared distance, so each player gives back in proportion to his share of the team's possessions.  Candidate
("flat"): the nearest point in possession-weighted squared distance, so every player of the team gives back the same
amount.  Diagnostic ("none"): no give-back, team totals move.  `swapadjust.solve(..., giveback=)`,
`91_swap_adjust.py --giveback=`; tests/test_swapadjust.py (+2).  Everything else as shipped; the minutes rule rebuilds
the incumbent exactly (difference 0.0 over 14,579 rows).  Chain: `scratch/2026-10-06_giveback/giveback_chain.sh`.

**What prompted it** (the owner: "how confident are we in the swaps in general re: being robust across all players
etc? white is #11 in consensus").  `scripts/113_swap_explain.py` rebuilds the shipped swap adjustment of every player
step by step (type x0.5, give-back, centering, spread hold) and matches the shipped table to 3e-14 (2024-2026).
Against the consensus (2024-26 pooled, 475 players with 1,000+ possessions; a sanity check, never a target): Spearman
0.809 before the swap step, 0.836 after the type, 0.827 after the give-back, 0.832 as shipped.  Each piece moves a
player toward the consensus 53-54% of the time, the whole adjustment 63%; for the 48 largest give-backs, 50%.  White:
consensus 10th, 51st before the swap step, 89th after (pooled give-back -0.95).  On average the adjustment holds up in
every test (swap test z +4.3, year over year every tier, in-season t -4.4); per player it is close to a coin flip.

| test (against the incumbent) | flat | none |
|---|---|---|
| year-over-year, team-game level | **+0.038, z +2.2, 18 of 56: worse** (8.6009 vs 8.5995) | +1.98, z +13.2: much worse |
| by quality tier, team-game | top 30 z +4.1, 31-90 +2.4, 91-150 +2.4, 151-300 +1.5, 301+ +1.0 | |
| year-over-year, stint level | **-0.213, z -7.9, 45 of 56: better** | z +11.7 |
| by quality tier, stint | better in every tier: top 30 z -1.8, 31-90 -4.8, 91-150 -6.7, 151-300 -7.0, 301+ -9.0 | |
| stint level, each side rescaled | -0.208, z -7.1, 42 of 56 | z +14.5 |
| swap test, order / gaps | -0.002, z -0.2 / -0.07, z -1.2: a tie | z -1.8 / z +8.3 |
| trade loss, offense / defense | +0.0006, z +1.9 / +0.0007, z +4.3: worse (top 30 offense z +3.9) | |
| consensus, 1,000+ possessions | 0.839 (from 0.834) | 0.824 |

- **The team constraint is essential**: with no give-back the type predictions move team totals and every test gets
  much worse.
- **Flat vs minutes splits by level**: the stint level (which sees bench lineups) prefers flat, most of all for low-tier
  players; the team-game level and the per-player trade loss prefer the shipped rule, most of all for the top 30.  By the
  decision rule (team-game z -2 or below) flat is not adoptable.
- **It is not White's fix.**  Site table, 2026: median move 0.06 per 100 for players with 2,000+ possessions (94 of 293
  move 0.1 or more; largest +0.38 Podziemski, -0.40 Jalen Williams).  White 1.25 to 1.47 (88th to 66th); pooled 2024-26
  rank 89th to 72nd against consensus 10th.  Boston's net offensive type is +0.75 per possession and has to come back
  from Boston's players under either rule (flat: 0.75 from each; minutes: about 8.3 x his share, 1.11 for White), while
  White's own type is small (+0.30 offense, -0.64 defense), so the charge survives any give-back rule.

## The shot-quality build (2026-10-06, the owner's request; in progress)

**The owner:** a VERY GOOD shot-quality model to replace the per-season distance curve, because noisy data in the
targets and priors is the biggest gap: far more from the play-by-play (a rebuilt shot clock, the possession around the
shot, the make or miss itself), and the public SportVU tracking data to teach it.  Rulings the same day: the quality is
**the shot alone** (a league-average shooter on this exact shot; the shooter's skill stays a separate padded term; no
defender ids or traits for now); "did it go in" means **update the estimate** (quality given the result: a made 18-footer
was probably more open); **no scorer shot-type tag is ever an input, not even dunk or putback**; defender distance from
STATS (2013-17) and from Second Spectrum (2017-18 on) are **different measurements** ("split eras"); KOBE is inspiration
only ("build our own"); the direct shot test first, then the year-over-year chain decides; the defensive target first.
Plan: `~/.claude/plans/read-handoff-md-what-i-dreamy-stonebraker.md` (ten gated stages, then experiment 42).

**Data built so far.**  `data/shotframe/<season>_<phase>.parquet`, one row per field-goal attempt 1997-2026
(`shotframe.py`, `scripts/114_shot_frame.py`): the stint parser's own shot logger (`stints.GameParser(log_shots=True)`)
records each attempt's x/y, the ten players on the floor, how its possession began (after a make, a made free throw, a
defensive rebound, a team rebound, a steal, a dead-ball turnover, a period start), the clock events since (offensive
rebounds, fouls, violations, timeouts) and the shot that handed the ball over; beside it `_aux`, every turnover and
free-throw trip.  The logger only appends: rebuilding every season with it on reproduces the cached stints exactly on all
252 parser columns (`--check=1`), and per shooter-game makes and attempts equal the stints' shooter table in 100% of rows.
Tracking: the 2014-15 shot log (OpenML 42806, 128,069 attempts of ~280 selected shooters to 2015-03-04), the 2015-16
movement archives (636, MIT licence), the 2015-16 per-date dashboards; all under `data/tracking/`.

**The direct shot test, registered before any model is scored** (`shottest.py`, `scripts/117_shot_test.py`,
`scripts/118_forward_test.py`).  Every arm is scored on seasons or games it never saw, paired by season.

| test | a new arm passes when |
|---|---|
| 1. held-out makes | lower log loss than the reference arm on twos and on threes in 24+ of 30 seasons at z <= -2; no era block worse at z >= +2; calibration slope 0.95-1.05 in every era block |
| arena veto | the arena's signal sd of expected points per 100 attempts no more than the reference's + 0.5, and the arena residual's correlation with it >= -0.5 (a scorer artefact reads near -0.9) |
| 2. tracking agreement | on 2015-16 movement labels or the 2013-14 to 2016-17 dashboards' per player-game defender bins, closer than its twin at z <= -2 |
| 3. other-half shooting | shooters with 100+ attempts a half, team offence, team defence on threes: lower error at z <= -2 for shooters and for defensive threes, nothing worse at z >= +2; the padding constant chosen on the other seasons for every arm |
| 4. forward test | first half of a team's games predicts its second half: lower at z <= -2 on one side, z < +2 on the other |

The tracking-taught model replaces the play-by-play make model only if it wins test 3 and test 2 and is no worse than
0.1% on test 1 in any era block, like for like (before against before; after against the make model plus the same
update).  Ties go to the make model.

**Reproduced first** (the stage-2 gate): the forward test of FINDINGS 35 to the third decimal -- raw 8.9066 / 8.2929,
every rate shrunk 8.4662 / 7.6607, outcomes only 8.4485 / 7.7106, and outcomes only with raw beside it 8.3157 / 7.4332
(z -2.76 / -2.65, 23 and 21 of 30).  The arena check already reads the shipped curve as partly the arena in 1997-99
(arena signal 2.0-2.4 points per 100 attempts against the offences' 1.2-1.9, residual correlation -0.55 to -0.75): the
arenas that recorded no coordinates for rim shots.

**Stages 1-4, 2026-10-06.**
- *Stage 1 (one row per attempt):* 6,328,669 attempts over 60 season-phases, no game failed, the stints identical on
  every season, 744,964 of 744,964 shooter-games equal to the stints' table.  The coordinate fix was changed after its
  gate failed: translating each arena-season to its densest near-rim point left the arena spread of the 0-3 ft share at
  8-9 points in 2021-26 (bar 4).  The three-point arc sits at the same distance in every arena (sd 0.23 ft), so the
  origin is not the problem; scorers place close shots 0.8-2.6 ft deep by arena.  `shotframe.rim_map` instead maps each
  arena-season's near-rim depth onto the league's, fitted on the VISITING teams' twos inside 10 ft (29 offences, so the
  league's mix): the arena spread falls to 1.9-2.7 points in 2021-26 and from 12-16 to 3-6 before 2011, while home
  teams keep a 4-6 point spread (their own style).  Coordinates only, no tags.
- *Stage 3 (join and shot clock):* the 2014-15 log joins 127,522 of 128,069 attempts by shot order (99.6%; the clock
  of the play-by-play runs 2.0 s behind tracking at the median).  The rebuilt shot clock against the log's, lags fitted
  on the other half of the games: median error 0.8 s, 62% within 1 s, 92% within 3 s, 77% in the same dashboard bin;
  arena bias sd 0.15 s; 79.4% of the log's 24.0 resets (tips and putbacks) rebuilt as an offensive-rebound reset within
  2 s (bar 80%).  Shot-clock violations, every season 1997-2026, after dropping the team rebound the feed logs at the
  whistle on a shot that never reached the rim: median 1-2 s, 68-74% within 2 s of zero, the same in 1997 as in 2015 --
  the rules carry across eras, but the bar as written (90% within 1 s) fails, because the feed's clock is in whole
  seconds and each kind of event is logged with its own delay.  So, the plan's fallback: the clock enters the model in
  four bands.  The 2018-19 rule is confirmed by the data: from 2019 the 14-second offensive-rebound reset rebuilds the
  violations tighter than the old 24 in all eight seasons (80-84% against 68-73% within 2 s of the median).
- *Stage 4 (the information-gap pilot, `scripts/119_info_gap.py`):* on the 127,152 tracked 2014-15 attempts, held out
  by game, every arm shooter-neutral.  Log loss: play-by-play 0.6484, the tracking teacher 0.6413 (-0.0072 a shot,
  game-paired z -20.8, better in 683 of 904 games), spot plus tracking alone 0.6421.  Learning curve (10 / 30 / 100% of
  the training games): teacher 0.6497 / 0.6434 / 0.6413, play-by-play 0.6544 / 0.6499 / 0.6484.  How much tracking
  quality varies among attempts the play-by-play cannot tell apart (the covariance of two teachers fitted on different
  games): sd 0.07-0.09 at the rim, 0.03-0.04 on other twos, 0.02-0.05 on threes (largest late in the clock), so the
  owner's update gives a shot's own result a weight of about 3% at the rim and 0.3-1% elsewhere.  Kill gate passed.

**Stages 5-9, 2026-10-06 (the owner's "go" after the pilot).**  The ablation table is `outputs/shottest/ablation.csv`
(`scripts/125_shot_ablation.py`) and the Google Sheet's "Shot quality" tab.  Every version prices a league-average
shooter, is trained on other seasons (never the rated season or either neighbour, never 2026) and is scored on
seasons or games it never saw; differences against the shipped distance curve unless named.
- *Two fixes found by the tests.*  (1) The rim map first left out the (0, 0) twos; arenas that logged their rim
  shots there had their located non-rim twos pulled toward the rim (+0.84 between an arena's (0, 0) share and its
  overpricing, 2005).  Counting (0, 0) twos at depth 0 fixed it: the arena spread of the 0-3 ft share is 1.8-3.4
  points in every season 1998-2026; 1997 stays at 8.4 (its coordinates are patchy game by game).  (2) The level set
  from the other half moved from four coarse bands to one-foot cells (padded 150 shots, like the curve's own bins).
- *Spot only* (distance and angle from x/y): ties the shipped curve on twos, slightly worse on threes (0.0003 a shot:
  the curve fits each season's deep threes in season); arena signal from 2011 cut from 2.0-3.6 to 0.5-1.0 points per
  100 attempts, the same as the curve before 2011.
- *The play-by-play model* (spot, possession start, putbacks, shot clock in four bands, game situation, time on court,
  the shot that handed the ball over; shooter-season and arena-season terms fitted and set to zero): held-out log loss
  -0.0056 on twos (29 of 30 seasons) and -0.0020 on threes (30 of 30) against the curve; 2018 log loss 0.6484 against
  Blackport's published 0.652 and the curve's 0.6532.  The possession start carries most of it, the shot clock a good
  share, game situation some on threes; time on court and the previous shot nothing; a fine clock curve ties the
  bands.  Gap to the tracking model's quality 40.8 / 47.5 squared points (2014-15 / 2015-16) against 55.0 / 63.5 for
  spot only and 61.8 / 66.8 for the curve; contest left -0.056 against -0.072 and -0.075 (se 0.006).  Forward test:
  with the model's team shot quality as the shrink target, -0.130 on defence (z -2.17) and -0.096 on offence (z -1.61)
  against the shipped luck adjustment.
- *Other-half shooting, against the flat league rate:* shooters' twos -1.37 (30 of 30), shooters' threes -0.47
  (z -3.5), team defence twos -0.086 (z -1.75; without the shooter and arena terms -0.104, z -2.13); but team defence
  threes +0.123 (z +2.0) and team offence twos and threes no better.  The shipped curve also loses to flat on defence
  threes (+0.068, z +2.0), and shrinking the model's within-spot spread on threes does not help (+0.12 to +0.32).
  Read: the threes a defence allows differ too little in quality (a corner three is about 3 points better than a wing
  three) for an estimate of it to beat the flat rate, which is what the shipped defensive target already uses.  **The
  registered pre-test for experiment 42 as planned (defensive threes) fails; defensive twos pass.**
- *Shooters' twos* lose ground once the possession start enters (+0.18 for its five early-transition flags alone):
  how much a fast break helps a shot varies by shooter, which one padded ratio per shooter cannot express.  Matters
  for the offensive target, not the defensive one.
- *The shooter and arena terms:* without them the model is slightly better in every other-half row (z -1.6 to -4.2)
  and slightly further from the tracking quality (+1.2 to +1.3); with them it is closer to "the shot alone".
- *The owner's update* (quality given the result; weight about 3% at the rim, 0.4-1% elsewhere, from both tracked
  seasons; table on the sheet): team offence twos -0.038 better (z -4.8), shooters' twos +0.036 worse (z +7.2),
  everything else within 0.006.
- *Tracking taught against makes taught* (`scripts/122_tracking_taught.py`, each tracked season held out, same rows in
  every version; the teacher's labels for this shooter, fixed after the first students were taught with the shooter
  taken out): the share of the label from tracking changes nothing (0.5, 0.8 and 1.0 identical to 0.003 squared
  points).  Counting the other tracked season 4x helps (-0.57 / -0.81 gap, z -20 / -23) -- and helps exactly as much
  with makes as the label, so it is the season next door, not tracking.  As the theory says: both labels have the
  same expected value, and against 6 million makes the extra precision is nil.  So tracking's value here is the
  yardstick (the gap and contest tests) and the size of the owner's update, not the training label.  Side finding:
  the 5-season closeness half-life is too flat.
- *Movement labels* (2015-16, `scripts/124_movement_labels.py`): 102,444 attempts from 631 games (96.4%).  On games
  the calibration never saw, defender bins agree with the official per player-game counts on 87.1% of attempts (bar
  85%), dribbles 88.4%, touch time 88.6%; the shot clock 72.0% (bar 90%) -- but the labels' clock distribution matches
  the 2014-15 log's true clock (medians 12.4 and 12.3 s, 10th percentiles 4.8 and 4.7), while the dashboards put 20%
  of all shots in "4-0 Very Late", which fits neither tracking source; the check is what fails.  The teacher does as
  well on the movement labels (out-of-fold log loss 0.6422) as on the official log (0.6412).
- Not run: the boosted-trees row of the ablation.  Test 2(c) used the dashboards pulled so far (2015 complete, 2016
  movement dates, 2017 to 3 March); rerun when the pull finishes.
- *Can the features learn true shot quality?* (the owner's challenge to the nested test; `scripts/126_quality_ceiling.py`,
  held-out tracked season 2014-15, 2015-16 in brackets, squared points of gap to the tracking quality, whose own spread
  is 235 (228)): distance alone 56.5 (63.9), 76% (72%) of true quality; the current model 38.7 (44.8), 84% (80%);
  boosted trees taught by the other tracked season's tracking labels 38.6 (43.8); the same trees taught by that
  season's makes 55.7 (58.0), and by 1.7 million makes of every other season 48.6 (56.6); the ceiling, boosted trees
  on the same season's tracking quality held out by game, 33.8 (39.2), 86% (83%).  So play-by-play can see at most
  83-86% of true quality, the current model already has 80-84%, and tracking labels ARE the better teacher for a
  flexible model (38.6 against 55.7 on the same shots) -- the simple shooter-neutral model just does not need them.
  Headroom for a tracking-taught correction on top: about 1-5 squared points (about 2% of true quality).  The owner:
  "Ok fair enough".
- *The level, and what happened with threes* (the owner asked, 2026-10-06).  The defensive-threes failure above was
  my design, not the shot quality: each half of a season took its level from the OTHER half's makes, and the league's
  make rate moves between the halves of a season by 0.40 points on threes (median 0.41, up to 0.85; 0.16 on twos) --
  as much as the whole spread in the quality of threes defences allow (sd 0.42 points; it repeats from half to half,
  r 0.69, against 1.72 points of pure chance per half in a defence's 3P% allowed).  With the level from the whole
  season, as the shipped curve and the flat rate already had, the play-by-play model beats the flat rate on every
  team row (`scratch` re-level of the saved raw logits, `data/shotq/pbp_full_seasonlevel`): defence threes -0.122
  (z -3.1, 23 of 30), defence twos -0.156 (z -3.2), offence threes -0.302 (z -6.8), offence twos a tie (-0.021),
  shooters' threes -0.827 (z -6.9), twos -1.457.  `shotmodel.relevel` now sets the season's level by default
  (`mode="half"` kept to rerun the record).  The leakage this admits is the curve's: one team's shots are about 1/30
  of a cell, and xshoot.align already sets each season's level from the season.  **The defensive-threes pre-test for
  experiment 42 now passes**; the ablation table above was built with the half levels and is superseded on this point.

### The shot-quality search: the protocol, registered before any trial (2026-10-06)

**The owner:** "let's go bananas on making the model as good as possible! random search, different models,
stacking, you name it. this is the most important thing we'll do!"  Spec: the design round's merged plan (saved with
the session; the protocol below is the binding part).  Code: `shotsearch.py` (split, rows, fast relevel, SALL),
`shotlearners.py`, `shotfeatures.py`, the regression's new options in `shotmodel.py`, scripts 127-132.

- **Split.**  SEARCH blocks 1997-99, 2003-05, 2009-11, 2018-20, 2021-23 (15 seasons): every trial is scored here and
  only here.  CONFIRM blocks 2000-02, 2006-08, 2012-14, 2015-17, 2024-26: read once, after the shortlist is registered
  below; they hold every tracked season and the live block.  Models for a search block train on its allowed other
  seasons as always (shotmodel.train_seasons), confirm seasons' attempts included; no confirm season is ever SCORED, and
  no choice is made on a confirm season's makes or on a tracking label.  `shotsearch.assert_phase` enforces it.
- **Objective: SALL**, shooter-adjusted held-out log loss: each scored regular-season attempt (heaves out) gets the
  arm's logit relevelled from the whole season, plus a ridge offset (25 Hessian units, never tuned) for its
  shooter-season and shot value fitted on the other half of the season's games under the same arm.  Raw makes reward
  knowing who takes which shots; SALL does not (unit test on a planted league).  Validated on the stored arms, search
  seasons only: the play-by-play model minus the version without shooter terms -0.062 per 1000 (z -1.0), minus spot
  only -4.27 (z -12.1, 15 of 15), minus the four-block model -0.215 (z -15.7); the same order at ridge 10 and 100.
- **Rows.**  One fixed permutation per season; the search uses the first 750,000 training rows of each block (nested in
  the 1.5M of a final build); penalties are entered per 1.5M rows and scaled.
- **Logged, never optimised:** raw log loss, calibration slope, arena signal, fit time.  Optuna (QMC then TPE, seed 0),
  studies resumable; trial 0 is the current model; a placebo family (the current model on other row samples) gives the
  noise floor.  Every trial's logits on the search seasons are kept for stacking.
- **Shortlist:** per family the plateau within one season-paired standard error of the best; at most two per family and
  six overall, stacks included; written here before the confirm read.  **Decision rule:** the current model enters
  experiment 42 by default; a challenger replaces it only if, on the CONFIRM seasons, it is better on SALL at
  Holm-corrected z in at least 4 of 5 blocks, closer to tracking quality (2015, 2016) at z <= -2 and worse on no
  tracking row, and passes the arena check, the other-half rows and the forward test; ties go to the simpler model.
- **Expected:** a gain of 0.0002-0.0005 log loss per shot over the current model, at most about 0.001: on the tracked
  seasons the current model already captures 80-84% of true quality against a ceiling of 83-86%.

### The timing leak: every clock input is now read from the row before the shot (2026-10-06)

**Found by** the first boosted search trial: the game clock was its top input, and it beat the regression by 1.21 per
1000 attempts, twice the regression's whole gain.  **Cause:** the play-by-play stamps makes and misses with different
delays, so any time measured to the shot's own stamp (seconds into the possession, the shot clock, time since the
offensive rebound, time on court, the game clock) carries the result.

**Measured on the 2015-16 movement data** (the true release frame of each tracked attempt), log loss per 1000:

| Timing measured from | Gain over no timing | Of which the result leaking |
|---|---|---|
| the shot's stamp, fine | -2.93 | -1.59 (z -6.6) |
| the true release (the real signal) | -1.35 | 0 |
| the stamp, in the model's four clock bands | -0.71 | -0.26 |
| the last row before the shot that cannot belong to it | -1.18 | none from the shot's stamp |

- **The fix:** `stints` logs each attempt's **anchor**, the last made or missed shot, free throw,
  rebound, turnover, substitution, timeout or jump ball before it.  Fouls, violations and blank steal or block rows are
  skipped: in 3-3.6% of attempts the previous row has the shot's own clock reading, and those rows are mostly the
  shot's own foul (make rate 0.59 against 0.43-0.47).  `shotframe.anchored` re-measures every clock input from the
  anchor, and the shot clock is rebuilt at the anchor.  The logger also counts the possession's clock events up to and
  including the anchor row (`anchor_nev`, FRAME_VERSION 4): the clock rebuild and the timeout and foul fields read only
  those, by feed order, so no clock reading is ever compared with the shot's stamp.  (The first version kept the
  same-second rule against the anchor's clock instead; it dropped the anchor's own timeout and every foul logged at
  the second of a substitution that followed it.)  The stamp-based columns stay in the table as `*_stamp`, banned
  as inputs (`shotfeatures.FORBIDDEN`).  The previous attempt of the possession keeps its own stamp: it is an anchor
  row, at or before this shot's anchor.
- **What it voids:** every search trial so far, the stage 6-8 ablation, the tracking teacher's play-by-play side, the
  update size, and the other-half test wins (their models read the stamp).  The forward test never reads a stamp.  All
  are rerun on the anchored table before the shortlist; the studies restart from trial 0.
- **The anchor's own stamp does not leak** (2026-10-06, the same 97,865 tracked attempts, `xgboost` five game folds,
  tracking inputs in every arm; log loss per 1000):

  | Comparison | Change | z |
  |---|---|---|
  | timing from the shot's stamp, minus timing from the true release (the leak, re-measured) | -1.41 | -6.3 |
  | timing from the true release, minus no timing (the real signal) | -1.50 | -5.8 |
  | timing from the anchor, minus no timing | -1.35 | -6.1 |
  | the anchor's timing added on top of the true release (real or leaking) | -0.02 | -0.1 |
  | anchor attempts within 1 s of the shot's stamp (4.3%), the same | +1.47 per 1000 of them | +1.0 |

  The anchor keeps 90% of the real timing signal and adds nothing the true release does not already know.  Across
  all 6.3M attempts the anchor shares the shot's second in 3.2%, almost all offensive rebounds (tip-ins), which make
  LESS than their spot (-2.7 to -11.2 points), the opposite of a leak.  Two tiny tied groups make more than their spot:
  a defensive rebound (0.06% of 2016-26 attempts, +11 points) and a turnover (0.03%, +12 points), both breakaways;
  left in.

### The feature step on the anchored table (2026-10-06)

The registered rule: a new input joins the regression search as a toggle only if it improves SALL in at least 4 of the
5 search era blocks on the attempts it targets.  Add-one arms on the current model, 750,000 rows, the same rows as
trial 0 (log loss per 1000, by search block 1997-99 / 2003-05 / 2009-11 / 2018-20 / 2021-23):

| Input | Target | Blocks better | Verdict |
|---|---|---|---|
| end of period (`late`) | all | 5 of 5 (-0.031 / -0.024 / -0.035 / -0.028 / -0.004) | in |
| scramble after a missed or blocked shot | rim | 4 of 5 (+0.050 / -0.065 / -0.020 / -0.536 / -0.448) | in |
| distance to the three-point line | long twos | 4 of 5; threes 2 of 5 (all of it 1997-99, -0.577) | in |
| previous attempt's location | all | 3 of 5 | out (passed on the stamp table) |
| after a timeout | all | 4 of 5, mean +0.002 | out |
| earlier defensive fouls | all | 2 of 5 | out |
| court side | all | 2 of 5 | out |

Placebo (the current model on row samples 1-3): all-attempts +0.017 / -0.020 / +0.011, rim +0.17 to +0.23 (seed 0
is a lucky rim draw), threes -0.28 to -0.37 (an unlucky one), so per-shot-type gains under about 0.3 are within row
noise.  The audits (same-second, arena, tag proxy) ran on the stamp table; the same-second audit is now replaced by
construction (`anchor_nev`), and 128 reruns in the final battery.  The regression search restarted from trial 0 with
toggles late, scramble, line.

### The shortlist, registered before the confirm read (2026-10-06)

The search half on the anchored table (750,000 rows, the 15 search seasons; SALL per 1000 attempts against the current
model on the same rows; every arm better in 15 of 15 seasons):

| Arm | All | z | Rim | Mid-range | Threes | Trials (complete) | Overfitting probability |
|---|---|---|---|---|---|---|---|
| best regression (`glm:48`) | -0.728 | -7.6 | -1.830 | -0.203 | -0.144 | 60 (39) | 0.00 |
| + XGBoost (`xgb:13`) | -1.110 | -8.8 | -2.665 | -0.388 | -0.283 | 40 (28) | 0.04 |
| + LightGBM (`lgbm:25`) | -1.098 | -9.6 | -2.711 | -0.347 | -0.243 | 40 (31) | 0.02 |
| + chimeraboost (`chimera:26`) | -1.106 | -9.0 | -2.756 | -0.326 | -0.271 | 30 (22) | 0.02 |
| + chimeraboost, quality 3 | -1.106 | -9.0 | | | | fixed row | |
| + chimeraboost, quality 5 (8-model bag) | -1.082 | -8.8 | -2.692 | -0.329 | -0.252 | fixed row | |
| + CatBoost (`cat:12`) | -1.016 | -9.4 | -2.537 | -0.311 | -0.217 | 30 (18) | 0.14 |
| equal stack of XGBoost, LightGBM, chimeraboost | -1.150 | -9.2 | -2.790 | -0.383 | -0.294 | | |
| fitted stack of the same | -1.152 | -9.1 | | | | | |
| fitted stack of all five | -1.154 | -9.2 | | | | | |

- Boosters sit on the best regression (offsets kept, zeroed when pricing).  XGBoost's gain ranking: the regression's
  own margin first, then the spot (x, y, distance, line distance, angle), then the anchor's game clock, time on court
  and time since the previous attempt.  The earlier coordinate test (exact x/y removed, distance to whole feet) moved
  the booster gain little, so the spot detail is not the scorer's words in disguise.
- **Stacks:** fitted weights move by up to 0.28 between search blocks (the regression gets 0 everywhere, CatBoost
  almost 0), so by the rule the EQUAL stack is the candidate.  The stack optimiser first stopped at its start (scipy's
  default 0.00025 first step from zero logits); it now starts with unit steps and fits on season-relevelled log loss
  (SALL refits the shooter ridge per call), scoring still by SALL.  Overfitting probability: best trial picked on 7
  random search seasons, share of 2000 splits where it ranks below the median on the other 8.
- **Shortlist (5):** `glm:48`, `xgb:13`, `lgbm:25`, `chimera:26`, the equal stack `eq:xgb:13+lgbm:25+chimera:26`.
  CatBoost is dominated and left out; the neural net was cut (first to cut in the spec).  The decision rule is the one
  registered above (protocol and spec section 5); ties go to the simpler model.
- **Confirm read:** `scripts/131_shot_confirm.py`, the current model and each candidate at 1.5M rows on the five
  confirm blocks, row samples 0, 1, 2; read once.

### The confirm read (2026-10-07)

`131_shot_confirm.py`: each candidate and the current model at 1.5M rows on the five confirm blocks (15 seasons no
choice had read), row samples 0, 1, 2; SALL per 1000 attempts against the current model on the same rows, sample 0
(season-paired z over 15 seasons; Holm across the shortlist; retention = confirm gain over search gain):

| Candidate | All | z | Rim | Mid-range | Threes | Blocks better | Retention | Sample sd |
|---|---|---|---|---|---|---|---|---|
| best regression | -0.509 | -6.1 | -1.237 | -0.239 | -0.116 | 5 (rim 4) | 0.70 | 0.040 |
| + XGBoost | -1.166 | -9.2 | -2.920 | -0.443 | -0.234 | 5 | 1.05 | 0.059 |
| + LightGBM | -1.200 | -8.7 | -3.097 | -0.401 | -0.209 | 5 | 1.09 | 0.067 |
| + chimeraboost | -1.175 | -9.7 | -3.119 | -0.380 | -0.138 | 5 | 1.06 (threes 0.51) | 0.051 |
| equal stack of the three | -1.259 | -10.0 | -3.199 | -0.452 | -0.238 | 5 | 1.09 | 0.058 |

Head to head (sample 0; the three-sample means agree): the stack minus LightGBM -0.059 (z -4.1), minus XGBoost
-0.092 (z -7.7), minus chimeraboost -0.084 (z -5.1); LightGBM minus the regression -0.690 (z -6.9); the regression
minus the current model -0.509 (z -6.1).  Every Holm p is below 1e-4.  Rule 1 (confirm SALL) passes for every
candidate; rule 7 (overfitting probability at most 0.25) passes for every family.  The remaining rules need the
battery on all 30 seasons: finalists the regression, LightGBM and the equal stack, priced by `132_shot_price.py`.

### The location leak: the scorer logs a made close shot nearer the rim (2026-10-07)

**Found by** the battery: the boosted models fit makes far better than the current model (confirm SALL -1.2 per 1000)
yet sat FARTHER from tracking quality (2015 +1.99 squared points, z +16; 2016 +2.41, z +16), and their departure from
the current model predicted makes beyond the tracking teacher and this shooter (coefficient 0.69-0.78, se 0.06).

**Cause:** at the same true distance, the play-by-play places made shots closer to the rim than missed ones.
- Against the NBA shot log's own tracking distance (2014-15, 127,152 attempts): made minus missed, logged minus true,
  -0.68 ft at 0-2.5 ft, -0.85 at 2.5-4.5, -0.90 at 4.5-6.5 (se 0.02-0.04), -0.17 at 6.5-10.5, none beyond.
- Against the movement data's release distance (2015-16, 97,865): -0.77 / -1.01 / -1.31 / -0.70 / -1.07 ft in the
  0-2 / 2-4 / 4-6 / 6-10 / 10-16 ft bands, none beyond 16 ft; the same within every release height (so not dunks).
- Two independent tracking pipelines agree; the play-by-play spot is the common factor.

**Measured** (xgboost, five game folds, possession facts, anchored timing and tracking contest inputs in every arm;
log loss per 1000 attempts; info = true coded distance vs none; leak = logged coded vs true coded, negative leaks):

| Distance coding | Info 2015 | Info 2016 | Leak 2015 | Leak 2016 |
|---|---|---|---|---|
| exact | -17.54 | -13.45 | -7.93 (z -14.5) | -8.07 (z -13.5) |
| whole feet | -17.20 | -13.07 | -6.29 | -6.24 |
| 2-ft bands | -16.74 | -12.81 | -4.73 | -4.30 |
| zones 0-4 / 4-10 ft, exact beyond | -15.35 | -10.74 | +0.77 | -1.95 (z -5.1) |
| zones 0-6 / 6-10 ft, exact beyond | -15.41 | -12.21 | +3.46 | +1.47 |
| one 0-10 ft zone, exact beyond | -10.78 | -9.93 | +2.09 | +1.52 |

The logged location (distance, angle, x, y) beats the true release distance by 8.0 per 1000 (z -12.9), 16.3 inside
16 ft; with the ball's release height added, still 5.9 (z -9.0).  Beyond 16 ft the logged spot is worse than the truth.

**What it touches:** every play-by-play shot model, the current one and the shipped curve included (all read the exact
spot); the boosters most (fine x/y near the rim); the rim / mid-range split (made at 4.3 ft logged at 3.5 moves
sub-model); the per-band level; and every test that prices a shot with its own leaky spot (other-half, forward).  The
shipped ratings are untouched: the defensive target reprices threes only, and threes show no bias.

**The fix:** inside 10 ft the spot is coded as two zones, 0-6 and 6-10 ft (the rim sub-model becomes 0-6 ft), with no
x, y, angle or side; exact beyond 10 ft.  The exact logged spot is kept as `*_logged` columns, banned as inputs.  The
tracking teacher gets the TRUE distance.  The search reruns on the coded table.

**The rebuild on the coded spot (2026-10-07).**  The tracking teacher with the TRUE distance and the coded logged spot:
out-of-fold log loss 0.6503 (2015) and 0.6504 (2016), against 0.6419 / 0.6428 with the exact logged spot -- about the
8 per 1000 the leak was worth.  The current model's search SALL rose from 0.65542 to 0.66469 (+9.3 per 1000): the leak
was that much of its apparent accuracy.  Placebo: all-attempts -0.03 to -0.06, rim +0.06 to +0.09, threes -0.28 to
-0.38 (row noise as before).  Feature step, the same rule (4 of 5 search blocks on the target):

| Input | Target | Blocks better (mean per 1000) | Verdict |
|---|---|---|---|
| end of period | all | 5 of 5 (-0.035); rim 5, mid 5 | in |
| distance to the three-point line | long twos | 5 of 5 (-0.013); threes 2 of 5 | in |
| scramble after a missed or blocked shot | rim | 3 of 5 (-0.066); mid 5 of 5, all 4 of 5 | out (passed only on the leaky spot) |
| previous attempt's location | all | 3 of 5 | out |
| after a timeout | all | 2 of 5 | out |
| earlier defensive fouls | all | 3 of 5 | out |
| court side | all | 1 of 5 | out |

The regression search restarted from trial 0 with toggles end of period and line distance; CatBoost is not rerun (it
was dominated on the leaky spot and costs an hour); XGBoost, LightGBM and chimeraboost are.

**The search on the coded spot (2026-10-07), search half, SALL per 1000 against the current model:** best regression
(`glm:58`, blocks + end of period, interactions spot2d / start x time / putback x distance / context) -0.217 (z -3.4,
14/15; rim -0.284, mid -0.101, threes -0.327 at z -1.3, inside row noise); + XGBoost (`xgb:38`) -0.594 (z -4.9; rim
-1.043); + LightGBM (`lgbm:32`) -0.571 (z -5.1; rim -1.026).  On the leaky spot the same families read -0.73 / -1.11 /
-1.10: most of the regression's gain was the leak, the trees' added part (-0.38) survives.  XGBoost's inputs by gain
(2018-20): the anchor's game clock 0.12, time on court 0.10, the regression's margin 0.09, score margin 0.07, time
since the previous attempt 0.06 -- game context, no fine spot.  The margin is the score BEFORE the shot (stints adds the
points after logging it).

**Against the clean teacher** (true distance; the 2015-17 block priced alone by 132 --block=2015):
- Gap to the teacher, minus the current model: regression +0.16 (z +2.5) / +0.47 (z +6.6); LightGBM +0.57 (z +5.8) /
  +1.21 (z +10.0) in 2015 / 2016 -- both FARTHER, so both fail registered rule 2 as written.
- Yet each one's departure from the current model predicts makes beyond the teacher and this shooter's season level:
  coefficient 0.53 / 0.47 (se 0.14-0.15) for the regression, 0.73 / 0.79 (se 0.09-0.10) for LightGBM.
- Reading: the teacher has the CURRENT model's play-by-play form plus tracking, so any added play-by-play structure
  (nonlinear game context) is something it cannot express; the gap test then penalises a change of form whether or
  not the extra is real.  The coefficient test says the extra is real make signal that tracking does not explain.
  Whether game-context make effects belong in "shot quality" is the owner's call (asked 2026-10-07).

### The shortlist on the coded spot, registered before its confirm read (2026-10-07)

Search half, SALL per 1000 against the current model (overfitting probability per family in brackets):
best regression `glm:58` -0.217 (z -3.4) [0.007]; + XGBoost `xgb:38` -0.594 (z -4.9) [0.000]; + LightGBM `lgbm:32`
-0.571 (z -5.1) [0.000]; + chimeraboost `chimera:13` -0.598 (z -4.8) [0.000]; equal stack of the three -0.627
(z -5.1; fitted weights move up to 0.36 between blocks, so equal).  **Shortlist (5):** `glm:58`, `xgb:38`,
`lgbm:32`, `chimera:13`, `eq:xgb:38+lgbm:32+chimera:13`.  The confirm read runs while the owner rules on whether game
context belongs in shot quality; with "tracking-visible only" the current model stays whatever the read says.

**The confirm read on the coded spot (2026-10-07)**, 1.5M rows, confirm seasons, SALL per 1000 against the current
model (sample 0; sample sd under 0.04 throughout; every block better unless noted):

| Candidate | All | z | Rim | Mid-range | Threes | Retention all / threes |
|---|---|---|---|---|---|---|
| best regression | -0.187 | -7.8 | -0.312 | -0.153 | -0.066 (4 blocks) | 0.86 / 0.20 |
| + XGBoost | -0.648 | -9.0 | -1.256 | -0.364 | -0.204 | 1.09 / 0.45 |
| + LightGBM | -0.636 | -10.1 | -1.256 | -0.344 | -0.193 (4 blocks) | 1.11 / 0.47 |
| + chimeraboost | -0.660 | -8.5 | -1.261 | -0.377 | -0.228 | 1.10 / 0.51 |
| equal stack | -0.692 | -9.5 | -1.307 | -0.398 | -0.255 | 1.10 / 0.54 |

Rule 1 (better on every sub-model at Holm z in 4+ blocks, retention at least 0.5, gain above 2 sample sd) passes for
the equal stack and chimeraboost only; the others miss on threes retention (the search-half three gains sat inside row
noise).  Finalists for the battery: the regression and the equal stack, priced on all 30 seasons by 132.

### The battery on the coded spot and the decision (2026-10-07)

All 30 seasons priced by `132_shot_price.py` (each block's model fitted exactly as in the confirm read); arms the
current model (`c_L7`), the best regression (`c_G58`), the equal stack (`c_stack3`); against the current model:

| Test | Regression | Equal stack | Registered bar |
|---|---|---|---|
| held-out makes, twos (30 seasons) | -0.00021 (z -10.3, 30/30) | -0.00087 (z -13.3, 30/30) | |
| held-out makes, threes | -0.00017 (z -1.9, 24/30) | -0.00038 (z -3.7, 28/30) | |
| threes calibration slope 2017-22 / 2023-26 | 0.942 / 0.927 | 0.914 / 0.911 | 0.95-1.05 (current 0.958 / 0.948) |
| gap to the clean teacher 2015 / 2016 | +0.16 (z +2.5) / +0.47 (z +6.6) | +0.37 (z +3.8) / +1.08 (z +9.0) | none worse at z >= +2 |
| contest left 2014-17 | about equal | slightly nearer zero, all 4 | not worse |
| other-half, team defence twos | +0.011 (z +3.5) | +0.008 (z +2.0) | none worse at z >= +2 |
| other-half, team defence threes | +0.000 | +0.004 (z +0.5) | |
| other-half, shooters' threes | -0.020 (z -1.4) | -0.054 (z -3.0) | |
| forward test, offence / defence (MSE; current 8.323 / 7.677) | 8.338 / 7.711 | 8.337 / 7.704 | not worse at z >= +2 |

**Decision (the registered rule): the current model, on the clean table (anchored timing, coded spot), enters
experiment 42; no challenger replaces it.**  The challengers win on makes through game context (the anchor's game
clock, time on court, score margin, time since the previous attempt), which is real pre-shot make signal (the
coefficient test) but does not carry to the team-level tests the ratings use: worse other-half team defence on twos,
slightly worse forward test on both sides, and the stack's three-point spread too wide in recent seasons.  So the
owner's question (does game context belong in shot quality) does not change the outcome under the registered rule.
Experiment 42's pre-test now passes: on the clean table the current model beats the flat rate on other-half team
defence threes (flat +0.088, z +2.2, flat better in 7/30 seasons); on the leaky table no estimate did.  The curve
(`lp`, which reads the exact logged spot) still "wins" defence twos and the forward defence -- the leak again.

**The update on the clean table (2026-10-07)**, `119 --base=c_L7` (the current model priced from other seasons as the
play-by-play quality; v from two teachers on different games, both with the true distance): the teacher beats the
play-by-play model by 17.7 per 1000 (z -45) -- the coded spot gave up the near-rim detail tracking has.  The weight a
shot's own result gets: rim 0.073-0.098 (was about 0.03), mid-range 0.009-0.022, threes 0.004-0.016 (corner 0.008).
Written to outputs/shottest/info_gap_v_clean_L7.csv.  Nearly every attempt sits in the 10+ s shot-clock class now
(the clock is read at the anchor, mostly the possession's start), so the clock classes of the update carry little.

### The verdict, corrected (2026-10-07; the owner: "So the logistic beats everything? Really?")

The decision above ("the current model stays; no challenger replaces it") was wrong as stated.  A verification run
(seven agents; scripts and outputs under scratch/2026-10-07_verify/, outputs/shottest/ctxfree_*) found:

1. **The trees fitted on the CURRENT logistic beat it on makes and are no worse on any team test.**  Arm LX = XGBoost
   (the parameters of `xgb:38`) as a correction on the current model's margin (`132 --spec`, data/shotq/c_L7_xgb38):

   | Test (arm minus current model) | LX | z | Stack (on the tuned logistic) |
   |---|---|---|---|
   | SALL x1000, all 30 seasons / 15 confirm seasons | -0.644 / -0.623 | -12.8 / -10.0 (30/30, 15/15) | -0.747 / -0.690 |
   | makes x1000, twos / threes | -0.79 / -0.25 | -15.7 / -7.8 | -0.87 / -0.38 |
   | other half: shooters 2s / 3s | -0.026 / -0.070 | -3.3 / -4.5 | -0.011 / -0.054 |
   | other half: offence 2s / 3s | -0.008 / -0.017 | -2.2 / -3.7 | -0.005 / +0.005 |
   | other half: defence 2s / 3s | -0.002 / -0.001 | -0.6 / -0.2 | +0.008 (z +2.0) / +0.004 |
   | forward test offence / defence (sel) | +0.0003 / -0.0041 | +0.1 / -0.4 | +0.013 / +0.027 (z +2.1) |
   | threes calibration 2017-22 / 2023-26 (current 0.958 / 0.948) | 0.925 / 0.915 | | 0.914 / 0.911 |

2. **The stack's team-test losses came from its base, not its trees.**  The tuned logistic's signed score-margin terms
   raise the make chance of trailing teams' shots, so a strong defence's "quality allowed" takes on its lead
   (defence-season shift r +0.51 with team strength; margin terms carry ~100% of it).  Removing the team-level shift
   halves the defence-twos loss (G58 z +3.47 -> +1.53; stack z +2.00 -> +1.15).  Against their own base the trees never
   lose a team test (stack minus tuned logistic: other-half cells z -3.1 to +0.9).
3. **The borderline failures were borderline.**  Forward test, paired z against the current model: offence noise for
   every challenger (z 0.6-1.1); defence the tuned logistic reliably worse (season z 3.0, block z 2.5), the trees
   borderline (season z 1.95-2.2, block z 1.5-1.7, under 1.7 without 1997-2001); every forecast moves by 0.03-0.07
   points per 100.  Other-half defence twos for the stack: z +2.00 on the battery split, +1.5 averaged over 200 random
   game splits, z +1.49 at half the padding constant.  Threes calibration: 95% intervals +-0.045, the current model
   itself 0.915 in 1997-2001 and 0.948 in 2023-26, so the 0.95-1.05 bar separates nothing; paired, the trees ARE
   over-spread on recent threes (stack minus current -0.044, z -6.3, surviving shooter fixed effects), costing about
   0.02 per 1000 -- a recalibration, not a mispricing.
4. **The tracking yardstick was biased toward the incumbent.**  With teachers that can express the challengers' form
   (T3 = LightGBM on the tuned form plus tracking, the best teacher out of fold), every challenger is CLOSER to tracking
   quality in 2015 (stack -1.04, z -9.6) and closer or tied in 2016 (-0.55, z -4.2; tie under the unscaled teacher).
   The tracking-specific part (teacher with tracking minus the same form without) endorses 18-41% of the trees'
   departures, all positive (z 5-17).
5. **The skeptic found no leak behind the trees' edge**, but found a third leak shared by every model: **unlocated (0, 0)
   twos are result-coded in older seasons** (1997-2004: 39.4% of made twos unlocated against 22.9% of misses; one arena
   in 2000: 46.3% against 1.4%; 2016-26: 2.9% against 2.0%).  Also a residual leak at the 6-ft zone edge (made shots
   logged inside 6 ft 85% vs 75% of misses at true 3-9 ft; worth -0.76 / -1.93 per 1000 of tracked attempts), and a
   minor stamp leak through `since_reset` on forced possession starts (0.044% of rows).  All three touch twos only;
   threes are clean.  The trees' edge does not grow with these gaps.

**Corrected reading:** the trees on the current logistic are the better shot model by every test that is not biased
toward the incumbent, except an over-spread on recent threes that a recalibration can fix.  LX was not on the
registered shortlist (it was built during the verification, after the confirm seasons had been read for related
arms), so the confirm half no longer counts as untouched for it; the decider is experiment 42's ratings chain, run on
both qualities.  The unlocated-shot leak must be fixed before any use of twos.

## Experiment 42: opponent threes priced at shot quality (registered 2026-10-07, before any result)

**The owner:** "go" (2026-10-07), on running experiment 42 on both the current logistic and the trees on top of it.
**What changes:** the defensive target `x3def_w0.25` prices each opponent three at the shooter's flat padded
other-half 3P%.  The candidates price each three at its shot quality AFTER its own result (`q_after`, the owner's
update, weight about 0.8% on threes) times the shooter's padded other-half ratio on quality BEFORE the result
(p_pad / p_mix, k 450).  Everything else is the incumbent's: offence `xpts_ft`, the kept share 0.25, the prior,
`onc_*`, every setting of 62, 99, 91.
- **42** `quality_logistic`, target `x3def_qL7_w0.25`: quality from the current logistic on the clean table
  (data/shotq/q42_L7).  **The primary arm.**
- **42x** `quality_trees`, target `x3def_qLX_w0.25`: XGBoost on the current logistic, threes recalibrated (alpha 0.87,
  fitted on the 15 search seasons' threes) (data/shotq/q42_LXr).  Adopted only if it passes the bar on its own AND
  beats 42 head to head on 63 at z <= -2; otherwise 42 is the reading (ties go to the simpler).
- **Bar (HANDOFF):** 63 both-directions team-game `game_armse` at z <= -2 against the incumbent's 8.5995 (a fresh 62
  reproduced the incumbent's 2015 and 2024 ratings exactly, max difference 0.0); no gross consensus miss (64); the 2026
  top 20 not worse (66); read 90, 88 by tier, 73 (on the incumbent's trade-loss target, and on the candidate's own as a
  sensitivity), scale_def (trap 17) and the absorbed share (trap 12).
- **Plumbing:** scripts 133 (quality per attempt), 134 (stint side tables and shooter tables; gates: every stint slot's
  attempts and every game x shooter's attempts and makes equal the cached stints; the frame and the quality agree on
  every attempt's value and result), `xshoot.expected_threes_q`; stints' shot logger places each attempt in the stint
  its possession record lands in (`rec_idx` set when the record is appended; FRAME_VERSION 5).  Checks passed on 1997:
  the incumbent target equals the committed code exactly; every quality at the league 3P% reproduces today's expected
  threes to 4.4e-16; an independent attempt-by-attempt recomputation matches to 8.9e-16.  An adversarial review found
  one blocking defect (attempts of a possession dropped at a period's end were placed in the next period's first stint;
  fixed, test added) and the small ones below (fixed where marked).
- **Accepted exceptions, recorded before the result:** (1) the update's size v comes from the 2015 and 2016 tracked
  attempts and is used in every season, rated 2014-2017 included (24 class-level variances; a +-30% change moves the
  target's own-result share by about 0.002); (2) alpha 0.87 and the trees' hyperparameters were fitted on the search
  seasons' makes, some of which are scored neighbours (one league-wide scalar); (3) a second-order channel: the prior
  for season s trains on labels of other seasons whose quality models may have trained on s's scored neighbours (league-
  wide, shooter- and arena-neutral pricing; far below 63's resolution); (4) the current logistic's quality has score-
  margin and end-of-period terms, so a leading defence's opponent threes are priced slightly lower (per defence-season
  0.02-0.05 points per 100; the owner's open question whether game context belongs in quality); (5) fixed: the threes'
  level is now fitted on located threes only (1997's unlocated misses had pulled located threes 1.3 points low).

**Result (2026-10-07): both arms are WORSE than the incumbent; neither passes the registered bar.**

| 63, both directions (56 season-pairs) | 42 (current logistic's quality) | 42x (trees' quality) |
|---|---|---|
| team-game `game_armse` (incumbent 8.5995) | 8.6043, +0.128, z +2.83, 23 of 56 better | 8.6026, +0.080, z +1.77, 24 of 56 |
| stint level / each side rescaled | z +4.70 / +4.61 | z +3.03 / +2.95 |
| scale_def (incumbent 1.0051) | 1.0186 | 1.0223 |
| every quality tier (top 30 ... 301+) | worse, z +2.3 to +2.8 | worse, z +1.0 to +2.3 |
| 90 swap test, defence order / gaps | -0.031 (z -1.8) / +0.17 (z +4.2) | |
| 73 trade loss, defence (incumbent's target / own target) | +0.0026 (z +3.7) / +0.0061 (z +8.4) | z +2.8 (incumbent's target) |
| 64 consensus rho_def / spread_def / top5 (incumbent 0.816 / 0.982 / 5) | 0.825 / 0.966 / 4 | 0.823 / 0.957 / 4 |

The loss is in the ranking (the rescaled rows match the raw ones), on defence, in every tier.  2026 top 20 (42): in
Curry (22 -> 12), Caruso (42 -> 14), Anunoby (21 -> 18); out Diabate (14 -> 27), Ighodaro (18 -> 51), Gonzalez (20
-> 21); defence moves sd 0.43 points among 1,000+ possession players.  Logged as experiments 42 and 42x (pending the
owner's call).  A diagnosis (target components, where the loss sits, run artefacts) is running.

**Diagnosis (2026-10-07, three independent angles; scratch/2026-10-07_exp42/diag_*):**
- **The target change itself is small and, at team level, better.**  Per defence-season it moves the target by sd 0.16
  points per 100; 83% of it is shot shape (spot, shot clock, possession start), which persists like something a
  defence controls (half to half r 0.77, next season 0.73) and is barely tied to team strength (r -0.09).  Game
  context is only 6% of it (sd 0.034 points; r -0.71 with team strength; prices strong defences slightly BETTER here,
  the opposite of the tuned logistic's problem) and has no detectable link to the rating changes.  At team level the
  quality piece predicts the other half's points allowed with weight 1.015 (se 0.167; the incumbent's mix gives it
  0.25) and the candidate's target predicts the other half (0.7280 against 0.7259) and the next season (0.5865
  against 0.5829) slightly better at every kept share; the best kept share is about 0.4 under both pricings.
- **The ratings loss is in the box-score prior's refit, not in the target's own games.**  96% of the loss is on
  defence and it is already in 62's raw output (99 and 91 do not cause it).  The candidate moves each player's
  defensive prior by sd 0.52 per 100 (players with 1,000+ possessions) but his own-games part by only 0.13; the
  change is 95% within team, does not carry to the same player's next season (r +0.03), and tracks no team context.
  The two candidates, whose targets are nearly identical (r 0.993), have priors that differ by sd 0.50 and ratings by
  0.43 -- as much as either differs from the incumbent.
- **Likely mechanism (in the code, not yet tested by a run):** `singleyear.stratified_player_folds` sorts players by
  LABEL and deals them in snake order into the five out-of-player folds, so any change to the labels re-deals the
  players among the boosters; each player's prior then comes from a booster trained on a different set of players.
  That makes every target experiment pay a re-deal noise the incumbent (fixed deal) does not.  One measured sample of
  that noise: L7 against LX, two re-deals on nearly the same target, differ by 0.048 on 63 (z -0.97), against the
  candidate's loss of 0.128 -- so the re-deal explains part of the loss, perhaps not all; the rest may be real or the
  incumbent's winner's-curse advantage from having been picked across earlier runs.
- **Tests proposed (one build each, about 95 min):** (1) the L7 arm with the player folds pinned to the incumbent's
  deal; (2) the incumbent target with a different, equally balanced deal (measures the re-deal noise and the
  incumbent's selection advantage, which bears on every target experiment so far).

**The two checks and the verdict (2026-10-07; the owner: "Sounds good" to rejecting 42 and holding the prior's
training groups fixed in target experiments).**  `62 --deal_seed=K` trains each defensive box-score prior alongside a
different, equally balanced set of players (seeded); `62 --deal_target_def=<name>` keeps the groups the named target's
labels make (both new 2026-10-07, default off; a default build still reproduces the incumbent exactly, 2015 max
difference 0.0).  63, both directions, against the incumbent's 8.5995:

| Run | What changed | Team-game | z | Seasons better | Stint z |
|---|---|---|---|---|---|
| 42 | quality pricing, training groups follow the new labels | +0.128 | +2.83 | 23 of 56 | +4.70 |
| C1 | nothing but the defensive training groups (seed 1; 80% of players move) | +0.056 | +1.15 | 25 of 56 | +1.32 |
| 42f | quality pricing, training groups held to the incumbent's (79% would have moved) | +0.043 | +1.29 | 23 of 56 | +2.22 |

- **Verdict: 42, 42x and 42f rejected.**  With the noise taken out the quality pricing of opponent threes is no gain
  (+0.043, z +1.3, no quality tier better).  It does move Ighodaro 18th to 46th in 2026 (the owner thinks he was too
  high); Caruso's rise to 14th in 42 was the training-group noise (C1 alone moves him 42nd to 14th; 42f leaves him
  50th).
- **The training-group noise is real and large:** one equally good regrouping of the incumbent costs 0.056 on the
  team-game criterion, and every experiment that changed the labels paid a draw of it while the incumbent kept its
  own.  **Standard from now on:** a target experiment runs 62 with `--deal_target_def=<the incumbent's defensive
  target>` (and the offence's equivalent if the offence target changes), so only the idea moves.  Earlier label
  experiments decided by less than about 0.06 may have been decided by this noise.
- **Next (HANDOFF):** average the box-score prior over several training groupings, which should remove this noise
  from the ratings themselves.

## Experiment 43: the box-score prior averaged over five training groupings (2026-10-07/08, awaiting the owner's call)

**What changed.**  `62 --prior_groupings=K` deals the out-of-player folds K times (grouping 0 the shipped snake deal,
grouping g `BalancedGroupKFold(seed=g)`) and gives each player the mean of his K out-of-player predictions, both sides;
the full-data booster is unchanged.  One grouping is bit-identical to before (`tests/test_prior_groupings.py`; a default
build reproduced the incumbent's 2015 rows to 0.0).  Run 43 = groupings 0-4; replicate 43b = groupings 5-9, none
shared.  Both through the standard chain (99 with the incumbent's multipliers, 91, the battery).  5K + 1 booster fits a
side: about 12-17 minutes a season, 6.5 hours for thirty.

**The noise it removes.**  One grouping moves a player's PRIOR by sd 0.13 per 100 on offence and 0.38 on defence (all
30 seasons; the offensive booster is already a bag of five, the defensive one is not).  In the finished ratings, players
with 1,000+ possessions: one grouping's draw is about sd 0.33 in total; 43 and 43b each move a player by sd 0.29 / 0.37
from the incumbent, and those two moves correlate 0.78 (theory for a shared grouping-0 draw: 0.82), so both averages
step away from the incumbent's own draw in the same direction.  The two averages still differ from each other by sd
0.23 in total (0.11 offence, 0.18 defence); one regrouping (C1) differs from the incumbent by 0.39.  2026 top 20: 18 of
20 the same in 43 and 43b, typical gap 0.18, largest Caruso 0.86.

**63, both directions (incumbent 8.5995):** 43 8.5970, -0.067, z -1.72, 31 of 56 (forward -0.100 z -1.99, back -0.034
z -0.56); 43b 8.5990, -0.015, z -0.33, 32 of 56; 43b against 43 +0.052, z +1.7 (C1 against the incumbent: +0.056).

**The robust review (the owner, 2026-10-08: "review our results in a not-brittle fashion ... systemic improvements
should be seen positive"; `scripts/135_robust_review.py`, outputs/robust_prior_avg5.csv, sheet tab "Robust review").**
Every instrument split into slices and the slices counted, beside the noise control C1:

| slices better than the incumbent | 43 | 43b | C1 (one regrouping) |
|---|---|---|---|
| prediction tests (63 team-game and stint, 88 groups, trade loss): 47 slices | 42 | 34 | 8 |
| 63 team-game, six era x direction cells sharing no games | 5 | 4 | 2 |
| trade loss defence, three decades | 3 | 3 | 0 |
| lineup-swap test, 12 slices | 5 | 3 | 8 |
| consensus rho (offence, defence, total) | 0 | 0 | 0 |

- Both replicates improve the prediction tests in most slices, every player-quality tier leaning better in 43 and four
  of six in 43b; the gain sits on defence (63 with only the defence swapped in: -0.049 / -0.022; offence only -0.018 /
  +0.008).  The 2007-2016 forward-looking cell is better in both (z -2.4, -2.0); 2007-2016 backward worse in both.
- The lineup-swap test is slightly worse in both: net order +0.019 / +0.021 (z +1.3 / +1.2), from defensive order (z
  +1.7 / +0.9); offensive order better in both (z -1.2 / -1.2).  Its 2017-2025 backward cell is worse in all three runs,
  C1 included, so the incumbent's own draw is unusually good there.
- Consensus total rho 0.822 / 0.824 against 0.834: the same drop as C1 (0.822), so it is the incumbent's draw that
  agrees unusually well, not the averaging (bootstrap z +3.1 / +2.5 / +2.6).
- Caveat as registered: 99's multipliers were fitted on single-grouping priors (2017-2026 folds); refit (97, then 99)
  only on adoption.  scale_def is 0.984 for both averages (the incumbent 1.005): the free prior scale grows on a less
  noisy prior, so defence is about 2% wider.

**Where the grouping noise comes from (2026-10-08, `scratch/2026-10-07_prior_groupings/seed_noise.py`, 2015).**  The
full-data booster refitted on IDENTICAL rows with random seeds 0-4 moves a player's prior by sd 0.068 (offence) and
0.271 (defence); five groupings at a fixed seed move it 0.131 and 0.401.  So about a quarter (offence) and half
(defence) of the grouping variance is the booster's own randomness (row and column subsampling, its early-stopping
split), which any single fit carries, leave-one-player-out included; the rest (sd about 0.11 / 0.30) is which players
are left out.  Averaging groupings redraws both parts at once, which is why it is the cheapest of the options compared
for the owner (one model; one grouping; averaged groupings; leave one player out, with and without a bag; twenty
folds; a bagged defensive booster; folds fixed by player id).

## Experiment 44: LightGBM as the prior's learner, then leave one player out (2026-10-08, awaiting the owner's call)

The owner, 2026-10-08: "go! (lightgbm mode)" -- ruling 5 ("keep chimeraboost") lifted for this experiment.
`eracoef.lgbprior`: deterministic LightGBM (no subsampling, a fixed tree count, one thread, weights normalised to
mean one); five refits on identical rows give identical predictions (chimeraboost's defensive prior moves sd 0.27).
`62 --booster_params=lgb1` (44, today's five player folds) and `--lopo=1` (44b, every rated player's prior from a
fit on every other player's rows; the six rows each player is asked about are answered inside the workers, since one
model can be 14 MB).  Settings from `scripts/136_tune_lightgbm.py`: 94's screen on 2026 and 2015, 150 trials a side;
both sides chose linear leaves, 250-300 rows a leaf (the top of the range), 31 bins (the bottom).

**The screen.**  Tuned LightGBM against the SHIPPED chimeraboost, every season and split: offence rescaled error
-1.5% (z -1.9 to -2.6), defence -2.0 to -2.6% (z -4.9 to -6.0).  But experiment 32's TUNED chimeraboost scores as
well or slightly better on the same 2026 folds (defence 1.423 / 1.417 against LightGBM's 1.426 / 1.425; offence 0.646
/ 0.649 against 0.647 / 0.644): the screen gain is tuning, not the learner.

**The test, both directions against the incumbent (8.5995), and each stage against the incumbent's same stage:**

| stage | 44, five folds | 44b, leave one out |
|---|---|---|
| prior alone (`63 --columns=prior`, raw tables) | -0.222, z -1.3 | -0.360, z -1.9 |
| raw ratings (steps 1-3) | -0.114, z -1.1 | -0.143, z -1.2 |
| after the prior shrink (99, the incumbent's multipliers) | +0.036, z +0.5 | -0.006, z -0.1 |
| final, after the swap adjustment (91) | **+0.169, z +2.2** (8.6055) | **+0.136, z +1.6** (8.6044) |
| stint level, final (rescaled) | z +7.2 (+7.0) | z +5.8 (+5.1) |

44b against 44: -0.033, z -0.5 (leave one out a little better at every stage).  Robust review (135, `robust_lgb`):
prediction slices better 4 of 47 (44) and 10 of 47 (44b), against 8 for one regrouping (C1) and 42 for experiment 43;
defensive with/without the one bright spot for 44b (8 of 9 slices, z -1.6); consensus 0.838 / 0.830 (from 0.834).
Ratings move sd 0.57 in total (1,000+ possessions).  2026 (44b): Zach Edey 39th -> 4th on 590 possessions, Caruso
42nd -> 6th, SGA 6th -> 17th, Jarrett Allen 16th -> 48th.  The defensive prior is far more extreme for some players:
Edey 2.17 (chimeraboost) -> 4.14 / 4.39, Caruso 0.77 -> 1.73 / 3.38.

**Reading.**  The LightGBM prior predicts the neighbouring seasons better than chimeraboost's, and leave one out is
better still, but the two later steps were fitted on chimeraboost's priors (99's multipliers on its within-season
folds; 91's strength and spread hold with the incumbent) and turn the gain into a loss; at the stint level the loss
is there from the raw stage on.  Untested, the likely reason for the extreme priors: linear leaves extrapolating on
unusual box scores, which the possession-weighted screen barely sees (a 590-possession row weighs little there).
Not adoptable as built.

**The rematch (2026-10-08; the owner: "fair rematch", "i like the lightgbm numbers better").**  Same raw tables; the
prior shrink refitted on LightGBM's own priors (62 `--booster_params=lgb1 --save_models=lgb1_within`, which rebuilt
44's 2017-2026 exactly, 0.0; 97 `--tag=lgb1_within --booster_params=lgb1`; 99 `--tag=lgb1_within`; leave one out
borrows the five-fold models' multipliers).  First, a default chimeraboost build still reproduces the incumbent's 2015
rows after the fold-prior refactor (0.0).  LightGBM's multipliers: offence 0.704-0.726 (chimeraboost's 0.704-0.728),
defence 0.882-0.922 (chimeraboost's 0.920-0.968) -- its defensive prior part holds up slightly WORSE in held-out games.
After the shrink: a tie (+0.009 / -0.038, z +0.1 / -0.4).  Final: 8.6052 (z +2.2) and 8.6039 (z +1.5), barely moved;
robust review (`robust_lgb_rs`) still worse in nearly every year-over-year slice.  Where it loses is WITHIN teams:
stint level worse from the raw stage on (rescaled z +5.0 / +2.6 raw, +4.7 / +2.4 after the shrink), the lineup-swap
test worse, the swap adjustment gains a quarter of what it gains on chimeraboost (-0.002 against -0.008), and the
ratings lean more on the team (one-way team share of the defensive rating 0.176-0.180 against 0.162).  Across teams
its prior is better (prior alone z -1.9).  Untested reading: the on-court plus-minus inputs (team context, experiment
39) carry more weight in LightGBM's prior.

## Experiment 45: the LightGBM prior without the on-court inputs (2026-10-08, awaiting the owner's call)

The owner: "try without the on court plus minus stuff for lightgbm".  Experiment 44's LightGBM (settings `lgb1`, five
player folds, the incumbent's prior-shrink multipliers) on `--features=boruta_noonc`: on-court plus-minus and on-court
possessions out (offence 21 -> 17 inputs, defence 17 -> 14).  `bash scripts/experiment_chain.sh lgb_noonc x3def_w0.25
"--booster_params=lgb1 --features=boruta_noonc"`.

- **63, both directions: 8.5897 against 8.5995, -0.245, z -2.38, 37 of 56 -- passes the rule.**  Stint level -1.14,
  z -9.1 (48 of 56); each side rescaled z -9.0 (the order, not the spread).
- Robust review (`robust_lgb_noonc`): prediction slices better 36 of 47 (15 past z -2), against 8 for one regrouping
  (C1), 42 for experiment 43 and 4 for experiment 44; year-over-year cells sharing no games 5 of 6; lineup-swap test 11
  of 12 better (6 past z -2), so the within-team split that sank 44 is fixed; ratings far less team-driven (one-way
  team share of the defensive rating 0.079, offence 0.094; incumbent 0.162 / 0.132).
- Weak spots: with/without on offence worse (8 of 9 slices, z +1.4 overall); consensus 0.865 / 0.788 / 0.818 (from
  0.851 / 0.816 / 0.834), top five 4 of 5 -- the same defensive-consensus drop experiment 19 saw when `onc_d` alone left
  the defensive list (then ruled: keep `onc_d`).
- Ratings move a lot: sd 0.81 in total (1,000+ possessions).  2026: Ajay Mitchell 9th -> 28th (-1.74), Hugo Gonzalez
  20th -> 113th (-2.07), Ighodaro 18th -> 69th, Giannis 2nd -> 5th; Cason Wallace 29th -> 11th (+1.69), Butler 15th ->
  8th, Jokic 4th -> 2nd, Gobert 43rd -> 20th, Edey 39th -> 17th; outside the top 20 Robert Williams 183rd -> 33rd,
  Jakucionis 33rd -> 202nd, McBride 49th -> 223rd.
- **Not yet attributed:** the learner and the inputs both differ from the incumbent.  Chimeraboost on the same inputs
  is the control (the old open item 5a, never run).
- **ADOPTED 2026-10-08** -- the owner, on the 2026 top 20: "oh hell yeah. this looks right. keep the one that likes
  butler/jokic/edey and drops hugo/ajay/oso down".  This reverses experiment 19's ruling to keep `onc_d`.  The adoption
  build (`scratch/2026-10-08_lightgbm/adopt45.sh`): models saved for 2017-2026, 97 on them (`--tag=lgb_noonc_within`),
  the test table with its own multipliers (`lgb_noonc_rs`), the product table (`season_ratings_product_lgb_noonc`).
  Publishing waits for the owner.

## The Robustness pass (2026-10-08, the owner's request)

The owner, on the calibration-step question at the top of HANDOFF.md: "research the most robust way to deal w/ these
biases.  we could do even more fanciful held-out stuff, and it's ok if that's where we land.  but ideally these biases
don't even exist in the first place."  Fourteen read-only agents researched it (code and data audits, the literature,
three designs, three adversarial reviews); the approved plan is to measure every input's lean on each season's own
held-out games, remove the causes inside the fit one build at a time, and correct after the fact only what survives.

**Rulings this session:**
- **Scope:** every input on both sides, the same method on offence and defence, and in groups of correlated inputs
  (average linkage on 1 - |rank correlation|), not single statistics -- "i dont think those chosen there are
  necessarily best. we should either do all or if too slow just find the ones with highest sum SHAP impact".  Group
  shares of the prior come from Owen values (shap's Partition masker); LightGBM 4.7.0 refuses SHAP on linear leaves.
- **A bug fix, or a fix that removes a lean the held-out audit counts, is adopted on a tie** (year-over-year z
  between -2 and +2), unless the robust review is systematically worse than the noise control, the consensus misses
  grossly, or the 2026 top 20 is worse.  Every other change keeps the standing rule.
- **Career counts are allowed as they are.**  Seasons played, career possessions and seasons with his team count
  every season before the rated one (`roles.career_inputs` at `roles.py:224`, called with the rated season at
  `scripts/49_role_panel.py:207`; its docstring was written for three-season blocks, where the count stopped three
  seasons back).  So in the year-over-year test a rating that looks back knows how much the player played in the
  season it is scored on: taking last season out of the counts moves the raw prior by sd 0.09-0.12 per 100 (2021,
  2026, saved models), on offence following last season's possessions (r -0.27 to -0.49).  A known look-back leak in
  the test, accepted.
- **Height and weight stay the middle value of all of a player's season listings** (`bio.py:42-57`, later seasons
  included; 5% of player-seasons differ from it by more than 10 lb).  The owner: "shouldn't matter much.  Your call";
  kept, because changing it costs a full build for a small effect.

**Two defects found, both checked in the code:**
1. **The prior never sees body weight.**  `prior_rows`, `season_rows`, `adjacent_rows` and `chunk_rows` write each
   training row's sample weight (the possessions behind its label) into the column `weight`
   (`singleyear.py:296, 334, 591, 716`), over the body-weight input that both `boruta_noonc` lists carry.  62 trains on
   it (`62:332`; values 0.4-133,082) and rates on the panel's body weight in pounds (`62:1111`; 133-360).  Feature
   selection kept `weight` through the same overwrite (`50_boruta.py:119`).  The fix is the Robustness pass's first
   experiment.
2. **The prior's scale is not cross-fitted for the shipped inputs.**  `--crossfit=scale` rebuilds only the on-court,
   off-court and RAPM-piece columns per fold (`62:495`); since experiment 45 the prior reads none of them, so the free
   scale is priced on box-score columns built from every game of the season, the held-out fold's included.  How much
   that inflates the scale is the subject of a check before any change.

### The held-out audit (138): the definition, frozen before any build (2026-10-08)

`scripts/138_heldout_audit.py` is the yardstick every Robustness-pass build is judged by.  **What it measures:** for
every held-out team-game of the within-season folds (97: a season rated from three quarters of its games, scored on the
fourth), the points per 100 as scored minus what the ratings predict -- each rated player's share of the possessions
times his rating as it ships (the prior part times 99's product-rule multiplier for that season, plus the games' part,
re-centred), stand-ins at their own values, the level and home edge free per fold -- is regressed on a correction per
player that depends on one input or one group's axis, entering the team-game the way a rating does.  The correction's
slope is how far the season's own games say players high on that input are underrated (+) or overrated (-).

- **Axis:** each fold's rebuilt input as a within-fold possession-weighted percentile and its normal score.  **Groups**
  (`params/audit_groups.json`, frozen): average linkage on 1 - |rank correlation| over the incumbent's whole-season rows
  with 500+ possessions (2017-2026), cut at 0.4, 0.5 and 0.6; 18 groups, the same on both sides (size: rebounds,
  defensive rebounds, height, weight, blocks, steals plus blocks; three-point shooting profile; scoring volume; free
  throws; playmaking; turnovers; shooting efficiency; three-point shooting; career length; draft and entry age; playing
  time with score state).  Steals group with nothing.  A group's axis is the first principal component of its members.
- **Statistics, per side:** a straight line per standard deviation of the axis; a bend (the square's coefficient); and the
  top tenth minus the bottom tenth (possession-weighted tenths), in points per 100.  Offence plus defence is read too.
- **The null and the threshold:** whole careers reassigned -- each player takes another player's inputs, all of them,
  season by season along the donor's career -- 400 times; each test's z is its estimate over its spread under that
  null, and the threshold is the 95th percentile of the largest |z| over all 64 decision tests' line and bend
  statistics on both sides (studentized max-T).  On 2017-2026: **3.64**.  The season jackknife would have needed 7.58,
  because a player's misfit carries over between seasons; it is printed but never decides.
- **Stages:** the prior part alone, before the prior shrink, after it (the decision stage); after the swap step once
  106 runs on the folds.  **Versions:** the scale pinned (primary), the two prior multipliers refitted with the lean, one
  free scale per side on the whole rating, and the pinned version plus playing-time terms (log possessions, the bench's
  share of the team-game) as information.
- **A lean counts** only if, after the shrink, its line or bend z passes the threshold, its tenths reading is at least
  0.1 per 100 bottom to top, its line keeps its sign and size (within a factor of two) under all three scale versions,
  and (with thirty seasons and the deadline folds) it keeps its sign in all three eras and the deadline folds do not
  contradict it at |z| 2.  Team context, playing time and score state are measured and never count.  A fix is credited
  with a lean only through its paired movement beyond the noise control; a lean that merely stops passing is never
  called removed.
- **Checks passed:** with only the two multipliers free the fit gives 99's 0.7130 / 0.9515; the readings match the design
  reviewers' independent re-measurement (defensive weight +1.26 against their +1.30, defensive rebounds +1.41 / +1.40,
  height +0.88 / +0.86, steals -0.37 / -0.36; offensive true shooting +0.68 / +0.69).

**The baseline, 2017-2026** (`outputs/csv/audit_lgb_noonc_2017_summary.csv`; points per 100 from the bottom tenth to the
top, + = underrated; z against the threshold 3.64):

| lean | offence | defence | offence + defence |
|---|---|---|---|
| size (rebounds, defensive rebounds, height, weight) | -0.51 (z -3.5) | **+1.83 (z +8.6)** | **+1.53 (z +6.4)** |
| body weight | -0.43 | **+1.26 (z +5.2)** | **+0.83 (z +3.8)** |
| height | -0.48 (z -3.4) | **+0.88 (z +4.7)** | +0.40 |
| steals | -0.43 (z -2.9) | -0.37 (z -1.8) | **-0.81 (z -3.8)** |
| blocks | -0.34 | +0.58 | +0.24 |
| turnovers per possession used | **-0.52 (z -3.8)** | -0.22 | -- |
| free-throw %, two-point %, effective FG % | **+0.75 / +0.64 / +0.58** | -- | -- |
| teams played for | -0.21 | **-0.53 (z -4.1)** | -- |

Read with care: the offensive shooting leans mostly vanish once playing time is held (true shooting +0.13 per standard
deviation -> -0.02), and before the prior shrink they point the other way (true shooting -0.70 raw, +0.68 shrunk): they
are the closed playing-time topic, not a price.  Steals do not count on either side alone, but on the total the
high-steal players are overrated by 0.81 per 100.  Big men are underrated on defence and, less, overrated on offence;
on the total they are still underrated.

### The prior's scale, priced in sample: measured (2026-10-09, `scratch/2026-10-08_robustness/scale_check.py`)

For 2017, 2021 and 2026 the scale was refitted with each of the ridge's five game folds' priors rebuilt honestly -- every
input (box rates and their padding, playing time, starts, shot totals, on- and off-court numbers) rebuilt from the
fold's training games through 97's World and asked of the same saved boosters -- against the shipped pricing (62's
fold builder, which rebuilds nothing the prior reads).  The as-shipped refit equals the shipped table's scales exactly.

| season | offence: shipped -> honest | defence: shipped -> honest | at penalty 3,000 (offence / defence) |
|---|---|---|---|
| 2017 | 2.673 -> 2.405 (-10%) | 1.090 -> 0.959 (-12%) | 2.791 -> 2.236 / 1.196 -> 0.980 |
| 2021 | 2.500 -> 2.408 (-4%) | 1.098 -> 0.986 (-10%) | 2.546 -> 2.172 / 1.099 -> 0.923 |
| 2026 | 2.008 -> 1.869 (-7%) | 1.358 -> 1.226 (-10%) | 2.057 -> 1.699 / 1.388 -> 1.144 |

The box score of the games a scale is regressed on reads part of their outcome back (a player's own points are part of
his team's points), so the shipped scale is 4-12% too large, about 10% on defence in every season, and the gap doubles
at a lighter residual penalty.  The plan's gate (more than 5% on either side in at least two of three seasons) passes in
three of three: the honest-scale build (the Robustness pass's step 6) is called for.  The prior shrink (x0.72 offence,
x0.95 defence) was fitted on folds that carry the same in-sample pricing, so part of what it corrects may be this.

### The held-out audit on thirty seasons (2026-10-09; `outputs/csv/audit_lgb_noonc_summary.csv`)

The same frozen audit on the incumbent's within-season folds for all thirty seasons (360 folds: 2017-2026 from the
adoption build, 1997-2016 built for this pass with the models saved and every rebuild equal to experiment 45's chunks,
maximum difference 0.0), with the trade-deadline folds (60) and the trade set on actual points as vetoes.  Threshold
3.63 (the season jackknife would have needed 6.95).  **39 leans count.**  Points per 100 from the bottom tenth to the
top, + = underrated:

| lean | offence | defence | offence + defence | eras (line per sd, 1997-2006 / 2007-2016 / 2017-2026) |
|---|---|---|---|---|
| size: rebounds, defensive rebounds, height, weight | **-0.59** | **+1.79 (z +11.6)** | **+1.20 (z +7.9)** | defence +0.31 / +0.59 / +0.54 |
| steals | -0.33 (z -3.5) | **-0.61 (z -5.4)** | **-0.95 (z -8.6)** | defence -0.20 / -0.25 / -0.08 |
| possessions he finishes, points (scoring volume) | +0.55 (playing time) | **-0.85 / -0.72** | -0.88 | defence -0.38 / -0.20 / -0.10 |
| three-point volume (attempts, makes, share of shots) | **+0.74 (z +5.9)** | -0.80 (made threes) | -- | offence +0.19 / +0.24 / +0.15 |
| blocks | **-0.64** | +0.45 | -- | offence -0.23 / -0.22 / -0.09 |
| turnovers per possession used | **-0.54** | -0.29 | -0.75 (turnovers) | offence -0.18 / -0.10 / -0.16 |
| free-throw % | **+0.78** | -- | -- | offence +0.24 / +0.17 / +0.18 |
| personal fouls | -0.28 | **+0.51** | -- | defence +0.30 / +0.05 / +0.09 |
| age entering the league | -- | **+0.49** | +0.79 | -- |

Not counting: true shooting (offence +0.59, but its size changes more than twofold across the scale versions and
playing time removes it), seasons played and career possessions (no straight-line lean once whole careers are
reassigned; the bend is modest and the offence eras disagree), assists.  Team context remains the largest lean and is
measured only: players with more playoff possessions are underrated (offence +0.96, defence +0.51), and the team's
results with him on and off the court lean up to z -13.9 on the season jackknife.  The three eras agree in sign on every
counting lean; the steal lean is weakest in 2017-2026.

### The noise control, the swap step on every fold, and the body-weight preview (2026-10-09)

- **The noise control** (`noise_regroup1`: the incumbent rebuilt with its training players regrouped on both sides,
  `62 --prior_grouping_first=1`, through the whole chain with its own folds): year over year 8.5902 against 8.5884
  (+0.04, z +0.6); held-out team-game error on each season's own games +0.013 (z +0.3, 5 of 10 seasons better).  The
  leans move by a median 0.023 per 100 from regrouping alone (90th percentile 0.051, largest 0.105, defensive body
  weight); the ratings move along an axis by a median 0.022 (90th percentile 0.057).  This is the yardstick: a fix is
  credited with a lean only through a paired movement beyond it.
- **The swap step on every fold** (106 with the LightGBM folders and tables): the shrunk table reproduces exactly and the
  swap to 0.005-0.013 per 100 (106's documented standardising change).  Its own lean, swapped minus shrunk with the
  same multipliers: it halves the scorers-on-defence lean (possessions he finishes -0.85 -> -0.37), trims size (defence
  +1.79 -> +1.57, offence -0.59 -> -0.31), leaves steals (defence -0.61 -> -0.61, offence -0.33 -> -0.42) and adds a
  playing-time lean on defence (+0.32).  The audit reads it as the stages `shrunk106` and `swap`.
- **The body-weight preview** (`scratch/2026-10-08_robustness/weight_check.py` on the noise control's dumped rows -- the
  incumbent's exact rows; the as-shipped refits reproduce the saved 2015, 2021 and 2026 priors exactly once the rows are
  indexed by player): restoring body weight moves the raw prior by sd 0.25 on offence and 0.38-0.48 on defence (the
  booster's scale, before the scale and shrink are refitted); heavy players' defensive priors improve by 0.22-0.41
  (heaviest tenth against lightest); and players with 2,500+ possessions fall against the bench by about 0.2-0.3 on
  offence and 0.3-0.4 on defence -- the overwritten column worked as a "how well measured is this label" input (trap 8),
  and the free scale (offence x2.0-2.7) may have been stretching back a prior it compressed.  Only the build can say how
  much survives the refitted scale and shrink.
- **The fix is in the code, not built:** `row_weight` everywhere the training weight was written or read (singleyear,
  62, stackprior, gbdt_prior, 93, 94, 71, 72, 85, tests); `singleyear._guard` refuses a bookkeeping column named like an
  input; `pytest`: 559 passed, 1 xfailed (2026-10-09).

### Each group's share of the prior (2026-10-09; `outputs/csv/audit_lgb_noonc_shares.csv`)

Shapley values with every group of correlated inputs as one player (the audit's 0.5-cut groups), against a background of
the season's rated rows, on the incumbent's saved prior models (`138 --shares=`; LightGBM gives no SHAP values for linear
leaves and shap's Partition explainer took 35 minutes a season and side).  A group's share is its mean |value| over the
rated rows over the sum; the mean of 2017, 2021 and 2026 (each season within a few points of it):

| offence | share | defence | share |
|---|---|---|---|
| scoring volume (possessions he finishes, shot attempts, points, missed twos) | 0.20 | the rated season's possessions | 0.21 |
| career length (seasons played, career possessions, age) | 0.18 | size (rebounds, defensive rebounds, height, blocks, weight, steals plus blocks) | 0.21 |
| playmaking (assists, assists minus turnovers, assists per possession used) | 0.13 | scoring volume | 0.19 |
| playing time and score state | 0.10 | playing time and score state | 0.16 |
| shot difficulty and offensive rebounding | 0.09 | career length | 0.11 |
| the rated season's possessions | 0.07 | steals | 0.09 |
| three-point volume | 0.05 | shooting efficiency | 0.03 |
| shooting efficiency | 0.05 | | |
| free throws, seasons with current team, steals, three-point % | 0.03 each | | |
| size (body weight is the only size input on offence, and it is broken) | 0.004 | | |

Scoring volume carries a fifth of the DEFENSIVE prior, which is the channel the audit's scorers-overrated-on-defence lean
(possessions he finishes -0.85) points at; steals carry 9% of it.

## Experiment 46: the body-weight fix (the Robustness pass, experiment 1; 2026-10-09, ADOPTED as the base, not published)

The owner: "Go".  The training weight moved from the column `weight` (where it overwrote the body-weight input) to
`row_weight`; the prior trains on body weight in pounds; everything else as the incumbent.  `FIRST_PASS=1 OWN_FOLDS=1
CONTROL_TAG=noise_regroup1_within bash scripts/experiment_chain.sh weightfix x3def_w0.25`, then the same without
FIRST_PASS; the robust review against the noise control.

- **First pass (2017-2026, its own folds, paired with the incumbent's on identical test games):** the defensive
  body-weight lean +1.26 -> +1.02 per 100 bottom to top (-0.24, z -5.2; noise control -0.11); the other size leans
  move within noise; held-out team-game error +0.056 (z +0.9; control +0.013).  Players under 1,000 possessions rise
  by about 0.3 per 100 on each side (root mean square of the move 0.83); the offensive possession-share lean grows
  +0.78 -> +1.00.  The overwritten column had been holding the bench down.
- **63, both directions: 8.5857 against 8.5894, -0.0997, z -2.15, 39 of 56** -- all of it in the rating looking back
  (-0.231, z -3.56, 23 of 28); the rating looking forward ties (+0.032, z +0.56).  On the incumbent's multipliers
  -0.089 (z -1.95).  Stint level -0.09 (z -1.2).  Every quality tier better on both directions together (z -1.9 to
  -2.6; top 30 a tie).
- Lineup-swap test: net order +0.033 (z +1.9), gaps -0.67 (z -5.2).  Trade loss: a tie (offence z -0.4, defence
  z +1.1).  Consensus 0.851 / 0.780 / 0.805 against 0.865 / 0.789 / 0.820 (noise control 0.864 / 0.790 / 0.819).
- Robust review against the noise control: year over year 8 slices better and 3 worse (control 2 and 9); quality
  tiers 6 and 0; lineup-swap test 10 and 2; consensus 0 and 3 (two clearly worse); trade loss as the control.
- 2026: Jokic 2nd -> 5th (-0.95), Wembanyama 1st -> 1st (-0.68), Giannis 5th -> 3rd (+0.60), Edey 17th -> 9th
  (+1.13), Jarrett Allen 19th -> 26th (-0.47); outside the top 20 Tyrese Maxey 31st -> 91st (-1.16).
- **Not the look-back leak:** with last season taken out of the career counts the fixed prior moves less than the
  incumbent's on offence (sd 0.07 against 0.09) and about as much on defence
  (`scratch/2026-10-08_robustness/leak_check.py`).  Why the gain is one-directional is not known.
- Under the owner's rule for bug fixes (adopt on a tie unless 63 is +2 or worse, the robust review is worse than the
  noise control, the consensus misses grossly or the 2026 top 20 is worse) it qualifies; the top 20 is the owner's call.
- **ADOPTED 2026-10-09 as the base of the Robustness pass, NOT published** (the owner: "go for it", after asking about
  the in-season reading and the biases).  The incumbent for scoring is now `outputs/season_ratings_weightfix.parquet`
  (8.5857), its multipliers from its own folds (`outputs/within/weightfix_within`, 2017-2026), its trade-loss alphas
  `tradeset_weightfix_alpha`; `experiment_chain.sh` defaults to it (`INC_NAME=weightfix`, `INC_FOLDS=weightfix_within`)
  and so does 135.  The site still shows experiment 45; publishing waits until the pass as a whole is better year over
  year AND in season with the counted biases smaller.

## Experiment 47: the honest scale (the Robustness pass, experiment 2; registered 2026-10-09, before any result)

The owner: "go for it".  `62 --crossfit=honest`: each of the ridge's five cross-fitting folds asks the boosters about
the season rebuilt from the fold's training games -- every input the prior reads, through `seasoninputs.SeasonWorld`
(97's season rebuild, moved to `src/eracoef/seasoninputs.py` unchanged; the 2017 shipped scale 2.673 / 1.090 and the
honest one 2.405 / 0.959 reproduce through it exactly) -- instead of 62's fold builder, which rebuilt only on-court
columns the prior no longer reads.  97 prices the scale the same way inside each of its folds when the saved models
say `crossfit: honest`.  Everything else as experiment 46 (the prior models refit identically: deterministic
LightGBM, the same rows).  Run: `FIRST_PASS=1 OWN_FOLDS=1 CONTROL_TAG=noise_regroup2_within bash
scripts/experiment_chain.sh honestscale x3def_w0.25 "--crossfit=honest"`, then the full chain on the first pass's gate.
**Expected:** the free scale 4-12% smaller, about 10% on defence; the refitted prior shrink nearer 1.  **Read:** in
season (the paired held-out error) and the biases (the audit, paired with experiment 46's folds and the new noise
control `noise_regroup2`, the incumbent regrouped on the fixed code), then year over year, the robust review, the
swap test, the trade loss, the consensus and the 2026 top 20.

**First pass (2017-2026, 2026-10-09).**  The honest path reproduces itself in 97 exactly (every season, maximum
difference 0.0; 2,640 checks, 0 failed), and the boosters are the incumbent's (the raw priors differ by a per-season
constant only).  The scale fell as expected: offence 14% (mean 2.22 against 2.58), defence 9% (1.00 against 1.11).
99 refitted on the candidate's own folds: offence 0.7064 -> 0.7535, defence 0.9553 -> 1.0257 -- so after the shrink
the prior carries 8% less weight on offence and 2% less on defence.  On each season's own held-out games:
- **before the shrink** the honest scale is better in all 10 seasons (team-game error -0.59; the noise control
  +0.11), and the scale the held-out games ask for is nearly 1 (offence 0.98, defence 1.07; the incumbent 0.89, 0.99);
- **after the shrink** (the decision stage) it is worse in all 10 seasons, by +0.097 (z +4.1; the noise control
  +0.070, z +1.5, 3 of 10 better).  The shrink, fitted on truly held-out games, already corrects the in-sample scale,
  and does it slightly better than pricing honestly at the source.
- The leans it moves beyond twice the noise control are all on offence and all one way: high-volume scorers more
  underrated (possessions he finishes +0.26 -> +0.45, shot attempts +0.55 -> +0.73, points +0.71 -> +0.85, bottom
  tenth to top); no lean the 2017 baseline counts is pushed further by 0.1 or more.  Players under 1,000 possessions
  move up (offence +0.16, defence +0.11 on average; root mean square 0.31, largest 0.75): a smaller scale lifts
  priors that sit below the mean.
Not stopped: the plan stops a first pass only when it moves nothing beyond the noise control, and a bug fix is judged
on the full chain (the tie rule), so the full chain runs.

**Full chain and verdict (2026-10-09): not adopted.**  Year over year, both directions: 8.5901 against 8.5857, +0.123
(z +3.4, 21 of 56 better) -- past the tie rule's veto of +2.  Looking forward +0.163 (z +3.2, 9 of 28), looking back
+0.084 (z +1.6, 12 of 28); per stint +0.296 (z +7.0).  On the incumbent's multipliers: +0.021 (z +0.5), per stint
+0.433 (z +8.3).  The scored seasons ask the offence to be widened by 1.13 (the incumbent 1.12): the honest scale
narrows it further, the wrong way.  Robust review (135, against noise_regroup2): worse in every year-over-year family
(team-game 1 better and 10 worse, 5 past z 2; the noise control 7 and 4, 1 past z 2), by quality tier 0 and 6, and in
the defensive trade loss 0 and 9; the lineup-swap test ties (net order +0.002, z +0.2); the consensus is unchanged
(total 0.807 against 0.805).  2026 top 20: the same players in nearly the same order, the stars a little lower
(Antetokounmpo -0.33, Edey -0.24, Jokic -0.20); Moussa Diabate (20th) leaves, Alex Caruso (21st) enters.
**What it settles:** the scale's in-sample pricing is real (defect 2 stands as measured), but it is not a defect of
the published ratings: the prior shrink (99) is fitted on truly held-out games and already pulls the oversized scale
back, and pricing the scale honestly at the source loses on both yardsticks once the shrink is refitted.  The honest
mode stays in the code (`62 --crossfit=honest`, `seasoninputs.honest_fold_prior`, 97 prices its folds the same way
when the saved models say so); the shipped default is unchanged (`--crossfit=scale`).  Logged as experiment 47.
