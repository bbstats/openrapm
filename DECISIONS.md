# OpenRAPM: what the out-of-season test decided, and what it could not

## What the test is

Hold out one NBA season H. Fit the whole system on a symmetric neighbourhood of seasons around it —
K = 2 is {H-1, H+1}, K = 3 is {H-2, H-1, H+1}, K = 4 is {H-2..H+2} minus H — so the held-out season
is identical across every method and every K, comparisons are exactly paired, and aging cancels.
Then predict every stint of H from the ten players on the floor, refitting only an intercept and a
home term on H (level "home", never "full": context that is a consequence of the score is not an
input). Score possession-weighted squared error in points per 100, at stint level and again after
summing each team's points within a game. The owner ruled on 2026-09-05 that **game level is the whole
test** — ratings remove about 10% of team-game error and 0.7% of stint error, and stint error is
mostly binomial noise — but the stint column is always reported, because the two have disagreed.
Baseline with no player ratings: 125.6 per 100 at game level. This is the only criterion the project
has that is both external to the model and legal to select on: it scores actual points in a season the
model never saw, and uses no outside data. `src/eracoef/holdout.py`, `scripts/45_holdout.py`, stop
rules in `scripts/48_ladder.py`, the mapped and timed version in `scripts/54_track.py`. A second
instrument, `investigate.attributable` (`scripts/57_investigate.py`), reads the same residual for what
a player ridge can still put on named players: the criterion decides forecasting, the investigator
decides attribution, and a candidate should lose neither.

## What was settled, and how

**Three-season windows, not five.** Ten three-season windows against six five-season ones over the
same thirty seasons, game-grouped folds inside each. Held-out weighted MSE 3624.5 against 3624.9, the
box prior worth 30.1 bp against 24.5, rating stability between adjacent windows 0.714 against 0.676.
Five years buys 15% tighter coefficient standard errors, which is mechanical, and blends too much
career change into one player effect. Later superseded in season by the rolling kernel; the owner's
standing note is that chunks are on the way out, so do not tune K.

**No defensive box prior (the hybrid).** The box term is an offset `Xbox @ beta` with separate
offensive and defensive columns, so zeroing beta's defensive half IS "no defensive prior" — one fit,
one penalty. Consensus agreement went 0.784 to 0.896 total, 0.755 to 0.888 on defense, defensive
spread 2.13 to 1.23, archetype bias +0.63 to +0.20. The mechanism: weighted R-squared of a player's 13
box rates on his own on-court RAPM is 0.53 on offense and 0.26 on defense, and the stats the defensive
prior was built from are the most conserved into the lineup sum (defensive rebounds 0.59, blocks 0.69,
against steals 0.90 and missed twos 0.88). **A lineup-level coefficient transfers to individuals
exactly to the extent the stat is not conserved.**

**The free-throw luck adjustment (shipped).** Each free throw is priced at the shooter's padded
leave-one-out FT%. Worth **+0.26 per 100 against raw points, mapped** (the unmapped figure was 0.459;
see the map-absorption trap). It is the only luck adjustment that has ever paid here, and the reason
is the padding constant: FT% pads at k = 24 attempts, so a shooter's own rate is trusted almost
entirely on 30 attempts, the outcome belongs to one identified player, and the defence is absent.

**The opponent-three-point adjustment, `x3def` (shipped).** For the DEFENSIVE fit only, every
opponent three-point make becomes 3 x the shooter's 3P% from the OTHER half of the block, padded
toward the league with k = 450 attempts. Team opponent 3P% stabilises only at 4,000-7,000 attempts
against about 2,000 in a season. Worth -0.39 / -0.47 per 100 at z -4.4 / -5.1; removing it altogether
costs +0.248 at z 3.25 mapped, and handing the defence back up to a quarter of its realised threes is
free (-0.007, z -0.34). Never leave-one-game-out — a leave-one-out mean is negatively correlated with
the left-out outcome by 1/(N-1). It cost consensus defensive agreement 0.888 -> 0.814, and the
component split explains that: every raw-points metric in the blend agrees less, every luck-adjusted
one more.

**Shot quality in the prior.** Six features from the per-shooter shot tables: difficulty (`xl / a`,
the league's make probability from his spots) and shot-making (`(m - xl) / a`) on twos, threes and in
points, each padded in attempts toward the BLOCK's own league level, never a constant (the league
two-point rate runs 0.4648 in 2000-02 to 0.5468 in 2024-26). Worth -0.045 on the accuracy line
(z -1.13) and nothing in shipping shape — it shipped for the floors, moving the offensive bigness gap
-0.303 to -0.270 (which unlocked the 0.7 offensive blend) and the defensive spread to 1.30, the
narrowest any candidate had measured. Later pruned by Boruta at no cost.

**The calibration map and its exposure term.** A per-side map fitted leave-one-season-out on the
pooled TEAM-GAME residuals (`src/eracoef/calmap.py`, `scripts/53_calmap.py`). A map of the RATING
ALONE is the identity at game level — the scalar the games want is 1.03 / 0.98 / 0.96 on offense at
K = 2/3/4, and every shape family lands within 0.03. **The exposure term is what pays**: `sat(poss) =
poss / (poss + 1000)`, worth **-0.79 to -1.01 per 100 (z -5.7 to -7.1, 25 of 28 seasons)**. At K = 3
the all-seasons fit is offense `0.85 x + 2.88 sat`, defense `0.885 x - 3.15 sat`: a rotation player is
about 5 points per 100 above a player the block never saw, and the term is flat among regulars, so it
reorders the low-minute end and leaves the top alone. The shipped family is `linear+log2&xlog` plus a
prior re-weighting term on OFFENSE only (on defense it takes consensus agreement to 0.751). Two
further terms are prediction-time only and cannot ship: a quadratic in the player's age at H (-0.14)
and a cubic bend on the team-game total and on the stint rows (-0.07 and -0.21). The earlier
stint-fitted rank map (-0.41 per 100, z -6.2) was superseded by this and is off.

**The boosted GBDT prior beats the linear one.** `mspi` scored 112.06 at K = 4 against 112.33 for the
linear-prior board, 20 of 28 seasons. Its prior is a third narrower and the on-court residual carries
half again as much of the rating (1997-99 offense: prior sd 1.15 against a residual of 0.67, where the
linear prior was 1.69 against 0.40 and correlated 0.98 with the final rating). That is the Stockton
problem answered: Jordan first by 0.5 and Stockton seventh, where the linear board had Stockton 5.18
over Jordan 5.27. The era term is in the model and does almost nothing — moving an assist from its
10th to its 90th percentile is worth 0.14 per 100 in 1998 and 0.16 in 2025.

**The role prior chain: APM -> Simple SPM -> RAPM_1 -> GBDT.** APM is the plug-in ridge with beta 0 at
penalty 100 (under 1% of the shipped 18,352); the Simple SPM is a possession-weighted ridge of APM on
possession share, starts share and age with squares, fitted leave-window-out (role prior sd 1.4-1.9
per 100 on offense, 0.7-0.9 on defense); RAPM_1 is the shipped ridge with the SPM as its offset; the
GBDT is trained on that. **What the prior is trained ON is the only thing about it that has ever
mattered**: RAPM_1 -> unshrunk APM is -0.29 per 100 at z -3.1, while capacity, twenty-three engineered
features and a distance-weighted target all move the criterion by less than 0.07. The pure APM prior
fails the consensus floors (defense 0.678, spread 1.56), so the shipped offensive target is
**0.7 APM + 0.3 RAPM_1** with RAPM_1 on defense. `scripts/49_role_panel.py` builds the panel.

**The estimator search.** `scripts/55_tune.py`, 625 TPE trials in 97 minutes, scored on 14 alternating
held-out seasons with the other 14 never shown to the optimizer. `tune501_b7` ships: 110.624 against
110.707, **z -3.02 over 18 of 28, and 37 s for the 28 fits against 46**. Four independent candidates
converged on the same structure — linear leaves on both sides, cross features on offense only, no
defensive bag, 64-128 bins, subsample 0.65-0.83 — an independent rediscovery of what a hand
decomposition had found. The best search trial (491, -0.194) regressed to -0.060 on the confirm half
while the eventual winner ranked 8th on search: the region is real, the ranking inside it is noise.

**The turnover-aware prior, evaluated at a settled context.** Teammate turnover is measured for
everyone (`src/eracoef/turnover.py`; movers sit at a season-to-season median of 1.0, stayers at 0.36).
The prior is trained on ordered window pairs with the target window's turnover as a feature and then
evaluated at a settled 0.35 for everyone, fixed a priori. **-0.054 at z -3.39 over 22 of 28, offense
only.** Why: the pooled target is a player's value over windows three years away, where turnover
averages 0.70, so the context-change penalty is baked in; the held-out season sits inside its own
block at 0.40.

**Binned height and weight.** Zero on the criterion (-0.006, z -0.32) and shipped on the owner's call,
because the prior's own fit puts height at -0.14 on defense and the archetype table shows what it
corrects: **the defensive prior under-rates 6-10-and-taller players by about a third of a point per
100 in nine windows of ten, with no era trend at all.** The criterion's resolution for a correction
shared by an archetype is about 0.015, so it cannot adjudicate this. Binned, never fine (trap five).

**His own past on-court record in the prior.** `past_apm` (his possession-weighted APM over the panel
windows BEFORE this one, discounted 0.5 per window), `past_poss`, `past_rapm`, on the offensive list,
on PAIR rows because a pooled row's target contains the past windows (`training_rows` raises on any
PAST name). **-0.085 at z -2.46 on the criterion and -0.204 at z -4.32 on the investigator: the first
candidate ever significant on both instruments.** The gain is in the body of the distribution, not the
top — the top offensive decile is still under-rated by +0.64 against +0.70 before.

**The Boruta prune.** 111 candidates on pair rows, 50 trials, 5.7 hours (`scripts/50_boruta.py
--modes=sink,sinknoagg`). Boruta's own accepted lists LOSE badly (+0.199 on the criterion at z 4.8 and
+0.574 on the investigator at z 9.1 for the offensive one). But removing its REJECTS from the shipped
lists costs nothing on either instrument: 48 offensive names to 31 and 25 defensive to 11, at -0.024
(z -0.54) and -0.044 (z -0.89), 58 s for the 28 fits against 65. Shipped on the parsimony tie-break.
The defensive prior is back to eleven box columns, the season and two role inputs.

**The in-season kernel.** A rating is ANCHORED at a season and fit on that season and the two before
it, `kernel = {0: 1, -1: 0.5, -2: 0.25}`, nothing after the anchor in it, at half the block board's
ridge. Chosen on the search half and replicated on the confirm half: **-0.225 per 100 (z -3.0) at a
quarter of a season and -0.205 (z -2.4) at half**, with attribution -0.363 (z -2.9). The optimum is a
plateau with three neighbours inside 0.11, so the simpler member wins: one halving per season. The
chunk board is not close in season — +2.7 to +6.2 per 100 on prediction and +4.2 to +7.3 on
attribution — and most of that is coverage, 0.716 against 0.988, because a block that ended before the
season began has no rating for a quarter of the players on the floor. Against a stratified permutation
null the season board has captured **84% of the player credit a lineup model can see**, the flat
rolling window 80%, the chunk board 45%.

**The season board ships beside the block board, which is untouched.** `scripts/60_season_board.py`
fits the kernel once per anchor season 1997-2026 (66 s for all thirty) and `scripts/52_site.py` writes
both with a Block/Season switch. It is the same board, slightly narrower: Spearman 0.969 on 580 shared
players, sd 2.52 against 2.93.

**Two knobs measured and left alone.** The offensive and defensive penalties were already separate and
already essentially optimal: the best cell of a 4x5 sweep is 110.047 against 110.049 for the single
multiplier. And on a single season the ridge does NOT collapse onto the box score — every target's
optimum is interior at x0.5, where a rating is 20% on-court evidence on offense and 44% on defense.

## What was tried and rejected

**The LRBoost branch (a boosted correction on a frozen linear prior).** Five things had to be right
before it did anything: pooling across all ten windows (shrinkage 0.26/0.03 within one window, 0.87/
0.59 pooled), weighting by the shrinkage diagonal, stripping the 8% of the linear span the plug-in fit
leaves in the residual, freezing playing time the way the scorer does (this alone dropped the
offensive permutation null from 0.42 to 0.00), and keeping playing time on defense and off offense. It
then won held-out stint MSE by +0.278 ± 0.090 (3.1 SE) and the next-window criterion by +0.0129. On
the ladder against the consensus it is 0 on the total and -0.020 on offense. Off — but the offset
machinery stays, because a correction must be carried INTO the fit, never added afterwards.

**The xPTS stage-4 shot term, and the shooter-level version after it.** The full geometric closure
(condition on the possession's first attempt, marginalise everything downstream with the lineup's four
cross-fitted factor rates) was built and it CLOSES: expected attempts within 0.4% of observed on all
112 training blocks, points ratio 0.994-0.999, no clipping. It loses at every `c_def`, both K, both
variants: +0.32 at game level (z +1.8, 9 of 28) and +1.02 at stint level. A diagnostic with league
make rates instead of lineup ones loses by +1.96, so shot-making is about 15% of everything the
ratings know and the lineup eFG fit recovers only 80% of it. The shooter-level rebuild (his own padded
other-half 2P%/3P% times a per-season shot-location curve) loses by more, +1.5 to +2.4 per 100, and
the padding constants say why: k = 175 attempts for 2P%, 226 for 3P%, against 24 for FT%. A heavily
padded shooter rate IS mostly the league rate. The location curve itself is worth 0.7 per 100 and is
kept.

**The four-factor defensive fit.** Opponent eFG%, turnovers forced, offensive rebounds allowed and
free-throw rate allowed, each solved on the points fit's own layout with its own ridge, the points
prior shared out by zero-prior slopes and recombined with the row-level gradients (eFG 1.55, TOV
-1.04, OREB 0.63, FTR 0.32 points per 100 per point of rate, R-squared 0.88). Twelve forms measured;
the best is -0.040 at z -0.94 and 7x slower, and re-run on the current board it is +0.005 / -0.010,
exactly nothing. What was learned is where the value is: **per-factor shrinkage WITHOUT a prior is
worth -0.11 (z -1.44); the fixed-share prior split costs more than that**, because a share puts a big's
prior into rebounding-points and a guard's into foul-points the same way for everyone. Per-factor
PRIORS were never built. Splitting eFG by zone was killed by measurement before it was built: REML's
four ratios match an independent split-half estimate (eFG 1.50 against 1.50), so the constants were
already right.

**The team-game leave-one-out luck adjustment — it works at team level and fails at player level.**
Every rate a possession's points depend on now has a measured skill share on each side, nothing is a
chosen constant, and the adjustment was tested directly: first half of a team's season predicts its
second, 1,784 team-seasons, no ratings pipeline in the way, scored so that generic shrinkage cannot
win. **It removes 6.6% of the forward error on offense and 10.4% on defense**, and three-point
shooting is almost the whole of it (-0.357 and -0.568). Shrinking a STYLE rate (rebound chance per
attempt, field goals per attempt) COSTS 0.09-0.17 at z 3-4. In the ratings, none of it survives: the
free-throw rung at team level recovers 0% of the shipped shooter-level gain (k = 297 attempts against
24, so it prices a 90% shooter and a 60% shooter identically), the two-point rung costs +1.63 mapped
because a defence controls 0.85 of two-point percentage, and the three-point rung — the one place the
premise holds — wins the search half at -0.170 (z -2.8) and gives back two thirds of it on the confirm
half (-0.055, z -0.90) while being significantly WORSE at stint level there. `src/eracoef/teamloo.py`
is kept as a self-contained single-season instrument.

**The Dredge play-by-play counter block.** Twenty-one event types per player-season out of the raw
play-by-play, twenty-six padded features, validated to 0.0000-0.0024 against the box score and to
0.00e+00 between the panel and prediction paths. **Seven candidates, best z -0.20, worst z 2.33.** The
mechanism is reliability: year-over-year, `blk` per 100 scores 0.918 and the Russell SHARE — Dredge's
headline decomposition — scores **0.126**. It is 0.575 for everybody and 98% of its spread is sampling
noise. Dredge's strong claim, that the split should REPLACE `blk`, reverses: dropping `blk` costs the
defensive prior +0.328 weighted MSE, the largest effect of any single column here, and its
decomposition recovers 87% of that. A decomposition is worth having when the model cannot make it, and
a depth-4-to-7 booster over 23 to 43 crossed features can. The assist-by-zone extension (`pot_ast`,
year-over-year 0.922, the most reliable feature in the block, -0.018 on the prior's own offensive fit)
reads +0.0005 at z 0.12 on the criterion. `OffFoulsDrawn100`, Dredge's best published coefficient, is
not buildable from the v3 feed and is untested.

**Schedule context.** Declined by the owner — *"evens out really well."* Independently, removing the
shooting team's own leave-one-out three-point rate from every team-game before estimating moves the
measured defensive three-point spread from 0.59 to 0.58 points, so schedule is not driving that
either.

**The playoff delta, as anything other than a separate view.** A rating can be the prior for another
rating: the regular-season number becomes the OFFSET for a fit that sees only playoff possessions, so
that fit's residual IS the delta and no playoff box score is needed. Penalty chosen on five blocks and
read on the other five: x2 the regular-season penalty, **-0.459 per 100 at z -2.16 on prediction and
-0.508 at z -2.27 on attribution**, surviving a per-side slope refit, so it is ranking and not
amplitude. But the delta's spread is 0.21 against 2.43 for the ratings — a playoff run moves a player
a tenth of the distance between players — so it changes no board and ships only as its own view. Two
bugs in shared code had to be fixed first: the exposure filtered to `phase == "RS"` always, so a
playoff-only design had zero exposure; and the unpenalized fixed block goes singular on a one-phase
design (`is_po` constant, `po_home` a copy of `home`), returning solves around 1e12.

**Career-experience features.** Seasons played, career possessions and entry age, counted before the
block's first season. **-0.077 weighted MSE on the prior's own leave-window-out fit — the largest gain
any block ever posted there, and -0.247 on the low-exposure rows — and +0.054 on the criterion at
z +1.65, 11 of 28.** See the fifth trap below; this is its purest instance.

**Also rejected, with the number.** Squared lineup-sum terms for diminishing returns: unanimous in
sample (eight of the top ten 2024-26 terms sign-reversed), 0.3-0.4 bp worse out of sample, rank
correlation 0.998. Single-season training panels for the prior: +0.52 at z 4.4, because a prior
trained on one-season feature lines and applied to three-season ones is attenuated and the defensive
prior loses a third to a HALF of its spread. Destination/"traded-to" features as the board's prior:
-0.072 on the criterion and +0.125 on the investigator — kept as the trade-to instrument. The
per-player trade delta applied to H: +0.069. On-court ORTG/DRTG in the prior: +0.05 at z 2.22. A
recursive prior-informed RAPM: converges worse (0.786 -> 0.765), because refitting beta on `Rbeta + u`
launders shrunk residual into unshrunk prior.

## The measurement traps

**Benchmarking a model against another model built from the same data.** The board's tilt toward big
men was certified as "evidence, not bias" by regressing the shipped rating on the same player's NEXT
window pure on-court RAPM: the bigness term came out **+0.045 at z +4.7**, and the conclusion was
published. The benchmark was built from the same margin data with the same blind spot and could not
catch a shared error. Against an external blend the same board had Robert Williams 25th to the
consensus's 134th, Jusuf Nurkic 21st to 166th, Trae Young 313th to 79th. *The check:* score against
something that does not share the estimate's construction — actual points in a season the model never
saw, and an outside metric read once.

**Calibration on a lineup sum is blind to attribution.** A stint row observes only the lineup SUM of
player effects, so reallocating credit between teammates barely moves it. Swept over the defensive
prior weight `c_def`, held-out stint MSE **prefers the wrong end at 3.4 standard errors while moving
0.065% in absolute terms**, choosing 1.0 over the 0.0 that raises player rank agreement from 0.78 to
0.89. Letting the stint regression estimate its own trust in the prior returns 1.12 on offense and
1.06 on defense — trust the box score MORE — and moves Robert Williams from 25th to 12th. *The check:*
ask which subspace a tunable moves. If it moves the design's null space, no in-sample loss can select
it; measure it externally or state that you are imposing a belief.

**An argmax on a grid boundary has chosen nothing.** The four-factor OREB fit first selected
`lam_ratio = 2.0`, the top of the configured grid; widening it moved the answer to 3.0, interior. The
same failure produced a whole ladder row's defensive knobs, where the internal criterion was flat
across the four weakest penalties (0.4789 to 0.4813) and then declined monotonically — no interior
optimum, so it picked whichever weak-end grid point won a coin flip. *The check:* verify the argmax is
interior with both neighbours worse, and say so. The in-season kernel and the single-season penalty
were both accepted on exactly this evidence.

**A target's gates cannot tell you whether it should be the target.** The xPTS shot term closed to
within 0.4% on all 112 training blocks with a points ratio of 0.994-0.999 and no clipping, and lost by
up to +1.7 per 100. The team leave-one-out target at zero shrinkage returns actual points per team-game
**to 1e-13** while changing every stint of that game, because it has spread the game's makes over its
attempts — so its gates pass everywhere and it erases which LINEUP did the shooting before any
shrinkage happens. *The check:* a gate that aggregates cannot see what the target did to the rows the
ridge actually reads. Only out-of-season prediction against ACTUAL points decides a target.

**A feature can predict how well-MEASURED a row is rather than how good the player is.** The prior's
target is a player's value pooled over his OTHER windows, so a player with more windows has a
lower-noise target and is intrinsically easier to predict. Career experience names exactly those
players: -0.077 offline, the best block ever measured there, and +0.054 on the criterion. Fine height
and weight are the purest form — the pair names a player almost uniquely, is -0.27 on the offensive
prior's own fit, and costs +0.20 on the criterion at z 4.7; binned, it keeps 0.124 of 0.145 on defense
(physiology) and 0.031 of 0.272 on offense (identification). Boruta, selecting against that same
target, ACCEPTED the career block. *The check:* bin any static player attribute and read the
binned-versus-fine gap as the identification share; splitting by exposure does NOT catch it, since the
low-exposure stratum liked experience more.

**A knob measured at a cheap operating point does not transfer to the shipped one.** A Huber loss is
-0.039 weighted MSE at the cheap booster and **+0.061 at `quality=4`, the booster that ships** — the
bag and the model-selection search already buy what the robust loss was buying. The same applies to
feature sets and to the base list: the play-by-play block read -0.050 on the 43-name offensive line at
`quality=4`, which would have been the largest feature gain ever measured here, and -0.012 on the
shipped 23-name defensive list with the shipped defensive booster. Same features, same data, opposite
conclusion. Parts interact in both directions: cross features are harmful alone (+0.004) and the best
thing available beside linear leaves (-0.013 together). *The check:* bench a candidate at the operating
point it would actually join, booster and base list both.

**An unmapped criterion gain can be entirely absorbed by the shipped calibration map.** The offensive
three-point target read **-0.21 per 100 unmapped and -0.024 at z -0.63 once the map was fitted**,
because the whole of it was offensive amplitude (`scale_off` moved 1.06 to 1.15) and a per-side
calibration curve is exactly what corrects amplitude. The audit of the pass quantified the absorption:
99% of the offensive three-point target's effect, 44% of the free-throw target's, 29% of the two-point
disaster's, 33% of the on-court prior's. *The check:* run `54_track.py` (which fits the shipping map
leave-one-season-out) and pair the systems before quoting any number as a gain, never `45_holdout.py`
alone. Near-total absorption is itself diagnostic: the candidate was only rescaling a side.

**The team-level and player-level luck tests are allowed to disagree, and must both be run.** Shrinking
the outcome rates removes 6.6% and 10.4% of a team's forward error; handing a defence back its real
three-point signal in the ratings is worse at every step up to full replacement (+0.248 at z 3.25 at
the end). A defence's 0.59 points of true three-point spread is a TEAM property, and the ridge's job is
to split it among five players against 1.05 points of season noise. The signal exists and does not
survive attribution. *The check:* a luck adjustment needs the team-level forward test AND the
out-of-season criterion; neither substitutes for the other, and only the second decides a rating.

**An identical-paths check proves the two paths agree, never that either is right.** The play-by-play
feature builder read the block's league totals from the frame's FIRST row, so with all ten windows
handed to it at once every row was padded toward 1997-1999 — the exact era contamination the block was
designed against. The comparison script reported 0.00e+00 throughout because it calls the same function
on both sides. Every bench number moved by 0.005 to 0.02 when it was fixed. *The check:* anything with
a per-block constant needs a test that puts two eras in ONE frame.

**A covariate measured on the held-out season needs a control from data the outcome could not have
touched.** A player's share of his team's possessions AT H was the largest single map gain ever
measured (-0.19 on the best line, z -3.7); the same share measured on HALF of H's games is worth
-0.006. It kept 3% of its value, so the whole of it was knowledge of who was good that season. The
team-game's mean log stint length reads -0.149 and its leave-one-game-out cousin +0.008: substitutions
follow the score. Teammate turnover survived the same control (-0.084 against an identical -0.084) and
is real. *The check:* the other half of the season, the team's other games, or the training block. Two
of three candidates died on it in one session.

**A search-half gain at z -2.8 can be two thirds noise.** The team leave-one-out three-point target
read -0.170 (z -2.8, 11 of 14) on the search half and -0.055 (z -0.90) on the confirm half, while being
significantly WORSE at stint level there. *The check:* the 14/14 alternating-seasons protocol — choose
on the search half, read on the confirm half, and read the stint column even when game level decides.

**Two more worth carrying.** A player-level split-half correlation rejects a defensive residual and
never selects its ridge: tightening the factor ridges 16-32x brought the factor sum level with the
points residual player by player and cost +0.50 and +0.83 on the criterion, because the criterion
values the residual's team-coherent content, which a correlation across players cannot see. And Boruta
cannot gate a deliberately collinear list: given `stocks` = `stl + blk`, a shadow copy of `blk` is as
good as `blk`, so `blk` fails its own test — accepted on offense and rejected on defense on the same
panel, and swapping the two costs +0.050 directly.

## Standing rules

**Accuracy wins, provided the testing is robust — and the model has to stay shippable as open source.**
The owner, 2026-09-06: *"The speed thing is more just like I don't want us to build some insanely
complex model that is overfit and too slow, because I want this to be open source! but ultimately
accuracy is the winner, provided the testing is very robust."* So fit every row and score every game;
fit time is a number the chart carries, not a term to optimise, and a real accuracy gain is never
traded away for seconds. "Robust" is the precondition, not a nicety: a gain that survives only the
search half, or only the prior's own fit, or only one map, is not a gain.

**Between two candidates the criterion cannot separate, take the simpler and faster one, every time.**
Complexity and fit time are a tie-break, and a strong one — somebody else has to be able to run this.
That ruling shipped the Boruta prune (42 names against 73 at -0.024, z -0.54), the simpler in-season
kernel from a plateau of four, offense-only PAST over the both-sides form, and the incumbent board over
every candidate inside noise.

**The consensus is a sanity check, never a fitting target.** The owner, 2026-09-06: *"disagreeing with
consensus is just a sanity check, never something to fully fit to."* The floors in
`tests/test_vs_consensus.py` say the same of themselves — they *"guard against a further fall, not the
old level."* So: never choose a model constant because it clears a floor (a defensive pooling constant
that scored -0.067 was deliberately NOT read against the floors, because choosing the value that just
clears a gate is fitting the gate). **A marginal miss is not a veto** — 0.7592 against a 0.76 floor is
noise on a sanity check and does not get to overturn a z -3.02 result; report the number honestly and
decide on the criterion. **A gross miss still is a veto**, because that is what the check is for: an
8% overshoot on the archetype gap is the thing the consensus exists to catch. When a floor is re-based
it is done once, deliberately, with the reason written into the test itself — and the model constant is
not touched. That distinction is the whole ruling: the criterion decides and the consensus
sanity-checks.
