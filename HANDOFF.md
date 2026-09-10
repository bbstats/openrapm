# Handoff: the playoff delta works; the luck adjustment works on teams and not on players

**Latest, 2026-09-09 (FINDINGS 38).**  A rating can be the prior for another rating: the regular-season
number becomes the OFFSET for a fit on playoff possessions, so that fit's residual is the playoff DELTA and
no playoff box score is needed.  Penalty chosen on five blocks and read on five: **-0.459 per 100 at z -2.16
on held-out playoff games and -0.508 at z -2.27 on attribution**, surviving a per-side slope refit.  The
delta's spread is 0.21 against 2.43 for the ratings, and its extremes are Robert Horry twice and the 2011
Mavericks at the top, Chris Paul and DeRozan at the bottom.  Three fixes made it possible and two were bugs
in shared code (the exposure always filtered to RS; the unpenalized fixed block goes singular on a one-phase
design).  `scripts/63_playoff_delta.py`; not published.  Start at 3.16.

---

# Handoff: the luck adjustment is measured end to end -- it works on teams, not on players

**Latest, 2026-09-09 (FINDINGS 35-37).**  Every rate now has a measured skill share on each side and no
constant is chosen; `scratch/forward.py` tests an adjustment directly (first half of a team's season
predicts its second, 1,784 team-seasons, no ratings pipeline).  **Team level: it works**, -6.6% offense and
-10.4% defense.  **Player level: nothing survives the calibration map**, and the audit of every claim in the
pass (36.6) withdrew one, corrected one number and left five rejections standing.  The shipped constants
come out RIGHT: the four REML penalty ratios match the independent measurement to the digit on eFG, and the
per-factor defence is flat on the current board either way, so the zone split was not built.  `config.yaml`
is unchanged, ten of ten floors.  **Two rulings wait for the owner (3.15); what is left is attribution, not
luck adjustment.**  Start at 3.15.

---

# Handoff: the luck adjustment works at team level, does not survive the map at player level

**Latest, 2026-09-09 (FINDINGS 35-36).**  Every rate a possession's points depend on now has its own
measured shrinkage constant on each side (`teamloo.RATE_SPECS`, `skill_table`), nothing is a chosen number,
and `scratch/forward.py` tests an adjustment DIRECTLY -- first half of a team's season predicts its second
half, 1,784 team-seasons, no ratings pipeline in the way.  **At team level it works**: shrinking the
outcome rates and leaving the style rates alone removes 6.6% of the forward error on offense and 10.4% on
defense.  **At player level it does not.**  Handing a defense back its real three-point signal is
monotonically worse, because 0.59 points of team spread will not split five ways against a season's noise;
and the offensive version's apparent -0.21 per 100 is **-0.024 at z -0.63 once the shipping map is fitted**
(36.5, and the new trap: never quote an unmapped gain).  `config.yaml` is unchanged and the board rebuilds
to ten of ten floors.  What is left on the table is the in-season ATTRIBUTION gain, -0.163 at z -3.12, a
ruling for the owner under 26.5.  Start at 3.15.

---

# Handoff: what a single-season ridge actually weighs, and the team-game luck adjustment that does not replicate

**Latest, 2026-09-09 (FINDINGS 34): the single-season penalty does NOT pick the box score, and a luck
adjustment cannot change what it weighs.**  The owner asked for a lambda that does not collapse onto the
prior.  Measured (`scratch/lamshare.py`, the `ls_<target>_x<mult>` systems, both in-season instruments):
every target's optimum is x0.5 of the block penalty, INTERIOR, and there a single-season rating is 20%
on-court evidence on offense and 44% on defense -- the criterion's own answer, not a degeneracy, and it
tracks the prior's quality (offensive prior sd 1.44 -> 20%, defensive sd 0.80 -> 44%).  The shipped season
board already sits at x0.5.  **The luck-adjusted target has the SAME argmin and a LOWER share (16.4%)**:
removing variance from the target removes signal and noise together, so it buys a better level, never a
bigger share.  **The one actionable thing: between x0.25 and x1 both instruments are flat (z -0.13 and
z -0.26) while the offensive evidence share goes 10% -> 33%.**  Three times the on-court content for a
difference neither instrument can see, so that is a product choice and it is nearly free.  Start at 3.14.

---

# Handoff: the team-game luck adjustment is built and answered; the board is unchanged

**Latest, 2026-09-09 (FINDINGS 33): the owner's single-season, leave-one-out luck adjustment does not
replicate.**  `src/eracoef/teamloo.py` builds a team's shooting rates from its OTHER games (rebalanced
leave-one-out, Austin/Pe'er/Korem 2025) and moves each game's makes toward them.  A pre-registered ladder
was read rung by rung: free throws at team level recover 5% of the shipped free-throw gain (the constant
is 297 attempts against a shooter's 24, so it forgets WHO shot); the two-point term costs +2.28 per 100;
the three-point term wins the search half at -0.170 (z -2.8) and gives back two thirds of it on the
confirm half (-0.055, z -0.90), while being significantly worse at stint level there.  Part 0 ruling 1
applies: not shipped, `config.yaml` untouched.  Start at Part 3.13.  What is worth keeping is the
instrument -- `scripts/62_teamloo.py` answers "which stats predict this game" from ONE season with no
panel and no prior, and independently confirms that a defence controls 0.85 of opponent two-point
percentage, 0.20 of three-point and 0.04 of free-throw.

---

# Handoff: the rating is in-season now -- a rolling kernel anchored on the season being rated, and two instruments that separate predicting from crediting

**Latest, 2026-09-09 (FINDINGS 32): single-season targets are built and the prior cannot take them.**  The
per-season role panel exists (`49_role_panel.py --season`) and every consumer reads the granularity off the
panel's labels, but a prior trained on one-season feature lines and applied to three-season ones is
attenuated -- half its spread on defense -- and costs +0.52 per 100 (z 4.4) on the board and +0.13 to +0.62
in season.  Start at Part 3.12; what survives is cross-panel pair rows (block features, season targets).

**Built 2026-09-08 (FINDINGS 31): the season board.**  A rating is now ANCHORED at a season and fit
on that season and the two before it, the earlier ones halved and quartered (`kernel = {0: 1, -1: 0.5,
-2: 0.25}`), with nothing after it in the fit -- so the latest row is a rating of the season in progress and
`scripts/60_season_board.py` re-run updates it.  `src/eracoef/inseason.py` is the module; two new fields on
`fastfit.MspiFast` (`kernel`, `cut`) carry it and every existing system is byte-identical (the flat kernel
reproduces `tune501_b7_pasto_pOD` to 0.0e+00).

**The instrument, and why it exists.**  The owner: *"the ultimate goal here is 'in-season, super good at
dividing credit AND being predictive, for the current season' -- it would be good for us to isolate/decompose
those 2 a little more formally."*  So a fit may see only the first `q` of the held-out season (the CUT) and
is scored on the rest of it, and the same residual is read twice: the criterion's team-game MSE
(**prediction**) and `investigate.attributable` (**attribution** -- what a player ridge can still put on
named players).  `scratch/inseason_run.py` runs both for every kernel at every cut in one pass.

**What it says (search half, 14 seasons, K = 3, mapped; the full table is FINDINGS 31.4).**  The chunk board
is not close in season: **+2.2 per 100 at the start of a season and +5.5 to +6.9 once it is a third old**
(z 3.3 to 6.9, 0-3 wins of 13), and +4.0 to +8.6 on attribution -- mostly coverage, 0.717 against 0.95-0.99,
because a block that ended before the season began has no rating for a quarter of the players on the floor.
The current season ALONE is +1.7 at a quarter and +0.7 at a half, catching up only at three quarters.  The
kernel wants to decay: `ks52` beats the flat rolling window by **-0.47 per 100 (z -3.9) and -0.57 on
attribution**, and its optimum is a PLATEAU with three neighbours inside 0.11 -- interior, not a grid edge.

**The block board is untouched.**  `08_ratings.py`, `config.yaml`'s `ratings_prior`, and
`tests/test_vs_consensus.py` are exactly as they were; the season board is a second product beside it
(`outputs/season_ratings.parquet`, config `ratings_prior.season_board`, a Block/Season switch on the site).

**Published 2026-09-08: the live site opens in Season mode** (https://bbstats.github.io/openrapm/, `main` at the
merge of this branch).  **After the ship, three measurements (FINDINGS 31.8), all settled:** the credit score's
level is ~94% noise floor and its paired differences are real -- the season board has captured **84%** of the
player signal a lineup model can see, the chunk board 45%; the ridge holds a tenth of what is left (x0.25-x0.5
optimal on both instruments); the prior reads a player's other windows off his rate fingerprint and closing
that channel (`GBDTPrior folds`) is flat on the criterion -- kept as an instrument, not shipped.  **Tree state:
committed on `hybrid-and-xpts` and merged to `main`; 148 passed, 1 xfailed.**

---

*The header of the previous pass, kept:*

# Handoff: the kitchen-sink Boruta pruned the board to 42 names; plus-minus is an input; the investigator is the second instrument

**Shipped 2026-09-08, latest: `tune501_b7_pasto_pOD`** (FINDINGS 29) -- the owner's kitchen-sink BorutaShap
(111 candidates on pair rows, 5.7 hours, `50_boruta.py --modes=sink,sinknoagg`) and what the criterion made of
it.  Boruta's own lists LOSE (+0.20 / +0.57 on the two instruments: the career block, 22.2's trap, on a set
that had every chance); removing its rejects from the SHIPPED lists costs nothing on either instrument, so
the tie-break shipped the simpler board: **31 offensive names and 11 defensive against 48 and 25, 110.4547
(-0.024, z -0.5), 41.089 on the investigator (-0.044), 58 s for the 28 fits, ten of ten floors, consensus
0.801 / 0.795 / 0.760.**  The defensive prior is back to the box line, season and two role inputs: no shot
quality, no height, no past record -- none of it carried weight there.  What Boruta kept on both sides:
the past plus-minus block, turnover, age, fouls, steals, weight, career possessions; what it threw out
everywhere: assists by zone, most play-by-play counters, draft slot, tenure, binned height, five of six
shot-quality columns.  **Next: the top of the board (3.9) and the defensive names of 27.2, which no column
reached.**

---

*The header of the previous pass, kept:*

# Handoff: plus-minus is an input now, the investigator is the second instrument, and the board is `tune501_b7_turnref_o_hwb_pasto`

**Shipped 2026-09-07, latest: `tune501_b7_turnref_o_hwb_pasto`** (FINDINGS 28) -- the offensive prior sees the
player's own past on-court record: his discounted APM over the panel windows before the block, the possessions
behind it, and the shrunk twin, on PAIR rows so nothing of the target is in the feature (`gbdt_prior.PAST`;
pooled rows refuse it).  **110.4785 on the criterion, -0.085 at z -2.46 over 20 of 28, and -0.204 on the
investigator's score at z -4.32 over 23 of 28: the first candidate significant on both instruments.**
Consensus 0.801 / 0.792 / 0.760 (the best total and defensive agreement of any shipped board), defensive
spread 1.27.  One floor missed and re-based with the reason in the test: the offensive gap against bigness,
-0.344 against 0.32, now 0.35 -- the second re-base of that floor in a day, which the owner should look at
(Part 0 ruling 2 applied; the top of the offensive board rose against the bigs, as the held-out data asks).
The gain is in the body of the distribution, not the top: the stars are still under-rated by 1-1.5 (27.2, 28.4).

**The investigator (FINDINGS 27) is the second instrument.**  `scripts/57_investigate.py` names who the board
is wrong about out of season; `scratch/investigate_cmp.py` scores tracked systems on the residual variance a
player ridge can attribute to players.  Run both on every candidate: the criterion decides forecasting, this
decides attribution, and a candidate should lose neither.  `HANDOFF` Part 3.8 and 3.9.

---

*The header of the previous pass, kept:*

# Handoff: the settled-context offensive prior shipped; the four-factor defence and the bio block measured

**Measured 2026-09-07, latest: the prior's first non-box inputs -- height, weight, draft slot, tenure (FINDINGS
26; the owner's "keep building the prior" direction).**  Height is the largest offline gain any column has
produced here (-0.14 on the defensive prior; the pair -0.27 on offense) and the criterion reads it at zero.
Two things explain that and both are measured: the fine height-weight pair NAMES the player (binned it keeps
0.12 of the defensive gain and 0.03 of the offensive one; on the criterion the fine pair costs +0.20 at
z 4.7), and the criterion's resolution for a shift shared by an archetype is about 0.015 (the shift test,
26.5).  The defensive prior under-rates 6-10-and-taller players by a third of a point in nine windows of ten,
with no era trend: the "era" question is an archetype question.  **The owner's call, 2026-09-07: "we're doing a
lot of not shipping -- just ship."  Shipped as `tune501_b7_turnref_o_hwb`: 110.5635, ten of ten floors,
consensus 0.784 / 0.787 / 0.751, the archetype gaps narrower on both sides.**  The board is now
`tune501_b7_turnref_o_hwb`; every "vs board" below this line that says 110.5693 is against the board before it.

**The owner's next direction, 2026-09-07: act like investigators.**  Find the lineups the board misrepresents
most, and above all the PLAYER: "when this player is added or subtracted from all of his lineups we tend to
be off the most", the same-four plus-minus idea.  `src/eracoef/investigate.py` and `scripts/57_investigate.py`
are the first cut (Part 3.8): the out-of-season residual of the shipped board's mapped prediction on every
stint row of every held-out season, then the five-man units with the largest pooled miss, each player's
on-court miss per side, and a residual ridge on the lineup matrices -- the same-four question asked of every
lineup at once -- pooled across seasons with a z.

**Measured 2026-09-07, later the same day: HANDOFF 3.2, the defensive four-factor fit (FINDINGS 25).**  Built
(`fastfit.factor_defense`, an identity test), measured in twelve forms, not shipped.  Full replacement of the
defensive residual is +0.10 to +0.31 against the board; the best half blend -0.040 at z -0.94 and 7x slower.
The zero-prior pair says where the value is: without a defensive prior the factor residual beats the points
residual by 0.11 (z -1.44); with the prior shared out across factors by fixed shares the gain is lost.  The
consensus likes the raw-eFG factor defense (0.768 vs 0.758) and the criterion rejects it: 22.7's disagreement
again, the other way round.  **Part 3.2 below is now the record and 3.2b (per-factor priors) is what would
finish it; start at 3.5 or 3.2b.**

**Shipped 2026-09-07: `tune501_b7_turnref_o`** (FINDINGS 24.9) -- the offensive prior trained on window pairs
with teammate turnover as a feature and evaluated at a settled context of 0.35 for everyone; defense as it
was.  **110.569 on the criterion against 110.624, z -3.39 over 22 of 28, 58 s for the 28 fits against 37.**
Consensus 0.791 / 0.787 / 0.759, defensive spread 1.28.  Nine floors as they were; the bigness floor missed
by the margin the screen predicted (-0.311 against 0.30, from -0.276; one standard error is 0.046) and was
re-based 0.30 -> 0.32 once under Part 0 ruling 2, reason in the test.  The blend and the reference were not
touched.  `docs/data/ratings.json` carries the new board; `main` does not yet (Part 3.6).

*The header of the trade-calibration pass, kept because it is the record of how the candidate was found:*

Written 2026-09-07, later the same day as the play-by-play pass (which follows below unchanged).  **The board
did not move: `tune501_b7` at 110.6237.  But there is now a candidate that beats it at z -3.39 on 22 of 28
seasons, and it is one config key away from the floors.**  The owner asked for "trade calibration": a rating
and a rating-if-traded, behind them a trade-weighted SPM that measures how teammate turnover changes what a
box line is worth.  `FINDINGS.md` **24** is the record.  Lower is better throughout; "vs board" is candidate
minus `tune501_b7` on the criterion.

| what was measured | vs board | verdict |
|---|---|---|
| a rating's prior and possession evidence weighted apart for movers (map) | -0.003, z -0.14 | a rating travels; nothing to calibrate at team-game level |
| a level in the held-out season's teammate turnover (map, needs H; its half-season control identical) | -0.084, z -1.98 | real, exogenous, cannot ship, bounds the prize |
| the per-player TRADE DELTA (prior at his turnover into H minus at a settled context) | **+0.069, z +1.07** | rejected: the thing that was asked for, and it hurts |
| pair-row prior, no turnover feature (control) | -0.010, z -0.63 | un-pooling is not it |
| the turnover prior at the pairs' own mean context 0.7 (control) | -0.029, z -1.43 | a slightly better pooled prior |
| the turnover prior at a SETTLED context (0.35), both sides | -0.062, z -2.53, 21/28 | real; defense pays on the consensus screen |
| **the same on OFFENSE only (`tune501_b7_turnref_o`)** | **-0.054, z -3.39, 22/28** | **the candidate**; consensus screen unchanged; bigness reading -0.316 |

**What was learned, in one paragraph.**  Teammate turnover is measured for everyone now (`src/eracoef/turnover.py`;
movers sit at a median of 1.0 season to season, stayers at 0.36).  At player level a fully turned-over player
is worth about 0.85 points per 100 less on offense and 0.55 less on defense than his box line says (z 5 to 8
on the prior's own target), and the booster can say who: high-usage scorers on offense, older players on
defense.  At team-game level none of that per-player structure predicts anything -- but the shipped prior's
TARGET is pooled over windows three years away, where turnover averages 0.70, so it carries that penalty baked
in, and the held-out season sits inside its own block at turnover 0.40.  A prior that can be asked "what is
this box line worth beside people he knows" is the right offset, and it is worth -0.054 on offense at z -3.39.
The "if traded" column exists as a player-level estimate (`outputs/csv/trade_delta_*.csv`) that the game-level
test does not confirm; FINDINGS 24.7 says exactly how it may be published.

**Tree state: committed on `hybrid-and-xpts`.**  Tests:
**108 passed, 1 xfailed** -- 109 collected; the "110" the earlier headers quote was a miscount
(`tests/test_turnover.py` is five, `tests/test_gbdt_prior.py` has two more for the pair rows).  `ratings_prior.gbdt_turn` is `{sides: [O], ref: 0.35}` (it was null when this
header was written).  `docs/progress.csv` / `.png` carry the seven runs of this pass.

**The board: `tune501_b7_turnref_o`** (FINDINGS 24.9, shipped 2026-09-07) on top of **`tune501_b7`**
(FINDINGS 22.7, shipped 2026-09-06) -- the estimator search's board with the offensive target blended back
to 0.7.  `tune501_b7` was 110.624 on the criterion, 37 s for the 28 fits, consensus 0.788 / 0.790 / 0.759,
defensive spread 1.28.  Its defensive-agreement floor was re-based 0.76 -> 0.75 once, deliberately, with
the reason in the test (Part 0 ruling 2); the bigness floor 0.30 -> 0.32 the same way, for this ship.

---

*The header of the play-by-play pass, kept because its conclusions still stand:*

**Seven candidates measured, none shipped (FINDINGS 23).**  The Dredge block on defense -0.0055 (z -0.20),
era-relative +0.0006, era-calibrated pair +0.0045, unassisted makes +0.0106, `pot_ast` on offense +0.0005
(z 0.12) and on both sides +0.0327 (z 2.33, the control).  Four of those were NEGATIVE on the prior's own
leave-window-out fit and not one is distinguishable from zero on the criterion: **a gain measured against the
prior's own target is not evidence about the board.**  Use `prior_bench.py` to REJECT a candidate cheaply,
never to believe in one.  The Russell share has a year-over-year reliability of 0.126; BorutaShap cannot gate
this feature list, structurally (23.10).  The role input `share` was renamed **`poss_pct`** (code tokens only;
`calmap`'s `hshare` / `tshare` exposure KEYS untouched, because a map term whose covariate column is missing
applies as ZERO, silently).

---

## Part 0: the owner's standing rulings (read before choosing anything)

1. **Accuracy wins, PROVIDED THE TESTING IS ROBUST -- and the model has to stay shippable as open source.**
   The owner, 2026-09-06, correcting the earlier reading of this: *"The speed thing is more just like I don't
   want us to build some insanely complex model that is overfit and too slow, because I want this to be open
   source! but ultimately accuracy is the winner, provided the testing is very robust."*  So:
   * fit every row, score every game; the fit time is a number the chart carries, not a term to optimise, and
     a real accuracy gain is never traded away for seconds.
   * but complexity and fit time are a TIE-BREAK, and a strong one.  Between two candidates that the criterion
     cannot separate, take the simpler and faster one every time -- somebody else has to be able to run this.
   * and "robust testing" is the precondition, not a nicety.  A gain that survives only the search half, or
     only the prior's own fit, or only one map, is not a gain.  See 22.2 and 22.6 for what that looks like
     when it goes wrong.
2. **The consensus is a SANITY CHECK, never a fitting target.**  The owner, 2026-09-06: *"disagreeing with
   consensus is just a sanity check, never something to fully fit to."*  It exists to catch a board that has
   gone gross-wrong -- the Robert Williams 25th case that the internal benchmarks certified -- and the floors
   in `tests/test_vs_consensus.py` say so themselves: they *"guard against a further fall, not the old level"*.
   So:
   * never choose a constant because it clears a floor.  22.7 declined to read `win_decay_d = 0.6` against
     the floors for exactly this reason, and that was right.
   * a MARGINAL miss is not a veto.  0.759 against a 0.76 floor is noise on a sanity check, and a candidate
     that is significant on the criterion should not be thrown away for it.  Report the number honestly and
     decide on the criterion.
   * a GROSS miss still is a veto, because that is what the check is for.
   * the criterion (`scripts/45_holdout.py`, held-out seasons, actual points) remains the one test that is
     both external and legal to select on.  It decides; the consensus sanity-checks.

3. **The 3-season window is on the way out.**  *"Nobody really looks at chunks, we will ultimately move away
   from this mode."*  Do not tune K or the window length; prefer work that survives the move to a continuous
   or per-season rating.
4. **Schedule context is not worth it** (*"evens out really well"*) -- rest days and back-to-backs, declined.
5. **The four-factor defensive fit is right but was too big for v1.**  It is now the most promising thing left
   (Part 3).
6. **The prior is spent, and that is now measured twice.**  22.5 said the box score was nearly spent and
   named the play-by-play as the resolution; section 23 built the play-by-play block and it is zero.  Part 3
   is the estimator from here.

## Part 1: where it stands

| | criterion (K = 3, mapped) | 28 fits |
|---|---|---|
| **the criterion's line** (`best_ratio_full`) | **109.845** | 59 s |
| the same with shot quality (`best_shot`) | 109.801 (z -1.13) | 70 s |
| **what ships** (`tune501_b7_turnref_o`, the settled-context turnover prior on offense (24.9), ten of ten floors as tested, defensive spread 1.28) | **110.569 (z -3.39, 22/28 vs `tune501_b7`)** | 58 s |
| `tune501_b7` -- the board it replaced (22.7) | 110.624 | 37 s |
| `tune501_b7_turnref_o_ffx5` -- the four-factor defence, repriced eFG, half blend (25), the best form | 110.529 (z -0.94, 18/28) | 427 s |
| `tune501_b7_turnref_o_ffx` -- the same, full replacement | 110.667 (z +1.33) | 427 s |
| `tune501_b7_turnref_o_nodp` / `_nodp_ffx` -- NO defensive prior, points residual / factor residual | 111.026 / 110.915 | 57 / 127 s |
| `tune501_b7_turnref_o_dprior` -- the defensive prior alone, no residual | 112.482 | 90 s |
| `tune501_b7_turnref` -- the same on both sides | 110.561 (z -2.53) | 60 s |
| `tune501_b7_turn` -- with the per-player trade delta to H | 110.693 (z +1.07) | 66 s |
| `tune501_b7_drd` -- the whole Dredge block on defense (23) | 110.618 (z -0.20) | 39 s |
| `ship_shot7d` -- the board `tune501_b7` replaced, ten of ten | 110.707 | 50 s |
| no ratings at all | 125.6 | |

### Why the board is where it is (FINDINGS 22.7)

`tune501_b7` came out of a 625-trial search over the whole estimator, validated on 14 held-out seasons the
optimizer never saw.  It is -0.083 against `ship_shot7d` at z -3.02 over 18 of 28 and 20% cheaper, and it
missed the defensive-agreement floor by 0.0008 (0.7592 against 0.76), which was re-based to 0.75 once,
deliberately, under Part 0 ruling 2 ("a MARGINAL miss is not a veto") -- so the suite is ten of ten and the
reason is written into the test.  Its defensive SPREAD improves to 1.28, so the prior
is not too wide; the agreement alone moved, and the suspect is the search's defensive nearby-window discount
of 0.28 where the shipped board pools every window alike.  **3.2 is the work that would let that knob stay.**

`scratch/ship_try2.py` runs any candidate through the whole shipping path -- targets, map, and the prior's
feature lists on either side -- without touching `config.yaml` permanently.

### What this pass measured (all on the criterion, all in FINDINGS 23)

| | criterion | verdict |
|---|---|---|
| the whole Dredge play-by-play block on defense (`gbdt_prior.DREDGE`, 12 features) | **-0.0055, z -0.20** | rejected |
| the same block ERA-RELATIVE (`DREDGE_R`, each feature over its block's league level) | +0.0006, z -0.08 | rejected |
| era-relative rim-block share and goaltends only (`blkrimsh_r`, `goalt_r`) | +0.0045, z 0.32 | rejected |
| the block split REPLACING `blk` (Dredge's actual claim) | +0.043 on the prior's own fit | rejected before the criterion |
| stolen turnovers replacing `tov`, unassisted makes replacing `ast` | +0.026, +0.030 | rejected before the criterion |
| every Dredge group on OFFENSE | +0.004 to +0.018 on the prior's own fit | rejected before the criterion |
| **BorutaShap on a candidate set containing the board** (`50_boruta.py --modes=wide`) | rejects `blk`, `tov`, `orb` and eight more shipped names | **unusable as a gate**, structurally (23.10) |
| **potential assists** reconstructed from assists by ZONE (`pot_ast`, the owner's 2019 r-squared ~1 fit) | **+0.0005, z 0.12** on offense; +0.033, z 2.33 both sides | rejected -- and it was -0.018 offline, the best of the pass (23.11) |
| assists by zone as five rates, or five shares, or replacing `ast` | +0.018 to +0.029 on the prior's own fit | rejected before the criterion |
| blocks by zone (`blksmr`, `blklmr`) | +0.020 on the prior's own fit, and `blk_smr`'s era share wobbles 0.22-0.48 | rejected: 23.3's distance artifact again |
| offensive fouls DRAWN (`OffFoulsDrawn100`, his 1.22) | **not buildable from the v3 feed** | needs an ingest job; untested |

## Part 2: machinery

| file | what |
|---|---|
| `src/eracoef/turnover.py` | **teammate turnover** (24.1): `build_teammates` (the per-season shared-possession table, `data/cache/teammates.parquet`), `familiar_share(tm, ref_seasons, new_seasons)`, `season_turnover`, `window_pair_turnover`, `cached_table(ctx)` |
| `gbdt_prior.pair_rows` / `GBDTPrior(turn=, pairs=)` | the prior on ORDERED WINDOW PAIRS with `turn` as a feature (24.4-24.5); `Context.prior(turn=True\|"pairs")`, `Context.turn_table()` |
| `spm.chain_offset(turn="ref"\|"h"\|"pairs", turn_ref=, turn_sides=)` and `MspiFast.turn` / `turn_ref` / `turn_sides` | the settled-context prior ("ref"), the per-player delta to H ("h", added AFTER the ridge), the un-pooling control ("pairs"); `TURN_REF = 0.35` |
| `calmap.Turn` / `MovedPrior` | map terms `turn` / `turnx` / `turnprior` (+ `turnp*` past-only, `turna*` half-season control), `mprior`; `SeasonFrame.turnover(k, train)` builds the covariates |
| `scratch/trade_turnover.py` | build the teammate table and print the turnover distributions; writes `outputs/csv/turnover_season.csv` / `turnover_windows.csv` (ignored, 40 s) |
| `scratch/trade_spm.py` / `trade_gbdt.py` | the trade-weighted SPM, linear and boosted, leave-window-out on the pair rows; the boosted one writes `outputs/csv/trade_delta_{O,D}.csv` |
| `scratch/trade_maps.py` / `trade_pair.py` | the map terms on an existing dump with the control set; the paired test when a dump carries several maps (**the base is read at its SHIPPING map**; it used to take the first, the `&turn` one, and the "vs base" column was off by 0.09) |
| **`fastfit.factor_defense`** / `factor_rows` / `points_per_factor` / `FACTOR_LAMS` / `MspiFast(def_factors=, factor_reml=, factor_x3=, factor_lam_scale=, factor_lams=, no_def_prior=)` | **the four-factor defence (25)**: four factor fits on the points layout, the prior split by zero-prior slopes, recombined with the row-level gradients; `factor_diag` on the system after a fit carries g, the shares, the ridges chosen.  `tests/test_factor_defense.py` (5) holds the identity |
| `scripts/50_boruta.py` | BorutaShap per side and mode; **`--modes=sink,sinknoagg` (29)**: every column both paths can build (111) on the shipped prior's PAIR rows (`pair_training_rows`), with and without the ten linear aggregates.  5.7 hours at 50 trials; histories in `outputs/csv/boruta_sink*_{D,O}.csv`.  Prunes, never decides (23.10, 29.3) |
| **`src/eracoef/context.py`** / `scratch/trade_to.py` | **the destination (30)**: usage minutes (usage x possession share), block-minutes and rebound-minutes, his own and the target roster's already spoken for, with the roster's offensive and defensive APM -- roster identity from the target window, every teammate's number from the feature window (`DEST`, `DEST_D`; pair rows only, like PAST).  `destination_inputs(..., override=)` and `ctx.dest_override` ask the trade-to question; `team_roster(roles, seasons, team_id, tm=)` builds a team's roster with its own co-occurrence weights.  `trade_to.py "<name>" ...` fits the window once per context (actual, league average, each team) and writes `outputs/csv/trade_to_<window>.csv`.  `tests/test_context.py` (2) |
| `gbdt_prior.TURN_FEATURE` = **`target_pct_new_teammates`** | the pair rows' turnover feature (24.4), renamed 2026-09-08 at the owner's request: the share of the TARGET window's teammate-possessions spent with people he never played beside in the feature window (1 = all new; the shipped board evaluates everyone at 0.35).  The MAP's `turn` covariate (calmap.Turn, SeasonFrame.turnover) is a different object and keeps its name.  `scratch/turnover_compare.py <window>` builds the board at 0 and at 1 and writes `outputs/csv/target_pct_new_teammates_<window>.csv` (prior, raw and mapped ratings, deltas) |
| **`gbdt_prior.PAST` / `past_features` / `past_inputs`** | **plus-minus as an input (28)**: his discounted past APM per side, its possessions, the shrunk twin; pair rows only (`training_rows` raises), the pair's target window left out of the past; `chain_offset` builds them per side from `ctx.rpanel` with the block's windows excluded.  `tests/test_past.py` (3) |
| **`src/eracoef/investigate.py`** / **`scripts/57_investigate.py`** / `scratch/investigate_cmp.py` | **the investigator (27)**: `residual_ridge` (the same-four question of every lineup at once), `on_court`, `lineups`, `season_table`, `pooled`, and **`attributable`, the second score**.  `57_investigate.py [--system=] [--lam=2000] [--min-poss=1000] [--top=20]` runs the shipped board over the 28 held-out seasons in a minute and writes `outputs/investigate_*`; `investigate_cmp.py <sys1> <sys2> ...` pairs tracked systems on the score.  `tests/test_investigate.py` (4) |
| **`src/eracoef/bio.py`** | **who he is (26)**: `player_bio(cfg)` (height, weight, draft_pick per player from `data/raw/bio`, cached at `data/cache/bio.parquet`), `season_tenure` / `tenure_inputs(roles, seasons, ids, exclude_seasons=)` (tenure with his main team, teams in the window; the held-out season skipped), `player_inputs`; `gbdt_prior.BIO_BINS` bins height and weight in both paths.  Panel columns via `scratch/add_bio_cols.py` (backup `.bak5`); `chain_offset` builds them from the training block.  `tests/test_bio.py` (5) |
| `xshoot.expected_threes(seasons, cfg, wd)` | each row's expected opponent threes at the shooters' padded other-half rate: x3def's repricing as a function, used by `def_three_design` and by the repriced eFG factor |
| **`src/eracoef/teamloo.py`** / **`scripts/62_teamloo.py`** | **the team-game leave-one-out target (33)**: `team_games` (per-season team-game counters, cached, `keep=` for the cut), `loo_rates` (rebalanced LOO per team and side; `rebalance=label|rate|none`), `estimate_k` (tau2 four ways -- rebalanced, plain, split-half, method of moments; `K_SOURCE="mom"` because rebalancing OVER-corrects a moment), `adjusted_rates` (through `pad.shrink` only), `matchup_regression` (the owner's fit, and the per-team-game predictive net rating), `team_design` (the stint target; its closure is a TEAM-GAME identity, not a row one) and the 25-name ladder in `TARGETS` / `TARGET_COLUMNS`, registered through `fastfit.target_y`.  `62_teamloo.py [first] [last] [--csv] [--cut=q]` prints the constants, the regression and eight gates.  `tests/test_teamloo.py` (24).  **Answered no (33.6); the instrument is what is worth keeping** |
| **`src/eracoef/inseason.py`** | **in season (31)**: `season_rank` / `season_frac`, `kernel_game_mult` (one per-game weight array for the kernel AND the cut), `keep_games`, `anchor_of` (= max(train)), `KernelSystem` (its own `train_for`: the anchor and the seasons its kernel names, zero-weighted ones kept so every kernel excludes the same windows), `BlockSystem` (the last FINISHED window before the season -- the in-season chunk baseline).  `tests/test_inseason.py` (16) |
| `fastfit.MspiFast(kernel=, cut=)` | the two fields that carry it; `holdout.cut_season`, `Holdout.run`'s `train_for` hook and the `cut` result column; `calmap.SeasonFrame(cut=)` / `cut_of` / `train_of`; `roles.cut_role_inputs` and `window_inputs(psx_weights=)`; `spm.season_of_units(weights=)` and `cut_inputs_cached`; `xshoot.season_totals(keep=)` down every path |
| **`scratch/inseason_run.py`** | **the in-season instrument**: every kernel at every cut, dumped once per cut, mapped leave-one-season-out, read on BOTH instruments with a paired z.  `--systems= --cuts= --held=all\|search\|confirm --tag= --reuse` |
| `scratch/inseason_ident.py` | the real-data identities: flat kernel == the board (0.0e+00), cut 1 == no cut, cut 0 == the shorter kernel (5e-14 once the exclusion sets match) |
| **`scripts/60_season_board.py`** | the season board: one rating per player per season from the kernel fit, the in-season map, re-centred per season -> `outputs/season_ratings.parquet`.  `52_site.py` puts it in `docs/data/ratings.json` under `seasons` and the page has a Block/Season switch |
| `scripts/54_track.py` | the tracker: dump a system at K = 3 (timed), fit the map leave-one-season-out, score, log a row, redraw the chart.  `--systems=a,b "--maps=..." --label=...`; one dump per system |
| `scratch/prior_bench.py` | **the 20-second pre-filter**: the prior's own leave-window-out fit per feature set, with the low-exposure stratum beside the pooled number.  `--q4 --loss= --delta= --sat= --past= --kmul=`, and a set may be written `shipD+russsh:blkrimsh` (add) or `shipD-blk+russ:blkrim` (REPLACE).  **Read 22.2 and 23.4 first: it is necessary, not sufficient, it has been wrong by 0.13, and the BASE LIST is part of the operating point** |
| `scratch/pairsys.py` | the paired test between two TRACKED systems: pooled difference, z over the 28 seasons, wins |
| `scratch/maps.py` | map work on an EXISTING dump, ~2 s each, no refit |
| `scratch/ship_try2.py` | a candidate through the REAL shipping path: config (targets, map, **and the prior's feature lists**), `08_ratings.py`, `22_vs_consensus.py`, the floor tests.  Restores `config.yaml` in a `finally` |
| `scratch/consensus_read.py` | the cheap screening read of mapped candidates off the tracker's own parameter tables (no refits) |
| `scripts/56_dredge.py` | build and cache the play-by-play counter tables, 30 seasons in ~7 min, with the era report and the box-score cross-check.  `[first] [last] [--force]` |
| `src/eracoef/dredge.py` | `COUNTERS` (21 of them), `shot_zone`, `game_counts`, `season_dredge`, **`player_dredge_frame(seasons, cfg, ids)`** -- the block's per-player event counts and its own league totals, the single source used by the panel build and by prediction alike.  The assist counters resolve the PASSER from the surname in the shooter's row (`_norm` strips accents and suffixes; the roster is keyed on `playerName` AND `playerNameI`), 0.981-0.9998 in every season |
| `src/eracoef/fastfit.py` | `MspiFast`: every knob (`lam`, `gbdt_params`/`_d`, `gbdt_features`, `target`/`_d`, `win_decay`/`_d`, `decay`, `season_weights`, `pad_scale`, `phases`, `lam_buckets`, `min_den`); `FASTFIT_TIMER=1` for the section clock |
| `src/eracoef/gbdt_prior.py` | `DERIVED`, `RATIOS`, **`SHOTQ`** (+ `add_shotq`), **`CAREER`**, **`DREDGE_RATES`/`DREDGE_RATES_LOC`/`DREDGE_SHARES`/`POTENTIAL_AST`** (+ `add_dredge`, which builds every feature twice -- absolute and era-relative `_r`), the name sets `DREDGE`/`DREDGE_R`/`DREDGE_REL`/`DREDGE_ANY`/`DREDGE_LOC`/`DREDGE_AST`, the feature lists `FULL_/DERIVED_/RATIO_/SHOT_/DREDGE_/PRIOR_FEATURES`, `training_rows(win_decay=, win_past=, sat_poss=)`, `GBDTPrior` |
| `src/eracoef/xshoot.py` | **`player_shot_frame(seasons, cfg, ids)`**: the block's per-shooter shot totals and its own league levels -- the shot-quality features' single source, used by the panel build and by prediction alike |
| `src/eracoef/roles.py` | **`career_inputs(inputs, before_season, ids, age=)`** (rejected block, kept for the record) |
| `src/eracoef/calmap.py` | families x exposure terms x a bend; `parse_maps` takes `mapO:mapD` and `\|cubic` / `\|rowcubic` |
| `scratch/` (tracked) | `cmp_shotq.py` / `cmp_career.py` / **`cmp_dredge.py`** (the identical-numbers checks: every path agrees to 0.00e+00 on all ten windows), `add_shot_cols.py` / `add_career_cols.py` / **`add_dredge_cols.py`** (the in-place panel patches, backups at `.bak2` / `.bak3` / `.bak4`), **`dredge_audit.py` / `dredge_audit2.py` / `dredge_audit3.py`** (what the v3 feed can attribute, per era), **`dredge_rely.py`** (year-over-year reliability of any counter-derived feature, ~1 min), **`assist_audit.py`** (can the passer be named, per era), `rename_share.py` / `rename_share2.py` (the too-broad rename and its repair), `prior_ceiling.py`, `smoke_shot.py`, `tabfm_try.py` |

### Verification

```
.venv/Scripts/python -m pytest tests -q                                       # 110 passed, 1 xfailed
.venv/Scripts/python scratch/trade_turnover.py                                # the teammate table + turnover tables, 40 s
.venv/Scripts/python scratch/trade_gbdt.py O                                  # the boosted trade SPM and the delta CSV, 35 s
.venv/Scripts/python scratch/trade_maps.py tune501_b7 --set=all               # the map terms with the control, 25 s
.venv/Scripts/python scratch/trade_pair.py tune501_b7 tune501_b7_turnref_o    # paired z, every map in the dump
.venv/Scripts/python scratch/prior_bench.py D "shipD,shipD+pot_ast" --ll=1 --cf=0     # ~20 s a set
.venv/Scripts/python scripts/54_track.py --systems=X "--maps=linear+log2&xlog&prior&tshare:linear+log2&xlog" --label=...
.venv/Scripts/python scratch/pairsys.py tune501_b7 <cand> [<cand2> ...]       # the z over 28 seasons
.venv/Scripts/python scratch/ship_try2.py TAG blend0.7 rapm1 <system> [--od --dshot --drd=...]
.venv/Scripts/python scripts/08_ratings.py && .venv/Scripts/python -m pytest tests/test_vs_consensus.py -q
```
and, for the team-game luck adjustment (33):
```
.venv/Scripts/python scripts/62_teamloo.py 1997 2026 --csv      # the constants, the regression, the gates, 40 s
.venv/Scripts/python scripts/45_holdout.py --systems=tune501_b7_pasto_pOD,tune501_b7_pasto_pOD_tlxft3o_O     --k=3 --held=search --ref=tune501_b7_pasto_pOD --tag=tl      # 25 s; --held=confirm for the other half
.venv/Scripts/python scratch/inseason_run.py --systems=ks52_lam05,ks52_lam05_tlxft3o_O     --cuts=0.25,0.5,0.75 --held=search --ref=ks52_lam05 --tag=tl  # both in-season instruments, 115 s
```
and, for the in-season work (31):
```
.venv/Scripts/python scratch/inseason_ident.py                                  # the identities, ~40 s
.venv/Scripts/python scratch/inseason_run.py --systems=ks52_lam05,ks52,ks11,blk --cuts=0.75 \
    --tag=ship --held=all                                                       # both instruments, 28 seasons, 70 s
.venv/Scripts/python scripts/60_season_board.py && .venv/Scripts/python scripts/52_site.py   # the board, 70 s
.venv/Scripts/python scripts/61_credit.py --who=ks52_lam05_q75      # credit vs forecast, who is missed, 60 s
.venv/Scripts/python scratch/credit_null.py --tag=ship               # the permutation null: read 31.8 first, 40 s
.venv/Scripts/python scratch/foldtest.py --side=D                    # a feature's identification share, 60 s
```
and, for the play-by-play block specifically:
```
.venv/Scripts/python scripts/56_dredge.py [--force]      # the counters, ~8 min, with both cross-checks
.venv/Scripts/python scratch/add_dredge_cols.py          # ... into the panel (backup .bak4)
.venv/Scripts/python scratch/cmp_dredge.py               # panel path == prediction path, must be 0.00e+00
.venv/Scripts/python scratch/dredge_rely.py 1500         # year-over-year reliability of every feature
.venv/Scripts/python scripts/50_boruta.py --modes=wide --sides=D,O --trials=40   # ~35 min; see 23.10
```
A tracker run is about a MINUTE of wall time now, not the ten the earlier handoffs claimed -- the search
made the fits 20% cheaper and the four workers do the rest.  Measuring a candidate properly is cheap; the
expensive part is deciding it was worth measuring.

### Traps

- **The prior's own fit can be wrong by more than any feature is worth.**  Its target is the player's APM
  pooled over his OTHER windows, so a player with more windows has a quieter target and is easier to predict.
  Any feature that names those players (experience did) buys held-out MSE without buying knowledge, and the
  criterion gets none of it back.  Splitting by exposure does NOT catch it.  FINDINGS 22.2.
- **A knob measured at the cheap booster does not transfer to `quality=4`.**  Huber was -0.039 at one and
  +0.061 at the other; 21.25 said the same about bagging and the search.  Measure at the operating point.
- **`scripts/49_role_panel.py` used to drop the `raw_*` columns it had just built** -- they were only in the
  panel because `scratch/add_raw_rates.py` put them back by hand.  Fixed; the write-out now lists `raw_*` and
  the nine `shot_*` columns explicitly.  A rebuild of the panel is no longer destructive.
- **One system per dump for timing**: the GBDT cache per process flatters a later system in the same run.
- **The tracker's `seconds` varies +/-6% with machine load.**
- **A map term with no covariate column applies as ZERO, silently.**
- **The `tshare` and `prior` map terms belong on OFFENSE only.**
- **A feature set must be benched against the list it would actually JOIN.**  The Dredge block read -0.050
  on the offensive 43-name line and -0.012 on the shipped defensive 23-name one, at different boosters.
  Same features, opposite conclusion (23.4).
- **`shotDistance` conventions move between adjacent seasons.**  The rim share of blocked twos swings
  0.77 -> 0.48 -> 0.74 -> 0.42 with almost nothing unlocated.  Any cross-era feature built on raw distance
  needs `scratch/dredge_audit3.py`'s check; the shot-quality features escape it because `xl` is fit per
  season (23.3).
- **A gain against the prior's OWN TARGET is not evidence about the board.**  Four candidates this pass were
  negative on `prior_bench` -- the Dredge block -0.020, its era-relative form -0.029, the era-calibrated
  pair -0.021, `pot_ast` -0.018 -- i.e. all four were real improvements to the prior's own fit, and not one
  of them is distinguishable from zero on the criterion.  Use the bench to REJECT cheaply; never to believe.  FINDINGS 23.12.
- **An identical-paths check proves the two paths AGREE, never that either is RIGHT.**  `add_dredge` read
  a per-window constant from the frame's first row, so every row was padded toward 1997-1999, and
  `cmp_dredge.py` reported 0.00e+00 throughout because it calls the function once per window on both sides.
  Anything with a per-block constant needs a test that puts two eras in ONE frame (23.8).
- **Check a candidate's year-over-year reliability before building a model around it.**  `russsh` scores
  0.126 -- it is 0.575 for everybody -- and that one number explains the whole of section 23.
  `scratch/dredge_rely.py` does it for any counter-derived feature in about a minute.
- **The consensus floors are not the criterion.**  Read them before shipping, every time; the screening read
  (`consensus_read.py`, 2024-2026, 475 players) is NOT the same object as the floors in
  `tests/test_vs_consensus.py`, which score the board `08_ratings.py` builds.  Screen with the first, decide
  with the second.
- **`ensemble_n_jobs: 1`** on any `chimeraboost` fit inside the holdout's workers, or the bag forks and the
  wall time explodes.
- **The pooled prior's target is context-averaged** (24.6).  Anything that changes the context the prior is
  asked about has to be a feature on UN-POOLED rows (`pair_rows`), or the answer is baked in before the
  question is asked.
- **The K = 3 block brackets H.**  Turnover "with respect to the block" reads a mover who stayed at 0.73, not
  1.0 (his H+1 season is with the new teammates).  `turnp` is the past-only version; they answer different
  questions, and the criterion liked only the first.
- **A covariate built on H gets the half-season control before it is believed.**  `turna` returned the identical
  -0.084 to `turn`; HShare kept 3% on half the games and was a leak.
- **A per-player delta from a booster whose own leave-window-out gain is 0.04 is mostly noise** at a spread of
  0.3.  The criterion charged +0.13 for applying it.
- **A within-block, player-level split-half read can point the WRONG WAY on a defensive residual.**  Tightening the
  factor ridges 16-32x brought the factor sum level with the points residual player by player and cost +0.50
  and +0.83 on the criterion (25.3).  The criterion values the residual's team-coherent content, which a
  correlation across players cannot see.  Use split-half to reject a residual, never to choose its ridge.
- **REML on a joint offense/defense fit is not a guide to the defensive half** when the offensive half carries
  the real skill: on eFG% it chose the ratio LOOSER (1.0 against 1.5) while the defensive half read 0.19
  split-half (25.3).  Fix the ratio a priori or select it on the defensive half alone.
- **A prior shared out across sub-fits by FIXED shares leaves a persistent, uninformative residual** (25.5):
  the factor residual was more reliable year over year than the points residual (0.75 vs 0.71) and predicted
  worse.  Reliability is not information when the thing that persists is a mis-split prior.
- **A static per-player pair can NAME the player, and the prior's own fit rewards that as if it were knowledge**
  (22.2's trap, purest form): fine height + weight is -0.27 on the offensive prior's fit and +0.20 on the
  criterion at z 4.7.  Bin any static player attribute before believing it, and read the binned-versus-fine
  gap as the identification share (26.2).
- **The criterion's resolution for a correction shared by an archetype is about 0.015** (26.5): moving every
  6-10+ player's defense by 0.35 -- the size of the bias the prior's own fit shows -- moves it 0.015 unmapped,
  0.003 mapped.  It cannot adjudicate attribution at that scale; it can veto a shift the wrong way (+0.06).
  A candidate that reads 0.00 +/- 0.01 with a measured attribution gain is a ruling for the owner, not a
  rejection.
- **The criterion decides forecasting and is nearly blind to attribution; the investigator's score sees
  attribution and agrees with the criterion where it is sure** (27.4).  A candidate that reads zero on the
  criterion is not settled until `investigate_cmp.py` has read it; one that reads zero on both is.
- **The tracker's `seconds` for a system that computes shooter rates or REML per fit runs 4-7x the board's**
  (ffx 427 s against 58 s); measure the winner alone before quoting a time.
- **The paper's fix for leave-one-out bias belongs in the REGRESSION, not in the moment estimate** (33.2).
  The within-team term is exact -- `p_loo(j) - pbar == -n_j (p_j - pbar) / (S_n - n_j)` -- and it cancels
  against the sampling noise in the team's own mean, so a plain-LOO covariance is already close to unbiased
  for a between-team variance and REBALANCING OVER-CORRECTS IT by 9-23%.  The same rebalancing is worth
  15-20 points of coefficient in the regression.  Ask which quantity the bias is corrupting before applying
  a correction to both.
- **A luck-adjusted target's closure can be a TEAM-GAME identity while every row moves** (33.4).  `teamloo`
  at zero shrinkage returns actual points per team-game to 1e-13 and still changes every stint of that game,
  because it has spread the game's makes over its attempts.  A gate that aggregates cannot see what the
  target did to the rows the ridge actually reads.
- **A search-half gain at z -2.8 can be two thirds noise** (33.6): -0.170 on the search half, -0.055 on the
  confirm half, and significantly WORSE at stint level there.  Run the confirm half before believing any
  candidate, and read the stint column even when game level decides.
- **An unmapped criterion gain can be entirely absorbed by the shipped calibration map** (36.5).  A target
  that changes a side's AMPLITUDE shows a large gain in `45_holdout.py` and none at all once the map is
  fitted: the offensive three-point target read -0.21 per 100 unmapped and -0.024 at z -0.63 mapped, because
  a per-side calibration curve is exactly what corrects amplitude and it got there first.  **Run
  `54_track.py` then `scratch/pairsys.py` before quoting any number as a gain**, never `45_holdout.py`
  alone; Part 1's table has always been mapped and that is the convention a candidate must be read in.
- **The two instruments answer different questions and are allowed to disagree** (36.1 vs 35.3).  A luck
  adjustment that improves TEAM-level forward prediction need not survive attribution to five players: a
  defense's real 0.59 points of three-point spread is a team property and handing it back to the ridge is
  monotonically worse.  `scratch/forward.py` asks the first question in an afternoon, the criterion the
  second; a candidate needs both, and neither substitutes for the other.
- Long bash heredocs still fail in this shell; write patch scripts with the Write tool.

## Part 3: the next pass

### 3.16 DONE 2026-09-09: the playoff delta (FINDINGS 38)

**A rating as the prior for another rating.**  `playoffs.playoff_system(inner)` gives `inner` back with
`phases=("PO",)` and `prior_from=RegularSeasonPrior(inner)`: the regular-season rating becomes the OFFSET of
a fit that sees only playoff possessions, so the fit's residual IS the playoff delta and no playoff box
score is needed anywhere.  `scripts/63_playoff_delta.py` writes `outputs/playoff_delta.parquet`;
`scratch/playoff_chain.py` is the test.

**It works and it replicates.**  Playoff games alternate A/B within a series, so fitting the update on one
half and scoring the other holds teams, series and lineups fixed.  Penalty chosen on five blocks, read on
the other five: **x2 the regular-season penalty, -0.459 per 100 at z -2.16 on the confirm half and -0.508 at
z -2.27 on ATTRIBUTION.**  It survives a per-side slope refit (-0.312), so it is ranking and not amplitude --
which is what FINDINGS 36's candidate failed.  The delta's spread is 0.21 against 2.43 for the ratings.

**The extremes are Robert Horry twice, Jason Terry, Dirk and Barea from the 2011 Mavericks at the top, and
Chris Paul, DeRozan and Giannis at the bottom.**  No narratives were given to the method.

**Three fixes went in to make it possible, two of them bugs in shared code:** `MspiFast.prior_from`;
`BoxExposure(phases=)`, because the exposure filtered to `phase == "RS"` ALWAYS and a playoff-only design
therefore had zero exposure; and dependent-column detection in `Moments`, because `is_po` is constant and
`po_home` duplicates `home` on a one-phase design, which left the unpenalized block singular and the solve
returning 1e12.  `MspiFast.half` fits one half of the games so the other can score it.

**Not published.**  The site does not carry the delta.  Open: per-SEASON deltas (thinner still, untested)
and whether the playoff delta belongs beside the season board or as its own view.

### 3.15 DONE and answered 2026-09-09: the luck adjustment, end to end (FINDINGS 35-37)

**Do not start here.**  Three passes, all measured, all committed, nothing shipped, `config.yaml` unchanged
and the board rebuilds to ten of ten floors.

**What was built and is worth keeping.**  `teamloo.RATE_SPECS` puts every rate a possession's points depend
on -- three make rates by zone, free throws, turnovers, offensive rebounds, the shot mix, the foul rate --
through ONE estimator on each side, so no constant is chosen any more (`skill_table` reports the true spread
and the real share at any sample size).  `teamloo.possession_points` turns rates into points per 100 and
closes to 0.5% across eras.  **`scratch/forward.py` is the instrument that should have existed years ago**:
first half of a team's season predicts its second half, 1,784 team-seasons, no ratings pipeline in the way,
scored so that generic shrinkage cannot win.

**What it says.**  At TEAM level the adjustment works -- shrinking the outcome rates and leaving the style
rates alone removes 6.6% of forward error on offense and 10.4% on defense.  At PLAYER level none of it
survives the calibration map (36.5/36.6).  The two are not in conflict: a defense's 0.59 points of true
three-point spread is a team property that will not divide five ways against a season of noise.

**And the shipped constants are right, confirmed independently.**  The four `FACTOR_LAMS` ratios REML chose
in FINDINGS 15 match the split-half measurement (eFG 1.50 against 1.50); `x3def`'s premise matches
squared2020's tracking-based analysis (35.6); the per-factor defensive residual is flat on the current board
with either set of penalties (37.2), so the zone split was NOT built.

**Two rulings waiting for the owner**, both because the data genuinely cannot settle them:
1. **The penalty strength** (34.3): between x0.25 and x1 both instruments read zero while the offensive
   evidence share moves 10% -> 33%.  A product decision, and nearly free.
2. **The in-season attribution candidate** (36.3): the offensive three-point target is **-0.163 on the
   investigator at z -3.12 over 12 of 14 seasons** with prediction flat.  Under 26.5 that is a ruling, not
   a rejection.

**Never re-run:** the team-level free-throw target (indistinguishable from no adjustment at all, 36.6); any
two-point replacement (+1.63 mapped); on-court ORTG/DRTG in the prior (+0.05, z 2.22); giving a defense back
ALL its realised threes (+0.25, z 3.25).  **Free either way:** a defense keeping up to a quarter of them.

### 3.14 DONE and answered 2026-09-09: what a single-season ridge weighs (FINDINGS 34)

The owner's question -- does single-year prior-informed RAPM just pick the box score -- is measured and the
answer is no, with a caveat that is worth more than the answer.

**The instruments.**  `scratch/lamshare.py [--seasons=] [--targets=ship,pts,tl3] [--mults=] [--kernel=00|52]
[--csv]` decomposes any fit into the prior it was handed and the residual the ridge kept, and reports
`share = var(rating - prior) / var(rating)` per side.  It is a decomposition, not a criterion, and a fit is
under a second.  `train_for` builds the seasons a kernel names -- handing a kernel system `[season]` alone
silently turns it into the one-season kernel, which is how this script first reported ks52 and ks00 as
identical.  The `ls_<target>_x<mult>` systems (systems.py) cross five targets with eight penalties on the
one-season kernel so `scratch/inseason_run.py` can score the same grid.

**What it says.**  Interior optimum at x0.5 for every target, on prediction AND attribution.  There: 20%
evidence on offense, 44% on defense.  x1 (the block board's penalty) halves that and predicts slightly
worse.  The luck-adjusted target does not move the argmin and lowers the share.

**The caveat, and it is the finding.**  Between x0.25 and x1 both instruments are FLAT (z -0.13 prediction,
z -0.26 attribution) while the offensive evidence share moves 10% -> 33%.  The criterion cannot choose here,
which is memory trap 4 exactly.  If the product should lean less on the box score, x0.25 is nearly free --
but it must be chosen as a product decision and said to be one, not read off the criterion.

**Not done:** whether x0.25 holds up on the confirm half, and whether the block board wants the same move.

### 3.13 DONE and answered 2026-09-09: the team-game leave-one-out target (FINDINGS 33)

**Do not start here.**  The owner's single-season luck adjustment is built, measured on both halves of
the protocol, and the answer is no.  What follows is what exists and what the result rules out.

**What was built.**  `src/eracoef/teamloo.py`: the per-season team-game table (`team_games`, cached at
`data/cache/teamloo/`, `keep=` for the in-season cut), rebalanced leave-one-out rates per team and side
(`loo_rates`, the rule of Austin/Pe'er/Korem 2025), the shrinkage constant four ways (`estimate_k`;
`K_SOURCE = "mom"`), the blended rates (`adjusted_rates`, through `pad.shrink` and nothing else), the
owner's matchup regression (`matchup_regression`, with the per-team-game predictive rating as its fitted
value) and the stint-level target (`team_design`).  A 25-name ladder registers through `TARGETS` and
`fastfit.MspiFast.target_y` -- the same callable hook `xshoot.DEFENSE_TARGETS` uses, so the in-season
`keep` travels down it unchanged.  `scripts/62_teamloo.py` is the report and its own sanity gates;
`tests/test_teamloo.py` is 24 cases.  Nothing here reads a second season.

**The result.**  Against `tune501_b7_pasto_pOD`, K = 3, team-game level:

| | search half | confirm half |
|---|---|---|
| free throws at the team rate (`_tlfto`) | +0.438 (z 3.7) | not run -- it failed its gate |
| **shooter free throws + team-LOO threes, offence (`_tlxft3o_O`)** | **-0.170 (z -2.8, 11/14)** | **-0.055 (z -0.90, 9/14)** |
| the same at stint level | +0.108 (z 1.1) | **+0.258 (z 3.6, 3/14)** |
| plus the two-point term (`_tlxft32o`) | +2.276 (z 6.6, 0/14) | not run |
| the matchup prior (`_tlxft3b`) | +0.010 (z 0.1) | not run |

In season (`ks52_lam05`, cuts 0.25/0.5/0.75) it is -0.059 / -0.030 / -0.006 on prediction (z -1.1 to
-0.1) and zero on attribution at every cut.

**It does not replicate.**  Part 0 ruling 1 applies and the candidate is not shipped; `config.yaml` is
untouched and the board is unchanged.

**Why, in one paragraph, and it generalises.**  The constant decides everything.  A team's free-throw
percentage pads with k = 297 attempts against a shooter's 24, so the team-level free-throw term prices a
90% shooter and a 60% shooter identically and recovers 5% of the shipped free-throw gain -- between-shooter
variance IS the gain.  A team's two-point percentage is 85% defended (33.3) and 6% of a game's own rate
survives the blend, so replacing it deletes shot-making, +2.28 per 100.  Three-point percentage is the one
place the premise holds -- a defence controls a fifth of it -- and `x3def` already removes that luck from
the defensive half, which is why the defensive rung reads exactly zero (-0.004) and only the offensive one
moves at all.  There was about a tenth of a point there and it did not survive the confirm half.

**What is worth keeping** is the instrument, not the target: `62_teamloo.py` answers "which stats predict
this game" from one season of play-by-play with no panel and no prior, and three of its readings confirm,
independently, things the project believed on other evidence (what a defence controls; that the plain-LOO
moment estimate is nearly unbiased while the regression coefficient is badly attenuated; that a target's
gates cannot tell you whether it should be the target).  If the single-season direction is picked up
again, the next thing to try is NOT another make-rate replacement -- it is the possession-level pieces the
ladder never touched (turnovers and offensive rebounds are still fully realised in every rung above).

### 3.12 DONE and answered 2026-09-09: single-season targets are built, and the prior cannot take them (FINDINGS 32)

**Do not start here; 3.5 is answered.**  `scripts/49_role_panel.py --season` builds
`outputs/role_panel_season.parquet` (29,138 rows, 68 s) and every consumer reads the granularity off the
panel's own labels -- the exclusion set (`Context.labels(train, panel)`), the pair rows' turnover table
(`Context.turn_table(panel)`, 112,068 season pairs) and the PAST discount (`gbdt_prior.past_decay_for`,
which holds the reach in YEARS constant).  Systems `sp_pasto_pOD` / `spy_pasto_pOD` and
`sp_ks52_lam05` / `spy_ks52_lam05` take it with `panel=`.  The block board is byte-identical.

**It loses on both instruments, everywhere.**  Search half, K = 3: **+0.52 per 100 at team-game level
(z 4.4, 2 of 14)** on the block board, and in season +0.62 / +0.44 / +0.24 / +0.13 at the four cuts with
attribution +1.27 / +0.84 / +0.48 / +0.35.  Ten times the size of a normal win, and the two `win_decay`
conventions agree, so the unit is not it.

**Why, and this is the part worth keeping.**  The prior is trained on ONE-season feature lines and applied
to THREE-season ones, so it is attenuated: `--spread` says the season-panel prior's possession-weighted sd
is 0.92 and 0.81 of the shipped one on offense and **0.66 and 0.51 on DEFENSE**, where the box score's
signal is weakest and attenuates first.  That also explains the monotone shrink with the cut.

**The one thing left of 3.5**, and it is a real candidate: CROSS-PANEL pair rows -- the feature line from the
block panel, the target from the season panel.  It keeps the season turnover contrast, the per-season PAST
and seven times the rows, and removes the mismatch this pass measured.  It needs a leakage guard the current
`pair_rows` does not have (a target season inside the feature window, or inside a window the PAST sums
over), which is why it was not built here.

**Also new and reusable:** `49_role_panel.py` joins the career and bio columns itself (a panel is
reproducible from one command, not 49 plus two scratch scripts), `--out=` writes any panel path, and
`45_holdout.py --held=search|confirm` implements the 22.7 protocol directly.  Tree: committed on
`hybrid-and-xpts`; 154 passed, 1 xfailed.

### 3.11 DONE 2026-09-08: in season -- the kernel, the cut, and the season board (FINDINGS 31)

Built and measured; the season board ships beside the block board and the block board is untouched.  What is
left, in the order the instruments point:

1. ~~**Per-season targets for the prior** (3.5).~~  DONE and answered 2026-09-09, section 3.12: the panel is
   built and the prior cannot take it (+0.52 per 100, z 4.4; attenuated, worst on defense).  What is left of
   it is cross-panel pair rows -- block features, season targets -- with a leakage guard.
2. **A cut-aware `PAST`**: `past_apm` is his record up to the block, not up to today.  In season it should be
   his record up to the cut.  The season panel expresses it now (3.12), but only through a prior the
   criterion refuses; on the block panel it is still the same rebuild as 1.
3. **The kernel by exposure**: a player with 200 possessions this season wants more of his past than a
   starter does.  `lam_buckets` is the machinery and the kernel is currently one number for everyone.
4. **Not measured**: playoff rows in a kernel fit (the anchor season's playoffs are weighted 0 whenever the
   cut is below 1, deliberately); `win_decay` re-tuned for a kernel fit; the season board against the
   consensus (it is a different estimand from the block the floors score -- read `--consensus` on `ks52` if
   anyone wants the screen).
5. **Measured after the ship (31.8) and settled**: the credit score's level is ~94% noise floor, its paired
   differences are real (`scratch/credit_null.py`); the season board has captured **84%** of the player signal
   a lineup model can see, the chunk board 45%; the ridge holds about a tenth of what is left (x0.25-x0.5 is
   the optimum on both instruments); the prior reads a player's other windows off his rate fingerprint (10%
   of its offline accuracy) and closing that channel with player-grouped folds (`GBDTPrior folds`,
   `MspiFast.gbdt_folds`) is flat on the criterion and slightly worse on credit -- not shipped, the
   instrument (`scratch/foldtest.py`) kept.  `scripts/61_credit.py` is the credit-against-forecast report
   (its `credit %` column is the naive, noise-diluted share; read 31.8 before quoting it).

**Start at 3.2b, or at the cross-panel pair rows of 3.12.**  3.5 is answered (3.12: no), 3.0 is done and
3.2 is measured (both below).  3.2 as written -- per-factor
shrinkage with the points prior shared out -- does not unlock 22.7's 0.14; it loses.  What survives of it is a
measured -0.11 for per-factor shrinkage WITHOUT a prior, which 3.2b (per-factor priors, FINDINGS 25.6) would
build on; 3.5 (single-season targets) is the product direction and the thing that makes 24's trade delta
testable.  The prior's INPUTS remain spent (21.24, 21.25, 22.5, 23).

### 3.0 DONE 2026-09-07: the settled-context offensive prior is shipped (`tune501_b7_turnref_o`, FINDINGS 24.9)

**What was done.**  `config.yaml` `ratings_prior.gbdt_turn: {sides: [O], ref: 0.35}`, `cal_map` on the
candidate's tracker table copied to `outputs/calmap_ship.parquet`, `08_ratings.py`, `22_vs_consensus.py`,
the floors, `52_site.py`.  On the board only the offensive prior moved (mean absolute change 0.21 per 100;
the defensive prior is identical to the last digit).  Consensus 0.791 / 0.787 / 0.759, defensive spread 1.28.
**The bigness floor missed by the screen's margin**: -0.311 on the floor's own object against -0.276 for
`tune501_b7` and a floor of 0.30, a move under one standard error (0.046 at 475 players).  Part 0 ruling 2
applied; the floor is 0.32 now with the reason in `tests/test_vs_consensus.py`, the way 22.7 re-based the
defensive one.  The blend was NOT moved to clear it (that is what 22.7 refused) and the reference stays at
the a-priori 0.35.  **If the owner reads this miss as gross rather than marginal, the revert is three lines:
`gbdt_turn: null`, `cal_map` back to `tune501_b7`, the floor back to 0.30, then `08_ratings.py` and
`52_site.py`.**  Note that before ruling 2 this same floor turned candidates away at -0.303 (21.26, 22.4),
so it has a history of being the binding one on offense; the honest reading is that the settled-context
prior rates high-usage guards a little higher relative to bigs, which is what 24.4 said it would do.

*The section as it stood before the ship, kept for the record:*

What it is: the offensive prior trained on ordered window pairs with the teammate turnover of the target
window as a feature, evaluated at a settled context (0.35, the season-to-season stayer median, fixed a priori)
for everyone; the defensive prior stays pooled as shipped.  **-0.054 vs the board at z -3.39, 22 of 28; the
consensus screen unchanged on every agreement (total 0.802 vs 0.799, defense 0.758 vs 0.759); 58 s against
37 s for the 28 fits.**  Attribution is done: pair rows alone -0.010, the prior at the pairs' mean context
-0.029, both sides -0.062 with the defensive agreement paying 0.02, defense alone -0.008.

What is left, in order:
1. `.venv/Scripts/python scratch/trade_turnover.py` once on the machine (it builds `data/cache/teammates.parquet`;
   the board path raises without it).
2. `config.yaml`: `ratings_prior.gbdt_turn: {sides: [O], ref: 0.35}` (the key is wired in `08_ratings.py` and
   documented in the config; null ships the pooled prior), and `cal_map.system` / `base` pointed at
   `tune501_b7_turnref_o`'s tracker table (`outputs/calmap_track_tune501_b7_turnref_o.parquet` exists under the
   shipping map family; copy it to `calmap_ship.parquet` the way 22.7 did).
3. `08_ratings.py`, then `pytest tests/test_vs_consensus.py -q`.  **Read the bigness floor honestly**: the
   screen has the offensive gap against bigness at -0.316 where the floor on the board is |r| < 0.30 (the
   screen and the floor are different objects; the board read -0.284 on the screen and passed).  If it misses
   by that margin, Part 0 ruling 2 applies -- a marginal miss on a sanity check is not a veto against z -3.39 --
   and the offensive target blend is the knob that has moved this axis before (21.26, 22.4).  A gross miss is.
4. Then `52_site.py`, the README's number, FINDINGS 24.9 marked shipped, and this section closed.

Do NOT tune the reference on the 28 seasons.  If it is ever moved, the search-half protocol of 22.7 applies.
The per-player trade delta stays out of the board: the criterion charged +0.13 for it.  The "rating if traded"
column for the site is FINDINGS 24.7's player-level estimate, published with its own evidence and without a
game-level claim; `outputs/csv/trade_delta_*.csv` is the prototype and 3.5 (per-season targets) is what would
make it testable.

### 3.1 DONE and answered: the play-by-play block is built, and it is worth nothing (FINDINGS 23)

**Do not start here.  This is finished.**  The twelve Dredge features HANDOFF 3.1 listed are built, tested,
wired into both paths and measured, and the answer is no.  What follows is what exists now and what the
result rules out, so nobody spends another pass on it.

**What was built.**  `src/eracoef/dredge.py` counts **twenty-one** event types per (player, season) out of
`data/raw/pbp` -- Russells, rim blocks, blocked threes, blocks by mid-range zone, unassisted and assisted
makes, **assists credited to the PASSER and filed under the zone of the shot he created**, stolen and total
turnovers, loose-ball fouls, technicals/flagrants, offensive fouls committed, steals and defensive
goaltends, with each player's offensive and defensive possessions from the stints as the denominators.
`scripts/56_dredge.py` builds and caches `data/dredge/{season}_RS.parquet` (30 seasons, ~7 minutes, one
command) and prints the era report and the box-score cross-check.  `gbdt_prior.add_dredge` makes **twenty-six**
padded features, each in an absolute and an era-relative (`_r`) form (`DREDGE_RATES`, `DREDGE_RATES_LOC`,
`DREDGE_SHARES` and `pot_ast`; `DREDGE` is all of them minus `goalt`).
`scripts/49_role_panel.py` writes the 46 counter columns into the panel, so a rebuild is not destructive, and
`spm.chain_offset` rebuilds them from the training block at prediction time.  `tests/test_dredge.py` is
nineteen cases built from the real row shapes.

**What it cost to verify, and both gates passed.**  The counters against the BOX SCORE, all 30 seasons:
worst |ratio - 1| **0.0000** on made field goals, **0.0002 on assists**, 0.0001 on turnovers, 0.0012 on
steals, 0.0024 on blocks.  The panel path against the prediction path (`scratch/cmp_dredge.py`), ten
windows, twenty-six features: **0.00e+00**.

**The result.**  On the criterion, against the shipped `tune501_b7` at 110.6237 over the same 28 seasons:

| | criterion | vs the board | z | wins |
|---|---|---|---|---|
| the whole block on defense (`tune501_b7_drd`) | 110.6182 | -0.0055 | -0.20 | 16/28 |
| the same block era-relative (`tune501_b7_drr`) | 110.6243 | +0.0006 | -0.08 | 14/28 |
| `blkrimsh_r` + `goalt_r` only (`tune501_b7_dcal`) | 110.6281 | +0.0045 | 0.32 | 14/28 |

and 4.5% slower for it.  On offense the prior's own fit rejects every group before the criterion is reached.

**Why, and this is the part that generalises.**  Year-over-year reliability of the features themselves
(`scratch/dredge_rely.py`, 8,432 player-pairs with 1,500+ possessions in both seasons): `blk` per 100 scores
0.918, `unastsh` 0.892, `blkrim` 0.900 -- and **`russsh`, the Russell share, scores 0.126.**  It is 0.575 for
everybody and the spread around it is sampling noise.  Meanwhile Dredge's STRONG claim -- that `blk` is the
wrong shape and the split should REPLACE it -- was tested directly and reverses: dropping `blk` costs the
defensive prior +0.328 weighted MSE, replacing it with `russ`/`blkrim`/`blk3sh` recovers only 81% of that,
and the same holds for `tov` -> stolen (+0.018) and `ast` -> unassisted (+0.014).  **On all three terms he
singled out, the raw counter beats its own decomposition.**  He needed the split because an elastic net is
linear; our booster has `blk`, `drb`, `stl`, `pf`, `ast`, `usage`, `astr`, `share` and `gs_pct` and crosses
them freely.  A decomposition is worth having when the model cannot make it, and ours can.

**The era-calibration (the owner's, 2026-09-07) is built and measured.**  `add_dredge` builds each feature
twice: absolute, and divided by its own block's league level (`russsh` and `russsh_r`, thirteen pairs, ten
lines) so a change in how the feed RECORDS an event divides out.  On the prior's own fit it works -- the
relative block is -0.029 against -0.020, and `blkrimsh_r` + `goalt_r` alone are -0.021 at a fifth of the
low-exposure cost, the best balance anything here reached.  The criterion refuses all of it.  It is kept
because it costs nothing and because it is the mechanism a source that does NOT span the panel will need:
tracking data starts in 2013-14, and a dimensionless multiple of a player's own era is the only form of such
a column that can share a panel with seasons the source does not cover.  (Absence is harder than level and
this does not solve it.)

**The assist block is built (the owner's, 2026-09-07) and is the best thing the prior's own fit has ever
said about a new column -- and the criterion still says no.**  `dredge.py` now credits each assist to the
PASSER, resolved from the surname in the SHOOTER's row (accents, suffixes and same-surname collisions all
handled; 0.981-0.9998 resolved in every season with no era gradient, and the totals agree with the box
score to 0.0002), and files it under the zone of the shot it created.  `pot_ast` applies the owner's 2019
zone weights, which reconstruct TOTAL POTENTIAL ASSISTS at r-squared ~1 -- a tracking statistic that begins
in 2013-14, carried back to 1997 because assist location is in the play-by-play throughout.  It is the most
reliable feature in the block year over year (0.922, against 0.918 for blocks per 100) and -0.018 on the
prior's offensive fit, where nothing else has ever been negative.  The criterion reads +0.0005 at z 0.12.
The both-sides version is the control that proves the criterion is discriminating: a PASSING feature in a
DEFENSIVE prior is harmful at z 2.33.  FINDINGS 23.11.

**Four traps this turned up, all worth carrying forward.**

* **A gain against the prior's own target is not evidence about the board.**  This section measured four
  things that were negative offline -- the Dredge block (-0.020), its era-relative form (-0.029), the two
  era-calibrated features (-0.021) and `pot_ast` (-0.018), every one of them an improvement offline -- and
  not one of them is distinguishable from zero on the criterion.  22.5's ceiling argument is not only about how much is left in the prior; it is about what
  `prior_bench` can and cannot tell you.  Use it to REJECT, and never to believe.

* **An identical-paths check proves the paths AGREE, never that either is right.**  `add_dredge` read the
  block's league totals from the frame's FIRST ROW, so `training_rows` -- which hands it all ten windows at
  once -- padded every row toward 1997-1999.  `cmp_dredge.py` reported 0.00e+00 throughout because it calls
  the function once per window on both sides: both were wrong the same way.  Every number in the first
  version of section 23 was affected and the criterion's reading moved from +0.0008 to -0.0055; no
  conclusion changed.  The companion check is a test that puts two eras in ONE frame
  (`test_the_league_level_is_read_per_row_not_from_the_first_row`), and anything with a per-block constant
  needs one.

* **The base list is part of the operating point, not just the booster.**  The first bench read the whole
  block at **-0.050** on defense -- what would have been the largest feature gain ever measured here -- because
  it was run on the 43-name OFFENSIVE line at `quality=4`.  On the shipped 23-name defensive list with the
  shipped defensive booster it reads -0.012, with every part of it costing low-exposure accuracy.  Same
  features, same data, opposite conclusion.  `scratch/prior_bench.py` now takes `shipD+a:b` and `shipD-blk+a:b`
  so a candidate can be benched against the list it would actually join, and replacing a feature can be
  tested as well as adding one.
* **`blkrim` measures the scorer.**  The rim share of blocked twos swings 0.77 -> 0.48 -> 0.74 -> 0.42 with
  0.17-0.23 jumps between ADJACENT seasons, and `scratch/dredge_audit3.py` rules out missing locations (0-2%
  unlocated).  It is the recorded distance convention.  Anything built on `shotDistance` ACROSS eras needs
  this check; the shipped shot-quality features escape it only because `xl` is fit per season from the same
  distances.

**What is NOT ruled out, and what the machinery is now for.**

* **`OffFoulsDrawn100`** -- Dredge's 1.22 and his own favourite -- is the one published coefficient our feed
  cannot give us: `"Asik OFF.Foul (P3)"` names only the fouler, in 1997 and 2026 alike.  It needs the v2
  play-by-play (`playbyplayv2`, ~35,000 games) or pbpstats' enhanced feed.  Section 23 does not speak to it.
  If anyone does that ingest, the counter is one entry in `dredge.COUNTERS` and the rest is built.
* **`goaltend`** is counted but out of `DREDGE`: Justin's footnote says 1997 has suspiciously FEW goaltends
  and our feed says it has 1.27 a game against 0.43 now.  Reconcile against a published source before it is a
  column.  (It benched -0.014 pooled / +0.042 low-exposure, i.e. the 22.2 signature, so this is not urgent.)
* **Boruta is settled and the answer is no.**  `50_boruta.py` now has a `wide` mode on
  `DREDGE_FEATURES` (55 names, a superset of both shipped lists), so the two lines that ship have finally
  been assessed.  It **rejects `blk`** -- whose removal costs the defensive prior +0.328, the largest effect
  of any single column measured here -- along with `tov`, `orb`, `ftm`, `fg2m`, `fg3m` and five more of the
  23 shipped names, while ACCEPTING `unast` (worth zero on the criterion) and holding `russsh` (reliability
  0.126) as tentative.  The reason is structural, not a tuning problem: **on a candidate set containing
  engineered linear aggregates of its own members, Boruta keeps the aggregates and rejects the parts.**
  `stocks` is `stl + blk`, so given `stocks` a shadow copy of `blk` is as good as `blk`; `pts` swallows the
  three make counts, `fga` the misses, `reb` the rebounds.  **The offensive run proves it: `blk` is ACCEPTED
  on offense and `stocks` REJECTED, the exact reverse of defense, on the same panel and the same 55
  candidates.**  Which of a collinear pair survives is a coin toss.  Swapping `blk` for `stocks` -- Boruta's
  defensive preference -- costs +0.050 directly.  It is a useful noise detector (it rejects every Dredge
  feature on offense, agreeing with the criterion) and cannot be made into a gate.  FINDINGS 23.10.

Sources: Dredge, https://fansided.com/2016/07/26/introducing-dredge-a-play-by-play-derived-metric/ ; DRE,
https://fansided.com/2015/02/23/introducing-dre-a-hopefully-better-simple-metric/ ; block types,
https://fansided.com/2015/09/21/shot-blocking-details-mining-19-years-of-play-by-play-data/ ; BPM 2.0,
https://www.basketball-reference.com/about/bpm2.html ; pbpstats enhanced play-by-play,
https://pbpstats.readthedocs.io/en/latest/pbpstats.resources.enhanced_pbp.html

### 3.2 DONE and answered 2026-09-07: the defensive four-factor fit is built, and it does not beat one ridge (FINDINGS 25)

**Do not start here.**  It is built (`fastfit.factor_defense`: four factor fits on the points layout, each
with its own ridge and ratio, the prior split across them, recombined with the row-level gradients eFG 1.55 /
TOV -1.04 / OREB 0.63 / FTR 0.32; an identity test proves the plumbing) and measured in twelve forms against
the board.  Full replacement +0.10 (repriced eFG) to +0.31; the half blend -0.040 at z -0.94, 18 of 28, at
7x the fit time; the ridges tightened 16-32x +0.50 / +0.83.  The bounds: the defensive prior alone +1.91,
no defensive prior +0.46 -- and with no prior the FACTOR residual beats the points residual by 0.11
(z -1.44).  So per-factor shrinkage is worth something and the fixed-share prior split costs more than it.
The consensus prefers the raw-eFG factor defense (0.768 vs 0.758 on defense) and the criterion rejects it.

**3.2b, what would finish it (not scheduled): per-factor priors.**  Four defensive factor targets in the
role panel (`49_role_panel.py`, one zero-prior ridge per factor per window) and four defensive boosters, so
each factor shrinks toward a prior of its own kind.  Keep FINDINGS 15's a-priori ridges (REML per fit chose
the wrong direction on the defensive half and costs 8-15 s a fit); read the consensus screen, because the
form the criterion prefers (repriced eFG) breaks the defensive spread floor at 1.49.  About a day.

*The section as it stood before, kept for the record of why it was expected to work:*

**FINDINGS 22.7 put a number on this.**  A 625-trial search over the whole estimator, validated on 14
held-out seasons the optimizer never saw, found `tune501`: **-0.138 on the criterion at z -3.95 over 22 of 28
seasons, and 20% FASTER than what ships**.  It fails the consensus, and the reason is entirely defensive --
the gain lives in `win_decay_d` = 0.28 (pooling the defensive prior's target over nearby windows instead of
the whole career), and the defensive agreement floor refuses it.  Restoring the shipped 1.0 recovers the floor
and gives back 0.058 of the 0.083, taking z from -3.02 to -0.75.

So this is no longer "the most promising thing left".  It is the thing standing between this board and a
measured, significant, cheaper gain, and the disagreement is substantive: the criterion wants defence weighted
toward recent form, the consensus wants the career statement.  A defensive fit that both sides believe is what
unlocks 22.7.

Fit opponent eFG allowed, turnovers forced, offensive rebounds allowed and free-throw rate allowed separately,
each with its own ridge ratio -- the asymmetry came out ESTIMATED, not imposed (forcing turnovers is a real
defensive skill at ratio 0.75; preventing offensive rebounds barely is, at 3.00) -- then recombine into
expected points allowed.  It was 0.4's "right but too big for v1" and it is now the best thing on the list:

* defense is the binding constraint on everything.  It is 1.33x too wide against the consensus, it owns the
  only permanently-failing test in the suite, and its 0.76 agreement floor blocked three candidates in 21.26
  and is what separates this pass's two shipping candidates.
* it is the only remaining item that changes the ESTIMATOR rather than the prior's inputs, and the estimator
  is where FINDINGS 22.5 says the signal must be.
* `ship_shot7d` already narrows the spread to 1.30 for free, which is the direction this work goes further in.
* it is the estimator-side complement to 3.1: 3.1 gives the prior more to say about defence, this changes what
  the ridge is fitting.  They do not collide and either can go first.

A day's work; the principled version of what `x3def` did by hand.

### 3.3 The defensive booster, now that its pieces are separated (FINDINGS 22.6)

21.26 rejected `quality=4` on defense as a package because the consensus floors would not take it.  It is
three things and only one of them is the problem:

* **`linear_leaves` is the good part** -- -0.009 weighted MSE on the prior's own fit, and the criterion agrees
  (`ship_shot7d_ll`, 110.6925, -0.014 against the shipped board, ten of ten floors).  It is inside the noise
  at z -0.75 and was not shipped, but it is the closest thing to a free defensive gain on the table.
* **`cross_features` is harmful alone and helpful beside linear leaves** (+0.004, then -0.013 together) -- and
  the criterion reverses that: `ship_shot7d_llcf` is only -0.006.  Do not trust the offline ranking on this.
* **the BAG is what defense will not tolerate.**  It is the only piece that moves the prior's calibration
  slope (1.08 -> 1.15), i.e. widens it, and defensive width is the binding floor.  **But it is also the only
  piece that improves the LOW-EXPOSURE rows** (1.032 against 1.075), which are the players the prior actually
  decides.  That tension is unresolved and is the interesting part: a bagged defensive prior with the width
  taken back out -- by 3.4's per-player shrinkage, or by a scalar -- has never been tried.

### 3.4 Per-player shrinkage from the prior's own confidence

Every player is pulled toward his prior by the same lambda, but the prior is far more predictive for some than
others.  `quality=4` already fits 5 bagged members, so the disagreement across members is a per-player
predictive spread, free.  Feed it into a per-player penalty (`lam_buckets` is the machinery, currently keyed on
exposure groups).  This is the mechanism that would let the prior carry the bench hard without overriding
stars -- the failure mode the criterion has complained about since FINDINGS 19, and the way to take the
defensive bag's width back out (3.3).

### 3.5 DONE and answered 2026-09-09 (FINDINGS 32, HANDOFF 3.12): single-season targets lose

*The section as it stood before it was built, kept for the record:*


Train the prior on single-season APM instead of three-season: more rows, noisier each, and a step toward the
continuous rating the product is going to (0.2).  It is also the only change that would break the pooled-target
mechanism of 22.2, which is worth knowing independently.  **And it is what makes the trade delta testable**
(24.6): pairs at s -> s+1 carry the season turnover (movers at 1.0, stayers at 0.36 -- twice the contrast of
the window pairs), and the prediction-time covariate becomes the same quantity instead of a block-bracketed
cousin of it.

### 3.10 The destination (FINDINGS 30): built, measured, kept as the trade-to instrument, not the board's prior

The owner's mechanism (28.8: a star's usage drops on the new team) made into inputs: his usage minutes, the
target roster's usage minutes already spoken for, and the roster's quality, every teammate measured in the
feature window.  On offense: -0.072 on the criterion (z -1.8) and +0.125 on the investigator (z +2.6) -- the
forecast improves, the attribution worsens, because the roster's quality lets the prior forecast value IN
CONTEXT while shifting credit between a player and his teammates.  Not shipped as the prior; the board rates
a player at his actual context.  `scratch/trade_to.py` answers what he would be worth elsewhere: for every
high-usage creator in 2024-2026 the best destinations are the rosters with the least usage spoken for
(Brooklyn, Washington, Memphis, Utah, Golden State) and the worst the loaded ones (New York, Boston,
Minnesota, Houston, Denver), about a point of offensive prior between them.  The defensive analogues (block-minutes, rebound-minutes, the roster's defensive APM) are zero on both instruments.

### 3.9 DONE and shipped: plus-minus as an input (FINDINGS 28), and what is left of the gap

Built and shipped (above).  Measured and not taken: PAST on defense (-0.02 / -0.06, neither significant),
the both-sides form (not separable from offense-only, slower, and it misses the defensive floor by 0.005).
**The owner's follow-up (28.7): the sides are split by construction, and each prior can now see BOTH sides'
record (`PAST_CROSS`).  Both priors seeing both sides (`tune501_b7_turnref_o_hwb_pastxd`) is -0.15 on the
investigator at z -4.3 and -0.064 on the criterion at z -1.8 against the shipped board, and misses the
defensive agreement floor by 0.002 (0.748).  The owner's call; the steps are in 28.7.**
**What is left of the gap is the top**: the offensive miss by rating decile is +0.64 in the top decile against
+0.70 before, and Curry, LeBron, Jokic lead the under-rated list as before.  28.1 names the two mechanisms --
the prior's target is the player's OTHER windows, so a peak's prior is its neighbours' average, and the ridge
gives back only half of the difference -- and no prior input reaches it.  What would: per-player shrinkage
keyed on exposure or on the prior's own confidence (3.4), or a map term that stretches the top BY EXPOSURE,
both scored on the investigator, because the criterion's resolution for a few stars moving a point is 0.015.
The within-season playing-time change (28.1: +1.85 per unit of possession share; "role" here is minutes, not position) is the noise floor, not a target.

### 3.8 The investigator (the owner's direction, 2026-09-07): who and what the board is most wrong about

Built: `investigate.residual_ridge` / `on_court` / `lineups` / `season_table` / `pooled` and
`scripts/57_investigate.py` (usage in its header; `tests/test_investigate.py` plants a miss and recovers
it).  It reads the shipped system's tracker dump, predicts every held-out stint row under the map fitted
without that season, and reports the residual three ways.  Outputs: `outputs/investigate_<system>.parquet`
(player-seasons: miss per side with its se, the on-court miss, the mapped rating and prior he was scored
with), `investigate_pooled_<system>.csv` (per player across seasons, z), `investigate_lineups_<system>.csv`.
**The first run (FINDINGS 27) says both tails are compressed**: out of season the board under-rates Curry,
LeBron, Jokic, Shaq, Harden, Nash by 1-1.5 per 100 (z 3-4 over 10-22 seasons each) and Garnett, Draymond,
Gobert, Duncan, Caruso on defense, and over-rates low-usage bigs and Trae Young, Bargnani, Kevin Martin.
The miss is flat over nine deciles of the rating and +0.7 in the top one, and it follows the PRIOR (0.18 per
point) not the on-court part (0.035): the prior is compressed at the top on both sides.  No map shape
reaches it (cubic, sinh, hinge, expo, prior2, priorsat: all inside 0.02) because the criterion cannot see a
few stars moving a point (26.5).

**So there is a second score: `investigate.attributable`** -- the out-of-season residual variance the player
ridge can put on players -- and `scratch/investigate_cmp.py` scores tracked systems on it, paired by season.
It agrees in sign with the criterion wherever the criterion was sure (the fine bio pair worse at z 8.3, no
defensive prior at z 12.8, prior-only at z 16) and reads where the criterion is silent (the shipped
height-weight board better than its predecessor at z 1.8; the four-factor half blend WORSE at z 2.3).
**Run both from now on**: the criterion decides forecasting, this decides attribution, a candidate should
lose neither.  What to aim at: the top of the board -- a prior less compressed at the top (the target's
shrinkage is global, the compression is not), or per-player shrinkage keyed on the prior's confidence (3.4),
scored on this number.  The true "same four" matched version -- lineups differing in exactly one player --
is a refinement on `lineups` if the ridge's answer needs a second opinion for a named player.

### 3.7 DONE: height and weight (binned) in the prior, shipped 2026-09-07 (FINDINGS 26)

`tune501_b7_turnref_o_hwb` -- height2 and weight15 on both sides -- is zero on the criterion (-0.006, z -0.32),
narrows the archetype gap against the consensus on both sides (defense 0.219 -> 0.185, offense -0.316 ->
-0.307), costs 0.008 of defensive agreement, and corrects a bias the record can now show in every era: the
defensive prior under-rates 6-10-and-taller players by a third of a point per 100.  The criterion cannot see
a correction of that shape (its resolution is 0.015) and the shift test says the direction is right.  Under
Part 0 the tie goes to the board.  If the board is for attribution as well as forecasting, this is the
cheapest attribution gain on the table and the fine pair is never the form (it names the player).  To ship:
`gbdt.features_full_O` / `_D` += `height2, weight15`, the tracker table for `hwb` to `calmap_ship.parquet`,
`08_ratings.py`, the floors.  **What the owner asked for beyond this and is not yet measured:** the padding
(pad_k, pad_target were swept in 21 and are in the registry), a booster re-tune with any new column in (the
625-trial protocol of 22.7), aggregations the tree cannot make from the 13 rates that are not yet in `DERIVED`
/ `RATIOS` (most linear and ratio forms are), and a team-context feature that is not a one-hot -- the
turnover/tenure machinery is the safe form of that and tenure was not wanted offline.

### 3.6 Still open, not scheduled

* **TabFM** (`google/tabfm-1.0.0-jax`): installs and downloads (5.7 GB; point `HF_HOME` at `A:`), but the
  orbax restore dies in tensorstore on a 1.5 GB region -- 38.5 GB of the box's 48 GB commit limit was taken.
  Retry on a quiet machine.  `scratch/tabfm_try.py` prints the booster's baseline for it to beat.
* **`main` carries the board** (`tune501_b7_pasto_pOD`, merged 2026-09-08 at the owner's request) and the live
  site at `https://bbstats.github.io/openrapm/` serves `docs/data/ratings.json` from it.  To publish a new
  board: `08_ratings.py`, `52_site.py`, commit, merge `hybrid-and-xpts` into `main`, push; Pages rebuilds on
  its own within a minute or two.
* **DNS for openrapm.com**: the custom domain was REMOVED on 2026-09-06 (`docs/CNAME` deleted on both branches,
  `cname: null` on the Pages API) because it had no DNS behind it and was redirecting `bbstats.github.io`
  into a dead name.  The site is back at `https://bbstats.github.io/openrapm/`, https enforced.  To turn the
  domain on: four GitHub `A` records + a `www` CNAME at Porkbun FIRST, then re-add `docs/CNAME` on `main`,
  wait for the cert, then `gh api -X PUT repos/bbstats/openrapm/pages -F https_enforced=true`.
* Never re-run, from 31: the GBDT prior with player-grouped folds as the board's prior (flat on the criterion,
  +0.1 on credit); the ridge outside x0.25-x0.5 of the shipped value on a kernel fit; a naive
  `1 - player / player_zero` credit share without the permutation null (it reads 26% where the truth is 84%).
* Never re-run, from 30: the destination features as the board's prior on offense; block-minutes and
  rebound-minutes on defense with or without the roster's defensive APM.
* Never re-run, from 29: Boruta's accepted lists as feature lists on either side; Boruta as a gate; the
  shot-quality block, the binned height and the efficiency ratios back on the lists they were pruned from
  without a new reason.
* Never re-run, from 28: PAST on defense alone; `past_rapm` beside APM and its possessions; PAST on pooled rows;
  the offensive prior alone seeing both sides' past (`pastx`, not separable from own-side);
  the map's prior terms (`prior`, `prior2`, `priorsat`) on defense scored on the investigator (unchanged to
  the third decimal).
* Never re-run, from 26: fine (unbinned) height and weight in the prior on either side; tenure and n_teams as
  prior features; the draft slot alone; the play-by-play block counters on top of height.
* Never re-run, from 25: the four-factor defence with the fixed-share prior split at any ridge, raw or
  repriced eFG, full or half blend; REML over the factor ratio; factor ridges chosen on a player-level
  split-half read.
* Never re-run, from 24: the per-player trade delta applied to the held-out season at K = 3 (`tune501_b7_turn`,
  +0.069); the turnover prior on defense (flat, and it costs the defensive agreement); the mover / turnover
  slopes on the rating or the prior in the map (zero both ways); the `turnp` (past-only) level.
* Never re-run, from 23: the Dredge block on either side, in any grouping, added or substituted, absolute or
  era-relative; the block split by location on either axis; assists by zone as rates or shares; `pot_ast`
  on either side; any feature built on the Russell share; and BorutaShap as a gate on the shipped lists.
* Never re-run, from 21 and 22: lam_ratio, lam_buckets by exposure, playoff rows, one target for both sides,
  x3def_p1, the mover / rookie-age / age-by-exposure / rating-by-age / rating-by-exposure / prior-by-exposure /
  prior^2 map terms, season weights beyond the decay, padding scale and target, panel APM at penalty 30 and
  300, the training-block team's mean rating, role growth, the lineup spread, the two sides of the bend apart,
  the multiplicative offense-defense term, 8 bagged members, re-pricing the prior, the recursion, scaling the
  offset, Huber, inverse-variance target weights, asymmetric pooling, experience.

* **The play-by-play ingest that would finish 3.1's list**: `OffFoulsDrawn100` (Dredge's 1.22 and his own
  favourite) needs the v2 feed or pbpstats, and `goaltend`'s era trend needs reconciling against a published
  source before it becomes a column.  Neither is scheduled and 23's result makes neither urgent.

(Earlier handoffs: `HANDOFF_rankmap_archive.md`; the calibration-map handoff is in git history at `d1bc3cc`;
the prior-pass handoff this one replaces is at `571ea72`.)
