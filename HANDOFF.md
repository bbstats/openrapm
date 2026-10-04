# Handoff: the single-year player rankings, one experiment at a time

**This file is transient.**  It starts the next session and is deleted when Phase 1 ships.  `DECISIONS.md`
is the permanent record and carries every number quoted here.  Do not let this grow into a lab notebook.

Branch `cleanup`; `main` is fast-forwarded to it at each publish, and the live site is served from `docs/`
on `main`.  `pytest -q`: **382 passed, 1 failed, 1 xfailed, ~120 s** with the scraped data.  The failure is
accepted and named at the bottom of this file; nothing else is red.

## START HERE (2026-10-03 evening): the within-season calibrator, experiment 33

The owner asked for a calibrator of the ratings' misses that does not shrink peak seasons, trained on each season's
own held-out games (2017-2026).  Built and run: `scripts/97_within_season.py` (rate from 3/4 of a season, every
input rebuilt from those games; `WITHIN_SEASON_LEDGER.md` lists what is held out and how it is checked) and
`scripts/98_calibrator.py`.  Full record: DECISIONS.md "Experiment 33".
- **One multiplier per side** (offence x0.75, defence x0.97-0.99, fitted on held-out games of other seasons):
  year-over-year **8.648** vs 8.666 (z -4.1), order unchanged, better in every quality tier; consensus unchanged;
  2026 offence-first stars drop (Curry 21st, Harden 42nd, Doncic 12th).  Superseded by 33b.
- **The trees on 78 inputs**: win within season (z -9.5 over the multiplier) but the order on neighbouring seasons is
  worse (z +3.3): they learn the team-season's context.  Not a candidate.
- **33b**: the part that comes back to the mean is the box-score PRIOR part (offence 0.70 on held-out games; the
  games' part holds up, 1.21).  Shrinking the prior part only (offence x0.71, defence x0.95): year-over-year 8.648
  (ties the one multiplier, z -4.1 vs incumbent), better within season (z -3.4, 9 of 10), consensus 0.840 (from
  0.836); 2026 Curry 24th, Harden 50th.  **ADOPTED for all thirty seasons (the owner: "Good, adopt for now"):
  the new incumbent `outputs/season_ratings_priorshrink.parquet`, 8.600 (z -12.2, 53 of 56); the product and the
  site are LIVE (commit 7e5bdcc, pushed to cleanup and main 2026-10-03, the owner: "Yes go live").**
- `tests/test_vs_consensus.py::test_star_guards_are_not_buried` fails on the product: LaMelo Ball 187th against
  a ceiling of 160.  It ALREADY failed on the 2026-10-01 product (162); the shrink moves him further, as it
  moves every offence-first guard (Trae Young 205th, Booker 85th, Curry 43rd, all inside their ceilings).
  CI reads the frozen artifact and passes.  Re-basing the ceiling is the owner's call.
- Not run yet on the candidate: trade loss (70 + 73), swap test (90).  Open: the multiplier for a WHOLE season
  (fitted on 3/4-season ratings; half-season folds would show how it moves with evidence).
- Rerun: `62 --exclude_neighbours=1 --score=0 --boards=2017,...,2026 --out=season_ratings_within_base
  --save_models=within`, then `97 --models=within --base=season_ratings_within_base`, then `98 --out=calibc`;
  splice + 91 + 63 as in `scratch/2026-10-03_within/` (`splice.py`).

## Earlier thread: the prior's model algorithm (2026-10-02, updated 2026-10-03)

The owner: *"let's focus on improving the prior, this time the model algorithm"*.  Their list, in order:
1. linear / elastic net, 2. chimeraboost hyperparameter tune, 3. an ensemble of models, 4. bagging
(chimeraboost `quality=4` = `n_ensembles=5`, `quality=5` = 8; `quality=3` = the default, no bag).
They said do FIRST: a stacking regressor, plus-minus -> elastic net, everything else -> Boruta -> chimeraboost.

**Experiment 31, the stack: DONE, NOT ADOPTED** (row in the record table below; the owner has the emailed summary).

```python
prior = StackingRegressor(                       # src/eracoef/stackprior.py: StackedSPM
  estimators=[("linear", ElasticNetCV(l1_ratio=[.1,.5,.9,1], cv=GroupKFold(5) by player) on PLUS_MINUS),
              ("booster", ChimeraBoostRegressor(quality=3, **cfg params) on the other 50 + chunk features)],
  final_estimator=LinearRegression(positive=True),   # intercept, sample_weight = row weight
  cv=the 5 player folds the prior already uses)      # fold fits = the out-of-fold preds; blend fit once
PLUS_MINUS = ONC + OFFC   # on- AND off-court (the owner insisted on off-court; net = on - off is implied)
```
- Run: `62 --features=stack --stack=1 --stack_quality=3`; chain `scratch/2026-10-02_stack/stack_chain.sh`
  (NAME=stack_q3); log `outputs/stack_q3_chain.log`; tables `outputs/season_ratings_stack_q3{_raw,}.parquet`.
- Booster list: `scripts/92_stack_boruta.py` (50 trials on `outputs/prior_rows_stackpool_2026_{O,D}.parquet`,
  dumped with `62 --features=sy --dump_rows=stackpool`) kept ALL 50 (`outputs/csv/boruta_stack_table.csv`) ->
  `singleyear.STACK_BOOSTER_O/D` = PRIOR_FEATURES minus ONC.
- Fitted (2026; same to 2 decimals every season): elastic net O `onc_o .247, offc_o -.072`; D `onc_d .692,
  offc_d -.213` (defence label = points allowed); blend O `-.064 + .571 lin + .655 boost`, D `.019 + .834 lin +
  .378 boost` (weights sum 1.22); out-of-fold rmse O lin .637 / boost .636 / stack .545, D 1.035 / 1.407 / .953.
  The ridge's prior_scale medians: incumbent O 2.09 / D 0.91, stack O 1.67 / D 0.48.
- Results vs incumbent: year-over-year 8.677 (z +1.8, 23/56); stint z +8.4, order-only z +12.7; swap test
  order worse (off z -6.8, def z -4.8); trade loss off z +5.0 / def z +7.8, worse in every tier but the top 30;
  top 30 better on year-over-year (z -2.7); consensus .876 / .859 total / def (from .836 / .813).  2026: Edey
  (590 poss) 8th, Keshad Johnson (621) 12th, Caruso 5th; Queta, Clingan, Harden out of the top 20.
- Timing: quality=3 2.2 min a season (~70 min for 30); quality=5 10 min a season (5 hours).

**Why it lost, measured 2026-10-03** (the owner picked option a; `scripts/93_prior_oof.py`; full table in
DECISIONS.md, "Experiment 31").  2026's training rows with outside labels, every prior out of player fold, scored
on the one-season rows (the rated row's shape) against a label that shares no game with them:
- the stack as built is far worse than the shipped booster: offence 0.754 vs 0.667 (z +11.7), defence 1.696 vs
  1.509 (z +12.3).  Most of that is width -- its elastic net's one-season predictions are twice too wide (slope
  0.49 / 0.46) and the blend gives it 57% / 83% -- but not all: with every prior rescaled by its own best line
  (the build fits the prior's scale, so order is what counts) the stack is still worse, 0.688 vs 0.657 and 1.495
  vs 1.464, z +3.2 / +3.2.  The logged 0.545 was on career labels, where the shipped booster had 0.506 anyway.
- trained on outside labels (chunk rows only), the stack ties the shipped booster on offence and beats it on
  defence: 1.443 vs 1.509 raw (z -5.1), 1.438 vs 1.464 rescaled (z -2.3).
- the screen's score from now on is the RESCALED error (`94_tune_booster.py`); raw rmse rewards a narrow prior.
- a screen, not the test: experiment 25 used outside labels in a build and lost the year-over-year test.

**The owner picked (a), chimeraboost tuning then bagging, screened before any build -- experiment 32** (DECISIONS.md
"Experiment 32").  `scripts/94_tune_booster.py` (60 Optuna trials a side on the rescaled screen) found settings that
beat the shipped booster on both sides and both player splits: offence 0.6568 -> 0.6424 (z -2.8 / -2.9), defence
1.4641 -> 1.4213 (z -7.3 / -6.0).  On defence most of it is one change: chimeraboost's early-stopping split holding
out whole players (`groups=`) instead of random rows (z -4.5 / -4.0 alone).  Bags add little.  **The owner then ruled one
holdout method on both sides -- whole players** ("we are not going to use different holdout methods for statistical
reasons that are 100% identical"); tune1's offence had come out on rows, so its build was stopped.  **tune2**
(`94 --split=players`, both sides; log `outputs/tuned2_chain.log`) **failed my gate on offence**: defence z -7.1 /
-6.5, offence only z -1.0 / -1.0 (bag of five), so nothing was built.  Asked the owner: (a) build both sides tuned
anyway, the test decides (recommended); (b) defence tuned, offence on its shipped settings with whole players held
out; (c) stop.  To build (a): write `booster_params_tune2b.json` as `gate_tune2.py` would and run
`tuned2_chain.sh` from its build loop.  Not chosen: (b) the
defence-only stack on outside labels (~5.5 h build).

**Uncommitted** (nothing committed since 2026-10-01): `src/eracoef/stackprior.py`, `tests/test_stackprior.py` (4
pass), `scripts/92_stack_boruta.py`, `scripts/93_prior_oof.py`, `62` (`--stack`, `--stack_quality`, and a fix:
`--dump_rows` with career labels crashed in a same-team print), `singleyear.py` (`STACK_BOOSTER_*`, `"stack"` set),
this file, `DECISIONS.md` (now with experiment 31), `scripts/77-82`, and `singleyear.py`'s `boruta_onoff` /
`boruta_net`.
The owner wants clock times in 12-hour American format.

## How we work (the owner, 2026-09-13/14)

- **Two jobs only: implement the owner's ideas, or propose new ones.**  Never run an idea the owner has
  not said "go" to.  One experiment at a time; never a queue of several overnight.
- **One test decides**: the year-over-year test below.  The consensus checks, the trade loss and the 2026
  top 20 (the eye test) are read every time and can veto; they are never fitted to.
- **Every experiment ends with the 2026 top 20** (`/experiment-comparison`; `scripts/66_compare.py`).
- **Plain words.**  No invented labels, no single-letter names, no "board" (say the player rankings), no
  "floor" (say the check).  Define a term the first time it is used.  End every message with a
  one-sentence TL;DR unless the message is itself one sentence.  **Movement is reported in points per 100,
  never in rank places**, and under 0.1 is nothing.
- **On the site and in anything published the ratings are "OpenRAPM"** -- never "ours" or "we" -- and the
  published pages are American English.  The repo's own prose is British; leave it.

## The rulings that bind

1. **One rating per player per season, from that season's games only**, regular season and playoffs.
2. **Single year or bust**: a player's own other seasons may not reach his rating, not even through the
   booster's memory (hence the out-of-player priors).  Model coefficients may be learned from other seasons.
3. **Never train on the current season until its Finals are over.**  Loading is always allowed.
4. **The SPM is trained on one row per player** (his career average with the rated season out) PLUS chunk
   rows of his seasons; not one row per player-season.
5. **Keep chimeraboost.  The consensus is a sanity check, never a fitting target**; a gross miss vetoes.
6. Each season is centred at possession-weighted zero per side (the RAPM convention), 2026-09-14.

## What ships (the incumbent, `scripts/62_single_year_board.py` defaults)

1. **Target**: `looseason.LeaveSeasonOutRAPM`, one RAPM per player over every season except the rated one
   (penalties 40,000 / 40,000 / 0, closed).
2. **SPM**: chimeraboost on `singleyear.chunk_rows` (the career row plus contiguous 1-, 2-, 3-season
   chunks, two features saying how much evidence a row rests on), features `boruta` (21 offence /
   17 defence, on-court columns in).  **Out-of-player**: five player folds balanced on the label, every
   player's prior from the fit that never saw his rows.  **The DEFENSIVE label is un-shrunk and the
   offensive one is not** (`--unshrink_label=def`, the default, adopted 2026-09-18).
3. **Rating**: `priorridge.PriorRidgeCV`, `scale x prior + residual`, the scale priced on cross-fitted
   prior columns (`--crossfit=scale`), the residual penalty fixed at **13,037** on both sides.  Then centred.
3b. **The prior shrink** (adopted by the owner 2026-10-03, "Good, adopt for now"; DECISIONS.md, experiment 33b): the
   box-score prior part of each rating times a held-out multiplier per side (about 0.72 offence, 0.95 defence),
   the games' part unchanged, re-centred -- `scripts/99_prior_shrink.py`, fitted on the within-season folds of
   2017-2026 (`scripts/97_within_season.py`, `WITHIN_SEASON_LEDGER.md`); earlier seasons borrow the pooled value.
4. **The swap adjustment, team version** (adopted by the owner 2026-10-01; DECISIONS.md, "The swap adjustment
   with the spread held"): credit inside each team is re-split by lineup swaps -- a type model at half strength,
   learned from other players and other seasons -- with each team's total fixed and the spread inside teams held
   to step 3's.  A post-hoc step on the finished table, `scripts/91_swap_adjust.py`, not part of script 62.

`outputs/season_ratings_product.parquet` (every other season allowed in the prior) is what
`docs/data/ratings.json` and the site are built from.  Since 2026-10-03 it is the prior-shrunk, swap-adjusted table:

    .venv/Scripts/python scripts/99_prior_shrink.py --base=season_ratings_product_pre_swap --out=season_ratings_product_priorshrink_pre_swap --rule=product
    .venv/Scripts/python scripts/91_swap_adjust.py --base=outputs/season_ratings_product_priorshrink_pre_swap.parquet --kappas=0.5 --taus= --hold_spread=within --exclude_near=0 --score=0 --tag=product_priorshrink_swapadj
    copy outputs/season_ratings_product_priorshrink_swapadj.parquet over outputs/season_ratings_product.parquet
    .venv/Scripts/python scripts/52_site.py

where `season_ratings_product_pre_swap.parquet` is steps 1-3 (the table as built on 2026-09-18).  The table before
the shrink is kept as `outputs/season_ratings_product_before_priorshrink.parquet`.
**The incumbent every candidate is scored against is now `outputs/season_ratings_priorshrink.parquet`** (8.600 on
the year-over-year test; adopted 2026-10-03): `season_ratings_unshrinkdef.parquet` (steps 1-3 at
`--exclude_neighbours=1`) through `99_prior_shrink.py --rule=test` (each season's multiplier fitted outside it and its
neighbours) into `season_ratings_priorshrink_raw.parquet`, then 91 with `--kappas=0.5 --taus= --hold_spread=within`
and the default `--exclude_near=1`, which keeps both scored seasons out of the type model.  The previous incumbent,
`outputs/season_ratings_swapadj_within.parquet` (8.666), is kept.  **A candidate that changes steps 1-3 must go
through the same 99 and 91 commands before it is scored**, or it is compared without the steps the incumbent has.
Caveat: 99 reads the multipliers off the incumbent's own within-season folds; a candidate that changes the prior
changes how far its prior part holds up, so a serious one needs its own folds (62 `--save_models`, then 97).
`artifacts/season_ratings.parquet` is the older system, kept for the tests; do not overwrite it.

Threads: the script pins BLAS to one thread and numba to four before importing anything.  **Never run two
builds at once**, and kill a chain's python children when you stop it.

## The test: year-over-year

Rate a season from its own games.  Predict every stint of the season before and after from the ten players'
ratings alone, refitting only the intercept and home edge on the scored season.  28 scored seasons, each
predicted twice, 56 observations.  The prior must not have seen the two scored seasons:

    .venv/Scripts/python scripts/62_single_year_board.py --exclude_neighbours=1 --score=0 --boards=<ten seasons> --out=season_ratings_<name>_<first>
        (three chunks of ten seasons, stitched BY HAND into outputs/season_ratings_<name>_raw.parquet: concatenate
         the three parquets; ~3 min a season, so ~95 min for thirty.  With --chunk_label=outside, ~11 min a
         season: 280-310 extra RAPM solves each)
    .venv/Scripts/python scripts/99_prior_shrink.py --base=season_ratings_<name>_raw --out=season_ratings_<name>_shrunk_raw --rule=test
    .venv/Scripts/python scripts/91_swap_adjust.py --base=outputs/season_ratings_<name>_shrunk_raw.parquet --kappas=0.5 --taus= --hold_spread=within --tag=<name>
        (the swap adjustment the incumbent carries; writes outputs/season_ratings_<name>.parquet, ~2 min)
    .venv/Scripts/python scripts/63_yoy.py --rankings=<name>=outputs/season_ratings_<name>.parquet,incumbent=outputs/season_ratings_priorshrink.parquet --ref=incumbent --tag=<name> --splits=
    .venv/Scripts/python scripts/64_consensus_report.py outputs/season_ratings_<name>.parquet outputs/season_ratings_priorshrink.parquet
    .venv/Scripts/python scripts/66_compare.py incumbent=outputs/season_ratings_priorshrink.parquet <name>=outputs/season_ratings_<name>.parquet --season=2026 --top=20 --yoy=outputs/yoy_<name>.parquet --ref=incumbent
    .venv/Scripts/python scripts/90_swap_test.py --rankings=incumbent=outputs/season_ratings_priorshrink.parquet,<name>=outputs/season_ratings_<name>.parquet --contexts=nofatigue --checks=0 --tag=<name>
        (the swap test: does it order teammates better; 40 s)

**Decision rule:** adopt only if `z` is -2 or below with no gross consensus miss and the 2026 top 20 not
worse; ties go to the simpler version.  Read the row "each side rescaled to the scored season" to see
whether the ORDER improved or only the spread.  `scale_*` below 1 on a neighbouring season is expected
(players change year to year) and is not a target.

**Saved priors**: `--save_priors=<name>` then `--priors_from=<name>` reruns the ridge alone in a second a
season, which is how the penalty was swept.

## The second criterion: the trade loss (new, 2026-09-18)

The year-over-year test is an error per team-game, so a 200-possession man is a rounding error in it.  The
trade loss counts every player once.  **It has already overturned one reading and confirmed six others.**

    .venv/Scripts/python scripts/70_tradeset.py --rankings=outputs/season_ratings_<name>.parquet --team_effects=team --out=tradeset_<name>   (85 s)
    .venv/Scripts/python scripts/73_tradeloss.py --alphas=incumbent=outputs/tradeset_priorshrink_alpha.parquet,<name>=outputs/tradeset_<name>_alpha.parquet --ref=incumbent

Paired by season on the exact intersection of eligible players; `--tier=` splits by exposure.  Read
`mean_diff` below zero as better and `z` against its own standard error.  It detects a defensive change of
0.0024 at z -2.2, so it has power at the size that matters.

**Every experiment is also split by player-quality tier: top 30, 31-90, 91-150, 151-300, 301+** (the owner,
2026-09-30: "in general we should do this").  Both tests, tiers set by the incumbent's rank in the rated season:

    .venv/Scripts/python scripts/88_yoy_by_player.py --cands=<name>=outputs/season_ratings_<name>.parquet --tag=<name>
        (the year-over-year difference shared among the players on the floor and summed by quality tier, by team
         change and by age; the groups add up to 63_yoy.py's number exactly; ~3 min)
    .venv/Scripts/python scripts/73_tradeloss.py --alphas=<as above> --ref=incumbent --quality=outputs/season_ratings_priorshrink.parquet --tier=each
    (88 takes the incumbent with --ref=incumbent=outputs/season_ratings_priorshrink.parquet)

Direction names in both year-over-year scripts: "prev" / "the PREVIOUS season's rankings predict this season"
is a rating looking FORWARD at the season after it; "next" is a rating looking BACK at the season before it.

## The record so far (year-over-year error per team-game, points per 100)

| rankings | error | verdict |
|---|---|---|
| shipped before this branch (`artifacts/`) | 8.587 | uses the banned `past_*` channel on offence |
| career row only, one per player | 8.804 | the starting point |
| + 1-3 season chunks (owner's design) | 8.736 | adopted |
| + scale cross-fitted, penalty not | 8.693 | adopted |
| + out-of-player priors | 8.705 | worse on the test, adopted by ruling 2 |
| + fixed penalty 13,037 | 8.697 | tie on the test; **the trade loss later read it z -2.19 on defence** |
| + the DEFENSIVE label un-shrunk (the incumbent until 2026-10-01) | **8.682** | z -5.24, 43 of 56; trade loss defence z -7.14, 27 of 30; adopted 2026-09-18 |
| 25: chunk rows labelled from the seasons OUTSIDE the chunk | 8.700 | z +5.0; trade loss z +4.5 / +5.0; rejected 2026-09-29 (25b, weighting those rows by their label's possessions: much worse) |
| 26: + RAPM pieces, same-team measure, game difficulty, rated as if every player changed teams | 8.682 | a tie (z +0.1); trade loss z +3.5 / +5.4; not adopted |
| 27: + a team-season random intercept | 8.679 | a tie (z -0.4); trade loss z +3.8 / +3.9; not adopted.  26 and 27 predict the season BEFORE better (z -2.5, -2.8) and the season after worse: an age tilt (veterans up, young players down), not a team-change effect |
| 28: + every chunk's label moved to its own age (aging curve) | 8.721 | z +4.1; trade loss z +5.1 / +4.1; rejected: over-corrects (Curry 92nd, Durant 136th), though consensus agreement rises to 0.818 |
| 29: the incumbent + the team-movement weight, floor 0.5, rebalanced within career bands | **8.669** | **z -4.1, 39 of 56**; trade loss a tie; consensus unchanged; 2026 top 20 close to the incumbent's.  Meets the adoption rule; awaiting the owner (it halves one-team players' weight, which they said they do not want) |
| 30: windows of two, four or six seasons split in the middle, each half labelled by the other; same-team measure as an input; rated as if every player changed teams | 8.706 | z +3.4, 16 of 56; trade loss z +2.8 / +7.6; rejected 2026-10-01.  Order alone leans better (z -1.6) but the ratings are too wide; best consensus agreement yet (0.829 total); loses as much on players who changed teams as on those who stayed.  LeBron James 126th in 2026 |
| 30b: experiment 30's models with the trade flag switched on only for the players who changed teams; and every player rated as if he stayed | 8.682 (matched) / **8.680** (stayed) | the flag does not help movers: matched against all-stayed a tie (z +0.8).  All-stayed against the incumbent: a tie (z -0.3) but stint level z -5.1 (mostly the bench), trade loss offence z -4.4 / defence z +4.1, best consensus yet (0.864).  Not adopted under the rule; owner's call.  Durant 72nd, LeBron 65th in 2026.  `62 --rate_same_team=both`, `89_stitch_by_move.py` |
| the swap adjustment, by player type at half strength (2026-10-01) | 8.698 | z +3.5, 16 of 56: worse at native scale; each side rescaled -1.02, z -10.0, 52 of 56 (the order is much better, the spread ~12% / 27% wider); swap test order +0.16, z +5.5; consensus 0.832 (from 0.794), top five 5 of 5; trade loss offense z -5.3.  Not adopted under the rule; next proposed: the same with each side's spread held to the incumbent's |
| the swap adjustment with the spread held to the incumbent's: teams_fixed / list_scaled (2026-10-01) | **8.666 / 8.652** | **z -5.2, 39 of 56 / z -7.7, 48 of 56**; better in every quality tier; consensus 0.836 / 0.834 (from 0.794); swap test order z +4.3 / +4.6 and gaps better; trade loss offense z -5.7 / -5.3, defense z +2.0 / a tie.  Both meet the rule.  **teams_fixed ADOPTED 2026-10-01 (the owner: "team version") -- the incumbent**, `outputs/season_ratings_swapadj_within.parquet`.  `91_swap_adjust.py --kappas=0.5 --taus= --hold_spread=within|whole` |
| all four on- and off-court ratings in both priors, on top of the incumbent (2026-10-01) | 8.667 | a tie (z +0.4); order-only worse (z +2.6); consensus 0.814 (from 0.836); Ajay Mitchell 9th to 7th, Caruso and Hugo González up ~0.9.  Not adopted.  `--features=boruta_onoff` |
| on/off alone in both priors, in place of the raw on-court ratings (2026-10-01) | 8.704 | z +6.9, 12 of 56: worse (order a tie, the offensive spread wider); trade loss offence z +5.6; Ajay Mitchell 9th to 6th.  Not adopted.  `--features=boruta_net` |
| the prior as a stack (2026-10-02): on- and off-court plus-minus into an elastic net, the other 50 columns (Boruta kept all, `92_stack_boruta.py`) into chimeraboost quality=3, blended on out-of-player-fold predictions | 8.677 | z +1.8, 23 of 56: worse; stint level z +8.4, order-only z +12.7; trade loss offence z +5.0 / defence z +7.8, worse in every tier but the top 30; swap test order worse (offence z -6.8, defence z -4.8); consensus 0.876 total / 0.859 defence (from 0.836 / 0.813); only the top 30 better (z -2.7).  2026: Zach Edey (590 poss) 8th, Keshad Johnson (621) 12th, Caruso 5th; Queta, Clingan, Harden out of the top 20.  Not adopted.  `62 --features=stack --stack=1 --stack_quality=3` (`stackprior.StackedSPM`).  Why (2026-10-03, `93_prior_oof.py`): trained on career labels its elastic net's one-season predictions are twice too wide, and even rescaled the stack orders players worse than the shipped booster (z +3.2 / +3.2) |
| 33b: the box-score prior part times a held-out multiplier per side (offence x0.72, defence x0.95), fitted on each season's own held-out games (within-season folds 2017-2026; 1997-2016 borrow the pooled value) (2026-10-03) | **8.600** | **z -12.2, 53 of 56**; every era better (1998-2006 18 of 18, 2007-2016 19 of 20, 2017-2025 16 of 18); swap test order z +3.1 (40 of 56), gaps z -6.2; trade loss offence z -3.1, defence z +5.5 (0.0007 on 0.248); stint level with each side rescaled z +2.1; consensus 0.834 (from 0.836), top five 5 of 5.  2026: Curry 21st, Harden 31st, Durant 41st (product).  **ADOPTED 2026-10-03 (the owner: "Good, adopt for now") -- the incumbent**, `outputs/season_ratings_priorshrink.parquet`.  `99_prior_shrink.py`; the trees of experiment 33 not a candidate |
| rejected | | every chunk size; booster settings; off-court features; one row per player-season; cross-fitting the penalty; the un-shrunk label on BOTH sides (the 2026 offensive spread collapses 1.60 to 1.14, Curry falls to 61st); `onc_d` off the defensive list (the trade loss finds nothing) |

## Open with the owner (2026-10-01)

0. **Shipped 2026-10-01: the swap adjustment, team version** -- the owner's goal is ranking players among their
   own team, judged by "player vs replacement in lineups".  The new instruments are the swap test
   (`90_swap_test.py`) and the adjustment (`91_swap_adjust.py`); every later candidate goes through 91 before it
   is scored (see "What ships").  Open from it: the time-on-court term measured an artifact, not fatigue; each
   player's own swaps added nothing on top of the type model; the defensive trade loss of the shipped version is
   slightly worse (z +2.0, players ranked 151-300).
1. **Experiment 29, the team-movement weight** (`--trade_weight=0.5 --trade_bands=1`): the only change that
   passes the rule (z -4.1), and it wins equally for traded and not-traded players.  It only reweights the box
   prior's training rows -- one career-wide number per player (`1 - sum(share ** 2)` + 0.5, so a one-team
   player counts half), nothing at rating time.  The owner does not want one-team players halved; not adopted.
2. **The stayed list** (experiment 30b): a tie on the test, better at stint level and on consensus, defensive
   trade loss worse, Durant 72nd and LeBron 65th in 2026.  The owner's call.
3. **Proposed next, none started:** experiment 31, experiment 30 rated as if traded plus `age` as an input (the
   owner's idea; years of experience is already in, `age` never has been); weighting each label by its own
   precision instead of team count (the non-blind version of 29); and a one-season check of whether the
   veteran penalty in the traded list comes from the window weights (rows weigh by the feature half's
   possessions, so traded veterans' windows lean 57% toward "season 1 predicts season 2").

The chains and one-off scripts behind experiments 25-30b are in `scratch/2026-10-01_experiments_25_to_30b/`
(gitignored; they still point at the session's old temp folder for `movement.py`).

## THE FINDING to carry forward: OpenRAPM is about a sixth too wide, and three sources agree

Full numbers in `DECISIONS.md`, "The amplitude finding".

| source | offence | defence | what it measures |
|---|---|---|---|
| the consensus, GLS scale | x0.85 | x0.85 | the same season, against public metrics |
| the trade set, three-season window | x0.749 | x0.919 | adjacent seasons' team-games |
| the year-over-year sweep, interior optimum | x0.65 | x0.85 | the neighbouring seasons' games |

**One sign, three independent readings.**  There is no offence/defence imbalance -- an earlier claim of one
was an artefact of the consensus being scaled to EPM, withdrawn the same day.

**A rescale is still not adoptable** (experiment 20): the two across-season sources cannot separate "too
wide" from "players regress", and a uniform rescale reorders nobody -- the order-only row is zero to 4e-14.
So this wants fixing INSIDE the fit, not bolted on afterwards.

**The lead that goes with it: the high-usage creators.**  Six of the eight highest-usage players sit below
their consensus offence after the spreads are matched -- LaMelo Ball -1.70, Ja Morant -1.61, Luka -1.59,
Jokic -1.50, Booker -1.06, Giannis -0.83 -- while the correlation with usage over all 391 players is
**-0.00**, so it is the extreme top and not a gradient.  The trade set independently found **shot creation,
three-point rate and offensive rebound share are under-credited by the box prior**, and shot creation is
what this group is.  Not the explanation, measured and dismissed: age, experience, team quality; the whole
per-36 box profile explains 21% of the player-by-player gap.

**Also named and unexploited: on defence the prior reads team offence as defensive credit** -- a player
whose team scored while he was on the floor has his defensive correction pushed down 0.062.

## What is closed.  Do not reopen without a new reason.

- **The exposure correction** (the prior is flat in exposure, the truth is steep).  Six attempts, none beat
  the incumbent; the owner: *"whatever our version is as last posted seems to be good enough."*
- **The trade set as a product.**  Alpha reads neighbouring seasons, so it can never be published under
  ruling 1, and a season's statistics see 3.7% of it on offence and 5.6% on defence.  It is an instrument.
- **`onc_d` off the defensive feature list** (experiment 19): the trade loss finds nothing (z -0.40) where
  it detects the penalty change at z -2.19, and agreement fell against every public defensive metric,
  furthest against the luck-adjusted ones, so it was not the `def3` pattern that excuses a drop.  Ruled:
  keep `onc_d`.  Note the 2026 top 20 was NOT the reason -- to the eye it was arguably better.
- **The amplitude as a rescale** (experiment 20).  See above.
- **The trade flag as a rating-time switch** (experiments 26, 30, 30b).  Rating everyone as if traded loses;
  switching it on only for the players who really changed teams ties switching it off for everyone (z +0.8).
  The flag learns who gets traded (veterans, often declining), not what a trade does.

## The instruments, all read-only

| script | what it answers |
|---|---|
| `63_yoy.py` | the criterion: team-game error, both directions, paired by season |
| `70_tradeset.py` + `73_tradeloss.py` | the per-player loss that counts a bench player once |
| `74_consensus_bars.py` | how far OpenRAPM sits from the consensus in units of its own uncertainty, each side scaled |
| `75_amplitude.py` | a per-side multiplier sweep on a finished table |
| `76_bias_groups.py` | the published page: bias by player type, `vs consensus` and `vs 2026 observed` |
| `64_consensus_report.py` | rank agreement and spreads for several tables side by side |
| `88_yoy_by_player.py` | the year-over-year difference split among the players on the court, by quality tier, team change and age; `tg_abs` gives each group its own typical miss |
| `89_stitch_by_move.py` | the trade-flag test's list: each player's "traded" or "stayed" rating (`62 --rate_same_team=both`) by what he did next; score it with `63_yoy.py` / `88_yoy_by_player.py` as `name=<fwd>|<bwd>` |
| `73_tradeloss.py --movers=1 --tier=each` | the trade loss for players who changed teams the season before or after, and for those who did not |
| `90_swap_test.py` | the swap test: does a ranking order teammates the way the games of the seasons either side do -- every pair of lineups that share four players, the two swapped players' rating gap against their swap difference (DECISIONS.md, "The swap test") |
| `91_swap_adjust.py` | the swap adjustment: re-split each team's credit by its lineup swaps (a type model, then each player's own swaps), team totals fixed; scores a grid on the swap test and writes the chosen arm (DECISIONS.md, "The swap adjustment") |
| `95_miss_by_group.py` | where the ratings miss most against expectation: each player's trade-set correction, adjusted for his rating level and the evidence behind it, by fifths of every statistic (plain names, `eracoef.glossary`); the owner's variance question, DECISIONS.md "Where the ratings miss most" |
| `93_prior_oof.py` | a prior's out-of-player-fold error on one season's training rows (dumped by `62 --chunk_label=outside --dump_rows=`), scored on the one-season rows against labels that share no games with them; ~6 min for four priors.  A screen, not the test |

`outputs/bgmm_proba.parquet` carries `player_id`, `season`, the winning player type and **all eight mixture
probabilities** for 2026, from a mixture fitted on 2023-2025 (one row per player-season) and used to place
2026 out of sample, standardised by the training seasons' own constants.  **The owner wants those eight
columns available to the prior code; that work has not started.**

Whether the two deltas on that page are related was the test the owner set for building position-level
categories: across the eight types, `vs consensus` against `vs 2026 observed` is **r -0.04** (the
2026-fitted types read +0.03).  No relationship, so position categories are not warranted by it.

## The site

`docs/index.html` (rankings, newest season first), `docs/bias.html` (bias by player type) and
`docs/decompose.html`, the "Decomposition" tab (vanilla RAPM split two exact ways: by player, on-court rtg + teammates + opponents + context +
ridge penalty; by possession, on court signal + off court adjustment (GP) + off court adjustment (DNP) + team SOS adjustment,
with the actual on and off court ratings beside them; single seasons
and the ten three-season windows, penalty 3,000).  Rebuild with `scripts/52_site.py`,
`scripts/76_bias_groups.py` and `scripts/83_decompose_site.py`, commit, and fast-forward `main` to publish.  The
tabs at the top and the dark-mode button are `docs/site.css` and `docs/site.js`, shared by all three pages (76
writes them into bias.html); a new page needs the same `<nav class="site">` block and its colours given again
under `:root[data-theme="dark"]`.  The
bias page is deliberately a title, one table and two short paragraphs; the owner has trimmed it twice.

## Traps that cost a day

1. **The consensus file was rescaled twice on 2026-09-18** -- once to make it a real consensus rather than
   one dominated by collinear raw-points metrics, once to turn off the scaling to EPM.  **No consensus
   spread or agreement figure is comparable across those changes.**  Both rescales moved `adj_*` for ~550 of
   582 players at correlation 0.99+, so a number can look like the same measurement and not be.
2. **`defense` in the rankings parquet is points allowed, negative-good.  `rating_def` is the positive-good
   column**, and `rating_total = rating_off + rating_def`.  Adding `offense + defense` gives a
   plausible-looking table with the defensive sign inverted.
3. **An uncentred table cannot be differenced against a centred one** without removing the level first.
   Anything built before 2026-09-14 predates the centring rule; 4b sits at +1.61 on offence in 2026.
4. **Stint level and team-game level disagree in this project**, and have given opposite signs on
   amplitude.  Say which one a number is.
5. `pip install -e` puts this working tree on `sys.path`: a fresh clone silently tests THIS repo's data.
6. Tests cannot reach the network (`tests/conftest.py`).
7. A score on a mask that cuts team-games is not a score; read stint-cutting splits on `mse`.
8. Two builds at once, or an unpinned numba pool, and a 15-second fit takes an hour.
9. `TaskStop` on a chain stops the bash wrapper only; its python children keep running.  Check
   `Get-Process python` before starting anything.
10. Seventeen more in `DECISIONS.md`, "The measurement traps".
11. chimeraboost 0.34 changed no rating but made `shap_values` exact interventional TreeSHAP (~200x slower):
    a Boruta trial on 33,000 rows took 30 minutes.  `gbdt_prior.run_boruta(explain_rows=1000)` fixes it.
12. Script 62 pins BLAS to one thread, so a 6,000-column label solve takes 5 s; outside labels need ~300 a
    season and are solved by Cholesky on `--label_threads` cores (0.6 s each) for that reason.

## The one failing test, accepted by the owner (2026-09-18)

`tests/test_vs_consensus.py::test_star_guards_are_not_buried` puts LaMelo Ball at rank 162 against a
hand-set ceiling of 160 on the rebuilt published table.  **In points that is 0.016 per 100** -- he sits
0.168, the player at rank 160 sits 0.184 -- six times smaller than the owner's threshold for nothing, and
the check is a rank ceiling, the instrument the owner has ruled does not measure size.  The owner: *"that
test can fail, no worries."*  The honest fix, not done, is to replace the rank ceiling with a points check.

## Where to start next

1. **Credit shot creation properly in the prior.**  The one live lead with two independent measurements
   behind it (the creator gap above, and the trade set's under-credited statistics).  A feature change,
   then the test and the trade loss like anything else.  One build.
2. **Split on-court plus-minus into its offensive and defensive halves** so the defensive fit cannot read
   team scoring as defensive credit.  Named mechanism, one build.
3. **The eight mixture probability columns into the prior**, which the owner has asked for.  The input is
   written; nothing is wired.
4. **A standard error per player for alpha**, from the inverse of the trade set's own matrix, so a
   correction can be read against its own noise.  Half a day.

## Verify you are where this file says

    .venv/Scripts/python -m pytest tests -q                                  # 382 passed, 1 failed (LaMelo, accepted), 1 xfailed
    .venv/Scripts/python scripts/63_yoy.py --rankings=incumbent=outputs/season_ratings_swapadj_within.parquet,ship=artifacts/season_ratings.parquet --ref=incumbent --tag=verify --splits=
                                                                             # incumbent 8.666
    .venv/Scripts/python scripts/64_consensus_report.py outputs/season_ratings_product.parquet
                                                                             # 0.848 / 0.813 / 0.833, spreads 1.04 / 1.05, top5 4
                                                                             # (before the swap adjustment: 0.787 / 0.803 / 0.795)
