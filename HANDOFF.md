# Handoff: the single-year player rankings, one experiment at a time

**This file is transient.**  It starts the next session and is deleted when Phase 1 ships.  `DECISIONS.md`
is the permanent record and carries every number quoted here.  Do not let this grow into a lab notebook.

Branch `cleanup`; `main` is fast-forwarded to it at each publish, and the live site is served from `docs/`
on `main`.  `pytest -q`: **543 passed, 1 failed, 1 xfailed, ~175 s** with the scraped data.  The failure is
accepted and named at the bottom of this file; nothing else is red.

## START HERE (2026-10-07): next is averaging the box-score prior over several training groupings

**Everything through experiment 42 is committed on `cleanup`; `main` has not moved and the site is not published**
(the Portable column is still the owner's call, below).  Commit only when the owner asks, and never name a model in
a commit.  A session continued on another machine stashes uncommitted work: `git stash apply` and keep the stash
(memory "Teleport auto-stash").

### The next job (the owner, 2026-10-07: "handoff.md will handle the avging box score prior thingie")

**Why.**  Each player's box-score prior comes from a model that never saw him: the prior is fit five times, each
time leaving out a fifth of the players, and which fifth a player is left out with is set by sorting everyone by his
label (`singleyear.stratified_player_folds`, `rloocv.BalancedGroupKFold`).  Any change to the labels therefore changes
which players each player's prior is trained alongside, and that alone moves the ratings: control C1 (nothing changed
but the defensive groupings, `62 --deal_seed=1`, 80% of players in another group) costs +0.056 on 63's team-game
criterion (z +1.2), moves a typical defensive prior by about 0.5 per 100 and Caruso 42nd to 14th in 2026.  The
current ratings carry one draw of that noise.  DECISIONS.md, "Experiment 42", the last part.

**What to build.**  A 62 option, e.g. `--prior_groupings=K` (default 1 = the shipped snake grouping, unchanged):
in `OutOfPlayerSPM.fit` (62, around line 300) fit the out-of-player boosters under K different, equally balanced
groupings -- grouping 0 the shipped one, groupings 1..K-1 from `BalancedGroupKFold(seed=k)` -- and give each player
the MEAN of his K out-of-player predictions (each from a booster that never saw him); the full-data model stays as it
is.  Both sides.  Cost: K times the fold fits (5K boosters per side per rated season); time one decade first.

**How to test.**  The standard chain with the incumbent's targets: `bash scripts/experiment_chain.sh
prior_avg5 x3def_w0.25 --prior_groupings=5` (the script takes extra 62 flags as its third argument).  Read 63
against the incumbent (adopt at z <= -2) AND its stability: a second run with other seeds (groupings 5..8) should
differ from the first by far less than C1's 0.056.  Caveat: 99's prior-shrink multipliers were fitted on the
incumbent's single-grouping priors; keep them for the first test (as every experiment has) and refit only on adoption
(97 then 99, trap 22).  If the incumbent's single grouping was a lucky draw (it was picked across many runs), the
average may read neutral against it on 63 while still being the more honest number; say so in the report.

**The standard for target experiments from now on:** run 62 with `--deal_target_def=<the incumbent's defensive
target>` (today `x3def_w0.25`) so every player's prior is trained alongside the same players as in the current
ratings and only the idea moves (experiment 42f; trap 27 in memory).  Label experiments decided earlier by less than
about 0.06 may have been decided by this noise (e.g. outside labels 25, age-adjusted labels 28, adjacent labels 30).

### What the shot-quality build and experiment 42 found (2026-10-06/07; DECISIONS.md has every number)

- **Two leaks in the play-by-play, fixed:** a shot's own timestamp knows its result (makes and misses are logged with
  different delays; timing is now measured from the row before the shot), and inside 10 ft the scorer logs a made
  shot about 1 ft nearer the rim than a miss (the spot is coded as 0-6 / 6-10 ft zones there).  A third, unlocated
  (0, 0) twos in 1997-2010, is result-coded too and still in every model's twos (not fixed; threes are clean).
- **The shot model:** boosted trees on the current logistic predict makes clearly better (shooter-adjusted -0.64 per
  1000, z -12.8) and lose no team test; the tuned logistic's score-margin terms are what failed the team tests.
- **Experiment 42** (opponent threes priced at shot quality x the shooter's other-half skill): rejected.  As first
  run +0.128 (z +2.8), but two thirds of that was the training-group noise above; held fixed (42f) +0.043 (z +1.3),
  no gain.  It drops Ighodaro 18th to 46th in 2026 (the owner thinks he was too high).
- Pieces: scripts 114-134, `src/eracoef/{shotframe,shotclock,shotmodel,shotfeatures,shotlearners,shotsearch,tracking,
  movement,shottest}.py`, `data/shotq/` (tables `_table`, priced arms `c_*`, per-attempt qualities `q42_*` with their
  stint side tables), the sheet's "Shot quality" tab (rows 77-99) and Experiments rows 42, 42x, 42f, C1.  Memory
  "Shot-quality build", traps 24-27.

### Waiting on the owner (most recent first)

0. **Experiments 42 / 42x / 42f: rejected** (the owner, 2026-10-07: "Sounds good").
1. **Publish the Portable column?**  Built and checked locally, not committed: `docs/index.html` (a sortable
   Portable column, portable offence / defence / total in the CSV download) and `docs/data/ratings.json` (team
   ratings unchanged, max difference 0.0 over 14,578 rows).  **No explanatory text on the page**: the owner called
   the drafted footer "AI slop, do not include" (memory "No drafted prose on the site").  To publish: commit,
   fast-forward `main`, push.
2. **Experiment 41, the swap adjustment's give-back rule**: recommend keeping the shipped (minutes) rule; flat is
   not adoptable (below).
3. **Experiment 40, the portable rating**: the owner on the ranks: "Portable rank looks great"; the verdict rides
   on item 1.
4. **Experiment 34, teammates' shot mix**: a tie (8.5993 vs 8.5995); recommend not adopting.
5. **From experiments 35-38**: (a) the prior without on-court plus-minus -- the strongest open candidate now:
   experiment 39 finds on-court numbers are team context, and experiment 38 that they hurt out of season;
   (b) RAPM on the linear box prior (B3) as the base, with the luck-adjusted targets and the swap adjustment on top
   (B3 beats OpenRAPM year over year, 8.587 vs 8.600); (c) the swap adjustment at x0.75.  Experiment 37's stages 4-6
   (era drift, targets, lockbox freeze) not run.
6. **Proposed, not started**: the swap adjustment's TYPE model for White-type cases (its strength, now 0.5, or its
   features).  Boston's +0.75 net offensive type is what charges Derrick White, whatever the give-back rule.
7. Older and still open: experiment 32's tuned booster (options a/b/c, DECISIONS.md "Experiment 32"); experiment 29,
   the team-movement weight (the owner does not want one-team players halved); the stayed list (30b).
8. Back pocket: https://vorp.app/nba/about (JavaScript-rendered; a plain fetch returns only "vorp").

### What experiments 34-41 found (DECISIONS.md has every number)

| exp | the owner's question | answer |
|---|---|---|
| 34 | teammates' shot quality with him on the court | a tie once the shrink's folds are pinned (trap 22) |
| 35 | slope vs error; part-season OpenRAPM against vanilla baselines | slope reads scale (about twice as sensitive), error ranks systems; seasons are the unit (more deals add nothing); **B3, RAPM on a linear box-only SPM prior tuned on other seasons, beats OpenRAPM year over year (8.587 vs 8.600)** and orders teammates better; OpenRAPM leads within season, mostly 2017-2026 where its shrink was fitted |
| 36 | does the best penalty change with sample size? | vanilla RAPM's does not (Bayes); OpenRAPM's prior weight falls with more games while a linear box prior's rises; under identical tuning the linear prior wins (B4, RAPM on our prior, 8.609) |
| 37 | a box prior fit directly on held-out games | every input group helps within season, none year over year; only fitting the shape year over year helps (t -3.4); probability of backtest overfitting 0.00 |
| 38 | what helps in-season vs other seasons | out of season wants 2-3x the penalty; role, same-season RAPM, on-court numbers, shot-making and assists help in-season and hurt out of season; age and career help out of season |
| 39 | traded players: real or regression to the mean? | the box score travels; defensive RAPM beyond the box score and on/off-court numbers stay with the team; eFG / TS / 2P% regress; the prior shrink and the swap adjustment are same-team corrections |
| 40 | a portable rating beside the team rating | half the plus-minus part travels (0.49 mid-season trades, 0.56 off-season moves, pooled 0.51); movers rate 0.8-1.0 worse but play to their rating once matched on minutes; not better on held-out error, so a second number, not a replacement |
| -- | how robust is the swap adjustment per player? | on average it holds up in every test; per player each piece is close to a coin flip (53-54% toward consensus, the whole 63%); the give-back is the weakest piece |
| 41 | the give-back rule (prompted by Derrick White) | flat give-back worse at the team-game level (z +2.2) and on the trade loss, better at the stint level (z -7.9); no give-back much worse (z +13); White 88th to 66th only |

### Where the pieces are

- Sheet "OpenRAPM experiments", https://docs.google.com/spreadsheets/d/1gPeRCFUtJNPtsBq_j2CNiTTdrarb98Yf3zXYx974jfI:
  tabs Definitions, Experiments (rows through 41), Scorecard, In vs out of season, Traded players, Portable rating.
- Chains and logs: `scratch/2026-10-05_scorecard/` (experiments 35-40) and `scratch/2026-10-06_giveback/` (41).
  Outputs: `outputs/within/<tag>/` (folds, incl. `deadline`), `outputs/scorecard/`, `outputs/heldout/`,
  `outputs/portable/`.
- `scripts/113_swap_explain.py --team=BOS --player="Derrick White" --also=2024,2025` rebuilds a team's swap
  adjustment step by step (type, give-back, centring, spread hold) and asserts it equals the shipped table.
- Emailed to the owner 2026-10-06, "OpenRAPM 2026: team vs portable ratings": a Drive CSV of all 582 players
  (team and portable rating, box part and plus-minus part per side).

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
        (the fold seasons are PINNED to 2017-2026 by default, the folds the adopted multipliers were fitted on; the
         `within` folder holds 1997-2026 since 2026-10-04, and an unpinned rerun shrank harder -- experiment 34's
         first run.  `--check=<table>` asserts a rerun reproduces a shipped table)
    .venv/Scripts/python scripts/91_swap_adjust.py --base=outputs/season_ratings_<name>_shrunk_raw.parquet --kappas=0.5 --taus= --hold_spread=within --tag=<name>
        (the swap adjustment the incumbent carries; writes outputs/season_ratings_<name>.parquet, ~2 min)
    .venv/Scripts/python scripts/63_yoy.py --rankings=<name>=outputs/season_ratings_<name>.parquet,incumbent=outputs/season_ratings_priorshrink.parquet --ref=incumbent --tag=<name> --splits=
    .venv/Scripts/python scripts/64_consensus_report.py outputs/season_ratings_<name>.parquet outputs/season_ratings_priorshrink.parquet
    .venv/Scripts/python scripts/66_compare.py incumbent=outputs/season_ratings_priorshrink.parquet <name>=outputs/season_ratings_<name>.parquet --season=2026 --top=20 --yoy=outputs/yoy_<name>.parquet --ref=incumbent
    .venv/Scripts/python scripts/90_swap_test.py --rankings=incumbent=outputs/season_ratings_priorshrink.parquet,<name>=outputs/season_ratings_<name>.parquet --contexts=nofatigue --checks=0 --tag=<name>
        (the swap test: does it order teammates better; 40 s)
    .venv/Scripts/python scripts/101_log_run.py --name=<name> --exp=<number> --idea=owner --verdict=pending --changed="<one plain sentence>"
        (the experiment table, the owner's 2026-10-04 ask: one row per experiment in experiments/runs.csv, every number read
         from the outputs above; then mirror the row into the Google Sheet "OpenRAPM experiments",
         https://docs.google.com/spreadsheets/d/1gPeRCFUtJNPtsBq_j2CNiTTdrarb98Yf3zXYx974jfI, tab Experiments.  Re-run with
         --verdict=adopted|rejected|"not adopted" once the owner rules; the same --exp replaces its row)

**A target experiment** (a new defensive or offensive target) adds `--deal_target_def=x3def_w0.25` to 62 so the box-score
prior is trained on the same groupings of players as the incumbent's; without it the experiment also pays a random
regrouping worth about 0.05 on this test (control C1, 2026-10-07).  `scripts/experiment_chain.sh NAME TARGET
"EXTRA 62 FLAGS"` runs this whole chain.

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
| 34: teammates' shot mix with him on the court (2026-10-04) | 8.599 | a tie (z -0.2) once the shrink's folds are pinned; the owner's call pending, recommend not adopting |
| 35 / 36: baselines -- B3, RAPM on a linear box-only SPM prior; B4, RAPM on OpenRAPM's own prior; both tuned on other seasons (2026-10-05) | **8.587** / 8.609 | B3 z -2.6 (better, every era) and orders teammates better; B4 z +2.7.  Reference rows; B3's recipe as the base is an open option |
| 41: the swap adjustment's give-back taken flat instead of by minutes (2026-10-06) | 8.601 | z +2.2, 18 of 56: worse; stint level z -7.9 (mostly the bench); trade loss worse (defence z +4.3, top 30 offence z +3.9); consensus 0.839.  Not adoptable.  No give-back at all: z +13.2 |
| rejected | | every chunk size; booster settings; off-court features; one row per player-season; cross-fitting the penalty; the un-shrunk label on BOTH sides (the 2026 offensive spread collapses 1.60 to 1.14, Curry falls to 61st); `onc_d` off the defensive list (the trade loss finds nothing) |

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
- **The swap adjustment's give-back rule** (experiment 41): flat loses at the team-game level and on the trade
  loss; no give-back loses everything.  The minutes rule stays.
- **The portable rating as a replacement** (experiment 40): given to same-team players it is worse (t +5 to +8).
  It is a second number beside the team rating.
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
| `97_within_season.py` | within-season folds, every input rebuilt from the rating games (`WITHIN_SEASON_LEDGER.md`); `--split=deadline` rates before the date 60% of the regular season was played and scores after, and the reverse |
| `101_log_run.py` | the experiment table (`experiments/runs.csv`, mirrored to the sheet's Experiments tab); `--sheet` prints the rows |
| `102`-`106` | the scorecard baselines (B1 linear box SPM, B2 vanilla RAPM, B3/B4 RAPM on a prior), the scorecard itself (`104`, error and slope kept apart), held-out swaps (`105`), OpenRAPM as shipped on every fold (`106`) |
| `107` / `108` | the box prior fit on held-out games: exact quadratics per penalty pair (`107`; `--traded=1` splits traded players' contributions) and the selection (`108`) |
| `110` / `111` / `112` | in-season vs other seasons; the traded-player check; the portable rating (fit, cross-fitted tests, production ratio in `outputs/portable/production.json`) |
| `113_swap_explain.py` | one team's swap adjustment step by step, checked against the shipped table |
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
**Built, not published (2026-10-06):** a sortable Portable column on the rankings page, from
`outputs/portable/production.json` applied by `52_site.py` to the product table (`portable.portable_table`).  **Never
add explanatory prose to a page** -- the owner rejected the drafted footer as "AI slop"; ask the owner for the wording.

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
13. Twenty-three traps now: 22, a re-fitted constant read from a folder that later grew (pin it; 99's
    `--fold_seasons`), and 23, scoring traded players by weighting team-game rows (it scores the other nine;
    split the prediction by player).  Memory "era-coefs measurement traps".

## The one failing test, accepted by the owner (2026-09-18)

`tests/test_vs_consensus.py::test_star_guards_are_not_buried` puts LaMelo Ball at rank 187 against a
hand-set ceiling of 160 on the published table (162 before the prior shrink of 2026-10-03, which moves every
offence-first guard down).  The numbers below are from the 2026-09-18 ruling.  **In points that is 0.016 per 100** -- he sits
0.168, the player at rank 160 sits 0.184 -- six times smaller than the owner's threshold for nothing, and
the check is a rank ceiling, the instrument the owner has ruled does not measure size.  The owner: *"that
test can fail, no worries."*  The honest fix, not done, is to replace the rank ceiling with a points check.

## Where to start next

1. **Get the owner's calls** on the list in START HERE; commit and publish only when the owner says so.
2. **If the owner says go: the prior without on-court plus-minus** (item 5a).  Experiments 37-39 point the same way: the
   on-court inputs carry team context, help in-season and hurt out of season.  One build, then the standard chain.
3. **The swap adjustment's type model** for White-type cases (item 6): its strength or its features, scored on the
   standard battery plus the per-player consensus pieces (`113_swap_explain.py`).
4. Older leads, unchanged: credit shot creation in the prior; split on-court plus-minus into offensive and
   defensive halves; the eight mixture probability columns into the prior; a standard error per player for alpha.

## Verify you are where this file says

    .venv/Scripts/python -m pytest tests -q                                  # 441 passed, 1 failed (LaMelo, accepted), 1 xfailed
    .venv/Scripts/python scripts/63_yoy.py --rankings=incumbent=outputs/season_ratings_priorshrink.parquet,ship=artifacts/season_ratings.parquet --ref=incumbent --tag=verify --splits=
                                                                             # incumbent 8.600 (8.5995)
    .venv/Scripts/python scripts/64_consensus_report.py outputs/season_ratings_product.parquet
                                                                             # 0.851 / 0.816 / 0.832, spreads 0.77 / 1.00, top5 4
    .venv/Scripts/python scripts/113_swap_explain.py                         # "reproduces the shipped swap adjustment ... 2.13e-14"
    .venv/Scripts/python scripts/52_site.py                                  # 14,578 rows; ratings.json with po / pd / pt fields
