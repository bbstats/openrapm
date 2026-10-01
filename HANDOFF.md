# Handoff: the single-year player rankings, one experiment at a time

**This file is transient.**  It starts the next session and is deleted when Phase 1 ships.  `DECISIONS.md`
is the permanent record and carries every number quoted here.  Do not let this grow into a lab notebook.

Branch `cleanup`; `main` is fast-forwarded to it at each publish, and the live site is served from `docs/`
on `main`.  `pytest -q`: **364 passed, 1 failed, 1 xfailed, ~126 s** with the scraped data.  The failure is
accepted and named at the bottom of this file; nothing else is red.

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

`outputs/season_ratings_product.parquet` (every other season allowed in the prior) is what
`docs/data/ratings.json` and the site are built from; it was rebuilt on 2026-09-18 and `poss_def` is a real
column now.  `outputs/season_ratings_unshrinkdef.parquet` is the same settings at
`--exclude_neighbours=1` and is **the incumbent every candidate is scored against**.
`artifacts/season_ratings.parquet` is the older system, kept for the tests; do not overwrite it.

Threads: the script pins BLAS to one thread and numba to four before importing anything.  **Never run two
builds at once**, and kill a chain's python children when you stop it.

## The test: year-over-year

Rate a season from its own games.  Predict every stint of the season before and after from the ten players'
ratings alone, refitting only the intercept and home edge on the scored season.  28 scored seasons, each
predicted twice, 56 observations.  The prior must not have seen the two scored seasons:

    .venv/Scripts/python scripts/62_single_year_board.py --exclude_neighbours=1 --score=0 --boards=<ten seasons> --out=season_ratings_<name>_<first>
        (three chunks of ten seasons, stitched BY HAND: concatenate the three parquets; ~3 min a season, so
         ~95 min for thirty.  With --chunk_label=outside, ~11 min a season: 280-310 extra RAPM solves each)
    .venv/Scripts/python scripts/63_yoy.py --rankings=<name>=outputs/season_ratings_<name>.parquet,incumbent=outputs/season_ratings_unshrinkdef.parquet --ref=incumbent --tag=<name> --splits=
    .venv/Scripts/python scripts/64_consensus_report.py outputs/season_ratings_<name>.parquet outputs/season_ratings_unshrinkdef.parquet
    .venv/Scripts/python scripts/66_compare.py incumbent=outputs/season_ratings_unshrinkdef.parquet <name>=outputs/season_ratings_<name>.parquet --season=2026 --top=20 --yoy=outputs/yoy_<name>.parquet --ref=incumbent

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
    .venv/Scripts/python scripts/73_tradeloss.py --alphas=incumbent=outputs/tradeset_unshrinkdef_alpha.parquet,<name>=outputs/tradeset_<name>_alpha.parquet --ref=incumbent

Paired by season on the exact intersection of eligible players; `--tier=` splits by exposure.  Read
`mean_diff` below zero as better and `z` against its own standard error.  It detects a defensive change of
0.0024 at z -2.2, so it has power at the size that matters.

**Every experiment is also split by player-quality tier: top 30, 31-90, 91-150, 151-300, 301+** (the owner,
2026-09-30: "in general we should do this").  Both tests, tiers set by the incumbent's rank in the rated season:

    .venv/Scripts/python scripts/88_yoy_by_player.py --cands=<name>=outputs/season_ratings_<name>.parquet --tag=<name>
        (the year-over-year difference shared among the players on the floor and summed by quality tier, by team
         change and by age; the groups add up to 63_yoy.py's number exactly; ~3 min)
    .venv/Scripts/python scripts/73_tradeloss.py --alphas=<as above> --ref=incumbent --quality=outputs/season_ratings_unshrinkdef.parquet --tier=each

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
| + the DEFENSIVE label un-shrunk (**the incumbent**) | **8.682** | z -5.24, 43 of 56; trade loss defence z -7.14, 27 of 30; adopted 2026-09-18 |
| 25: chunk rows labelled from the seasons OUTSIDE the chunk | 8.700 | z +5.0; trade loss z +4.5 / +5.0; rejected 2026-09-29 (25b, weighting those rows by their label's possessions: much worse) |
| 26: + RAPM pieces, same-team measure, game difficulty, rated as if every player changed teams | 8.682 | a tie (z +0.1); trade loss z +3.5 / +5.4; not adopted |
| 27: + a team-season random intercept | 8.679 | a tie (z -0.4); trade loss z +3.8 / +3.9; not adopted.  26 and 27 predict the season BEFORE better (z -2.5, -2.8) and the season after worse: an age tilt (veterans up, young players down), not a team-change effect |
| 28: + every chunk's label moved to its own age (aging curve) | 8.721 | z +4.1; trade loss z +5.1 / +4.1; rejected: over-corrects (Curry 92nd, Durant 136th), though consensus agreement rises to 0.818 |
| 29: the incumbent + the team-movement weight, floor 0.5, rebalanced within career bands | **8.669** | **z -4.1, 39 of 56**; trade loss a tie; consensus unchanged; 2026 top 20 close to the incumbent's.  Meets the adoption rule; awaiting the owner (it halves one-team players' weight, which they said they do not want) |
| 30: windows of two, four or six seasons split in the middle, each half labelled by the other; same-team measure as an input; rated as if every player changed teams | 8.706 | z +3.4, 16 of 56; trade loss z +2.8 / +7.6; rejected 2026-10-01.  Order alone leans better (z -1.6) but the ratings are too wide; best consensus agreement yet (0.829 total); loses as much on players who changed teams as on those who stayed.  LeBron James 126th in 2026 |
| 30b: experiment 30's models with the trade flag switched on only for the players who changed teams; and every player rated as if he stayed | 8.682 (matched) / **8.680** (stayed) | the flag does not help movers: matched against all-stayed a tie (z +0.8).  All-stayed against the incumbent: a tie (z -0.3) but stint level z -5.1 (mostly the bench), trade loss offence z -4.4 / defence z +4.1, best consensus yet (0.864).  Not adopted under the rule; owner's call.  Durant 72nd, LeBron 65th in 2026.  `62 --rate_same_team=both`, `89_stitch_by_move.py` |
| rejected | | every chunk size; booster settings; off-court features; one row per player-season; cross-fitting the penalty; the un-shrunk label on BOTH sides (the 2026 offensive spread collapses 1.60 to 1.14, Curry falls to 61st); `onc_d` off the defensive list (the trade loss finds nothing) |

## Open with the owner (2026-10-01)

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

    .venv/Scripts/python -m pytest tests -q                                  # 364 passed, 1 failed (LaMelo, accepted), 1 xfailed
    .venv/Scripts/python scripts/63_yoy.py --rankings=incumbent=outputs/season_ratings_unshrinkdef.parquet,ship=artifacts/season_ratings.parquet --ref=incumbent --tag=verify --splits=
                                                                             # incumbent 8.682
    .venv/Scripts/python scripts/64_consensus_report.py outputs/season_ratings_product.parquet
                                                                             # 0.787 / 0.803 / 0.795, spreads 1.03 / 1.04, top5 4
