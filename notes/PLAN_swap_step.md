# Plan: what the swap step is actually doing

Written 2026-10-18 for the next session.  Every stage is a read-only check: nothing here changes the shipped
ratings, and any change to step 5 is the owner's call after the report.

**Run the same evening**, `scripts/checks/153_swap_audit.py`: the results, two arms the plan did not have
(`step5_other` and the equal-move control `step5_share037`) and where the plan's stage-5 reading rule was wrong are
in notes/DECISIONS.md, "Check 153: what the swap step does".

## The question

Step 5 moves credit between teammates by what a type model says about each player.  The type model is a ridge
regression of each lineup swap's residual on the two swapped players' box-score, role, body and prior numbers,
fitted on other seasons.  Every team's total stays fixed.  It was adopted on 2026-10-01 on the year-over-year test,
with a paired t-statistic of -5.2, better in 39 of 56 predictions, and it improved the guard again on 2026-10-16,
paired t-statistic -2.66.  Two things are unknown:

- The decider, check 151, has never scored it: step 5 is not in the folds.
- Nothing has separated a real misrating of certain kinds of players from the other explanations below.

The owner's question, 2026-10-18: does step 5 fix a real, systematic misrating of certain kinds of players, or does
its gain come from something else?

## Five explanations to separate

1. **Real type misses.**  The step-4 ratings give some kinds of players too little of their team's credit and others
   too much, and the type model finds which.
2. **Within-team shrink alone.**  The swap test's within-team slope was 0.69-0.72: the swaps asked every gap inside a
   team to be multiplied by about 0.7.  Holding the spread multiplies every player's distance from the team's mean by
   about 0.85 in 2026.  That shrink alone may be the gain, whatever the type model says.
3. **Same-games luck.**  The type features are counted in the same games the swaps read.  A player who shot 42% from
   three made that player's lineups score more, so "made threes" predicts the swap residual because the threes are in it.  Step
   5 already leaves the games' part out of the features for this reason; the box counts stayed in.
4. **Coaching context.**  Who replaces whom, and when, is not random.  Garbage-time share is the largest defensive
   coefficient, -1.07 per standard deviation, positive good: it may be context the controls miss, not a trait.
5. **Undoing step 4.**  Step 4 multiplies the box-score part by about 0.66-0.73.  Step 5 may be handing part of it back
   to scorers and passers inside each team, so steps 4 and 5 pull against each other.

## Ground rules

- One process at a time, numba at 4 threads; stopping a run stops its python children.
- New code goes in `scripts/checks/153_swap_audit.py`, one subcommand per stage, each caching its tables under
  `outputs/swap_audit/` so a stopped run resumes.  A new model-layer function gets a test in `tests/`.  The behaviour
  of `src/openrapm/swap_adjust.py` does not change.
- Traps that bind here, from `notes/DECISIONS.md`: 37, a season-total input leaks into the decider, worth z -31, so
  every rated-season input on a fold comes from its fit games; 36, one build's swap-test order has a standard
  deviation of 0.022 from regrouping alone, so a swap-test difference is read beside the noise control; 31, score
  every stage before blaming one.
- Nothing is committed, pushed or published.
- Results: a section in `notes/DECISIONS.md`, "Check 153: what the swap step does"; long tables to the Google Sheet
  "OpenRAPM experiments", new tab "Swap step"; one report to the owner at the end in the four-part format, one
  quantity per column, no parentheses.

## Stage 0: reproduce, about 10 minutes

- `113_swap_explain.py` reproduced the shipped 2026 step 5 on 2026-10-18 to 1.5e-14.
- The new fold code, handed every game of a season as its fit games, must rebuild the test-setting table
  `outputs/season_ratings_bags10.parquet` from `outputs/season_ratings_bags10_step4.parquet` to 1e-9 for every
  season, with `SwapSettings.exclude_near` 1, the test setting.  Stop if it does not.

## Stage 1: does step 5 help on the decider?

**Build: step 5 on every fold** of `bags10_early` and `bags10_within`, 30 seasons x 12 folds.

| input | where it comes from on a fold |
|---|---|
| the ratings it adjusts | check 151's step 4 for the fold, `step4_parts` |
| the type model | the test setting as shipped: other seasons more than one away, their whole-season swaps against `season_ratings_bags10_step4.parquet`, five player folds |
| the rated players' type features | the season panel rebuilt from the fold's fit games, `season_panel` with `fit_game_ids`; the two prior features from the fold's step-4 prior part; standardised with the whole test table's means and standard deviations |
| team possessions for the give-back, main teams, spread weights | the fold's fit-game stints |

Fit games are the season's games minus the fold's held-back games, which `test_<stem>.npz` lists.  The adjusted
ratings go to check 151 through its `extra` stages hook, `stage_ratings`, as a stage `step5`, positive good in the
fold's row order.

**Arms:**

| arm | what it is |
|---|---|
| `step4` | the decider as today |
| `step5` | step 5 as shipped, on the fold |
| `step5_noise` | the same on `noise_bags10`'s folds: what regrouping alone does |
| `step5_leak` | step 5 with whole-season type features: a leak control, expected to look far too good |

**Readings:** the decider, all held-back stints and the DNP slice, `step5` against `step4` season by season: mean
difference, standard error, paired t-statistic, seasons better.  The noise control beside it.

| result | what it means |
|---|---|
| `step5` better than `step4` at a paired t-statistic of -2 or below, beyond the noise control | step 5 helps on the owner's test |
| a tie | step 5 has no within-season evidence; its case rests on the guard and the swap test |
| worse at +2 or above | step 5 hurts on the owner's test |
| `step5_leak` not clearly better than `step5` | the leak control failed to detect a leak: stop and check the fold code |

**Cost:** about 2-3 hours of code and tests.  Machine time unmeasured: the fold panels are the slow part, 360 of them;
step 5 itself took 23 seconds for a whole table of 30 seasons on 2026-10-18.

## Stage 2: which part does the work?  Explanation 2

**Arms**, each on the folds through stage 1's code and on the whole-season test table:

| arm | what it is |
|---|---|
| `shipped` | step 5 as shipped |
| `shrink_only` | no type move: each player's distance from the main team's mean multiplied by the shipped spread factor for that season and side, factors from rerunning step 5 on `season_ratings_bags10_step4.parquet` |
| `placebo_type` | the type features shuffled among players within each season, in training and in use, then half strength, give-back and spread held as shipped; five shuffle seeds, read as their mean |
| `no_spread_hold` | half strength and give-back, spread not held: it failed the year-over-year test on 2026-10-01; the decider has never seen it |

**Readings, each arm against `step4`:** the decider; the guard, `63_yoy.py`; the swap test, `90_swap_test.py`, order and
gaps; the 2026 top 20, `66_compare.py`, for `shrink_only` and `placebo_type`.

| result | what it means |
|---|---|
| `shrink_only` as good as `shipped` | the gain is the within-team shrink; the type model adds nothing |
| `placebo_type` as good as `shipped` | the gain is the give-back and spread mechanics, not the type content |
| `shipped` beats both at a paired t-statistic of -2 or below | the type content is real information |

**Cost:** about 30 minutes of code, about 1 hour of machine time.

## Stage 3: is each coefficient real, and is it a trait?  Explanations 1 and 4

No rebuild: each season's normal equations are already kept, `PairGrams`.

- **Season jackknife:** the type model refitted 30 times, one season left out each time; each coefficient's jackknife
  standard error and t-statistic.  44 coefficients, so a Bonferroni bar is about 3.3 beside the plain 2.
- **By era:** fitted on 1997-2006, 2007-2016, 2017-2026 alone, and on odd and even seasons alone.
- **Player-level stability:** the correlation, over player-seasons, of each player's type prediction from the odd-season
  model and from the even-season model.
- **Context refits**, the swaps rebuilt three ways and the model refitted on each: garbage-time stints removed from the
  lineup results; the fatigue clocks back in the context, `CONTEXTS["full"]`; each swap's weight capped at its 95th
  percentile.

**Readings:** one full table, every feature on both sides: coefficient, jackknife standard error, t-statistic, the three
era coefficients, odd, even, and each context refit.  Never a prose list of winners.

| result | what it means |
|---|---|
| a coefficient keeps its sign in all three eras and clears the bar | a stable pattern |
| garbage-time share collapses with garbage-time stints removed | that coefficient is context, explanation 4 |
| the odd and even models' player predictions barely agree | the type prediction is mostly noise per player |

**Cost:** about 30 minutes of code, under 10 minutes of machine time.

## Stage 4: same-games luck?  Explanation 3

On the folds, with stage 1's fold features and fold step-4 ratings, two fits of the type model pooled over every fold
and season:

| fit | features from | swaps from |
|---|---|---|
| `same_games` | the fold's fit games | the fold's fit games: the shipped situation |
| `other_games` | the fold's fit games | the fold's held-back games |

The held-back quarter holds fewer swaps; each game is held back once in each of the three repeats, so every swap is
used three times, and the season jackknife gives the standard errors.

**Readings:** per coefficient, `other_games` beside `same_games`, each with its standard error, and their ratio.

| result | what it means |
|---|---|
| made and missed threes and twos fall below half in `other_games` | the model is reading that season's shooting luck in the very lineups it scores |
| the coefficients hold | the box counts describe a trait that carries over to other games |

**Cost:** about 1 hour of code, about 30 minutes of machine time.

## Stage 5: whose mistake is it?  Explanation 5

The type model fitted, whole seasons as shipped, against the residuals of four rating tables per season, and each
table scored on the swap test, order and gaps:

| table | file |
|---|---|
| plain single-season RAPM, penalty 3,000 | `swap_test.plain_rapm` |
| the box-score part alone | `season_ratings_bags10_step3.parquet`, `prior_off` and `prior_def` |
| step 3 | `season_ratings_bags10_step3.parquet` |
| step 4 | `season_ratings_bags10_step4.parquet`, what step 5 reads today |

**Readings:** the full coefficient table for each, side by side.

| result | what it means |
|---|---|
| small against step 3, large against step 4 | step 5 is undoing step 4's shrink of the box-score part inside teams |
| large against the box-score part and step 3 alike | the box-score prior mis-splits teammates; the fix belongs in step 2 |
| large against plain RAPM only | expected: the box score carries skill RAPM misses; not a misrating |

**Cost:** about 30 minutes of code, about 15 minutes of machine time.

## The report

One table, an explanation per row: the verdict and the reading that decides it.  Then the 2026 top 20 for any arm that
would change the shipped table, and one question for the owner.  Stages 1-5 run back to back as one resumable
background chain: about a day of session time, most of it code.
