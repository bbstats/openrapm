# Glossary

The words this project uses, each in one or two lines, in the order a reader meets them; then the raw box
counts; then every column the box-score prior can read, written from the code. Reports to the owner use the
plain names, never the codes.

## Method terms

**Stint.** A stretch of play with the same ten players on the court; a substitution or the end of a period
starts a new one. The 2003 playoffs, 88 games, give 2,653 stints, about 30 a game.

**Possession.** One team's turn with the ball, from when it gains the ball until the other team gains it or the
period ends; an offensive rebound continues it. Every rating is in points per 100 possessions.

**RAPM** (regularised adjusted plus-minus). A ridge regression of every stint's points per 100 possessions on
the ten players on the court, plus context columns; a player's coefficient is his rating. In the published
tables positive is good on both ends.

**Design.** The stints of one or more seasons as the rows of that regression: one column per player per side
(offence, defence), plus the context columns (season intercept, home court, playoffs, garbage time, score
margin, margin x share of the game left). Each row's target is its points per 100 possessions (adjusted as
the two targets below say) and its weight its possessions (`design.py`).

**Garbage time.** A stint in the fourth quarter with a margin of 20 or more, or of 15 or more with under six
minutes left (`config.yaml`, `garbage_time`). It is flagged, and the design carries the flag as a context
column; a garbage-time row weighs the same as any other (`gt_weight: 1.0`).

**Penalty.** The ridge's lambda. Seen alone, a player with n possessions keeps about n / (n + lambda) of his raw
impact (the coefficient an unpenalised fit would give him). With lambda = 40,000, a player with 54,000
possessions keeps 57% of it and one with 2,600 keeps 6%.

**Label.** What the box-score prior learns to predict: a player's RAPM from one fit over every season except the
unseen ones, one number per side for all those seasons together (step 1, `rapm.py`).

**Leave-season-out.** Fitted without the rated season (and, in the test setting, without the seasons either side
of it). Each season's normal equations are stored once, so leaving a season out is one solve on the other
seasons' sums, not thirty refits.

**Un-shrink.** Dividing a coefficient by the share n / (n + lambda) the ridge kept, so long and short careers sit
on one scale. The division stops at 4,444 possessions and the rest comes from the level of the player's
possession tier. Only the defensive label is un-shrunk (`notes/DECISIONS.md`, experiment 22): un-shrinking
offence too helped the year-over-year test but made the order within each side worse and dropped the
consensus agreement from 0.771 to 0.680. Example: with
lambda = 40,000 the division stops at 4,444 / 44,444 = 0.10, so a 1,000-possession player (kept share 0.024)
with coefficient 0.05, in a tier whose level is -3.0, gets 0.05 / 0.10 + (1 - 0.24) x -3.0 = -1.77.

**Season panel.** One row per player, season and side: his possessions, padded box-score rates, role, career,
body, team, and his team's record with him on and off the court (`outputs/season_panel.parquet`).

**Padded rate.** A per-100 box-score rate pulled toward the rate of players with about as many possessions, by
k possessions' worth of it: padded = (n x rate + k x bin rate) / (n + k), with n his possessions. The bin rate
is that season's rate among the players of his possession bin (six ranges of possessions, equally wide on a log
scale). k is measured per count and season from every player's odd- against even-numbered games: the noise
variance of one possession's worth of the rate over the variance of the true rates between players
(`box_rates.py`). A 200-possession player with 2.1 steals per 100, k = 450 and a bin rate of 1.4 is padded to
(200 x 2.1 + 450 x 1.4) / 650 = 1.6.

**Box-score prior.** LightGBM's (gradient-boosted trees') prediction of a player's label from his season-panel
row: 17 inputs on offence, 14 on defence (`features.SHIPPED`). Step 2, `box_prior.py`.

**Booster.** One fitted LightGBM model. Each rated season and side has six: one fitted on every training row
and one without each of the five player folds (below). `--save-models` writes them to
`outputs/prior_models_NAME.pkl`.

**Career row and chunk rows.** A player's training rows for the prior. The career row averages his box score over
every season outside the unseen ones; a chunk row averages a run of 1, 2 or 3 consecutive seasons. Every row
carries his career label, and the chunk rows together weigh what the career row weighs. Example: seasons of
1,000, 2,000 and 3,000 possessions and a label resting on 6,000 give six chunk rows weighing 300, 600, 900,
900, 1,500 and 1,800, together 6,000.

**Out-of-player.** Players are sorted by label and assigned to five player folds, 0,1,2,3,4,4,3,2,1,0,..., and each
player's prior comes from the model fitted without his fold, so it cannot recognise his box score and return
his own label.

**Prior-informed RAPM.** Step 3 (`prior_rapm.py`): the season's RAPM with each player pulled toward a multiple of
his prior instead of toward 0, penalty 13,037. A rating is s x prior + r, where r is how far the season's games
moved him.

**Prior scale.** That multiple, s: one per side and season, the coefficient on the summed priors of the five on
the court, unpenalised, so the season's games set it. In 2026 it is 2.35 on offence (product setting): a
player whose box-score prior is +0.5 per 100 starts at 2.35 x 0.5 = +1.18. Why s is well above 1 on offence:
the offensive label is a ridge coefficient that is not un-shrunk, so it keeps only n / (n + 40,000) of a
player's raw impact, and the prior learned from it is on that shrunk scale; s puts it back on the season's
scale. The defensive label is un-shrunk, and its scale stays near 1. Over the thirty published seasons s runs
from 2.32 to 4.47 on offence and from 0.93 to 1.39 on defence. With both labels un-shrunk (experiment 1 of
2026-09-14 evening) the scales read 0.84 and 0.92.

**Centring.** Per season and side, the possession-weighted mean rating is subtracted, so the average possession
is played by a 0 player.

**Prior part and games' part.** The two parts of a rating after step 3: the prior part s x prior (`prior_off`,
`prior_def`), and the games' part r, also called u (`u_off`, `u_def`).

**Prior shrink.** Step 4 (`prior_shrink.py`): every rating's prior part multiplied by a multiplier per side and
season, fitted on games the rating never saw, the games' part unchanged, then centred again. The published
multipliers run from 0.699 to 0.715 on offence and from 0.929 to 0.976 on defence. Example: 2026, m = 0.704 on
offence, so Jokic's offensive prior part of 5.394 becomes 3.80, 3.78 after centring; his games' part stays
0.97, and his offensive rating goes from 6.37 to 4.75.

**Within-season folds.** Each season of 2017-2026 rated from three quarters of its games (the fit games) and
scored on the other quarter (the test games), which the rating never saw. The games are assigned at random to
four folds, three times over: twelve folds a season (`openrapm folds`, `notes/WITHIN_SEASON_LEDGER.md`).

**Lineup swap.** Two lineups of one team in one season that share four players. The four are held fixed, so the
difference between the two lineups' results is evidence about the fifth player of each.

**Swap test.** Does a ratings table order teammates the way their lineup swaps do: for every lineup swap in the
seasons either side, the two swapped players' rating gap against the swap's result (`scripts/checks/90_swap_test.py`).

**Main team.** The team a player played the most possessions for that season.

**Swap adjustment.** Step 5 (`swap_adjust.py`): a type model (a ridge regression of each lineup swap's unexplained
result on the two swapped players' standardised box-score, role, body and prior numbers), fitted on other seasons,
predicts which players the ratings give too much or too little credit; half of that prediction is applied.

**Give-back.** Each team's total adjustment is taken back from its own players in proportion to their share of
its possessions, so no team's total moves. A starter on the court for 75% of his team's possessions has a
share of 0.15 (five players per possession).

**Spread held.** After the give-back, each player's distance from his main team's mean is scaled, one factor per
season and side, until the spread inside teams equals step 4's. Team means and the order inside each team stay;
over the thirty seasons the factors average 0.84 on offence and 0.86 on defence.

**Product setting and test setting.** Product builds the published ratings: a season's labels, the prior's
training rows, step 4's multipliers and step 5's type model all leave that season out. Test builds the table
the year-over-year test scores: they also leave out the seasons either side, because those are the seasons the
test predicts (`settings.py`, `PRODUCT` and `TEST`).

**The two targets.** What each side's ridge explains, per 100 possessions. Offence,
`points_with_expected_free_throws`: points with each made free throw replaced by the shooter's padded
free-throw percentage. Defence, `points_with_expected_opponent_threes`: the same, less 3 x 0.75 x (threes made -
threes expected), so three quarters of the shooters' three-point luck is taken off. Scoring is always against
the points actually scored.

**Year-over-year test.** The test that decides (`scripts/checks/63_yoy.py`): each season's ratings predict every
stint of the season before and the season after from the ten players' ratings alone, refitting only that
season's intercept and home edge; the misses are summed to team-games. 28 scored seasons, each predicted from
both sides: 56 observations. The score is sqrt(team-game mean squared error) x sqrt(2 / pi), in points per 100
(`game_armse`, `scoring.py`): the incumbent reads 8.5857.

**Team-game.** One team's offence in one game: the unit the year-over-year test scores, because the noise of
single possessions cancels inside a team's hundred-odd possessions before the miss is squared.

**Paired z.** For each of the 56 observations, one table's team-game mean squared error minus the other's; the
mean of those differences divided by its standard error (a paired t-statistic). A change is adopted at -2 or
below.

**Incumbent.** The table every candidate is scored against: today `outputs/season_ratings_weightfix.parquet`,
experiment 46 built at the test setting.

**Trade set.** For each rated season, the team-games of that season and the seasons either side, each player's
share of his team's possessions in each, and the rated season's ratings subtracted; what is left is ridged back
onto the players. A player's coefficient, alpha, is what his comings and goings say that his rating did not
already know (`scripts/checks/70_tradeset.py`).

**Trade loss.** Two tables compared by the size of their trade-set corrections on the same players, paired by
season (`scripts/checks/73_tradeloss.py`). It counts every player once, so it sees the bench players the
year-over-year test, an error per team-game, barely weighs.

**Consensus.** A blend of public all-in-one metrics for 2024-26 (`data/external/consensus.csv`). The ratings are
compared with it by Spearman's rank correlation over players with 1,000 or more possessions, each pooled over
the three seasons: 0.850 on offence, 0.781 on defence and 0.806 in total over 475 players for the published
table. A sanity check, read every time and never a fitting target: a gross miss can veto a change, a marginal
one cannot.

**Noise control.** The incumbent rebuilt with nothing changed but which players share each of the prior's five
folds (`--fold-assignment 1`). It shows how far every reading moves when nothing real changes: on the
year-over-year test it reads 8.5853 against the incumbent's 8.5857, paired z -0.24.

**Replicate.** The same change built a second time with another random assignment of players to the prior's
folds, so every reading is seen twice (`scripts/checks/135_robust_review.py`).

**Held-out audit.** On the within-season folds' held-back games, are players high on some input rated too high
or too low (`scripts/checks/138_heldout_audit.py`).

**Quality tiers.** Players grouped by the incumbent's rank in the rated season: top 30, 31-90, 91-150, 151-300,
301 and below. Every experiment is read in each tier (`scripts/checks/88_yoy_by_player.py`).

## The raw box counts

The thirteen per-game counts every design carries, in the order of `config.yaml`'s `box_counts`. They keep the
NBA's abbreviations in the box table, the stint counters and the bias page; the per-100 rates built from them
have plain names (the table below).

| code | plain name |
|---|---|
| fg3m / fg3_miss | made threes / missed threes |
| fg2m / fg2_miss | made twos / missed twos |
| ftm / ft_miss | made free throws / missed free throws |
| orb / drb | offensive rebounds / defensive rebounds |
| ast | assists |
| tov | turnovers |
| stl | steals |
| blk | blocks |
| pf | personal fouls |

Files and notes from before 2026-10-11 use older column names (`poss_pct`, `onc_o`, `chunk_poss`, ...);
`src/openrapm/data/legacy_names.py` maps each to the name below.

## Every column the prior can read

Written from src/openrapm/features.py by `openrapm glossary`; edit the catalogue there, not the table here.
The shipped prior reads 17 of these on offence and 14 on defence (`features.SHIPPED`), plus `row_possessions`
and `row_seasons`; the rest are candidates an experiment may try (the on-court columns came out in experiment
45).

- Offence: assists; assists minus turnovers; effective field-goal %; career possessions before this season;
  seasons played before this one; three-point %; shot attempts; free-throw attempts; offensive rebounds;
  offensive-rebound share of his rebounds; share of shots that are threes; share of team possessions played;
  points; steals; seasons with his current team; true shooting %; weight.
- Defence: blocks; defensive rebounds; career possessions before this season; seasons played before this one;
  shot attempts; share of games started; height; share of team possessions played; points; steals; steals plus
  blocks; true shooting %; possessions he finishes; weight.

<!-- generated: features -->
| name | meaning | formula | unit | scale |
|---|---|---|---|---|
| `threes_made_per100` | made threes | padded rate minus the league level | per 100 possessions | centred |
| `threes_missed_per100` | missed threes | padded rate minus the league level | per 100 possessions | centred |
| `twos_made_per100` | made twos | padded rate minus the league level | per 100 possessions | centred |
| `twos_missed_per100` | missed twos | padded rate minus the league level | per 100 possessions | centred |
| `free_throws_made_per100` | made free throws | padded rate minus the league level | per 100 possessions | centred |
| `free_throws_missed_per100` | missed free throws | padded rate minus the league level | per 100 possessions | centred |
| `off_rebounds_per100` | offensive rebounds | padded rate minus the league level | per 100 possessions | centred |
| `def_rebounds_per100` | defensive rebounds | padded rate minus the league level | per 100 possessions | centred |
| `assists_per100` | assists | padded rate minus the league level | per 100 possessions | centred |
| `turnovers_per100` | turnovers | padded rate minus the league level | per 100 possessions | centred |
| `steals_per100` | steals | padded rate minus the league level | per 100 possessions | centred |
| `blocks_per100` | blocks | padded rate minus the league level | per 100 possessions | centred |
| `fouls_per100` | personal fouls | padded rate minus the league level | per 100 possessions | centred |
| `threes_made_per100_uncentred` | made threes, league level kept | padded rate | per 100 possessions | level |
| `threes_missed_per100_uncentred` | missed threes, league level kept | padded rate | per 100 possessions | level |
| `twos_made_per100_uncentred` | made twos, league level kept | padded rate | per 100 possessions | level |
| `twos_missed_per100_uncentred` | missed twos, league level kept | padded rate | per 100 possessions | level |
| `free_throws_made_per100_uncentred` | made free throws, league level kept | padded rate | per 100 possessions | level |
| `free_throws_missed_per100_uncentred` | missed free throws, league level kept | padded rate | per 100 possessions | level |
| `off_rebounds_per100_uncentred` | offensive rebounds, league level kept | padded rate | per 100 possessions | level |
| `def_rebounds_per100_uncentred` | defensive rebounds, league level kept | padded rate | per 100 possessions | level |
| `assists_per100_uncentred` | assists, league level kept | padded rate | per 100 possessions | level |
| `turnovers_per100_uncentred` | turnovers, league level kept | padded rate | per 100 possessions | level |
| `steals_per100_uncentred` | steals, league level kept | padded rate | per 100 possessions | level |
| `blocks_per100_uncentred` | blocks, league level kept | padded rate | per 100 possessions | level |
| `fouls_per100_uncentred` | personal fouls, league level kept | padded rate | per 100 possessions | level |
| `def_rebounds_of_free_throws_per100` | defensive rebounds of missed free throws | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `def_rebounds_of_threes_per100` | defensive rebounds of missed threes | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `def_rebounds_of_rim_twos_per100` | defensive rebounds of missed twos within 6 ft of the rim, or logged without a location | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `def_rebounds_of_outside_twos_per100` | defensive rebounds of missed twos from beyond 6 ft | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `bad_pass_steals_per100` | steals of a pass (a bad-pass turnover) | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `lost_ball_steals_per100` | steals of the ball from a dribbler (a lost-ball turnover) | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `assisted_points_per100` | points from assisted field goals | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `unassisted_points_per100` | points from unassisted field goals | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `corner_threes_attempted_per100` | corner threes attempted | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `above_break_threes_attempted_per100` | above-the-break threes attempted | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `assisted_threes_made_per100` | assisted made threes | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `unassisted_threes_made_per100` | unassisted made threes | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `russell_blocks_per100` | blocks his team recovered (Russell blocks) | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `rim_blocks_per100` | blocks of twos within 6 ft of the rim, or logged without a location | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `mid_range_blocks_per100` | blocks of twos from beyond 6 ft | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `three_point_blocks_per100` | blocks of threes | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `offensive_fouls_per100` | offensive fouls committed | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `stolen_turnovers_per100` | turnovers stolen by the defence (live-ball turnovers) | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `assists_to_rim_per100` | assists on makes at the rim (within 6 ft, or logged without a location) | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `assists_to_mid_range_per100` | assists on made twos from beyond 6 ft | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `assists_to_corner_threes_per100` | assists on made corner threes | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `assists_to_above_break_threes_per100` | assists on made above-the-break threes | padded play-by-play count per 100 minus the league level | per 100 possessions | centred |
| `drives_vs_league_avg` | drives to the basket | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `drive_points_vs_league_avg` | points scored on drives | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `potential_assists_vs_league_avg` | potential assists (passes that led to a shot) | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `passes_made_vs_league_avg` | passes made | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `touches_vs_league_avg` | touches of the ball | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `paint_touches_vs_league_avg` | touches in the paint | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `post_touches_vs_league_avg` | post touches | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `elbow_touches_vs_league_avg` | elbow touches | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `minutes_with_ball_vs_league_avg` | minutes with the ball in his hands | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `miles_on_offense_vs_league_avg` | miles run on offence | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `off_reb_chances_vs_league_avg` | offensive rebound chances (a rebound within his reach) | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `off_reb_contested_vs_league_avg` | contested offensive rebounds | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `def_reb_chances_vs_league_avg` | defensive rebound chances (a rebound within his reach) | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `def_reb_contested_vs_league_avg` | contested defensive rebounds | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `rim_shots_defended_vs_league_avg` | shots at the rim he was the closest defender on | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `twos_defended_vs_league_avg` | two-point shots he was the closest defender on | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `threes_defended_vs_league_avg` | three-point shots he was the closest defender on | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `miles_on_defense_vs_league_avg` | miles run on defence | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `rim_makes_over_expected_per_shot` | shots made within 6 ft against him, above what the league's shooters make against those defenders, per shot faced | (makes - expected makes) / (shots faced + pad); the pad is the shots in data/dashboard_rates.py | makes per shot | centred |
| `twos_makes_over_expected_per_shot` | two-point shots made against him above expected, per shot faced | (makes - expected makes) / (shots faced + pad); the pad is the shots in data/dashboard_rates.py | makes per shot | centred |
| `threes_makes_over_expected_per_shot` | three-point shots made against him above expected, per shot faced | (makes - expected makes) / (shots faced + pad); the pad is the shots in data/dashboard_rates.py | makes per shot | centred |
| `deflections_vs_league_avg` | deflections | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `contested_twos_vs_league_avg` | contested two-point shots | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `contested_threes_vs_league_avg` | contested three-point shots | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `charges_drawn_vs_league_avg` | charges drawn | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `loose_balls_recovered_vs_league_avg` | loose balls recovered | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `def_box_outs_vs_league_avg` | defensive box-outs | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `off_box_outs_vs_league_avg` | offensive box-outs | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `screen_assists_vs_league_avg` | screen assists | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `screen_assist_points_vs_league_avg` | points from his screen assists | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centred |
| `deflections_vs_league_avg_shuffled` | deflections, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centred |
| `contested_twos_vs_league_avg_shuffled` | contested two-point shots, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centred |
| `contested_threes_vs_league_avg_shuffled` | contested three-point shots, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centred |
| `charges_drawn_vs_league_avg_shuffled` | charges drawn, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centred |
| `loose_balls_recovered_vs_league_avg_shuffled` | loose balls recovered, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centred |
| `def_box_outs_vs_league_avg_shuffled` | defensive box-outs, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centred |
| `off_box_outs_vs_league_avg_shuffled` | offensive box-outs, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centred |
| `screen_assists_vs_league_avg_shuffled` | screen assists, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centred |
| `screen_assist_points_vs_league_avg_shuffled` | points from his screen assists, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centred |
| `attackers_guarded_scoring_vs_league_avg` | the scoring of the attackers he guarded (points per matchup possession without the rows against him), against the league's | (attackers' points per matchup possession, each without his rows against the player, padded 50 possessions toward the league's, mean weighted by possessions - league rate) / league rate; the second sign-flipped | fraction above the league average | centred |
| `defenders_faced_stinginess_vs_league_avg` | how stingy the defenders were who guarded him (points allowed per matchup possession without the rows against him, sign-flipped), against the league's | (attackers' points per matchup possession, each without his rows against the player, padded 50 possessions toward the league's, mean weighted by possessions - league rate) / league rate; the second sign-flipped | fraction above the league average | centred |
| `points_per100` | points | 2 twos_made_per100 + 3 threes_made_per100 + free_throws_made_per100 | per 100 possessions | centred |
| `shots_per100` | shot attempts | twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 | per 100 possessions | centred |
| `free_throw_attempts_per100` | free-throw attempts | free_throws_made_per100 + free_throws_missed_per100 | per 100 possessions | centred |
| `three_point_attempts_per100` | three-point attempts | threes_made_per100 + threes_missed_per100 | per 100 possessions | centred |
| `used_possessions_per100` | possessions he finishes | twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100 | per 100 possessions | centred |
| `rebounds_per100` | rebounds | off_rebounds_per100 + def_rebounds_per100 | per 100 possessions | centred |
| `steals_plus_blocks_per100` | steals plus blocks | steals_per100 + blocks_per100 | per 100 possessions | centred |
| `assists_minus_turnovers_per100` | assists minus turnovers | assists_per100 - turnovers_per100 | per 100 possessions | centred |
| `threes_minus_twos_attempted_per100` | threes minus twos taken | threes_made_per100 + threes_missed_per100 - twos_made_per100 - twos_missed_per100 | per 100 possessions | centred |
| `effective_fg_pct` | effective field-goal % | (twos_made_per100 + 1.5 threes_made_per100 + 3 x 0.5) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 3), uncentred rates | fraction | level |
| `true_shooting_pct` | true shooting % | (2 twos_made_per100 + 3 threes_made_per100 + free_throws_made_per100 + 6 x 0.54) / (2 twos_made_per100 + 2 twos_missed_per100 + 2 threes_made_per100 + 2 threes_missed_per100 + 0.88 free_throws_made_per100 + 0.88 free_throws_missed_per100 + 6), uncentred rates | fraction | level |
| `three_point_attempt_share` | share of shots that are threes | (threes_made_per100 + threes_missed_per100 + 3 x 0.25) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 3), uncentred rates | fraction | level |
| `free_throws_per_shot` | free throws per shot | (free_throws_made_per100 + free_throws_missed_per100 + 3 x 0.25) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 3), uncentred rates | fraction | level |
| `three_point_pct` | three-point % | (threes_made_per100 + 3 x 0.34) / (threes_made_per100 + threes_missed_per100 + 3), uncentred rates | fraction | level |
| `two_point_pct` | two-point % | (twos_made_per100 + 3 x 0.48) / (twos_made_per100 + twos_missed_per100 + 3), uncentred rates | fraction | level |
| `free_throw_pct` | free-throw % | (free_throws_made_per100 + 2 x 0.75) / (free_throws_made_per100 + free_throws_missed_per100 + 2), uncentred rates | fraction | level |
| `assists_per_used_possession` | assists per possession used | (assists_per100 + 3 x 0.18) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100 + 3), uncentred rates | fraction | level |
| `turnovers_per_used_possession` | turnovers per possession used | (turnovers_per100 + 3 x 0.13) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100 + 3), uncentred rates | fraction | level |
| `off_rebound_share` | offensive-rebound share of his rebounds | (off_rebounds_per100 + 3 x 0.22) / (off_rebounds_per100 + def_rebounds_per100 + 3), uncentred rates | fraction | level |
| `effective_fg_pct_vs_league` | effective field-goal % above the league's | (twos_made_per100 + 1.5 threes_made_per100 - L x (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100)) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 3), uncentred rates; L the same ratio of the league's rates (each uncentred minus centred) | fraction | centred |
| `true_shooting_pct_vs_league` | true shooting % above the league's | (2 twos_made_per100 + 3 threes_made_per100 + free_throws_made_per100 - L x (2 twos_made_per100 + 2 twos_missed_per100 + 2 threes_made_per100 + 2 threes_missed_per100 + 0.88 free_throws_made_per100 + 0.88 free_throws_missed_per100)) / (2 twos_made_per100 + 2 twos_missed_per100 + 2 threes_made_per100 + 2 threes_missed_per100 + 0.88 free_throws_made_per100 + 0.88 free_throws_missed_per100 + 6), uncentred rates; L the same ratio of the league's rates (each uncentred minus centred) | fraction | centred |
| `three_point_attempt_share_vs_league` | share of shots that are threes above the league's | (threes_made_per100 + threes_missed_per100 - L x (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100)) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 3), uncentred rates; L the same ratio of the league's rates (each uncentred minus centred) | fraction | centred |
| `free_throws_per_shot_vs_league` | free throws per shot above the league's | (free_throws_made_per100 + free_throws_missed_per100 - L x (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100)) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 3), uncentred rates; L the same ratio of the league's rates (each uncentred minus centred) | fraction | centred |
| `three_point_pct_vs_league` | three-point % above the league's | (threes_made_per100 - L x (threes_made_per100 + threes_missed_per100)) / (threes_made_per100 + threes_missed_per100 + 3), uncentred rates; L the same ratio of the league's rates (each uncentred minus centred) | fraction | centred |
| `two_point_pct_vs_league` | two-point % above the league's | (twos_made_per100 - L x (twos_made_per100 + twos_missed_per100)) / (twos_made_per100 + twos_missed_per100 + 3), uncentred rates; L the same ratio of the league's rates (each uncentred minus centred) | fraction | centred |
| `free_throw_pct_vs_league` | free-throw % above the league's | (free_throws_made_per100 - L x (free_throws_made_per100 + free_throws_missed_per100)) / (free_throws_made_per100 + free_throws_missed_per100 + 2), uncentred rates; L the same ratio of the league's rates (each uncentred minus centred) | fraction | centred |
| `assists_per_used_possession_vs_league` | assists per possession used above the league's | (assists_per100 - L x (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100)) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100 + 3), uncentred rates; L the same ratio of the league's rates (each uncentred minus centred) | fraction | centred |
| `turnovers_per_used_possession_vs_league` | turnovers per possession used above the league's | (turnovers_per100 - L x (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100)) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100 + 3), uncentred rates; L the same ratio of the league's rates (each uncentred minus centred) | fraction | centred |
| `off_rebound_share_vs_league` | offensive-rebound share of his rebounds above the league's | (off_rebounds_per100 - L x (off_rebounds_per100 + def_rebounds_per100)) / (off_rebounds_per100 + def_rebounds_per100 + 3), uncentred rates; L the same ratio of the league's rates (each uncentred minus centred) | fraction | centred |
| `two_point_expected_pct` | difficulty of his twos | (shot_twos_expected_made + 50 league_two_point_pct) / (shot_twos_attempted + 50) | fraction | level |
| `three_point_expected_pct` | difficulty of his threes | (shot_threes_expected_made + 50 league_three_point_pct) / (shot_threes_attempted + 50) | fraction | level |
| `two_point_pct_over_expected` | two-point shot-making | (shot_twos_made - shot_twos_expected_made) / (shot_twos_attempted + 250) | makes per attempt | level |
| `three_point_pct_over_expected` | three-point shot-making | (shot_threes_made - shot_threes_expected_made) / (shot_threes_attempted + 450) | makes per attempt | level |
| `expected_points_per_shot` | expected points per shot | (2 shot_twos_expected_made + 3 shot_threes_expected_made + 100 league_points_per_shot) / (shot_twos_attempted + shot_threes_attempted + 100) | points | level |
| `points_per_shot_over_expected` | points per shot above expected | (2 (shot_twos_made - shot_twos_expected_made) + 3 (shot_threes_made - shot_threes_expected_made)) / (shot_twos_attempted + shot_threes_attempted + 300) | points | level |
| `possessions` | possessions behind the row: offensive on an offence row, defensive on a defence row | panel column | possessions | level |
| `team_possession_share` | share of team possessions played | panel column | share | level |
| `games_started_share` | share of games started | panel column | share | level |
| `age` | age | panel column | years | level |
| `garbage_time_share` | share of possessions in garbage time | panel column | share | level |
| `closeness` | how close his games were: mean of 1 / max(\|margin\|, 1) | panel column | 1 / points | level |
| `mean_abs_margin` | average score gap while he played | panel column | points | level |
| `shot_twos_attempted` | two-point attempts | panel column | attempts | level |
| `shot_twos_made` | two-point makes | panel column | makes | level |
| `shot_twos_expected_made` | expected two-point makes, league rates from his spots | panel column | makes | level |
| `shot_threes_attempted` | three-point attempts | panel column | attempts | level |
| `shot_threes_made` | three-point makes | panel column | makes | level |
| `shot_threes_expected_made` | expected three-point makes, league rates from his spots | panel column | makes | level |
| `league_two_point_pct` | league two-point % | panel column | fraction | level |
| `league_three_point_pct` | league three-point % | panel column | fraction | level |
| `league_points_per_shot` | league points per shot | panel column | points | level |
| `seasons_before` | seasons played before this one | panel column | seasons | level |
| `career_possessions_before_k` | career possessions before this season | panel column | thousands | level |
| `age_at_first_season` | age entering the league | panel column | years | level |
| `height_in` | height | panel column | inches | level |
| `weight_lb` | weight | panel column | pounds | level |
| `draft_pick` | draft position | panel column | pick | level |
| `seasons_with_team` | seasons with his current team | panel column | seasons | level |
| `teams_this_season` | teams he played for this season | panel column | teams | level |
| `seasons_before_from_debut` | seasons played before this one, counted from his debut (before 1997 too) | panel column | seasons | level |
| `career_possessions_before_from_debut_k` | career possessions before this season, counted from his debut (before 1997 from his minutes) | panel column | thousands | level |
| `age_at_debut` | age in his first NBA season (before 1997 too) | panel column | years | level |
| `seasons_with_team_from_debut` | seasons with his current team, counted from his debut (before 1997 too) | panel column | seasons | level |
| `on_court_scored_per100` | team points scored per 100, him on court | panel column | per 100 possessions | centred |
| `on_court_allowed_per100` | team points allowed per 100, him on court | panel column | per 100 possessions | centred |
| `on_court_off_possessions` | offensive possessions on court | panel column | possessions | level |
| `on_court_def_possessions` | defensive possessions on court | panel column | possessions | level |
| `off_court_scored_per100` | team points scored per 100, him off court | panel column | per 100 possessions | centred |
| `off_court_allowed_per100` | team points allowed per 100, him off court | panel column | per 100 possessions | centred |
| `off_court_off_possessions` | offensive possessions off court | panel column | possessions | level |
| `off_court_def_possessions` | defensive possessions off court | panel column | possessions | level |
| `on_minus_off_scored_per100` | on court minus off court, points scored | panel column | per 100 possessions | centred |
| `on_minus_off_allowed_per100` | on court minus off court, points allowed | panel column | per 100 possessions | centred |
| `playoff_share` | share of possessions in the playoffs | panel column | share | level |
| `row_possessions` | possessions behind the row | panel column | possessions | level |
| `row_seasons` | seasons the row covers | panel column | seasons | level |
| `season_from_2017` | 1 for a season from 2017 on, 0 before (an era marker) | season >= 2017 | flag | level |
<!-- end generated -->
