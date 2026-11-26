# Glossary

The words this project uses, each in a few lines, in the order a reader meets them, so a term is defined before
another entry leans on it; then the raw box counts; then every column the box-score prior can read, written from the
code. Reports to the owner use the plain names, never the codes. Every setting named here has its value and the
experiment that set it in `src/openrapm/settings.py`.

## The data

**Stint.** A stretch of play with the same ten players on the court; a substitution or the end of a period starts a
new one. The 2003 playoffs, 88 games, give 2,653 stints, about 30 a game.

**Possession.** One team's turn with the ball, from when it gains the ball until the other team gains it or the
period ends; an offensive rebound continues it. Every rating is in points per 100 possessions.

**Garbage time.** A stint in the fourth quarter with a margin of 20 or more, or of 15 or more with under six minutes
left (`config.yaml`, `garbage_time`). It is flagged, and the design carries the flag as a context column; a
garbage-time row weighs the same as any other.

**Season panel.** One row per player, season and side: his possessions, padded box-score rates, role, career, body,
team, his team's record with him on and off the court, the play-by-play parts and the hustle rates
(`outputs/season_panel.parquet`).

**Padded rate.** A per-100 box-score rate pulled toward the rate of players with about as many possessions, by k
possessions' worth of it: padded = (n x rate + k x bin rate) / (n + k), with n his possessions. The bin rate is that
season's rate among the players of his possession bin, six ranges of possessions equally wide on a log scale. k is
measured per count and season from every player's odd- against even-numbered games: the noise variance of one
possession's worth of the rate over the variance of the true rates between players (`box_rates.py`). A 200-possession
player with 2.1 steals per 100, k = 450 and a bin rate of 1.4 is padded to (200 x 2.1 + 450 x 1.4) / 650 = 1.6. The
bin rate is itself padded toward the league rate by the bin's possessions, so early in a season, when a bin's rate
rests on a few dozen events, the league rate carries most of the weight.

**Hustle inputs.** Screen assists, points from screen assists, box-outs, loose balls recovered, deflections, contested
twos and threes and charges drawn, from the league's hustle box scores, one per game since 2017; each padded per 100
possessions and read against the league's rate: (player - league) / league. Missing before 2017. A season in progress
reads the league's season-to-date totals instead (`data/dashboard_rates.py`).

**Self-creation.** The owner's estimate of the shots a player creates for himself, per 100 possessions: (shot
attempts + 0.44 x free-throw attempts) x (1 - share assisted), with share assisted = 0.73 - 0.03 x assists per 100 -
0.01 x field goals made per 100, clipped to 0-1. An input of the offensive box-score prior.

**Season in progress.** The season whose Finals are not over, 2027 as of this writing (`data/seasons.py`). It is
rated every morning and never trained on.

## How a rating is made

**RAPM** (regularized adjusted plus-minus). A ridge regression of every stint's points per 100 possessions on the
ten players on the court, plus context columns; a player's coefficient is his rating. In the published tables
positive is good on both ends.

**Design.** The stints of one or more seasons as the rows of that regression: one column per player per side, offense
and defense, plus the context columns: season intercept, home court, playoffs, garbage time, score margin, margin x
share of the game left. Each row's target is its points per 100 possessions, adjusted as the two targets below say,
and its weight its possessions (`design.py`).

**The two targets.** What each side's ridge explains, per 100 possessions. Offense,
`points_with_expected_free_throws`: points with each made free throw replaced by the shooter's padded free-throw
percentage. Defense, `points_with_expected_opponent_threes_late_as_scored`: the same, less 3 x 0.75 x (threes made -
threes expected), so three quarters of the shooters' three-point luck is taken off; a three of a period's last two
seconds counts as it happened. Scoring is always against the points actually scored.

**Penalty.** The ridge's lambda. Seen alone, a player with n possessions keeps about n / (n + lambda) of his raw
impact, the coefficient an unpenalized fit would give him. With lambda = 40,000, a player with 54,000 possessions keeps
57% of it and one with 2,600 keeps 6%.

**Label.** What the box-score prior learns to predict: a player's RAPM from one fit over every season except the
unseen ones, one number per side for all those seasons together (step 1, `rapm.py`).

**Leave-season-out.** Fitted without the rated season, and in the test setting without the seasons either side of it.
Each season's normal equations are stored once, so leaving a season out is one solve on the other seasons' sums, not
thirty refits.

**Un-shrink.** Dividing a coefficient by the share n / (n + lambda) the ridge kept, so long and short careers sit on
one scale. The division stops at 4,444 possessions and the rest comes from the level of the player's possession tier.
Only the defensive label is un-shrunk. Example: with lambda = 40,000 the division stops at 4,444 / 44,444 = 0.10, so a
1,000-possession player, kept share 0.024, with coefficient 0.05, in a tier whose level is -3.0, gets 0.05 / 0.10 + (1
- 0.24) x -3.0 = -1.77.

**Box-score prior.** LightGBM's, gradient-boosted trees', prediction of a player's label from his season-panel row:
18 inputs on offense and 18 on defense, marked in the table below, plus `row_possessions` and `row_seasons`. Step 2,
`box_prior.py`.

**Booster.** One fitted LightGBM model. `--save-models` writes every rated season's boosters to
`outputs/prior_models_NAME.pkl`.

**Bags.** The box-score prior averages 10 versions of itself, each trained on a resample of the players drawn with
replacement; a player drawn twice counts twice in that version.

**Career row and chunk rows.** A player's training rows for the prior. The career row averages his box score over
every season outside the unseen ones; a chunk row averages a run of 1, 2 or 3 consecutive seasons. Every row carries
his career label, and the chunk rows together weigh what the career row weighs. Example: seasons of 1,000, 2,000 and
3,000 possessions and a label resting on 6,000 give six chunk rows weighing 300, 600, 900, 900, 1,500 and 1,800,
together 6,000.

**Part-season rows.** Each training season so far at a quarter, a half and three quarters of its regular season, one
chunk row each, so the prior has seen early-season box scores like the ones a season in progress has.

**Out-of-player.** Players are sorted by label and assigned to five player folds, 0,1,2,3,4,4,3,2,1,0,..., and each
player's prior comes from the booster fitted without his fold, so it cannot recognize his box score and return his own
label.

**Held-back games.** Each season, 1997 to 2026, rated from three quarters of its games, the fit games, and scored on
the other quarter, which the rating never saw. The games are assigned at random to four folds, three times over:
twelve folds a season (`openrapm folds`, `notes/WITHIN_SEASON_LEDGER.md`). The code calls them the within-season
folds. Box Adj's weights and the shrink's multipliers are fitted on them.

**Box Adj.** The part of a player's prior that corrects his box-score prior for who he played with and against; the
Detail view's column, the code's second prior. The prior is a straight line over his box-score prior on both ends,
his teammates' and opponents' box-score priors with him on and off the court, his box rates, role and body, and his
team's shots by zone, free throws, offensive rebounds and turnovers with him on and off the court, on defense the
opponents': 56 inputs an end and two more on defense, his offensive rebound share and his close-game share of team possessions, every one from the rated season and none of them his age (`second_prior.py`). Its weights are fitted on
held-back games through the ridge (`rating_line.py`). Each team's total is held, so Box Adj only moves credit inside a
team. The Detail view's Box score column refits his own stats' inputs alone, and Box Adj is the rest of the same
prior.

**Zones.** Where a shot was taken from, five ways: the rim, 3 feet or closer; short mid, to 13 feet; long mid, to the
three-point line; three, to 26 feet; long three, 27 feet or more. Box Adj reads a team's makes and misses in each.

**Half-life.** Box Adj's line counts each season it is fitted on 0.5 to the power of its distance from the rated
season over 8: a season 8 years away counts half, 16 away a quarter.

**The rating, prior-informed RAPM.** Step 3 (`prior_rapm.py`): the season's RAPM with each player pulled toward a
multiple of his prior instead of toward 0, penalty 13,037. A rating is s x prior + r, where r is how far the season's
games moved him.

**Prior scale.** That multiple, s: one per side and season, the coefficient on the summed priors of the five on the
court, unpenalized, so the season's games set it. In 2026 it is 0.84 on offense and 0.71 on defense. A season in
progress holds it at the 8 nearest finished seasons' mean through a quarter of its regular season, then moves on a
straight line to its own at three quarters.

**Centering.** Per season and side, the possession-weighted mean rating is subtracted, so the average possession is
played by a 0 player.

**Prior part and games' part.** The two parts of a rating after step 3: the prior part s x prior (`prior_off`,
`prior_def`), and the games' part r, also called u (`u_off`, `u_def`).

**Standard error of a rating.** How far a rating would land from the one shown had the season's games come out as
another draw of the same schedule (`rating_error.py`): the season is rated 30 more times, each without a tenth of its
games, and the spread of those ratings is scaled up to a whole season's. A regular reads about 0.8 on his total. It
holds the games' noise in the box score, Box Adj and the ridge; it leaves out how far the box-score prior can miss a
player whatever games are played.

**The shrink.** Step 4 (`prior_shrink.py`): each part of a rating multiplied by what holds up on held-back games, in
two layers. Each part is split into the player's team mean and his deviation from it; the team means take four
multipliers per season and the deviations four of their own, the prior's part and the games' part on offense and
defense. The prior's part is multiplied down, the games' part up: in the published tables of 2026-11-16 the team
layer's prior multiplier runs 0.600 to 0.707 on offense and 0.629 to 0.799 on defense, its games' multiplier 1.292 to
1.439 and 1.224 to 1.309. Each season's multipliers come from the 8 fold seasons nearest it. The code calls it the
prior shrink.

**Defense factor.** One more factor on the shrink's two defensive team multipliers: 0.8682 before 2017 and 0.8704 from
2017. The within layer's multipliers take 0.7663 on offense and 0.9968 on defense.

**Read to a full season.** A fold's rating is built from three quarters of a season, the published one from all of it.
So each multiplier is fitted twice, on folds holding back a quarter and on folds holding back a tenth, and read off
the straight line through the two at nothing held back.

**Lineup swap.** Two lineups of one team in one season that share four players, a same-four comparison. The four are
held fixed, so the difference between the two lineups' results is evidence about the fifth player of each.

**Swap adjustment.** Step 5 (`swap_adjust.py`), off since 2026-10-20: a ridge regression of each lineup swap's
unexplained result on the two swapped players' box-score, role and body numbers predicted which kinds of player the
ratings gave too much or too little credit, and half of that prediction was applied. Box Adj learns from the same
comparisons before the ridge.

**Give-back.** In the swap adjustment, each team's total adjustment taken back from its own players in proportion to
their share of its possessions, so no team's total moved.

**Spread held.** In the swap adjustment, each player's distance from his main team's mean scaled, one factor per
season and side, until the spread inside teams equaled step 4's.

**Main team.** The team a player played the most possessions for that season.

**Product setting and test setting.** Product builds the published ratings: a season's labels, the prior's training
rows, Box Adj's line and step 4's multipliers all leave that season out. Test builds the table the year-over-year
test scores: they also leave out the seasons either side, because those are the seasons the test predicts
(`settings.py`, `PRODUCT` and `TEST`).

## How a change is judged

**Within-season WOWY test.** The test that decides since 2026-10-15 (`scripts/checks/151_wowy.py`): each season is
rated from three quarters of its games, twelve times over, and scored stint by stint on the quarter held back, with
each team's own level in those games taken out, so what is scored is how the ratings split credit among teammates.
WOWY is with or without you. A change must be better in each of 1997-2006, 2007-2016 and 2017-2026, read beside the
noise control.

**Year-over-year test.** The test that guards it (`scripts/checks/63_yoy.py`), and decided until 2026-10-15: each
season's ratings predict every stint of the season before and the season after from the ten players' ratings alone,
refitting only that season's intercept and home edge; the misses are summed to team-games. 28 scored seasons, each
predicted from both sides: 56 readings. The score is the root of the team-game mean squared error times the square
root of 2 / pi, in points per 100 (`game_armse`, `scoring.py`): the incumbent reads 8.6908. A change must not be worse
at a paired z of +2, read with each side's spread matched to the incumbent's.

**Team-game.** One team's offense in one game: the unit the year-over-year test scores, because the noise of single
possessions cancels inside a team's hundred-odd possessions before the miss is squared.

**Paired z.** For each reading, one table's error minus the other's; the mean of those differences divided by its
standard error, a paired t-statistic. Below zero is better. The within-season WOWY test pairs its 30 seasons, the
year-over-year test its 56 readings.

**Incumbent.** The table every candidate is scored against: `INC_NAME` at the top of `scripts/experiment_chain.sh`,
today `rd143`, experiment 143 built at the test setting, with its held-back games `rd143_early` and
`rd143_within`.

**Noise control.** The incumbent rebuilt with nothing changed but which players share each of the box-score prior's
five folds (`--fold-assignment 1`): `CONTROL_NAME` at the top of `scripts/experiment_chain.sh`, today
`noise_rd143`. It shows how far every reading moves when nothing real changes: -0.012 on the within-season WOWY
test, paired z -1.37, better in 17 of 30 seasons.

**Replicate.** The same change built a second time with another random assignment of players to the prior's folds,
so every reading is seen twice (`scripts/checks/135_robust_review.py`).

**Swap test.** Does a ratings table order teammates the way their lineup swaps do: for every lineup swap in the
seasons either side, the two swapped players' rating gap against the swap's result
(`scripts/checks/90_swap_test.py`).

**Trade set.** For each rated season, the team-games of that season and the seasons either side, each player's share
of his team's possessions in each, and the rated season's ratings subtracted; what is left is ridged back onto the
players. A player's coefficient is what his comings and goings say his rating did not already know
(`scripts/checks/70_tradeset.py`).

**Trade loss.** Two tables compared by the size of their trade-set corrections on the same players, paired by season
(`scripts/checks/73_tradeloss.py`). It counts every player once, so it sees the bench players the year-over-year test,
an error per team-game, barely weighs.

**Held-out audit.** On held-back games, are players high on some input rated too high or too low
(`scripts/checks/138_heldout_audit.py`).

**Quality tiers.** Players grouped by the incumbent's rank in the rated season: top 30, 31-90, 91-150, 151-300, 301 and
below (`scripts/checks/88_yoy_by_player.py`).

**Consensus.** A blend of public all-in-one metrics for 2024-26 (`data/external/consensus.csv`). The ratings are
compared with it by Spearman's rank correlation over players with 1,000 or more possessions, each pooled over the three
seasons (`scripts/checks/64_consensus_report.py`). A sanity check, never a fitting target: a gross miss can veto a
change, a marginal one cannot.

## The site

**Detail view.** The Ratings page's Detail button: each published rating split, offense and defense separately, into
Box score + Box Adj + On adj + Off adj + Team SOS, which add up to the rating. On adj is the part of the games' part
from the possessions he was on the court for, Off adj the part from his team's possessions without him, Team SOS the
part from every other possession (`decomposition.py`).

**The Bias page's consensus.** For the season the pages open on, the weighted mean of five outside systems' ratings
per side, 30% xRAPM, 20% xrapm.com's 3-year RAPM, 20% LOKI, 15% DARKO and 15% Pulse (the owner's weights, 2026-11-05;
`bias.py`), over the players all five list. The page shows where OpenRAPM rates players higher or lower than it, per
one standard deviation of each box-score stat and of on-court and off-court plus-minus. Never a fitting target.

**Width match.** Before the Bias page subtracts, each side of OpenRAPM is multiplied once, league-wide, to the
consensus's standard deviation, so a row shows where the two disagree about who is good and not that one is spread
wider. The page prints both multipliers.

**Similarity.** For each player-season with 1,000 or more offensive possessions, the 50 nearest player-seasons of other
players by the straight-line distance between their standardized inputs (`similarity.py`).

**Power.** Each team as it has played lately: its players' current ratings times their shares of its possessions
over its last 10 games, the latest five counted double and blowout possessions half, shifted so the average team
reads 0. The page's one rating multiplies Power and adds the team's last-10 shooting, rebounding, turnover and
free-throw stats, with weights that fade as the season goes on (`power_forecast.py`).

**Morning job.** `scripts/daily_update.sh`: at 6 AM Eastern it fetches the night's games, rates the season in progress
in place of yesterday's, and publishes the site's data files (`PIPELINE.md`, "The morning job").

**WOSA**, wins over starter average. How many regular-season wins a player added to his team over an average starter,
offense, defense and pace together (`wins.py`). No rating moves. A +9.0 player against a bar of +0.8, over 5,300
possessions at each end at an average weight: 8.2 x 5,300 / 100 / 30 = 14.5 wins. It reads each rating from the
regular season alone, so it stops changing when the regular season ends.

**Starter average.** The bar WOSA is read against: the mean rating of the season's 150 players with the most
regular-season games started, minutes breaking ties. It sits above zero, since ratings are centered on the average
possession and starters are better than that.

**Win model.** The chance the home team wins from a margin with a share of the game left, fitted on each finished
season's games: Phi((lead kept x margin + home edge x share left) / spread). The spread is the measured standard
deviation of what is left of a final margin, by time left.

**Lead kept.** The share of a lead that reaches the final margin, a line in the share of the game left: 0.93 at the end
and 0.74 at the half, in every era. A team ahead coasts and a team behind presses.

**Leverage.** What one point moves the win model by at a stint's game state. WOSA uses it as a weight inside each game,
a stint's leverage over its own game's mean: every game carries the same total weight, and 1 is an average possession
of that game, about 2.3 late in a close game, 0.03 in garbage time.

**Points per win.** Rating points, weighted by leverage inside each game, for one win: 30, read from how teams'
records follow their lineups' regular-season ratings over 30 seasons.

**Pace effect.** A player's effect on the possessions per 48 minutes each team plays, from a ridge regression of each
stint's pace on the ten players on the court.

## The raw box counts

The thirteen per-game counts every design carries, in the order of `config.yaml`'s `box_counts`. They keep the NBA's
abbreviations in the box table and the stint counters; the per-100 rates built from them have plain names, in the
table below, which the Bias page's rows use.

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

Files and notes from before 2026-10-11 use older column names, `poss_pct`, `onc_o`, `chunk_poss` and others;
`src/openrapm/data/legacy_names.py` maps each to the name below.

## Every column the prior can read

Written from `src/openrapm/features.py` by `openrapm glossary`; edit the catalogue there, not the table here. The
shipped column says which sides of the shipped box-score prior read each input, offense, defense or both, from
`settings.PriorSettings.feature_set`; the rest are candidates an experiment may try. A centered column has the
possession-weighted league mean subtracted, so 0 is a league-average rate; a level column keeps its units.

<!-- generated: features -->
| name | shipped | meaning | formula | unit | scale |
|---|---|---|---|---|---|
| `threes_made_per100` |  | made threes | padded rate minus the league level | per 100 possessions | centered |
| `threes_missed_per100` |  | missed threes | padded rate minus the league level | per 100 possessions | centered |
| `twos_made_per100` |  | made twos | padded rate minus the league level | per 100 possessions | centered |
| `twos_missed_per100` |  | missed twos | padded rate minus the league level | per 100 possessions | centered |
| `free_throws_made_per100` |  | made free throws | padded rate minus the league level | per 100 possessions | centered |
| `free_throws_missed_per100` |  | missed free throws | padded rate minus the league level | per 100 possessions | centered |
| `off_rebounds_per100` | offense | offensive rebounds | padded rate minus the league level | per 100 possessions | centered |
| `def_rebounds_per100` | defense | defensive rebounds | padded rate minus the league level | per 100 possessions | centered |
| `assists_per100` | offense | assists | padded rate minus the league level | per 100 possessions | centered |
| `turnovers_per100` |  | turnovers | padded rate minus the league level | per 100 possessions | centered |
| `steals_per100` | both | steals | padded rate minus the league level | per 100 possessions | centered |
| `blocks_per100` | defense | blocks | padded rate minus the league level | per 100 possessions | centered |
| `fouls_per100` |  | personal fouls | padded rate minus the league level | per 100 possessions | centered |
| `threes_made_per100_uncentred` |  | made threes, league level kept | padded rate | per 100 possessions | level |
| `threes_missed_per100_uncentred` |  | missed threes, league level kept | padded rate | per 100 possessions | level |
| `twos_made_per100_uncentred` |  | made twos, league level kept | padded rate | per 100 possessions | level |
| `twos_missed_per100_uncentred` |  | missed twos, league level kept | padded rate | per 100 possessions | level |
| `free_throws_made_per100_uncentred` |  | made free throws, league level kept | padded rate | per 100 possessions | level |
| `free_throws_missed_per100_uncentred` |  | missed free throws, league level kept | padded rate | per 100 possessions | level |
| `off_rebounds_per100_uncentred` |  | offensive rebounds, league level kept | padded rate | per 100 possessions | level |
| `def_rebounds_per100_uncentred` |  | defensive rebounds, league level kept | padded rate | per 100 possessions | level |
| `assists_per100_uncentred` |  | assists, league level kept | padded rate | per 100 possessions | level |
| `turnovers_per100_uncentred` |  | turnovers, league level kept | padded rate | per 100 possessions | level |
| `steals_per100_uncentred` |  | steals, league level kept | padded rate | per 100 possessions | level |
| `blocks_per100_uncentred` |  | blocks, league level kept | padded rate | per 100 possessions | level |
| `fouls_per100_uncentred` |  | personal fouls, league level kept | padded rate | per 100 possessions | level |
| `def_rebounds_of_free_throws_per100` |  | defensive rebounds of missed free throws | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `def_rebounds_of_threes_per100` |  | defensive rebounds of missed threes | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `def_rebounds_of_rim_twos_per100` |  | defensive rebounds of missed twos within 6 ft of the rim, or logged without a location | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `def_rebounds_of_outside_twos_per100` |  | defensive rebounds of missed twos from beyond 6 ft | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `bad_pass_steals_per100` |  | steals of a pass (a bad-pass turnover) | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `lost_ball_steals_per100` |  | steals of the ball from a dribbler (a lost-ball turnover) | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `assisted_points_per100` |  | points from assisted field goals | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `unassisted_points_per100` |  | points from unassisted field goals | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `corner_threes_attempted_per100` |  | corner threes attempted | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `above_break_threes_attempted_per100` |  | above-the-break threes attempted | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `assisted_threes_made_per100` |  | assisted made threes | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `unassisted_threes_made_per100` |  | unassisted made threes | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `russell_blocks_per100` |  | blocks his team recovered (Russell blocks) | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `rim_blocks_per100` |  | blocks of twos within 6 ft of the rim, or logged without a location | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `mid_range_blocks_per100` |  | blocks of twos from beyond 6 ft | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `three_point_blocks_per100` |  | blocks of threes | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `offensive_fouls_per100` |  | offensive fouls committed | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `stolen_turnovers_per100` |  | turnovers stolen by the defense (live-ball turnovers) | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `assists_to_rim_per100` |  | assists on makes at the rim (within 6 ft, or logged without a location) | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `assists_to_mid_range_per100` |  | assists on made twos from beyond 6 ft | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `assists_to_corner_threes_per100` |  | assists on made corner threes | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `assists_to_above_break_threes_per100` |  | assists on made above-the-break threes | padded play-by-play count per 100 minus the league level | per 100 possessions | centered |
| `unassisted_points_per100_uncentred` |  | points from unassisted field goals, league level kept | padded play-by-play count per 100 | per 100 possessions | level |
| `unassisted_threes_made_per100_uncentred` |  | unassisted made threes, league level kept | padded play-by-play count per 100 | per 100 possessions | level |
| `drives_vs_league_avg` |  | drives to the basket | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `drive_points_vs_league_avg` |  | points scored on drives | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `potential_assists_vs_league_avg` |  | potential assists (passes that led to a shot) | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `passes_made_vs_league_avg` |  | passes made | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `touches_vs_league_avg` |  | touches of the ball | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `paint_touches_vs_league_avg` |  | touches in the paint | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `post_touches_vs_league_avg` |  | post touches | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `elbow_touches_vs_league_avg` |  | elbow touches | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `minutes_with_ball_vs_league_avg` |  | minutes with the ball in his hands | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `miles_on_offense_vs_league_avg` |  | miles run on offense | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `off_reb_chances_vs_league_avg` |  | offensive rebound chances (a rebound within his reach) | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `off_reb_contested_vs_league_avg` |  | contested offensive rebounds | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `def_reb_chances_vs_league_avg` |  | defensive rebound chances (a rebound within his reach) | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `def_reb_contested_vs_league_avg` |  | contested defensive rebounds | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `rim_shots_defended_vs_league_avg` |  | shots at the rim he was the closest defender on | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `twos_defended_vs_league_avg` |  | two-point shots he was the closest defender on | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `threes_defended_vs_league_avg` |  | three-point shots he was the closest defender on | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `miles_on_defense_vs_league_avg` |  | miles run on defense | tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `rim_makes_over_expected_per_shot` |  | shots made within 6 ft against him, above what the league's shooters make against those defenders, per shot faced | (makes - expected makes) / (shots faced + pad); the pad is the shots in data/dashboard_rates.py | makes per shot | centered |
| `twos_makes_over_expected_per_shot` |  | two-point shots made against him above expected, per shot faced | (makes - expected makes) / (shots faced + pad); the pad is the shots in data/dashboard_rates.py | makes per shot | centered |
| `threes_makes_over_expected_per_shot` |  | three-point shots made against him above expected, per shot faced | (makes - expected makes) / (shots faced + pad); the pad is the shots in data/dashboard_rates.py | makes per shot | centered |
| `deflections_vs_league_avg` | defense | deflections | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `contested_twos_vs_league_avg` | defense | contested two-point shots | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `contested_threes_vs_league_avg` | defense | contested three-point shots | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `charges_drawn_vs_league_avg` | defense | charges drawn | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `loose_balls_recovered_vs_league_avg` | both | loose balls recovered | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `def_box_outs_vs_league_avg` | defense | defensive box-outs | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `off_box_outs_vs_league_avg` | offense | offensive box-outs | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `screen_assists_vs_league_avg` | offense | screen assists | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `screen_assist_points_vs_league_avg` | offense | points from his screen assists | hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / league rate | fraction above the league average | centered |
| `deflections_vs_league_avg_shuffled` |  | deflections, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centered |
| `contested_twos_vs_league_avg_shuffled` |  | contested two-point shots, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centered |
| `contested_threes_vs_league_avg_shuffled` |  | contested three-point shots, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centered |
| `charges_drawn_vs_league_avg_shuffled` |  | charges drawn, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centered |
| `loose_balls_recovered_vs_league_avg_shuffled` |  | loose balls recovered, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centered |
| `def_box_outs_vs_league_avg_shuffled` |  | defensive box-outs, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centered |
| `off_box_outs_vs_league_avg_shuffled` |  | offensive box-outs, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centered |
| `screen_assists_vs_league_avg_shuffled` |  | screen assists, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centered |
| `screen_assist_points_vs_league_avg_shuffled` |  | points from his screen assists, shuffled among the season's players (a placebo) | the hustle column with each season's player profiles shuffled jointly, missing kept missing | fraction above the league average | centered |
| `attackers_guarded_scoring_vs_league_avg` |  | the scoring of the attackers he guarded (points per matchup possession without the rows against him), against the league's | (attackers' points per matchup possession, each without his rows against the player, padded 50 possessions toward the league's, mean weighted by possessions - league rate) / league rate; the second sign-flipped | fraction above the league average | centered |
| `defenders_faced_stinginess_vs_league_avg` |  | how stingy the defenders were who guarded him (points allowed per matchup possession without the rows against him, sign-flipped), against the league's | (attackers' points per matchup possession, each without his rows against the player, padded 50 possessions toward the league's, mean weighted by possessions - league rate) / league rate; the second sign-flipped | fraction above the league average | centered |
| `points_per100` | both | points | 2 twos_made_per100 + 3 threes_made_per100 + free_throws_made_per100 | per 100 possessions | centered |
| `shots_per100` | both | shot attempts | twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 | per 100 possessions | centered |
| `free_throw_attempts_per100` | offense | free-throw attempts | free_throws_made_per100 + free_throws_missed_per100 | per 100 possessions | centered |
| `three_point_attempts_per100` |  | three-point attempts | threes_made_per100 + threes_missed_per100 | per 100 possessions | centered |
| `used_possessions_per100` | defense | possessions he finishes | twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100 | per 100 possessions | centered |
| `rebounds_per100` |  | rebounds | off_rebounds_per100 + def_rebounds_per100 | per 100 possessions | centered |
| `steals_plus_blocks_per100` | defense | steals plus blocks | steals_per100 + blocks_per100 | per 100 possessions | centered |
| `assists_minus_turnovers_per100` | offense | assists minus turnovers | assists_per100 - turnovers_per100 | per 100 possessions | centered |
| `threes_minus_twos_attempted_per100` |  | threes minus twos taken | threes_made_per100 + threes_missed_per100 - twos_made_per100 - twos_missed_per100 | per 100 possessions | centered |
| `potential_assists_estimated_per100` |  | potential assists, estimated from where the assists went | 1.656 assists_to_rim_per100 + 2.261 assists_to_mid_range_per100 + 2.057 assists_to_corner_threes_per100 + 2.503 assists_to_above_break_threes_per100 | per 100 possessions | centered |
| `effective_fg_pct` | offense | effective field-goal % | (twos_made_per100 + 1.5 threes_made_per100 + 3 x 0.5) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 3), uncentered rates | fraction | level |
| `true_shooting_pct` | both | true shooting % | (2 twos_made_per100 + 3 threes_made_per100 + free_throws_made_per100 + 6 x 0.54) / (2 twos_made_per100 + 2 twos_missed_per100 + 2 threes_made_per100 + 2 threes_missed_per100 + 0.88 free_throws_made_per100 + 0.88 free_throws_missed_per100 + 6), uncentered rates | fraction | level |
| `three_point_attempt_share` | offense | share of shots that are threes | (threes_made_per100 + threes_missed_per100 + 3 x 0.25) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 3), uncentered rates | fraction | level |
| `free_throws_per_shot` |  | free throws per shot | (free_throws_made_per100 + free_throws_missed_per100 + 3 x 0.25) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 3), uncentered rates | fraction | level |
| `three_point_pct` |  | three-point % | (threes_made_per100 + 3 x 0.34) / (threes_made_per100 + threes_missed_per100 + 3), uncentered rates | fraction | level |
| `two_point_pct` |  | two-point % | (twos_made_per100 + 3 x 0.48) / (twos_made_per100 + twos_missed_per100 + 3), uncentered rates | fraction | level |
| `free_throw_pct` |  | free-throw % | (free_throws_made_per100 + 2 x 0.75) / (free_throws_made_per100 + free_throws_missed_per100 + 2), uncentered rates | fraction | level |
| `assists_per_used_possession` |  | assists per possession used | (assists_per100 + 3 x 0.18) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100 + 3), uncentered rates | fraction | level |
| `turnovers_per_used_possession` |  | turnovers per possession used | (turnovers_per100 + 3 x 0.13) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100 + 3), uncentered rates | fraction | level |
| `off_rebound_share` | offense | offensive-rebound share of his rebounds | (off_rebounds_per100 + 3 x 0.22) / (off_rebounds_per100 + def_rebounds_per100 + 3), uncentered rates | fraction | level |
| `effective_fg_pct_vs_league` |  | effective field-goal % above the league's | (twos_made_per100 + 1.5 threes_made_per100 - L x (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100)) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 3), uncentered rates; L the same ratio of the league's rates (each uncentered minus centered) | fraction | centered |
| `true_shooting_pct_vs_league` |  | true shooting % above the league's | (2 twos_made_per100 + 3 threes_made_per100 + free_throws_made_per100 - L x (2 twos_made_per100 + 2 twos_missed_per100 + 2 threes_made_per100 + 2 threes_missed_per100 + 0.88 free_throws_made_per100 + 0.88 free_throws_missed_per100)) / (2 twos_made_per100 + 2 twos_missed_per100 + 2 threes_made_per100 + 2 threes_missed_per100 + 0.88 free_throws_made_per100 + 0.88 free_throws_missed_per100 + 6), uncentered rates; L the same ratio of the league's rates (each uncentered minus centered) | fraction | centered |
| `three_point_attempt_share_vs_league` |  | share of shots that are threes above the league's | (threes_made_per100 + threes_missed_per100 - L x (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100)) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 3), uncentered rates; L the same ratio of the league's rates (each uncentered minus centered) | fraction | centered |
| `free_throws_per_shot_vs_league` |  | free throws per shot above the league's | (free_throws_made_per100 + free_throws_missed_per100 - L x (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100)) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 3), uncentered rates; L the same ratio of the league's rates (each uncentered minus centered) | fraction | centered |
| `three_point_pct_vs_league` |  | three-point % above the league's | (threes_made_per100 - L x (threes_made_per100 + threes_missed_per100)) / (threes_made_per100 + threes_missed_per100 + 3), uncentered rates; L the same ratio of the league's rates (each uncentered minus centered) | fraction | centered |
| `two_point_pct_vs_league` |  | two-point % above the league's | (twos_made_per100 - L x (twos_made_per100 + twos_missed_per100)) / (twos_made_per100 + twos_missed_per100 + 3), uncentered rates; L the same ratio of the league's rates (each uncentered minus centered) | fraction | centered |
| `free_throw_pct_vs_league` |  | free-throw % above the league's | (free_throws_made_per100 - L x (free_throws_made_per100 + free_throws_missed_per100)) / (free_throws_made_per100 + free_throws_missed_per100 + 2), uncentered rates; L the same ratio of the league's rates (each uncentered minus centered) | fraction | centered |
| `assists_per_used_possession_vs_league` |  | assists per possession used above the league's | (assists_per100 - L x (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100)) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100 + 3), uncentered rates; L the same ratio of the league's rates (each uncentered minus centered) | fraction | centered |
| `turnovers_per_used_possession_vs_league` |  | turnovers per possession used above the league's | (turnovers_per100 - L x (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100)) / (twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 + turnovers_per100 + 3), uncentered rates; L the same ratio of the league's rates (each uncentered minus centered) | fraction | centered |
| `off_rebound_share_vs_league` |  | offensive-rebound share of his rebounds above the league's | (off_rebounds_per100 - L x (off_rebounds_per100 + def_rebounds_per100)) / (off_rebounds_per100 + def_rebounds_per100 + 3), uncentered rates; L the same ratio of the league's rates (each uncentered minus centered) | fraction | centered |
| `two_point_expected_pct` |  | difficulty of his twos | (shot_twos_expected_made + 50 league_two_point_pct) / (shot_twos_attempted + 50) | fraction | level |
| `three_point_expected_pct` |  | difficulty of his threes | (shot_threes_expected_made + 50 league_three_point_pct) / (shot_threes_attempted + 50) | fraction | level |
| `two_point_pct_over_expected` |  | two-point shot-making | (shot_twos_made - shot_twos_expected_made) / (shot_twos_attempted + 250) | makes per attempt | level |
| `three_point_pct_over_expected` |  | three-point shot-making | (shot_threes_made - shot_threes_expected_made) / (shot_threes_attempted + 450) | makes per attempt | level |
| `expected_points_per_shot` |  | expected points per shot | (2 shot_twos_expected_made + 3 shot_threes_expected_made + 100 league_points_per_shot) / (shot_twos_attempted + shot_threes_attempted + 100) | points | level |
| `points_per_shot_over_expected` |  | points per shot above expected | (2 (shot_twos_made - shot_twos_expected_made) + 3 (shot_threes_made - shot_threes_expected_made)) / (shot_twos_attempted + shot_threes_attempted + 300) | points | level |
| `possessions` |  | possessions behind the row: offensive on an offense row, defensive on a defense row | panel column | possessions | level |
| `team_possession_share` | both | share of team possessions played | panel column | share | level |
| `games_started_share` | defense | share of games started | panel column | share | level |
| `age` |  | age | panel column | years | level |
| `garbage_time_share` |  | share of possessions in garbage time | panel column | share | level |
| `closeness` |  | how close his games were: mean of 1 / max(\|margin\|, 1) | panel column | 1 / points | level |
| `mean_abs_margin` |  | average score gap while he played | panel column | points | level |
| `shot_twos_attempted` |  | two-point attempts | panel column | attempts | level |
| `shot_twos_made` |  | two-point makes | panel column | makes | level |
| `shot_twos_expected_made` |  | expected two-point makes, league rates from his spots | panel column | makes | level |
| `shot_threes_attempted` |  | three-point attempts | panel column | attempts | level |
| `shot_threes_made` |  | three-point makes | panel column | makes | level |
| `shot_threes_expected_made` |  | expected three-point makes, league rates from his spots | panel column | makes | level |
| `league_two_point_pct` |  | league two-point % | panel column | fraction | level |
| `league_three_point_pct` |  | league three-point % | panel column | fraction | level |
| `league_points_per_shot` |  | league points per shot | panel column | points | level |
| `seasons_before` |  | seasons played before this one | panel column | seasons | level |
| `career_possessions_before_k` |  | career possessions before this season | panel column | thousands | level |
| `age_at_first_season` |  | age entering the league | panel column | years | level |
| `height_in` | defense | height | panel column | inches | level |
| `weight_lb` | both | weight | panel column | pounds | level |
| `draft_pick` |  | draft position | panel column | pick | level |
| `seasons_with_team` |  | seasons with his current team | panel column | seasons | level |
| `teams_this_season` |  | teams he played for this season | panel column | teams | level |
| `seasons_before_from_debut` |  | seasons played before this one, counted from his debut (before 1997 too) | panel column | seasons | level |
| `career_possessions_before_from_debut_k` |  | career possessions before this season, counted from his debut (before 1997 from his minutes) | panel column | thousands | level |
| `age_at_debut` |  | age in his first NBA season (before 1997 too) | panel column | years | level |
| `seasons_with_team_from_debut` |  | seasons with his current team, counted from his debut (before 1997 too) | panel column | seasons | level |
| `on_court_scored_per100` |  | team points scored per 100, him on court | panel column | per 100 possessions | centered |
| `on_court_allowed_per100` |  | team points allowed per 100, him on court | panel column | per 100 possessions | centered |
| `on_court_off_possessions` |  | offensive possessions on court | panel column | possessions | level |
| `on_court_def_possessions` |  | defensive possessions on court | panel column | possessions | level |
| `off_court_scored_per100` |  | team points scored per 100, him off court | panel column | per 100 possessions | centered |
| `off_court_allowed_per100` |  | team points allowed per 100, him off court | panel column | per 100 possessions | centered |
| `off_court_off_possessions` |  | offensive possessions off court | panel column | possessions | level |
| `off_court_def_possessions` |  | defensive possessions off court | panel column | possessions | level |
| `on_minus_off_scored_per100` |  | on court minus off court, points scored | panel column | per 100 possessions | centered |
| `on_minus_off_allowed_per100` |  | on court minus off court, points allowed | panel column | per 100 possessions | centered |
| `playoff_share` |  | share of possessions in the playoffs | panel column | share | level |
| `row_possessions` | both | possessions behind the row | panel column | possessions | level |
| `row_seasons` | both | seasons the row covers | panel column | seasons | level |
| `season_from_2017` |  | 1 for a season from 2017 on, 0 before (an era marker) | season >= 2017 | flag | level |
| `self_creation_per100` | offense | self-created shot attempts (the owner's self-creation) | (shot attempts + 0.44 x free-throw attempts) x (1 - share assisted), share assisted = 0.73 - 0.03 x assists - 0.01 x field goals made, clipped to 0-1; uncentered rates | per 100 possessions | level |
| `assisted_makes_inside_per100_uncentred` |  | assisted makes, rim and paint | padded per-game count per 100, level kept | per 100 possessions | level |
| `assisted_makes_two_per100_uncentred` |  | assisted makes, other twos | padded per-game count per 100, level kept | per 100 possessions | level |
| `assisted_makes_three_per100_uncentred` |  | assisted makes, threes | padded per-game count per 100, level kept | per 100 possessions | level |
| `expected_assisted_makes_inside_per100_uncentred` |  | expected assisted makes from the season's cell shares, rim and paint | padded per-game count per 100, level kept | per 100 possessions | level |
| `expected_assisted_makes_two_per100_uncentred` |  | expected assisted makes from the season's cell shares, other twos | padded per-game count per 100, level kept | per 100 possessions | level |
| `expected_assisted_makes_three_per100_uncentred` |  | expected assisted makes from the season's cell shares, threes | padded per-game count per 100, level kept | per 100 possessions | level |
| `misses_inside_per100_uncentred` |  | missed shots, rim and paint | padded per-game count per 100, level kept | per 100 possessions | level |
| `misses_two_per100_uncentred` |  | missed shots, other twos | padded per-game count per 100, level kept | per 100 possessions | level |
| `misses_three_per100_uncentred` |  | missed shots, threes | padded per-game count per 100, level kept | per 100 possessions | level |
| `expected_passed_misses_inside_per100_uncentred` |  | misses expected off a pass, before his lean, rim and paint | padded per-game count per 100, level kept | per 100 possessions | level |
| `expected_passed_misses_two_per100_uncentred` |  | misses expected off a pass, before his lean, other twos | padded per-game count per 100, level kept | per 100 possessions | level |
| `expected_passed_misses_three_per100_uncentred` |  | misses expected off a pass, before his lean, threes | padded per-game count per 100, level kept | per 100 possessions | level |
| `trips_without_make_per100_uncentred` |  | free-throw trips without a make, technicals out | padded per-game count per 100, level kept | per 100 possessions | level |
| `expected_passed_trips_per100_uncentred` |  | trips without a make expected off a pass, before his lean | padded per-game count per 100, level kept | per 100 possessions | level |
| `unassisted_makes_inside_per100_uncentred` |  | unassisted makes, rim and paint | padded per-game count per 100, level kept | per 100 possessions | level |
| `unassisted_makes_two_per100_uncentred` |  | unassisted makes, other twos | padded per-game count per 100, level kept | per 100 possessions | level |
| `unassisted_makes_three_per100_uncentred` |  | unassisted makes, threes | padded per-game count per 100, level kept | per 100 possessions | level |
| `late_clock_attempts_per100_uncentred` |  | shot attempts 20 seconds or more into the possession | padded per-game count per 100, level kept | per 100 possessions | level |
| `transition_attempts_per100_uncentred` |  | shot attempts in transition, within 7 seconds of a defensive rebound or a steal | padded per-game count per 100, level kept | per 100 possessions | level |
| `putback_attempts_per100_uncentred` |  | putback attempts, within 3 seconds of his team's offensive rebound | padded per-game count per 100, level kept | per 100 possessions | level |
| `and_ones_per100_uncentred` |  | and-ones | padded per-game count per 100, level kept | per 100 possessions | level |
| `unassisted_makes_inside_per100` |  | unassisted makes, rim and paint | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `unassisted_makes_two_per100` |  | unassisted makes, other twos | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `unassisted_makes_three_per100` |  | unassisted makes, threes | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `late_clock_attempts_per100` |  | shot attempts 20 seconds or more into the possession | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `transition_attempts_per100` |  | shot attempts in transition, within 7 seconds of a defensive rebound or a steal | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `putback_attempts_per100` |  | putback attempts, within 3 seconds of his team's offensive rebound | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `and_ones_per100` |  | and-ones | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `trips_without_make_per100` |  | free-throw trips without a make, technicals out | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `goaltends_per100` |  | defensive goaltending violations called on him | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `shooting_fouls_per100` |  | shooting fouls he commits, blocking fouls on a shooter included | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `loose_ball_fouls_per100` |  | loose-ball fouls he commits | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `shots_blocked_per100` |  | his missed shots that were blocked | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `three_shot_trips_per100` |  | three-shot free-throw trips he earns: fouled shooting a three | padded per-game count per 100 minus the league level | per 100 possessions | centered |
| `goaltends_over_expected_from_blocks_per100` |  | goaltends over those expected from his blocks | goaltends_per100 minus a line on blocks_per100, fitted on the season's players on his side, weighted by possessions | per 100 possessions | centered |
| `blocks_over_expected_from_fouls_per100` |  | blocks over those expected from his personal fouls | blocks_per100 minus a line on fouls_per100, fitted on the season's players on his side, weighted by possessions | per 100 possessions | centered |
| `steals_over_expected_from_fouls_per100` |  | steals over those expected from his personal fouls | steals_per100 minus a line on fouls_per100, fitted on the season's players on his side, weighted by possessions | per 100 possessions | centered |
| `shots_blocked_over_expected_from_attempts_per100` |  | his shots blocked over those expected from his twos and threes taken | shots_blocked_per100 minus a line on twos_made_per100 + twos_missed_per100 and threes_made_per100 + threes_missed_per100, fitted on the season's players on his side, weighted by possessions | per 100 possessions | centered |
| `three_shot_trips_over_expected_from_threes_per100` |  | three-shot free-throw trips over those expected from his threes taken | three_shot_trips_per100 minus a line on threes_made_per100 + threes_missed_per100, fitted on the season's players on his side, weighted by possessions | per 100 possessions | centered |
| `assists_over_expected_from_turnovers_per100` |  | assists over those expected from his turnovers | assists_per100 minus a line on turnovers_per100, fitted on the season's players on his side, weighted by possessions | per 100 possessions | centered |
| `turnovers_over_expected_from_attempts_and_assists_per100` |  | turnovers over those expected from his true-shooting attempts and assists | turnovers_per100 minus a line on twos_made_per100 + twos_missed_per100 + threes_made_per100 + threes_missed_per100 + 0.44 free_throws_made_per100 + 0.44 free_throws_missed_per100 and assists_per100, fitted on the season's players on his side, weighted by possessions | per 100 possessions | centered |
| `self_creation_estimated_per100` |  | self-created shot attempts, passes estimated (experiment 111) | unassisted makes + misses not off a pass + free-throw trips without a make not off a pass; off a pass = his lean x the expected passes, by spot group, held to his count; lean = (assisted makes + 0.5) / (expected assisted makes + 0.5); uncentered rates | per 100 possessions | level |
| `attackers_guarded_height_in` |  | height of the attackers he guarded | mean over the fit games' per-game matchup pairs (data/matchups.py), weighted by matchup possessions, padded toward his bin by a split-half pad k | inches | level |
| `defenders_faced_height_in` |  | height of the defenders who guarded him | mean over the fit games' per-game matchup pairs (data/matchups.py), weighted by matchup possessions, padded toward his bin by a split-half pad k | inches | level |
| `attackers_guarded_self_creation_per100` |  | self-creation of the attackers he guarded | mean over the fit games' per-game matchup pairs (data/matchups.py), weighted by matchup possessions, padded toward his bin by a split-half pad k | per 100 possessions | level |
| `attackers_guarded_total_creation_per100` |  | total creation of the attackers he guarded: self-creation plus estimated potential assists | mean over the fit games' per-game matchup pairs (data/matchups.py), weighted by matchup possessions, padded toward his bin by a split-half pad k | per 100 possessions | level |
| `top_self_creator_guarded_share` |  | share of his defense on the opposing team's top self-creator of the game | mean over the fit games' per-game matchup pairs (data/matchups.py), weighted by matchup possessions, padded toward his bin by a split-half pad k | fraction | level |
| `top_total_creator_guarded_share` |  | share of his defense on the opposing team's top total creator of the game | mean over the fit games' per-game matchup pairs (data/matchups.py), weighted by matchup possessions, padded toward his bin by a split-half pad k | fraction | level |
| `top_self_creator_guarded_share_on_shared_floor` |  | share of his time on the floor with the opposing top self-creator spent guarding him | mean over the fit games' per-game matchup pairs (data/matchups.py), weighted by the shared floor with the opposing top creator, in matchup possessions, padded toward his bin by a split-half pad k | fraction | level |
| `top_total_creator_guarded_share_on_shared_floor` |  | share of his time on the floor with the opposing top total creator spent guarding him | mean over the fit games' per-game matchup pairs (data/matchups.py), weighted by the shared floor with the opposing top creator, in matchup possessions, padded toward his bin by a split-half pad k | fraction | level |
<!-- end generated -->
