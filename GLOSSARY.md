# What the prior's features mean

The prior is the model that turns a player's stats into his starting estimate, before his own games adjust it.
These are its inputs, by the names the code uses.  Rates are per 100 possessions he played, pulled toward that
season's average for players with similar playing time (one of six possession bins, `config.yaml` `pad_target:
poss_conditional`) when he played little.  Reports to the owner use the plain names, never the codes.

## Box-score rates (per 100 possessions)

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

## Combinations of those (per 100)

| code | plain name | how it is made |
|---|---|---|
| pts | points | 2 x twos + 3 x threes + free throws |
| fga | shot attempts | twos and threes taken |
| fta | free-throw attempts | |
| fg3a | three-point attempts | |
| usage | possessions he finishes | shots + 0.44 x free-throw attempts + turnovers |
| reb | rebounds | offensive + defensive |
| stocks | steals plus blocks | |
| creation | assists minus turnovers | |
| shotmix | threes minus twos taken | |
| bigness | "plays like a centre" score | offensive rebounds + blocks + 0.3 x defensive rebounds - 0.5 x assists - 0.4 x made threes; high = rebounds and blocks with little passing or shooting, low = a perimeter guard.  A leftover from an old check (`22_vs_consensus`) |

## Percentages and shares

| code | plain name |
|---|---|
| efg | effective field-goal % (a made three counts as 1.5 makes) |
| ts | true shooting % (points per shooting possession, free throws included) |
| fg3p / fg2p / ftp | three-point % / two-point % / free-throw % |
| p3r | share of his shots that are threes |
| ftr | free-throw attempts per shot attempt |
| astr | assists per possession he uses |
| tovr | turnovers per possession he uses |
| orbsh | offensive rebounds as a share of all his rebounds |

## Shot quality (from where he shoots)

| code | plain name |
|---|---|
| q2 / q3 | difficulty of his twos / threes: the league's make rate from the spots he shoots from |
| m2 / m3 | shot-making on twos / threes: his makes above the league's from those same spots |
| xps | expected points per shot, from his spots |
| mpts | points per shot above that expectation |

## Role, body and career

| code | plain name |
|---|---|
| poss_pct | share of his team's possessions he was on the floor for |
| gs_pct | share of games he started |
| age | age |
| height / weight | height / weight |
| draft_pick | draft position (61 = undrafted) |
| exp_yrs | seasons played before this stretch |
| exp_poss | career possessions before this stretch (thousands) |
| entry_age | age he entered the league |
| tenure | seasons with his current main team |
| n_teams | teams he played for in this stretch |

## His team's results with him on the court (plus-minus)

| code | plain name |
|---|---|
| onc_o | his team's points scored per 100 while he is on the court |
| onc_d | his team's points allowed per 100 while he is on the court |
| onc_poss_o / onc_poss_d | the possessions behind each |

## How much evidence a training row rests on

| code | plain name |
|---|---|
| chunk_poss | possessions behind the row's stats |
| chunk_seasons | seasons the row covers (1, 2, 3, or his whole career) |

## Never offered to the shipped model

| code | plain name |
|---|---|
| offc_o / offc_d | his team's points scored / allowed per 100 while he is OFF the court |
| net_o / net_d | on-court minus off-court ("on/off") |
| gt_share | share of his possessions in garbage time |
| closeness | how close the games were while he played |
| abs_margin | average score gap while he played |
| po_share | share of his possessions in the playoffs |

## Every column the prior can read

Written from src/openrapm/features.py by `openrapm glossary`; edit the catalogue there, not the table here.

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
| `two_point_expected_pct` | difficulty of his twos | (shot_twos_expected_made + 50 league_two_point_pct) / (shot_twos_attempted + 50) | fraction | level |
| `three_point_expected_pct` | difficulty of his threes | (shot_threes_expected_made + 50 league_three_point_pct) / (shot_threes_attempted + 50) | fraction | level |
| `two_point_pct_over_expected` | two-point shot-making | (shot_twos_made - shot_twos_expected_made) / (shot_twos_attempted + 250) | makes per attempt | level |
| `three_point_pct_over_expected` | three-point shot-making | (shot_threes_made - shot_threes_expected_made) / (shot_threes_attempted + 450) | makes per attempt | level |
| `expected_points_per_shot` | expected points per shot | (2 shot_twos_expected_made + 3 shot_threes_expected_made + 100 league_points_per_shot) / (attempts + 100) | points | level |
| `points_per_shot_over_expected` | points per shot above expected | (2 (shot_twos_made - shot_twos_expected_made) + 3 (shot_threes_made - shot_threes_expected_made)) / (attempts + 300) | points | level |
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
<!-- end generated -->
