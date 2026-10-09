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
