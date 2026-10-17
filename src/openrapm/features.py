"""The box-score prior's inputs: what each column means, how the derived ones are built, and which ones ship.

The season panel (one row per player, season and side) carries a player's padded box-score rates and his role,
career, body and on-court numbers.  A tree splits one column at a time, so it can never form a ratio such as
true shooting from the two columns it is made of; the derived columns hand those over ready-made.

Words used below
    tree        one decision tree of the box-score prior's model (LightGBM, gradient-boosted trees; step 2)
    rate        a count per 100 possessions, padded before it is used: pulled toward the average of players
                with a similar number of possessions (one of six possession bins) by k possessions' worth, k
                estimated per rate and season from odd- versus even-game halves (box_rates.py).  With k = 450
                and a bin average of 1.4 (illustrative values), a 200-possession bench player's 2.1 steals per
                100 become (200 x 2.1 + 450 x 1.4) / (200 + 450) = 1.6
    centred     the possession-weighted league mean subtracted, so 0 is a league-average rate; "level"
                columns keep their units
    uncentred   the same padded rate with the level kept (the `_uncentred` columns); a ratio needs the level back
    combination a fixed linear sum of the centred rates
    ratio       an efficiency built from the uncentred rates, its denominator padded

The derived families:

    combination   = sum over its rates of  weight x centred rate
    ratio         = (numerator + pad x typical value) / (denominator + pad)
    vs league     = (numerator - L x denominator) / (denominator + pad)

    numerator, denominator   weighted sums of the uncentred rates (`RATIOS` lists the weights)
    pad                      how many denominator units of the typical value are added, so a bench player's
                             ratio is not a ratio of two roundings
    typical value            roughly the league's value of the ratio (0.54 for true shooting)
    L                        the league's value of the ratio in his seasons, from the league's rates (`VS_LEAGUE`)

Example: a player with 6.0 made twos, 2.0 made threes and 3.0 made free throws per 100 (uncentred) and the league
at 5.0, 1.5 and 2.5 has `points_per100` = 2 x 1.0 + 3 x 0.5 + 0.5 = 4.0 points per 100 above the league.

The names are plain words (since 2026-10-11; files written before then use short codes, which
data/legacy_names.py translates as they are read).  The order of the terms and the constants below are
load-bearing: the shipped booster reads its inputs by position, and the sums are written in the order the
published ratings were built with.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["Feature", "BOX_RATES", "UNCENTRED_RATES", "ROLE", "CLOSENESS", "SHOT_TOTALS", "SHOT_LEAGUE", "CAREER",
           "BIO", "ON_COURT", "OFF_COURT", "ON_MINUS_OFF", "PLAYOFF_SHARE", "EVENT_RATES", "CAREER_FROM_DEBUT",
           "FROM_DEBUT", "TRACKING_RATES", "HUSTLE_RATES", "MATCHUP_RATES", "DASHBOARD_RATES", "MISSING_OK", "DERIVED",
           "RATIOS", "VS_LEAGUE", "SHOTQ", "SHOTQ_PADS", "PRIOR_FEATURES", "PANEL_INPUTS", "PANEL_KEYS",
           "SEASON_PANEL_COLUMNS", "ROW_FEATURES", "SHIPPED", "REBOUND_PARTS", "STEAL_PARTS", "POINTS_PARTS",
           "THREE_POINT_PARTS", "FEATURE_SETS", "CATALOGUE", "add_features", "add_shotq", "meaning",
           "glossary_markdown"]


@dataclass(frozen=True)
class Feature:
    """One column the prior can read: its name, what it means, how it is computed, its unit, centred or level."""
    name: str
    meaning: str
    formula: str
    unit: str
    scale: str          # "centred" or "level"
    group: str


# ------------------------------------------------------------------------------ the panel's own columns
# The thirteen padded rates per 100 possessions, centred, in the order of config.yaml's box_counts (the per-game
# counts they are built from: fg3m, fg3_miss, fg2m, ...), which is the order the padding returns them in.
BOX_RATES = ["threes_made_per100", "threes_missed_per100", "twos_made_per100", "twos_missed_per100",
             "free_throws_made_per100", "free_throws_missed_per100", "off_rebounds_per100", "def_rebounds_per100",
             "assists_per100", "turnovers_per100", "steals_per100", "blocks_per100", "fouls_per100"]
(THREES_MADE, THREES_MISSED, TWOS_MADE, TWOS_MISSED, FREE_THROWS_MADE, FREE_THROWS_MISSED, OFF_REBOUNDS,
 DEF_REBOUNDS, ASSISTS, TURNOVERS, STEALS, BLOCKS, FOULS) = BOX_RATES
UNCENTRED_RATES = [f"{rate}_uncentred" for rate in BOX_RATES]
ROLE = ["team_possession_share", "games_started_share", "age"]
# the score state a player's statistics were compiled in (data/panel.py, score_state)
CLOSENESS = ["garbage_time_share", "closeness", "mean_abs_margin"]
# per shooter, summed over the season: attempts, makes, and the league's expected makes from his spots
SHOT_TOTALS = ["shot_twos_attempted", "shot_twos_made", "shot_twos_expected_made", "shot_threes_attempted",
               "shot_threes_made", "shot_threes_expected_made"]
SHOT_LEAGUE = ["league_two_point_pct", "league_three_point_pct", "league_points_per_shot"]
# counted BEFORE the season: a 25-year-old rookie and a 25-year-old in year seven are different players
CAREER = ["seasons_before", "career_possessions_before_k", "age_at_first_season"]
BIO = ["height_in", "weight_lb", "draft_pick", "seasons_with_team", "teams_this_season"]
# his team's record with him on the court, with him off it in the same games, and the difference
ON_COURT = ["on_court_scored_per100", "on_court_allowed_per100", "on_court_off_possessions",
            "on_court_def_possessions"]
OFF_COURT = ["off_court_scored_per100", "off_court_allowed_per100", "off_court_off_possessions",
             "off_court_def_possessions"]
ON_MINUS_OFF = ["on_minus_off_scored_per100", "on_minus_off_allowed_per100"]
PLAYOFF_SHARE = ["playoff_share"]
# Parts of four box counts, read from the play-by-play (data/event_counts.py says how each is read) and padded per
# 100 possessions like the thirteen rates (box_rates.py), centred; no `_uncentred` twins, since no ratio is built from
# them.  The owner's stat-splits plan (2026-10-12, experiments 52-55): each part sits beside its box count, so the
# prior can price two kinds of the same count differently.  Each part is padded with its own k, so the parts' rates
# need not add up to the box count's rate.  Example: 9 defensive rebounds per 100 might be 1 of missed free throws,
# 2.5 of missed threes, 4.5 of missed twos at the rim and 1 of missed twos from farther out.
EVENT_RATES = ["def_rebounds_of_free_throws_per100", "def_rebounds_of_threes_per100",
               "def_rebounds_of_rim_twos_per100", "def_rebounds_of_outside_twos_per100",
               "bad_pass_steals_per100", "lost_ball_steals_per100",
               "assisted_points_per100", "unassisted_points_per100",
               "corner_threes_attempted_per100", "above_break_threes_attempted_per100",
               "assisted_threes_made_per100", "unassisted_threes_made_per100",
               # the stepwise selection's wider pool (the owner, 2026-10-12): the old counter block's cuts
               "russell_blocks_per100", "rim_blocks_per100", "mid_range_blocks_per100", "three_point_blocks_per100",
               "offensive_fouls_per100", "stolen_turnovers_per100",
               "assists_to_rim_per100", "assists_to_mid_range_per100", "assists_to_corner_threes_per100",
               "assists_to_above_break_threes_per100"]
# The career columns counted from each player's debut (experiment 57): CAREER's three and seasons_with_team with his
# seasons before 1997 added from his career totals (data/careers.py), where the columns above start counting in 1997.
# For a player who debuted in 1997 or later they equal the 1997-counted ones.  Example: A.C. Green's 1997 row reads
# seasons_before 0 and seasons_before_from_debut 11.
CAREER_FROM_DEBUT = ["seasons_before_from_debut", "career_possessions_before_from_debut_k", "age_at_debut",
                     "seasons_with_team_from_debut"]
# {the column counted from 1997: the same counted from the debut}
FROM_DEBUT = dict(zip([*CAREER, "seasons_with_team"], CAREER_FROM_DEBUT))
# The dashboard inputs (the owner, 2026-10-14: "let's look for hustle, tracking, synergy play types, and matchups"):
# season totals from stats.nba.com's tracking and hustle dashboards (data/dashboards.py fetches them,
# data/dashboard_rates.py pads them), the regular season and playoffs added.  The tracking dashboards start in 2014 and
# the hustle ones in 2016 (2017 for nearly every player), so every column is missing (NaN) for the seasons before
# and for a player a file does not list.  Most are a count per 100 possessions, padded like the box rates, then read
# against the season's league average: a fraction above it (0.2 is 20% above), because their definitions moved
# between seasons (box-outs per player fell by four fifths between 2018 and 2026) and a level difference would not
# follow.  The three "over expected" columns are makes minus the makes the league's shooters would have had against the
# shots he was the closest defender on, per shot faced, padded toward 0 in shots (the SHOTQ way).  Example: 8.4
# deflections per 100 possessions in a league at 7.0 is 0.2.
TRACKING_RATES = [
    "drives_vs_league_avg", "drive_points_vs_league_avg", "potential_assists_vs_league_avg",
    "passes_made_vs_league_avg", "touches_vs_league_avg", "paint_touches_vs_league_avg", "post_touches_vs_league_avg",
    "elbow_touches_vs_league_avg", "minutes_with_ball_vs_league_avg", "miles_on_offense_vs_league_avg",
    "off_reb_chances_vs_league_avg", "off_reb_contested_vs_league_avg", "def_reb_chances_vs_league_avg",
    "def_reb_contested_vs_league_avg", "rim_shots_defended_vs_league_avg", "twos_defended_vs_league_avg",
    "threes_defended_vs_league_avg", "miles_on_defense_vs_league_avg",
    "rim_makes_over_expected_per_shot", "twos_makes_over_expected_per_shot", "threes_makes_over_expected_per_shot"]
HUSTLE_RATES = ["deflections_vs_league_avg", "contested_twos_vs_league_avg", "contested_threes_vs_league_avg",
                "charges_drawn_vs_league_avg", "loose_balls_recovered_vs_league_avg", "def_box_outs_vs_league_avg",
                "off_box_outs_vs_league_avg", "screen_assists_vs_league_avg", "screen_assist_points_vs_league_avg"]
# The matchup assignments, from the matchups dashboard (2017 on): who a player was matched with, never how the matchup
# went.  The first is the scoring of the attackers he guarded, the second how stingy the defenders were who guarded him
# (above 0 is better defenders).  Each is a rate against the league's (0.05 is 5% above), computed without the rows
# against the player himself and padded toward the league's (data/dashboard_rates.py).  The mirror of a defender's
# assignment is not an offence's choice, so the second is built to test that.
MATCHUP_RATES = ["attackers_guarded_scoring_vs_league_avg", "defenders_faced_stinginess_vs_league_avg"]
# The placebo for experiment 66 (2026-10-15): the hustle columns with each season's players' profiles shuffled among
# them, jointly, the missing ones kept missing (data/dashboard_rates.shuffled_rows).  As many columns, the same pattern
# of missing seasons, no information about the player: what it gains is what any extra columns gain.
HUSTLE_PLACEBO = [f"{name}_shuffled" for name in HUSTLE_RATES]
DASHBOARD_RATES = [*TRACKING_RATES, *HUSTLE_RATES, *MATCHUP_RATES, *HUSTLE_PLACEBO]
# An era marker (2026-10-16, overnight): 1 for a season from 2017 on, 0 before; a training row averaging several seasons
# averages it.  The hustle placebo gained only in the seasons before 2017, where its columns are missing; this asks
# whether a plain marker of that split does the same.
ERA_MARKER = ["season_from_2017"]
# the columns a prior row may lack: a training row averages them over the seasons that have them (box_prior.py) and is
# kept when they are missing, where a missing value in any other input drops the row
MISSING_OK = list(DASHBOARD_RATES)
(DEF_REBOUNDS_OF_FREE_THROWS, DEF_REBOUNDS_OF_THREES, DEF_REBOUNDS_OF_RIM_TWOS, DEF_REBOUNDS_OF_OUTSIDE_TWOS,
 BAD_PASS_STEALS, LOST_BALL_STEALS, ASSISTED_POINTS, UNASSISTED_POINTS, CORNER_THREES, ABOVE_BREAK_THREES,
 ASSISTED_THREES, UNASSISTED_THREES, RUSSELL_BLOCKS, RIM_BLOCKS, MID_RANGE_BLOCKS, THREE_POINT_BLOCKS,
 OFFENSIVE_FOULS, STOLEN_TURNOVERS, ASSISTS_TO_RIM, ASSISTS_TO_MID_RANGE, ASSISTS_TO_CORNER_THREES,
 ASSISTS_TO_ABOVE_BREAK_THREES) = EVENT_RATES

# ------------------------------------------------------------------------------ the derived columns
# Linear sums of the centred rates.  A tree splits on one column at a time (x_j < c), so it can only approximate
# a sum such as points = 2 x made twos + 3 x made threes + made free throws with many splits; the sum is handed
# over as its own column.  {name: {rate: weight}}, terms in the order they are summed.
DERIVED = {
    "points_per100": {TWOS_MADE: 2.0, THREES_MADE: 3.0, FREE_THROWS_MADE: 1.0},
    "shots_per100": {TWOS_MADE: 1.0, TWOS_MISSED: 1.0, THREES_MADE: 1.0, THREES_MISSED: 1.0},
    "free_throw_attempts_per100": {FREE_THROWS_MADE: 1.0, FREE_THROWS_MISSED: 1.0},
    "three_point_attempts_per100": {THREES_MADE: 1.0, THREES_MISSED: 1.0},
    # the possessions he finishes: a shot, a trip to the line (0.44 possessions per free throw), a turnover
    "used_possessions_per100": {TWOS_MADE: 1.0, TWOS_MISSED: 1.0, THREES_MADE: 1.0, THREES_MISSED: 1.0,
                                FREE_THROWS_MADE: 0.44, FREE_THROWS_MISSED: 0.44, TURNOVERS: 1.0},
    "rebounds_per100": {OFF_REBOUNDS: 1.0, DEF_REBOUNDS: 1.0},
    "steals_plus_blocks_per100": {STEALS: 1.0, BLOCKS: 1.0},
    "assists_minus_turnovers_per100": {ASSISTS: 1.0, TURNOVERS: -1.0},
    "threes_minus_twos_attempted_per100": {THREES_MADE: 1.0, THREES_MISSED: 1.0, TWOS_MADE: -1.0, TWOS_MISSED: -1.0},
}
# Efficiencies from the uncentred rates (a ratio of centred rates means nothing), each denominator padded so a
# 200-possession player is not a ratio of two roundings.  {name: (numerator, denominator, pad, typical value)}.
_SHOTS = {TWOS_MADE: 1, TWOS_MISSED: 1, THREES_MADE: 1, THREES_MISSED: 1}
_USED = {TWOS_MADE: 1, TWOS_MISSED: 1, THREES_MADE: 1, THREES_MISSED: 1, FREE_THROWS_MADE: 0.44,
         FREE_THROWS_MISSED: 0.44, TURNOVERS: 1}
RATIOS = {
    "effective_fg_pct": ({TWOS_MADE: 1.0, THREES_MADE: 1.5}, _SHOTS, 3.0, 0.50),
    "true_shooting_pct": ({TWOS_MADE: 2.0, THREES_MADE: 3.0, FREE_THROWS_MADE: 1.0},
                          {TWOS_MADE: 2, TWOS_MISSED: 2, THREES_MADE: 2, THREES_MISSED: 2, FREE_THROWS_MADE: 0.88,
                           FREE_THROWS_MISSED: 0.88}, 6.0, 0.54),
    "three_point_attempt_share": ({THREES_MADE: 1, THREES_MISSED: 1}, _SHOTS, 3.0, 0.25),
    "free_throws_per_shot": ({FREE_THROWS_MADE: 1, FREE_THROWS_MISSED: 1}, _SHOTS, 3.0, 0.25),
    "three_point_pct": ({THREES_MADE: 1}, {THREES_MADE: 1, THREES_MISSED: 1}, 3.0, 0.34),
    "two_point_pct": ({TWOS_MADE: 1}, {TWOS_MADE: 1, TWOS_MISSED: 1}, 3.0, 0.48),
    "free_throw_pct": ({FREE_THROWS_MADE: 1}, {FREE_THROWS_MADE: 1, FREE_THROWS_MISSED: 1}, 2.0, 0.75),
    "assists_per_used_possession": ({ASSISTS: 1}, _USED, 3.0, 0.18),
    "turnovers_per_used_possession": ({TURNOVERS: 1}, _USED, 3.0, 0.13),
    "off_rebound_share": ({OFF_REBOUNDS: 1}, {OFF_REBOUNDS: 1, DEF_REBOUNDS: 1}, 3.0, 0.22),
}
# Each ratio against the league's value of it in the player's seasons (the owner, 2026-10-12: "vs league average";
# notes/DECISIONS.md, experiment 56).  A ratio keeps its era's level, so a share of shots that are threes of 0.35 is
# high in 1999 and average in 2024, while the centred rates are already measured from their season's league level.
# The league's rates are each `<rate>_uncentred - <rate>`: the season's possession-weighted level, the same on every
# row of a season and side, and on a row averaged over seasons the possession-weighted level of his seasons.  The
# ratio is padded toward L instead of the typical value, then L is subtracted, so a league-average player and a
# player with no attempts both read 0.  {name: the ratio it is built from}.
VS_LEAGUE = {f"{name}_vs_league": name for name in RATIOS}
# Shot quality from where his attempts came from: difficulty (the league's make probability from his spots) and
# shot-making (how far he beats a league shooter from those spots), padded in attempts toward the season's own
# league level.  The pads are attempts: threes need about 450 before a shooter's own rate is trusted.
SHOTQ_PADS = {"two_point_expected_pct": 50.0, "three_point_expected_pct": 50.0, "two_point_pct_over_expected": 250.0,
              "three_point_pct_over_expected": 450.0, "expected_points_per_shot": 100.0,
              "points_per_shot_over_expected": 300.0}
SHOTQ = list(SHOTQ_PADS)

# Every input the prior could read (131): the rates, the role, every derived column, career, body, on-court, the
# play-by-play parts, the ratios against the league, the career counted from the debut, and the dashboard inputs with
# the hustle placebo.
# scripts/checks/95_miss_by_group.py and commands/within_season.py (its PLAYER_COLUMNS) report on all of them.
PRIOR_FEATURES = [*BOX_RATES, *ROLE, *DERIVED, *RATIOS, *SHOTQ, *CAREER, *BIO, *ON_COURT, *EVENT_RATES, *VS_LEAGUE,
                  *CAREER_FROM_DEBUT, *DASHBOARD_RATES, *ERA_MARKER]
# The panel columns averaged into a multi-season training row, before anything is derived from them; the play-by-play
# parts came after every column the published ratings were built from (2026-10-12), the career counted from the debut
# after them (2026-10-13), the dashboard inputs last (2026-10-14).
PANEL_INPUTS = [*BOX_RATES, *ROLE, *CLOSENESS, *UNCENTRED_RATES, *SHOT_TOTALS, *SHOT_LEAGUE, *CAREER, *BIO,
                *ON_COURT, *OFF_COURT, *ON_MINUS_OFF, *PLAYOFF_SHARE, *EVENT_RATES, *CAREER_FROM_DEBUT,
                *DASHBOARD_RATES, *ERA_MARKER]
# The season panel's columns, in order: who and when, the possessions behind the row (his offensive possessions
# on an offence row, defensive on a defence row), then the inputs.
PANEL_KEYS = ["side", "player_id", "season"]
SEASON_PANEL_COLUMNS = [*PANEL_KEYS, "possessions", *PANEL_INPUTS]
# How much evidence a training row rests on; the rated season's own row reads its possessions and 1.
ROW_FEATURES = ["row_possessions", "row_seasons"]

# The shipped inputs, each side, in the order the booster reads them: the columns Boruta (a feature-selection
# method that keeps a column only when its importance beats that of randomly shuffled copies of every column)
# accepted or left undecided, sorted alphabetically by their codes of the time (ast, creation, efg, ...), on-court
# columns removed (notes/DECISIONS.md, experiment 45).  The booster reads its inputs by position, so the positions are
# load-bearing: the plain names keep the old order, which is no longer alphabetical.
SHIPPED = {
    "O": ["assists_per100", "assists_minus_turnovers_per100", "effective_fg_pct", "career_possessions_before_k",
          "seasons_before", "three_point_pct", "shots_per100", "free_throw_attempts_per100", "off_rebounds_per100",
          "off_rebound_share", "three_point_attempt_share", "team_possession_share", "points_per100",
          "steals_per100", "seasons_with_team", "true_shooting_pct", "weight_lb"],
    "D": ["blocks_per100", "def_rebounds_per100", "career_possessions_before_k", "seasons_before", "shots_per100",
          "games_started_share", "height_in", "team_possession_share", "points_per100", "steals_per100",
          "steals_plus_blocks_per100", "true_shooting_pct", "used_possessions_per100", "weight_lb"],
}
# The stat-splits experiments' lists (the owner, 2026-10-12): the shipped lists with each experiment's parts appended,
# so every shipped input keeps its position, and a part goes on every list that reads its box count (the owner's
# same-method rule: the reason to split a count is the same on both sides).
REBOUND_PARTS = [DEF_REBOUNDS_OF_FREE_THROWS, DEF_REBOUNDS_OF_THREES, DEF_REBOUNDS_OF_RIM_TWOS,
                 DEF_REBOUNDS_OF_OUTSIDE_TWOS]
STEAL_PARTS = [BAD_PASS_STEALS, LOST_BALL_STEALS]
POINTS_PARTS = [ASSISTED_POINTS, UNASSISTED_POINTS, FREE_THROWS_MADE]
THREE_POINT_PARTS = [CORNER_THREES, ABOVE_BREAK_THREES, ASSISTED_THREES, UNASSISTED_THREES, "free_throw_pct"]
# The incumbent's lists since experiment 59: the shipped ones without the career counts.
_NO_CAREER = {side: [name for name in SHIPPED[side] if name not in FROM_DEBUT] for side in ("O", "D")}
# The dashboard arms (the owner, 2026-10-14): each side reads the inputs that describe what it does on that side of
# the ball, appended to the incumbent's list so every shipped input keeps its position.
TRACKING_OFFENCE = ["drives_vs_league_avg", "drive_points_vs_league_avg", "potential_assists_vs_league_avg",
                    "passes_made_vs_league_avg", "touches_vs_league_avg", "paint_touches_vs_league_avg",
                    "post_touches_vs_league_avg", "elbow_touches_vs_league_avg", "minutes_with_ball_vs_league_avg",
                    "miles_on_offense_vs_league_avg", "off_reb_chances_vs_league_avg"]
TRACKING_DEFENCE = ["rim_shots_defended_vs_league_avg", "twos_defended_vs_league_avg", "threes_defended_vs_league_avg",
                    "def_reb_chances_vs_league_avg", "def_reb_contested_vs_league_avg",
                    "miles_on_defense_vs_league_avg", "rim_makes_over_expected_per_shot",
                    "twos_makes_over_expected_per_shot", "threes_makes_over_expected_per_shot"]
HUSTLE_OFFENCE = ["screen_assists_vs_league_avg", "screen_assist_points_vs_league_avg", "off_box_outs_vs_league_avg",
                  "loose_balls_recovered_vs_league_avg"]
HUSTLE_DEFENCE = ["deflections_vs_league_avg", "contested_twos_vs_league_avg", "contested_threes_vs_league_avg",
                  "charges_drawn_vs_league_avg", "def_box_outs_vs_league_avg", "loose_balls_recovered_vs_league_avg"]
# The matchup arm (the owner, 2026-10-14): each side reads the assignment that describes it, appended to the incumbent's
# list.  A defender's is the scoring of the attackers he guarded; an attacker's the stinginess of his defenders.
MATCHUP_OFFENCE = ["defenders_faced_stinginess_vs_league_avg"]
MATCHUP_DEFENCE = ["attackers_guarded_scoring_vs_league_avg"]
FEATURE_SETS = {
    # the box score, the role, the career and the body, no on-court record (experiment 45's boruta_noonc)
    "box_score_and_bio": SHIPPED,
    # experiment 52: defensive rebounds by the miss they ended (only the defensive list reads defensive rebounds)
    "rebound_parts": {"O": list(SHIPPED["O"]), "D": SHIPPED["D"] + REBOUND_PARTS},
    # experiment 53: steals by the turnover they took (both lists read steals)
    "steal_parts": {"O": SHIPPED["O"] + STEAL_PARTS, "D": SHIPPED["D"] + STEAL_PARTS},
    # experiment 54: points by whether an assist set them up, and the free throws (both lists read points)
    "points_parts": {"O": SHIPPED["O"] + POINTS_PARTS, "D": SHIPPED["D"] + POINTS_PARTS},
    # experiment 55: threes by where and how they came, and free-throw % (only the offensive list reads threes)
    "three_point_parts": {"O": SHIPPED["O"] + THREE_POINT_PARTS, "D": list(SHIPPED["D"])},
    # what forward stepwise selection (scripts/checks/141_stepwise_inputs.py) added on each half of the seasons
    # (2026-10-12): 1997-2011 ("early") and 2012-2026 ("late"); each is tested on the other half
    "stepwise_early": {"O": SHIPPED["O"] + [DEF_REBOUNDS_OF_THREES], "D": SHIPPED["D"] + [ASSISTS_TO_MID_RANGE]},
    "stepwise_late": {"O": SHIPPED["O"] + [CORNER_THREES], "D": SHIPPED["D"] + [FREE_THROWS_MADE]},
    # experiment 56: every shipped ratio against the league's value of it, in the ratio's place
    "ratios_vs_league": {side: [f"{name}_vs_league" if name in RATIOS else name for name in SHIPPED[side]]
                         for side in ("O", "D")},
    # experiment 57: every shipped career count counted from the debut, in its place
    "career_from_debut": {side: [FROM_DEBUT.get(name, name) for name in SHIPPED[side]] for side in ("O", "D")},
    # experiment 59 (the owner, 2026-10-14): the career counts off both lists -- career possessions before, seasons
    # played before and seasons with his current team; every other shipped input keeps its order
    "no_career_counts": _NO_CAREER,
    # experiment 74 (2026-10-16, overnight): the hustle lists plus the five tracking columns the per-game tracking box
    # score holds (2014 on), so a fold rebuilds every one from its fit games
    "hustle_and_tracking_games": {
        "O": _NO_CAREER["O"] + HUSTLE_OFFENCE + ["passes_made_vs_league_avg", "touches_vs_league_avg",
                                                 "off_reb_chances_vs_league_avg"],
        "D": _NO_CAREER["D"] + HUSTLE_DEFENCE + ["rim_shots_defended_vs_league_avg", "def_reb_chances_vs_league_avg"]},
    # experiment 73 (2026-10-16, overnight): the era marker appended to both lists
    "no_career_era_marker": {side: [*_NO_CAREER[side], *ERA_MARKER] for side in ("O", "D")},
    # experiment 72 (2026-10-16, overnight): experiment 54's points split on today's inputs, both lists reading points
    # by whether an assist set them up, and free throws made: unassisted scoring is shot creation (the Jordan look)
    "no_career_points_parts": {side: _NO_CAREER[side] + POINTS_PARTS for side in ("O", "D")},
    # experiment 69 (2026-10-15, overnight): age added to both lists, a this-season fact; the top 5% of offensive
    # box-score parts aged 33+ came in 0.34 per 100 below the prior, the younger ones 0.00 to 0.12
    "no_career_plus_age": {side: [*_NO_CAREER[side], "age"] for side in ("O", "D")},
    # experiment 64 (2026-10-15): experiment 56 again on today's inputs, the ratios against the league in their place
    "no_career_ratios_vs_league": {side: [f"{name}_vs_league" if name in RATIOS else name for name in _NO_CAREER[side]]
                                   for side in ("O", "D")},
    # the dashboard arms (screen: scripts/checks/146_dashboard_screen.py): tracking, hustle, and both, appended
    "tracking": {"O": _NO_CAREER["O"] + TRACKING_OFFENCE, "D": _NO_CAREER["D"] + TRACKING_DEFENCE},
    "hustle": {"O": _NO_CAREER["O"] + HUSTLE_OFFENCE, "D": _NO_CAREER["D"] + HUSTLE_DEFENCE},
    # the booster tune on the decider (check 152, 2026-10-16): the hustle lists with the era marker, its switch
    "hustle_era_marker": {"O": _NO_CAREER["O"] + HUSTLE_OFFENCE + ERA_MARKER,
                          "D": _NO_CAREER["D"] + HUSTLE_DEFENCE + ERA_MARKER},
    # experiment 66's placebo: the same lists with every hustle column shuffled among the season's players
    "hustle_placebo": {"O": _NO_CAREER["O"] + [f"{name}_shuffled" for name in HUSTLE_OFFENCE],
                       "D": _NO_CAREER["D"] + [f"{name}_shuffled" for name in HUSTLE_DEFENCE]},
    "tracking_and_hustle": {"O": _NO_CAREER["O"] + TRACKING_OFFENCE + HUSTLE_OFFENCE,
                            "D": _NO_CAREER["D"] + TRACKING_DEFENCE + HUSTLE_DEFENCE},
    # the matchup arm (screen: scripts/checks/149_matchup_screen.py): one assignment per side
    "matchups": {"O": _NO_CAREER["O"] + MATCHUP_OFFENCE, "D": _NO_CAREER["D"] + MATCHUP_DEFENCE},
}


# ------------------------------------------------------------------------------ building them
def _add_combinations(frame: pd.DataFrame) -> None:
    """Each `DERIVED` column whose rates the frame carries, summed term by term in the listed order.

    Example: `assists_minus_turnovers_per100` = 1 x assists + (-1) x turnovers, so 3.0 assists and 1.0 turnovers
    above the league give 2.0.
    """
    for name, weights in DERIVED.items():
        if name in frame.columns or not all(rate in frame.columns for rate in weights):
            continue
        frame[name] = sum(weight * frame[rate].to_numpy(dtype=float) for rate, weight in weights.items())


def _add_ratios(frame: pd.DataFrame) -> None:
    """Each `RATIOS` column on the uncentred (`_uncentred`) rates:

        ratio = (numerator + pad x typical value) / (denominator + pad)

    Example: offensive-rebound share with 3.0 offensive and 9.0 defensive rebounds per 100, pad 3 and typical
    value 0.22: (3.0 + 3 x 0.22) / (3.0 + 9.0 + 3) = 0.244.
    """
    for name, (numerator, denominator, pad, typical) in RATIOS.items():
        rates = set(numerator) | set(denominator)
        if name in frame.columns or not all(f"{rate}_uncentred" in frame.columns for rate in rates):
            continue
        top = sum(weight * frame[f"{rate}_uncentred"].to_numpy(dtype=float) for rate, weight in numerator.items())
        bottom = sum(weight * frame[f"{rate}_uncentred"].to_numpy(dtype=float) for rate, weight in denominator.items())
        frame[name] = (top + pad * typical) / (bottom + pad)


def _add_ratios_vs_league(frame: pd.DataFrame) -> None:
    """Each `VS_LEAGUE` column, from the uncentred rates and the league's (`<rate>_uncentred - <rate>`):

        L          = (numerator of the league's rates) / (denominator of the league's rates)
        vs league  = (numerator + pad x L) / (denominator + pad) - L
                   = (numerator - L x denominator) / (denominator + pad)

    Example: offensive-rebound share with 3.0 offensive and 9.0 defensive rebounds per 100 in a league at 2.5 and 7.5:
    L = 2.5 / 10.0 = 0.25 and (3.0 - 0.25 x 12.0) / (12.0 + 3) = 0.0, his share being the league's.
    """
    for name, ratio in VS_LEAGUE.items():
        numerator, denominator, pad, _ = RATIOS[ratio]
        rates = set(numerator) | set(denominator)
        if name in frame.columns or not all(f"{rate}_uncentred" in frame.columns and rate in frame.columns
                                            for rate in rates):
            continue
        own = {rate: frame[f"{rate}_uncentred"].to_numpy(dtype=float) for rate in rates}
        league = {rate: own[rate] - frame[rate].to_numpy(dtype=float) for rate in rates}
        top = sum(weight * own[rate] for rate, weight in numerator.items())
        bottom = sum(weight * own[rate] for rate, weight in denominator.items())
        level = (sum(weight * league[rate] for rate, weight in numerator.items())
                 / sum(weight * league[rate] for rate, weight in denominator.items()))
        frame[name] = (top - level * bottom) / (bottom + pad)


def add_shotq(frame: pd.DataFrame) -> pd.DataFrame:
    """The `SHOTQ` columns, when the frame carries the shot totals and the league levels.

        two_point_expected_pct       = (expected two-point makes + 50 x league 2P%) / (two-point attempts + 50)
        two_point_pct_over_expected  = (two-point makes - expected) / (two-point attempts + 250)
        expected_points_per_shot     = (2 x expected 2s + 3 x expected 3s + 100 x league points per shot)
                                       / (attempts + 100)
        points_per_shot_over_expected = (2 x (makes - expected) on twos + 3 x the same on threes) / (attempts + 300)
    and the two three-point columns the same on threes (pads 50 and 450).  "Expected" makes are the league's make
    probability from each of his shot locations, summed over his attempts; "attempts" counts twos and threes
    together.

    Example: 300 three-point attempts, 111 made and 105 expected give three_point_pct_over_expected =
    (111 - 105) / (300 + 450) = 0.008.
    """
    if "two_point_expected_pct" in frame.columns or not all(c in frame.columns for c in (*SHOT_TOTALS, *SHOT_LEAGUE)):
        return frame
    column = {name: frame[name].to_numpy(dtype=float) for name in (*SHOT_TOTALS, *SHOT_LEAGUE)}
    attempts2, makes2, expected2 = (column["shot_twos_attempted"], column["shot_twos_made"],
                                    column["shot_twos_expected_made"])
    attempts3, makes3, expected3 = (column["shot_threes_attempted"], column["shot_threes_made"],
                                    column["shot_threes_expected_made"])
    league2, league3 = column["league_two_point_pct"], column["league_three_point_pct"]
    league_points = column["league_points_per_shot"]
    pad, attempts = SHOTQ_PADS, attempts2 + attempts3
    frame["two_point_expected_pct"] = ((expected2 + pad["two_point_expected_pct"] * league2)
                                       / (attempts2 + pad["two_point_expected_pct"]))
    frame["three_point_expected_pct"] = ((expected3 + pad["three_point_expected_pct"] * league3)
                                         / (attempts3 + pad["three_point_expected_pct"]))
    # padded toward 0, not toward the league: a league-average shooter beats his expected makes by nothing
    frame["two_point_pct_over_expected"] = (makes2 - expected2) / (attempts2 + pad["two_point_pct_over_expected"])
    frame["three_point_pct_over_expected"] = (makes3 - expected3) / (attempts3 + pad["three_point_pct_over_expected"])
    frame["expected_points_per_shot"] = ((2.0 * expected2 + 3.0 * expected3
                                          + pad["expected_points_per_shot"] * league_points)
                                         / (attempts + pad["expected_points_per_shot"]))
    frame["points_per_shot_over_expected"] = ((2.0 * (makes2 - expected2) + 3.0 * (makes3 - expected3))
                                              / (attempts + pad["points_per_shot_over_expected"]))
    return frame


def add_features(frame: pd.DataFrame, names) -> pd.DataFrame:
    """Add every combination, ratio and shot-quality column the frame can make (in place), then check `names`.

    A family whose inputs are absent is skipped, so the check is what stops a prior from training on a
    quietly shorter list: every name asked for must exist afterwards, or this raises.
    """
    _add_combinations(frame)
    _add_ratios(frame)
    _add_ratios_vs_league(frame)
    add_shotq(frame)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"cannot build {missing}: the rows lack their inputs. "
                       f"Needed: {[column for column in PANEL_INPUTS if column not in frame.columns]}")
    return frame


# ------------------------------------------------------------------------------ the catalogue
def _linear(weights: dict) -> str:
    """{"twos_made_per100": 2.0, "free_throws_made_per100": 1.0} -> "2 twos_made_per100 + free_throws_made_per100"."""
    out = ""
    for position, (name, weight) in enumerate(weights.items()):
        sign = ("-" if weight < 0 else "") if position == 0 else (" - " if weight < 0 else " + ")
        out += f"{sign}{'' if abs(weight) == 1 else f'{abs(weight):g} '}{name}"
    return out


def _family(names, meanings, formula, unit, scale, group) -> list:
    """One `Feature` per name, all sharing a formula, unit, scale and group."""
    return [Feature(name, text, formula, unit, scale, group) for name, text in zip(names, meanings)]


_RATE_MEANINGS = ["made threes", "missed threes", "made twos", "missed twos", "made free throws",
                  "missed free throws", "offensive rebounds", "defensive rebounds", "assists", "turnovers", "steals",
                  "blocks", "personal fouls"]
_EVENT_MEANINGS = ["defensive rebounds of missed free throws", "defensive rebounds of missed threes",
                   "defensive rebounds of missed twos within 6 ft of the rim, or logged without a location",
                   "defensive rebounds of missed twos from beyond 6 ft", "steals of a pass (a bad-pass turnover)",
                   "steals of the ball from a dribbler (a lost-ball turnover)", "points from assisted field goals",
                   "points from unassisted field goals", "corner threes attempted", "above-the-break threes attempted",
                   "assisted made threes", "unassisted made threes",
                   "blocks his team recovered (Russell blocks)", "blocks of twos within 6 ft of the rim, or logged "
                   "without a location", "blocks of twos from beyond 6 ft", "blocks of threes",
                   "offensive fouls committed", "turnovers stolen by the defence (live-ball turnovers)",
                   "assists on makes at the rim (within 6 ft, or logged without a location)",
                   "assists on made twos from beyond 6 ft", "assists on made corner threes",
                   "assists on made above-the-break threes"]
# the first 18 of TRACKING_RATES: counts per 100 possessions; the last three are makes over expected (below)
_TRACKING_MEANINGS = ["drives to the basket", "points scored on drives",
                      "potential assists (passes that led to a shot)", "passes made", "touches of the ball",
                      "touches in the paint", "post touches", "elbow touches",
                      "minutes with the ball in his hands", "miles run on offence",
                      "offensive rebound chances (a rebound within his reach)", "contested offensive rebounds",
                      "defensive rebound chances (a rebound within his reach)", "contested defensive rebounds",
                      "shots at the rim he was the closest defender on",
                      "two-point shots he was the closest defender on",
                      "three-point shots he was the closest defender on", "miles run on defence"]
_OVER_EXPECTED_ROWS = [
    ("rim_makes_over_expected_per_shot", "shots made within 6 ft against him, above what the league's shooters make "
     "against those defenders, per shot faced"),
    ("twos_makes_over_expected_per_shot", "two-point shots made against him above expected, per shot faced"),
    ("threes_makes_over_expected_per_shot", "three-point shots made against him above expected, per shot faced")]
_HUSTLE_MEANINGS = ["deflections", "contested two-point shots", "contested three-point shots", "charges drawn",
                    "loose balls recovered", "defensive box-outs", "offensive box-outs", "screen assists",
                    "points from his screen assists"]
_MATCHUP_ROWS = [
    ("attackers_guarded_scoring_vs_league_avg", "the scoring of the attackers he guarded (points per matchup "
     "possession without the rows against him), against the league's"),
    ("defenders_faced_stinginess_vs_league_avg", "how stingy the defenders were who guarded him (points allowed per "
     "matchup possession without the rows against him, sign-flipped), against the league's")]
_DERIVED_MEANINGS = {"points_per100": "points", "shots_per100": "shot attempts",
                     "free_throw_attempts_per100": "free-throw attempts",
                     "three_point_attempts_per100": "three-point attempts",
                     "used_possessions_per100": "possessions he finishes", "rebounds_per100": "rebounds",
                     "steals_plus_blocks_per100": "steals plus blocks",
                     "assists_minus_turnovers_per100": "assists minus turnovers",
                     "threes_minus_twos_attempted_per100": "threes minus twos taken"}
_RATIO_MEANINGS = {"effective_fg_pct": "effective field-goal %", "true_shooting_pct": "true shooting %",
                   "three_point_attempt_share": "share of shots that are threes",
                   "free_throws_per_shot": "free throws per shot", "three_point_pct": "three-point %",
                   "two_point_pct": "two-point %", "free_throw_pct": "free-throw %",
                   "assists_per_used_possession": "assists per possession used",
                   "turnovers_per_used_possession": "turnovers per possession used",
                   "off_rebound_share": "offensive-rebound share of his rebounds"}
_SHOTQ_ROWS = [
    # name, meaning, formula ({pad} is the name's SHOTQ_PADS entry), unit
    ("two_point_expected_pct", "difficulty of his twos",
     "(shot_twos_expected_made + {pad} league_two_point_pct) / (shot_twos_attempted + {pad})", "fraction"),
    ("three_point_expected_pct", "difficulty of his threes",
     "(shot_threes_expected_made + {pad} league_three_point_pct) / (shot_threes_attempted + {pad})", "fraction"),
    ("two_point_pct_over_expected", "two-point shot-making",
     "(shot_twos_made - shot_twos_expected_made) / (shot_twos_attempted + {pad})", "makes per attempt"),
    ("three_point_pct_over_expected", "three-point shot-making",
     "(shot_threes_made - shot_threes_expected_made) / (shot_threes_attempted + {pad})", "makes per attempt"),
    ("expected_points_per_shot", "expected points per shot",
     "(2 shot_twos_expected_made + 3 shot_threes_expected_made + {pad} league_points_per_shot) "
     "/ (shot_twos_attempted + shot_threes_attempted + {pad})", "points"),
    ("points_per_shot_over_expected", "points per shot above expected",
     "(2 (shot_twos_made - shot_twos_expected_made) + 3 (shot_threes_made - shot_threes_expected_made)) "
     "/ (shot_twos_attempted + shot_threes_attempted + {pad})", "points"),
]
_PANEL_ROWS = [
    # name, meaning, unit, scale, group
    ("possessions", "possessions behind the row: offensive on an offence row, defensive on a defence row",
     "possessions", "level", "role"),
    ("team_possession_share", "share of team possessions played", "share", "level", "role"),
    ("games_started_share", "share of games started", "share", "level", "role"),
    ("age", "age", "years", "level", "role"),
    ("garbage_time_share", "share of possessions in garbage time", "share", "level", "score state"),
    ("closeness", "how close his games were: mean of 1 / max(|margin|, 1)", "1 / points", "level", "score state"),
    ("mean_abs_margin", "average score gap while he played", "points", "level", "score state"),
    ("shot_twos_attempted", "two-point attempts", "attempts", "level", "shot totals"),
    ("shot_twos_made", "two-point makes", "makes", "level", "shot totals"),
    ("shot_twos_expected_made", "expected two-point makes, league rates from his spots", "makes", "level",
     "shot totals"),
    ("shot_threes_attempted", "three-point attempts", "attempts", "level", "shot totals"),
    ("shot_threes_made", "three-point makes", "makes", "level", "shot totals"),
    ("shot_threes_expected_made", "expected three-point makes, league rates from his spots", "makes", "level",
     "shot totals"),
    ("league_two_point_pct", "league two-point %", "fraction", "level", "shot totals"),
    ("league_three_point_pct", "league three-point %", "fraction", "level", "shot totals"),
    ("league_points_per_shot", "league points per shot", "points", "level", "shot totals"),
    ("seasons_before", "seasons played before this one", "seasons", "level", "career"),
    ("career_possessions_before_k", "career possessions before this season", "thousands", "level", "career"),
    ("age_at_first_season", "age entering the league", "years", "level", "career"),
    ("height_in", "height", "inches", "level", "body"),
    ("weight_lb", "weight", "pounds", "level", "body"),
    ("draft_pick", "draft position", "pick", "level", "body"),
    ("seasons_with_team", "seasons with his current team", "seasons", "level", "career"),
    ("teams_this_season", "teams he played for this season", "teams", "level", "career"),
    ("seasons_before_from_debut", "seasons played before this one, counted from his debut (before 1997 too)",
     "seasons", "level", "career"),
    ("career_possessions_before_from_debut_k", "career possessions before this season, counted from his debut "
     "(before 1997 from his minutes)", "thousands", "level", "career"),
    ("age_at_debut", "age in his first NBA season (before 1997 too)", "years", "level", "career"),
    ("seasons_with_team_from_debut", "seasons with his current team, counted from his debut (before 1997 too)",
     "seasons", "level", "career"),
    ("on_court_scored_per100", "team points scored per 100, him on court", "per 100 possessions", "centred",
     "on court"),
    ("on_court_allowed_per100", "team points allowed per 100, him on court", "per 100 possessions", "centred",
     "on court"),
    ("on_court_off_possessions", "offensive possessions on court", "possessions", "level", "on court"),
    ("on_court_def_possessions", "defensive possessions on court", "possessions", "level", "on court"),
    ("off_court_scored_per100", "team points scored per 100, him off court", "per 100 possessions", "centred",
     "off court"),
    ("off_court_allowed_per100", "team points allowed per 100, him off court", "per 100 possessions", "centred",
     "off court"),
    ("off_court_off_possessions", "offensive possessions off court", "possessions", "level", "off court"),
    ("off_court_def_possessions", "defensive possessions off court", "possessions", "level", "off court"),
    ("on_minus_off_scored_per100", "on court minus off court, points scored", "per 100 possessions", "centred",
     "on/off"),
    ("on_minus_off_allowed_per100", "on court minus off court, points allowed", "per 100 possessions", "centred",
     "on/off"),
    ("playoff_share", "share of possessions in the playoffs", "share", "level", "playoffs"),
    ("row_possessions", "possessions behind the row", "possessions", "level", "row evidence"),
    ("row_seasons", "seasons the row covers", "seasons", "level", "row evidence"),
]


def _ratio_formula(numerator: dict, denominator: dict, pad: float, typical: float) -> str:
    """A `RATIOS` entry written out: "(threes_made_per100 + 3 x 0.34) / (threes_made_per100 + threes_missed_per100
    + 3), uncentred rates"."""
    return f"({_linear(numerator)} + {pad:g} x {typical:g}) / ({_linear(denominator)} + {pad:g}), uncentred rates"


def _vs_league_formula(numerator: dict, denominator: dict, pad: float, typical: float) -> str:
    """A `VS_LEAGUE` entry written out: "(threes_made_per100 - L x (threes_made_per100 + threes_missed_per100)) /
    (threes_made_per100 + threes_missed_per100 + 3), uncentred rates; L the same ratio of the league's rates"."""
    return (f"({_linear(numerator)} - L x ({_linear(denominator)})) / ({_linear(denominator)} + {pad:g}), "
            f"uncentred rates; L the same ratio of the league's rates (each uncentred minus centred)")


def _catalogue() -> tuple:
    """Every column the prior can read, family by family, in the order the panel and the derivations list them."""
    rows = _family(BOX_RATES, _RATE_MEANINGS, "padded rate minus the league level", "per 100 possessions",
                   "centred", "box rate")
    rows += _family(UNCENTRED_RATES, [f"{text}, league level kept" for text in _RATE_MEANINGS], "padded rate",
                    "per 100 possessions", "level", "box rate, uncentred")
    rows += _family(EVENT_RATES, _EVENT_MEANINGS, "padded play-by-play count per 100 minus the league level",
                    "per 100 possessions", "centred", "play-by-play rate")
    rows += _family(TRACKING_RATES[:18], _TRACKING_MEANINGS,
                    "tracking-dashboard season total per 100 possessions, padded, (rate - league rate) / league rate",
                    "fraction above the league average", "centred", "tracking rate")
    rows += [Feature(name, text, "(makes - expected makes) / (shots faced + pad); the pad is the shots in "
                     "data/dashboard_rates.py", "makes per shot", "centred", "tracking rate")
             for name, text in _OVER_EXPECTED_ROWS]
    rows += _family(HUSTLE_RATES, _HUSTLE_MEANINGS,
                    "hustle box scores summed over the games, per 100 possessions, padded, (rate - league rate) / "
                    "league rate", "fraction above the league average", "centred", "hustle rate")
    rows += _family(HUSTLE_PLACEBO, [f"{text}, shuffled among the season's players (a placebo)"
                                     for text in _HUSTLE_MEANINGS],
                    "the hustle column with each season's player profiles shuffled jointly, missing kept missing",
                    "fraction above the league average", "centred", "hustle placebo")
    rows += [Feature(name, text, "(attackers' points per matchup possession, each without his rows against the player, "
                     "padded 50 possessions toward the league's, mean weighted by possessions - league rate) / league "
                     "rate; the second sign-flipped", "fraction above the league average", "centred", "matchup")
             for name, text in _MATCHUP_ROWS]
    rows += [Feature(name, _DERIVED_MEANINGS[name], _linear(weights), "per 100 possessions", "centred",
                     "combination") for name, weights in DERIVED.items()]
    rows += [Feature(name, _RATIO_MEANINGS[name], _ratio_formula(*spec), "fraction", "level", "ratio")
             for name, spec in RATIOS.items()]
    rows += [Feature(name, f"{_RATIO_MEANINGS[ratio]} above the league's", _vs_league_formula(*RATIOS[ratio]),
                     "fraction", "centred", "ratio vs league") for name, ratio in VS_LEAGUE.items()]
    rows += [Feature(name, text, formula.format(pad=f"{SHOTQ_PADS[name]:g}"), unit, "level", "shot quality")
             for name, text, formula, unit in _SHOTQ_ROWS]
    rows += [Feature(name, text, "panel column", unit, scale, group)
             for name, text, unit, scale, group in _PANEL_ROWS]
    rows += [Feature("season_from_2017", "1 for a season from 2017 on, 0 before (an era marker)", "season >= 2017",
                     "flag", "level", "era")]
    return tuple(rows)


CATALOGUE = _catalogue()


def meaning(name: str) -> str:
    """What a column means, in the catalogue's words (reports print this); an unknown name comes back as it is,
    so a missing entry shows up as a name rather than as a wrong meaning."""
    return next((feature.meaning for feature in CATALOGUE if feature.name == name), name)


def glossary_markdown() -> str:
    """The catalogue as a markdown table, one row per name, for GLOSSARY.md."""
    def cell(text: str) -> str:
        return text.replace("|", "\\|")          # a literal |margin| would end the cell

    lines = ["| name | meaning | formula | unit | scale |", "|---|---|---|---|---|"]
    lines += [f"| `{feature.name}` | {cell(feature.meaning)} | {cell(feature.formula)} | {feature.unit} | "
              f"{feature.scale} |" for feature in CATALOGUE]
    return "\n".join(lines) + "\n"
