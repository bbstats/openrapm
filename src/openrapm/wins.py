"""Wins over starter average (WOSA): how many regular-season wins a player added to his team over an average starter.

The owner's MVP score (2026-11-02): "how many wins you improve your team vs the average player in the top 150".  It
reads the regular season's stints and each player's rating from that regular season alone (`commands/wins.py`, the
lock), and changes no published rating.  notes/DECISIONS.md, "WOSA", holds the checks behind every choice here
(scripts/checks/157_wins_model.py, 158_high_leverage.py).

Words used below
    starter average  the bar: the mean rating, on each side, of the 150 players with the most regular-season games
                     started, minutes played breaking ties.  Every season's ratings are centred at possession-weighted
                     zero per side, so the bar sits above zero: starters are better than the average possession
    share left       the share of 48 minutes left in the game; an overtime reads its own clock, so 120 seconds left in
                     any overtime is 120 / 2,880
    lead kept        the share of a lead that reaches the final margin, a line in the share left: about 0.94 with no
                     time left and 0.75 at the half, in every era.  A team ahead coasts and a team behind presses, so
                     a margin is not a random walk (check 157: with every point kept, a team up 7 to 12 in the first
                     half is given 81% and wins 73%)
    win model        the chance the home team wins from a margin with a share of the game left,
                         P = Phi((lead kept(share left) x margin + home edge x share left) / spread(share left))
                     where spread is the measured standard deviation of what is left of a final margin after that, by
                     time left, at average team strength
    leverage         what one point moves the win model by at a stint's game state: (P(margin + 1) - P(margin - 1)) / 2,
                     the same for both teams.  Read at the stint's starting margin, so its own result never weights it,
                     and at the middle of its time
    weight           a stint's leverage over the possession-weighted mean of its own game.  Every game carries the same
                     total weight, and leverage only divides it up inside that game (the owner, 2026-11-02: "if you
                     cause the blowout you shouldn't be penalized").  A blowout win moves a team from about 50% to 100%
                     as a buzzer-beater does, so a lopsided game counts as much as a close one: its weight goes to the
                     possessions that built the lead, and its decided end stays near 0.  Read against the season's mean
                     instead (`relative_leverage`, shipped first), San Antonio's 2026 games read 0.89 with Wembanyama
                     and 1.06 without: a player was marked down for the blowouts he caused
    points per win   how many rating points, weighted by the weight, a team needs for one more win: one number, 30,
                     read from the standings over 30 seasons (check 157; 28.7, 28.9 and 30.4 by era, each season's
                     correlation 0.92 to 0.98).  The win model's own 1 over its mean leverage reads 41 to 45 and is not
                     the scale: it values a point after leads shrink, where the published ratings already sit below
                     teams' actual margins, so it would discount twice and leave team totals well short of their records
    pace effect      a player's effect on the possessions per 48 minutes each team plays, from a ridge regression of
                     each stint's pace on the ten players on the court (pace RAPM)

The equation, for a player over the regular-season stints he was on the court for, his ratings from that regular season:

    value     = weight / points per win / 100
    wosa_off  = sum value x (rating_off - bar_off) x his team's possessions
    wosa_def  = sum value x (rating_def - bar_def) x the opponent's possessions
    wosa_pace = sum value x (margin - (rating_total - bar_total)) x extra possessions
    extra possessions = (pace effect - bar pace) x seconds / 2,880
    wosa      = wosa_off + wosa_def + wosa_pace

margin is the stint's expected margin per 100 for his team, from the ten players' published ratings and the season's
home edge.  This is exactly his team's expected margin on the stint with him minus with an average starter in his place
(`counterfactual_wins` computes it that way).  The extra possessions are valued at the margin WITHOUT his surplus
because his rating on them is already in the first two lines; at the full margin a star would be paid for it twice.

The same two lines as three numbers a page can show, per side (the Detail view): above_off = rating_off - bar_off, his
possessions at that end, and leverage_off = his mean weight over them, the season's mean weight being 1, so

    wosa_off  = above_off x his team's possessions x leverage_off / 100 / points per win

and the same at the other end.  `leverage` is his mean weight over the possessions at both ends.

Example: a player at +9.0 per 100 against a bar of +0.8, over 5,300 possessions at each end, at an average weight: 8.2 x
5,300 / 100 / 30 = 14.5 wins.  Three extra possessions per 48 over 2,500 minutes are 156 extra possessions; on a lineup
at +8 per 100 with his surplus of +8.2 taken out they are worth -0.2 x 156 / 100 / 30 = -0.01 wins, while on a lineup at
+8 that owes him nothing they would be worth 8 x 156 / 100 / 30 = 0.42 wins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import isotonic_regression
from scipy.stats import norm

from .rapm import normal_equations, solve_ridge
from .scoring import weighted_least_squares
from .swap_test import AWAY, HOME, PERIOD_END_SECONDS

__all__ = ["STINT_COLUMNS", "WINS_COLUMNS", "SPREAD_BIN_EDGES", "share_left", "game_results", "strength_gap",
           "WinModel", "fit_win_model", "spread_at", "lead_kept", "win_probability", "leverage", "relative_leverage",
           "leverage_inside_games", "mean_leverage", "net_points_per_win", "rating_points_per_win", "PACE_CONTEXT",
           "PaceRows", "PaceFit",
           "pace_rows", "solve_pace", "pace_error", "pace_effects", "player_index", "starts_and_minutes",
           "expected_margins", "Bar", "starter_bar", "season_wins", "counterfactual_wins", "wins_columns"]

REGULATION_SECONDS = 2880.0
SHARE_LEFT_MINIMUM = 1.0 / REGULATION_SECONDS      # one second: a stint at the buzzer must not divide by zero
SPREAD_MINIMUM = 0.5                               # points: the rest of a game is never known closer than half a point
MARGIN_CLIP = 25.0                                 # config margin_clip, as the design and the swap test clip it

# what the wins read from a season's stints, plus home_team_id and away_team_id from the loader
STINT_COLUMNS = ["game_id", "phase", "period", "start_clock", "margin_h", "is_gt", "neutral", "half",
                 "poss_h", "poss_a", "pts_h", "pts_a", *HOME, *AWAY]
WINS_COLUMNS = ["wosa_off", "wosa_def", "wosa_pace", "wosa", "pace_effect", "leverage", "starts_rs", "minutes_rs",
                "above_off", "above_def", "leverage_off", "leverage_def"]

# The bins of seconds left the spread is measured in: 10 seconds in the last minute, where a possession decides it; 30
# to two minutes left; 1 minute to 12 left; 4 minutes before that, since a one-minute bin holding only each game's
# opening seconds is 25 stints and read 17 points beside neighbours at 14 (2026).
SPREAD_BIN_EDGES = np.concatenate([np.arange(0.0, 60.0, 10.0), np.arange(60.0, 120.0, 30.0),
                                   np.arange(120.0, 720.0, 60.0), np.arange(720.0, 2881.0, 240.0)])


# ------------------------------------------------------------------------------------------- the win model
def share_left(period, start_clock, seconds) -> tuple[np.ndarray, np.ndarray]:
    """(at the stint's start, at its middle): the share of 48 minutes left, never under one second's.

    Example: a 60-second stint from 300 seconds left in the fourth reads 300 / 2,880 and 270 / 2,880; from 300 seconds
    left in the second, (720 x 2 + 300) / 2,880 at its start.  In an overtime only that overtime's clock counts."""
    periods_after = np.clip(4 - np.asarray(period, dtype=float), 0.0, None)
    left = periods_after * 720.0 + np.asarray(start_clock, dtype=float)
    middle = left - np.asarray(seconds, dtype=float) / 2.0
    return (np.maximum(left / REGULATION_SECONDS, SHARE_LEFT_MINIMUM),
            np.maximum(middle / REGULATION_SECONDS, SHARE_LEFT_MINIMUM))


def game_results(stints: pd.DataFrame) -> pd.DataFrame:
    """One row per game: game_id, home_team_id, away_team_id, neutral and margin, the home team's final margin as its
    stints add up.  A margin of 0 is a game the parser lost points in: the league has no ties."""
    net = stints.pts_h.to_numpy(dtype=float) - stints.pts_a.to_numpy(dtype=float)
    return (stints.assign(_net=net).groupby("game_id", sort=False)
            .agg(home_team_id=("home_team_id", "first"), away_team_id=("away_team_id", "first"),
                 neutral=("neutral", "first"), margin=("_net", "sum")).reset_index())


def strength_gap(results: pd.DataFrame) -> np.ndarray:
    """Per game of `game_results`: the home team's net points per game minus the away team's, each over its OTHER games
    of the season.  With the game itself in, its own result sits in its regressor, the rest of the game looks more
    predictable than it is and every leverage reads too high.  A team's only game counts as no information: 0.

    Example: a team 410 points up over 41 games reads +10.0 per game; in a game it won by 30, (410 - 30) / 40 = +9.5."""
    margin = results.margin.to_numpy(dtype=float)
    frame = pd.DataFrame({"team": np.concatenate([results.home_team_id, results.away_team_id]),
                          "net": np.concatenate([margin, -margin])})
    by_team = frame.groupby("team").net
    others = (by_team.transform("size") - 1).to_numpy(dtype=float)
    rest = (by_team.transform("sum") - frame.net).to_numpy(dtype=float)
    strength = np.divide(rest, others, out=np.zeros(len(frame)), where=others > 0)
    return strength[:len(results)] - strength[len(results):]


@dataclass(frozen=True)
class WinModel:
    """One season's win model: the level it drifts by and the spread of the rest of a game by time left."""
    season: int
    home_edge: float               # points a full game adds for the home team, between teams of equal strength
    strength_slope: float          # points a full game adds per point of `strength_gap`, about 0.9: a net is noisy
    lead_kept_at_end: float        # the share of a lead that lasts with no time left
    lead_kept_per_game: float      # how that share moves per whole game left: negative, a long way to go loses more
    seconds_left: np.ndarray       # the middle of each spread bin, ascending
    spread: np.ndarray             # the spread there, never smaller with more time left
    n_games: int                   # the games it read
    n_ties: int                    # the games left out with a final margin of 0


def _per_stint(results: pd.DataFrame, stints: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """Per stint, the value its game has in `values` (one per row of `results`)."""
    return pd.Series(values, index=results.game_id.to_numpy()).reindex(stints.game_id.to_numpy()).to_numpy(dtype=float)


def _monotone_spread(variance: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """The variances, by more time left, as the closest non-decreasing sequence: more of a game left cannot add less to
    its margin, so a bin that dips is noise, and a running maximum would carry one noisy high to every bin above it."""
    return np.sqrt(isotonic_regression(variance, weights=weight, increasing=True).x)


def fit_win_model(stints: pd.DataFrame, seconds: np.ndarray, season: int) -> WinModel:
    """`WinModel` from one season's regular-season stints in time order within each game.

    The home edge and the strength slope come from the games' final margins by least squares, and the lead kept from
    the final margin less the drift still expected on the margin and its product with the share left.  The line stands
    in for a slope per bin because a margin at the tip-off is near zero and its slope anything at all (-0.41 in 2011).
    The spread is the possession-weighted root mean square of what is left, by bin of seconds left at the stint's
    middle.  One sigma x sqrt(share left) was tried first and does not fit: late possessions are noisier than a straight
    line in time allows (notes/DECISIONS.md, "WOSA")."""
    results = game_results(stints)
    played = results.margin.to_numpy() != 0.0
    gaps = strength_gap(results)
    at_home = ~results.neutral.fillna(False).to_numpy(dtype=bool)
    columns = np.column_stack([at_home.astype(float), gaps])[played]
    margins = results.margin.to_numpy(dtype=float)[played]
    home_edge, strength_slope = weighted_least_squares(columns, margins, np.ones(int(played.sum())))

    level = home_edge * at_home + strength_slope * gaps
    left = share_left(stints.period, stints.start_clock, seconds)[1]
    final = _per_stint(results, stints, results.margin.to_numpy(dtype=float))
    margin = stints.margin_h.to_numpy(dtype=float)
    beyond_drift = final - _per_stint(results, stints, level) * left
    keep = final != 0.0
    weight = (stints.poss_h + stints.poss_a).to_numpy(dtype=float)
    at_end, per_game = weighted_least_squares(np.column_stack([margin, margin * left])[keep], beyond_drift[keep],
                                              weight[keep])
    rest = beyond_drift - (at_end + per_game * left) * margin
    n_bins = len(SPREAD_BIN_EDGES) - 1
    bins = np.clip(np.searchsorted(SPREAD_BIN_EDGES, left * REGULATION_SECONDS, side="right") - 1, 0, n_bins - 1)
    total = np.bincount(bins[keep], weight[keep], minlength=n_bins)
    squares = np.bincount(bins[keep], (weight * rest ** 2)[keep], minlength=n_bins)
    filled = total > 0.0
    middles = (SPREAD_BIN_EDGES[:-1] + SPREAD_BIN_EDGES[1:]) / 2.0
    return WinModel(int(season), float(home_edge), float(strength_slope), float(at_end), float(per_game),
                    middles[filled],
                    _monotone_spread(squares[filled] / total[filled], total[filled]),
                    int(played.sum()), int((~played).sum()))


def spread_at(left, model: WinModel) -> np.ndarray:
    """The spread of the rest of a game with a share `left` of it left: read between the bins' middles, flat past."""
    seconds_left = np.asarray(left, dtype=float) * REGULATION_SECONDS
    return np.maximum(np.interp(seconds_left, model.seconds_left, model.spread), SPREAD_MINIMUM)


def lead_kept(left, model: WinModel) -> np.ndarray:
    """The share of a lead that reaches the final margin with a share `left` of the game left."""
    return model.lead_kept_at_end + model.lead_kept_per_game * np.asarray(left, dtype=float)


def win_probability(margin, left, model: WinModel, neutral=False, strength=0.0) -> np.ndarray:
    """The chance the home team wins from `margin` with a share `left` of the game left; `strength` is `strength_gap`,
    0 for two teams of average strength.

    Example: tied at tip-off with a home edge of 3 and a spread of 11, Phi(3 / 11) = 0.61."""
    left = np.asarray(left, dtype=float)
    edge = model.home_edge * (1.0 - np.asarray(neutral, dtype=float))
    drift = (edge + model.strength_slope * np.asarray(strength, dtype=float)) * left
    return norm.cdf((lead_kept(left, model) * np.asarray(margin, dtype=float) + drift) / spread_at(left, model))


def leverage(margin, left, model: WinModel, neutral=False) -> np.ndarray:
    """What one point is worth in wins at this game state, at average team strength: half the difference between a
    point more and a point less.

    Example: tied with a minute left and a spread of 3.5, about 0.11; up 20 with a minute left, about 0."""
    margin = np.asarray(margin, dtype=float)
    above = win_probability(margin + 1.0, left, model, neutral)
    return (above - win_probability(margin - 1.0, left, model, neutral)) / 2.0


def _stint_leverage(stints: pd.DataFrame, seconds: np.ndarray, model: WinModel) -> np.ndarray:
    """Each stint's leverage, at its starting margin and the middle of its time."""
    left = share_left(stints.period, stints.start_clock, seconds)[1]
    neutral = stints.neutral.fillna(False).to_numpy(dtype=float)
    return leverage(stints.margin_h.to_numpy(dtype=float), left, model, neutral)


def mean_leverage(stints: pd.DataFrame, seconds: np.ndarray, model: WinModel) -> float:
    """The possession-weighted mean leverage of these stints."""
    return float(np.average(_stint_leverage(stints, seconds, model), weights=stints.poss_h + stints.poss_a))


def relative_leverage(stints: pd.DataFrame, seconds: np.ndarray, model: WinModel) -> np.ndarray:
    """Each stint's leverage over the mean of these stints: 1 is an average possession.

    Example: 0.09 late in a close game against a season's mean of 0.03 reads 3."""
    return _stint_leverage(stints, seconds, model) / mean_leverage(stints, seconds, model)


def leverage_inside_games(stints: pd.DataFrame, seconds: np.ndarray, model: WinModel) -> np.ndarray:
    """Each stint's weight: its leverage over the possession-weighted mean of its own game, so every game's possessions
    average 1.

    Example: a game decided after 60 possessions, the first 60 at a leverage of 0.03 and the last 40 at about 0, reads
    about 1.67 and 0; a close game, 80 at 0.03 and the last 20 at 0.09, reads 0.71 and 2.14."""
    lever = _stint_leverage(stints, seconds, model)
    possessions = (stints.poss_h + stints.poss_a).to_numpy(dtype=float)
    frame = pd.DataFrame({"game": stints.game_id.to_numpy(), "levered": lever * possessions,
                          "possessions": possessions})
    totals = frame.groupby("game")[["levered", "possessions"]].transform("sum")
    mean = np.divide(totals.levered.to_numpy(), totals.possessions.to_numpy(), out=np.zeros(len(frame)),
                     where=totals.possessions.to_numpy() > 0.0)
    return np.divide(lever, mean, out=np.ones(len(frame)), where=mean > 0.0)


def _team_season(results: pd.DataFrame, home_values: pd.Series) -> pd.DataFrame:
    """Per team: wins above half its games, and `home_values` (the home team's view, per game) from its own view."""
    margin = results.margin.to_numpy(dtype=float)
    values = home_values.reindex(results.game_id.to_numpy()).to_numpy(dtype=float)
    return pd.DataFrame({"team": np.concatenate([results.home_team_id, results.away_team_id]),
                         "wins_above": np.concatenate([(margin > 0.0) - 0.5, (margin < 0.0) - 0.5]),
                         "value": np.concatenate([values, -values])}).groupby("team").sum()


def net_points_per_win(results: pd.DataFrame) -> float:
    """Net points per win from the standings alone: 1 over the slope of a team's wins on its net points, across a
    season's teams.

    Example: 29.9 in 2006, the usual rule of thumb; 37.9 in 2026, when games are decided by more."""
    season = _team_season(results, results.set_index("game_id").margin)
    return 1.0 / float(np.polyfit(season.value, season.wins_above, 1)[0])


def rating_points_per_win(stints: pd.DataFrame, weight: np.ndarray, margin: np.ndarray) -> float:
    """Rating points per win: 1 over the slope of a team's wins on its expected margin from the published ratings, in
    points and weighted by `weight`, across a season's teams.  What `WinsSettings.points_per_win` is read
    from: at this number a team's lineups add up to its record."""
    points = weight * margin * (stints.poss_h + stints.poss_a).to_numpy(dtype=float) / 2.0 / 100.0
    per_game = pd.Series(points).groupby(stints.game_id.to_numpy()).sum()
    season = _team_season(game_results(stints), per_game)
    return 1.0 / float(np.polyfit(season.value, season.wins_above, 1)[0])


# ------------------------------------------------------------------------------------------- the pace ridge
@dataclass(frozen=True)
class PaceRows:
    """The pace ridge's rows: one per stint long enough to read a pace from."""
    player_ids: np.ndarray         # ascending; a column of `players` per id
    players: sp.csr_matrix         # stints x players: 1 for each of the ten on the court, both teams share the pace
    context: np.ndarray            # stints x PACE_CONTEXT
    target: np.ndarray             # possessions per 48 minutes per team
    weight: np.ndarray             # seconds


PACE_CONTEXT = ("level", "garbage_time", "margin", "margin_x_share_left", "period_end")


def _lineup_matrix(lineups: np.ndarray, player_ids: np.ndarray) -> sp.csr_matrix:
    """Rows x players: a 1 for each id in a row of `lineups`; `player_ids` ascending and holding every id."""
    rows, width = lineups.shape
    return sp.csr_matrix((np.ones(rows * width), (np.repeat(np.arange(rows), width),
                                                  np.searchsorted(player_ids, lineups.ravel()))),
                         shape=(rows, len(player_ids)))


def pace_rows(stints: pd.DataFrame, seconds: np.ndarray, min_seconds: float) -> PaceRows:
    """The rows of `stints` lasting `min_seconds` or more (and any time at all).

    The target is 2,880 x (possessions at both ends / 2) / seconds and the weight seconds, the design's rule: response
    numerator / denominator, weight denominator.  A short stint's clock is mostly the moment of a substitution at a free
    throw, not pace: in 2025 the stints under 15 seconds are 0.5% of the time and 43% of the weighted squared error of
    this target (mean under 5 seconds: 875 possessions per 48).  They leave this fit only.

    Blowouts, close games and period ends change how fast anyone plays, so they are context, not the ten players."""
    keep = (seconds > 0.0) & (seconds >= float(min_seconds))
    rows = stints.loc[keep]
    kept_seconds = seconds[keep]
    lineups = rows[[*HOME, *AWAY]].to_numpy(dtype=np.int64)
    player_ids = np.unique(lineups)
    margin = np.minimum(np.abs(rows.margin_h.to_numpy(dtype=float)), MARGIN_CLIP)
    left = share_left(rows.period, rows.start_clock, kept_seconds)[0]
    context = np.column_stack([np.ones(len(rows)), rows.is_gt.to_numpy(dtype=float), margin, margin * left,
                               (rows.start_clock.to_numpy(dtype=float) <= PERIOD_END_SECONDS).astype(float)])
    target = REGULATION_SECONDS * (rows.poss_h + rows.poss_a).to_numpy(dtype=float) / 2.0 / kept_seconds
    return PaceRows(player_ids, _lineup_matrix(lineups, player_ids), context, target, kept_seconds)


@dataclass(frozen=True)
class PaceFit:
    """A fitted pace ridge."""
    player_ids: np.ndarray
    effects: np.ndarray            # possessions per 48 minutes, per player
    context: np.ndarray            # per PACE_CONTEXT


def solve_pace(rows: PaceRows, penalty: float, sums=None) -> PaceFit:
    """The ridge of `rows`, one penalty on the players, none on the context.  `sums` are the rows' normal equations,
    kept by a caller that solves one set of rows at several penalties."""
    sums = sums if sums is not None else normal_equations(rows.players, rows.context, rows.target, rows.weight)
    gram = np.block([[sums.gram_players, sums.gram_players_context], [sums.gram_players_context.T, sums.gram_context]])
    right_hand_side = np.concatenate([sums.rhs_players, sums.rhs_context])
    n_players = len(rows.player_ids)
    penalties = np.concatenate([np.full(n_players, float(penalty)), np.zeros(rows.context.shape[1])])
    theta = solve_ridge(gram, right_hand_side, penalties)
    return PaceFit(rows.player_ids, theta[:n_players], theta[n_players:])


def pace_error(fit: PaceFit, rows: PaceRows) -> float:
    """The seconds-weighted mean squared miss of `fit` on `rows`; a player the fit never saw reads 0."""
    place = np.searchsorted(fit.player_ids, rows.player_ids)
    place = np.minimum(place, len(fit.player_ids) - 1)
    known = fit.player_ids[place] == rows.player_ids
    effects = np.where(known, fit.effects[place], 0.0)
    predicted = rows.players @ effects + rows.context @ fit.context
    return float(np.average((rows.target - predicted) ** 2, weights=rows.weight))


def pace_effects(stints: pd.DataFrame, seconds: np.ndarray, penalty: float, min_seconds: float) -> pd.DataFrame:
    """player_id, pace_effect: each player's effect on possessions per 48 minutes in these stints."""
    fit = solve_pace(pace_rows(stints, seconds, min_seconds), penalty)
    return pd.DataFrame({"player_id": fit.player_ids, "pace_effect": fit.effects})


# ------------------------------------------------------------------------------------------- the wins
def player_index(stints: pd.DataFrame) -> tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix]:
    """(every player id in the stints, ascending; stints x players for the home five; the same for the away five)."""
    home, away = stints[HOME].to_numpy(dtype=np.int64), stints[AWAY].to_numpy(dtype=np.int64)
    player_ids = np.unique(np.concatenate([home.ravel(), away.ravel()]))
    return player_ids, _lineup_matrix(home, player_ids), _lineup_matrix(away, player_ids)


def starts_and_minutes(stints: pd.DataFrame, seconds: np.ndarray) -> pd.DataFrame:
    """player_id, starts_rs, minutes_rs: games in which he is in the opening stint, and his seconds over 60.

    The stints are in time order within a game, so a game's first stint is its opening lineup.  The roles table is not
    the source: it pools playoff starts in, so a deep run would move a player up the list the bar reads."""
    player_ids, home, away = player_index(stints)
    opening = (~stints.game_id.duplicated()).to_numpy(dtype=float)
    return pd.DataFrame({"player_id": player_ids,
                         "starts_rs": np.asarray(home.T @ opening + away.T @ opening).round().astype(int),
                         "minutes_rs": np.asarray(home.T @ seconds + away.T @ seconds) / 60.0})


def _ratings_for(player_ids: np.ndarray, ratings: pd.DataFrame, column: str) -> np.ndarray:
    """`column` of the season's ratings in the order of `player_ids`; a player without a row counts as average, 0."""
    return ratings.set_index("player_id")[column].reindex(player_ids).fillna(0.0).to_numpy(dtype=float)


def expected_margins(stints: pd.DataFrame, ratings: pd.DataFrame) -> tuple[np.ndarray, float]:
    """(per stint, the home team's expected margin per 100 from the ten published ratings and the home edge; that edge).

    The edge is the possession-weighted mean of what the ratings leave of the home team's actual margin per 100, over
    stints at a home court; ratings are centred, so nothing else about the level matters to a margin."""
    player_ids, home, away = player_index(stints)
    totals = _ratings_for(player_ids, ratings, "rating_total")
    from_ratings = np.asarray(home @ totals - away @ totals)
    possessions = (stints.poss_h + stints.poss_a).to_numpy(dtype=float) / 2.0
    at_home = ~stints.neutral.fillna(False).to_numpy(dtype=bool)
    usable = at_home & (possessions > 0.0)
    actual = 100.0 * (stints.pts_h - stints.pts_a).to_numpy(dtype=float)[usable] / possessions[usable]
    edge = float(weighted_least_squares(np.ones((int(usable.sum()), 1)), actual - from_ratings[usable],
                                        possessions[usable])[0])
    return from_ratings + edge * at_home, edge


@dataclass(frozen=True)
class Bar:
    """The starter average: the mean of the 150 most-started players."""
    off: float
    defense: float
    total: float
    pace: float
    n: int                         # the players it averages: 150, or everyone rated in a season's first days
    fewest_starts: int             # the games started by the last of them


def starter_bar(ratings: pd.DataFrame, starts: pd.DataFrame, pace: pd.DataFrame, n_starters: int) -> Bar:
    """The `n_starters` rated players with the most regular-season starts, minutes breaking ties (the owner, 2026-11-02:
    "sort by games started, minutes played is the tiebreaker"), their player ids breaking any tie left, so the bar never
    depends on the order of a file.  A starter the pace ridge never saw (his stints were all too short) reads 0."""
    ranked = starts.merge(ratings[["player_id", "rating_off", "rating_def", "rating_total"]], on="player_id")
    ranked = ranked.merge(pace, on="player_id", how="left").fillna({"pace_effect": 0.0})
    top = ranked.sort_values(["starts_rs", "minutes_rs", "player_id"], ascending=[False, False, True]).head(n_starters)
    return Bar(float(top.rating_off.mean()), float(top.rating_def.mean()), float(top.rating_total.mean()),
               float(top.pace_effect.mean()), len(top), int(top.starts_rs.min()))


def season_wins(stints: pd.DataFrame, seconds: np.ndarray, ratings: pd.DataFrame, weight: np.ndarray,
                pace: pd.DataFrame, bar: Bar, points_per_win: float) -> pd.DataFrame:
    """One row per player in the stints: player_id and the columns WINS_COLUMNS, from the equation at the top of this
    module; `weight` is each stint's `leverage_inside_games`."""
    lever = weight / float(points_per_win)
    margin = expected_margins(stints, ratings)[0]
    player_ids, home, away = player_index(stints)
    poss_h, poss_a = stints.poss_h.to_numpy(dtype=float), stints.poss_a.to_numpy(dtype=float)
    above_off = _ratings_for(player_ids, ratings, "rating_off") - bar.off
    above_def = _ratings_for(player_ids, ratings, "rating_def") - bar.defense
    above_total = above_off + above_def
    effect = pace.set_index("player_id").pace_effect.reindex(player_ids).fillna(0.0).to_numpy(dtype=float)
    above_pace = effect - bar.pace

    own_end = np.asarray(home.T @ (lever * poss_h) + away.T @ (lever * poss_a))
    other_end = np.asarray(home.T @ (lever * poss_a) + away.T @ (lever * poss_h))
    per_share = lever * seconds / REGULATION_SECONDS
    margin_part = np.asarray(home.T @ (per_share * margin) - away.T @ (per_share * margin))
    time_part = np.asarray(home.T @ per_share + away.T @ per_share)
    wosa_off, wosa_def = above_off * own_end / 100.0, above_def * other_end / 100.0
    wosa_pace = above_pace * (margin_part - above_total * time_part) / 100.0

    own_side = np.asarray(home.T @ poss_h + away.T @ poss_a)
    other_side = np.asarray(home.T @ poss_a + away.T @ poss_h)
    mean_lever = float(np.average(lever, weights=poss_h + poss_a))

    def mean_weight(levered: np.ndarray, possessions: np.ndarray) -> np.ndarray:
        """His mean weight over these possessions, against the season's; 0 with none."""
        return np.divide(levered, possessions, out=np.zeros(len(player_ids)), where=possessions > 0.0) / mean_lever

    starts = starts_and_minutes(stints, seconds)
    return pd.DataFrame({"player_id": player_ids, "wosa_off": wosa_off, "wosa_def": wosa_def, "wosa_pace": wosa_pace,
                         "wosa": wosa_off + wosa_def + wosa_pace, "pace_effect": effect,
                         "leverage": mean_weight(own_end + other_end, own_side + other_side),
                         "starts_rs": starts.starts_rs.to_numpy(), "minutes_rs": starts.minutes_rs.to_numpy(),
                         "above_off": above_off, "above_def": above_def,
                         "leverage_off": mean_weight(own_end, own_side), "leverage_def": mean_weight(other_end, other_side)})


def counterfactual_wins(stints: pd.DataFrame, seconds: np.ndarray, ratings: pd.DataFrame, weight: np.ndarray,
                        pace: pd.DataFrame, bar: Bar, points_per_win: float) -> pd.Series:
    """Each player's wins computed the long way, one stint and player at a time: his team's expected points for and
    against with him, minus the same with an average starter in his place on the possessions that would then be played.
    `season_wins` must equal it to rounding: `tests/test_wins.py` holds it there, and the check over every season."""
    lever = weight / float(points_per_win)
    edge = expected_margins(stints, ratings)[1]
    player_ids, home, away = player_index(stints)
    offense = _ratings_for(player_ids, ratings, "rating_off")
    defense = _ratings_for(player_ids, ratings, "rating_def")
    at_home = ~stints.neutral.fillna(False).to_numpy(dtype=bool)
    home_off, home_def = np.asarray(home @ offense), np.asarray(home @ defense)
    away_off, away_def = np.asarray(away @ offense), np.asarray(away @ defense)
    half_edge = edge * at_home / 2.0
    # per side: the possessions at his team's end, at the other end, and the two levels that add up to its margin
    sides = ((HOME, stints.poss_h.to_numpy(dtype=float), stints.poss_a.to_numpy(dtype=float),
              home_off - away_def + half_edge, home_def - away_off + half_edge),
             (AWAY, stints.poss_a.to_numpy(dtype=float), stints.poss_h.to_numpy(dtype=float),
              away_off - home_def - half_edge, away_def - home_off - half_edge))
    effects = pace.set_index("player_id").pace_effect
    totals = pd.Series(0.0, index=player_ids)
    for slots, own, other, scored, stopped in sides:
        for slot in slots:
            player = stints[slot].to_numpy(dtype=np.int64)
            above_off = pd.Series(offense, index=player_ids).reindex(player).to_numpy() - bar.off
            above_def = pd.Series(defense, index=player_ids).reindex(player).to_numpy() - bar.defense
            extra = (effects.reindex(player).fillna(0.0).to_numpy() - bar.pace) * seconds / REGULATION_SECONDS
            with_him = scored * own + stopped * other
            without = (scored - above_off) * (own - extra) + (stopped - above_def) * (other - extra)
            totals = totals.add(pd.Series(lever * (with_him - without) / 100.0).groupby(player).sum(),
                                fill_value=0.0)
    return totals


def wins_columns(rows: pd.DataFrame, wins: pd.DataFrame) -> pd.DataFrame:
    """`rows`, one season of a ratings table, with the columns WINS_COLUMNS after its own, in place of any it has.  A
    player with no regular-season stints, a playoff call-up, reads 0 on every one."""
    kept = rows.drop(columns=[column for column in WINS_COLUMNS if column in rows.columns])
    joined = kept.merge(wins[["player_id", *WINS_COLUMNS]], on="player_id", how="left", validate="one_to_one")
    joined[WINS_COLUMNS] = joined[WINS_COLUMNS].fillna(0.0)
    joined["starts_rs"] = joined.starts_rs.astype(int)
    return joined
