"""The Bias page's second column: where the season's team-game results rate players higher or lower than OpenRAPM,
stat by stat, with no ridge.

The question is the consensus column's, asked of the games instead: among the season's players, do those with more of
a stat get more from OpenRAPM than the games give them, or less?

Words used below
    team-game     one team's offense in one game: its points per 100 possessions on the build's target for that side
                  (settings.Targets), weighted by its possessions
    share         a player's share of his team's possessions in that game; the five on the court sum to 5
    prediction    the attackers' published offense plus the defenders' published defense, each times his share
    miss          the team-game's points minus the prediction
    z             a player's stat in standard deviations of the page's compared players (bias.stat_z); 0, the mean,
                  for a player the season panel does not hold
    width         one free multiplier per side on the prediction, so OpenRAPM's spread being too narrow or too wide for
                  the games is taken out first, as the consensus column's width match takes it out

The fit, per stat, is ordinary weighted least squares, no penalty, on the season's team-games:

    miss = b_O (the attackers' share x z, summed) + b_D (the defenders' share x z, summed)
           + m_O (the attackers' prediction) + m_D (the defenders' prediction) + the season's level, home court and
           the playoff columns

b_O is read off the fit on the offense target and b_D off the fit on the defense target, the same team-games and
columns both times.  If the games rate every player c z points per 100 higher on offense than OpenRAPM does, b_O is c.
On defense a better defender allows fewer points, so the games' lean there is -b_D.  The column is OpenRAPM minus the
games, per player and per standard deviation, as the consensus column is OpenRAPM minus the consensus:

    vs game results = -(b_O - b_D) = b_D - b_O

Its standard error is how far the number moves when each team's games are left out in turn (the leave-one-team-out
jackknife, K the teams):
    se^2 = (K - 1) / K x sum over teams of (the number without that team's games - their mean)^2
A player's miss repeats in every game he plays, nearly all for one team, so leaving a team out resamples its players
along with its games.  Two narrower errors were tried first and are too small (2026-11-24):
  - clustered by game alone: on a simulated season, the true ratings handed in read +0.20 against 0.05;
  - the spread over 200 shuffles of the stat among the players: 0.21 to 0.26 on 2027's rows, against 0.16 to 0.92
    left-one-team-out; a shuffled stat loses its overlap with the ratings themselves, which is what widens the error
    of a stat such as made twos.

Example: OpenRAPM rates every player 0.5 per 100 too low on offense per standard deviation of assists and is right on
defense.  Then b_O = 0.5, b_D = 0, and the assists row reads -0.5: OpenRAPM rates good passers 0.5 lower than the
games do.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import trade_set

SIDES = ("offense", "defense")


@dataclass
class SeasonGames:
    """One season's team-games on each side's target, with the published rating in raw sign for every player column:
    {side: TradeSet}, the offset (offense then defense, `trade_set.offset_vector`) and each team-game's two teams,
    team_off and team_def, in the TradeSets' row order."""
    games: dict
    offset: np.ndarray
    teams: pd.DataFrame

    @property
    def player_ids(self) -> np.ndarray:
        return self.games["offense"].player_ids

    @property
    def team_games(self) -> int:
        return int(self.games["offense"].Z.shape[0])


def raw_ratings(table: pd.DataFrame, season: int) -> pd.DataFrame:
    """player_id, o, d, poss of `season` from a ratings table stored positive-good (rating_off, rating_def, poss_off):
    raw sign, as the design's columns carry it.  Example: rating_def +1.5 reads d = -1.5, 1.5 fewer points allowed."""
    rows = table[table.season == season]
    return pd.DataFrame({"player_id": rows.player_id.astype("int64").to_numpy(),
                         "o": rows.rating_off.astype(float).to_numpy(),
                         "d": -rows.rating_def.astype(float).to_numpy(),
                         "poss": rows.poss_off.astype(float).to_numpy()})


def season_games(designs: dict, table: pd.DataFrame, season: int, teams_of: Callable) -> SeasonGames:
    """The season's team-games from `designs`, {"offense": design on the offense target, "defense": on the defense
    target} (the command loads them), the published rating of every player column, and the teams of each row,
    `teams_of(keys)` (data.tables.team_of_rows in the command).  A player the ratings table has no row for is filled
    at a quarter of its under-500-possession players' mean (trade_set.replacement_fill)."""
    games = {side: trade_set.team_game_design(designs[side]) for side in SIDES}
    if not np.array_equal(games["offense"].player_ids, games["defense"].player_ids):
        raise ValueError("the two targets built different player columns")
    if not (np.allclose(games["offense"].w, games["defense"].w)
            and games["offense"].keys.game_id.equals(games["defense"].keys.game_id)):
        raise ValueError("the two targets hold their team-games in different orders or weights")
    ratings = raw_ratings(table, season)
    offset = trade_set.offset_vector(games["offense"].player_ids, ratings, *trade_set.replacement_fill(ratings))
    teams = teams_of(games["offense"].keys).reset_index(drop=True)
    if len(teams) != len(games["offense"].keys):
        raise ValueError("the teams name a different number of team-games")
    return SeasonGames(games, offset, teams)


def _columns(ts, offset: np.ndarray, z: np.ndarray | None) -> np.ndarray:
    """The fit's columns: [attackers' z, defenders' z] when `z` is given, then the two predictions, then the fixed
    columns that are not all zero (a regular season has no playoff rows)."""
    n = ts.n_players
    blocks = [] if z is None else [np.column_stack([ts.Z[:, :n] @ z, ts.Z[:, n:] @ z])]
    fixed = ts.F[:, np.any(ts.F != 0.0, axis=0)]
    return np.column_stack([*blocks, trade_set.rating_columns(ts, offset), fixed])


def _fit(ts, offset: np.ndarray, z: np.ndarray | None) -> np.ndarray:
    """The coefficients of the weighted least squares of the miss on `_columns`."""
    columns = _columns(ts, offset, z)
    weighted = columns * ts.w[:, None]
    return np.linalg.solve(columns.T @ weighted, weighted.T @ (ts.y - ts.Z @ offset))


def multipliers(season: SeasonGames) -> dict:
    """{"offense": m, "defense": m}: what each side's prediction is multiplied by to fit the games, no stat in the
    fit.  Above 1, OpenRAPM's spread is narrower than the games'."""
    fits = {side: _fit(season.games[side], season.offset, None) for side in SIDES}
    return {"offense": 1.0 + float(fits["offense"][0]), "defense": 1.0 + float(fits["defense"][1])}


def _point(games: dict, offset: np.ndarray, z: np.ndarray) -> float:
    """b_D - b_O for one stat: OpenRAPM minus the games, per standard deviation."""
    defense = _fit(games["defense"], offset, z)[1]
    return float(defense - _fit(games["offense"], offset, z)[0])


def _without(ts, keep: np.ndarray):
    """The TradeSet's rows where `keep` is true."""
    return replace(ts, Z=ts.Z[keep], F=ts.F[keep], y=ts.y[keep], w=ts.w[keep],
                   keys=ts.keys[keep].reset_index(drop=True))


def lean(season: SeasonGames, z: np.ndarray) -> tuple[float, float]:
    """(OpenRAPM minus the games per standard deviation of one stat, its leave-one-team-out standard error).  `z` is
    the stat in standard deviations for every player column, in `season.player_ids` order.  A stat the fit cannot
    tell from the other columns (no spread on the court) has no lean: (nan, nan)."""
    team_off, team_def = season.teams.team_off.to_numpy(), season.teams.team_def.to_numpy()
    try:
        value = _point(season.games, season.offset, z)
        left_out = np.array([_point({side: _without(season.games[side], (team_off != team) & (team_def != team))
                                     for side in SIDES}, season.offset, z)
                             for team in np.unique(np.concatenate([team_off, team_def]))])
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    k = len(left_out)
    return value, float(np.sqrt((k - 1) / k * np.sum((left_out - left_out.mean()) ** 2)))


def lean_table(season: SeasonGames, z: pd.DataFrame, stats) -> pd.DataFrame:
    """stat, games, games_se: `lean` for each of `stats`, from `z` (player_id and one column per stat, bias.stat_z).
    A player column `z` does not hold counts at 0, the mean."""
    aligned = pd.DataFrame({"player_id": season.player_ids}).merge(z, on="player_id", how="left").fillna(0.0)
    rows = []
    for stat in stats:
        value, error = lean(season, aligned[stat].to_numpy(float))
        rows.append({"stat": stat, "games": value, "games_se": error})
    return pd.DataFrame(rows)
