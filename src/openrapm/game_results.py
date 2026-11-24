"""The Bias page's game-results columns: where the team-game results rate players higher or lower than OpenRAPM, stat by
stat, with no ridge, for the season the page opens on and for the last 10 seasons together.

The question is the consensus column's, asked of the games instead: among the players, do those with more of a stat
get more from OpenRAPM than the games give them, or less?

Words used below
    team-game     one team's offense in one game: its points per 100 possessions on the build's target for that side
                  (settings.Targets), weighted by its possessions
    share         a player's share of his team's possessions in that game; the five on the court sum to 5
    prediction    the attackers' published offense plus the defenders' published defense of that season, each times
                  his share
    miss          the team-game's points minus the prediction
    z             a player's stat in standard deviations of his season's players (bias.stat_z); 0, the mean, for a
                  player the season panel does not hold
    width         one free multiplier per side and season on the prediction, so OpenRAPM's spread being too narrow or
                  too wide for a season's games is taken out first, as the consensus column's width match takes it out

The fit, per stat, is ordinary weighted least squares, no penalty, on the team-games of every season asked for:

    miss = b_O (the attackers' share x z, summed) + b_D (the defenders' share x z, summed)
           + per season: m_O (the attackers' prediction) + m_D (the defenders' prediction) + its level, home court and
             the playoff columns

b_O and b_D are one number each for all the seasons; everything after them is the season's own.  b_O is read off the
fit on the offense target and b_D off the fit on the defense target, the same team-games and columns both times.  If
the games rate every player c z points per 100 higher on offense than OpenRAPM does, b_O is c.  On defense a better
defender allows fewer points, so the games' lean there is -b_D.  The column is OpenRAPM minus the games, per player
and per standard deviation, as the consensus column is OpenRAPM minus the consensus:

    vs game results = -(b_O - b_D) = b_D - b_O

Its standard error is how far the number moves when each team's games are left out in turn, in every season at once
(the leave-one-team-out jackknife, K the teams; a franchise keeps its NBA id when it moves):
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
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import block_diag

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
    target} (the command loads them), the season's published rating of every player column, and the teams of each
    row, `teams_of(keys)` (data.tables.team_of_rows in the command).  A player the ratings table has no row for is
    filled at a quarter of its under-500-possession players' mean (trade_set.replacement_fill)."""
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


@dataclass
class Rows:
    """Every asked season's team-games on one side's target, stacked: the fit's columns (the two stat columns when a
    stat is asked, then each season's own columns, block by block), the miss, the weights and each row's two teams."""
    columns: np.ndarray
    miss: np.ndarray
    w: np.ndarray
    team_off: np.ndarray
    team_def: np.ndarray

    def without(self, team) -> Rows:
        keep = (self.team_off != team) & (self.team_def != team)
        return Rows(self.columns[keep], self.miss[keep], self.w[keep], self.team_off[keep], self.team_def[keep])

    def fit(self) -> np.ndarray:
        """The coefficients of the weighted least squares of the miss on the columns."""
        weighted = self.columns * self.w[:, None]
        return np.linalg.solve(self.columns.T @ weighted, weighted.T @ self.miss)


def rows(seasons: list, side: str, zs: list | None = None) -> Rows:
    """`Rows` of `seasons` (SeasonGames) on `side`'s target; with `zs`, one z vector per season in its player-column
    order, the attackers' and the defenders' share x z, summed, come first.  A season's own columns are its two
    predictions and its fixed columns that are not all zero (a regular season so far has no playoff rows)."""
    shared, own, miss, w, team_off, team_def = [], [], [], [], [], []
    for position, season in enumerate(seasons):
        ts = season.games[side]
        n = ts.n_players
        if zs is not None:
            shared.append(np.column_stack([ts.Z[:, :n] @ zs[position], ts.Z[:, n:] @ zs[position]]))
        own.append(np.column_stack([trade_set.rating_columns(ts, season.offset), ts.F[:, np.any(ts.F != 0.0, axis=0)]]))
        miss.append(ts.y - ts.Z @ season.offset)
        w.append(ts.w)
        team_off.append(season.teams.team_off.to_numpy())
        team_def.append(season.teams.team_def.to_numpy())
    columns = block_diag(*own)
    if zs is not None:
        columns = np.column_stack([np.vstack(shared), columns])
    return Rows(columns, np.concatenate(miss), np.concatenate(w), np.concatenate(team_off), np.concatenate(team_def))


def multipliers(season: SeasonGames) -> dict:
    """{"offense": m, "defense": m}: what each side's prediction is multiplied by to fit the season's games, no stat
    in the fit.  Above 1, OpenRAPM's spread is narrower than the games'."""
    return {"offense": 1.0 + float(rows([season], "offense").fit()[0]),
            "defense": 1.0 + float(rows([season], "defense").fit()[1])}


def lean(seasons: list, zs: list) -> tuple[float, float]:
    """(OpenRAPM minus the games per standard deviation of one stat over `seasons`, its leave-one-team-out standard
    error).  `zs` holds the stat in standard deviations for every player column of each season, in its
    `player_ids` order.  A stat the fit cannot tell from the other columns (no spread on the court) has no lean:
    (nan, nan)."""
    offense, defense = rows(seasons, "offense", zs), rows(seasons, "defense", zs)
    try:
        value = float(defense.fit()[1] - offense.fit()[0])
        left_out = np.array([defense.without(team).fit()[1] - offense.without(team).fit()[0]
                             for team in np.unique(np.concatenate([offense.team_off, offense.team_def]))])
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    k = len(left_out)
    return value, float(np.sqrt((k - 1) / k * np.sum((left_out - left_out.mean()) ** 2)))


def lean_table(seasons: list, zs: list, stats) -> pd.DataFrame:
    """stat, games, games_se: `lean` over `seasons` for each of `stats`, from `zs`, one table per season (player_id
    and one column per stat, bias.stat_z).  A player column a season's table does not hold counts at 0, the mean."""
    aligned = [pd.DataFrame({"player_id": season.player_ids}).merge(z, on="player_id", how="left").fillna(0.0)
               for season, z in zip(seasons, zs)]
    out = []
    for stat in stats:
        value, error = lean(seasons, [table[stat].to_numpy(float) for table in aligned])
        out.append({"stat": stat, "games": value, "games_se": error})
    return pd.DataFrame(out)
