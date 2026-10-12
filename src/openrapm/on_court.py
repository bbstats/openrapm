"""His team's record with him on the court and with him off it: the season panel's on-court and off-court columns.

A player's on-court rating is his team's points per 100 possessions over the stints he played; his off-court rating
is the same over the stints of the same games that his team played without him.  Neither separates him from his
teammates (RAPM does that), so both lean toward whoever he played with; they are far less noisy than RAPM, and the
box-score prior may weigh the one against the other.  Both are measured on the luck-adjusted targets: the offence
design explains points with expected free throws, the defence design points with every opponent three repriced at
the shooter's padded percentage (design.py, data/shots.py), so the free-throw and opponent-three luck is already out.

Words used below
    design       one season's stints as rows (design.py): the five players on offence, the five on defence, the
                 target y (points per 100) and the weight w (possessions)
    on court     the possession-weighted mean of y over the rows he was on the court for: on offence for the
                 scored rating, on defence for the allowed rating
    off court    the same over the rows of the games he played in where his team was on the court without him
    level        the possession-weighted mean of a rating over the players, subtracted so 0 is average
    pad k        sigma^2 / tau^2 in possessions, the method of moments below

Both ratings are centred on the season's level and padded toward it (box_rates.shrink):

    padded = n x (x - level) / (n + k)
    tau^2  = Var_n(x - level) - sigma^2 E_n[1 / n]        k = sigma^2 / tau^2

    x        his rating, n the possessions behind it
    sigma^2  the possession-weighted variance of y over the design's rows: the noise of one possession
    tau^2    the spread of the true ratings between players: the observed spread minus the noise of n possessions

Example: with sigma^2 = 15,000 (points per 100, squared) and tau^2 = 21, k is about 700 possessions, a fifth of a
season's play: a 300-possession player's on-court +10 becomes 300 x 10 / 1,000 = +3.

The allowed ratings keep the design's raw sign (points the opponent scored per 100): lower is a better defence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .box_rates import shrink
from .features import OFF_COURT, ON_COURT

__all__ = ["on_court", "oncourt_rates", "offcourt_rates"]

NO_SPREAD_K = 1e9               # the pad k when the ratings show no spread beyond the noise: everyone at the level


def on_court(Zo, Zd, r, w) -> tuple:
    """Per player, the weighted mean of `r` over the rows he was on offence for and over those he was on defence for,
    with the weights behind each: (mean on offence, weight on offence, mean on defence, weight on defence).

    `Zo` and `Zd` are (rows x players) indicators of who was on offence and on defence; a player with no weight gets 0.
    """
    w = np.asarray(w, dtype=float)
    r = np.asarray(r, dtype=float)
    out = []
    for Z in (Zo, Zd):
        Z = sp.csr_matrix(Z)
        p = np.asarray(Z.T @ w).ravel()
        s = np.asarray(Z.T @ (w * r)).ravel()
        out += [np.where(p > 0, s / np.where(p > 0, p, 1.0), 0.0), p]
    return tuple(out)


def _court_columns(design) -> tuple:
    """(offence, defence): the design's two (rows x players) indicator blocks."""
    n_players = design.spec.n_players
    players = sp.csr_matrix(design.parts["Z"] if design.parts is not None else design.X[:, :2 * n_players])
    return players[:, :n_players], players[:, n_players:]


def _centre_and_pad(rating: np.ndarray, possessions: np.ndarray, response, weight) -> np.ndarray:
    """The ratings minus their level, padded toward 0 by the method of moments (the module's equations); unchanged
    when nobody has possessions."""
    if possessions.sum() <= 0:
        return rating
    level = float(np.average(rating, weights=possessions))
    centred = rating - level
    sigma2 = float(np.average((response - np.average(response, weights=weight)) ** 2, weights=weight))
    played = possessions > 0
    tau2 = float(np.average(centred[played] ** 2, weights=possessions[played])
                 - sigma2 * np.average(1.0 / possessions[played], weights=possessions[played]))
    k = sigma2 / tau2 if tau2 > 1e-9 else NO_SPREAD_K
    return shrink(centred, possessions, k, 0.0)


def _same_players(offence_design, defence_design) -> None:
    if offence_design.spec.n_players != defence_design.spec.n_players:
        raise ValueError("the two designs must share a player layout; build them from the same season")


def oncourt_rates(offence_design, defence_design) -> pd.DataFrame:
    """One row per player (design column order): `on_court_scored_per100` from the rows he was on offence for in the
    offence design, `on_court_allowed_per100` from the rows he was on defence for in the defence design, each centred
    and padded, and the possessions behind each."""
    _same_players(offence_design, defence_design)
    columns = {}
    for rating, possessions, design, side in ((ON_COURT[0], ON_COURT[2], offence_design, 0),
                                              (ON_COURT[1], ON_COURT[3], defence_design, 1)):
        mean, weight = on_court(*_court_columns(design), design.y, design.w)[2 * side:2 * side + 2]
        columns[rating] = _centre_and_pad(mean, weight, design.y, design.w)
        columns[possessions] = weight
    return pd.DataFrame(columns)[ON_COURT]


def _team_groups(design) -> tuple:
    """The design's rows grouped by (game, which team is on offence): group 2g + 1 when the home team of game g is on
    offence, 2g when the away team is.  Returns (rows x groups indicator, number of groups)."""
    game = design.rows["game_idx"].to_numpy()
    home_off = design.rows["is_home_off"].to_numpy(dtype=bool)
    group = 2 * game + home_off.astype(int)
    n_groups = 2 * (int(game.max()) + 1)
    return sp.csr_matrix((np.ones(len(group)), (np.arange(len(group)), group)), shape=(len(group), n_groups)), n_groups


def _his_team(offence_players, defence_players, groups, n_groups: int, side: int) -> np.ndarray:
    """(players x groups): 1 for the groups in which his team was on offence (side 0) or on defence (side 1), in the
    games he played.  His team in a game is read off the design: he is on the home team if he was on offence in more
    home-offence rows plus defence in away-offence rows than the other way round."""
    on_offence, on_defence = (offence_players.T @ groups).tocsr(), (defence_players.T @ groups).tocsr()
    home = np.asarray((on_offence[:, 1::2] + on_defence[:, 0::2]).todense())
    away = np.asarray((on_offence[:, 0::2] + on_defence[:, 1::2]).todense())
    is_home = home > away
    played = (home + away) > 0
    team = np.zeros((offence_players.shape[1], n_groups))
    home_groups, away_groups = (slice(1, None, 2), slice(0, None, 2)) if side == 0 else (slice(0, None, 2),
                                                                                         slice(1, None, 2))
    team[:, home_groups] = played & is_home
    team[:, away_groups] = played & ~is_home
    return team


def _off_court(design, side: int) -> tuple:
    """(his team's points per 100 without him, the possessions behind it) on offence (side 0) or defence (side 1):
    his team's totals over its groups in his games, minus his own on-court totals."""
    offence_players, defence_players = _court_columns(design)
    y, w = np.asarray(design.y, dtype=float), np.asarray(design.w, dtype=float)
    groups, n_groups = _team_groups(design)
    group_points = np.asarray(groups.T @ (w * y)).ravel()
    group_possessions = np.asarray(groups.T @ w).ravel()
    team = _his_team(offence_players, defence_players, groups, n_groups, side)
    his = offence_players if side == 0 else defence_players
    points = team @ group_points - np.asarray(his.T @ (w * y)).ravel()
    possessions = np.maximum(team @ group_possessions - np.asarray(his.T @ w).ravel(), 0.0)
    rating = np.where(possessions > 0, points / np.where(possessions > 0, possessions, 1.0), 0.0)
    return _centre_and_pad(rating, possessions, y, w), possessions


def offcourt_rates(offence_design, defence_design) -> pd.DataFrame:
    """One row per player (design column order): his team's points per 100 scored (`off_court_scored_per100`, from the
    offence design) and allowed (`off_court_allowed_per100`, from the defence design) in the games he played, over
    the stints he sat, centred and padded the same way as the on-court ones so on minus off is on one scale, and the
    possessions behind each.  Games he did not play in count nothing: his team without him for a whole game is
    another question."""
    _same_players(offence_design, defence_design)
    scored, scored_possessions = _off_court(offence_design, 0)
    allowed, allowed_possessions = _off_court(defence_design, 1)
    return pd.DataFrame(dict(zip(OFF_COURT, (scored, allowed, scored_possessions, allowed_possessions))))
