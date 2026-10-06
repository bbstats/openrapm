"""B1, experiment 35's first baseline: a LINEAR SPM from the box score alone.

Standard practice, with no play-by-play anywhere in the inputs: each player's 13 box counts per 100 box-estimated
possessions, centered on the league and padded toward it, plus his share of the minutes available in the games he
played; one weighted least squares per side onto a RAPM label.

Box-estimated possessions, per team-game:  FGA + 0.44 FTA - OREB + TOV, averaged with the opponent's (one number per
game); a player's possessions are his minutes over a fifth of his team's minutes, times the game's possessions.  The
stints' possession counts are play-by-play and are not used.

Every number a rating reads comes from the games it is handed: league rates from the same games, padding constants
from OTHER seasons (passed in), the label from seasons the scored ones never touch (the caller's job).  Model layer:
nothing here reads a file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .exposure import split_half_k
from .pad import shrink

COUNTS = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl", "blk", "pf"]
RATES = [f"r_{c}" for c in COUNTS]
INPUTS = RATES + ["min_share"]


def with_possessions(box: pd.DataFrame) -> pd.DataFrame:
    """Per player-game box rows plus `poss` (box-estimated) and `avail` (a fifth of his team's minutes)."""
    b = box.copy()
    b["_fga"] = b.fg3m + b.fg3_miss + b.fg2m + b.fg2_miss
    b["_fta"] = b.ftm + b.ft_miss
    team = b.groupby(["game_id", "team_id"]).agg(fga=("_fga", "sum"), fta=("_fta", "sum"), orb=("orb", "sum"),
                                                 tov=("tov", "sum"), minutes=("minutes", "sum")).reset_index()
    team["raw"] = team.fga + 0.44 * team.fta - team.orb + team.tov
    both = team.groupby("game_id").raw.agg(["sum", "size"]).rename(columns={"sum": "raw2", "size": "n_teams"})
    team = team.join(both, on="game_id")
    # one number per game: the mean of the two teams' estimates (a game with one team's rows uses that team's)
    team["game_poss"] = np.where(team.n_teams == 2, team.raw2 / 2.0, team.raw)
    b = b.merge(team[["game_id", "team_id", "minutes", "game_poss"]].rename(columns={"minutes": "team_minutes"}),
                on=["game_id", "team_id"], how="left")
    b["avail"] = b.team_minutes / 5.0
    b["poss"] = np.where(b.avail > 0, b.minutes / b.avail.where(b.avail > 0, 1.0) * b.game_poss, 0.0)
    return b.drop(columns=["_fga", "_fta"])


def padding_k(box: pd.DataFrame) -> np.ndarray:
    """Per-count padding constants (possession units) for one season's player-games: odd/even split halves
    (`exposure.split_half_k`); NaN where a count cannot be estimated."""
    b = box[box.poss > 0]
    ps = pd.factorize(b.player_id)[0]
    game = pd.factorize(b.game_id, sort=True)[0]
    k, _, _, _ = split_half_k(ps, game, b[COUNTS].to_numpy(float), b.poss.to_numpy(float),
                              np.zeros(ps.max() + 1, dtype=int), 1)
    return k[0]


def player_inputs(box: pd.DataFrame, k: np.ndarray) -> pd.DataFrame:
    """One row per player: possessions, minutes share, and the 13 per-100 rates centered on THESE games' league rate
    and padded toward it with constants `k` (13 values, from other seasons)."""
    b = box[box.poss > 0]
    tot = b.groupby("player_id")[COUNTS + ["poss", "minutes", "avail"]].sum()
    league = 100.0 * tot[COUNTS].sum() / tot.poss.sum()
    out = pd.DataFrame(index=tot.index)
    out["poss"] = tot.poss
    out["min_share"] = np.where(tot.avail > 0, tot.minutes / tot.avail.where(tot.avail > 0, 1.0), 0.0)
    for j, c in enumerate(COUNTS):
        rate = 100.0 * tot[c] / tot.poss - league[c]
        out[f"r_{c}"] = shrink(rate.to_numpy(float), tot.poss.to_numpy(float), float(k[j]), 0.0)
    return out.reset_index()


class LinearBoxSPM:
    """One weighted ridge per side: label ~ intercept + INPUTS, the inputs standardized, a light penalty."""

    def __init__(self, penalty: float = 1e-3):
        self.penalty = float(penalty)

    def fit(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> "LinearBoxSPM":
        A = X[INPUTS].to_numpy(float)
        w = np.asarray(w, dtype=float)
        self.mean_ = np.average(A, axis=0, weights=w)
        self.sd_ = np.sqrt(np.average((A - self.mean_) ** 2, axis=0, weights=w))
        self.sd_[self.sd_ == 0] = 1.0
        S = np.column_stack([np.ones(len(A)), (A - self.mean_) / self.sd_])
        pen = self.penalty * w.sum() * np.eye(S.shape[1])
        pen[0, 0] = 0.0
        self.coef_ = np.linalg.solve((S * w[:, None]).T @ S + pen, (S * w[:, None]).T @ np.asarray(y, float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        A = X[INPUTS].to_numpy(float)
        return np.column_stack([np.ones(len(A)), (A - self.mean_) / self.sd_]) @ self.coef_

    def coefficients(self) -> pd.Series:
        """Per input, points per 100 per one-unit change in the (padded) input."""
        return pd.Series(np.r_[self.coef_[0] - (self.coef_[1:] * self.mean_ / self.sd_).sum(),
                               self.coef_[1:] / self.sd_], index=["intercept"] + INPUTS)


def centered(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Possession-weighted mean zero, as every rankings table is centered."""
    w = np.asarray(weights, dtype=float)
    return np.asarray(values, dtype=float) - np.average(values, weights=w) if w.sum() > 0 else values
