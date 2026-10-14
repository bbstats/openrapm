"""The decomposition pages' model code: which rows of a ridge fit a player's rating comes from.

A ridge coefficient is a weighted sum of the rows' outcomes, so grouping the rows splits the coefficient exactly.
Both decomposition pages (commands/site.py) use this: the vanilla RAPM page, and the OpenRAPM page, which splits
step 3's games' part this way.

Words used below
    row             one stint seen from one side (design.py): five players on offence, five on defence
    games' part     u, how far the season's games moved a player from his box-score prior (step 3, prior_rapm.py);
                    one number per player per side, positive good
    unpenalised     the columns the ridge does not shrink: the context columns and the two lineup priors P_O, P_D
    on court        the rows he is one of the ten players of
    off court       his team's rows without him: in games he played, and in games he missed in a season he was on
                    the team
    team SOS        every other row: other teams' games, and his team's games in seasons he was not on it

The split.  Step 3 solves (G + Lambda) theta = B' W y, with B = [Z | C | P_O | P_D] (the players' columns, then the
unpenalised ones), W the diagonal matrix of the rows' possessions and Lambda the penalties.  Write M for the matrix
that was solved.  A player's coefficient is then a sum over rows,

    theta_j = sum over rows r of  h_r y_r,      h = W B M^-1 e_j

and h is orthogonal to every unpenalised column (B'h = e_j - Lambda M^-1 e_j, which is 0 where Lambda is 0).  So
the unpenalised columns' fitted part can be taken out of every outcome without changing the sum:

    u_j = sum over rows r of  h_r z_r,      z = y - C gamma - s_O P_O - s_D P_D

z is how far a row's result beat what its context and its ten players' box-score priors predicted.  Each group of
rows (on court, off court, team SOS) adds up its own h_r z_r.  Where the solve needed a jitter on the unpenalised
diagonal (rapm.solved_ridge: the 2020 bubble has an empty playoff home-court column), B'h is not exactly 0; the
leftover goes to team SOS and is reported, about 1e-8.

Example: Victor Wembanyama's 2026 offensive rating is 2.07 = box score 0.69 + games' part 1.20 + swap adjustment
0.18, and his games' part splits as 1.41 on court - 0.27 off court + 0.06 team SOS: his own possessions say he was 1.41
per 100 better than his context and priors predicted, and his team's possessions without him took 0.27 of it back.

Model layer: everything here takes frames and arrays.  commands/site.py loads the tables and the box scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import prior_rapm, rapm

__all__ = ["TEAM_CODE", "side_team", "team_codes", "his_teams", "row_groups", "lineups", "row_keys",
           "GamesPartFit", "refit", "row_weights", "split_games_part"]

TEAM_CODE = 10 ** 10           # season (or game) x TEAM_CODE + team id: one number per team-season or team-game


def side_team(teams: pd.Series, game_ids: np.ndarray, players: np.ndarray) -> np.ndarray:
    """Each row's team on one side: the team of that side's first player the box score lists (-1 if none is)."""
    got = np.full(len(game_ids), np.nan)
    for k in range(players.shape[1]):
        todo = np.isnan(got)
        if not todo.any():
            break
        index = pd.MultiIndex.from_arrays([game_ids[todo], players[todo, k]])
        got[todo] = teams.reindex(index).to_numpy(dtype=float)
    return np.where(np.isnan(got), -1, got).astype(np.int64)


def team_codes(rows: dict) -> dict:
    """Each row's attacking and defending team as team-season and team-game codes, and the share of rows the box
    scores cannot place.  A side the box scores cannot name gets a code of its own, so it is nobody's team."""
    team_att, team_def, season = rows["team_att"], rows["team_def"], rows["season"]
    game = pd.factorize(rows["game_ids"])[0].astype(np.int64)
    n_rows = len(team_att)
    unknown_att, unknown_def = -1 - np.arange(n_rows), -1 - n_rows - np.arange(n_rows)
    team_season = np.concatenate([np.where(team_att >= 0, season * TEAM_CODE + team_att, unknown_att),
                                  np.where(team_def >= 0, season * TEAM_CODE + team_def, unknown_def)])
    team_game = np.concatenate([np.where(team_att >= 0, game * TEAM_CODE + team_att, unknown_att),
                                np.where(team_def >= 0, game * TEAM_CODE + team_def, unknown_def)])
    season_code, game_code = pd.factorize(team_season)[0], pd.factorize(team_game)[0]
    return dict(ts_att=season_code[:n_rows], ts_def=season_code[n_rows:], gt_att=game_code[:n_rows],
                gt_def=game_code[n_rows:], n_ts=int(season_code.max()) + 1, n_gt=int(game_code.max()) + 1,
                unplaced=float(((team_att < 0) | (team_def < 0)).mean()))


def his_teams(rows_off, rows_def, keys: dict) -> tuple:
    """Per row: is the attacking (defending) side one of his team-seasons, and one of his team-games."""
    his_season, his_game = np.zeros(keys["n_ts"], dtype=bool), np.zeros(keys["n_gt"], dtype=bool)
    his_season[keys["ts_att"][rows_off]], his_season[keys["ts_def"][rows_def]] = True, True
    his_game[keys["gt_att"][rows_off]], his_game[keys["gt_def"][rows_def]] = True, True
    return (his_season[keys["ts_att"]], his_season[keys["ts_def"]],
            his_game[keys["gt_att"]], his_game[keys["gt_def"]])


def row_groups(rows_off, rows_def, keys: dict, n_rows: int) -> dict:
    """The groups of rows one player's coefficient is split over, as boolean masks over all `n_rows` rows.

    `rows_off` and `rows_def` are the rows he is on offence and on defence in; `keys` is `team_codes`.
        on        he is one of the ten players
        without   his team is playing, he is not, and he played in that game
        missed    his team is playing in a season he was on it, he is not, and he did not play in that game
        off_att   his team attacks without him in a game he played (`without`'s attacking half)
        off_def   his team defends without him in a game he played (`without`'s defending half)
    For a traded player, each of his teams' games he did not play counts as `missed` for the season he was there.
    Every other row is team SOS.
    """
    on = np.zeros(n_rows, dtype=bool)
    on[rows_off], on[rows_def] = True, True
    team_att, team_def, played_att, played_def = his_teams(rows_off, rows_def, keys)
    without = ((team_att & played_att) | (team_def & played_def)) & ~on
    missed = ((team_att & ~played_att) | (team_def & ~played_def)) & ~on & ~without
    return dict(on=on, without=without, missed=missed, off_att=team_att & played_att & ~on,
                off_def=team_def & played_def & ~on)


# ===================================================================================== one season of step 3
def lineups(players, n_players: int) -> tuple[np.ndarray, np.ndarray]:
    """(lineup_o, lineup_d): each row's five offensive and five defensive player columns, read off Z.

    The pricing of the defensive target drops the design's `parts`, so the lineups are read from the matrix: every
    row of Z holds ten ones, the offence's five in the first `n_players` columns and the defence's five after them.
    """
    z = players.tocsr().copy()
    z.sort_indices()
    if not (np.diff(z.indptr) == 10).all():
        raise ValueError("a row of Z does not hold ten players")
    columns = z.indices.reshape(-1, 10)
    if columns[:, :5].max() >= n_players or columns[:, 5:].min() < n_players:
        raise ValueError("a row of Z does not hold five offensive and five defensive players")
    return columns[:, :5].astype(np.int64), (columns[:, 5:] - n_players).astype(np.int64)


def row_keys(design, lineup_o: np.ndarray, lineup_d: np.ndarray, teams: pd.Series) -> dict:
    """`team_codes` of one season's design: each row's teams from the box scores (`teams` maps (game id, player id)
    to a team id; commands/site.py reads it from data/raw/box)."""
    player_ids = design.spec.players["player_id"].to_numpy()
    game_ids = (design.games.set_index("game_idx").game_id.reindex(design.rows.game_idx.to_numpy())
                .to_numpy().astype(str))
    season = np.full(len(game_ids), int(design.spec.seasons[0]), dtype=np.int64)
    return team_codes(dict(team_att=side_team(teams, game_ids, player_ids[lineup_o]),
                           team_def=side_team(teams, game_ids, player_ids[lineup_d]),
                           season=season, game_ids=game_ids))


@dataclass
class GamesPartFit:
    """One season's step-3 fit on one target, with what the split needs: the rows and the matrix that was solved."""
    player_ids: np.ndarray
    players: sp.csr_matrix       # Z
    unpenalised: np.ndarray      # [C | P_O | P_D]
    y: np.ndarray
    w: np.ndarray
    matrix: np.ndarray           # M = G + Lambda, with the jitter if the solve needed one
    theta: np.ndarray            # [deviation (offence, defence) | context | s_O, s_D]
    retried: bool                # the solve added a jitter (rapm.solved_ridge)

    @property
    def n_players(self) -> int:
        return len(self.player_ids)


def refit(design, prior_offense: dict, prior_defense: dict, penalty: float) -> GamesPartFit:
    """Step 3's fit on one season's design (`prior_rapm.fit`'s system, through the same code), keeping the rows and
    the matrix.  The priors are {player_id: prior}; any multiple of the booster's prior, plus a constant, gives the
    same deviations, because the prior scale is free and the season intercept absorbs the constant."""
    system = prior_rapm.ridge_system(design, prior_offense, prior_defense, penalty)
    theta, matrix = rapm.solved_ridge(system.gram, system.rhs, system.penalties)
    retried = not np.array_equal(np.diag(matrix), np.diag(system.gram) + system.penalties)
    return GamesPartFit(player_ids=design.spec.players["player_id"].to_numpy(), players=system.players,
                        unpenalised=system.unpenalised, y=np.asarray(design.y, dtype=float),
                        w=np.asarray(design.w, dtype=float), matrix=matrix, theta=theta, retried=retried)


def row_weights(fit: GamesPartFit, solved_column: np.ndarray) -> np.ndarray:
    """h = W B M^-1 e_j for one column j of the system, given M^-1 e_j: how much each row's outcome counts for
    that coefficient."""
    n = fit.n_players
    return fit.w * (fit.players @ solved_column[:2 * n] + fit.unpenalised @ solved_column[2 * n:])


def split_games_part(fit: GamesPartFit, keys: dict, side: str) -> tuple[pd.DataFrame, dict]:
    """Every player's games' part on one side, split over his rows: (frame, the largest misses).

    `side` is "off" (his offensive column) or "def" (his defensive column, sign flipped so positive is good, as the
    published tables have it).  The frame has player_id, u, on_court, off_court and team_sos, and
    u = on_court + off_court + team_sos for every player; team_sos includes the leftover of a jittered solve.
    The misses: `sum_hy`, the largest |sum of h y - the coefficient|, `leftover`, the largest |coefficient - the
    three groups' sum before the leftover is added to team SOS|, and whether the solve was jittered.
    """
    n, n_rows = fit.n_players, len(fit.w)
    first, sign = (0, 1.0) if side == "off" else (n, -1.0)
    unit = np.zeros((fit.matrix.shape[0], n))
    unit[first + np.arange(n), np.arange(n)] = 1.0
    solved = np.linalg.solve(fit.matrix, unit)
    z = fit.y - fit.unpenalised @ fit.theta[2 * n:]
    by_column = fit.players.tocsc()
    records, sum_hy, leftover = [], 0.0, 0.0
    for j in range(n):
        rows_off = by_column.indices[by_column.indptr[j]:by_column.indptr[j + 1]]
        rows_def = by_column.indices[by_column.indptr[n + j]:by_column.indptr[n + j + 1]]
        h = row_weights(fit, solved[:, j])
        hz = h * z
        groups = row_groups(rows_off, rows_def, keys, n_rows)
        off = groups["without"] | groups["missed"]
        on_court, off_court = hz[groups["on"]].sum(), hz[off].sum()
        rest = hz[~(groups["on"] | off)].sum()
        coefficient = fit.theta[first + j]
        sum_hy = max(sum_hy, abs(float(h @ fit.y) - coefficient))
        leftover = max(leftover, abs(coefficient - (on_court + off_court + rest)))
        records.append((fit.player_ids[j], sign * coefficient, sign * on_court, sign * off_court,
                        sign * (coefficient - on_court - off_court)))
    frame = pd.DataFrame(records, columns=["player_id", "u", "on_court", "off_court", "team_sos"])
    return frame, dict(sum_hy=sum_hy, leftover=leftover, retried=fit.retried)
