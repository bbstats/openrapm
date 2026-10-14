"""Step 1 of 5: plain RAPM, and the label the box-score prior learns from.

RAPM (regularised adjusted plus-minus) asks of every stint: given the ten players on the court, how many points
per 100 possessions did the offence score?  Five players share every possession, so one stint cannot say who
was good; thousands of stints with different lineups can, once a ridge penalty keeps the answer from chasing
noise.

Words used below
    stint       a stretch of play with the same ten players on the court
    row         one stint seen from one side: five players on offence, five on defence
    target      what the ridge explains for each row, y = 100 x adjusted points / possessions; the row's weight
                w is its possessions.  One adjustment per side ships (settings.Targets):
                  offence, "points_with_expected_free_throws": points - made free throws + expected free
                      throws, each free throw counted at the shooter's padded free-throw %, that game's own
                      attempts left out
                  defence, "points_with_expected_opponent_threes": the same free-throw adjustment, and three
                      quarters of the shooters' three-point luck taken off: points - 3 x 0.75 x (threes made -
                      threes expected), threes expected = attempts x the shooter's three-point % from the other
                      half of the season's games, padded toward the league by 450 attempts; then one number per
                      season rescales the target so the season's total equals the points actually scored
                      (data/shots.align)
    design      one column per player per side (offence, defence) plus the context columns
    context     per-season intercepts, home court, playoffs, playoff home court, garbage time, score margin
                (clipped at +-25), margin x share of the game left
    raw impact  the coefficient an unpenalised fit would give a player
    penalty     the ridge's lambda.  Seen alone, a player with n possessions keeps about n / (n + lambda) of
                his raw impact
    rated season    the season whose ratings are being built
    unseen seasons  the seasons a label may not use: the rated one, and in the test setting its two
                neighbours as well
    label       a player's RAPM from one ridge fit over every season except the unseen ones: ONE coefficient
                per player per side for all those seasons together, his career level, assumed constant from
                season to season.  Step 2's prior learns to predict it
    team-season flags   optional (`LabelSettings.team_offense_penalty`, `team_defense_penalty`; off when both are
                0, the shipped setting): two more context blocks per season, one column per team that season.
                The attacking flag is 1 on the rows that team is on offence, the defending flag 1 on the rows it is
                on defence.  Each block has its own penalty; the flags' values are not part of any label.  Inside
                one season a flag is the same as "every player on that team +x", so only players who change teams
                between seasons tell the two apart (notes/DECISIONS.md, experiment 60)

The fit, for one set of seasons:

    minimise   sum over rows  w (y - B theta)^2  +  lambda_O |theta_O|^2  +  lambda_D |theta_D|^2
    solution   (G + Lambda) theta = b,     G = B' W B,   b = B' W y

    B          the design matrix, one line per row: [Z_O | Z_D | C]
    Z_O, Z_D   the player columns: Z_O[row, j] = 1 when player j is one of the five on offence, Z_D the same
               for the five on defence; Z = [Z_O | Z_D]
    C          the context columns
    y, w       each row's target and weight (its possessions); W the diagonal matrix of the weights
    theta      the coefficients: theta_O on Z_O, theta_D on Z_D, then gamma on C
    lambda_O, lambda_D   the offence and defence penalties (40,000 each); the context is not penalised
    Lambda     the diagonal matrix of per-column penalties: lambda_O, lambda_D, 0 (`block_penalty`)
    G, b       the Gram matrix and right-hand side of the weighted least-squares normal equations

G and b are sums over rows, so they add up over seasons.  `SeasonSums` stores each season's share once; the fit
that leaves the rated season out is then one solve on the sums of the other seasons, not thirty refits.  Each
season keeps its own context columns, so leaving a season out drops its rows and its context together.

Example: with lambda = 40,000, a player with 54,000 career possessions keeps 57% of his raw impact
(54,000 / 94,000) and a player with 2,600 keeps 6%.

The defensive label is "un-shrunk" (`unshrink`) before the prior learns from it, so long and short careers sit on
one scale; the offensive label is not.  Un-shrinking both sides helped the year-over-year test (paired difference
in team-game mean squared error -0.84, z -3.77) but made the order within each side worse (each side rescaled:
+1.79, z +7.75, better in 8 of 56) and cut the consensus agreement from 0.771 to 0.680.  Un-shrinking defence alone
passed every check (-0.42, z -5.24, better in 43 of 56; notes/DECISIONS.md, experiment 22).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

__all__ = ["design_blocks", "NormalEquations", "normal_equations", "player_possessions", "block_penalty",
           "solved_ridge", "solve_ridge", "team_flags", "SeasonSums", "unshrink", "career_labels"]


def design_blocks(design):
    """Split a season's design matrix into (Z as CSR sparse, C as a dense array).

    Z = [Z_O | Z_D] has 2 x n columns, n = the season's players (`design.spec.n_players`); C is the next
    `len(design.spec.context_names)` columns.  The matrix's last column holds each row's game number, used to group
    rows by game, and is not a regressor.
    """
    n_players = design.spec.n_players
    n_context = len(design.spec.context_names)
    players = design.X[:, :2 * n_players].tocsr()
    context = np.asarray(design.X[:, 2 * n_players:2 * n_players + n_context].todense())
    return players, context


class NormalEquations(NamedTuple):
    """G and b of one design, kept in blocks so seasons can be added up:

        G = [[Z'WZ, Z'WC], [C'WZ, C'WC]],   b = [Z'Wy, C'Wy]
    """
    gram_players: np.ndarray           # Z'WZ
    gram_players_context: np.ndarray   # Z'WC
    gram_context: np.ndarray           # C'WC
    rhs_players: np.ndarray            # Z'Wy
    rhs_context: np.ndarray            # C'Wy


def normal_equations(players, context, target, weights) -> NormalEquations:
    """The weighted least-squares sums of one design: G = B'WB, b = B'Wy, block by block.

    The arithmetic, and the order of every sum, is the one every published rating was built with, so the bits
    match: change the order and the published ratings move in their last digits.
    """
    weighted = players.T.multiply(weights)
    weighted_context = (context * weights[:, None]).T
    return NormalEquations(gram_players=np.asarray((weighted @ players).todense()),
                           gram_players_context=np.asarray(weighted @ context),
                           gram_context=weighted_context @ context,
                           rhs_players=np.asarray(weighted @ target).ravel(),
                           rhs_context=weighted_context @ target)


def player_possessions(players, possessions, n_players: int) -> np.ndarray:
    """Each player's possessions in the design: the possessions of the rows he is on offence in.

    Only one side is counted.  His defensive possessions are within a few of the offensive ones, and counting
    both would credit every player with about twice the trips he was on the court for.
    """
    return np.asarray(players[:, :n_players].multiply(possessions[:, None]).sum(axis=0)).ravel()


def block_penalty(sizes, penalties) -> np.ndarray:
    """One penalty per column from one penalty per block.

    Example: block sizes (3, 3, 2) with penalties (40000, 40000, 0) give three columns at 40,000 for offence,
    three at 40,000 for defence and two context columns at 0.
    """
    return np.concatenate([np.full(size, penalty) for size, penalty in zip(sizes, penalties)])


def solved_ridge(gram, rhs, penalty) -> tuple[np.ndarray, np.ndarray]:
    """(theta, the matrix that was solved): theta solves (G + diag(penalty)) theta = b, one penalty per column.

    An unpenalised context column can be all zeros (the 2020 bubble's playoff home court) and then the plain
    solve raises.  The retry adds a jitter of 1e-9 x the mean diagonal to the UNPENALISED diagonal only, which
    leaves the player block untouched; least squares is the last resort.  The matrix returned has the jitter in
    it, so a caller who needs theta's rows as sums over the data rows (decomposition.py) solves the system that
    was solved, not the one that was meant.
    """
    penalty = np.asarray(penalty, dtype=float)
    matrix = gram + np.diag(penalty)
    try:
        return np.linalg.solve(matrix, rhs), matrix
    except np.linalg.LinAlgError:
        jitter = 1e-9 * max(float(np.mean(np.abs(np.diag(matrix)))), 1.0)
        matrix[np.diag_indices_from(matrix)] += np.where(penalty == 0.0, jitter, 0.0)
        try:
            return np.linalg.solve(matrix, rhs), matrix
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(matrix, rhs, rcond=None)[0], matrix


def solve_ridge(gram, rhs, penalty) -> np.ndarray:
    """Solve (G + diag(penalty)) theta = b, one penalty per column: `solved_ridge`'s theta."""
    return solved_ridge(gram, rhs, penalty)[0]


def team_flags(team_off, team_def) -> tuple[np.ndarray, np.ndarray]:
    """(flags, teams): one season's team-season flags as a dense (rows x 2 n_teams) array, [attacking | defending],
    and the season's team ids in column order (ascending).

    Example: in a season of teams 10, 20 and 30, a row in which 20 attacks 30 gets [0, 1, 0, 0, 0, 1].
    """
    team_off, team_def = np.asarray(team_off, dtype=np.int64), np.asarray(team_def, dtype=np.int64)
    if team_off.shape != team_def.shape:
        raise ValueError("team_off and team_def must name one team per row each")
    teams = np.unique(np.concatenate([team_off, team_def]))
    rows, n_teams = np.arange(len(team_off)), len(teams)
    flags = np.zeros((len(team_off), 2 * n_teams))
    flags[rows, np.searchsorted(teams, team_off)] = 1.0
    flags[rows, n_teams + np.searchsorted(teams, team_def)] = 1.0
    return flags, teams


class SeasonSums:
    """Each season's normal equations, stored once; any set of seasons can then be left out and solved.

    Columns: [offence of every player seen | defence of every player seen | each season's context].  A player's
    column is fixed the first time a season brings him, so seasons must be added in ascending order for the
    columns (and so the bits of every solve) to match the published ratings.  A season added with its rows' teams
    carries its team-season flags at the end of its context: [plain context | attacking flags | defending flags].
    """

    def __init__(self, min_possessions: float = 100.0):
        self.min_possessions = float(min_possessions)
        self.player_ids = np.empty(0, dtype=np.int64)
        self._column: dict = {}
        self._seasons: dict = {}

    @property
    def seasons(self) -> list:
        """The seasons added so far, ascending."""
        return sorted(self._seasons)

    def _columns_for(self, ids) -> np.ndarray:
        """Each player's column, giving a new column to a player seen for the first time."""
        new = [int(pid) for pid in ids if int(pid) not in self._column]
        if new:
            start = len(self._column)
            for offset, pid in enumerate(new):
                self._column[pid] = start + offset
            self.player_ids = np.concatenate([self.player_ids, np.asarray(new, dtype=np.int64)])
        return np.array([self._column[int(pid)] for pid in ids], dtype=np.int64)

    def add(self, season: int, design, teams=None) -> "SeasonSums":
        """Store one season's normal equations and each player's possessions in it.

        `teams`, optional: (team_off, team_def), each row's attacking and defending team id.  When given, the
        season's context gains its team-season flags (`team_flags`)."""
        players, context = design_blocks(design)
        team_ids = np.empty(0, dtype=np.int64)
        if teams is not None:
            flags, team_ids = team_flags(*teams)
            if len(flags) != len(context):
                raise ValueError(f"{season}: {len(flags)} rows of teams for {len(context)} design rows")
            context = np.column_stack([context, flags])
        self._seasons[int(season)] = dict(
            columns=self._columns_for(design.spec.players["player_id"].to_numpy()),
            n_context=context.shape[1],
            team_ids=team_ids,
            sums=normal_equations(players, context, design.y, design.w),
            possessions=player_possessions(players, design.rows["poss"].to_numpy(), design.spec.n_players))
        return self

    @property
    def has_team_flags(self) -> bool:
        """Whether any season was added with its team-season flags."""
        return any(len(block["team_ids"]) for block in self._seasons.values())

    def player_columns(self, season: int) -> np.ndarray:
        """The column of each of a season's players, in the order of that season's design.

        `openrapm site --decomposition` reads this and `context_size` to place each season inside `assemble`'s G.
        """
        return self._seasons[int(season)]["columns"]

    def context_size(self, season: int) -> int:
        """How many context columns a season brought."""
        return self._seasons[int(season)]["n_context"]

    def _context_starts(self, kept: list) -> tuple[dict, int]:
        """({season: its first context column}, total columns): the context blocks follow the 2 x players."""
        starts, total = {}, 2 * len(self._column)
        for season in kept:
            starts[season] = total
            total += self._seasons[season]["n_context"]
        return starts, total

    def _kept(self, unseen) -> list:
        """The seasons not in `unseen`, ascending."""
        dropped = {int(season) for season in unseen}
        return [season for season in self.seasons if season not in dropped]

    def assemble(self, unseen=()):
        """G, b, each player's possessions and the number of context columns, over the seasons not in `unseen`.

        The context columns follow the 2 x players in ascending season order.
        """
        kept = self._kept(unseen)
        n_players = len(self._column)
        starts, total = self._context_starts(kept)
        gram = np.zeros((total, total))
        rhs = np.zeros(total)
        possessions = np.zeros(n_players)
        for season in kept:
            block = self._seasons[season]
            sums, columns = block["sums"], block["columns"]
            both_sides = np.concatenate([columns, columns + n_players])
            first = starts[season]
            stop = first + block["n_context"]
            gram[np.ix_(both_sides, both_sides)] += sums.gram_players
            gram[np.ix_(both_sides, np.arange(first, stop))] += sums.gram_players_context
            gram[np.ix_(np.arange(first, stop), both_sides)] += sums.gram_players_context.T
            gram[first:stop, first:stop] += sums.gram_context
            rhs[both_sides] += sums.rhs_players
            rhs[first:stop] += sums.rhs_context
            possessions[columns] += block["possessions"]
        return gram, rhs, possessions, total - 2 * n_players

    def penalty(self, unseen=(), penalties=(40000.0, 40000.0, 0.0), team_penalties=(0.0, 0.0)) -> np.ndarray:
        """One penalty per column of `assemble(unseen)`.

        penalties = (offence, defence, context) for the players and the plain context; team_penalties = (attacking,
        defending) for the team-season flags.  Flags need both above 0: unpenalised, a season's attacking flags (and
        its defending flags) add up to its intercept and the solve is singular.
        """
        kept = self._kept(unseen)
        n_players = len(self._column)
        n_teams = [len(self._seasons[season]["team_ids"]) for season in kept]
        if not any(n_teams):
            n_context = sum(self._seasons[season]["n_context"] for season in kept)
            return block_penalty((n_players, n_players, n_context), [float(value) for value in penalties])
        if min(team_penalties) <= 0.0:
            raise ValueError(f"team-season flags need both team penalties above 0, got {tuple(team_penalties)}")
        parts = [np.full(n_players, float(penalties[0])), np.full(n_players, float(penalties[1]))]
        for season, teams in zip(kept, n_teams):
            parts += [np.full(self._seasons[season]["n_context"] - 2 * teams, float(penalties[2])),
                      np.full(teams, float(team_penalties[0])), np.full(teams, float(team_penalties[1]))]
        return np.concatenate(parts)

    def flag_columns(self, unseen=()) -> pd.DataFrame:
        """Where each team-season's flags sit in `assemble(unseen)`: season, team_id, attacking, defending (column
        numbers).  Empty when no season carries flags."""
        starts, _ = self._context_starts(self._kept(unseen))
        frames = []
        for season, first in starts.items():
            block = self._seasons[season]
            teams = block["team_ids"]
            attacking = first + block["n_context"] - 2 * len(teams) + np.arange(len(teams))
            frames.append(pd.DataFrame({"season": season, "team_id": teams, "attacking": attacking,
                                        "defending": attacking + len(teams)}))
        columns = {"season": "int64", "team_id": "int64", "attacking": "int64", "defending": "int64"}
        return pd.concat(frames, ignore_index=True).astype(columns) if frames else pd.DataFrame(columns=columns)

    def solve(self, unseen=(), penalties=(40000.0, 40000.0, 0.0), team_penalties=(0.0, 0.0)):
        """(theta, each player's possessions) over the seasons not in `unseen`; theta is [offence | defence | each
        kept season's context], the layout of `assemble`."""
        gram, rhs, possessions, _ = self.assemble(unseen)
        return solve_ridge(gram, rhs, self.penalty(unseen, penalties, team_penalties)), possessions

    def ratings(self, unseen=(), penalties=(40000.0, 40000.0, 0.0), team_penalties=(0.0, 0.0)) -> pd.DataFrame:
        """One row per player with at least `min_possessions` in the seasons used (every season not in `unseen`).

        penalties = (offence, defence, context); team_penalties = (attacking, defending), read only when a season
        carries team-season flags.  Columns: player_id, offense, defense (raw sign: defence is points allowed, lower
        is better), possessions.
        """
        coefficients, possessions = self.solve(unseen, penalties, team_penalties)
        n_players = possessions.size
        keep = possessions >= self.min_possessions
        return pd.DataFrame({"player_id": self.player_ids[keep],
                             "offense": coefficients[:n_players][keep],
                             "defense": coefficients[n_players:2 * n_players][keep],
                             "possessions": possessions[keep]})


def unshrink(ratings: pd.DataFrame, column: str, penalty: float, full_trust_possessions: float = 4444.0,
             tier_edges=(0.0, 500.0, 1500.0, 4444.0)) -> pd.DataFrame:
    """Put every player's label on one scale, and pull a player with few possessions toward his tier's level.

        label = beta / max(s, s_cap)  +  (1 - s / max(s, s_cap)) x m(tier)

        beta    the player's ridge coefficient in `column`
        n       his possessions
        lambda  the ridge penalty the coefficient was fitted with
        s       = n / (n + lambda), the share of his raw impact the ridge kept
        s_cap   = N / (N + lambda), N = full_trust_possessions
        m(tier) = mean beta / mean s over the players in his possession tier: the tier's level, un-shrunk

    Above N possessions the label is the raw-scale coefficient beta / s.  Below it, dividing by s would
    multiply noise by hundreds, so the division stops at s_cap and the rest comes from the player's tier:
    possessions in [0, 500), [500, 1500), [1500, 4444) or 4444 and more.

    Example: lambda = 40,000, so s_cap = 4,444 / 44,444 = 0.10.  A 1,000-possession player (s = 0.024) with
    beta = 0.05 in a tier whose level is -3.0 gets 0.05 / 0.10 + (1 - 0.24) x -3.0 = -1.77.
    """
    out = ratings.copy()
    possessions = out.possessions.to_numpy(float)
    tier = np.digitize(possessions, list(tier_edges[1:]))
    beta = out[column].to_numpy(float)
    kept = possessions / (possessions + penalty)
    kept_capped = np.maximum(kept, full_trust_possessions / (full_trust_possessions + penalty))
    tier_level = np.zeros(len(tier_edges))
    for tier_index in range(len(tier_level)):
        members = tier == tier_index
        if members.any():
            tier_level[tier_index] = beta[members].mean() / kept[members].mean()
    out[column] = beta / kept_capped + (1.0 - kept / kept_capped) * tier_level[tier]
    return out


def career_labels(sums: dict, unseen, label_settings, targets) -> dict:
    """{"O": offence labels, "D": defence labels}: each side's RAPM over every season not in `unseen`.

    `sums` maps a target name to its `SeasonSums`; `label_settings` is `settings.LabelSettings` and `targets`
    `settings.Targets`.  The offence label is the `offense` column of the fit on the offence target, the defence
    label the `defense` column of the fit on the defence target.  A column named in
    `label_settings.unshrink_columns` (shipped: defence only; notes/DECISIONS.md, experiment 22) is un-shrunk.

    With team-season flags on (`label_settings.team_offense_penalty` and `team_defense_penalty` above 0), every
    `SeasonSums` must have been built with each row's teams; with them off, without.

    Example: rating 2026 in the product setting, unseen = [2026]; each table holds player_id, offense, defense
    and possessions for every player with 100+ possessions in 1997-2025.
    """
    penalties = (label_settings.offense_penalty, label_settings.defense_penalty, label_settings.context_penalty)
    team_penalties = (label_settings.team_offense_penalty, label_settings.team_defense_penalty)
    flags_on = label_settings.team_flags
    out = {}
    for side, column, target, penalty in (("O", "offense", targets.offence, label_settings.offense_penalty),
                                          ("D", "defense", targets.defence, label_settings.defense_penalty)):
        if sums[target].has_team_flags != flags_on:
            raise ValueError(f"team-season flags are {'on' if flags_on else 'off'} in the label settings but the "
                             f"sums on {target} were built {'without' if flags_on else 'with'} each row's teams")
        table = sums[target].ratings(unseen, penalties, team_penalties)
        if column in label_settings.unshrink_columns:
            table = unshrink(table, column, penalty, label_settings.full_trust_possessions,
                             label_settings.tier_edges)
        out[side] = table
    return out
