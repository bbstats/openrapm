"""The design: the stints of one or more seasons turned into the rows, columns, target and weight every fit reads.

Step 1's RAPM, step 3's prior-informed RAPM and the season panel's on-court numbers all read the same design.  It is
built in two stages, so that a season is prepared once and reused by every set of seasons it belongs to:

    piece_from_frames   one season and phase (RS regular season, PO playoffs) of stints and box scores -> a piece:
                        its games in order, its player-seasons, every stint's lineup, flags and counters
    assemble            the pieces of one or more seasons -> the design

The data layer keeps pieces on disk (data/designs.py) and the simulated tests build them from simulated stints
(`pieces_from_frames`); both then call `assemble`, so there is one design builder.

Words used below
    stint          a stretch of play with the same ten players on the court (data/stints.py)
    row            one stint seen from one side: five players on offence, five on defence.  A stint gives two rows;
                   every home-offence row of the design comes first, then every away-offence row
    player-season  one player in one season, the unit the possession and box-score tables are kept in.  Its key is
                   season x 100,000,000 + player id (`player_season_key`)
    player column  the unit a coefficient belongs to: one per player over all the design's seasons (config
                   player_unit "player", the shipped setting), or one per player-season ("player_season").  In a
                   design of one season, which is what the five steps fit, the two give the same matrix
    context        the columns every row carries besides its players: home court, one intercept per season,
                   playoffs, playoff home court, garbage time, score margin (clipped at +-25), margin x share of the
                   game left
    target         a TARGETS entry: the row's response is scale x numerator / denominator and its weight is the
                   denominator; a row whose denominator is 0 is dropped
    half           each game's half, A or B: regular-season games alternate A, B, A, ... in date order within a
                   season, playoff games within a series (data/stints.assign_halves).  A shooter's attempts in a
                   half-A game are priced from his makes in the half-B games, regular season and playoffs
                   together, and the other way round (data/shots.py), so a shot never prices itself

The design, one line per row:

    X = [Z_O | Z_D | C | game]
    y = scale x numerator / denominator
    w = denominator x (gt_weight in garbage time, 1 otherwise)

    Z_O, Z_D   Z_O[row, j] = 1 when player column j is one of the five on offence; Z_D the same for defence
    C          the context columns
    game       the row's game number (0, 1, 2, ... by season, regular season before playoffs, then date); it groups
               rows by game and is not a regressor

Example: a stint in which the home team scores 8 points on 6 possessions and the visitors 5 on 6 gives, on the "pts"
target, a home-offence row with y = 100 x 8 / 6 = 133.3, w = 6, home = +1, and an away-offence row with
y = 100 x 5 / 6 = 83.3, w = 6, home = -1.  With config gt_weight = 1.0 a garbage-time row weighs the same.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

# A season is its regular season and its playoffs, and they are ONE ENTITY.  Owner, 2026-09-10:
# "the playoff delta is gone; playoff games just join the fit like any other games", and then
# "RS + playoffs should be considered a single entity in our new version".  So there is one constant
# and no second one for scoring: every fit trains on both phases and the criterion scores both.
#
# The design's `is_po` and `po_home` context columns stay, and are not a contradiction.  They are level
# controls on the ENVIRONMENT, the same kind of thing as the per-season intercepts `int_<s>` -- which
# nobody would call treating each season as a separate entity.  Dropping them would make the playoffs'
# lower scoring level something the fit has to explain with the players on the court, and the players
# on the court in the playoffs are disproportionately the good ones.  A level control is what keeps
# them one entity rather than an advantage for whoever got there.
SEASON_PHASES = ("RS", "PO")

FEATURES = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss",
            "orb", "drb", "ast", "tov", "stl", "blk", "pf"]
HOME_SLOTS = [f"h{place}" for place in range(1, 6)]
AWAY_SLOTS = [f"a{place}" for place in range(1, 6)]
STINT_COLUMNS = ["game_id", "season", "phase", "series_id", "period", *HOME_SLOTS, *AWAY_SLOTS,
                 "poss_h", "poss_a", "pts_h", "pts_a", "margin_h", "frac_rem", "is_gt", "neutral"]
SEASON_KEY = 100_000_000          # a player-season key is season x SEASON_KEY + player id
# config player_unit: one player column per player over the design's seasons, or one per player-season (the words in
# the module docstring).  Before 2026-10-11 these were "window" and "season"; data/config.py still reads those.
PLAYER_UNITS = ("player", "player_season")
# A piece's per-game tables (game_poss, game_box) number its player-seasons by position in the piece's keys_u, in a
# column named `psx_idx`: the name the sixty pieces stored in data/cache/pieces carry, kept so they are read without a
# rebuild and a piece written now is the same bytes.  `assemble` renumbers them across the design as `player_season`.
PIECE_PLAYER_SEASON = "psx_idx"

# The stints' counter columns.  They are defined here, beside the other stint columns, because the design
# reads them; the stint parser (data/stint_parser.py) imports them from here and fills them.
#
# Per-possession counters, carried alongside `points` and summed per side into the stints.
#
# `att` is an ATTEMPT EVENT, and it is the unit the rebound chain is built on: every field-goal
# attempt, plus every free-throw trip that is not an and-1.  A shot that draws a shooting foul
# records no FGA at all, so counting only FGA misses 9% of possessions -- see FINDINGS.md section 14,
# where this definition reconciles the possession count to 0.25% across 1998, 2005 and 2024 while
# the box-score convention (FGA only) is out by 9%.
#
# Two identities hold exactly, per possession, and are tested in tests/test_stints.py:
#     points - pts_tech == 2*(fgm - fg3m) + 3*fg3m + ftm
#     reb_cont - cont_dead + att_retained == att - 1     (whenever the possession reaches an attempt)
# There are exactly three ways a possession gets an attempt after its first, and the identity names
# all of them:
#   reb_cont      it rebounded its own miss (including `oreb_unattr`, where the feed recorded no
#                 rebound row -- a block or a tip -- but play plainly continued)
#   att_retained  a flagrant, clear-path or away-from-play foul handed the ball back
#   cont_dead     subtracts a continuation that led to NO further attempt: rebound your own miss and
#                 then turn it over, or have the period expire.  Only the last continuation can be
#                 dead, since between any two rebounds there is necessarily a shot.
POSS_COUNTERS = (
    "pts_tech",                                             # technical/deferred FT points
    "fga", "fgm", "fg3a", "fg3m", "fta", "ftm",             # box-score shooting
    "fta_tech", "ftm_tech", "fta1", "tov",
    "att", "and1", "trip_shoot", "trip_ns", "fga_fouled", "att_retained",
    "reb_chance", "reb_cont", "reb_drop", "cont_dead",      # the rebound chain
    "oreb_p", "oreb_t", "oreb_unattr",
    "fga_rim", "fgm_rim", "fga_mid", "fgm_mid", "fga_thr", "fgm_thr",
    "att1_rim", "att1_mid", "att1_thr", "att1_ft",          # the possession's FIRST attempt only
    "xftm", "xftm_tech",                                    # EXPECTED free-throw makes (data/stint_parser.py)
    "pts1", "chance1",                                      # the first attempt's realised points (incl. its
                                                            # free throws) and whether it produced a rebound chance
    "fg3a_late", "fg3m_late",                               # threes taken (made) in a period's last two seconds,
                                                            # heaves mostly (stint_parser.LATE_THREE_SECONDS)
)

# Per-SHOOTER counters, kept by lineup slot so the shooter's own padded rate can price them when the
# design is built (data/shots.py).  The rate depends on which seasons are pooled and on which half of
# them is the other half, so it cannot be baked into the stints; the counts can.
#   fg2a, fg3a, fta   his two-point, three-point and (non-technical) free-throw attempts
#   xl2, xl3          the sum over those attempts of the LEAGUE make probability at that distance this
#                     season (data/shot_curves.py); xv2, xv3 the sum of l(1-l), for the logit variant
#   *1                the same restricted to the possession's first attempt (fg2a1/xl2_1/fg3a1/xl3_1),
#                     the free throws that belong to it (fta1), and its trip's last free throw (ftlast1),
#                     the one whose miss creates a rebound chance
# Slots are positions in the stint's SORTED lineup (h1..h5 / a1..a5); `x` is a shooter outside the lineup the
# possession is credited to, priced at the league rate.  Since a possession that ends inside its free-throw
# trip is credited to the players on the court at the foul (data/stint_parser.py), the shooter of a trip is
# in it; before 2026-10-22 a shooter subbed out after his free throws went to `x`.
SHOOTER_COUNTERS = ("fg2a", "xl2", "xv2", "fg3a", "xl3", "xv3", "fta",
                    "fg2a1", "xl2_1", "fg3a1", "xl3_1", "fta1", "ftlast1",
                    "fg3a_late")                    # his threes in a period's last two seconds, part of his fg3a
SLOTS = ("1", "2", "3", "4", "5", "x")
SLOT_COUNTERS = tuple(f"{counter}_s{slot}" for counter in SHOOTER_COUNTERS for slot in SLOTS)

# What the model is asked to explain.  `num` is a weighted sum of stint counters, `den` the exposure
# that is both the rate's denominator and the row's weight.
#
# The denominator is NOT always possessions: an eFG% row carries information in proportion to its
# field-goal attempts, not its possessions, so weighting it by possessions would misstate how noisy it is.
# The ridge penalties are calibrated to the points-per-100 scale and do NOT transfer to a rate target.
# Only the two points targets are fitted by the shipped steps; the four rates stay because every stored
# piece keeps the stint columns of every target (`target_stint_columns`), and dropping them would make a
# freshly built piece differ from the sixty already in data/cache/pieces.
TARGETS = {
    "pts":  dict(num={"pts": 1.0}, den="poss", scale=100.0),                    # points per 100
    # Luck-adjusted points: made free throws replaced by expected ones.  A fixed linear combination of
    # counters, with no fitted quantity in it, so it is computed here rather than stored.
    "points_with_expected_free_throws": dict(num={"pts": 1.0, "ftm": -1.0, "ftm_tech": -1.0, "xftm": 1.0,
                                                  "xftm_tech": 1.0}, den="poss", scale=100.0),
    "efg":  dict(num={"fgm": 1.0, "fg3m": 0.5}, den="fga", scale=100.0),        # effective FG%
    "tov":  dict(num={"tov": 1.0}, den="poss", scale=100.0),                    # turnovers per 100
    "oreb": dict(num={"reb_cont": 1.0}, den="reb_chance", scale=100.0),         # offensive rebound %
    "ftr":  dict(num={"fta": 1.0}, den="att", scale=100.0),                     # FTs per attempt
}


def player_season_key(player_id, season) -> np.ndarray:
    """The integer key of a player-season: season x 100,000,000 + player id.

    Example: player 2544 in 2026 has key 202,600,002,544.  Sorting the keys sorts by season, then player."""
    return np.asarray(season, dtype=np.int64) * SEASON_KEY + np.asarray(player_id, dtype=np.int64)


def target_stint_columns() -> list:
    """The stint columns any TARGETS response reads, both sides, sorted: what a piece keeps for the response."""
    names = set()
    for target in TARGETS.values():
        names.update(target["num"])
        names.add(target["den"])
    return sorted(f"{name}_{side}" for name in names for side in ("h", "a"))


@dataclass
class DesignSpec:
    """Column layout of the design matrix and the player behind each player column.

    Two numberings, because the coefficients and the padded rates want different units:

    player column   what a coefficient belongs to: one per player over the design's seasons (player_unit
                    "player") or one per player-season (player_unit "player_season").  There are `n_players` of
                    them, numbered 0, 1, 2, ... (the `player_column` of a table), so Z = [Z_O | Z_D] has
                    2 x n_players columns: player column j is Z column j on offence and n_players + j on defence.
    player-season   one player in one season, always.  There are `n_player_seasons` of them, numbered 0, 1, 2, ...
                    in key order (the `player_season` of a table).  `game_box` and `game_poss` are kept per
                    player-season, so the padding constants and the league means stay per season even when a
                    player column spans three seasons.  `player_of_player_season` maps one numbering to the other.

    Example: a design of 2025 and 2026 in which 400 players play both seasons and 100 only one has 900
    player-seasons; under player_unit "player" it has n_players = 500 and Z has 1,000 columns, under
    "player_season" n_players = 900 and Z has 1,800.
    """
    n_players: int                    # the number of player columns
    seasons: list                     # the design's seasons, ascending
    season_of_player: np.ndarray      # (n_players,) each player column's index into `seasons`; under player_unit
                                      # "player", the season in which he had the most regular-season possessions
    players: pd.DataFrame             # one row per player column: player_column, player_id (+ season under
                                      # player_unit "player_season")
    context_names: list               # the context columns' names, in the matrix's order
    box_counts: list                  # the box-score counts of game_box (config box_counts)
    # diagnostic groups of Z columns (0 .. 2 x n_players - 1), by regular-season possessions (`design_spec`)
    possession_groups: dict = field(default_factory=dict)
    player_seasons: pd.DataFrame = None            # player_season, player_id, season, player_column
    season_of_player_season: np.ndarray = None     # (n_player_seasons,) each player-season's index into `seasons`
    player_of_player_season: np.ndarray = None     # (n_player_seasons,) each player-season's player column
    player_unit: str = "player_season"             # "player" or "player_season" (config player_unit)

    @property
    def n_player_seasons(self) -> int:
        """The number of player-seasons."""
        return self.n_players if self.player_seasons is None else len(self.player_seasons)

    @property
    def offense_columns(self) -> slice:
        """Z_O's columns in the matrix: 0 .. n_players - 1."""
        return slice(0, self.n_players)

    @property
    def defense_columns(self) -> slice:
        """Z_D's columns in the matrix: n_players .. 2 x n_players - 1."""
        return slice(self.n_players, 2 * self.n_players)

    def context_column(self, name: str) -> int:
        """The matrix column of the context column `name`: the context columns follow Z_O and Z_D."""
        return 2 * self.n_players + self.context_names.index(name)

    @property
    def n_seasons(self) -> int:
        return len(self.seasons)


@dataclass
class Design:
    """One design: the matrix, the response and weight of every row, and the tables the fits read beside them."""
    # the design matrix, or a zero-argument builder for it: a fit that works from `parts` never needs it
    X_src: object
    y: np.ndarray
    w: np.ndarray
    groups: np.ndarray         # "G:<game id>" for a regular-season row, "S:<series id>" for a playoff row
    spec: DesignSpec
    game_box: pd.DataFrame     # game_idx, player_season, phase, <box counts>
    game_poss: pd.DataFrame    # game_idx, player_season, poss_off, poss_def
    rows: pd.DataFrame         # per row: game_idx, season, phase, poss, den, is_home_off, is_gt, stint, half
    games: pd.DataFrame        # game_idx, game_id, season, phase, series_id, half (+ game_date)
    # the offence's per-possession counters (POSS_COUNTERS, SLOT_COUNTERS) summed over the row, the five's ids
    # (pid_s1..pid_s5) and the half, when the stints carry counters; the luck-adjusted targets are priced from them
    counters: pd.DataFrame | None = None
    # the matrix's blocks, so a fit need not slice X back apart: Z (rows x 2 n_players, CSR), F (rows x context, dense),
    # lineup_o / lineup_d (rows x 5 player columns, increasing), game_idx (rows,)
    parts: dict | None = None

    @property
    def X(self) -> sp.csr_matrix:
        """The design matrix, built on first use if it was passed as a builder."""
        if callable(self.X_src):
            self.X_src = self.X_src()
        return self.X_src

    def subset(self, mask: np.ndarray) -> "Design":
        """The rows in `mask`; the game, box-score and possession tables are kept whole."""
        idx = np.flatnonzero(mask)
        return Design(self.X[idx], self.y[idx], self.w[idx], self.groups[idx], self.spec,
                      self.game_box, self.game_poss, self.rows.iloc[idx].reset_index(drop=True), self.games,
                      None if self.counters is None else self.counters.iloc[idx].reset_index(drop=True))

    def with_target(self, y: np.ndarray, w: np.ndarray | None = None) -> "Design":
        """The same design with a different response (and, optionally, weight): how a derived target
        such as expected points is attached without rebuilding anything."""
        return Design(self.X_src, np.asarray(y, dtype=float), self.w if w is None else np.asarray(w, dtype=float),
                      self.groups, self.spec, self.game_box, self.game_poss, self.rows, self.games, self.counters)


# ------------------------------------------------------------------------------------------ one season's piece
def order_games(stints: pd.DataFrame) -> pd.DataFrame:
    """One row per game, numbered 0, 1, 2, ... (game_idx) by season, regular season before playoffs, then date
    and game id, with the game's half.

    The stint files store each game's half (data/stints.assign_halves applies the rule of `game_halves`) and it is
    taken as stored, so the design and the shooter tables can never disagree; stints without it get it here."""
    columns = ["game_id", "season", "phase", "series_id"]
    if "game_date" in stints.columns:
        columns.append("game_date")
    stored = "half" in stints.columns and stints["half"].notna().all()
    if stored:
        columns.append("half")
    games = stints[columns].drop_duplicates("game_id").copy()
    games["_ph"] = (games["phase"] == "PO").astype(int)
    sort_columns = ["season", "_ph"] + (["game_date"] if "game_date" in games.columns else []) + ["game_id"]
    games = games.sort_values(sort_columns).drop(columns="_ph").reset_index(drop=True)
    games["game_idx"] = np.arange(len(games), dtype=np.int64)
    if not stored:
        games["half"] = game_halves(games)
    return games


def game_halves(games: pd.DataFrame) -> np.ndarray:
    """Each game's half: regular-season games alternate A, B, A, ... in order within a season, playoff games
    within a series, so a whole game (both teams) sits in one half and both halves get the same mix."""
    regular = (games["phase"] == "RS").to_numpy()
    half = np.empty(len(games), dtype=object)
    rank = games[regular].groupby("season").cumcount().to_numpy()
    half[regular] = np.where(rank % 2 == 0, "A", "B")
    if (~regular).any():
        rank = games[~regular].groupby(["season", "series_id"]).cumcount().to_numpy()
        half[~regular] = np.where(rank % 2 == 0, "A", "B")
    return half


def piece_from_frames(stints: pd.DataFrame, box: pd.DataFrame, season: int, phase: str, box_counts) -> dict:
    """One season and phase made ready for `assemble`.

    `stints`: one row per stint, STINT_COLUMNS (+ game_date, half and the counter columns when the stint files carry
    them).  `box`: one row per player and game, game_id, player_id, season, phase and the `box_counts` columns.

    Everything is indexed locally: games 0..n_games-1 (`order_games`) and player-seasons by position in `keys_u`,
    the sorted keys of everyone in a lineup or a box score.  The piece holds
        games, n_games, keys_u, n_st           the game table, the player-season keys, the number of stints
        slot_loc                               (stints x 10) each lineup slot's player-season, home five first
        is_po, neutral, is_gt, margin_h, frac_rem, game_idx, groups     one value per stint
        tcols                                  the stint columns of every target (`target_stint_columns`)
        game_poss, game_box                    per (game, player-season): possessions; box-score counts, the
                                               player-season as its position in keys_u (`PIECE_PLAYER_SEASON`)
        psx_poss                               each player-season's regular-season offensive possessions
        have, ccols, counters, pids            the counters present, the counter matrix's columns (have + pts,
                                               poss), per side the (stints x columns) counters and the five ids
    """
    stints = stints.reset_index(drop=True).copy()
    for slot in HOME_SLOTS + AWAY_SLOTS:
        stints[slot] = stints[slot].astype(np.int64)
    games = order_games(stints)
    stints = stints.merge(games[["game_id", "game_idx"]], on="game_id", how="left")
    keys, slot_loc = _player_season_keys(stints, box)
    game_poss = _game_possessions(stints, slot_loc)
    piece = dict(season=int(season), phase=str(phase), games=games, n_games=len(games), keys_u=keys,
                 slot_loc=slot_loc, game_poss=game_poss, game_box=_game_box(box, games, keys, box_counts),
                 psx_poss=_regular_season_possessions(game_poss, len(keys), str(phase)),
                 tcols={name: stints[name].to_numpy().astype(float) for name in target_stint_columns()
                        if name in stints.columns})
    piece.update(_stint_values(stints))
    piece.update(_counter_arrays(stints))
    return piece


def _player_season_keys(stints: pd.DataFrame, box: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(keys, slot_loc): the sorted keys of every player-season in a lineup or the box score, and each lineup
    slot's position among them."""
    season = stints["season"].to_numpy()
    slot_keys = np.stack([player_season_key(stints[slot].to_numpy(), season) for slot in HOME_SLOTS + AWAY_SLOTS],
                         axis=1)
    box_keys = player_season_key(box["player_id"].to_numpy(), box["season"].to_numpy())
    keys = np.unique(np.concatenate([slot_keys.ravel(), box_keys]))
    return keys, np.searchsorted(keys, slot_keys)


def _game_possessions(stints: pd.DataFrame, slot_loc: np.ndarray) -> pd.DataFrame:
    """Per (game, player-season): the possessions his team had the ball (poss_off) and did not (poss_def)."""
    game = stints["game_idx"].to_numpy()
    games, players, offence, defence = [], [], [], []
    for slots, own, other in ((range(0, 5), "poss_h", "poss_a"), (range(5, 10), "poss_a", "poss_h")):
        own_possessions, other_possessions = stints[own].to_numpy(), stints[other].to_numpy()
        for slot in slots:
            games.append(game)
            players.append(slot_loc[:, slot])
            offence.append(own_possessions)
            defence.append(other_possessions)
    table = pd.DataFrame({"game_idx": np.concatenate(games), PIECE_PLAYER_SEASON: np.concatenate(players),
                          "poss_off": np.concatenate(offence), "poss_def": np.concatenate(defence)})
    return table.groupby(["game_idx", PIECE_PLAYER_SEASON], as_index=False)[["poss_off", "poss_def"]].sum()


def _game_box(box: pd.DataFrame, games: pd.DataFrame, keys: np.ndarray, box_counts) -> pd.DataFrame:
    """Per (game, player-season, phase): the box-score counts, for the games that have stints."""
    counts = list(box_counts)
    merged = box.merge(games[["game_id", "game_idx"]], on="game_id", how="inner")
    merged[PIECE_PLAYER_SEASON] = np.searchsorted(keys, player_season_key(merged["player_id"].to_numpy(),
                                                                          merged["season"].to_numpy()))
    table = merged[["game_idx", PIECE_PLAYER_SEASON, "phase"] + counts].copy()
    return table.groupby(["game_idx", PIECE_PLAYER_SEASON, "phase"], as_index=False)[counts].sum()


def _regular_season_possessions(game_poss: pd.DataFrame, n_keys: int, phase: str) -> np.ndarray:
    """Each player-season's offensive possessions in a regular-season piece; zeros in a playoff piece."""
    if phase != "RS":
        return np.zeros(n_keys)
    return np.bincount(game_poss[PIECE_PLAYER_SEASON].to_numpy(), weights=game_poss["poss_off"].to_numpy(dtype=float),
                       minlength=n_keys)


def _stint_values(stints: pd.DataFrame) -> dict:
    """The per-stint values the rows and the context read, and each stint's group (its game, or its playoff
    series)."""
    n_stints = len(stints)
    is_po = stints["phase"].to_numpy() == "PO"
    neutral = stints["neutral"].to_numpy().astype(bool) if "neutral" in stints.columns else np.zeros(n_stints, bool)
    series = stints["series_id"].astype(str).to_numpy()
    game_id = stints["game_id"].astype(str).to_numpy()
    groups = np.where(is_po, np.char.add("S:", series.astype(str)), np.char.add("G:", game_id.astype(str)))
    return dict(n_st=n_stints, is_po=is_po, neutral=neutral, is_gt=stints["is_gt"].to_numpy().astype(bool),
                margin_h=stints["margin_h"].to_numpy().astype(float),
                frac_rem=stints["frac_rem"].to_numpy().astype(float),
                game_idx=stints["game_idx"].to_numpy(), groups=groups)


def _counter_arrays(stints: pd.DataFrame) -> dict:
    """The counters the stints carry, as one (stints x columns) matrix per side, and each side's five ids."""
    have = [name for name in POSS_COUNTERS + SLOT_COUNTERS
            if f"{name}_h" in stints.columns and f"{name}_a" in stints.columns]
    columns = have + ["pts", "poss"]
    counters = {side: np.column_stack([stints[f"{name}_{side}"].to_numpy(dtype=float) for name in columns])
                for side in ("h", "a")}
    pids = {"h": stints[HOME_SLOTS].to_numpy(dtype=np.int64), "a": stints[AWAY_SLOTS].to_numpy(dtype=np.int64)}
    return dict(have=have, ccols=columns, counters=counters, pids=pids, counters_h=counters["h"],
                counters_a=counters["a"], pids_h=pids["h"], pids_a=pids["a"])


def pieces_from_frames(stints: pd.DataFrame, box: pd.DataFrame, box_counts) -> list:
    """One piece per season and phase present in `stints`, seasons ascending, the regular season first: the order
    `assemble` concatenates them in, which for stints sorted the same way keeps each stint's place."""
    pieces = []
    for season in sorted(stints["season"].unique()):
        for phase in SEASON_PHASES:
            in_stints = (stints["season"] == season) & (stints["phase"] == phase)
            if in_stints.any():
                in_box = (box["season"] == season) & (box["phase"] == phase)
                pieces.append(piece_from_frames(stints[in_stints], box[in_box], season, phase, box_counts))
    return pieces


# ------------------------------------------------------------------------------------------ assembling pieces
class Players(NamedTuple):
    """The design's player-seasons and player columns (DesignSpec's two numberings)."""
    keys: np.ndarray                         # every player-season key in the pieces, sorted
    player_seasons: pd.DataFrame             # key, season, player_id, player_season, player_column
    seasons: list                            # the design's seasons, ascending
    season_of_player_season: np.ndarray      # each player-season's index into `seasons`
    columns: pd.DataFrame                    # the player columns: player_column, player_id (+ season under
                                             # player_unit "player_season")
    player_of_player_season: np.ndarray      # each player-season's player column
    unit: str                                # "player" or "player_season"


def assemble(pieces: list, cfg: dict, target: str = "pts") -> Design:
    """The design of `pieces` (one or more seasons, each regular season before its playoffs) on `target`.

    Reads from `cfg`: player_unit ("player_season" by default), gt_weight (1.0), margin_clip (25), and the
    possession thresholds of the diagnostic column groups, low_poss_threshold (500) and starter_poss_threshold
    (1500).

    The rows are every piece's home-offence rows, then every piece's away-offence rows; a design built from the
    same stints on another target has the same players, the same context columns and, where both keep a row, the
    same row, so their fits compare directly."""
    players = player_columns(pieces, str(cfg.get("player_unit", "player_season")))
    stints, to_block, game_offset = block_stints(pieces, players)
    rows = kept_rows(pieces, stints, target)
    context_names, context = context_columns(rows, players.seasons, float(cfg.get("margin_clip", 25)))
    lineup_o, lineup_d = sorted_lineups(rows["offence"], rows["defence"])
    n_players = len(players.columns)
    games = pd.concat([piece["games"].assign(game_idx=piece["games"]["game_idx"].to_numpy() + offset)
                       for piece, offset in zip(pieces, game_offset)], ignore_index=True)
    table = row_table(rows, games, players.seasons)
    return Design(X_src=partial(design_matrix, lineup_o, lineup_d, context, rows["game"], n_players),
                  y=TARGETS[target]["scale"] * rows["num"] / rows["den"],
                  w=rows["den"] * np.where(rows["is_gt"], float(cfg.get("gt_weight", 1.0)), 1.0),
                  groups=stints["groups"][rows["stint"]],
                  spec=design_spec(pieces, players, to_block, context_names, cfg),
                  game_box=_block_table(pieces, "game_box", game_offset, to_block),
                  game_poss=_block_table(pieces, "game_poss", game_offset, to_block),
                  rows=table, games=games, counters=counters_frame(pieces, rows["stint_by_side"], table),
                  parts=dict(Z=player_matrix(lineup_o, lineup_d, n_players), F=context, lineup_o=lineup_o,
                             lineup_d=lineup_d, game_idx=rows["game"].astype(np.int64)))


def player_columns(pieces: list, unit: str) -> Players:
    """Every player-season of the pieces, numbered in key order (season, then player), and the player columns."""
    keys = np.unique(np.concatenate([piece["keys_u"] for piece in pieces]))
    player_seasons = pd.DataFrame({"key": keys, "season": keys // SEASON_KEY, "player_id": keys % SEASON_KEY})
    player_seasons["player_season"] = np.arange(len(player_seasons), dtype=np.int64)
    seasons = sorted(player_seasons["season"].unique().tolist())
    season_of_player_season = np.searchsorted(np.asarray(seasons), player_seasons["season"].to_numpy())
    columns, player_of_player_season = _player_units(player_seasons, unit)
    player_seasons["player_column"] = player_of_player_season
    return Players(keys, player_seasons, seasons, season_of_player_season, columns, player_of_player_season, unit)


def _player_units(player_seasons: pd.DataFrame, unit: str) -> tuple[pd.DataFrame, np.ndarray]:
    """(the player columns, each player-season's column).

    `unit` "player" gives one column per player over the whole design, "player_season" one per player-season."""
    if unit not in PLAYER_UNITS:
        raise ValueError(f"player_unit must be one of {PLAYER_UNITS}, got {unit!r}")
    if unit == "player":
        players = np.unique(player_seasons["player_id"].to_numpy())
        columns = pd.DataFrame({"player_id": players, "player_column": np.arange(len(players), dtype=np.int64)})
        return columns, pd.Index(players).get_indexer(player_seasons["player_id"].to_numpy())
    columns = player_seasons[["player_season", "player_id", "season"]].rename(
        columns={"player_season": "player_column"}).copy()
    return columns, player_seasons["player_season"].to_numpy()


def block_stints(pieces: list, players: Players) -> tuple[dict, list, np.ndarray]:
    """(stints, to_block, game_offset): every stint of the pieces in order, with its two fives as player columns
    (home, away), its game numbered across the design (game) and its season's index (season); each piece's map
    from its local player-season positions to the design's; each piece's first game number."""
    to_block = [np.searchsorted(players.keys, piece["keys_u"]) for piece in pieces]
    game_offset = np.cumsum([0] + [piece["n_games"] for piece in pieces])[:-1]
    slots = [players.player_of_player_season[local[piece["slot_loc"]]] for piece, local in zip(pieces, to_block)]
    season = np.concatenate([np.full(piece["n_st"], piece["season"]) for piece in pieces])
    stints = {name: np.concatenate([piece[name] for piece in pieces])
              for name in ("is_po", "neutral", "is_gt", "margin_h", "frac_rem", "groups")}
    stints.update(home=np.concatenate([ten[:, :5] for ten in slots]),
                  away=np.concatenate([ten[:, 5:] for ten in slots]),
                  game=np.concatenate([piece["game_idx"] + offset for piece, offset in zip(pieces, game_offset)]),
                  season=np.searchsorted(np.asarray(players.seasons), season))
    return stints, to_block, game_offset


def kept_rows(pieces: list, stints: dict, target: str) -> dict:
    """Every row the target keeps (denominator above 0), home-offence rows first: its two fives (offence, defence),
    possessions, the target's denominator and numerator, home (+1, -1, or 0 at a neutral site), the margin from the
    offence's side, its stint, and the stint's playoff, garbage-time, share-left, season and game values."""
    _require_target_columns(pieces, target)
    rule = TARGETS[target]
    sides = [_side_rows(pieces, stints, rule, on_offence, on_defence, side, sign)
             for on_offence, on_defence, side, sign in (("home", "away", "h", 1.0), ("away", "home", "a", -1.0))]
    rows = {name: np.concatenate([side[name] for side in sides]) for name in sides[0]}
    stint = rows["stint"]
    rows.update(is_po=stints["is_po"][stint], is_gt=stints["is_gt"][stint], frac_rem=stints["frac_rem"][stint],
                season=stints["season"][stint], game=stints["game"][stint],
                stint_by_side=[side["stint"] for side in sides])
    return rows


def _require_target_columns(pieces: list, target: str) -> None:
    rule = TARGETS[target]
    needed = {f"{name}_{side}" for name in [*rule["num"], rule["den"]] for side in ("h", "a")}
    missing = sorted(needed - set.intersection(*(set(piece["tcols"]) for piece in pieces)))
    if missing:
        raise KeyError(f"target {target!r} needs stint columns {missing}; rebuild the stints "
                       f"(openrapm data stints <first> <last> RS --force)")


def _side_rows(pieces: list, stints: dict, rule: dict, on_offence: str, on_defence: str, side: str,
               sign: float) -> dict:
    """The rows of one side (side "h": the home team on offence), those whose denominator is above 0.  `on_offence`
    and `on_defence` name the stints' fives, "home" or "away"."""
    def column(name):
        return np.concatenate([piece["tcols"][name] for piece in pieces])

    possessions = column(f"poss_{side}")
    denominator = column(f"{rule['den']}_{side}")
    numerator = sum(weight * column(f"{name}_{side}") for name, weight in rule["num"].items())
    keep = denominator > 0.0
    return dict(offence=stints[on_offence][keep], defence=stints[on_defence][keep], poss=possessions[keep],
                den=denominator[keep], num=numerator[keep], home=np.where(stints["neutral"][keep], 0.0, sign),
                margin=sign * stints["margin_h"][keep], stint=np.flatnonzero(keep),
                is_home_off=np.full(int(keep.sum()), sign > 0))


def context_columns(rows: dict, seasons: list, margin_clip: float) -> tuple[list, np.ndarray]:
    """(names, C): the context columns of every row, as a dense (rows x columns) array.

    home, int_<season> (1 in that season's rows), is_po, po_home = is_po x home, is_gt, margin (clipped at
    +-margin_clip), margin_frem = margin x share of the game left."""
    playoff = rows["is_po"].astype(float)
    margin = np.clip(rows["margin"], -margin_clip, margin_clip)
    names = ["home"] + [f"int_{season}" for season in seasons] + ["is_po", "po_home", "is_gt", "margin", "margin_frem"]
    columns = ([rows["home"]] + [(rows["season"] == index).astype(float) for index in range(len(seasons))]
               + [playoff, playoff * rows["home"], rows["is_gt"].astype(float), margin, margin * rows["frac_rem"]])
    return names, np.column_stack(columns)


def sorted_lineups(offence: np.ndarray, defence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row's five offensive and five defensive player columns in increasing order; a player twice in one five
    is an error."""
    lineup_o, lineup_d = np.sort(offence, axis=1), np.sort(defence, axis=1)
    if not ((np.diff(lineup_o, axis=1) > 0).all() and (np.diff(lineup_d, axis=1) > 0).all()):
        raise ValueError("duplicate player ids inside a lineup")
    return lineup_o, lineup_d


def design_matrix(lineup_o, lineup_d, context, game, n_players: int) -> sp.csr_matrix:
    """X = [Z_O | Z_D | C | game] in CSR form.

    Every row has the same pattern: its five offensive columns, its five defensive columns (+ n_players), every
    context column and the game number.  The context's zeros are stored, not dropped, so each row holds exactly
    10 + context + 1 entries in increasing column order and the matrix is written straight, with no sort."""
    n_rows, n_context = context.shape
    columns = 2 * n_players + np.arange(n_context + 1)
    indices = np.hstack([lineup_o, lineup_d + n_players, np.broadcast_to(columns, (n_rows, n_context + 1))]).ravel()
    data = np.hstack([np.ones((n_rows, 10)), context, game.astype(float)[:, None]]).ravel()
    per_row = 10 + n_context + 1
    row_starts = np.arange(0, n_rows * per_row + 1, per_row, dtype=np.int64)
    matrix = sp.csr_matrix((data, indices, row_starts), shape=(n_rows, 2 * n_players + n_context + 1))
    matrix.has_sorted_indices = True
    return matrix


def player_matrix(lineup_o, lineup_d, n_players: int) -> sp.csr_matrix:
    """Z = [Z_O | Z_D]: ten ones per row."""
    n_rows = len(lineup_o)
    matrix = sp.csr_matrix((np.ones(n_rows * 10), np.hstack([lineup_o, lineup_d + n_players]).ravel(),
                            np.arange(0, n_rows * 10 + 1, 10, dtype=np.int64)), shape=(n_rows, 2 * n_players))
    matrix.has_sorted_indices = True
    return matrix


def _block_table(pieces: list, name: str, game_offset: np.ndarray, to_block: list) -> pd.DataFrame:
    """A per-game, per-player-season table of every piece, renumbered to the design's games and player-seasons (the
    piece's `PIECE_PLAYER_SEASON` column becomes the design's `player_season`, in the same place)."""
    tables = [piece[name].rename(columns={PIECE_PLAYER_SEASON: "player_season"}) for piece in pieces]
    return pd.concat([table.assign(game_idx=table["game_idx"].to_numpy() + offset,
                                   player_season=local[table["player_season"].to_numpy()])
                      for table, offset, local in zip(tables, game_offset, to_block)], ignore_index=True)


def design_spec(pieces: list, players: Players, to_block: list, context_names: list, cfg: dict) -> DesignSpec:
    """The column layout, with each player column's season and the diagnostic column groups: player columns with
    under low_poss_threshold regular-season possessions ("low_poss") and with starter_poss_threshold or more
    ("high_poss"; "high_poss_O" and "high_poss_D" its offence and defence columns)."""
    n_players = len(players.columns)
    possessions = np.zeros(len(players.player_seasons))
    for piece, local in zip(pieces, to_block):
        possessions[local] += piece["psx_poss"]
    column_possessions = np.bincount(players.player_of_player_season, weights=possessions, minlength=n_players)
    low = np.flatnonzero(column_possessions < float(cfg.get("low_poss_threshold", 500)))
    high = np.flatnonzero(column_possessions >= float(cfg.get("starter_poss_threshold", 1500)))
    possession_groups = {"low_poss": np.concatenate([low, low + n_players]),
                         "high_poss": np.concatenate([high, high + n_players]),
                         "high_poss_O": high, "high_poss_D": high + n_players}
    return DesignSpec(n_players=n_players, seasons=players.seasons,
                      season_of_player=_season_of_columns(possessions, players, n_players),
                      players=players.columns.copy(), context_names=context_names,
                      box_counts=list(pieces[0]["game_box"].columns[3:]), possession_groups=possession_groups,
                      player_seasons=players.player_seasons[["player_season", "player_id", "season",
                                                             "player_column"]].copy(),
                      season_of_player_season=players.season_of_player_season,
                      player_of_player_season=players.player_of_player_season, player_unit=players.unit)


def _season_of_columns(possessions: np.ndarray, players: Players, n_players: int) -> np.ndarray:
    """Each player column's season index: its own season, or under player_unit "player" the season in which the
    player had the most regular-season possessions (the later season on a tie)."""
    if players.unit != "player":
        return players.season_of_player_season
    season_of_player = np.zeros(n_players, dtype=np.int64)
    best = np.full(n_players, -1.0)
    for player_season in np.argsort(possessions):
        column = players.player_of_player_season[player_season]
        if possessions[player_season] >= best[column]:
            best[column] = possessions[player_season]
            season_of_player[column] = players.season_of_player_season[player_season]
    return season_of_player


def row_table(rows: dict, games: pd.DataFrame, seasons: list) -> pd.DataFrame:
    """One line per row: game_idx, season, phase, poss, den, is_home_off, is_gt, stint and the game's half."""
    game_half = np.empty(len(games), dtype=object)
    game_half[games["game_idx"].to_numpy()] = games["half"].to_numpy()
    return pd.DataFrame({"game_idx": rows["game"], "season": np.asarray(seasons)[rows["season"]],
                         "phase": pd.Series(np.where(rows["is_po"], "PO", "RS").astype(object), dtype=object),
                         "poss": rows["poss"], "den": rows["den"], "is_home_off": rows["is_home_off"],
                         "is_gt": rows["is_gt"], "stint": rows["stint"],
                         "half": pd.Series(game_half[rows["game"]], dtype=object)})


def counters_frame(pieces: list, stint_by_side: list, table: pd.DataFrame) -> pd.DataFrame | None:
    """The offence's counters of every row (home-offence rows, then away), with the offensive five's ids
    (pid_s1..pid_s5) and the game's half; None when the stints carry no counters."""
    if not pieces[0]["have"]:
        return None
    first_stint = np.cumsum([0] + [piece["n_st"] for piece in pieces])
    blocks = []
    for stint, side in zip(stint_by_side, ("h", "a")):
        keep = np.zeros(first_stint[-1], dtype=bool)
        keep[stint] = True
        for piece, start, stop in zip(pieces, first_stint[:-1], first_stint[1:]):
            blocks.append(piece["counters"][side][np.flatnonzero(keep[start:stop])])
    counters = pd.DataFrame(np.vstack(blocks), columns=pieces[0]["ccols"])
    fives = np.vstack([np.vstack([piece["pids"][side] for piece in pieces])[stint]
                       for stint, side in zip(stint_by_side, ("h", "a"))])
    for slot in range(5):
        counters[f"pid_s{slot + 1}"] = fives[:, slot]
    counters["half"] = pd.Series(table["half"].to_numpy(), dtype=object)
    return counters
