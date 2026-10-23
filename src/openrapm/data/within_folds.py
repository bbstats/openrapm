"""The within-season folds on disk, read back for step 4 (prior_shrink.py) and the held-out audit.

`openrapm folds` (commands/within_season.py) rates a season from three quarters of its games and keeps the other
quarter back.  Each fold is three files under paths.within/<tag>/ with one stem, <season>_r<repeat>_f<fold>:

    fold_<stem>.json         the fold's summary (season, repeat, fold, the stand-in ratings) and its leak checks
    players_<stem>.parquet   one row per player and side, rated from the fit games: rating, prior part, games' part
    test_<stem>.npz          the held-back team-games

Words used below
    repeat           one random assignment of a season's games to four folds; a season has three repeats
    test team-game   one team's offence in one held-back game: its points per 100 possessions as scored (y), its
                     possessions (w), the controls (F: home, the season's intercept, playoffs, playoff home), and
                     each player's share of its possessions, offence and defence (Z; each side's shares sum to 5)
    rated player     a player with offensive possessions in the fit games, so with a rating of his own
    stand-in rating  what a player without one carries in the test games: a quarter of the possession-weighted
                     mean rating of the fold's players with under 500 possessions (`fill_o`, `fill_d` in the
                     summary), the year-over-year test's rule
    base             each test team-game's prediction from the ten players alone,
                     sum over the offence of share x rating_off + sum over the defence of share x (-rating_def)
                     (defence in points allowed), a stand-in rating where a player has none

Example: 2026, repeat 0, fold 0 holds 329 test games (658 test team-games) and 581 rated players (its test games
carry 582 player columns: one player appears only there, at the stand-in rating); its stand-in ratings are
-0.31 on offence and +0.19 on defence (points allowed).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .legacy_names import translate_frame

__all__ = ["Fold", "read_fold", "fold_stems", "check_fold_record", "check_repeats", "check_fold_files", "load_folds",
           "fold_games", "HeldStints", "held_back_stints"]



@dataclass
class Fold:
    """One fold of one season: its rated players and its test team-games."""
    stem: str
    season: int
    repeat: int
    fold: int
    fill_o: float              # the stand-in ratings (offence; defence in points allowed)
    fill_d: float
    y: np.ndarray              # points per 100 as scored, one per test team-game
    w: np.ndarray              # possessions, one per test team-game
    F: np.ndarray              # the controls, one row per test team-game
    game_idx: np.ndarray       # the game each test team-game belongs to
    base: np.ndarray           # the ten players' prediction (module docstring)
    rows: dict                 # "O" / "D": the rated players' rows for that side, in Z's column order
    Z: dict                    # "O" / "D": test team-games x rated players, each player's share of the possessions


def _test_games(directory: Path, stem: str) -> tuple:
    """(Z over every player, the players' ids, y, w, F, game_idx) from test_<stem>.npz."""
    stored = np.load(directory / f"test_{stem}.npz", allow_pickle=False)
    shares = sp.csr_matrix((stored["Z_data"], stored["Z_indices"], stored["Z_indptr"]),
                           shape=tuple(stored["Z_shape"]))
    return (shares, stored["player_ids"], stored["y"].astype(float), stored["w"].astype(float),
            np.asarray(stored["F"], dtype=float), stored["game_idx"])


def _rated(players: pd.DataFrame) -> pd.DataFrame:
    """One row per rated player, indexed by player_id."""
    rated = players.drop_duplicates("player_id").set_index("player_id")
    return rated[rated.poss_off > 0]


def _base(shares, player_ids, rated: pd.DataFrame, fill_o: float, fill_d: float) -> np.ndarray:
    """The ten players' prediction of every test team-game, a stand-in rating where a player has none.

    Each player carries what the fold's own score gave him (`within_season.score_test_games`): his rating, or the
    stand-in rating.
    """
    n_players = len(player_ids)
    offence = pd.Series(player_ids).map(rated.rating_off).to_numpy(float)
    defence = pd.Series(player_ids).map(-rated.rating_def).to_numpy(float)
    offence = np.where(np.isnan(offence), fill_o, offence)
    defence = np.where(np.isnan(defence), fill_d, defence)
    return np.asarray(shares[:, :n_players] @ offence).ravel() + np.asarray(shares[:, n_players:] @ defence).ravel()


def _side_blocks(players: pd.DataFrame, rated: pd.DataFrame, shares, player_ids) -> tuple[dict, dict]:
    """({side: the rated players' rows}, {side: their columns of Z}), in the same player order."""
    n_players = len(player_ids)
    rows, blocks = {}, {}
    for side, block in (("O", shares[:, :n_players]), ("D", shares[:, n_players:])):
        side_rows = players[(players.side == side) & players.player_id.isin(rated.index)].reset_index(drop=True)
        column = pd.Series(np.arange(n_players), index=player_ids).reindex(side_rows.player_id).to_numpy()
        if np.isnan(column.astype(float)).any():
            raise ValueError("a rated player has no column in the test games")
        rows[side] = side_rows
        blocks[side] = sp.csr_matrix(block[:, column.astype(np.int64)])
    return rows, blocks


def read_fold(directory: Path, stem: str) -> Fold:
    """One fold's three files, as a `Fold`."""
    directory = Path(directory)
    summary = json.loads((directory / f"fold_{stem}.json").read_text())["summary"]
    # a fold written before 2026-10-11 names its input columns the old way (data/legacy_names.py)
    players = translate_frame(pd.read_parquet(directory / f"players_{stem}.parquet"))
    shares, player_ids, y, w, controls, game_idx = _test_games(directory, stem)
    rated = _rated(players)
    fill_o, fill_d = float(summary["fill_o"]), float(summary["fill_d"])
    rows, blocks = _side_blocks(players, rated, shares, player_ids)
    return Fold(stem=stem, season=int(summary["season"]), repeat=int(summary["repeat"]), fold=int(summary["fold"]),
                fill_o=fill_o, fill_d=fill_d, y=y, w=w, F=controls, game_idx=game_idx,
                base=_base(shares, player_ids, rated, fill_o, fill_d), rows=rows, Z=blocks)


def fold_stems(directory: Path, seasons: tuple[int, int]) -> list[str]:
    """The stems of the folds in `directory` whose season is in [first, last], sorted."""
    first, last = seasons
    stems = sorted(path.stem[len("fold_"):] for path in Path(directory).glob("fold_*.json"))
    stems = [stem for stem in stems if first <= int(stem[:4]) <= last]
    if not stems:
        raise ValueError(f"no folds in {directory} for seasons {first}-{last}")
    return stems


def check_fold_record(record: dict, stem: str) -> None:
    """A fold must be a three-quarter-season fold whose leak checks all passed.

    `rate_from` "rest" is the fold size the shipped multipliers were fitted at; a fold written before that key
    existed has none, or NaN, and is "rest" too.
    """
    mode = record["summary"].get("rate_from")
    if mode is None or (isinstance(mode, float) and np.isnan(mode)):
        mode = "rest"
    if mode != "rest":
        raise ValueError(f"fold {stem} rates from {mode!r}, not the three-quarter-season `rest` size")
    failed = [name for name, passed in record["checks"].items() if isinstance(passed, bool) and not passed]
    if failed:
        raise ValueError(f"fold {stem}: leak checks failed {failed}")


def check_repeats(records: dict) -> None:
    """Every repeat of every season must have exactly its folds, 0 to K - 1, K its records' `folds_per_repeat`:
    four for a fold written before that key existed, as step 4's folds are built; `records` maps a stem to its fold
    json."""
    repeats: dict = {}
    for record in records.values():
        summary = record["summary"]
        held = repeats.setdefault((int(summary["season"]), int(summary["repeat"])), (set(), set()))
        held[0].add(int(summary["fold"]))
        held[1].add(int(summary.get("folds_per_repeat", 4)))
    short = {repeat: sorted(folds) for repeat, (folds, sizes) in repeats.items()
             if len(sizes) != 1 or folds != set(range(min(sizes)))}
    if short:
        raise ValueError(f"(season, repeat) without exactly its folds: {sorted(short.items())[:5]}")


def check_fold_files(directory: Path, seasons: tuple[int, int]) -> list[str]:
    """The stems of the folds in `directory` whose season is in [first, last], after checking that each fold has
    its three files, each passed its leak checks and every repeat has its folds.  Reads only the small json
    files, so a command can call it before the minutes of work that end in `load_folds`; raises ValueError naming
    the first problem.

    Example: on a fresh clone `check_fold_files(paths.within / "weightfix_within", (2017, 2026))` raises
    "no folds in .../outputs/within/weightfix_within for seasons 2017-2026".
    """
    directory = Path(directory)
    stems = fold_stems(directory, seasons)
    missing = [name for stem in stems for name in (f"players_{stem}.parquet", f"test_{stem}.npz")
               if not (directory / name).exists()]
    if missing:
        raise ValueError(f"{directory} lacks {len(missing)} fold files, among them {missing[0]}")
    records = {stem: json.loads((directory / f"fold_{stem}.json").read_text()) for stem in stems}
    for stem, record in records.items():
        check_fold_record(record, stem)
    check_repeats(records)
    return stems


def load_folds(directory: Path, seasons: tuple[int, int]) -> list[Fold]:
    """Every fold in `directory` whose season is in [first, last], after `check_fold_files`.

    Example: `load_folds(paths.within / "weightfix_within", (2017, 2026))` reads 120 folds: ten seasons, three
    repeats each, four folds a repeat.
    """
    return [read_fold(Path(directory), stem) for stem in check_fold_files(directory, seasons)]


def fold_games(directory: Path, fold: Fold) -> tuple[set, set]:
    """(fit games, held-back games) of a fold, as game ids: its season's assignment, folds_<season>.parquet, column
    r<repeat>.  The held-back games must be the ones its test file stores; a ValueError says so if not.

    Example: 2024, repeat 0, fold 0 holds back 328 of the season's 1,312 games and fits on the other 984.
    """
    directory = Path(directory)
    assignment = pd.read_parquet(directory / f"folds_{fold.season}.parquet")
    held_mask = assignment[f"r{fold.repeat}"].to_numpy() == fold.fold
    games = assignment.index.astype(str)
    held = set(games[held_mask])
    stored = set(np.load(directory / f"test_{fold.stem}.npz", allow_pickle=True)["game_id"].astype(str))
    if stored != held:
        raise ValueError(f"{fold.stem}: the test file holds {len(stored)} games, the assignment {len(held)}")
    return set(games[~held_mask]), held


@dataclass
class HeldStints:
    """A fold's held-back stints, every stint of its test games with possessions, as step 4's within layer fits on
    them (prior_shrink.stint_equations, experiment 106)."""
    y: np.ndarray              # points per 100 as scored
    w: np.ndarray              # possessions
    lineup_o: np.ndarray       # stints x 5: the offence's player ids, -1 where the design has none
    lineup_d: np.ndarray       # stints x 5: the defence's
    team_game: np.ndarray      # one code per team-game: the game and which team has the ball


def held_back_stints(design, held_ids) -> HeldStints:
    """The rows of a season's points design (openrapm.data.designs) whose game is in `held_ids` (`fold_games`'s
    second set) and that carry possessions.

    Example: 2026, repeat 0, fold 0: the 329 held-back games' 19,000 or so stint rows, both teams on offence.
    """
    games = design.games.set_index("game_idx").game_id.astype(str)
    game_id = design.rows.game_idx.map(games).to_numpy().astype(str)
    keep = np.isin(game_id, np.array(sorted(held_ids), dtype=str)) & (np.asarray(design.w) > 0)
    ids = design.spec.players.set_index("player_column").player_id

    def lineup(name):
        return ids.reindex(design.parts[name][keep].ravel()).fillna(-1).to_numpy(np.int64).reshape(-1, 5)
    home = design.rows.is_home_off.to_numpy()[keep]
    team_game, _ = pd.factorize(pd.Series(game_id[keep]) + np.where(home, "h", "a"))
    return HeldStints(y=np.asarray(design.y)[keep], w=np.asarray(design.w)[keep], lineup_o=lineup("lineup_o"),
                      lineup_d=lineup("lineup_d"), team_game=team_game)
