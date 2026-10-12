"""The design pieces on disk, and `DesignCache`, the way every command gets a season's design.

A piece is everything about one season and phase that a design needs (design.piece_from_frames): its games in
order, its player-seasons, every stint's lineup, flags and counters.  Building one reads the season's stint file and
its game log, so pieces are kept on disk under paths.cache, in pieces/{season}_{phase}/ (data/cache/pieces by
default): meta.pkl holds the tables, and four .npy files hold the counter and lineup arrays, memory-mapped on load.
A design is then `design.assemble` on the pieces of its seasons, the same function the simulated tests use.

Words used below
    stamp    what a stored piece was built from: a version number, the box-count names (config box_counts), and the
             modification time and size of the season's stint file and game log.  A piece is reused only when its
             stamp equals the one computed now, compared whole, so editing a stint file or the box counts rebuilds it
    block    one design over several consecutive seasons: the rows of every season, one column per player
"""
from __future__ import annotations

import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..design import SEASON_PHASES, TARGETS, Design, assemble, piece_from_frames
from .box_scores import season_box
from .config import resolve
from .shots import DEFENSE_TARGETS
from .stints import build_season

PIECES_VERSION = 1
ARRAY_KEYS = ("counters_h", "counters_a", "pids_h", "pids_a")     # memory-mapped from .npy on disk
# the rest of a piece, in meta.pkl, in the order the stored pieces were written in, so that this code writes the same
# meta.pkl, byte for byte, as the code that wrote them
META_KEYS = ("season", "phase", "stamp", "tcols", "games", "n_games", "keys_u", "slot_loc", "n_st", "is_po",
             "neutral", "is_gt", "margin_h", "frac_rem", "game_idx", "groups", "game_poss", "game_box", "psx_poss",
             "have", "ccols")
MAX_PIECES = 8          # pieces a DesignCache holds in memory: a season's two, for the few seasons a build is on


def _stamp(season: int, phase: str, cfg: dict) -> dict:
    """What the stored piece must have been built from: the stints and the game log files, the box counts.

    The box counts are stored under the key `features`, their name before config.yaml called them
    `box_counts`: the stamp is compared whole, so keeping the key keeps every piece already on disk valid."""
    stint_file = resolve(cfg, "stints") / f"{season}_{phase}.parquet"
    game_log = resolve(cfg, "raw") / "gamelog" / f"{season}_{phase}.parquet"
    out = dict(version=PIECES_VERSION, features=tuple(cfg["box_counts"]))
    for name, path in (("stints", stint_file), ("gamelog", game_log)):
        if path.exists():
            info = path.stat()
            out[name] = (int(info.st_mtime_ns), int(info.st_size))
    return out


def _pieces_dir(cfg: dict) -> Path:
    return resolve(cfg, "cache") / "pieces"


def _load_piece(season: int, phase: str, cfg: dict) -> dict | None:
    """The stored piece, or None when there is none, it cannot be read, or its stamp is not today's."""
    folder = _pieces_dir(cfg) / f"{season}_{phase}"
    meta = folder / "meta.pkl"
    if not meta.exists():
        return None
    try:
        with open(meta, "rb") as handle:
            piece = pickle.load(handle)
        if piece.get("stamp") != _stamp(season, phase, cfg):
            return None
        for key in ARRAY_KEYS:
            piece[key] = np.load(folder / f"{key}.npy", mmap_mode="r")
    except Exception:            # a piece that cannot be read, for whatever reason, is rebuilt rather than trusted
        return None
    piece["counters"] = {"h": piece["counters_h"], "a": piece["counters_a"]}
    piece["pids"] = {"h": piece["pids_h"], "a": piece["pids_a"]}
    return piece


def _save_piece(piece: dict, season: int, phase: str, cfg: dict) -> None:
    folder = _pieces_dir(cfg) / f"{season}_{phase}"
    folder.mkdir(parents=True, exist_ok=True)
    small = {key: piece[key] for key in META_KEYS}
    for key in ARRAY_KEYS:
        np.save(folder / f"{key}.npy", np.ascontiguousarray(piece[key]))
    partial = folder / "meta.pkl.tmp"
    with open(partial, "wb") as handle:
        pickle.dump(small, handle, protocol=pickle.HIGHEST_PROTOCOL)
    partial.replace(folder / "meta.pkl")


def build_piece(season: int, phase: str, cfg: dict) -> dict:
    """One season and phase's piece from its stint file and game log, stamped."""
    stints = build_season(int(season), phase, cfg)[0]
    box = season_box([int(season)], [phase], cfg)
    piece = piece_from_frames(stints, box, int(season), str(phase), cfg["box_counts"])
    piece["stamp"] = _stamp(int(season), str(phase), cfg)
    return piece


def stored_piece(season: int, phase: str, cfg: dict) -> dict:
    """The piece of one season and phase: read from disk, or built and written there when it is missing or stale.
    A folder that cannot be written to only means the piece is built again next time."""
    piece = _load_piece(int(season), str(phase), cfg)
    if piece is None:
        piece = build_piece(season, phase, cfg)
        try:
            _save_piece(piece, int(season), str(phase), cfg)
        except OSError:
            pass
    return piece


def block_design(seasons, cfg: dict, target: str = "pts", phases=SEASON_PHASES) -> Design:
    """The design of `seasons` on a `design.TARGETS` target, from the stored pieces of `phases` (both, by default:
    a season is its regular season and its playoffs)."""
    return assemble([stored_piece(season, phase, cfg) for season in seasons for phase in phases], cfg, target)


class DesignCache:
    """One season's design (or a run of seasons') on one target, built from the stored pieces and kept for the
    next few requests.

    `target` is a `design.TARGETS` key ("pts", "points_with_expected_free_throws") or a `data.shots.DEFENSE_TARGETS`
    key ("points_with_expected_opponent_threes" and its "_all" variant), which reprices the same seasons' points
    design.  Once `keep` designs are held the cache empties: a season's design is hundreds of megabytes, and a
    build asks for them in season order.  The last `MAX_PIECES` pieces stay in memory, so the targets of one
    season share them.
    """

    def __init__(self, cfg: dict, keep: int = 4):
        self.cfg, self.keep = cfg, int(keep)
        self._designs: dict = {}
        self._pieces: OrderedDict = OrderedDict()

    def season(self, season: int, target: str) -> Design:
        """The design of `season` on `target`, built on first request."""
        return self.block([season], target)

    def block(self, seasons, target: str) -> Design:
        """One design over several seasons on `target` (rows of every season, one column per player), built on
        first request.  The trade set (scripts/checks/70_tradeset.py) fits a season and its two neighbours
        together."""
        key = (tuple(int(season) for season in seasons), target)
        if key not in self._designs:
            design = self._build(list(key[0]), target)
            if len(self._designs) >= self.keep:
                self._designs.clear()
            self._designs[key] = design
        return self._designs[key]

    def clear(self) -> None:
        """Forget every held design."""
        self._designs.clear()

    def piece(self, season: int, phase: str) -> dict:
        """The piece of one season and phase, from memory when it is among the last `MAX_PIECES` asked for."""
        key = (int(season), str(phase))
        if key not in self._pieces:
            self._pieces[key] = stored_piece(*key, self.cfg)
            while len(self._pieces) > MAX_PIECES:
                self._pieces.popitem(last=False)
        self._pieces.move_to_end(key)
        return self._pieces[key]

    def _build(self, seasons: list, target: str) -> Design:
        if target in DEFENSE_TARGETS:
            design = DEFENSE_TARGETS[target](seasons, self.cfg, self.block(seasons, "pts"))
            return design[0] if isinstance(design, tuple) else design
        if target not in TARGETS:
            raise KeyError(f"unknown target {target!r}; have {sorted(TARGETS)} and {sorted(DEFENSE_TARGETS)}")
        pieces = [self.piece(season, phase) for season in seasons for phase in SEASON_PHASES]
        return assemble(pieces, self.cfg, target)
