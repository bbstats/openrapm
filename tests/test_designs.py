"""The design-piece cache (src/openrapm/data/designs.py): where the pieces live, and the stamp that keeps every
piece already on disk valid across the 2026-10 cleanup.

A piece is reused only when the stamp stored with it equals the stamp the code computes now, compared whole.
The cleanup renamed the config key `features` to `box_counts` and moved the cache folder behind config
`paths.cache`; if either had changed the stamp, every one of the sixty pieces in data/cache/pieces would
silently rebuild on the next run.  The expected values below are written out by hand from the old code
(the piece cache's `_stamp` at tag archive/research-2026-10), not imported from it.
"""
import os
import pickle

import numpy as np

from openrapm.data import designs

BOX_COUNTS = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl", "blk", "pf"]
STINTS_NS, STINTS_SIZE = 1_700_000_000_000_000_000, 1234     # multiples of 100 ns: NTFS keeps 100 ns ticks
GAMELOG_NS, GAMELOG_SIZE = 1_700_000_000_123_456_700, 567

# The old `_stamp` built this dict, in this order: version=PIECES_VERSION (1), features=tuple(cfg["features"]),
# then for each of the stints file and the game log file that exists, (st_mtime_ns, st_size).
OLD_STAMP = {"version": 1, "features": tuple(BOX_COUNTS),
             "stints": (STINTS_NS, STINTS_SIZE), "gamelog": (GAMELOG_NS, GAMELOG_SIZE)}
OLD_STAMP_REPR = ("{'version': 1, 'features': ('fg3m', 'fg3_miss', 'fg2m', 'fg2_miss', 'ftm', 'ft_miss', 'orb', "
                  "'drb', 'ast', 'tov', 'stl', 'blk', 'pf'), 'stints': (1700000000000000000, 1234), "
                  "'gamelog': (1700000000123456700, 567)}")


def _cfg(tmp_path, **paths):
    return {"_root": str(tmp_path), "box_counts": list(BOX_COUNTS),
            "paths": {"raw": "data/raw", "stints": "data/stints", **paths}}


def _source_files(tmp_path):
    st = tmp_path / "data" / "stints" / "2026_RS.parquet"
    gl = tmp_path / "data" / "raw" / "gamelog" / "2026_RS.parquet"
    for path, size, ns in ((st, STINTS_SIZE, STINTS_NS), (gl, GAMELOG_SIZE, GAMELOG_NS)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"x" * size)
        os.utime(path, ns=(ns, ns))


def test_the_piece_stamp_is_the_one_the_old_code_wrote(tmp_path):
    _source_files(tmp_path)
    stamp = designs._stamp(2026, "RS", _cfg(tmp_path))
    assert stamp == OLD_STAMP                    # what `_load_piece` compares
    assert repr(stamp) == OLD_STAMP_REPR         # the same keys, in the same order, with the same values
    assert pickle.loads(pickle.dumps(stamp)) == OLD_STAMP


def test_the_stamp_without_source_files(tmp_path):
    assert repr(designs._stamp(2026, "RS", _cfg(tmp_path))) == OLD_STAMP_REPR.split(", 'stints'")[0] + "}"


def test_a_piece_written_by_the_old_code_still_loads(tmp_path):
    """A meta.pkl carrying the old stamp is reused, not rebuilt."""
    _source_files(tmp_path)
    cfg = _cfg(tmp_path)
    folder = tmp_path / "data" / "cache" / "pieces" / "2026_RS"
    folder.mkdir(parents=True)
    with open(folder / "meta.pkl", "wb") as fh:
        pickle.dump({"season": 2026, "phase": "RS", "stamp": dict(OLD_STAMP)}, fh)
    for key in designs.ARRAY_KEYS:
        np.save(folder / f"{key}.npy", np.zeros((2, 3)))
    piece = designs._load_piece(2026, "RS", cfg)
    assert piece is not None and piece["season"] == 2026
    assert set(piece["counters"]) == {"h", "a"} and set(piece["pids"]) == {"h", "a"}
    # and a different box-count list is a different stamp: the piece is rebuilt
    assert designs._load_piece(2026, "RS", {**cfg, "box_counts": BOX_COUNTS[:-1]}) is None


def test_the_pieces_live_under_paths_cache(tmp_path):
    assert designs._pieces_dir(_cfg(tmp_path)) == tmp_path / "data" / "cache" / "pieces"      # the default
    elsewhere = tmp_path / "verify" / "cache"
    assert designs._pieces_dir(_cfg(tmp_path, cache=str(elsewhere))) == elsewhere / "pieces"
