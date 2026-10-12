"""The data layer reads each box-score, bio and shots table once per process, and keys each cache on the file or
folder it read.

Until 2026-10-11 the box scores were cached on (season, phase) alone, so a run that pointed paths.raw at another tree
after reading the real one got the real tree's rows back.  Each cache is now `functools.lru_cache` on a loader that
is handed the resolved path, so two trees never share an entry.
"""
import pandas as pd

from openrapm.data.bio import player_bio
from openrapm.data.box_scores import season_box
from openrapm.data.shots import load_shots

GAME = "0022500001"
GAMELOG_COUNTS = ("MIN", "FG3M", "FG3A", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB", "AST", "TOV", "STL", "BLK", "PF",
                  "PTS")


def _write(frame: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _tree(root, player_id: int) -> dict:
    """A data tree under `root` with one row for `player_id` in each table: the 2026 regular-season game log, the bio
    table and the 2026 regular-season shots table.  Returns a configuration that points at it."""
    _write(pd.DataFrame([{"GAME_ID": GAME, "PLAYER_ID": player_id, "TEAM_ID": 1, "PLAYER_NAME": f"Player {player_id}",
                          "season": 2026, "phase": "RS", **{count: 1.0 for count in GAMELOG_COUNTS}}]),
           root / "data" / "raw" / "gamelog" / "2026_RS.parquet")
    _write(pd.DataFrame({"player_id": [player_id], "height": [80.0], "weight": [220.0], "draft_pick": [1.0]}),
           root / "data" / "cache" / "bio.parquet")
    _write(pd.DataFrame({"game_id": [GAME], "player_id": [player_id], "half": ["A"]}),
           root / "data" / "stints" / "2026_RS_shots.parquet")
    return {"_root": str(root), "paths": {"raw": "data/raw", "cache": "data/cache", "stints": "data/stints"}}


def _players(cfg: dict) -> tuple:
    return (season_box([2026], ["RS"], cfg).player_id.tolist(), player_bio(cfg).player_id.tolist(),
            load_shots([2026], cfg, ("RS",)).player_id.tolist())


def test_each_tree_gets_its_own_tables_even_after_another_was_read(tmp_path):
    first, second = _tree(tmp_path / "first", 101), _tree(tmp_path / "second", 202)
    assert _players(first) == ([101], [101], [101])
    assert _players(second) == ([202], [202], [202])
    assert _players(first) == ([101], [101], [101])


def test_a_table_is_read_once(tmp_path):
    cfg = _tree(tmp_path, 101)
    box, bio, shots = _players(cfg)
    (tmp_path / "data" / "raw" / "gamelog" / "2026_RS.parquet").unlink()
    (tmp_path / "data" / "stints" / "2026_RS_shots.parquet").unlink()
    assert season_box([2026], ["RS"], cfg).player_id.tolist() == box
    assert load_shots([2026], cfg, ("RS",)).player_id.tolist() == shots
    assert player_bio(cfg) is player_bio(cfg)
