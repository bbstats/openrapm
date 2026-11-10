"""`openrapm data check-games`: the morning job rates the season in progress only when every game of its game log has
stints and the right points (data/stints.check_games), so a game that failed to load can no longer go missing from
the published table with nobody told.  Small files under a temporary folder, as tests/test_seasons.py builds them."""
import argparse
import json

import pandas as pd
import pytest
from test_seasons import _gamelog, _write

from openrapm.commands import data as data_command
from openrapm.data import ingest, stints

GAMES = [f"00226000{k:02d}" for k in range(1, 5)]


def _season(tmp_path, logged=GAMES, built=GAMES, points=None, errors=None, playoff_log=True) -> dict:
    """A tree where 2026 is finished and 2027 in progress: 2027's game log holds `logged`, its stints `built`, its
    diagnostics each built game's points ({game: (parsed home, parsed away, box home, box away)}, 100-98 both ways by
    default) and an `error` row for each game of `errors`.  Every logged game has its three kinds of raw file."""
    cfg = _write(tmp_path, 2026, _gamelog(16))
    cfg = {**cfg, "first_season": 2026, "last_season": 2027, "paths": {**cfg["paths"], "stints": "data/stints"}}
    raw, folder = tmp_path / "data" / "raw", tmp_path / "data" / "stints"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"GAME_ID": [game for game in logged for _ in range(2)]}).to_parquet(
        raw / "gamelog" / "2027_RS.parquet")
    if playoff_log:                # fetched before the playoffs: on disk, every column and no rows
        pd.DataFrame({"TEAM_ID": pd.Series([], dtype=int), "GAME_ID": pd.Series([], dtype=str),
                      "WL": pd.Series([], dtype=str)}).to_parquet(raw / "gamelog" / "2027_PO.parquet")
    pd.DataFrame({"game_id": [game for game in built for _ in range(3)]}).to_parquet(folder / "2027_RS.parquet")
    rows = []
    for game in built:
        home, away, box_home, box_away = (points or {}).get(game, (100, 98, 100, 98))
        rows.append(dict(game_id=game, pts_home=home, pts_away=away, box_pts_home=box_home, box_pts_away=box_away))
    for game, error in (errors or {}).items():
        rows.append(dict(game_id=game, error=error))
    pd.DataFrame(rows).to_parquet(folder / "2027_RS_diag.parquet")
    for game in logged:
        for kind, name in (("pbp", game), ("box", game), ("periodbox", f"{game}_1"), ("periodbox", f"{game}_5")):
            path = raw / kind / "2027" / f"{name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"raw")
    return cfg


def _run(cfg, season=2027) -> int:
    return data_command.run_check_games(argparse.Namespace(season=season), cfg)


def _raw_files(tmp_path, game) -> list:
    return sorted(path.name for path in (tmp_path / "data" / "raw").rglob(f"{game}*.parquet"))


def test_a_season_whose_every_game_was_built_passes_and_leaves_a_record(tmp_path, capsys):
    cfg = _season(tmp_path)
    assert _run(cfg) == 0
    record = json.loads((tmp_path / "data" / "stints" / "2027_games_checked.json").read_text(encoding="utf-8"))
    assert record["games"] == {"RS": GAMES}
    assert "2027 RS: 4 games" in capsys.readouterr().out
    assert _raw_files(tmp_path, GAMES[0]) == [f"{GAMES[0]}.parquet"] * 2 + [f"{GAMES[0]}_1.parquet",
                                                                           f"{GAMES[0]}_5.parquet"]


def test_a_game_that_failed_to_parse_stops_the_morning_and_only_its_raw_files_are_removed(tmp_path, capsys):
    """The stints step drops such a game with one printed line and goes on; removed, its files are fetched again by the
    next `data ingest`, which fetches whatever is missing."""
    cfg = _season(tmp_path, built=GAMES[:3], errors={GAMES[3]: "ValueError('empty pbp')"})
    assert _run(cfg) == 1
    out = capsys.readouterr().out
    assert GAMES[3] in out and "empty pbp" in out
    assert _raw_files(tmp_path, GAMES[3]) == [] and len(_raw_files(tmp_path, GAMES[0])) == 4
    assert not (tmp_path / "data" / "stints" / "2027_games_checked.json").exists()


def test_a_game_in_the_log_with_no_stints_and_no_error_stops_it_too(tmp_path, capsys):
    cfg = _season(tmp_path, built=GAMES[:3])
    assert _run(cfg) == 1 and "no stints" in capsys.readouterr().out
    assert _raw_files(tmp_path, GAMES[3]) == []


def test_a_game_whose_points_differ_from_its_box_score_stops_it_unless_the_config_accepts_it(tmp_path, capsys):
    """A play-by-play cut off before the game ended parses, and scores fewer points than the box score holds."""
    cfg = _season(tmp_path, points={GAMES[1]: (100, 91, 100, 98)})
    assert _run(cfg) == 1
    assert "91" in capsys.readouterr().out and _raw_files(tmp_path, GAMES[1]) == []
    accepted = _season(tmp_path / "accepted", points={GAMES[1]: (100, 91, 100, 98)})
    assert _run({**accepted, "points_mismatch_accepted": [GAMES[1]]}) == 0
    assert len(_raw_files(tmp_path / "accepted", GAMES[1])) == 4


def test_a_game_log_that_lost_a_game_stops_it_even_when_another_was_added(tmp_path, capsys):
    """Containment, not a count: one game gone and one new leaves the count where it was."""
    cfg = _season(tmp_path, logged=GAMES[:3], built=GAMES[:3])
    assert _run(cfg) == 0
    swapped = [GAMES[0], GAMES[1], GAMES[3]]
    cfg = _season(tmp_path, logged=swapped, built=swapped)
    assert _run(cfg) == 1
    out = capsys.readouterr().out
    assert GAMES[2] in out and "2027_games_checked.json" in out
    assert len(_raw_files(tmp_path, GAMES[0])) == 4                        # nothing removed: no file is at fault


def test_the_playoffs_before_they_start_are_skipped_and_a_phase_never_logged_is_too(tmp_path):
    assert [check.phase for check in stints.check_games(2027, _season(tmp_path))] == ["RS"]
    without = _season(tmp_path / "without", playoff_log=False)
    assert [check.phase for check in stints.check_games(2027, without)] == ["RS"]


def test_a_finished_season_is_refused_and_nothing_of_it_is_removed(tmp_path):
    """Five finished regular seasons hold a game whose parsed points differ from its box score; their raw files must
    never be removed by a check meant for last night's games."""
    cfg = _season(tmp_path)
    with pytest.raises(SystemExit, match="in progress"):
        _run(cfg, season=2026)


def test_the_job_is_told_3_once_a_team_has_won_the_title(tmp_path, capsys):
    """`openrapm data in-progress`: the morning job stops on 3, since a finished season is rated with every other
    season by a full build.  Fifteen playoff wins is still a season in progress; the sixteenth ends it."""
    def asked(root, wins: int, season: int = 2027) -> int:
        cfg = _write(root, 2026, _gamelog(16))
        _write(root, 2027, _gamelog(wins))
        return data_command.run_in_progress(argparse.Namespace(season=season),
                                            {**cfg, "first_season": 2026, "last_season": 2027})
    assert asked(tmp_path / "june_14", wins=15) == 0 and "in progress" in capsys.readouterr().out
    assert asked(tmp_path / "june_15", wins=16) == 3 and "Finals are over" in capsys.readouterr().out
    assert asked(tmp_path / "october", wins=15, season=2026) == 3            # a season long finished
    with pytest.raises(SystemExit, match="last_season"):
        asked(tmp_path / "typo", wins=15, season=2028)


def test_forgetting_a_game_removes_its_play_by_play_its_box_score_and_its_period_box_scores(tmp_path):
    cfg = _season(tmp_path)
    removed = ingest.forget_game(GAMES[2], cfg)
    assert sorted(path.name for path in removed) == sorted(
        [f"{GAMES[2]}.parquet"] * 2 + [f"{GAMES[2]}_1.parquet", f"{GAMES[2]}_5.parquet"])
    assert _raw_files(tmp_path, GAMES[2]) == [] and len(_raw_files(tmp_path, GAMES[1])) == 4
    assert ingest.forget_game(GAMES[2], cfg) == []                         # nothing left to remove
