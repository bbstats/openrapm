"""`openrapm wins` (src/openrapm/commands/wins.py): the columns go in, a finished season fits its own win model, a
season in progress reads the latest finished one's, and a partial run on a table without them is refused."""
import numpy as np
import pandas as pd
import pytest

from openrapm import swap_test, wins
from openrapm.commands import wins as command
from openrapm.settings import PRODUCT
from openrapm.simulate import simulate


def _season_files(seed=3):
    stints = simulate(n_seasons=1, n_teams=6, players_per_team=9, games_per_season=60, stints_per_game=(20, 30),
                      seed=seed)["stints"].copy()
    stints["home_team_id"], stints["away_team_id"] = stints.h1 // 1000, stints.a1 // 1000
    stints = swap_test.order_stints(stints)
    return stints, swap_test.stint_durations(stints)


def _table(seasons=(2025, 2026)) -> pd.DataFrame:
    stints, _ = _season_files()
    player_ids = wins.player_index(stints)[0]
    rng = np.random.default_rng(1)
    frames = []
    for season in seasons:
        offense, defense = rng.normal(0.0, 2.0, len(player_ids)), rng.normal(0.0, 1.5, len(player_ids))
        frames.append(pd.DataFrame({"player_id": player_ids, "season": season, "player_name": "P",
                                    "rating_off": offense, "rating_def": defense, "rating_total": offense + defense,
                                    "poss_season": 100.0}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def files(monkeypatch):
    """Every season's stints the same simulated ones; 2026 the latest finished unless a test moves it."""
    seen = []

    def load(cfg, season, phases):
        seen.append(season)
        return _season_files()

    monkeypatch.setattr(command, "season_stints", load)
    monkeypatch.setattr(command, "last_finished", lambda cfg: 2026)
    monkeypatch.setattr(command, "share_played", lambda season, cfg: 1.0)
    return seen


def test_the_columns_go_in_after_the_tables_own_and_the_summary_names_each_season(files):
    table, summary = command.add_wins({}, _table(), PRODUCT)
    assert list(table.columns) == [*_table().columns, *wins.WINS_COLUMNS]
    assert len(table) == len(_table()) and table[wins.WINS_COLUMNS].notna().all().all()
    assert summary.season.tolist() == [2025, 2026] and summary.bar_players.tolist() == [54, 54]
    assert (table.wosa == table.wosa_off + table.wosa_def + table.wosa_pace).all()


def test_the_old_columns_do_not_move(files):
    before = _table()
    table, _ = command.add_wins({}, before, PRODUCT)
    pd.testing.assert_frame_equal(table[before.columns], before)


def test_one_season_can_be_computed_again_and_the_others_keep_theirs(files):
    table, _ = command.add_wins({}, _table(), PRODUCT)
    marked = table.assign(wosa=lambda frame: np.where(frame.season == 2025, 99.0, frame.wosa))
    again, summary = command.add_wins({}, marked, PRODUCT, seasons=[2026])
    assert summary.season.tolist() == [2026]
    assert (again[again.season == 2025].wosa == 99.0).all()
    pd.testing.assert_frame_equal(again[again.season == 2026].reset_index(drop=True),
                                  table[table.season == 2026].reset_index(drop=True))


def test_one_season_on_a_table_without_the_columns_is_refused(files):
    with pytest.raises(SystemExit, match="compute every season first"):
        command.add_wins({}, _table(), PRODUCT, seasons=[2026])


def test_a_table_of_one_season_needs_no_other(files):
    table, summary = command.add_wins({}, _table(seasons=(2026,)), PRODUCT, seasons=[2026])
    assert table.season.unique().tolist() == [2026] and summary.season.tolist() == [2026]


def test_a_finished_season_fits_its_own_win_model_and_one_in_progress_the_latest_finished(files, monkeypatch):
    stints, seconds = _season_files()
    assert command.win_model_for({}, 2025, stints, seconds, ("RS",), {}).season == 2025
    kept = {}
    assert command.win_model_for({}, 2027, stints, seconds, ("RS",), kept).season == 2026
    assert command.win_model_for({}, 2027, stints, seconds, ("RS",), kept).season == 2026
    assert files == [2026]                 # fitted once a run, from the finished season's own stints


def test_a_season_in_progress_reports_no_standings_of_its_own(files):
    _, summary = command.add_wins({}, _table(seasons=(2027,)), PRODUCT)
    assert summary.win_model_season.tolist() == [2026]
    assert summary.net_points_per_win.isna().all() and summary.rating_points_per_win.isna().all()


def test_the_summary_keeps_the_lines_of_seasons_not_computed(tmp_path, monkeypatch):
    monkeypatch.setattr(command, "resolve", lambda cfg, key: tmp_path)
    command.write_wins_summary(pd.DataFrame({"season": [2025, 2026], "bar_total": [1.0, 2.0]}), {}, "trial")
    command.write_wins_summary(pd.DataFrame({"season": [2026], "bar_total": [9.0]}), {}, "trial")
    written = pd.read_csv(tmp_path / "csv" / "wins_trial_seasons.csv")
    assert written.season.tolist() == [2025, 2026] and written.bar_total.tolist() == [1.0, 9.0]


def test_the_command_refuses_a_table_that_is_not_there(tmp_path, monkeypatch):
    monkeypatch.setattr(command, "resolve", lambda cfg, key: tmp_path)
    with pytest.raises(SystemExit, match="does not exist"):
        command.run_wins({}, PRODUCT, "absent")


def test_the_command_writes_the_table_in_place(tmp_path, monkeypatch, files, capsys):
    monkeypatch.setattr(command, "resolve", lambda cfg, key: tmp_path)
    _table().to_parquet(tmp_path / "season_ratings_trial.parquet", index=False)
    assert command.run_wins({}, PRODUCT, "trial") == 0
    written = pd.read_parquet(tmp_path / "season_ratings_trial.parquet")
    assert "wosa" in written.columns and (tmp_path / "csv" / "wins_trial_seasons.csv").exists()
    assert "2026: " in capsys.readouterr().out
