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
                                    "poss_season": 100.0, "prior_off": 0.0, "prior_def": 0.0, "u_off": 0.0,
                                    "u_def": 0.0, "poss_off": 100.0, "poss_def": 100.0}))
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
    monkeypatch.setattr(command, "has_playoff_stints", lambda season, cfg: False)
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


def _regular_rows(table: pd.DataFrame, shift: float) -> pd.DataFrame:
    """A regular-season table: the rating `shift` higher for everyone, on offense."""
    rows = table.assign(rating_off=table.rating_off + shift, rating_total=table.rating_total + shift)
    return rows[command.REGULAR_COLUMNS]


def test_a_season_without_playoff_games_reads_its_own_rows_even_with_older_ones_stored(files):
    own = _table(seasons=(2027,))
    assert command.rating_rows({}, 2027, own, _regular_rows(own, 5.0)) is own


def test_a_season_with_playoff_games_reads_its_regular_season_rows(files, monkeypatch):
    monkeypatch.setattr(command, "has_playoff_stints", lambda season, cfg: True)
    own = _table(seasons=(2026,))
    regular = _regular_rows(own, 0.0)
    regular.loc[regular.player_id == regular.player_id.iloc[0], "rating_off"] += 6.0
    regular.loc[regular.player_id == regular.player_id.iloc[0], "rating_total"] += 6.0
    plain, _ = command.add_wins({}, own, PRODUCT, regular=_regular_rows(own, 0.0))
    locked, _ = command.add_wins({}, own, PRODUCT, regular=regular)
    first = own.player_id.iloc[0]
    assert locked.set_index("player_id").loc[first, "wosa"] > plain.set_index("player_id").loc[first, "wosa"]
    pd.testing.assert_frame_equal(locked[own.columns], own)            # the published rating is not touched


def test_a_season_with_playoff_games_and_no_regular_season_rating_stops_and_says_how_to_make_one(files, monkeypatch):
    monkeypatch.setattr(command, "has_playoff_stints", lambda season, cfg: True)
    with pytest.raises(SystemExit, match="--rebuild-regular"):
        command.add_wins({}, _table(seasons=(2026,)), PRODUCT)


def test_the_lock_is_the_last_rows_stored_before_the_first_playoff_game(files, monkeypatch):
    """Each morning without a playoff game stores the season's rows; the first morning with one leaves them, so the wins
    stop moving however the rating moves after it."""
    morning_1 = _regular_rows(_table(seasons=(2027,)), 1.0)
    morning_1.loc[:, "rating_off"] = 1.0
    stored = command.with_seasons_so_far({}, morning_1, None)
    morning_2 = morning_1.assign(rating_off=2.0)
    stored = command.with_seasons_so_far({}, morning_2, stored)
    assert (stored.rating_off == 2.0).all()
    monkeypatch.setattr(command, "has_playoff_stints", lambda season, cfg: True)
    playoffs = morning_1.assign(rating_off=9.0)
    assert command.with_seasons_so_far({}, playoffs, stored) is stored
    assert (command.rating_rows({}, 2027, playoffs, stored).rating_off == 2.0).all()


def test_the_regular_rows_of_other_seasons_stay_when_one_season_is_stored(files):
    older = _regular_rows(_table(seasons=(2026,)), 0.0)
    now = _regular_rows(_table(seasons=(2027,)), 0.0)
    stored = command.with_seasons_so_far({}, now, older)
    assert sorted(stored.season.unique()) == [2026, 2027] and len(stored) == len(older) + len(now)


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


def test_regular_ratings_rates_only_finished_seasons_with_playoff_games(monkeypatch):
    from openrapm.commands import build
    asked = []

    def table_of(cfg, settings, seasons, name, jobs):
        asked.extend(seasons)
        return pd.DataFrame({column: [1.0] * len(seasons) for column in command.REGULAR_COLUMNS}).assign(season=seasons)

    monkeypatch.setattr(build, "regular_season_table", table_of)
    monkeypatch.setattr(build, "last_finished", lambda cfg: 2026)
    monkeypatch.setattr(build, "has_playoff_stints", lambda season, cfg: season != 2025)
    table = pd.DataFrame({"season": [2024, 2025, 2026, 2027]})
    stored = pd.DataFrame({column: [7.0] for column in command.REGULAR_COLUMNS}).assign(season=[2027])
    result = build.regular_ratings({}, PRODUCT, table, stored, "trial")
    assert asked == [2024, 2026]                                       # 2025 no playoffs here, 2027 locked already
    assert sorted(result.season.tolist()) == [2024, 2026, 2027]
    assert result[result.season == 2027].rating_off.tolist() == [7.0]  # a locked season is left as it stood


def test_a_stopped_run_goes_on_from_the_seasons_it_finished(tmp_path):
    from openrapm.commands import build
    (tmp_path / "2024.parquet").write_bytes(b"done")
    (tmp_path / "2025.parquet.part").write_bytes(b"cut short")        # a season stopped mid-write counts as not done
    assert build.seasons_still_to_rate(tmp_path, [2024, 2025, 2026]) == [2025, 2026]


def test_a_child_process_rates_its_seasons_and_leaves_the_rest_to_its_parent(monkeypatch, tmp_path):
    from openrapm.commands import build, parallel
    rated = []
    monkeypatch.setattr(parallel, "is_child", lambda: True)
    monkeypatch.setattr(build, "regular_parts_folder", lambda cfg, name: tmp_path)
    monkeypatch.setattr(build, "rate_regular_seasons", lambda cfg, settings, seasons, folder: rated.extend(seasons))
    assert command.run_wins({}, PRODUCT, "trial", [2024, 2028], rebuild_regular=True, jobs=1) == 0
    assert rated == [2024, 2028]


def test_child_processes_rate_the_regular_seasons_only_at_the_shipped_settings(monkeypatch, tmp_path):
    """They run `openrapm wins`, which reads the shipped settings and takes no flags, so a build of a new line named by
    flags must rate its regular seasons itself or they come out on the shipped line."""
    from dataclasses import replace

    from openrapm.commands import build, parallel

    class Rated(Exception):
        pass

    def rated_by(who):
        def stop(*args, **kwargs):
            raise Rated(who)
        return stop

    monkeypatch.setattr(build, "regular_parts_folder", lambda cfg, name: tmp_path)
    monkeypatch.setattr(parallel, "affordable_jobs", lambda jobs: jobs)
    monkeypatch.setattr(parallel, "run_children", rated_by("children"))
    monkeypatch.setattr(build, "rate_regular_seasons", rated_by("this process"))
    cfg = {"paths": {"outputs": str(tmp_path)}}
    with pytest.raises(Rated, match="children"):
        build.regular_season_table(cfg, PRODUCT, [2024, 2025], "trial", jobs=2)
    new_line = replace(PRODUCT, prior=replace(PRODUCT.prior, second_prior="trial"))
    with pytest.raises(Rated, match="this process"):
        build.regular_season_table(cfg, new_line, [2024, 2025], "trial", jobs=2)


def test_the_designs_are_cut_for_the_rated_season_alone():
    from openrapm.commands import build

    class Designs:
        cfg = {}

        def season(self, season, target):
            return f"whole {season} {target}"

    cut = build.RegularSeasonDesigns(Designs(), 2026, {"pts": "cut 2026 pts"})
    assert cut.season(2026, "pts") == "cut 2026 pts" and cut.season(2025, "pts") == "whole 2025 pts"
