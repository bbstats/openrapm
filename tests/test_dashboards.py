"""The league dashboards (src/openrapm/data/dashboards.py): what is asked of nba_api, what is cached, what a failure and
an empty season look like, and how much of the rated players a file covers.  The endpoint is made up: tests cannot
reach the network."""
import numpy as np
import pandas as pd
import pytest

from openrapm.commands import data as data_command
from openrapm.data import dashboards
from openrapm.data.ingest import OfflineFetch


@pytest.fixture
def cfg(tmp_path):
    return {"_root": str(tmp_path), "last_season": 2026, "paths": {"raw": str(tmp_path / "raw"),
                                                                    "outputs": str(tmp_path / "outputs"),
                                                                    "panel": str(tmp_path / "panel.parquet")}}


class Endpoint:
    """A made-up nba_api endpoint class: remembers how it was called, answers with `answers[season]` (a frame), an
    empty frame for 'none', or raises for 'broken'."""
    answers: dict = {}
    calls: list = []

    def __init__(self, **arguments):
        type(self).calls.append(arguments)
        answer = type(self).answers.get(arguments["season"], "none")
        if isinstance(answer, str) and answer == "broken":
            raise KeyError("resultSet")
        self.frame = pd.DataFrame(columns=["PLAYER_ID", "DEFLECTIONS"]) if isinstance(answer, str) else answer

    def get_data_frames(self):
        return [self.frame]


@pytest.fixture
def endpoint(monkeypatch):
    Endpoint.answers, Endpoint.calls = {}, []
    monkeypatch.setattr(dashboards, "_endpoint_class", lambda name: Endpoint)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    return Endpoint


def hustle() -> dashboards.Pull:
    return dashboards.pulls(("hustle",))[0]


def test_the_pulls_are_the_four_sources_with_their_own_arguments():
    every = dashboards.pulls()
    assert len(every) == 34 and {pull.source for pull in every} == set(dashboards.SOURCES)
    assert len({(pull.source, pull.kind) for pull in every}) == 34, "one folder each"
    assert [pull.kind for pull in dashboards.pulls(("matchups",))] == ["season"]
    defend = [pull for pull in every if pull.endpoint == "LeagueDashPtDefend"]
    assert {pull.id_column for pull in defend} == {"CLOSE_DEF_PERSON_ID"} and len(defend) == 4
    assert len([pull for pull in every if pull.source == "synergy"]) == 2 * len(dashboards.PLAY_TYPES)
    matchups = dashboards.pulls(("matchups",))[0]
    assert matchups.season_type_argument == "season_type_playoffs"


def test_a_fetch_asks_for_the_season_and_phase_and_is_cached(cfg, endpoint):
    endpoint.answers = {"2025-26": pd.DataFrame({"PLAYER_ID": [1, 2], "DEFLECTIONS": [10, 20]})}
    pull = hustle()
    frame, cached = dashboards.fetch_dashboard(pull, 2026, "PO", cfg)
    assert not cached and len(frame) == 2
    assert endpoint.calls[0]["season"] == "2025-26" and endpoint.calls[0]["season_type_all_star"] == "Playoffs"
    assert endpoint.calls[0]["per_mode_time"] == "Totals"
    assert dashboards.dashboard_path(cfg, pull, 2026, "PO").parts[-3:] == ("hustle", "players", "2026_PO.parquet")
    again, cached = dashboards.fetch_dashboard(pull, 2026, "PO", cfg)
    assert cached and len(endpoint.calls) == 1 and again.equals(frame)


def test_a_season_with_no_rows_is_cached_empty_and_a_broken_answer_is_not(cfg, endpoint):
    endpoint.answers = {"2024-25": "none", "2023-24": "broken"}
    wanted = [(hustle(), [2025, 2024])]
    records = dashboards.pull_all(cfg, wanted, phases=("RS",), delay=0.0, tries=2)
    status = {record["season"]: record["status"] for record in records}
    assert status == {2025: "empty", 2024: "failed"}
    assert "resultSet" in records[1]["error"] and dashboards.dashboard_path(cfg, hustle(), 2025, "RS").exists()
    assert not dashboards.dashboard_path(cfg, hustle(), 2024, "RS").exists()
    assert records[0]["rows"] == 0
    second = dashboards.pull_all(cfg, wanted, phases=("RS",), delay=0.0, tries=2)
    assert second[0]["status"] == "cached" and second[1]["status"] == "failed"


def test_offline_a_missing_file_stops_the_pull_instead_of_being_recorded_as_failed(cfg, endpoint, monkeypatch):
    monkeypatch.setenv("OPENRAPM_OFFLINE", "1")
    with pytest.raises(OfflineFetch, match="hustle players dashboard 2026 RS"):
        dashboards.pull_all(cfg, [(hustle(), [2026])], phases=("RS",), delay=0.0)
    assert endpoint.calls == []


def test_a_probe_asks_the_season_before_the_start_the_start_itself_the_break_and_the_latest():
    assert dashboards.probe_seasons(hustle(), 2026) == [2015, 2016, 2018, 2026]
    tracking = dashboards.pulls(("tracking",))[0]
    assert dashboards.probe_seasons(tracking, 2026) == [2013, 2014, 2018, 2026]
    matchups = dashboards.pulls(("matchups",))[0]
    assert dashboards.probe_seasons(matchups, 2026) == [2017, 2018, 2026]       # the break's season is its first


def test_coverage_is_the_share_of_the_rated_players_and_of_their_possessions_a_file_holds():
    panel = pd.DataFrame({"player_id": [1, 2, 3, 1, 2, 3, 9], "season": [2026] * 6 + [2025],
                          "side": ["O", "O", "O", "D", "D", "D", "O"],
                          "possessions": [3000.0, 1000.0, 0.0] * 2 + [1.0]})
    frame = pd.DataFrame({"PLAYER_ID": [1, 7], "DEFLECTIONS": [5, 6]})
    records = [dict(pull=hustle(), season=2026, phase="RS", status="fetched", rows=2, error="", frame=frame),
               dict(pull=hustle(), season=2026, phase="PO", status="failed", rows=0, error="timeout", frame=None),
               dict(pull=hustle(), season=2025, phase="RS", status="empty", rows=0, error="", frame=frame.iloc[:0])]
    table = dashboards.coverage(records, panel).set_index(["season", "phase"])
    row = table.loc[(2026, "RS")]
    # a player with no possessions is not rated, and player 7 is not in the panel
    assert (row.rated_players, row.in_file) == (2, 1)
    assert row.share_players == pytest.approx(0.5) and row.share_possessions == pytest.approx(0.75)
    assert row.n_columns == 2
    assert table.loc[(2026, "PO")].status == "failed" and np.isnan(table.loc[(2026, "PO")].share_players)
    assert np.isnan(table.loc[(2025, "RS")].share_possessions)


def test_fields_list_each_pulls_columns_once(endpoint, cfg):
    frame = pd.DataFrame({"PLAYER_ID": [1], "DEFLECTIONS": [5]})
    records = [dict(pull=hustle(), season=2016, phase="RS", status="empty", rows=0, error="", frame=frame.iloc[:0]),
               dict(pull=hustle(), season=2018, phase="RS", status="fetched", rows=1, error="", frame=frame),
               dict(pull=hustle(), season=2026, phase="RS", status="fetched", rows=1, error="", frame=frame)]
    listed = dashboards.fields(records)
    assert listed.season.tolist() == [2018] and listed.fields.iloc[0] == "PLAYER_ID, DEFLECTIONS"


def test_the_command_probes_writes_the_coverage_and_rejects_an_unknown_source(cfg, endpoint, tmp_path, capsys):
    pd.DataFrame({"player_id": [1, 2], "season": [2026, 2026], "side": ["O", "O"], "possessions": [100.0, 50.0]}
                 ).to_parquet(tmp_path / "panel.parquet")
    endpoint.answers = {"2025-26": pd.DataFrame({"PLAYER_ID": [1], "DEFLECTIONS": [3]})}
    args = data_command.argparse.Namespace(probe=True, sources="hustle", phases="RS", delay=0.0, force=False,
                                           first_season=None, last_season=None)
    assert data_command.run_dashboards(args, cfg) == 0
    out = capsys.readouterr().out
    assert "wrote" in out and (tmp_path / "outputs" / "csv" / "dashboard_probe.csv").exists()
    table = pd.read_csv(tmp_path / "outputs" / "csv" / "dashboard_probe.csv")
    assert sorted(table.season) == [2015, 2016, 2018, 2026]
    assert table[table.season == 2026].share_possessions.iloc[0] == pytest.approx(100 / 150)
    args.sources = "hustle,bogus"
    with pytest.raises(SystemExit, match="bogus"):
        data_command.run_dashboards(args, cfg)


class GameEndpoint:
    """A made-up BoxScoreHustleV2: answers `answers[game_id]`, raises for 'broken', remembers the games asked."""
    answers: dict = {}
    calls: list = []

    def __init__(self, game_id, timeout):
        type(self).calls.append(game_id)
        answer = type(self).answers.get(game_id)
        if isinstance(answer, str):
            raise KeyError("resultSet")
        self.frame = answer

    def get_data_frames(self):
        return [self.frame, pd.DataFrame()]


def test_hustle_games_are_fetched_once_each_and_a_broken_game_is_listed_not_cached(cfg, monkeypatch):
    GameEndpoint.calls = []
    GameEndpoint.answers = {"0022500001": pd.DataFrame({"personId": [1, 2], "deflections": [3, 0]}),
                            "0022500002": pd.DataFrame({"personId": [], "deflections": []}),
                            "0042500101": "broken"}
    monkeypatch.setattr(dashboards, "_endpoint_class", lambda name: GameEndpoint)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    monkeypatch.setattr(dashboards, "season_game_ids",
                        lambda cfg, season, phases: ["0022500001", "0022500002", "0042500101"])
    out = dashboards.fetch_hustle_games(cfg, [2026], log_every=0)
    assert (out["games"], out["fetched"], out["empty"]) == (3, 2, 1)
    assert [game for game, _ in out["failed"]] == ["0042500101"]
    path = dashboards.hustle_game_path(cfg, "0022500001")
    assert path.parts[-2:] == ("2026", "0022500001.parquet")
    assert pd.read_parquet(path).deflections.tolist() == [3, 0]
    assert not dashboards.hustle_game_path(cfg, "0042500101").exists()
    GameEndpoint.calls = []
    again = dashboards.fetch_hustle_games(cfg, [2026], log_every=0)
    assert again["fetched"] == 0 and GameEndpoint.calls == ["0042500101"] * 4      # only the missing game, retried


def test_offline_a_missing_hustle_game_stops_the_run(cfg, monkeypatch):
    monkeypatch.setenv("OPENRAPM_OFFLINE", "1")
    monkeypatch.setattr(dashboards, "season_game_ids", lambda cfg, season, phases: ["0022500001"])
    with pytest.raises(OfflineFetch):
        dashboards.fetch_hustle_games(cfg, [2026], log_every=0)
