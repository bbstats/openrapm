"""data/public_ratings.py: the three outside systems' pages read, checked and stored, on invented pages.

Every row below is made up, in each system's own format: the repository is public and the real ratings are their
makers'.  No test reaches the network (conftest.py blocks it); `requests.get` and `fetch` are replaced.
"""
import gzip
import json

import pandas as pd
import pytest
import requests

from openrapm.commands import cli
from openrapm.data import public_ratings
from openrapm.data.ingest import OFFLINE_ENV, OfflineFetch
from openrapm.data.public_ratings import FormatChanged, PublicRatings

# three invented players: (NBA id, offense, defense), defense positive good
PLAYERS = [(900001, 3.5, 8.0), (900002, -1.2, 0.4), (900003, 0.0, -2.1)]


@pytest.fixture(autouse=True)
def small_tables(monkeypatch):
    """Three players are a table here; the real minimum has its own test."""
    monkeypatch.setattr(public_ratings, "MIN_ROWS", 3)


def xrapm_page(players=PLAYERS, styled=False, date="2026-11-04", total_sign=1.0) -> str:
    """xrapm.com's table: Defense printed as points allowed, each number followed by a percentile."""
    cell = '<td style="padding-left:2px;">' if styled else "<td>"
    rows = "".join(
        f'{cell}<a href="https://xrapm.com/player_pages/{player}.html">Player {player}</td>\n{cell}TM</td>\n'
        f"{cell}{offense:.1f} (50)</td>\n{cell}{-defense:.1f} (50)</td>\n"
        f"{cell}{total_sign * (offense + defense):.1f} (50)</td>\n</tr>\n" for player, offense, defense in players)
    header = f"<h6>Metrics updated on: {date}</h6>" if date else "<h6>3-year RAPM</h6>"
    return f"<html>{header}<table><tbody>\n{rows}</tbody></table></html>"


def darko_page(players=PLAYERS, season=2027, through="2026-11-04", missing=()) -> str:
    """darko.app's __data.json: one flat list, every value written once and referred to by its position."""
    columns = {"nba_id": [player for player, _, _ in players], "season": [season] * len(players),
               "o_dpm": [None if player in missing else offense for player, offense, _ in players],
               "d_dpm": [defense for _, _, defense in players],
               "dpm": [round(offense + defense, 2) for _, offense, defense in players]}
    flat = [None, None]

    def put(value) -> int:
        flat.append(value)
        return len(flat) - 1

    names = [put(name) for name in columns]
    lists = [put([-1 if value is None else put(value) for value in values]) for values in columns.values()]
    flat[1] = {"keys": put(names), "values": put(lists)}
    flat[0] = {"players": 1, "ratingsThrough": put(through)}
    return json.dumps({"type": "data", "nodes": [None, {"type": "data", "data": flat}]})


PAGES = {"xrapm": xrapm_page, "rapm_3y": lambda: xrapm_page(styled=True, date=None), "darko": darko_page}
EXPECTED = pd.DataFrame(PLAYERS, columns=["player_id", "offense", "defense"])


# ------------------------------------------------------------------------------------ the two parsers
def test_xrapm_defense_is_flipped_to_positive_good_and_the_date_read():
    ratings = public_ratings.parse_xrapm(xrapm_page(), 2027)
    pd.testing.assert_frame_equal(ratings.table, EXPECTED)
    assert ratings.as_of == "2026-11-04" and ratings.pulled == ""


def test_the_three_year_page_has_styled_cells_and_no_date():
    ratings = public_ratings.parse_xrapm(xrapm_page(styled=True, date=None), 2027)
    pd.testing.assert_frame_equal(ratings.table, EXPECTED)
    assert ratings.as_of is None


def test_a_total_that_is_not_offense_plus_defense_is_a_changed_page():
    """If xrapm.com printed Defense as positive good, its Total would no longer be Offense minus that column."""
    with pytest.raises(FormatChanged, match="total misses offense"):
        public_ratings.parse_xrapm(xrapm_page(total_sign=-1.0), 2027)


def test_darko_reads_its_columns_through_the_flat_list_and_leaves_out_a_player_without_a_rating():
    ratings = public_ratings.parse_darko(darko_page(), 2027)
    pd.testing.assert_frame_equal(ratings.table, EXPECTED)
    assert ratings.as_of == "2026-11-04"
    four = [*PLAYERS, (900004, 1.0, 1.0)]
    assert list(public_ratings.parse_darko(darko_page(four, missing={900004}), 2027).table.player_id) == [
        900001, 900002, 900003]


def test_another_seasons_table_is_refused():
    with pytest.raises(FormatChanged):
        public_ratings.parse_darko(darko_page(season=2026), 2027)


@pytest.mark.parametrize("parse", [public_ratings.parse_xrapm, public_ratings.parse_darko])
@pytest.mark.parametrize("page", ["", "<html>Service unavailable</html>", '{"nodes": [], "rows": 3}'])
def test_a_page_that_is_not_the_table_is_a_changed_format_and_nothing_else(parse, page):
    with pytest.raises(FormatChanged):
        parse(page, 2027)


# ------------------------------------------------------------------------------------ the checks
def test_the_real_minimum_is_300_players(monkeypatch):
    monkeypatch.setattr(public_ratings, "MIN_ROWS", 300)
    with pytest.raises(FormatChanged, match="3 rated players read, under the 300"):
        public_ratings.parse_darko(darko_page(), 2027)


def test_a_player_listed_twice_or_a_rating_beyond_25_is_refused():
    with pytest.raises(FormatChanged, match="900001 is listed twice"):
        public_ratings.parse_darko(darko_page([*PLAYERS, PLAYERS[0]]), 2027)
    with pytest.raises(FormatChanged, match="beyond 25"):
        public_ratings.parse_darko(darko_page([*PLAYERS, (900004, 40.0, 1.0)]), 2027)


# ------------------------------------------------------------------------------------ fetch, pull, latest
class Response:
    def __init__(self, body: bytes):
        self.content = body

    def raise_for_status(self) -> None:
        pass


def test_fetch_makes_one_request_that_names_openrapm_and_reads_gzip(monkeypatch, tmp_path):
    seen = []

    def get(address, headers, timeout):
        seen.append((address, headers["User-Agent"], timeout))
        return Response(gzip.compress("Dončić".encode()))

    monkeypatch.setattr(requests, "get", get)
    text = public_ratings.fetch(public_ratings.SYSTEMS["darko"], 2027, tmp_path)
    assert text == "Dončić"
    assert seen == [("https://www.darko.app/__data.json", public_ratings.USER_AGENT, public_ratings.TIMEOUT)]
    assert "OpenRAPM" in public_ratings.USER_AGENT and "openrapm.com" in public_ratings.USER_AGENT


def _cfg(tmp_path) -> dict:
    return {"_root": str(tmp_path), "paths": {"public_ratings": "data/public_ratings"}}


def test_a_pull_stores_every_system_and_a_system_that_fails_keeps_its_last_pull(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(public_ratings, "fetch", lambda system, season, folder: PAGES[system.key]())
    first = public_ratings.pull(cfg, 2027, "2026-11-05", pause=0)
    assert [(record["key"], record["status"], record["rows"]) for record in first] == [
        (key, "pulled", 3) for key in public_ratings.SYSTEMS]
    assert {record["key"]: record["as_of"] for record in first} == {
        "xrapm": "2026-11-04", "rapm_3y": None, "darko": "2026-11-04"}

    def second_day(system, season, folder):
        if system.key == "darko":
            raise requests.ConnectionError("no answer")
        return "<html>a new design</html>" if system.key == "rapm_3y" else PAGES[system.key]()

    monkeypatch.setattr(public_ratings, "fetch", second_day)
    second = {record["key"]: record for record in public_ratings.pull(cfg, 2027, "2026-11-06", pause=0)}
    assert second["darko"]["status"] == "failed" and "ConnectionError" in second["darko"]["error"]
    assert second["rapm_3y"]["status"] == "failed" and second["xrapm"]["status"] == "pulled"
    folder = tmp_path / "data" / "public_ratings" / "2027"
    assert sorted(path.name for path in (folder / "rapm_3y").iterdir()) == ["2026-11-05.parquet",
                                                                           "2026-11-06.unparsed"]
    assert sorted(path.name for path in (folder / "darko").iterdir()) == ["2026-11-05.parquet"]

    stored = public_ratings.latest(cfg, 2027)
    assert {key: ratings.pulled for key, ratings in stored.items()} == {
        "xrapm": "2026-11-06", "rapm_3y": "2026-11-05", "darko": "2026-11-05"}
    assert stored["rapm_3y"].as_of is None and stored["darko"].as_of == "2026-11-04"
    pd.testing.assert_frame_equal(stored["darko"].table, EXPECTED)
    assert public_ratings.latest(cfg, 2028) == {}                    # another season's pulls are another folder


def test_offline_nothing_is_asked_and_nothing_written(monkeypatch, tmp_path):
    monkeypatch.setenv(OFFLINE_ENV, "1")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: pytest.fail("a request was made while offline"))
    with pytest.raises(OfflineFetch):
        public_ratings.pull(_cfg(tmp_path), 2027, "2026-11-05", pause=0)
    assert not any(tmp_path.rglob("*.parquet"))


def test_the_three_systems_are_the_owners_and_each_names_its_maker():
    """The owner, 2026-11-24: LOKI and Pulse dropped; LEBRON wanted, but no 2026-27 LEBRON is published."""
    assert list(public_ratings.SYSTEMS) == ["xrapm", "rapm_3y", "darko"]
    assert all(system.by and system.link.startswith("https://") for system in public_ratings.SYSTEMS.values())


# ------------------------------------------------------------------------------------ the command
def test_the_command_prints_a_line_per_system_and_ends_1_when_one_failed(monkeypatch, capsys):
    asked = {}

    def pull(cfg, season, day, keys):
        asked.update(season=season, keys=keys)
        return [dict(key="xrapm", status="pulled", rows=582, as_of="2026-11-04"),
                dict(key="rapm_3y", status="pulled", rows=737, as_of=None),
                dict(key="darko", status="failed", error="ConnectionError: no answer")]

    kept = PublicRatings(EXPECTED, "2026-11-01", "2026-11-03")
    monkeypatch.setattr(public_ratings, "pull", pull)
    monkeypatch.setattr(public_ratings, "latest", lambda cfg, season: {"darko": kept})
    assert cli.main(["data", "public-ratings", "--season", "2027", "--systems", "xrapm,rapm_3y,darko"]) == 1
    assert asked == dict(season=2027, keys=["xrapm", "rapm_3y", "darko"])
    out = capsys.readouterr().out
    assert "xRAPM         582 players, as of 2026-11-04" in out
    assert "3-year RAPM   737 players, as of no date on its page" in out
    assert "DARKO        FAILED: ConnectionError: no answer; its pull of 2026-11-03 stays" in out

    monkeypatch.setattr(public_ratings, "pull", lambda cfg, season, day, keys: [
        dict(key="darko", status="pulled", rows=660, as_of="2026-11-04")])
    assert cli.main(["data", "public-ratings"]) == 0


@pytest.mark.parametrize("name", ["epm", "loki", "pulse"])
def test_the_command_refuses_a_system_it_does_not_know(name):
    with pytest.raises(SystemExit, match=f"no system named {name}"):
        cli.main(["data", "public-ratings", "--systems", name])
