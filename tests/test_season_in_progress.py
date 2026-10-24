"""The season in progress (2026-10-23; the owner: "let's pull data for the current season so far. we also need to pull
data regularly").  No test here reaches the network: conftest.py blocks sockets, and the one door every pull goes
through (`ingest._retry`) is replaced by a fake that hands back a game log.
"""
import pandas as pd

from openrapm.data import ingest

DATES = {"0022600001": "2026-10-21", "0022600002": "2026-10-22", "0022600003": "2026-10-23"}


def _cfg(tmp_path):
    return {"_root": str(tmp_path), "paths": {"raw": "data/raw"}}


def _log_rows():
    """The game log stats.nba.com would send on the evening of 2026-10-23: two rows (home, away) a game, the last
    game still being played."""
    rows = []
    for game_id, date in DATES.items():
        rows.append(dict(GAME_ID=game_id, GAME_DATE=date, TEAM_ID=1, MATCHUP="AAA vs. BBB"))
        rows.append(dict(GAME_ID=game_id, GAME_DATE=date, TEAM_ID=2, MATCHUP="BBB @ AAA"))
    return pd.DataFrame(rows)


def _fake_network(monkeypatch):
    """The game log pull returns `_log_rows`, the pbpstats pull an empty table; the games fetched are recorded."""
    pulls, fetched = [], []

    def fake_retry(fn, *args, **kwargs):
        what = kwargs.get("what", "")
        pulls.append(what)
        return _log_rows() if what.startswith("gamelog") else pd.DataFrame()

    monkeypatch.setattr(ingest, "_retry", fake_retry)
    monkeypatch.setattr(ingest, "_fetch_games", lambda todo, *args: fetched.extend(todo) or [])
    monkeypatch.setattr(ingest, "eastern_today", lambda: pd.Timestamp("2026-10-23"))
    return pulls, fetched


def _cache_empty_log(tmp_path):
    """The 2027 log as it sat on disk: fetched on 2026-10-01, before the season, with no rows."""
    path = tmp_path / "data" / "raw" / "gamelog" / "2027_RS.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(columns=["GAME_ID", "GAME_DATE", "TEAM_ID", "MATCHUP", "season", "phase"]).to_parquet(path)
    return path


def test_refresh_fetches_the_log_again_and_keeps_the_games_before_today(tmp_path, monkeypatch):
    path = _cache_empty_log(tmp_path)
    pulls, fetched = _fake_network(monkeypatch)
    summary = ingest.ingest_season(2027, ("RS",), _cfg(tmp_path), delay=0.0, verbose=False, refresh=True)
    cached = pd.read_parquet(path)
    assert sorted(cached["GAME_ID"].unique()) == ["0022600001", "0022600002"], "a game of today may still be in play"
    assert sorted(fetched) == ["0022600001", "0022600002"]
    assert summary["RS"]["n_games"] == 2
    assert "gamelog 2027 RS" in pulls and "pbpstats games 2027 RS" in pulls


def test_without_refresh_the_cached_log_is_read_and_nothing_is_fetched(tmp_path, monkeypatch, capsys):
    _cache_empty_log(tmp_path)
    pulls, fetched = _fake_network(monkeypatch)
    summary = ingest.ingest_season(2027, ("RS",), _cfg(tmp_path), delay=0.0, refresh=False)
    assert not pulls and not fetched
    assert summary["RS"]["n_games"] == 0
    assert "no games yet" in capsys.readouterr().out


def test_eastern_today_is_a_date_without_a_time_zone():
    today = ingest.eastern_today()
    assert today.tz is None and today == today.normalize()


def test_until_4_am_eastern_today_is_still_the_night_before():
    """The west coast's last game of 2026-10-23 ends near 1 AM Eastern: its play-by-play must wait for the morning."""
    assert ingest.eastern_today(pd.Timestamp("2026-10-24 04:45", tz="UTC")) == pd.Timestamp("2026-10-23")
    assert ingest.eastern_today(pd.Timestamp("2026-10-24 10:00", tz="UTC")) == pd.Timestamp("2026-10-24")
