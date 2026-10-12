"""OPENRAPM_OFFLINE=1: every network fetch raises, naming the file it would have written, instead of fetching.

A rebuild from a copied raw folder (the verification runs of the 2026-10 cleanup) must fail on the first file
that was not copied, not quietly download it and carry on: a downloaded file can differ from the one the
published numbers were built from.  No test here reaches the network -- conftest.py blocks sockets, and the one
door every pull goes through (`ingest._retry`) is replaced by a recorder.
"""
import pandas as pd
import pytest

from openrapm.data import ingest, roles
from openrapm.data.ingest import OFFLINE_ENV, OfflineFetch, offline

GAME = "0022500001"
FETCHES = {
    "game log": (lambda cfg: ingest.load_gamelog(2026, "RS", cfg), ("gamelog", "2026_RS.parquet")),
    "play-by-play": (lambda cfg: ingest.fetch_pbp(GAME, cfg), ("pbp", "2026", f"{GAME}.parquet")),
    "box score": (lambda cfg: ingest.fetch_boxscore(GAME, cfg), ("box", "2026", f"{GAME}.parquet")),
    "period box score": (lambda cfg: ingest.fetch_period_box(GAME, 1, cfg), ("periodbox", "2026", f"{GAME}_1.parquet")),
    "pbpstats games": (lambda cfg: ingest.pbpstats_games(2026, "RS", cfg), ("pbpstats_games", "2026_RS.parquet")),
    "bio stats (ages)": (lambda cfg: roles.season_ages(2026, cfg), ("bio", "2026_RS.parquet")),
}


class Fetched(Exception):
    """Raised by the recorder in place of a real pull."""


@pytest.fixture
def pulls(monkeypatch):
    """Every pull goes through `_retry`; record it and stop instead of fetching."""
    seen = []

    def recorder(fn, *args, **kwargs):
        seen.append(kwargs.get("what"))
        raise Fetched(kwargs.get("what"))

    monkeypatch.setattr(ingest, "_retry", recorder)
    monkeypatch.setattr(roles, "_retry", recorder)
    return seen


def _cfg(tmp_path):
    return {"_root": str(tmp_path), "paths": {"raw": "data/raw"}}


@pytest.mark.parametrize("what", sorted(FETCHES))
def test_offline_a_missing_file_raises_instead_of_fetching(what, tmp_path, monkeypatch, pulls):
    fetch, parts = FETCHES[what]
    monkeypatch.setenv(OFFLINE_ENV, "1")
    with pytest.raises(RuntimeError, match=f"{OFFLINE_ENV}=1") as caught:
        fetch(_cfg(tmp_path))
    assert isinstance(caught.value, OfflineFetch)
    missing = tmp_path.joinpath("data", "raw", *parts)
    assert str(missing) in str(caught.value), "the message must name the file that is missing"
    assert not pulls, f"{what}: a fetch was attempted while offline"
    assert not any(tmp_path.rglob("*.parquet")), "nothing may be written"


@pytest.mark.parametrize("what", sorted(FETCHES))
def test_online_the_default_still_fetches(what, tmp_path, monkeypatch, pulls):
    """Without the switch nothing changes: a missing file goes to the network (here, to the recorder)."""
    fetch, _ = FETCHES[what]
    monkeypatch.delenv(OFFLINE_ENV, raising=False)
    with pytest.raises(Fetched):
        fetch(_cfg(tmp_path))
    assert len(pulls) == 1


def test_offline_a_cached_file_is_still_read(tmp_path, monkeypatch, pulls):
    monkeypatch.setenv(OFFLINE_ENV, "1")
    path = tmp_path / "data" / "raw" / "gamelog" / "2026_RS.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"GAME_ID": ["0022500001"], "season": [2026]}).to_parquet(path, index=False)
    got = ingest.load_gamelog(2026, "RS", _cfg(tmp_path))
    assert list(got.GAME_ID) == ["0022500001"] and not pulls


def test_offline_a_forced_refetch_is_refused(tmp_path, monkeypatch, pulls):
    monkeypatch.setenv(OFFLINE_ENV, "1")
    path = tmp_path / "data" / "raw" / "gamelog" / "2026_RS.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"GAME_ID": ["0022500001"]}).to_parquet(path, index=False)
    with pytest.raises(OfflineFetch):
        ingest.load_gamelog(2026, "RS", _cfg(tmp_path), force=True)
    assert not pulls


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("YES", True), ("", False), ("0", False),
                                            ("no", False)])
def test_the_switch_reads_its_value(value, expected, monkeypatch):
    monkeypatch.setenv(OFFLINE_ENV, value)
    assert offline() is expected


# ----------------------------------------------------------------------------- the loops that forgive a failed game
# Five places catch any error from one game (or one cross-check) and carry on: the stint build records it in the
# diagnostics and drops the game, the shot curve skips up to 5% of a season, and the ingest driver and the stint
# diagnostics note a failed pbpstats pull.  Offline, a missing file is a copying mistake, so each must let the
# refusal through: otherwise a file missing from a copied raw folder shows up as a quietly different stint file.
N_GAMES = 20


def _game_ids(n=N_GAMES):
    return [f"00225000{k:02d}" for k in range(1, n + 1)]


def _write_gamelog(tmp_path, n=N_GAMES):
    """A real cached game log: two rows (home, away) per game, the columns `ingest.game_table` reads."""
    rows = []
    for k, gid in enumerate(_game_ids(n)):
        date = f"2025-11-{1 + k % 28:02d}"
        rows.append(dict(GAME_ID=gid, GAME_DATE=date, TEAM_ID=1, MATCHUP="AAA vs. BBB", season=2026, phase="RS"))
        rows.append(dict(GAME_ID=gid, GAME_DATE=date, TEAM_ID=2, MATCHUP="BBB @ AAA", season=2026, phase="RS"))
    path = tmp_path / "data" / "raw" / "gamelog" / "2026_RS.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _loop_cfg(tmp_path):
    return {"_root": str(tmp_path), "paths": {"raw": "data/raw", "stints": "data/stints"}}


def _stint_build(tmp_path, monkeypatch):
    """`stints.build_season` with one game's play-by-play missing; every other input faked."""
    from openrapm.data import box_scores, shot_curves, stints
    monkeypatch.setattr(box_scores, "box_from_gamelog", lambda gl: None)
    monkeypatch.setattr(box_scores, "ft_totals", lambda box: {})
    monkeypatch.setattr(box_scores, "ft_padding", lambda box: (0.75, 100.0))
    monkeypatch.setattr(shot_curves, "curve_for", lambda *a, **k: None)
    monkeypatch.setattr(stints, "build_game", lambda game_id, home, away, cfg, *rest: stints.fetch_pbp(game_id, cfg))
    stints.build_season(2026, "RS", _loop_cfg(tmp_path), force=True, verbose=False)


def _stint_diagnostics(tmp_path, monkeypatch):
    """`stints.season_diagnostics` with the pbpstats file missing."""
    from openrapm.data import stints
    diag = pd.DataFrame({c: [1.0] for c in (
        "n_poss_home", "n_poss_away", "n_valid_poss", "n_poss", "unresolved_subs", "sub_out_not_on_floor",
        "forced_switches", "periods_from_box", "n_periods", "periods_starters_failed", "pts_home", "box_pts_home",
        "pts_away", "box_pts_away")}).assign(game_id=_game_ids(1))
    monkeypatch.setattr(stints, "build_season", lambda *a, **k: (pd.DataFrame(), diag))
    stints.season_diagnostics(2026, "RS", _loop_cfg(tmp_path))


def _shot_curve(tmp_path, monkeypatch):
    """`shot_curves.season_shot_rows` with 19 of 20 play-by-play files present: enough to fit a curve on."""
    from openrapm.data import shot_curves
    for gid in _game_ids()[1:]:
        path = tmp_path / "data" / "raw" / "pbp" / "2026" / f"{gid}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"actionType": ["Made Shot"]}).to_parquet(path, index=False)
    monkeypatch.setattr(shot_curves, "shot_rows", lambda pbp: pd.DataFrame({"three": [False], "made": [1], "bin": [0]}))
    shot_curves.season_shot_rows(2026, _loop_cfg(tmp_path))


def _ingest_pbpstats(tmp_path, monkeypatch):
    """`ingest.ingest_season` with the pbpstats file missing."""
    ingest.ingest_season(2026, ("RS",), _loop_cfg(tmp_path), delay=0.0, verbose=False)


def _ingest_games(tmp_path, monkeypatch):
    """`ingest.ingest_season` with the pbpstats file cached and the play-by-play missing."""
    path = tmp_path / "data" / "raw" / "pbpstats_games" / "2026_RS.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"GameId": ["x"]}).to_parquet(path, index=False)
    ingest.ingest_season(2026, ("RS",), _loop_cfg(tmp_path), workers=1, delay=0.0, verbose=False)


FORGIVING_LOOPS = {"stint build": _stint_build, "stint diagnostics": _stint_diagnostics,
                   "shot curve": _shot_curve, "ingest, pbpstats": _ingest_pbpstats, "ingest, games": _ingest_games}


@pytest.mark.parametrize("loop", sorted(FORGIVING_LOOPS))
def test_offline_a_loop_that_forgives_a_failed_game_still_stops(loop, tmp_path, monkeypatch, pulls):
    monkeypatch.setenv(OFFLINE_ENV, "1")
    _write_gamelog(tmp_path)
    with pytest.raises(OfflineFetch):
        FORGIVING_LOOPS[loop](tmp_path, monkeypatch)
    assert not pulls, f"{loop}: a fetch was attempted while offline"
    stints_dir = tmp_path / "data" / "stints"
    assert not (stints_dir.exists() and any(stints_dir.iterdir())), f"{loop}: a stint file was written"
