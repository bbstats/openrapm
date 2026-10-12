"""No test may reach the network.

This exists because of a bug it would have caught.  The README claims `pytest -q` passes on a fresh
clone with no data, and it did -- by silently scraping stats.nba.com mid-run and writing three game
logs into the clone.  `data.box_scores.season_box` calls `data.ingest.load_gamelog`, which downloads when the
cache is cold, so a test that was meant to skip for want of data quietly fetched it instead.

That is bad three ways.  It makes the suite slow and flaky, it hits a third party on every CI run
and every contributor's first `pytest`, and it means a test that looks like it is exercising the
model is exercising the network.

Why this is session-scoped
--------------------------
The first version of this block was `scope="function"` and did not work.  pytest instantiates
higher-scoped fixtures first, so a `scope="module"` fixture -- `test_vs_consensus.ratings`, which
scraped at the time -- was set up BEFORE the function-scoped block was installed.  The suite went
on downloading and the block looked like it was passing.  `tests/test_network_block.py` is
there so that never goes unnoticed again.  It lives in its own file because pytest does not collect
tests out of `conftest.py` -- the guard sat here for weeks and was never once run.

A test that genuinely needs the network -- and it should be very rare -- takes the `allow_network`
fixture.

Shared fixtures live here too, so no test module imports another and no test repeats another's set-up:
    small_cfg          the design settings every simulated test uses, as a function of the keys a test adds
    synthetic_league   a simulated league (openrapm.simulate) with persistent talent, as a function of what differs
    synthetic_design   the design of some of a simulated league's seasons, through the one assembly (design.assemble)
    lineup_league      the league of random lineups the swap test (test_swap_test.py) and step 5
                       (test_swap_adjust.py) are checked on
    network_blocked    the error a blocked connection raises (test_network_block.py)
and one marker: `@pytest.mark.needs_data("stints/*_PO.parquet", ...)` skips a test unless every configured data path
it names exists (see `data_present`), so a fresh clone with no scraped data skips it instead of failing.
"""
import socket

import numpy as np
import pandas as pd
import pytest

from openrapm.data.config import load_config, resolve
from openrapm.design import FEATURES, assemble, pieces_from_frames
from openrapm.simulate import simulate

_REAL = {"socket": socket.socket, "create_connection": socket.create_connection,
         "getaddrinfo": socket.getaddrinfo}


class NetworkBlocked(RuntimeError):
    """A test tried to reach the network.  See tests/conftest.py."""


def _blocked(*a, **kw):
    raise NetworkBlocked(
        "a test tried to open a network connection.  Tests must run offline: if this is a loader "
        "reaching stats.nba.com for data that is not cached, the caller should skip instead.  See "
        "tests/conftest.py.")


def _block():
    socket.socket = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked


def _unblock():
    for name, fn in _REAL.items():
        setattr(socket, name, fn)


@pytest.fixture(autouse=True, scope="session")
def _no_network():
    """Installed before any module- or class-scoped fixture, which is the whole point."""
    _block()
    try:
        yield
    finally:
        _unblock()


@pytest.fixture
def allow_network():
    """Restore the network for one test.  Mark it `@pytest.mark.network` too, so it is greppable."""
    _unblock()
    try:
        yield
    finally:
        _block()


@pytest.fixture
def network_blocked():
    """The error a blocked connection raises."""
    return NetworkBlocked


# ------------------------------------------------------------------------------ tests that need the scraped data
def data_present(spec: str) -> bool:
    """Whether a configured data path exists: "<key>" or "<key>/<pattern>", the key one of config.yaml's paths.

    Example: "stints/*_PO.parquet" is true when paths.stints holds a playoff stint file; "panel" when the season panel
    exists.  Read through config.yaml (or $OPENRAPM_CONFIG), so a verification config points it elsewhere.
    """
    key, _, pattern = spec.partition("/")
    base = resolve(load_config(), key)
    return any(base.glob(pattern)) if pattern else base.exists()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Skip a test marked needs_data(...) before its fixtures are built, when any path it names is missing."""
    for marker in item.iter_markers("needs_data"):
        missing = [spec for spec in marker.args if not data_present(spec)]
        if missing:
            pytest.skip(f"needs {', '.join(missing)} on disk (a key of config.yaml's paths, then a file pattern)")


# ------------------------------------------------------------------------------ simulated leagues and their designs
# What most simulated tests share: talent that persists from season to season and players who change teams, so a
# season's ratings say something about the next; a stint's points from the ten impacts, the context and noise alone
PERSISTENT_LEAGUE = dict(n_teams=8, players_per_team=10, games_per_season=60, stints_per_game=(20, 30), rho=0.85,
                         turnover=0.3, eps_var=0.2, leak=False)


@pytest.fixture(scope="session")
def small_cfg():
    """The design settings of the simulated tests as a function of the keys a test adds:
    `small_cfg(player_unit="window")` is a new dict each call."""
    def config(**extra) -> dict:
        return {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500, "box_counts": FEATURES, **extra}
    return config


@pytest.fixture(scope="session")
def synthetic_league():
    """`simulate` with PERSISTENT_LEAGUE's settings, as a function of the keywords that differ:
    `synthetic_league(n_seasons=3, seed=4)`.  Any keyword of `simulate` may be changed."""
    def league(**changes) -> dict:
        return simulate(**{**PERSISTENT_LEAGUE, **changes})
    return league


@pytest.fixture(scope="session")
def synthetic_design(small_cfg):
    """The design of a simulated league's `seasons` (every season when None) through the one assembly:
    `synthetic_design(sim, [2002])`, `synthetic_design(sim, cfg=small_cfg(player_unit="window"))`."""
    def design(sim: dict, seasons=None, cfg=None, target: str = "pts"):
        stints, box = sim["stints"], sim["box"]
        if seasons is not None:
            stints, box = stints[stints.season.isin(seasons)], box[box.season.isin(seasons)]
        return assemble(pieces_from_frames(stints, box, FEATURES), small_cfg() if cfg is None else cfg, target)
    return design


def _lineup_league(seed=0, teams=4, players=8, games=240, per_period=6, poss=5, noise=1.2):
    """(stints, truth, roster) of a league with known ratings: random lineups, five possessions a side per stint,
    points drawn around 1.1 a possession plus the ten players' ratings (offence adds, defence = points allowed).

    `stints` has swap_test.STINT_COLUMNS plus home_team_id / away_team_id; `truth` is (player_id, o, d);
    `roster` maps each team to its players' ids.
    """
    rng = np.random.default_rng(seed)
    roster = {t: np.arange(t * 100 + 1, t * 100 + 1 + players) for t in range(1, teams + 1)}
    every = np.concatenate(list(roster.values()))
    truth = pd.DataFrame({"player_id": every, "o": rng.normal(0, 3, len(every)), "d": rng.normal(0, 2, len(every))})
    o, d = dict(zip(truth.player_id, truth.o)), dict(zip(truth.player_id, truth.d))
    rows = []
    for g in range(games):
        h, a = rng.choice(np.arange(1, teams + 1), 2, replace=False)
        margin = 0
        for period in range(1, 5):
            for k in range(per_period):
                five_h = rng.choice(roster[h], 5, replace=False)
                five_a = rng.choice(roster[a], 5, replace=False)
                mu_h = 1.1 + (sum(o[p] for p in five_h) + sum(d[p] for p in five_a)) / 100.0
                mu_a = 1.1 + (sum(o[p] for p in five_a) + sum(d[p] for p in five_h)) / 100.0
                pts_h = poss * mu_h + rng.normal(0, noise * np.sqrt(poss))
                pts_a = poss * mu_a + rng.normal(0, noise * np.sqrt(poss))
                start = 720.0 - k * 720.0 / per_period
                rows.append(dict(game_id=f"g{g:04d}", phase="RS", period=period, start_clock=start,
                                 frac_rem=((4 - period) * 720.0 + start) / 2880.0, margin_h=margin, is_gt=False,
                                 neutral=False, poss_h=poss, poss_a=poss, pts_h=pts_h, pts_a=pts_a,
                                 home_team_id=h, away_team_id=a,
                                 **{f"h{i + 1}": int(p) for i, p in enumerate(five_h)},
                                 **{f"a{i + 1}": int(p) for i, p in enumerate(five_a)}))
                margin += int(round(pts_h - pts_a))
    return pd.DataFrame(rows), truth, roster


@pytest.fixture(scope="session")
def lineup_league():
    """The synthetic league as a function of its seed and size: `lineup_league(seed=0, games=240)`."""
    return _lineup_league
