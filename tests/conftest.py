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
higher-scoped fixtures first, so a `scope="module"` fixture -- `test_vs_consensus.board`, the one
that actually scrapes -- was set up BEFORE the function-scoped block was installed.  The suite went
on downloading and the block looked like it was passing.  `tests/test_network_block.py` is
there so that never goes unnoticed again.  It lives in its own file because pytest does not collect
tests out of `conftest.py` -- the guard sat here for weeks and was never once run.

A test that genuinely needs the network -- and it should be very rare -- takes the `allow_network`
fixture.

Shared fixtures live here too, so no test module imports another: `lineup_league` is the synthetic league the swap
test (test_swap_test.py) and step 5 (test_swap_adjust.py) are checked on.
"""
import socket

import numpy as np
import pandas as pd
import pytest

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
