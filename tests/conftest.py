"""No test may reach the network.

This exists because of a bug it would have caught.  The README claims `pytest -q` passes on a fresh
clone with no data, and it did -- by silently scraping stats.nba.com mid-run and writing three game
logs into the clone.  `boxtable.season_box` calls `ingest.load_gamelog`, which downloads when the
cache is cold, so a test that was meant to skip for want of data quietly fetched it instead.

That is bad three ways.  It makes the suite slow and flaky, it hits a third party on every CI run
and every contributor's first `pytest`, and it means a test that looks like it is exercising the
model is exercising the network.

Why this is session-scoped
--------------------------
The first version of this block was `scope="function"` and did not work.  pytest instantiates
higher-scoped fixtures first, so a `scope="module"` fixture -- `test_vs_consensus.board`, the one
that actually scrapes -- was set up BEFORE the function-scoped block was installed.  The suite went
on downloading and the block looked like it was passing.  `test_the_network_block_works` below is
there so that never goes unnoticed again.

A test that genuinely needs the network -- and it should be very rare -- takes the `allow_network`
fixture.
"""
import socket

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


def test_the_network_block_works():
    """The regression for the scope bug.  If this passes and the suite still writes to data/raw,
    something is holding a reference to the real socket functions from before the block."""
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("stats.nba.com", 443), timeout=1)
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("stats.nba.com", 443)
    with pytest.raises(NetworkBlocked):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
