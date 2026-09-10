"""No test may reach the network.

This exists because of a bug it would have caught.  The README claims `pytest -q` passes on a fresh
clone with no data, and it did -- by silently scraping stats.nba.com mid-run and writing three game
logs into the clone.  `boxtable.season_box` calls `ingest.load_gamelog`, which downloads when the
cache is cold, so a test that was meant to skip for want of data quietly fetched it instead.

That is bad three ways.  It makes the suite slow and flaky, it hits a third party on every CI run
and every contributor's first `pytest`, and it means a test that looks like it is exercising the
model is exercising the network.

So: sockets are blocked for the whole session.  A test that needs scraped data now raises
`NetworkBlocked`, which is what the calling code's own "not built yet" skip is there to catch.  If
you are writing a test that genuinely needs the network -- and it should be very rare -- mark it
`@pytest.mark.network` and it will be allowed.
"""
import socket

import pytest

_real_socket = socket.socket
_real_create_connection = socket.create_connection


class NetworkBlocked(RuntimeError):
    """A test tried to reach the network.  See tests/conftest.py."""


def _blocked(*a, **kw):
    raise NetworkBlocked(
        "a test tried to open a network connection.  Tests must run offline: if this is a loader "
        "reaching stats.nba.com for data that is not cached, the caller should skip instead.  See "
        "tests/conftest.py.")


@pytest.fixture(autouse=True, scope="function")
def _no_network(request):
    if request.node.get_closest_marker("network"):
        yield
        return
    socket.socket = _blocked
    socket.create_connection = _blocked
    try:
        yield
    finally:
        socket.socket = _real_socket
        socket.create_connection = _real_create_connection
