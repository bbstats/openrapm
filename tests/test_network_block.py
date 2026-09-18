"""The regression for the network block, in a file pytest actually collects.

These three assertions lived at the bottom of `tests/conftest.py`, where pytest's default
`python_files` never looked at them: the guard against the suite silently scraping stats.nba.com had
itself never run.  If this file passes and the suite still writes into `data/raw`, something is
holding a reference to the real socket functions from before the block was installed.
"""
import socket

import pytest

from conftest import NetworkBlocked, _blocked


def test_the_network_block_works():
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("stats.nba.com", 443), timeout=1)
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("stats.nba.com", 443)
    with pytest.raises(NetworkBlocked):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


@pytest.mark.network
def test_allow_network_hands_back_the_real_socket(allow_network):
    """The escape hatch has to work, or the next test that genuinely needs the network will be told
    to use a fixture that does nothing.  Nothing is dialled here: holding the real function is the
    whole claim, and opening a connection would make the suite depend on a third party again."""
    assert socket.socket is not _blocked
    assert socket.create_connection is not _blocked
    assert socket.socket.__module__ == "socket"
    assert socket.create_connection.__module__ == "socket"
