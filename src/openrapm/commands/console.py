"""How the commands and the scripts run: the console's encoding and the number of linear-algebra threads.

`use_utf8()` is the first line of every entry point (`openrapm.commands.cli.main` and each script's `main`).  It
used to run as a side effect of importing the config module, so a program printed accents correctly only if it
happened to import that module first.

`one_thread()` wraps the work of every check that solves least squares.  `openrapm` itself holds the libraries to
one thread through the environment before numpy loads (`openrapm.commands.cli`); a script imports numpy at its top,
so it limits the threads at run time instead.  Both give the same bits.
"""
from __future__ import annotations

import sys

from threadpoolctl import threadpool_limits

__all__ = ["one_thread", "use_utf8"]


def use_utf8() -> None:
    """Write stdout and stderr as UTF-8: player names carry accents (Schröder, Ginóbili), and a Windows console
    defaults to cp1252, which cannot encode some of them."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def one_thread() -> threadpool_limits:
    """Hold every linear-algebra library (OpenMP, MKL, OpenBLAS) to one thread inside a `with` block.

    A multi-threaded library splits a sum across threads and adds the pieces in another order, which moves the last
    digits of a solve; one thread gives the same numbers on any machine (DECISIONS.md, the cleanup's thread rule,
    2026-10-09).

        with one_thread():
            swap_test(args)
    """
    return threadpool_limits(limits=1)
