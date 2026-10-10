"""A game's position in its own season, the one piece of the in-season machinery the board still reads.

`season_frac` puts every game of a season on [0, 1] in chronological order, playoff games at 1.0: the
single-year board (scripts/62_single_year_board.py) uses it to split a season's rows by how far into the
season they were played.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def season_frac(games: pd.DataFrame) -> np.ndarray:
    """Each game's position in its own season on [0, 1], indexed by `game_idx`.  A regular-season game is
    its chronological rank over the count of them, so it lands in [0, 1).  A PLAYOFF game is 1.0: it comes
    after every regular-season game, which is the only thing the cut needs to know about it.

    A cut q trains on the games with frac < q and scores the ones with frac >= q, so this is what makes the
    held-out season one entity -- a fit that saw the first three quarters of the regular season is asked to
    predict the rest of the season, and the rest of the season includes its playoffs.  This was -1 until
    2026-09-10, which put playoff games on the training side of every comparison and then dropped them from
    the scored side, so they were in neither.
    """
    n = int(games["game_idx"].max()) + 1
    out = np.full(n, 1.0)
    rs = games[games["phase"] == "RS"].sort_values("game_idx")
    for _, g in rs.groupby("season", sort=False):
        idx = g["game_idx"].to_numpy()
        out[idx] = np.arange(len(idx), dtype=float) / float(len(idx))
    return out
