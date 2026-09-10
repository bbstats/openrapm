"""In-season ratings: the rolling kernel fit, the game cut, and the two things a rating is for.

The shipped board is one fit per disjoint 3-season block.  This module is the other mode: a rating
ANCHORED at a season, fit on that season and the two before it with the earlier seasons
down-weighted, and -- for measurement -- fit on only the first `cut` share of the anchor season's
games, so it can be scored on the games it has not seen.

    kernel   {0: 1.0, -1: w1, -2: w2}: the weight of each season, keyed on its offset from the
             ANCHOR (= max(train), so it is the held-out season H in the criterion, the rating's own
             season on the board, and the last season of the block under --consensus).  An offset the
             kernel does not name gets 0.  {0: 1, -1: 1, -2: 1} is the flat rolling 3-year window;
             {0: 1} is a single season.
    cut      q in [0, 1]: the fit may see the anchor season's regular-season games whose chronological
             share of that season is BELOW q, and nothing else of it.  q = 0.25 is "a quarter of the
             way through the season"; q = 1 (or None) is the whole season.

One array carries the cut and the kernel: `kernel_game_mult` returns a per-game weight, and the ridge
rows, the games behind the padded box rates, and the possessions in `Ratings.poss` are all derived
from it, so they cannot disagree.  `keep_games` names the same games for the inputs that are built
from season tables rather than from the design (the shooters' totals, the role inputs).

The decomposition this exists to measure
----------------------------------------
For a held-out season H and a cut q, fit on {H-2, H-1, H<=q} and predict the games of H after the
cut.  There is ONE residual, and two things to ask of it:

    prediction    the team-game mean squared error of that residual (`holdout.score`'s `tg`): the
                  criterion, unchanged except that it now scores the rest of a season in progress
                  rather than a season the fit sat beside.
    attribution   the residual variance a player ridge on the same rows can still put on named
                  players (`investigate.attributable`): what the board failed to credit to the right
                  man.  Lower is better on both.

Both come from the same `Prediction`, so they pair season by season exactly.  Two knobs trade one
against the other: the kernel (more past = a lower-variance rating and staler credit for the season
being played) and the ridge (looser = more of the current season's residual reaches the rating, at
the cost of noise).  q = 0 with a flat kernel is the year-over-year reliability test in its trailing
form: last year's rating, this year's games.

What the cut does NOT reach is written down in HANDOFF and in the leak table of FINDINGS 31; the one
accepted leak is `xftm`, the shooter's leave-one-game-out season free-throw percentage, which is
baked into the stints at build time and would need a rebuild per cut.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace

import numpy as np
import pandas as pd

from .windows import window_seasons


def season_rank(games: pd.DataFrame) -> np.ndarray:
    """The 0-based chronological rank of every regular-season game within its own season, indexed by
    `game_idx` (-1 for playoff games).  `design._order_games` sorts by season, phase, date and id, so
    `game_idx` is already chronological within a season."""
    n = int(games["game_idx"].max()) + 1
    out = np.full(n, -1.0)
    rs = games[games["phase"] == "RS"].sort_values("game_idx")
    for _, g in rs.groupby("season", sort=False):
        idx = g["game_idx"].to_numpy()
        out[idx] = np.arange(len(idx), dtype=float)
    return out


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


def kernel_game_mult(wd, anchor: int, kernel: dict | None, cut: float | None = None) -> np.ndarray:
    """The per-game weight of a kernel fit, indexed by `game_idx`.

    A game of season s takes `kernel[s - anchor]`, or 0 if the kernel does not name that offset.  With
    a cut below 1, every game of the ANCHOR season at or past the cut takes 0 as well, and that now
    includes its playoff games without a special case: `season_frac` scores them 1.0, so any cut below
    1 excludes them.  Seasons other than the anchor are never cut, playoffs and all.
    """
    games = wd.games
    n = int(games["game_idx"].max()) + 1
    gm = np.zeros(n)
    k = {int(o): float(v) for o, v in (kernel or {0: 1.0}).items()}
    season = np.full(n, -1, dtype=np.int64)
    season[games["game_idx"].to_numpy()] = games["season"].to_numpy().astype(np.int64)
    for off, v in k.items():
        gm[season == int(anchor) + off] = v
    if cut is not None and float(cut) < 1.0:
        frac = season_frac(games)
        is_anchor = season == int(anchor)
        gm[is_anchor & (frac >= float(cut))] = 0.0        # played after the cut, playoff games included
    return gm


def keep_games(wd, anchor: int, cut: float | None) -> dict | None:
    """{season: the game_ids of the anchor season the fit may see}, for the inputs that are built from
    season tables instead of from the design (`xshoot.season_totals`, `roles.cut_role_inputs`).  None
    when there is no cut, which every existing caller passes through unchanged."""
    if cut is None or float(cut) >= 1.0:
        return None
    games = wd.games
    frac = season_frac(games)
    g = games[(games["season"] == int(anchor)) & (games["phase"] == "RS")]
    idx = g["game_idx"].to_numpy()
    return {int(anchor): g["game_id"].to_numpy()[frac[idx] < float(cut)]}


def anchor_of(train) -> int:
    """The season a fit's kernel is keyed on: the last one it trains on.  In the criterion that is the
    held-out season H (the kernel fit sees part of it), on the season board the rating's own season,
    and under `45_holdout.py --consensus` the last season of the block."""
    return int(max(int(s) for s in train))


def _has_field(obj, name: str) -> bool:
    return any(f.name == name for f in fields(obj)) if is_dataclass(obj) else False


# ------------------------------------------------------------------------------------ the systems
@dataclass
class KernelSystem:
    """`inner` (a `fastfit.MspiFast`) fit on the anchor season and the ones its kernel names, with the
    anchor season cut at `cut`.  The runner asks `train_for` for the training seasons instead of
    `Context.neighbourhood`, because a kernel fit never uses a season after the one it rates."""
    name: str
    inner: object
    cut: float | None = None

    def train_for(self, h: int, ctx) -> list | None:
        s0 = int(ctx.cfg["first_season"])
        offs = sorted(int(o) for o in (self.inner.kernel or {0: 1.0}))
        return [int(h) + o for o in offs if int(h) + o >= s0] or None

    def fit(self, train, ctx):
        if self.cut is None or not _has_field(self.inner, "cut"):
            return self.inner.fit(train, ctx)
        return replace(self.inner, cut=self.cut).fit(train, ctx)


@dataclass
class BlockSystem:
    """The in-season baseline a chunk product actually offers: the last disjoint window that had
    FINISHED before season h, with no part of h in it.  `cut` is a scoring mask only -- the fit never
    sees the anchor season at all -- so it exists to make the comparison against a kernel fair."""
    name: str
    inner: object
    cut: float | None = None

    def train_for(self, h: int, ctx) -> list | None:
        wins = [w for w in window_seasons(ctx.cfg) if int(w[1]) < int(h)]
        return list(range(int(wins[-1][0]), int(wins[-1][1]) + 1)) if wins else None

    def fit(self, train, ctx):
        return self.inner.fit(train, ctx)
