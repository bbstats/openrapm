"""The scorecard: a rating's ERROR and its SLOPE on held-out team-games, kept apart (experiment 35).

A rating is scored on games it never saw: a within-season fold's held-out games (scripts/97_within_season.py), or a
neighbouring season (the year-over-year test).  Each scored set is a `Block` of team-game rows:

    y   points per 100 for the team on offence        w   its possessions
    Z   possession shares, offence players then defence players (each half sums to 5 per row)
    F   the level block (home, season intercept, playoffs, playoff home)

Ratings are in RAW sign here, as the scoring code has them: `o` adds points scored, `d` adds points ALLOWED.  A
player the rating has no row for is scored at the system's fill (the stand-in), kept as a fixed offset.

What each number answers:

    error     possession-weighted mean squared team-game miss, the level (intercept and home edge) refit on the scored
              games -- the year-over-year test's `tg`.  Ranks systems; nearly blind to a rating's spread.
    slope     least squares of the outcome on each side's contribution, the level refit block by block, pooled over
              blocks.  1 = calibrated, below 1 = too wide.  Outcome noise widens it and does not bias it.
    rescaled  the error after each side is multiplied by a slope fitted on OTHER seasons: what the rating knows once
              its spread is set fairly.
    encompass the weight on candidate-minus-baseline in a regression of the baseline's miss: 1 = the candidate's
              change is all signal, 0 = all noise, above 0.5 = the candidate has the lower error.

All of them see the SUM of the five players on the court and nothing about how it is split among teammates who play
together; the controls (`team_mean_control`, `shuffle_within_team`) measure how much each one can see of the split.

Model layer: no file is read here (tests/test_layer_boundary.py); the scripts load blocks and ratings and pass them in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Block:
    """One scored set of team-games."""
    key: str
    season: int                 # the cluster: the scored season (within-season: the rated season too)
    deal: int                   # the repeat of a within-season deal; 0 for the year-over-year test
    Z: sp.csr_matrix
    y: np.ndarray
    w: np.ndarray
    F: np.ndarray
    home: np.ndarray
    player_ids: np.ndarray

    @property
    def n_players(self) -> int:
        return len(self.player_ids)


@dataclass
class Parts:
    """A system's prediction on a block, split by side: rated players' sums and the stand-ins' sums."""
    c_o: np.ndarray
    c_d: np.ndarray
    s_o: np.ndarray
    s_d: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.c_o + self.c_d + self.s_o + self.s_d

    @property
    def side_o(self) -> np.ndarray:
        return self.c_o + self.s_o

    @property
    def side_d(self) -> np.ndarray:
        return self.c_d + self.s_d


# ------------------------------------------------------------------------------------------ the prediction
def _aligned(block: Block, ratings: pd.DataFrame, column: str) -> np.ndarray:
    """`ratings[column]` on the block's player columns, NaN where the rating has no row."""
    s = ratings.drop_duplicates("player_id").set_index("player_id")[column]
    return pd.Series(block.player_ids).map(s).to_numpy(dtype=float)


def parts(block: Block, ratings: pd.DataFrame, fill: tuple = (0.0, 0.0), o: str = "o", d: str = "d") -> Parts:
    """The block's prediction from `ratings` (columns player_id, `o`, `d`; raw sign), stand-ins at `fill`."""
    n = block.n_players
    vo, vd = _aligned(block, ratings, o), _aligned(block, ratings, d)
    rated_o, rated_d = ~np.isnan(vo), ~np.isnan(vd)
    Zo, Zd = block.Z[:, :n], block.Z[:, n:]
    return Parts(c_o=np.asarray(Zo @ np.where(rated_o, vo, 0.0)).ravel(),
                 c_d=np.asarray(Zd @ np.where(rated_d, vd, 0.0)).ravel(),
                 s_o=np.asarray(Zo @ np.where(rated_o, 0.0, fill[0])).ravel(),
                 s_d=np.asarray(Zd @ np.where(rated_d, 0.0, fill[1])).ravel())


def side_column(block: Block, ratings: pd.DataFrame, column: str, side: str) -> np.ndarray:
    """One side's share-weighted sum of `ratings[column]` (0 where the player has no row): a part of a rating."""
    n = block.n_players
    v = np.nan_to_num(_aligned(block, ratings, column), nan=0.0)
    Z = block.Z[:, :n] if side == "O" else block.Z[:, n:]
    return np.asarray(Z @ v).ravel()


def level_matrix(block: Block, level: str = "home") -> np.ndarray:
    """"home": an intercept and the home edge (what the scoring code refits); "full": the whole level block."""
    if level == "home":
        return np.column_stack([np.ones(len(block.y)), block.home])
    if level == "full":
        return np.asarray(block.F, dtype=float)
    raise ValueError(f"level must be 'home' or 'full', got {level!r}")


def _wls_residual(A: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`v` minus its weighted least-squares fit on the columns of A (columns of `v` at once)."""
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(A * sw[:, None], (v.T * sw).T, rcond=None)[0]
    return v - A @ coef


# ------------------------------------------------------------------------------------------ error
def error(block: Block, prediction: np.ndarray | None, level: str = "home") -> float:
    """Possession-weighted mean squared team-game miss, the level refit on the block (prediction None = no
    ratings, the year-over-year test's `tg_base`)."""
    y = block.y if prediction is None else block.y - prediction
    r = _wls_residual(level_matrix(block, level), y, block.w)
    return float(np.average(r ** 2, weights=block.w))


# ------------------------------------------------------------------------------------------ slopes
@dataclass
class Normal:
    """One block's profiled normal equations: X'WX and X'Wy after the level is taken out, for named columns."""
    season: int
    names: tuple
    xx: np.ndarray
    xy: np.ndarray


def normal(block: Block, columns: dict, offset: np.ndarray | None = None, level: str = "full") -> Normal:
    """The block's normal equations for regressing y - offset on `columns`, the level profiled out."""
    names = tuple(columns)
    X = np.column_stack([np.asarray(columns[k], dtype=float) for k in names])
    y = block.y if offset is None else block.y - offset
    R = _wls_residual(level_matrix(block, level), np.column_stack([y, X]), block.w)
    ry, RX = R[:, 0], R[:, 1:]
    return Normal(block.season, names, (RX * block.w[:, None]).T @ RX, RX.T @ (block.w * ry))


def solve(eqs: list) -> pd.Series:
    """The pooled coefficients of a list of `Normal`s (same names)."""
    names = eqs[0].names
    xx = sum(e.xx for e in eqs)
    xy = sum(e.xy for e in eqs)
    return pd.Series(np.linalg.solve(xx, xy), index=list(names))


def pooled(eqs: list) -> pd.DataFrame:
    """Pooled coefficients with a leave-one-season-out jackknife standard error (seasons are the clusters)."""
    coef = solve(eqs)
    seasons = sorted({e.season for e in eqs})
    if len(seasons) < 3:
        return pd.DataFrame({"coef": coef, "se": np.nan})
    jack = np.array([solve([e for e in eqs if e.season != s]).to_numpy() for s in seasons])
    k = len(seasons)
    se = np.sqrt((k - 1) / k * ((jack - jack.mean(axis=0)) ** 2).sum(axis=0))
    return pd.DataFrame({"coef": coef, "se": se})


def side_normal(block: Block, p: Parts, level: str = "full") -> Normal:
    """Slope per side: y on each side's rated contribution, the stand-ins as a fixed offset."""
    return normal(block, {"O": p.c_o, "D": p.c_d}, offset=p.s_o + p.s_d, level=level)


def crossfit_multipliers(eqs: list, seasons, exclude=lambda s: {s - 1, s, s + 1}) -> pd.DataFrame:
    """Per rated season, the per-side slopes pooled over every OTHER block (seasons in `exclude(s)` left out)."""
    rows = []
    for s in sorted(set(seasons)):
        keep = [e for e in eqs if e.season not in exclude(s)]
        coef = solve(keep)
        rows.append({"season": int(s), **{f"m_{k}": float(v) for k, v in coef.items()},
                     "blocks": len(keep)})
    return pd.DataFrame(rows)


def rescaled_error(block: Block, p: Parts, m_o: float, m_d: float, level: str = "home") -> float:
    """The error after multiplying each side's rated contribution by its cross-fitted slope."""
    return error(block, m_o * p.c_o + m_d * p.c_d + p.s_o + p.s_d, level=level)


def encompass_normal(block: Block, a: Parts, b: Parts, level: str = "full", by_side: bool = True) -> Normal:
    """Baseline `a`'s miss regressed on candidate-minus-baseline (per side, or the total)."""
    if by_side:
        cols = {"O": b.side_o - a.side_o, "D": b.side_d - a.side_d}
    else:
        cols = {"total": b.total - a.total}
    return normal(block, cols, offset=a.total, level=level)


# ------------------------------------------------------------------------------------------ the quadratic
def quadratic(block: Block, p: Parts, level: str = "home", score_weights: np.ndarray | None = None) -> dict:
    """Six weighted cross-products that give the block's error at ANY per-side multipliers (m_o, m_d) on the rated
    contributions, the stand-ins held: u0 = P(y - stand-ins), u_o = P(c_o), u_d = P(c_d), P the level refit.
    At (1, 1) it is `error`; the (m_o, m_d) minimizing a pooled set is the per-side slope at that level.

    `score_weights` re-weights the rows AFTER the level refit (which always uses the block's own weights), so two
    weightings that sum to the block's weights split its error exactly -- experiment 39's traded / not-traded parts."""
    U = _wls_residual(level_matrix(block, level), np.column_stack([block.y - p.s_o - p.s_d, p.c_o, p.c_d]), block.w)
    sw_rows = block.w if score_weights is None else np.asarray(score_weights, dtype=float)
    W = (U * sw_rows[:, None]).T @ U
    return dict(sw=float(sw_rows.sum()), g00=W[0, 0], g0o=W[0, 1], g0d=W[0, 2], goo=W[1, 1], god=W[1, 2], gdd=W[2, 2])


def traded_entries(block: Block, team_of_player: pd.Series, team_off: np.ndarray, team_def: np.ndarray) -> sp.csr_matrix:
    """Z restricted to the entries of players playing for a team other than their team in the rating games
    (`team_of_player`: player_id -> that team; players without one count as not traded)."""
    n = block.n_players
    t = pd.Series(block.player_ids).map(team_of_player).to_numpy(dtype=float)
    known = ~np.isnan(t)
    Z = block.Z.tocoo()
    side_o = Z.col < n
    col = np.where(side_o, Z.col, Z.col - n)
    row_team = np.where(side_o, np.asarray(team_off, dtype=float)[Z.row], np.asarray(team_def, dtype=float)[Z.row])
    hit = known[col] & (t[col] != row_team)
    return sp.csr_matrix((Z.data[hit], (Z.row[hit], Z.col[hit])), shape=Z.shape)


def split_traded(block: Block, Zt: sp.csr_matrix) -> tuple:
    """(the block with the other players' entries, the block with only the traded players' entries): the two Z's sum to
    the block's, so any linear prediction splits exactly into a stayed part and a traded part (experiment 39)."""
    from dataclasses import replace
    return replace(block, Z=sp.csr_matrix(block.Z - Zt)), replace(block, Z=sp.csr_matrix(Zt))


def moved_share(block: Block, team_of_player: pd.Series, team_off: np.ndarray, team_def: np.ndarray) -> np.ndarray:
    """Per held-out row, the share of the ten players on the court who are playing for a team other than their team in
    the rating games (`team_of_player`: player_id -> that team; players without one count as not moved)."""
    Zt = traded_entries(block, team_of_player, team_off, team_def)
    return np.asarray(Zt.sum(axis=1)).ravel() / 10.0


def error_at(q, m_o=1.0, m_d=1.0):
    """The error of stored quadratics at multipliers (m_o, m_d) (rows of a frame, or one dict)."""
    return (q["g00"] - 2 * (m_o * q["g0o"] + m_d * q["g0d"]) + m_o ** 2 * q["goo"] + 2 * m_o * m_d * q["god"]
            + m_d ** 2 * q["gdd"]) / q["sw"]


def multipliers(q: pd.DataFrame) -> np.ndarray:
    """The (m_o, m_d) minimizing the pooled (possession-weighted) error of a set of stored quadratics."""
    s = q[["g0o", "g0d", "goo", "god", "gdd"]].sum()
    return np.linalg.solve(np.array([[s.goo, s.god], [s.god, s.gdd]]), np.array([s.g0o, s.g0d]))


def multipliers_jackknife(q: pd.DataFrame) -> pd.DataFrame:
    """`multipliers` with a leave-one-season-out jackknife standard error (column `season` is the cluster)."""
    coef = multipliers(q)
    seasons = sorted(q.season.unique())
    k = len(seasons)
    jack = np.array([multipliers(q[q.season != s]) for s in seasons])
    se = np.sqrt((k - 1) / k * ((jack - jack.mean(axis=0)) ** 2).sum(axis=0)) if k > 2 else np.full(2, np.nan)
    return pd.DataFrame({"coef": coef, "se": se}, index=["O", "D"])


# ------------------------------------------------------------------------------------------ comparisons
def paired_by_season(frame: pd.DataFrame, a: str, b: str, value: str = "error") -> dict:
    """`frame` has one row per (system, block) with `season`; b minus a, averaged per season, then across
    seasons: mean, its standard error over seasons, z, and seasons where b is lower."""
    p = frame.pivot_table(index=["season", "key"], columns="system", values=value)
    diff = (p[b] - p[a]).groupby(level="season").mean().dropna()
    n = len(diff)
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return dict(a=a, b=b, value=value, mean_diff=float(diff.mean()), se=se,
                z=float(diff.mean() / se) if se and se > 0 else np.nan, wins=int((diff < 0).sum()), seasons=n)


def variance_components(diff: pd.DataFrame) -> dict:
    """A paired difference per block (`season`, `deal`, `d`) split into the season part and the within-season part,
    and the standard error of the overall mean now and with unlimited deals (the season part only)."""
    per_deal = diff.groupby(["season", "deal"]).d.mean()
    per_season = per_deal.groupby(level="season").mean()
    deals = per_deal.groupby(level="season").size()
    k = len(per_season)
    within = per_deal.groupby(level="season").var(ddof=1).mean() if deals.min() > 1 else np.nan
    total_var = per_season.var(ddof=1)
    season_var = max(total_var - (within / deals.mean() if within == within else 0.0), 0.0)
    return dict(seasons=k, deals=float(deals.mean()), se_now=float(np.sqrt(total_var / k)),
                se_unlimited_deals=float(np.sqrt(season_var / k)), deal_var=float(within), season_var=float(season_var))


# ------------------------------------------------------------------------------------------ controls
def team_mean_control(ratings: pd.DataFrame, team: pd.Series, weight: pd.Series, cols=("o", "d")) -> pd.DataFrame:
    """Every player's rating replaced by his main team's weighted mean: team sums kept, the split destroyed."""
    out = ratings.copy()
    t = out.player_id.map(team)
    wv = out.player_id.map(weight).fillna(0.0).to_numpy(float)
    for c in cols:
        frame = pd.DataFrame({"t": t, "v": out[c].to_numpy(float) * wv, "w": wv})
        g = frame.groupby("t")[["v", "w"]].transform("sum")
        out[c] = np.where(g.w > 0, g.v / g.w.where(g.w > 0, 1.0), out[c])
    return out


def shuffle_within_team(ratings: pd.DataFrame, team: pd.Series, weight: pd.Series, rng, cols=("o", "d")) -> pd.DataFrame:
    """Ratings permuted among teammates (same main team), then each team's weighted mean restored."""
    out = ratings.copy()
    t = out.player_id.map(team).to_numpy()
    wv = out.player_id.map(weight).fillna(0.0).to_numpy(float)
    for c in cols:
        v = out[c].to_numpy(float).copy()
        new = v.copy()
        for team_id in pd.unique(t):
            idx = np.flatnonzero(t == team_id)
            new[idx] = v[rng.permutation(idx)]
            if wv[idx].sum() > 0:
                new[idx] += np.average(v[idx], weights=wv[idx]) - np.average(new[idx], weights=wv[idx])
        out[c] = new
    return out


def shuffle_within_bins(ratings: pd.DataFrame, poss: pd.Series, rng, n_bins: int = 10, cols=("o", "d")) -> pd.DataFrame:
    """Ratings permuted among players with similar possessions, league-wide: the information destroyed, the
    relation of rating level to playing time kept."""
    out = ratings.copy()
    p = out.player_id.map(poss).fillna(0.0).to_numpy(float)
    bins = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
    for c in cols:
        v = out[c].to_numpy(float).copy()
        new = v.copy()
        for b in np.unique(bins):
            idx = np.flatnonzero(bins == b)
            new[idx] = v[rng.permutation(idx)]
        out[c] = new
    return out
