"""How ratings are scored on games they never saw, and how a miss is reported.

The year-over-year test (`openrapm.commands.year_over_year`, run by scripts/checks/63_yoy.py) asks one season's
ratings to predict the games of the season before or after it.  Only the scored season's level is refitted; the
ratings are held exactly as they were handed in, so what is measured is the ratings table and nothing fitted here.

Words used below
    scored season   the season whose games are predicted
    rated season    the season whose ratings predict them
    row             one stint seen from one side: five players on offence, five on defence (design.py)
    raw sign        `o` adds points scored and `d` adds points ALLOWED, the sign of the design's defensive
                    columns.  A table stored positive-good is flipped on the way in: rating_def +1.5 is d = -1.5
    fill            what a player the ratings have no row for scores: 0 (the average player) unless a
                    replacement level is set (`replacement_level`)
    level           the intercept and the home edge, refitted on the scored season: scoring drifts across eras,
                    and that is not what is under test
    team-game       one team's offence in one game: the unit the test decides on, because the noise inside a
                    team's hundred-odd possessions cancels before the miss is squared

The prediction of a row r of the scored season:

    prediction_r = a + b home_r + sum over the offensive five of o_i + sum over the defensive five of d_j

    a, b      the level: a weighted least-squares fit of the row's points per 100 on [1, home], with the
              ratings' part held fixed; weight = the row's possessions
    home_r    +1 when the offence is at home, -1 away, 0 at a neutral site

Two errors, each a weighted mean square:

    stint MSE      = sum over rows w_r (y_r - prediction_r)^2 / sum w_r
    team-game MSE  = sum over team-games n_g (100 (P_g - Phat_g) / n_g)^2 / sum n_g

    y_r, w_r      the row's points per 100 possessions and its weight (possessions)
    P_g, Phat_g   the team's actual and predicted points in the game: sum over its rows of y_r x poss_r / 100
    n_g           the team's possessions in the game

Every error is stored as a mean square, because mean squares pool and pair.  It is reported as ARMSE:

    ARMSE = sqrt(mean squared error) x sqrt(2 / pi)

A root mean square reads bigger than the typical miss, because squaring pays extra attention to the tail.  For a
normal error the mean absolute deviation is sqrt(2 / pi) = 0.7979 of the standard deviation, so the scaled root
says what a typical team-game misses by, in points per 100.

Example: a home offence whose five ratings sum to +6.0 against a defence whose five sum to -2.0 (two points per
100 fewer allowed than average), on a scored season whose level is a = 112.0 and b = 1.4, is predicted at
112.0 + 1.4 + 6.0 - 2.0 = 117.4 points per 100.  A mean square of 100 is a root mean square of 10 and an ARMSE
of 7.98.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["MAE_SCALE", "armse", "Ratings", "replacement_level", "Prediction", "weighted_least_squares",
           "predict_season", "team_game_mse", "keeps_whole_team_games", "score", "movers_split",
           "GAME_BENCH_SHARE_EDGES", "game_bench_share_split", "POOL_BY", "pooled", "paired", "QUALITY_EDGES",
           "QUALITY_LABELS", "quality_tier", "quality_tiers"]

MAE_SCALE = 0.7978845608028654          # sqrt(2 / pi)

# The share-of-the-team-game edges and the barely-seen minimum of `game_bench_share_split` (notes/DECISIONS.md,
# 2026-09-10: the split that partitions the team-games, so its groups add up to the pooled score).
GAME_BENCH_SHARE_EDGES = (0.0, 0.05, 0.15, 0.30, 1.01)
BARELY_SEEN_POSSESSIONS = 500.0

# The owner's player-quality tiers (2026-09-30): a player's rank by total rating within his season.
QUALITY_EDGES = (0, 30, 90, 150, 300, 10**9)
QUALITY_LABELS = ("top 30", "31-90", "91-150", "151-300", "301+")

POOL_BY = ("k", "lam", "system", "split", "group")


def armse(mean_squared_error):
    """A mean squared error on the mean-absolute scale: sqrt(mse) x sqrt(2 / pi)."""
    return np.sqrt(np.asarray(mean_squared_error, dtype=float)) * MAE_SCALE


# ------------------------------------------------------------------------------------------- the ratings
@dataclass(frozen=True)
class Ratings:
    """One row per player: `o`, `d` in raw sign and `poss`, the rated season's possessions.

    `fill_o` / `fill_d` are what a player absent from the table scores: 0 (the average player) unless a
    replacement level is set.
    """
    df: pd.DataFrame
    fill_o: float = 0.0
    fill_d: float = 0.0

    def aligned(self, player_ids) -> pd.DataFrame:
        """The ratings in the order of `player_ids`; an absent player scores the fill and 0 possessions."""
        r = pd.DataFrame({"player_id": np.asarray(player_ids)}).merge(self.df, on="player_id", how="left")
        return r.fillna({"o": float(self.fill_o), "d": float(self.fill_d), "poss": 0.0})


def replacement_level(table: pd.DataFrame, max_poss: float = 500.0, shrink: float = 1.0) -> tuple[float, float]:
    """(offence, defence) in raw sign for a player the table has no row for: the possession-weighted mean rating
    of the table's own players under `max_poss` possessions, times `shrink`; (0, 0) when it has none.

    The year-over-year test settled on a quarter of that level (notes/DECISIONS.md: the only depth better in both
    directions).  `table` is one season with columns `o`, `d` and `poss`.

    Example: players at -2.0 (100 possessions) and -4.0 (300) and a starter at +8.0 (5,000): with max_poss 500
    and shrink 0.25 the offence fill is 0.25 x (100 x -2.0 + 300 x -4.0) / 400 = -0.875.
    """
    low = table[(table.poss > 0) & (table.poss < float(max_poss))]
    if len(low) == 0:
        return 0.0, 0.0
    weight = low.poss.to_numpy(dtype=float)
    return (float(shrink) * float(np.average(low.o, weights=weight)),
            float(shrink) * float(np.average(low.d, weights=weight)))


# ---------------------------------------------------------------------------------------- the prediction
@dataclass
class Prediction:
    """Everything about one set of ratings' prediction of one scored season that the scores and splits need."""
    y: np.ndarray
    w: np.ndarray
    poss: np.ndarray
    home: np.ndarray
    c_off: np.ndarray          # the offensive five's ratings summed, per row
    c_def: np.ndarray          # the defensive five's ratings summed, per row
    pred: np.ndarray           # the level refitted on the season, plus c_off + c_def
    base_pred: np.ndarray      # the level alone, fitted with no ratings
    game_idx: np.ndarray
    is_home_off: np.ndarray
    rat: pd.DataFrame          # the ratings aligned to the season's player columns
    A: np.ndarray              # the level's columns: [1, home]


def weighted_least_squares(columns: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The coefficients of (A'WA) c = A'Wy, by least squares, so a column of zeros (ratings that cover nobody)
    gets a coefficient of 0 rather than an error."""
    weighted = (columns * w[:, None]).T
    return np.linalg.lstsq(weighted @ columns, weighted @ y, rcond=None)[0]


def predict_season(ratings: Ratings, design) -> Prediction:
    """Predict every row of the scored season's `design` from `ratings`; only the level [1, home] is refitted."""
    m = design.spec.n_players
    r = ratings.aligned(design.spec.players["player_id"].to_numpy())
    c_off = design.X[:, :m] @ r["o"].to_numpy()
    c_def = design.X[:, m:2 * m] @ r["d"].to_numpy()
    y, w = design.y, design.w
    home = np.asarray(design.X[:, design.spec.context_column("home")].todense()).ravel()
    level = np.column_stack([np.ones(design.X.shape[0]), home])
    contribution = c_off + c_def
    pred = level @ weighted_least_squares(level, y - contribution, w) + contribution
    base_pred = level @ weighted_least_squares(level, y, w)
    return Prediction(y=y, w=w, poss=design.rows["poss"].to_numpy(dtype=float), home=home, c_off=c_off,
                      c_def=c_def, pred=pred, base_pred=base_pred, game_idx=design.rows["game_idx"].to_numpy(),
                      is_home_off=design.rows["is_home_off"].to_numpy(), rat=r, A=level)


# ------------------------------------------------------------------------------------------- the scores
def team_game_mse(y, pred, poss, game_idx, is_home_off) -> tuple[float, float]:
    """Actual and predicted POINTS summed over each team's rows in a game, then scored: (mse, possessions)."""
    g = pd.DataFrame({"g": game_idx, "h": is_home_off, "poss": poss,
                      "act": y * poss / 100.0, "prd": pred * poss / 100.0}).groupby(["g", "h"]).sum()
    return float(np.average(((g.act - g.prd) / g.poss * 100.0) ** 2, weights=g.poss)), float(g.poss.sum())


def keeps_whole_team_games(prediction: Prediction, rows: np.ndarray) -> bool:
    """True if `rows` takes every row of each team-game it takes any row of."""
    key = np.stack([np.asarray(prediction.game_idx), np.asarray(prediction.is_home_off, dtype=np.int64)])
    _, inverse, count = np.unique(key, axis=1, return_inverse=True, return_counts=True)
    kept = np.bincount(inverse[rows], minlength=len(count))
    return bool(np.all((kept == 0) | (kept == count)))


def score(p: Prediction, mask=None) -> dict:
    """The scores of a prediction, on all rows or on a subset (the level stays the whole season's refit).

      mse, base          weighted stint MSE with the ratings, and with the level alone
      calib, calib_side  stint MSE after one scalar (calib_side: one per side) fitted ON the scored season:
                         an amplitude diagnostic and an upper bound, never a score
      scale_*            those scalars; 1.0 = calibrated, below 1 = the ratings are too wide
      covered            share of the season's players the ratings have a row for
      tg, tg_base        the same two errors per team-game; NaN when `mask` cuts a team-game

    `tg` is only defined on a mask that keeps whole team-games.  It sums a team's points over its rows in a
    game; on a mask that takes some of a team-game's rows and not others the sum is a partial point total
    scored against a level fitted on complete ones, which is not the test restricted to those rows.  Measured
    on the 2026-09 ratings, a stint-level split's groups recombined to 337.6 where the pooled score was 113.6
    (notes/DECISIONS.md).  So a mask that labels STINTS is read on `mse`; `game_bench_share_split` labels team-games
    and recombines to the pooled score exactly.
    """
    idx = np.arange(len(p.y)) if mask is None else np.flatnonzero(mask)
    y, w = p.y[idx], p.w[idx]

    def wmse(pred):
        return float(np.average((y - pred) ** 2, weights=w))

    one_scale = np.column_stack([p.A[idx], p.c_off[idx] + p.c_def[idx]])
    one_coef = weighted_least_squares(one_scale, y, w)
    two_scales = np.column_stack([p.A[idx], p.c_off[idx], p.c_def[idx]])
    two_coef = weighted_least_squares(two_scales, y, w)
    if mask is None or keeps_whole_team_games(p, idx):
        tg, tg_n = team_game_mse(y, p.pred[idx], p.poss[idx], p.game_idx[idx], p.is_home_off[idx])
        tg_base, _ = team_game_mse(y, p.base_pred[idx], p.poss[idx], p.game_idx[idx], p.is_home_off[idx])
    else:
        tg = tg_base = tg_n = np.nan
    return dict(n=float(w.sum()), mse=wmse(p.pred[idx]), base=wmse(p.base_pred[idx]), calib=wmse(one_scale @ one_coef),
                calib_side=wmse(two_scales @ two_coef), scale=float(one_coef[-1]), scale_off=float(two_coef[-2]),
                scale_def=float(two_coef[-1]), covered=float((p.rat.poss.to_numpy() > 0.0).mean()), tg=tg,
                tg_base=tg_base, tg_n=tg_n)


# ------------------------------------------------------------------------------------------- the splits
def movers_split(design, main_team_now: dict, main_team_before: dict) -> np.ndarray:
    """Each row by how many of the ten on the court changed team between the rated and the scored season.

    A player moved if his main team in the scored season (`main_team_now`, player_id -> team) differs from his
    main team in the rated season (`main_team_before`).  A player missing from either did not move.  This is the
    traded-to-a-new-team test: a rating that is really his old teammates' credit stops predicting once he plays
    with new ones.  It labels stints, so its groups are read on `mse`.
    """
    m = design.spec.n_players
    ids = design.spec.players["player_id"].to_numpy()
    moved = np.array([1.0 if (q in main_team_now and q in main_team_before
                              and main_team_now[q] != main_team_before[q]) else 0.0 for q in ids])
    n_movers = design.X[:, :2 * m] @ np.tile(moved, 2)
    return np.where(n_movers == 0, "no movers", np.where(n_movers <= 2, "1-2 movers", "3+ movers"))


def game_bench_share_split(p: Prediction, design, low_exposure: float = BARELY_SEEN_POSSESSIONS,
                           edges=GAME_BENCH_SHARE_EDGES) -> np.ndarray:
    """Each TEAM-GAME by the share of its possessions played by players the ratings barely saw.

    A player is barely seen with fewer than `low_exposure` possessions in the rated season (500: a deep bench
    player).  Each of the ten on the court counts as a tenth of the row's possessions.  The label is constant
    within a team-game, so the groups partition the test's own unit and their scores add up to the pooled one.

    Example: a team-game of 100 possessions in which two barely-seen players shared the court for 30 of them has
    a share of (2 x 30 / 10) / 100 = 6%, in the 5%-15% group.
    """
    m = design.spec.n_players
    edges = [float(e) for e in edges]
    low = (p.rat.poss.to_numpy(dtype=float) < float(low_exposure)).astype(float)
    n_low = design.X[:, :2 * m] @ np.tile(low, 2)
    d = pd.DataFrame({"g": p.game_idx, "h": p.is_home_off, "poss": p.poss, "lp": p.poss * n_low / 10.0})
    tg = d.groupby(["g", "h"]).sum()
    share = (tg.lp / tg.poss.clip(lower=1e-9)).rename("share")
    labels = [f"{a:.0%}-{b:.0%}" if b <= 1.0 else f"{a:.0%}+" for a, b in zip(edges[:-1], edges[1:])]
    b = np.clip(np.digitize(share.to_numpy(), edges) - 1, 0, len(labels) - 1)
    lab = pd.Series(np.asarray(labels, dtype=object)[b], index=share.index)
    return lab.reindex(pd.MultiIndex.from_arrays([d.g, d.h])).to_numpy()


# ----------------------------------------------------------------------------------------- the summaries
def pooled(res: pd.DataFrame, by=POOL_BY) -> pd.DataFrame:
    """Possession-weighted means over the scored seasons, one row per group of `by`.

    `vs_no_ratings` is the share of the level-only error the ratings remove; `share` is the group's share of the
    pooled rows.  The team-game columns are NaN for a group whose mask cut team-games (`score`).
    """
    def agg(d):
        n, tn = d.n.sum(), d.tg_n.sum()
        return pd.Series({
            "seasons": int(d.held_out.nunique()),
            "mse": float((d.mse * d.n).sum() / n),
            "base": float((d.base * d.n).sum() / n),
            "vs_no_ratings": 1.0 - float((d.mse * d.n).sum() / (d.base * d.n).sum()),
            "calib_side": float((d.calib_side * d.n).sum() / n),
            "scale_off": float((d.scale_off * d.n).sum() / n),
            "scale_def": float((d.scale_def * d.n).sum() / n),
            "covered": float(d.covered.mean()),
            "game": float((d.tg * d.tg_n).sum() / tn) if tn > 0 else np.nan,
            "game_base": float((d.tg_base * d.tg_n).sum() / tn) if tn > 0 else np.nan,
            "game_vs_no_ratings": (1.0 - float((d.tg * d.tg_n).sum() / (d.tg_base * d.tg_n).sum())
                                   if tn > 0 else np.nan),
            "share": float(n),
            "armse": armse((d.mse * d.n).sum() / n),
            "game_armse": armse((d.tg * d.tg_n).sum() / tn) if tn > 0 else np.nan,
        })
    out = res.groupby(list(by), sort=True).apply(agg, include_groups=False).reset_index()
    tot = out.groupby([b for b in by if b != "group"] if "group" in by else list(by))["share"].transform("sum")
    out["share"] = out["share"] / tot
    return out


def clustered_se(diff: pd.Series, cluster: pd.Series) -> float:
    """The standard error of the mean of `diff`, the readings that share a `cluster` counted together (the
    cluster-robust form): sqrt(G / (G - 1) x sum over the G clusters of (the sum of its residuals)^2) / n, residual =
    reading minus the mean.  With one reading per cluster it is the plain standard error, sd / sqrt(n), exactly.

    Example: readings -1.0 and -1.0 from one table and +0.5 and +0.5 from another.  The plain standard error is
    0.866 / sqrt(4) = 0.43; clustered, the residual sums are -1.5 and +1.5 and it is sqrt(2 x 4.5) / 4 = 0.75, since
    each table's two readings say the same thing twice."""
    residual = diff - diff.mean()
    sums = residual.groupby(np.asarray(cluster.reindex(diff.index))).sum()
    clusters, n = len(sums), len(diff)
    return float(np.sqrt(clusters / (clusters - 1) * np.sum(sums ** 2)) / n) if clusters > 1 else np.nan


def paired(res: pd.DataFrame, ref: str, value: str = "mse", by=("k", "lam", "split", "group")) -> pd.DataFrame:
    """Every system against `ref`, paired by scored season: the mean difference, its standard error, the paired
    t-statistic z = mean / standard error, and the seasons won.

    `value` is "mse" (stint), "tg" (team-game) or "calib_side".  A negative difference is better than `ref`.

    Beside them, when `res` names each reading's rated season (`train`), `se_clustered` and `z_clustered`: the same
    with the readings of one rated season's table counted together (`clustered_se`).  Both directions together hold
    two readings per table, the season before predicting forward and the season after predicting back, and the two
    move together; counted as independent they overstate the evidence (experiment 93 against 82: z -2.82, clustered
    -2.48; notes/DECISIONS.md, the bug-fix pass of 2026-10-22).  One direction alone has one reading per table, and
    the two columns agree.
    """
    rows = []
    for key, d in res.groupby(list(by), sort=True):
        p = d.pivot_table(index="held_out", columns="system", values=value)
        if ref not in p.columns:
            continue
        rated = (d.drop_duplicates("held_out").set_index("held_out")["train"] if "train" in d.columns else None)
        for s in p.columns:
            if s == ref:
                continue
            diff = (p[s] - p[ref]).dropna()
            n = len(diff)
            sd = diff.std(ddof=1) if n > 1 else np.nan
            se_clustered = clustered_se(diff, rated) if rated is not None and n > 1 else np.nan
            rows.append(dict(**dict(zip(by, key)), system=s, ref=ref, value=value, mean_diff=float(diff.mean()),
                             se=float(sd / np.sqrt(n)) if n > 1 else np.nan,
                             z=float(diff.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
                             wins=int((diff < 0).sum()), n_seasons=n, se_clustered=se_clustered,
                             z_clustered=(float(diff.mean() / se_clustered) if se_clustered and se_clustered > 0
                                          else np.nan)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ the quality tiers
def quality_tier(rank) -> np.ndarray:
    """The quality tier of each rank (1 = the season's best total rating).

    Example: ranks 1, 30, 31 and 301 give "top 30", "top 30", "31-90" and "301+".
    """
    return np.asarray(QUALITY_LABELS, dtype=object)[np.digitize(rank, QUALITY_EDGES[1:-1], right=True)]


def quality_tiers(table: pd.DataFrame) -> pd.Series:
    """(player_id, season) -> quality tier, from each player's rank by `rating_total` within his season.

    Ties keep the table's order.  One table fixes the tiers for every arm of a comparison, so all of them are cut
    the same way: name the reference's ratings.
    """
    rank = table.groupby("season").rating_total.rank(ascending=False, method="first").to_numpy()
    return pd.Series(quality_tier(rank), index=pd.MultiIndex.from_arrays([table.player_id.to_numpy(),
                                                                          table.season.to_numpy()]))
