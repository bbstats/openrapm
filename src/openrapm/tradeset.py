"""The trade set: what a player's absence says about him, one row per team-game.

The rating a season gives a player is fitted on the stints of that season alone.  Inside one season
the same five men keep appearing together, so how the credit for a good lineup is SPLIT among them is
barely identified -- moving a point from one to another leaves every lineup sum where it was.  The
thing that does identify it is a player being gone: a trade, an injury, a rotation change.  Across
three adjacent seasons roughly half a roster turns over, and every one of those departures is a
natural experiment the single-season fit never sees.

This module builds that experiment as a regression.  For a rated season:

  * rows are TEAM-GAMES, one per team per game, over the rated season and the two either side of it;
  * a player's entry in a row is his SHARE of that team's possessions in that game -- 0 when he did
    not play, and 0 for his old team's games after he is traded.  The five on the floor sum to 5;
  * the response is the team's points per 100 possessions on that side, on the same targets the
    rating was fitted on;
  * the season's rating is SUBTRACTED from every row, and what is left is ridged back onto the same
    player columns.  That coefficient is `alpha`: what the player's comings and goings say about him
    that his rating did not already know.

Every teammate has a column too, which is the whole reason for a regression rather than a difference
of averages.  A team's record without one player is mostly a statement about the other four; the
straight off-court number was tried on 2026-09-14 and failed exactly there, rating role players on
deep teams highly because their teams stayed good without them.  Here the other four are estimated
alongside him and the contrast is what is left.

**Alpha is not a rating and may never become one.**  It reads the games of the seasons either side,
so a published rating carrying it would break ruling 1 (one rating per player per season, from that
season's games only).  It is a diagnostic -- how much a season's rating misses, per player, with
bench players counted the same as starters -- and a training target for the box-score prior, whose
COEFFICIENTS are allowed to learn from other seasons (ruling 2).

A player who never misses a game has no with-and-without contrast of his own.  The contrast is
measured on the two NEIGHBOURING seasons only (`exposure_table` counts the rated season out), so
it is his absences there that carry it.
`exposure_table` measures that directly as `without_poss`, and everything downstream reads only
players for whom it is positive.

Aggregating to team-games, rather than fitting the stints, is deliberate and cheap: within a game the
same men are on the floor in most stints, so the with-and-without contrast lives between games, not
within them.  One season-triple aggregates to about 7,400 rows against about 1,500 player columns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .priorridge import MAE_SCALE, solve_diag      # MAE_SCALE: a mean square on the typical-miss scale

__all__ = ["TradeSet", "team_game_design", "replacement_fill", "offset_vector", "fit_alpha",
           "normal_equations", "solve_alpha", "rating_columns", "team_columns", "exposure_table",
           "eligibility", "trade_loss", "CONTROL_PREFIXES", "SIDES", "LOSS_COLUMNS"]

# The fixed columns kept when the stints are pooled into team-games.  Each is constant within a
# team-game, so its possession-weighted mean over the team-game is the value itself and nothing is
# smeared.  `is_gt`, `margin` and `margin_frem` are NOT: they vary stint by stint, and their weighted
# mean is a different covariate from the one the rating was fitted with.
CONTROL_PREFIXES = ("home", "int_", "is_po", "po_home")

SIDES = ("offense", "defense")


@dataclass
class TradeSet:
    """One season-triple pooled to team-games.

    Z       (n_team_games x 2*n_players) possession shares: columns 0..n_players-1 are the players on
            the floor for the team whose points the row counts, the rest are the five defending them.
            Each block sums to 5.0 in every row.
    F       (n_team_games x n_controls) the kept fixed columns, in `control_names` order.
    y       points per 100 possessions for that team in that game, on the design's target.
    w       possessions behind the row.
    keys    game_idx, is_home_off, season, phase, game_id -- one row per row of Z.
    """
    Z: sp.csr_matrix
    F: np.ndarray
    y: np.ndarray
    w: np.ndarray
    keys: pd.DataFrame
    player_ids: np.ndarray
    control_names: list

    @property
    def n_players(self) -> int:
        return len(self.player_ids)


def _fixed_block(wd) -> np.ndarray:
    """The design's dense fixed block, from `parts` when it is there and from X when it is not."""
    if wd.parts is not None and "F" in wd.parts:
        return np.asarray(wd.parts["F"], dtype=float)
    m, n_f = wd.spec.n_ps, len(wd.spec.f_names)
    return np.asarray(wd.X[:, 2 * m:2 * m + n_f].todense(), dtype=float)


def _player_block(wd) -> sp.csr_matrix:
    """The design's Z block (ones), from `parts` when it is there and from X when it is not."""
    if wd.parts is not None and "Z" in wd.parts:
        return sp.csr_matrix(wd.parts["Z"])
    return sp.csr_matrix(wd.X[:, :2 * wd.spec.n_ps])


def team_game_design(wd, control_prefixes=CONTROL_PREFIXES) -> TradeSet:
    """Pool a stint design into one row per team-game, possession-weighted.

    The weight is `rows["den"]`, the design's own denominator, never `w`: `w` is `den` times the
    garbage-time weight and the project runs that at 0.5 and 0 for robustness, which would silently
    reweight the shares as well as the rows.  For every target used here `den` is possessions.
    """
    rows = wd.rows
    den = rows["den"].to_numpy(dtype=float)
    if not np.all(den > 0):
        raise ValueError("a design row with no possessions reached the trade set")
    game = rows["game_idx"].to_numpy(dtype=np.int64)
    home = rows["is_home_off"].to_numpy().astype(np.int64)
    key, inverse = np.unique(np.stack([game, home]), axis=1, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    n_rows, n_groups = len(den), key.shape[1]

    pool = sp.csr_matrix((den, (inverse, np.arange(n_rows))), shape=(n_groups, n_rows))
    weight = np.asarray(pool @ np.ones(n_rows))
    inv_weight = sp.diags(1.0 / weight)

    Z = inv_weight @ (pool @ _player_block(wd))
    Z = sp.csr_matrix(Z)
    Z.eliminate_zeros()
    y = np.asarray(pool @ wd.y) / weight

    names = list(wd.spec.f_names)
    keep = [j for j, name in enumerate(names)
            if any(name == p or name.startswith(p) for p in control_prefixes)]
    F = (np.asarray(pool @ _fixed_block(wd)[:, keep]) / weight[:, None]) if keep else np.zeros((n_groups, 0))

    games = wd.games.drop_duplicates("game_idx").set_index("game_idx")
    game_of_group = key[0]
    keys = pd.DataFrame({
        "game_idx": game_of_group,
        "is_home_off": key[1].astype(bool),
        "season": games["season"].reindex(game_of_group).to_numpy(),
        "phase": games["phase"].reindex(game_of_group).to_numpy(),
        "game_id": games["game_id"].reindex(game_of_group).to_numpy(),
    })
    return TradeSet(Z=Z, F=F, y=y, w=weight, keys=keys,
                    player_ids=wd.spec.ps_table["player_id"].to_numpy(dtype=np.int64),
                    control_names=[names[j] for j in keep])


def replacement_fill(ratings, max_poss: float = 500.0, shrink: float = 0.25) -> tuple:
    """What a player the rated season's table has no row for is worth, per side, in raw sign.

    The same rule as `holdout.ReplacementSystem`: the possession-weighted mean rating of the table's
    own players under `max_poss` possessions, times `shrink`.  The year-over-year test settled on a
    quarter of that level, the only depth better in both directions.  `ratings` is one season, with
    columns `o`, `d` (raw sign: `d` is points allowed) and `poss`.
    """
    low = ratings[(ratings.poss > 0) & (ratings.poss < float(max_poss))]
    if len(low) == 0:
        return 0.0, 0.0
    weight = low.poss.to_numpy(dtype=float)
    return (float(shrink) * float(np.average(low.o, weights=weight)),
            float(shrink) * float(np.average(low.d, weights=weight)))


def offset_vector(player_ids, ratings, fill_o: float = 0.0, fill_d: float = 0.0) -> np.ndarray:
    """The rated season's rating for every player column, raw sign, offence then defence.

    Raw sign throughout: `o` adds points scored and `d` adds points ALLOWED, which is the sign the
    design's defensive columns carry.  A rating table stored positive-good has already had its
    defensive column negated on the way in (`scripts/63_yoy.py: load_table`); negating it a second
    time here would double the defensive signal and look like a result.
    """
    aligned = (pd.DataFrame({"player_id": np.asarray(player_ids, dtype=np.int64)})
               .merge(ratings[["player_id", "o", "d"]], on="player_id", how="left")
               .fillna({"o": float(fill_o), "d": float(fill_d)}))
    return np.concatenate([aligned.o.to_numpy(dtype=float), aligned.d.to_numpy(dtype=float)])


def rating_columns(ts: TradeSet, offset: np.ndarray) -> np.ndarray:
    """The rating summed over the five on the floor, per side: two columns, one per side.

    Handed to the fit unpenalised, these let the trade set say the whole rating is too wide or too
    narrow in ONE coefficient per side instead of writing the same tilt into every player's alpha.
    Without them a rating whose spread is 10% too wide gives every strong player a negative alpha and
    every weak one a positive alpha, and the correction is a rescale wearing a per-player costume.
    `PriorRidgeCV(free_prior_scale=True)` does the same thing at the stint level; the fitted
    coefficient plus one is the multiplier that side is asking for.
    """
    n = ts.n_players
    offset = np.asarray(offset, dtype=float)
    return np.column_stack([ts.Z[:, :n] @ offset[:n], ts.Z[:, n:] @ offset[n:]])


def team_columns(ts: TradeSet, team_of_row: pd.DataFrame, per_season: bool = True) -> tuple:
    """One free column per team on offence and per team on defence: (matrix, names).

    This is the control that answers the objection that sank the off-court record.  A team's output
    is the sum of its players' columns, so without a team term nothing stops a player's alpha from
    absorbing "his team was good that year" -- a role player on a deep team reads as good because the
    team is good with him too.  Freeing the team's own level per side leaves the player columns
    identified only by variation in WHO WAS ON THE FLOOR inside that team, which is the with-and-
    without contrast and nothing else.

    `per_season=True` frees each team's level in each season separately: the strictest version, under
    which a man who played every game of his team's season is collinear with his team's own column and
    keeps nothing.  `per_season=False` frees one level per team across the whole block, which removes
    persistent team quality -- the franchise, the coaching, the building -- while leaving a team's
    change from one season to the next available to identify the players who moved.

    Two sets of columns, offence and defence, because a team's scoring and its defending are separate
    facts and a single column per team would force them to share one number.
    """
    # The same guard `exposure_table` has.  `team_of_row` is built from ONE side's design and reused
    # for the other in scripts/70_tradeset.py; if the two ever stop agreeing row for row, every team
    # column here is assembled against the wrong rows and the fit still returns numbers.
    if len(team_of_row) != len(ts.keys):
        raise ValueError("team_of_row must have one row per team-game, in the design's order")
    season = ts.keys["season"].to_numpy() if per_season else np.zeros(len(ts.keys), dtype=np.int64)
    rows, columns, names = [], [], []
    for side, column in (("offense", "team_off"), ("defense", "team_def")):
        team = team_of_row[column].to_numpy(dtype=np.int64)
        label = pd.Series(list(zip(team, season)))
        for key, index in label.groupby(label).groups.items():
            rows.append(np.asarray(index, dtype=np.int64))
            columns.append(len(names))
            names.append(f"{side} {key[0]}" + (f" {key[1]}" if per_season else ""))
    data = np.concatenate([np.ones(len(r)) for r in rows]) if rows else np.zeros(0)
    row_index = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    col_index = np.concatenate([np.full(len(r), c, dtype=np.int64) for r, c in zip(rows, columns)]) \
        if rows else np.zeros(0, dtype=np.int64)
    matrix = sp.csr_matrix((data, (row_index, col_index)), shape=(len(ts.keys), len(names)))
    return matrix, names


def normal_equations(ts: TradeSet, offset: np.ndarray, free_scale: bool = True, extra=None) -> tuple:
    """(gram, rhs, n_free) of the weighted least squares for `y - Z @ offset` on [Z | scale | F | extra].

    Built once per season and reused at every penalty: the grid is then a sequence of solves of a
    matrix that is already formed, which is what makes a penalty sweep cheap here.  `n_free` counts
    the unpenalised columns at the end -- the two rating columns when `free_scale`, then the controls,
    then `extra` (`team_columns`, when it is asked for).
    """
    blocks = [ts.Z]
    if free_scale:
        blocks.append(sp.csr_matrix(rating_columns(ts, offset)))
    blocks.append(sp.csr_matrix(ts.F))
    n_extra = 0
    if extra is not None and extra.shape[1] > 0:
        blocks.append(sp.csr_matrix(extra))
        n_extra = extra.shape[1]
    design = sp.hstack(blocks, format="csr")
    residual = ts.y - ts.Z @ np.asarray(offset, dtype=float)
    weighted = design.multiply(ts.w[:, None]).tocsr()
    gram = np.asarray((design.T @ weighted).todense(), dtype=float)
    rhs = np.asarray(design.T @ (ts.w * residual), dtype=float).ravel()
    return gram, rhs, (2 if free_scale else 0) + len(ts.control_names) + n_extra


def solve_alpha(gram, rhs, n_players: int, n_free: int, lam: float) -> tuple:
    """(alpha, free) from normal equations already formed: the penalty sweep's inner loop.

    Every player column carries the penalty; the `n_free` columns at the end carry none, so each
    season's level, the home edge, the playoff terms and the two rating scales are unshrunk.  Taking
    the gram and the right-hand side rather than a TradeSet is what lets a whole grid of penalties be
    solved without the design in memory.
    """
    n = 2 * int(n_players)
    penalty = np.concatenate([np.full(n, float(lam)), np.zeros(int(n_free))])
    solution = solve_diag(gram, rhs, penalty)
    return solution[:n], solution[n:]


def fit_alpha(ts: TradeSet, offset: np.ndarray, lam: float, free_scale: bool = True) -> tuple:
    """(alpha, free): the ridged correction on every player column, and the unpenalised coefficients.

    `alpha` is 2*n_players long, offence then defence, in raw sign and in the same units as the
    rating -- points per 100 possessions.  When `free_scale`, the first two of `free` are the rating
    scales MINUS ONE, offence then defence.
    """
    gram, rhs, n_free = normal_equations(ts, offset, free_scale=free_scale)
    return solve_alpha(gram, rhs, ts.n_players, n_free, lam)


def exposure_table(ts: TradeSet, team_of_row: pd.DataFrame, team_of_player: dict, season: int) -> pd.DataFrame:
    """How much evidence each player's own comings and goings carry, per side.

    `with_poss`     his on-floor possessions over the whole three seasons, on that side.
    `without_poss`  possessions his RATED-SEASON team played in the neighbouring seasons with him not
                    on the floor -- games he missed, plus his old team's games once he is traded, plus
                    his new team's games before he arrives.  This is the trade set's actual sample
                    size for him, and it is zero for a player who never left the floor and for one
                    whose team is unknown.

    `team_of_row` must have `team_off` and `team_def` for every row of `ts` in row order: the team
    whose points the row counts and the team defending it.  A player's DEFENSIVE evidence lives in the
    rows where his team is defending, which are the opponent's offensive rows of the same games.
    """
    if len(team_of_row) != len(ts.keys):
        raise ValueError("team_of_row must have one row per team-game, in the design's order")
    n = ts.n_players
    team_off = team_of_row["team_off"].to_numpy(dtype=np.int64)
    team_def = team_of_row["team_def"].to_numpy(dtype=np.int64)
    neighbour = (ts.keys["season"].to_numpy(dtype=np.int64) != int(season))

    rated_team = np.array([int(team_of_player.get(int(p), -1)) for p in ts.player_ids], dtype=np.int64)
    known = rated_team >= 0

    with_poss = np.asarray(ts.Z.T @ ts.w).ravel()

    coo = ts.Z.tocoo()
    is_offense = coo.col < n
    column_player = np.where(is_offense, coo.col, coo.col - n)
    row_team = np.where(is_offense, team_off[coo.row], team_def[coo.row])
    played_for_rated_team = (neighbour[coo.row] & (row_team == rated_team[column_player])
                             & known[column_player])
    played = np.bincount(coo.col[played_for_rated_team], weights=ts.w[coo.row[played_for_rated_team]],
                         minlength=2 * n)

    # every neighbouring-season team-game of each team, on each side
    total_off = pd.Series(ts.w[neighbour]).groupby(team_off[neighbour]).sum()
    total_def = pd.Series(ts.w[neighbour]).groupby(team_def[neighbour]).sum()
    available = np.concatenate([total_off.reindex(rated_team).fillna(0.0).to_numpy(),
                                total_def.reindex(rated_team).fillna(0.0).to_numpy()])
    available = np.where(np.concatenate([known, known]), available, 0.0)

    return pd.DataFrame({
        "player_id": np.concatenate([ts.player_ids, ts.player_ids]),
        "side": np.repeat(np.asarray(SIDES), n),
        "team_id": np.concatenate([rated_team, rated_team]),
        "with_poss": with_poss,
        "without_poss": np.maximum(available - played, 0.0),
    })


def eligibility(roles: pd.DataFrame, season: int, min_poss: float = 100.0) -> pd.DataFrame:
    """Players whose rated season is one team and enough of it: player_id, team_id, poss_on.

    One team, because a player traded mid-season has no single rated-season team for the neighbouring
    games to be compared against -- his alpha is still estimated, it is simply not read.  Roster rows
    for a man who dressed and never played carry `poss_on` 0 and would otherwise make a one-team
    player look like a two-team one, so they go first.
    """
    played = roles[(roles.season == int(season)) & (roles.poss_on > 0)]
    total = played.groupby("player_id", as_index=False).agg(poss_on=("poss_on", "sum"),
                                                            n_teams=("team_id", "nunique"))
    main = (played.sort_values("poss_on").drop_duplicates("player_id", keep="last")
            [["player_id", "team_id"]])
    out = total.merge(main, on="player_id")
    return out[(out.n_teams == 1) & (out.poss_on >= float(min_poss))][["player_id", "team_id", "poss_on"]]


# What `trade_loss` returns, in order.  Named so an empty result carries the same columns as a full
# one and every caller can group by `side` without asking whether there was anything to group.
LOSS_COLUMNS = {"season": "int64", "side": "object", "tier": "object", "players": "float64",
                "missed": "float64", "missed_eff": "float64", "points": "float64"}

TIERS = [(0.0, 500.0, "under 500 possessions"),
         (500.0, 1500.0, "500 to 1,500 possessions"),
         (1500.0, np.inf, "over 1,500 possessions")]


def possession_tier(poss) -> np.ndarray:
    """The three exposure tiers, so the loss can be read for bench players separately."""
    poss = np.asarray(poss, dtype=float)
    out = np.empty(len(poss), dtype=object)
    for low, high, name in TIERS:
        out[(poss >= low) & (poss < high)] = name
    return out


def trade_loss(alpha_table: pd.DataFrame, min_without: float = 1.0) -> pd.DataFrame:
    """What the trade set says a season's ratings cost, per side, every player counting once.

    Three numbers per group, over players the trade set actually has evidence on:

    `missed`      the mean of alpha squared, on the mean-absolute scale (`sqrt` times sqrt(2/pi)) so
                  it reads as the points per 100 a typical player's rating is out by.
    `missed_eff`  the same, weighting each player by the effective sample of his own with-and-without
                  contrast, `with * without / (with + without)`.  A player who missed two games and a
                  player who missed forty do not know the same amount.
    `points`      the sum over players of |alpha| times his rated-season possessions over 100: points
                  a season mispriced, which is the shape a dollars loss takes.

    Reported pooled and by possession tier, because a team-game criterion cannot see the bench and
    this is the loss built to.
    """
    rows = alpha_table[(alpha_table.eligible) & (alpha_table.without_poss >= float(min_without))].copy()
    # No player with a with-and-without contrast is a legitimate answer, not a failure: `--block=0`
    # fits one season on its own, so nobody's team plays a neighbouring game without him.  Return the
    # schema anyway -- an empty frame with no columns reaches the caller as `KeyError: 'side'`, which
    # reads like a bug in the fit rather than the absence of evidence it is.
    if rows.empty:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in LOSS_COLUMNS.items()})
    rows["tier"] = possession_tier(rows.poss_on.to_numpy())
    effective = (rows.with_poss * rows.without_poss / (rows.with_poss + rows.without_poss)).to_numpy()
    rows["effective_poss"] = np.where(np.isfinite(effective), effective, 0.0)

    def one(group: pd.DataFrame) -> pd.Series:
        alpha = group.alpha_good.to_numpy(dtype=float)
        weight = group.effective_poss.to_numpy(dtype=float)
        weighted = (float(np.average(alpha ** 2, weights=weight)) if weight.sum() > 0 else np.nan)
        return pd.Series({
            "players": float(len(group)),
            "missed": MAE_SCALE * float(np.sqrt(np.mean(alpha ** 2))),
            "missed_eff": MAE_SCALE * float(np.sqrt(weighted)),
            "points": float(np.sum(np.abs(alpha) * group.poss_on.to_numpy(dtype=float) / 100.0)),
        })

    pooled = rows.groupby(["season", "side"], as_index=False).apply(one, include_groups=False)
    pooled["tier"] = "all"
    by_tier = rows.groupby(["season", "side", "tier"], as_index=False).apply(one, include_groups=False)
    return pd.concat([pooled, by_tier], ignore_index=True)
