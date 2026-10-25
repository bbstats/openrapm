"""Check 154, self-creation in lineups: on games the ratings never saw, do the amount and the concentration of
self-creation among the five on offense move a lineup's points per 100 away from what OpenRAPM expects, and does it
show more after a trade?

The owner, 2026-10-22: RAPM does not pick up self-creation well, and team evaluations read through salaries look like
"PI RAPM + usage".  OpenRAPM predicts every lineup as the plain sum of its ten players' ratings, so two lineups with the
same ratings are predicted alike however their self-creation is piled up or spread out; this check asks whether
held-back games say otherwise (notes/DECISIONS.md, "Self-creation in lineups").

Words used below
    self-creation   the owner's formula: true shooting attempts per 100 x (1 - share assisted).  True shooting
                    attempts are field-goal attempts + 0.44 x free-throw attempts; the share assisted is
                    0.73 - 0.03 x assists per 100 - 0.01 x field goals made per 100, clipped to 0 through 1.  Per 100
                    of the player's offensive possessions, from the padded rates of the fold's fit games, the inputs
                    the fold's box-score prior reads, so no held-back game is in it
    amount          a stint's sum of self-creation over the five on offense
    concentration   1 - the sum of the five's squared shares of that sum, the Gini-Simpson index the owner chose for
                    team movement: 0.8 when five hold equal shares, 0 when one player holds it all
    miss            a held-back stint's points per 100 minus its ten players' ratings after the shipped shrink, both
                    layers and the defense factor, exactly as check 151 builds it (a stand-in rating for a player the
                    fold did not rate)
    free terms      what the miss may also depend on: a level for every attacking and every defending team in each
                    fold, home and playoffs (check 151's team levels); garbage time, the score margin and the margin
                    times the share of the game left (the design's context columns, which the miss leaves out); the
                    sums of the offense's and the defense's ratings, so a ratings width that is off cannot pass for
                    creation (the two tests read width in opposite directions: notes/DECISIONS.md, "The widths"); the
                    five's summed share of team possessions, since heavy-minute lineups outscore their offensive
                    ratings in held-back stints; and how many of the ten have no fold rating
    lineup shape    the owner, 2026-10-24: "if good lineups in general are multiplicative as are high concentration
                    ones, it should be relatively easy to tease out that benefit from that benefit plus self-creation".
                    Four more free terms: the square of the offense's rating sum, the square of the defense's, their
                    product, each centred on the fold's possession-weighted mean (good lineups multiplying), and the
                    standard deviation of the five offensive ratings, the talent concentrated in one or two players at
                    the same sum (high concentration multiplying)
    career null     check 138's yardstick on this one input: every player is handed another player's self-creation,
                    season by season along that player's career (the donor's last season when his career is shorter),
                    and every reading is refitted.  A lineup sum of ANY number a player carries picks up the misfit
                    of the players who carry it, and a player's misfit carries over from season to season, so the
                    spread of the per-season estimates understates the uncertainty; the null keeps that carry-over
    after a trade   on the date folds of `openrapm folds --cut`: a held-back stint whose offense lost or gained, from
                    the first held-back game on, a rotation player who moved between teams that season, 20 or more
                    minutes a game for the team he left or the one he joined; the move dated by his first game for
                    his new team.  The box scores cannot tell a trade from a release and a signing, so both count

The fit, per season, its folds pooled (weighted least squares, weight = possessions):

    miss  =  free terms  +  a x amount  +  c x concentration

each fold's team levels, home and playoffs fitted out of every column first, which gives the same a and c as one fit
with them in.  a and c are reported per standard deviation of the season's amount and concentration over its
held-back possessions: points per 100 beyond what OpenRAPM expects.  Read twice: with the free terms, and with the
lineup shape held as well, beside the lineup-shape terms' own effects in that fit: each square at one standard
deviation of its rating sum, the product at one of each, the talent spread per standard deviation.  With --trade=1
the amount and the concentration also get a coefficient on the stints after a trade, the difference, beside a free
level for those stints; both are centred on the fold's possession-weighted mean first, so that level reads at the
fold's average lineup.

Across seasons: the mean of the per-season estimates, its standard error across seasons, the z, the Newey-West z
(check 151's `newey_west_se`), the seasons with the mean's sign, each era; beside it the same fit with no team levels.
Against the career null (--null draws): the null's mean and standard deviation of that same across-season mean, the z
against it, and the share of draws at least as far from the null's mean.

The table: each season's held-back stints in fifths of the amount and, within each fifth, thirds of the concentration
(possession-weighted); each cell's mean miss after the free terms alone, in points per 100, and its share of the
possessions, pooled over the seasons.

Example: one 2026 fold gives LeBron James 10.2 assists and 11.3 field goals made per 100, so a share assisted of
0.73 - 0.306 - 0.113 = 0.311, and 25.76 true shooting attempts per 100, so self-creation 25.76 x 0.689 = 17.75.

With no terms but the team levels each season's error is check 151's `step4` / `all` error; --check=FILE:SYSTEM prints
the largest difference against a check 151 seasons file.

    python scripts/checks/154_self_creation.py --folds=parts107 --tag=selfcreation
    python scripts/checks/154_self_creation.py --folds=creation111 --creation=actual_makes --tag=selfcreation_actual
        (the self-creation of experiment 111: his own unassisted makes, the formula on the misses and free throws)
    python scripts/checks/154_self_creation.py --folds=parts107_cut50 --step4-from=parts107 --trade=1 --null=0
        --top=0 --tag=selfcreation_trade
        (each season rated from its first half and scored on the rest, with the shrink of whole seasons)

Writes outputs/csv/<tag>_seasons.csv (every season, reading and term), <tag>.csv (across seasons), <tag>_null.csv and
<tag>_null_seasons.csv (the career null), <tag>_table.csv (the fifths and thirds, pooled and by era) and, with --top,
<tag>_top.csv (that season's players by self-creation).  With --save-rows=DIR, DIR/<season>.parquet: every held-back
stint with its possessions, the five's summed self-creation, the miss, the miss after the team levels, and what is
above expected after every free term and the lineup shape, the y of the table, so other cuts need no rerun.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from openrapm.commands.console import one_thread, use_utf8
from openrapm.settings import PRODUCT, factor_for

SHARE_ASSISTED = (0.73, 0.03, 0.01)     # the owner's formula, 2026-10-22: level, per assist per 100, per make per 100
FREE_THROW_WEIGHT = 0.44                # true shooting attempts = field-goal attempts + 0.44 x free-throw attempts
STAND_IN_POSS = 500                     # an unrated player carries the mean self-creation of the players under this
TRADE_MINUTES = 20.0                    # a move counts when its player played this many minutes a game for either team
TRADE_MIN_POSS = 1000                   # a season's after-a-trade reading needs this many held-back possessions in it
TOP_POSS = 1500                         # the players table: this many offensive possessions or more
NULL_SEED = 20261024
DEFENCE_FACTORS = PRODUCT.shrink.defence_factors
ERAS = ((1997, 2006), (2007, 2016), (2017, 2026))
TERMS = ("amount", "concentration")
TRADE_TERMS = ("amount_after_trade", "concentration_after_trade", "trade")
CONTEXT = ("is_gt", "margin", "margin_frem")
FREE = (*CONTEXT, "rating_sum_off", "rating_sum_def", "playing_time", "stand_ins")
SHAPE = ("rating_curve_off", "rating_curve_def", "rating_cross", "talent_spread_off")
SHAPE_NAMES = ("offense rating sum squared, at one standard deviation",
               "defense rating sum squared, at one standard deviation",
               "offense times defense rating sums, at one standard deviation each",
               "spread of the five offensive ratings, per standard deviation")
READINGS = (("within team", FREE), ("within team, lineup shape held", (*FREE, *SHAPE)))
RAW = (*TERMS, "rating_sum_off", "rating_sum_def", "talent_spread_off")
FIFTHS, THIRDS = 5, 3
CREATION_READ = {"name": "owner"}       # --creation: "owner" (the formula above) or "actual_makes" (experiment 111)


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # its dataclasses look their module up while being defined
    spec.loader.exec_module(module)
    return module


def eras() -> tuple:
    """(label, (first, last)): every season, then each era."""
    return (("all", (1997, 2026)), *((f"{first}-{last}", (first, last)) for first, last in ERAS))


# ------------------------------------------------------------------------------------------------ the metric
def share_assisted(assists, makes) -> np.ndarray:
    """The owner's estimate of the share of a player's shots that were assisted, clipped to 0 through 1."""
    level, per_assist, per_make = SHARE_ASSISTED
    raw = level - per_assist * np.asarray(assists, dtype=float) - per_make * np.asarray(makes, dtype=float)
    return np.clip(raw, 0.0, 1.0)


def self_creation(rows: pd.DataFrame) -> np.ndarray:
    """Self-created true shooting attempts per 100, from the padded per-100 rates with the league level kept; with
    --creation=actual_makes, his own unassisted makes in place of the estimate on his makes
    (openrapm.features.CREATION_ACTUAL_MAKES, experiment 111)."""
    if CREATION_READ["name"] == "actual_makes":
        from openrapm.features import CREATION_ACTUAL_MAKES, UNCENTRED_EVENT_RATES, UNCENTRED_RATES, add_features
        frame = rows[[*UNCENTRED_RATES, *UNCENTRED_EVENT_RATES]].reset_index(drop=True)
        return add_features(frame, CREATION_ACTUAL_MAKES)[CREATION_ACTUAL_MAKES[0]].to_numpy(float)

    def rate(name: str) -> np.ndarray:
        return rows[f"{name}_per100_uncentred"].to_numpy(float)
    makes = rate("twos_made") + rate("threes_made")
    attempts = (makes + rate("twos_missed") + rate("threes_missed")
                + FREE_THROW_WEIGHT * (rate("free_throws_made") + rate("free_throws_missed")))
    return attempts * (1.0 - share_assisted(rate("assists"), makes))


def concentration(values: np.ndarray) -> np.ndarray:
    """1 - the sum of squared shares across each row's values (the Gini-Simpson index); 0 for a row summing to 0."""
    values = np.asarray(values, dtype=float)
    total = values.sum(axis=1, keepdims=True)
    shares = np.divide(values, total, out=np.zeros_like(values), where=total > 0)
    return np.where(total.ravel() > 0, 1.0 - (shares ** 2).sum(axis=1), 0.0)


def fold_players(fold) -> tuple[dict, float, dict]:
    """({player id: self-creation}, the stand-in value, {player id: share of team possessions}) from the fold's
    offensive rows, its fit games only."""
    rows = fold.rows["O"]
    values = self_creation(rows)
    poss = rows.poss_off.to_numpy(float)
    low = poss < STAND_IN_POSS
    stand_in = float(np.average(values[low], weights=poss[low])) if poss[low].sum() > 0 else float(values.mean())
    ids = rows.player_id.to_numpy()
    return dict(zip(ids, values)), stand_in, dict(zip(ids, rows.team_possession_share.to_numpy(float)))


# ------------------------------------------------------------------------------------------- one fold's stints
def held_back_rows(teams: dict, design) -> pd.DataFrame:
    """The season design's stint rows of the fold's held-back team-games, with both teams, as check 151's
    `stint_fold_errors` finds them; `index` is the row's place in the design."""
    key = pd.DataFrame({"game_id": teams["game_id"].astype(str), "is_home_off": teams["is_home_off"].astype(bool),
                        "team_off": teams["team_off"], "team_def": teams["team_def"]})
    games = design.games.set_index("game_idx").game_id.astype(str)
    rows = design.rows.assign(game_id=design.rows.game_idx.map(games).to_numpy())
    return rows.reset_index().merge(key, on=["game_id", "is_home_off"], how="inner")


def lineup_ids(design, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row's five offensive and five defensive player ids."""
    columns = design.spec.players.set_index("player_column").player_id
    return tuple(columns.reindex(design.parts[part][index].ravel()).to_numpy().reshape(-1, 5)
                 for part in ("lineup_o", "lineup_d"))


def mapped(lineup: np.ndarray, values: dict, fill: float) -> tuple[np.ndarray, np.ndarray]:
    """(each player's value, `fill` where he has none, as check 151 fills a stand-in rating; where he had none)."""
    raw = pd.DataFrame(lineup).apply(lambda column: column.map(values)).to_numpy(float)
    missing = np.isnan(raw)
    return np.where(missing, fill, raw), missing


def context_columns(design, index: np.ndarray) -> dict:
    """The design's garbage-time, margin and margin-by-time-left columns for the rows."""
    context = design.parts["F"][index]
    context = context.toarray() if sp.issparse(context) else np.asarray(context)
    names = list(design.spec.context_names)
    return {name: context[:, names.index(name)].astype(float) for name in CONTEXT}


def stint_frame(wowy, audit, fold, teams: dict, design, step4: tuple) -> tuple[pd.DataFrame, np.ndarray, float]:
    """(one row per held-back stint: its miss, weight, teams, free terms and lineup self-creation; the five offensive
    player ids of each row; the stand-in self-creation), for one fold."""
    rows = held_back_rows(teams, design)
    index = rows["index"].to_numpy()
    lineup_o, lineup_d = lineup_ids(design, index)
    rating_off, allowed = wowy.stage_ratings(audit, fold, step4)["step4"]
    sum_off, missing_o = mapped(lineup_o, rating_off, fold.fill_o)
    sum_def, missing_d = mapped(lineup_d, allowed, fold.fill_d)
    creation, stand_in, share = fold_players(fold)
    created = mapped(lineup_o, creation, stand_in)[0]
    frame = pd.DataFrame({"game_idx": rows.game_idx.to_numpy(), "team_off": rows.team_off.to_numpy(),
                          "team_def": rows.team_def.to_numpy(), "is_home_off": rows.is_home_off.to_numpy(float),
                          "playoffs": (rows.phase == "PO").to_numpy(float), "y": design.y[index], "w": design.w[index],
                          "rating_sum_off": sum_off.sum(axis=1), "rating_sum_def": sum_def.sum(axis=1),
                          "talent_spread_off": sum_off.std(axis=1),
                          "amount": created.sum(axis=1), "concentration": concentration(created),
                          "playing_time": mapped(lineup_o, share, 0.0)[0].sum(axis=1),
                          "stand_ins": (missing_o.sum(axis=1) + missing_d.sum(axis=1)).astype(float),
                          **context_columns(design, index)})
    frame["miss"] = frame.y.to_numpy() - (frame.rating_sum_off.to_numpy() + frame.rating_sum_def.to_numpy())
    return frame, lineup_o, stand_in


def add_trade(frame: pd.DataFrame, moves: pd.DataFrame) -> list:
    """The after-a-trade flag and its two interactions; their names."""
    first = frame.game_idx.min()
    start = moves[moves.game_idx >= first].groupby("team_id").game_idx.min()
    since = frame.team_off.map(start).to_numpy(float)
    trade = (np.nan_to_num(since, nan=np.inf) <= frame.game_idx.to_numpy(float)).astype(float)
    w = frame.w.to_numpy(float)
    frame["trade"] = trade
    for term in TERMS:
        values = frame[term].to_numpy(float)
        frame[f"{term}_after_trade"] = (values - np.average(values, weights=w)) * trade
    return list(TRADE_TERMS)


# ------------------------------------------------------------------------------------------- the regressions
def team_levels(frame: pd.DataFrame, by_team: bool = True) -> sp.csr_matrix:
    """A free level for every attacking and every defending team, home and playoffs, as check 151 fits them in a
    fold; without `by_team`, one level for the fold, home and playoffs."""
    n = len(frame)
    home = sp.csr_matrix(frame.is_home_off.to_numpy(float)[:, None])
    playoffs = sp.csr_matrix(frame.playoffs.to_numpy(float)[:, None])
    if not by_team:
        return sp.hstack([sp.csr_matrix(np.ones((n, 1))), home, playoffs]).tocsr()
    off_codes, _ = pd.factorize(frame.team_off)
    def_codes, _ = pd.factorize(frame.team_def)
    return sp.hstack([
        sp.csr_matrix((np.ones(n), (np.arange(n), off_codes)), shape=(n, off_codes.max() + 1)),
        sp.csr_matrix((np.ones(n), (np.arange(n), def_codes)), shape=(n, def_codes.max() + 1))[:, 1:],
        home, playoffs]).tocsr()


def project(levels: sp.csr_matrix, w: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Each column with the levels fitted out by weighted least squares, as check 151's `wowy_miss` does the miss."""
    root = np.sqrt(w)
    design = levels.multiply(root[:, None]).toarray()
    coef, *_ = np.linalg.lstsq(design, columns * root[:, None], rcond=None)
    return columns - np.asarray(levels @ coef)


def projector(levels: sp.csr_matrix, w: np.ndarray) -> Callable:
    """The same projection as `project` from the levels' normal equations, cheap enough to repeat for every draw of
    the career null."""
    weighted = sp.csr_matrix(levels.multiply(w[:, None]))
    inverse = np.linalg.pinv((levels.T @ weighted).toarray())
    return lambda columns: columns - np.asarray(levels @ (inverse @ np.asarray(weighted.T @ columns)))


def fit(target: np.ndarray, columns: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares coefficients of `columns` for `target`."""
    root = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(columns * root[:, None], target * root, rcond=None)
    return coef


def weighted_sd(values: np.ndarray, w: np.ndarray) -> float:
    """The possession-weighted standard deviation."""
    mean = np.average(values, weights=w)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=w)))


@dataclass
class NullFold:
    """What one fold needs for the career null: its lineups as places in its sorted player ids, its stand-in
    self-creation, and the projection that fits its team levels out of a column."""
    players: np.ndarray
    codes: np.ndarray
    stand_in: float
    project: Callable


def add_shape(frame: pd.DataFrame, w: np.ndarray) -> None:
    """The lineup-shape terms beside the talent spread: each rating sum's square and their product, centred on the
    fold's possession-weighted means."""
    offense, defense = frame.rating_sum_off.to_numpy(float), frame.rating_sum_def.to_numpy(float)
    offense, defense = offense - np.average(offense, weights=w), defense - np.average(defense, weights=w)
    frame["rating_curve_off"], frame["rating_curve_def"] = offense ** 2, defense ** 2
    frame["rating_cross"] = offense * defense


def fold_part(wowy, audit, fold, teams: dict, design, step4: tuple, moves) -> tuple[pd.DataFrame, NullFold]:
    """(one fold's held-back stints with possessions: every column with the fold's team levels fitted out, p_; the
    miss and the terms with only a level, home and playoffs fitted out, q_; the raw terms and the weight; what the
    career null needs of the fold)."""
    frame, lineup_o, stand_in = stint_frame(wowy, audit, fold, teams, design, step4)
    extra = add_trade(frame, moves) if moves is not None else []
    levels = team_levels(frame)
    keep = frame.w.to_numpy() > 0
    frame, levels, lineup_o = frame[keep].reset_index(drop=True), levels[keep], lineup_o[keep]
    w = frame.w.to_numpy(float)
    add_shape(frame, w)
    names = [*TERMS, *FREE, *SHAPE, *extra]
    out = pd.DataFrame(project(levels, w, frame[names].to_numpy(float)), columns=[f"p_{n}" for n in names])
    out["p_miss"] = wowy.wowy_miss(frame.miss.to_numpy(float), levels, w)
    plain = [*TERMS, *FREE, "miss"]
    level_only = project(team_levels(frame, by_team=False), w, frame[plain].to_numpy(float))
    out = pd.concat([out, pd.DataFrame(level_only, columns=[f"q_{n}" for n in plain])], axis=1)
    for name in (*RAW, *(n for n in extra if n == "trade")):
        out[f"raw_{name}"] = frame[name].to_numpy(float)
    out["w"], out["raw_miss"] = w, frame.miss.to_numpy(float)
    out["game_idx"], out["team_off"], out["team_def"] = frame.game_idx, frame.team_off, frame.team_def
    out["repeat"], out["fold"] = fold.repeat, fold.fold
    players, codes = np.unique(lineup_o.astype(np.int64), return_inverse=True)
    return out, NullFold(players, codes.reshape(lineup_o.shape), stand_in, projector(levels, w))


def reading(parts: pd.DataFrame, prefix: str, terms: tuple, scales: tuple, free: tuple = FREE) -> np.ndarray:
    """The terms' coefficients beside the free terms, each times its scale: the standard deviation of the named raw
    column over the season, or 1."""
    w = parts.w.to_numpy(float)
    columns = parts[[f"{prefix}{name}" for name in (*terms, *free)]].to_numpy(float)
    coef = fit(parts[f"{prefix}miss"].to_numpy(float), columns, w)[:len(terms)]
    sizes = [weighted_sd(parts[scale].to_numpy(float), w) if scale else 1.0 for scale in scales]
    return coef * np.array(sizes)


def shape_reading(parts: pd.DataFrame) -> dict:
    """The lineup-shape terms' own effects in the fit that holds them, points per 100: each square at one standard
    deviation of its rating sum, the product at one of each, the talent spread per standard deviation."""
    w = parts.w.to_numpy(float)
    names = (*TERMS, *FREE, *SHAPE)
    coef = fit(parts.p_miss.to_numpy(float), parts[[f"p_{name}" for name in names]].to_numpy(float), w)[-len(SHAPE):]
    sd = {name: weighted_sd(parts[f"raw_{name}"].to_numpy(float), w) for name in RAW[len(TERMS):]}
    offense, defense, spread = sd["rating_sum_off"], sd["rating_sum_def"], sd["talent_spread_off"]
    return dict(zip(SHAPE_NAMES, coef * np.array([offense ** 2, defense ** 2, offense * defense, spread])))


def trade_reading(parts: pd.DataFrame) -> dict:
    """The after-a-trade terms: each coefficient with no trade, after a trade, the difference, and the slice's
    share of the held-back possessions."""
    w = parts.w.to_numpy(float)
    share = float(w[parts.raw_trade.to_numpy() > 0].sum() / w.sum()) if "raw_trade" in parts else 0.0
    out = {"share of possessions after a trade": share}
    if share * w.sum() < TRADE_MIN_POSS:
        return out
    coef = reading(parts, "p_", (*TERMS, *TRADE_TERMS), ("raw_amount", "raw_concentration", "raw_amount",
                                                         "raw_concentration", None))
    for k, term in enumerate(TERMS):
        out[f"{term}, no trade"], out[f"{term}, difference after a trade"] = coef[k], coef[k + 2]
        out[f"{term}, after a trade"] = coef[k] + coef[k + 2]
    out["level after a trade"] = coef[4]
    return out


def season_readings(parts: pd.DataFrame, season: int, trade: bool) -> list:
    """One row per reading and term for the season."""
    w = parts.w.to_numpy(float)
    rows = [dict(season=season, reading="baseline", term="error with team levels only",
                 value=float(np.sum(w * parts.p_miss.to_numpy() ** 2) / w.sum()))]
    rows += [dict(season=season, reading="one standard deviation", term=term,
                  value=weighted_sd(parts[f"raw_{term}"].to_numpy(float), w)) for term in TERMS]
    if trade:
        return rows + [dict(season=season, reading="after a trade", term=term, value=float(value))
                       for term, value in trade_reading(parts).items()]
    scales = ("raw_amount", "raw_concentration")
    found = {name: reading(parts, "p_", TERMS, scales, free) for name, free in READINGS}
    found["no team levels"] = reading(parts, "q_", TERMS, scales)
    rows += [dict(season=season, reading=name, term=term, value=float(value))
             for name, values in found.items() for term, value in zip(TERMS, values)]
    return rows + [dict(season=season, reading="lineup shape", term=term, value=float(value))
                   for term, value in shape_reading(parts).items()]


# ----------------------------------------------------------------------------------------------- the null
class Career:
    """Every player's whole-season self-creation in career order, from the season panel: what the career null hands
    from one player to another."""

    def __init__(self, panel: pd.DataFrame, last: int):
        rows = panel[(panel.side == "O") & (panel.season <= last)].reset_index(drop=True)
        rows = rows.assign(value=self_creation(rows)).sort_values(["player_id", "season"], kind="stable")
        self.players = rows.player_id.unique()
        self.place = {player: k for k, player in enumerate(self.players)}
        grouped = rows.groupby("player_id", sort=True)
        self.seasons = {player: group.season.to_numpy() for player, group in grouped}
        self.values = {player: group.value.to_numpy() for player, group in grouped}

    def permutations(self, draws: int) -> np.ndarray:
        """draws x players: each draw's donor for every player, by place."""
        rng = np.random.default_rng(NULL_SEED)
        return np.array([rng.permutation(len(self.players)) for _ in range(draws)]).reshape(draws, -1)

    def donated(self, permutation: np.ndarray, players: np.ndarray, season: int) -> np.ndarray:
        """Each player's donor's self-creation at the same step of the donor's career (the donor's last when his is
        shorter); NaN for a player the panel does not hold in the season."""
        out = np.full(len(players), np.nan)
        for k, player in enumerate(players):
            place = self.place.get(player)
            steps = np.flatnonzero(self.seasons[player] == season) if place is not None else ()
            if len(steps):
                donor = self.players[permutation[place]]
                out[k] = self.values[donor][min(int(steps[0]), len(self.values[donor]) - 1)]
        return out


def null_readings(parts: pd.DataFrame, nulls: list, season: int, career: Career, permutations: np.ndarray) -> dict:
    """{reading: draws x terms}: each within-team reading's coefficients, per standard deviation, with every player
    handed his donor's self-creation."""
    w = parts.w.to_numpy(float)
    target = parts.p_miss.to_numpy(float)
    frees = {name: parts[[f"p_{column}" for column in free]].to_numpy(float) for name, free in READINGS}
    players = np.unique(np.concatenate([null.players for null in nulls]))
    out = {name: np.empty((len(permutations), len(TERMS))) for name in frees}
    for draw, permutation in enumerate(permutations):
        donated = career.donated(permutation, players, season)
        raw, projected = [], []
        for null in nulls:
            values = donated[np.searchsorted(players, null.players)]
            created = np.where(np.isnan(values), null.stand_in, values)[null.codes]
            terms = np.column_stack([created.sum(axis=1), concentration(created)])
            raw.append(terms)
            projected.append(null.project(terms))
        raw, projected = np.vstack(raw), np.vstack(projected)
        sizes = np.array([weighted_sd(raw[:, k], w) for k in range(len(TERMS))])
        for name, free in frees.items():
            out[name][draw] = fit(target, np.hstack([projected, free]), w)[:len(TERMS)] * sizes
    return out


def against_null(mine: pd.Series, theirs: pd.DataFrame, seasons: list) -> dict:
    """The real across-season mean against the null draws' means over the same seasons."""
    value, draws = float(mine[seasons].mean()), theirs[seasons].mean(axis=1).to_numpy()
    centre, spread = float(draws.mean()), float(draws.std(ddof=1))
    return dict(mean=value, null_mean=centre, null_sd=spread,
                z_against_null=(value - centre) / spread if spread > 0 else np.nan,
                share_of_draws_as_far=float(np.mean(np.abs(draws - centre) >= abs(value - centre))),
                draws=len(draws), seasons=len(seasons))


def null_summary(per_season: pd.DataFrame, null_seasons: pd.DataFrame) -> pd.DataFrame:
    """Each within-team reading's terms: the across-season mean against the career null's draws of the same mean."""
    rows = []
    for name, _ in READINGS:
        real, null = per_season[per_season.reading == name], null_seasons[null_seasons.reading == name]
        for term in TERMS:
            mine = real[real.term == term].set_index("season").value
            theirs = null[null.term == term].pivot(index="draw", columns="season", values="value")
            for label, (first, last) in eras():
                seasons = [season for season in mine.index if first <= season <= last]
                if seasons:
                    rows.append(dict(reading=name, term=term, era=label, **against_null(mine, theirs, seasons)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- the table
def weighted_bins(values: np.ndarray, w: np.ndarray, parts: int) -> np.ndarray:
    """0 to parts - 1: each value's possession-weighted quantile bin."""
    order = np.argsort(values, kind="stable")
    middle = (np.cumsum(w[order]) - 0.5 * w[order]) / w.sum()
    bins = np.empty(len(values), dtype=int)
    bins[order] = np.minimum((middle * parts).astype(int), parts - 1)
    return bins


def residual_after(parts: pd.DataFrame, free: tuple) -> np.ndarray:
    """Each row's miss after the team levels and the `free` terms alone, fitted over the season: what is left for the
    self-creation terms to explain."""
    w = parts.w.to_numpy(float)
    columns = parts[[f"p_{name}" for name in free]].to_numpy(float)
    target = parts.p_miss.to_numpy(float)
    return target - columns @ fit(target, columns, w)


def season_table(parts: pd.DataFrame, season: int, free: tuple, held: str) -> pd.DataFrame:
    """The season's fifths of the amount by thirds of the concentration within each fifth: possession-weighted sums
    of the miss after the `free` terms alone, of the amount and of the concentration."""
    w = parts.w.to_numpy(float)
    left = residual_after(parts, free)
    amount, spread = parts.raw_amount.to_numpy(float), parts.raw_concentration.to_numpy(float)
    fifth = weighted_bins(amount, w, FIFTHS)
    third = np.zeros(len(w), dtype=int)
    for k in range(FIFTHS):
        inside = fifth == k
        third[inside] = weighted_bins(spread[inside], w[inside], THIRDS)
    cells = pd.DataFrame({"fifth": fifth + 1, "third": third + 1, "w": w, "miss": w * left, "amount": w * amount,
                          "concentration": w * spread})
    return cells.groupby(["fifth", "third"], as_index=False).sum().assign(season=season, held=held)


def pooled_table(tables: pd.DataFrame) -> pd.DataFrame:
    """The cells pooled over all seasons and over each era: mean miss per 100, share of possessions, mean amount and
    mean concentration; once for each set of free terms held."""
    out = []
    for held, table in tables.groupby("held", sort=False):
        for label, (first, last) in eras():
            part = table[(table.season >= first) & (table.season <= last)]
            sums = part.groupby(["fifth", "third"], as_index=False)[["w", "miss", "amount", "concentration"]].sum()
            out.append(pd.DataFrame({"held": held, "era": label, "amount_fifth": sums.fifth,
                                     "concentration_third": sums.third, "share_of_possessions": sums.w / sums.w.sum(),
                                     "mean_miss": sums.miss / sums.w, "mean_amount": sums.amount / sums.w,
                                     "mean_concentration": sums.concentration / sums.w}))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------------------------------- across seasons
def across_seasons(values: pd.Series, newey_west) -> dict:
    """The mean of per-season readings, its standard error, z, Newey-West z and the seasons with the mean's sign."""
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    se_nw = newey_west(values) if n > 2 else np.nan
    return dict(mean=mean, standard_error=se, z=mean / se if se > 0 else np.nan,
                z_newey_west=mean / se_nw if se_nw > 0 else np.nan,
                seasons_with_its_sign=int((np.sign(values) == np.sign(mean)).sum()), seasons=n)


def summary(per_season: pd.DataFrame, newey_west) -> pd.DataFrame:
    """Every reading and term across all seasons and each era."""
    rows = []
    for (name, term), group in per_season.groupby(["reading", "term"], sort=False):
        values = group.set_index("season").value.sort_index()
        for label, (first, last) in eras():
            part = values[(values.index >= first) & (values.index <= last)]
            rows.append(dict(reading=name, term=term, era=label, **across_seasons(part, newey_west)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ the inputs
def step4_of(fold, multipliers: pd.DataFrame, layer: pd.DataFrame | None) -> tuple:
    """The fold's shrink multipliers, built as check 151's `system_errors` builds them."""
    from openrapm.prior_shrink import WITHIN_COLUMNS
    row = multipliers.loc[fold.season]
    factor = factor_for(DEFENCE_FACTORS, fold.season)
    step4 = (float(row.multiplier_off), factor * float(row.multiplier_def),
             float(row.get("games_multiplier_off", 1.0)), factor * float(row.get("games_multiplier_def", 1.0)))
    if layer is not None:
        step4 += tuple(float(layer.at[fold.season, column]) for column in WITHIN_COLUMNS)
    return step4


def trade_moves(box: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """(team_id, game_idx): for every move of a rotation player, TRADE_MINUTES a game or more for the team he left
    or the one he joined, both teams, each dated by his first game for the new team."""
    played = box[box.minutes > 0].merge(games[["game_id", "game_idx"]], on="game_id")
    per_game = played.groupby(["player_id", "team_id"]).minutes.mean()
    played = played.sort_values(["player_id", "game_idx"], kind="stable")
    moves = []
    for player, group in played.groupby("player_id"):
        team, when = group.team_id.to_numpy(), group.game_idx.to_numpy()
        for k in np.flatnonzero(team[1:] != team[:-1]) + 1:
            if max(per_game[(player, team[k - 1])], per_game[(player, team[k])]) >= TRADE_MINUTES:
                moves += [(team[k - 1], when[k]), (team[k], when[k])]
    return pd.DataFrame(moves, columns=["team_id", "game_idx"])


def season_moves(cfg: dict, season: int, design) -> pd.DataFrame:
    """The season's moves between teams, from its box scores in the design's date order."""
    from openrapm.data.box_scores import season_box
    from openrapm.data.ingest import played_phases
    box = season_box([season], played_phases(season, cfg), cfg)
    games = design.games.assign(game_id=design.games.game_id.astype(str))
    return trade_moves(box.assign(game_id=box.game_id.astype(str)), games)


def stop_on_stale_pieces(cfg: dict, seasons: list) -> None:
    """Stop before a design piece would be rebuilt into the cache: every piece must match its stint file."""
    from openrapm.data.designs import _load_piece
    from openrapm.data.ingest import played_phases
    stale = [(season, phase) for season in seasons for phase in played_phases(season, cfg)
             if _load_piece(season, phase, cfg) is None]
    if stale:
        raise SystemExit(f"{len(stale)} design pieces are missing or stale, among them {stale[:3]}: a run would "
                         f"rebuild them into the cache; rebuild them on purpose first")


def stop_unless_date_folds(folds: list, where: dict) -> None:
    """The after-a-trade reading needs folds scored on the rest of a season after a cut day."""
    for fold in folds:
        record = json.loads((where[fold.stem] / f"fold_{fold.stem}.json").read_text())["summary"]
        if record.get("split") != "date":
            raise SystemExit(f"--trade=1 needs date folds (openrapm folds --cut); {fold.stem} is "
                             f"{record.get('split')!r}")


@dataclass
class Inputs:
    """What every season of a run reads: the checks, the folds, the shrink, the null, where the stint rows go."""
    wowy: object
    audit: object
    folds: list
    where: dict
    multipliers: pd.DataFrame
    layer: pd.DataFrame | None
    career: Career | None
    permutations: np.ndarray | None
    save_rows: Path | None


def trainable_panel(cfg: dict) -> pd.DataFrame:
    """The season panel, gated like every reader of it: a season still being played never enters a reading."""
    from openrapm.data.config import resolve
    from openrapm.data.seasons import drop_untrainable
    panel, _ = drop_untrainable(pd.read_parquet(resolve(cfg, "panel")), cfg, what="the season panel")
    return panel


def run_inputs(args: argparse.Namespace, cfg: dict, within: Path, wowy) -> Inputs:
    """The folds of the run's seasons, their shrink multipliers and the career null's draws, after the checks that
    stop a run before it starts."""
    folds, where = wowy.system_folds(within, args.folds)
    scored = [fold for fold in folds if args.first <= fold.season <= args.last]
    seasons = sorted({fold.season for fold in scored})
    multipliers, layer = wowy.step4_tables(within, folds, where, seasons, DEFENCE_FACTORS, args.step4_from or None)
    if args.trade:
        stop_unless_date_folds(scored, where)
    stop_on_stale_pieces(cfg, seasons)
    career = Career(trainable_panel(cfg), args.last) if args.null else None
    return Inputs(wowy, wowy._audit(), scored, where, multipliers, layer, career,
                  career.permutations(args.null) if career is not None else None,
                  Path(args.save_rows) if args.save_rows else None)


def run_season(inputs: Inputs, cfg: dict, season: int, design, trade: bool) -> tuple:
    """(the season's reading rows, its table cells, its null draws or None)."""
    moves = season_moves(cfg, season, design) if trade else None
    built = [fold_part(inputs.wowy, inputs.audit, fold, inputs.wowy.team_arrays(inputs.where[fold.stem], fold.stem),
                       design, step4_of(fold, inputs.multipliers, inputs.layer), moves)
             for fold in inputs.folds if fold.season == season]
    parts = pd.concat([part for part, _ in built], ignore_index=True)
    null = None
    if inputs.career is not None:
        draws = null_readings(parts, [fold for _, fold in built], season, inputs.career, inputs.permutations)
        null = pd.DataFrame([dict(season=season, reading=name, draw=draw, term=term, value=float(values[draw, k]))
                             for name, values in draws.items() for draw in range(len(values))
                             for k, term in enumerate(TERMS)])
    tables = pd.concat([season_table(parts, season, free, name) for name, free in READINGS], ignore_index=True)
    if inputs.save_rows is not None:
        inputs.save_rows.mkdir(parents=True, exist_ok=True)
        stint_rows(parts, [fold for _, fold in built], season).to_parquet(inputs.save_rows / f"{season}.parquet",
                                                                          index=False)
    return season_readings(parts, season, trade), tables, null


def stint_rows(parts: pd.DataFrame, nulls: list, season: int) -> pd.DataFrame:
    """One row per held-back stint, for cuts made after the run (--save-rows): its fold, game, teams, possessions, the
    five's summed self-creation and concentration, the miss three ways, the five offensive player ids.  Each game is
    held back once in each of the season's three repeats, so every stint appears three times, once per repeat."""
    lineups = np.vstack([null.players[null.codes] for null in nulls])
    out = pd.DataFrame({"season": np.int16(season), "repeat": parts.repeat.astype(np.int8),
                        "fold": parts.fold.astype(np.int8), "game_idx": parts.game_idx.astype(np.int32),
                        "team_off": parts.team_off, "team_def": parts.team_def,
                        "possessions": parts.w.astype(np.float32),
                        "self_creation_sum": parts.raw_amount.astype(np.float32),
                        "concentration": parts.raw_concentration.astype(np.float32),
                        "miss": parts.raw_miss.astype(np.float32),
                        "miss_after_team_levels": parts.p_miss.astype(np.float32),
                        "above_expected": residual_after(parts, (*FREE, *SHAPE)).astype(np.float32)})
    for k in range(5):
        out[f"offense_player_{k + 1}"] = lineups[:, k]
    return out


def run(args: argparse.Namespace, cfg: dict, within: Path, wowy) -> tuple:
    """(every season's readings, table cells, null draws)."""
    from openrapm.data.designs import DesignCache
    inputs = run_inputs(args, cfg, within, wowy)
    cache, rows, tables, nulls, started = DesignCache(cfg, keep=1), [], [], [], time.time()
    for season in sorted({fold.season for fold in inputs.folds}):
        found, table, null = run_season(inputs, cfg, season, cache.season(season, "pts"), bool(args.trade))
        rows += found
        tables.append(table)
        nulls.append(null)
        print(f"  {season}: done ({time.time() - started:.0f}s)", flush=True)
    nulls = [null for null in nulls if null is not None]
    return pd.DataFrame(rows), pd.concat(tables, ignore_index=True), pd.concat(nulls) if nulls else None


# --------------------------------------------------------------------------------------------- the report
def top_table(cfg: dict, season: int, count: int = 20) -> pd.DataFrame:
    """The season's players with TOP_POSS offensive possessions or more, by self-creation from the season panel's
    whole-season padded rates, with the published offense and total beside."""
    from openrapm.data.config import resolve
    panel = trainable_panel(cfg)
    rows = panel[(panel.season == season) & (panel.side == "O")].reset_index(drop=True)
    makes = rows.twos_made_per100_uncentred + rows.threes_made_per100_uncentred
    rows = rows.assign(self_creation=self_creation(rows), share_assisted=share_assisted(rows.assists_per100_uncentred,
                                                                                        makes))
    product = pd.read_parquet(resolve(cfg, "outputs") / "season_ratings_product.parquet")
    product = product[product.season == season][["player_id", "player_name", "rating_off", "rating_total", "poss_off"]]
    table = rows[["player_id", "self_creation", "share_assisted"]].merge(product, on="player_id")
    return table[table.poss_off >= TOP_POSS].sort_values("self_creation", ascending=False).head(count)


def check_against(per_season: pd.DataFrame, spec: str) -> None:
    """Each season's error with team levels only beside check 151's `step4` / `all` error from FILE:SYSTEM."""
    path, system = spec.rsplit(":", 1)
    theirs = pd.read_csv(path)
    theirs = theirs[(theirs.system == system) & (theirs.stage == "step4") & (theirs.slice == "all")]
    mine = per_season[per_season.reading == "baseline"].set_index("season").value
    joined = pd.DataFrame({"this check": mine, "check 151": theirs.set_index("season").error}).dropna()
    joined["difference"] = joined["this check"] - joined["check 151"]
    print(f"\nthe error with team levels only against check 151 ({system}): largest difference "
          f"{joined.difference.abs().max():.2e} over {len(joined)} seasons")
    print(joined.iloc[[0, len(joined) // 2, -1]].to_string())


def write_and_print(args: argparse.Namespace, cfg: dict, outputs: Path, found: tuple, newey_west) -> None:
    """The output files, then the readings, the null, the table and the players."""
    per_season, tables, null_seasons = found
    csv = outputs / "csv"
    csv.mkdir(exist_ok=True)
    across, table = summary(per_season, newey_west), pooled_table(tables)
    per_season.to_csv(csv / f"{args.tag}_seasons.csv", index=False)
    across.to_csv(csv / f"{args.tag}.csv", index=False)
    table.to_csv(csv / f"{args.tag}_table.csv", index=False)
    pd.set_option("display.width", 220)
    if args.check:
        check_against(per_season, args.check)
    print("\neach term in points per 100 per standard deviation, beyond what OpenRAPM expects")
    print(across[across.reading != "baseline"].round(4).to_string(index=False))
    if null_seasons is not None:
        null_seasons.to_csv(csv / f"{args.tag}_null_seasons.csv", index=False)
        against = null_summary(per_season, null_seasons)
        against.to_csv(csv / f"{args.tag}_null.csv", index=False)
        print(f"\nthe within-team readings against the career null, {args.null} draws")
        print(against.round(4).to_string(index=False))
    for held, part in table[table.era == "all"].groupby("held", sort=False):
        print(f"\nthe fifths of the amount by thirds of the concentration, all seasons, {held}")
        print(part.drop(columns=["held", "era"]).round(4).to_string(index=False))
    if args.top:
        top = top_table(cfg, args.top)
        top.to_csv(csv / f"{args.tag}_top.csv", index=False)
        print(f"\n{args.top} by self-creation, {TOP_POSS:,} or more offensive possessions")
        print(top.round(3).to_string(index=False))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-creation in lineups, on the within-season folds.",
                                     allow_abbrev=False)
    parser.add_argument("--folds", default="parts107",
                        help="the system whose folds are read: outputs/within/<name>_early and _within, or the folder "
                             "<name> (default %(default)s)")
    parser.add_argument("--step4-from", default="",
                        help="fit the shrink on this system's folds instead, as for a season so far read with the "
                             "shrink of whole seasons")
    parser.add_argument("--trade", type=int, default=0, help="1: the after-a-trade reading, on date folds")
    parser.add_argument("--null", type=int, default=200,
                        help="career-null draws for the within-team reading, 0 for none (default %(default)s)")
    parser.add_argument("--check", default="", help="FILE:SYSTEM, a check 151 seasons file to compare against")
    parser.add_argument("--save-rows", default="",
                        help="a folder: one parquet per season of every held-back stint, for cuts made later")
    parser.add_argument("--top", type=int, default=2026,
                        help="the season of the players table by self-creation, 0 for none (default %(default)s)")
    parser.add_argument("--tag", default="selfcreation", help="names the output files (default %(default)s)")
    parser.add_argument("--first", type=int, default=1997)
    parser.add_argument("--last", type=int, default=2026)
    parser.add_argument("--creation", default="owner", choices=("owner", "actual_makes"),
                        help="owner: the formula on every attempt; actual_makes: his own unassisted makes, the formula "
                             "on the misses and free throws (experiment 111) (default %(default)s)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    use_utf8()
    args = parse_args(argv)
    CREATION_READ["name"] = args.creation
    from openrapm.data.config import load_config, resolve
    cfg = load_config()
    outputs = resolve(cfg, "outputs")
    within = resolve(cfg, "within") if "within" in cfg.get("paths", {}) else outputs / "within"
    wowy = _load("wowy_check", "151_wowy.py")
    started = time.time()
    with one_thread():
        found = run(args, cfg, within, wowy)
    write_and_print(args, cfg, outputs, found, wowy.newey_west_se)
    print(f"\ndone in {time.time() - started:.0f}s: outputs/csv/{args.tag}*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
