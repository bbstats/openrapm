"""The multi-stat network: one model that reads a player's earlier games of the season and forecasts, for his next
game, every box-score rate the box-score prior is built from, against an average opponent.

The owner's handoff of 2026-10-27 (scratch/2026-10-27_multistat/: the network, its inputs and its padding baseline,
tested on simulated leagues) built on the real seasons, with the owner's changes: single-season inputs only; the stats
are the ones the box-score prior reads; each game measured against the opponent and the league.

    link_j = padding link_j + network_j(state)        log of a count per 100, logit of a make percentage

Words used below
    stat             one of the 21 forecasts in `STATS`: 18 counts per 100 possessions and 3 make percentages, the
                     counts and hustle numbers every input of the shipped box-score prior is built from
    exposure         a stat's denominator in one game: his offensive possessions, his defensive possessions, the
                     average of the two (loose balls, read on both sides), or, for a percentage, its attempts
    league level     the season's value of a stat over the games handed in: 100 x counts / exposure for a count,
                     makes / attempts for a percentage
    opponent factor  for a count, what the night's opponent allowed of it in its earlier games of the season, over the
                     league level, padded toward 1 by a pad fitted per stat:
                         factor = (100 x allowed / league level + pad) / (exposure faced + pad)
                     It multiplies the game's exposure, so the forecast is a rate against an average opponent.  The
                     make percentages have none (the owner: simpler is better)
    padding          the per-stat recursion, each stat on its own, the network's starting point: before his first game
                     of the season the league level, then after each game
                         forecast = forecast + w x (observed - forecast),   w = e / (e + k)
                     e the game's exposure times the opponent factor, k one pad per stat fitted on the training seasons
    state            what the network reads before a game, all of it from his earlier games of the same season: his
                     counts and exposures summed four ways (discounted over `TAUS` possessions, and the season so
                     far), his playing time so far, home or away, playoffs, the padding's 21 forecasts
    NLL              negative log-likelihood of one game under a forecast, without the terms that do not depend on
                     it: a count, Poisson over its exposure, m - y log m with m = rate x exposure / 100; a percentage,
                     binomial over its attempts, a x softplus(link) - y x link.  Summed over the 21 stats, the loss

Single season (the owner, 2026-10-27: "single-season only obviously!"): every input comes from the player's earlier
games of the same season.  Nothing from his other seasons reaches a forecast, and neither do height, weight or age: an
exact listing identifies a player across seasons.  The network's weights are learned from other seasons, as every
model coefficient may be.

Example: before his 41st game a guard has 312 assists in 2,940 offensive possessions; with the league at 4.9 per 100
and opponents that allowed 3% more assists than the league, his season-so-far input is
log((31,200 + 50 x 4.9) / (1.03 x 2,940 + 50) / 4.9) = +0.73, and the network's forecast for an average opponent
might be 10.1 assists per 100.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Stat", "STATS", "NAMES", "IS_PCT", "ATTEMPTS_OF", "HUSTLE_STATS", "TAUS", "BEST", "Games",
           "games_from_rows", "league_levels", "league_by_row", "opponent_sums", "opponent_factors",
           "padding_forecasts", "padding_loss", "fit_pads", "nll", "input_names", "playing_time", "network_inputs",
           "Standardised", "standardise", "neutral_inputs", "train_network", "forward_numpy", "forecast"]


@dataclass(frozen=True)
class Stat:
    """One forecast: its name, what it counts, the per-game columns summed into its count (its makes, for a
    percentage), and its exposure: "off", "def", "both", or the name of the count it is a share of."""
    name: str
    meaning: str
    counts: tuple
    exposure: str


STATS = [
    Stat("three_point_attempts", "three-point attempts per 100 offensive possessions", ("fg3m", "fg3_miss"), "off"),
    Stat("three_point_pct", "share of three-point attempts made", ("fg3m",), "three_point_attempts"),
    Stat("two_point_attempts", "two-point attempts per 100 offensive possessions", ("fg2m", "fg2_miss"), "off"),
    Stat("two_point_pct", "share of two-point attempts made", ("fg2m",), "two_point_attempts"),
    Stat("free_throw_attempts", "free-throw attempts per 100 offensive possessions", ("ftm", "ft_miss"), "off"),
    Stat("free_throw_pct", "share of free throws made", ("ftm",), "free_throw_attempts"),
    Stat("off_rebounds", "offensive rebounds per 100 offensive possessions", ("orb",), "off"),
    Stat("def_rebounds", "defensive rebounds per 100 defensive possessions", ("drb",), "def"),
    Stat("assists", "assists per 100 offensive possessions", ("ast",), "off"),
    Stat("turnovers", "turnovers per 100 offensive possessions", ("tov",), "off"),
    Stat("steals", "steals per 100 defensive possessions", ("stl",), "def"),
    Stat("blocks", "blocks per 100 defensive possessions", ("blk",), "def"),
    Stat("screen_assists", "screen assists per 100 offensive possessions", ("screen_assists",), "off"),
    Stat("screen_assist_points", "points off his screens per 100 offensive possessions", ("screen_ast_pts",), "off"),
    Stat("off_box_outs", "offensive box-outs per 100 offensive possessions", ("off_boxouts",), "off"),
    Stat("deflections", "deflections per 100 defensive possessions", ("deflections",), "def"),
    Stat("contested_twos", "contested two-point shots per 100 defensive possessions", ("contested_shots_2pt",), "def"),
    Stat("contested_threes", "contested threes per 100 defensive possessions", ("contested_shots_3pt",), "def"),
    Stat("charges_drawn", "charges drawn per 100 defensive possessions", ("charges_drawn",), "def"),
    Stat("def_box_outs", "defensive box-outs per 100 defensive possessions", ("def_boxouts",), "def"),
    Stat("loose_balls_recovered", "loose balls recovered per 100 possessions, the average of his two ends",
         ("loose_balls_recovered",), "both"),
]
NAMES = [stat.name for stat in STATS]
IS_PCT = np.array([stat.exposure not in ("off", "def", "both") for stat in STATS])
ATTEMPTS_OF = {j: NAMES.index(stat.exposure) for j, stat in enumerate(STATS) if IS_PCT[j]}
HUSTLE_STATS = np.arange(NAMES.index("screen_assists"), len(STATS))     # counted from 2017 only
TAUS = (100.0, 400.0, 1500.0)    # possessions: about 1, 5 and 20 games of a starter
K0, A0 = 50.0, 10.0              # the inputs' light pads, possessions and attempts (the handoff's)
FIRST_FACTOR_SEASON = 0          # the opponent factor reads every season handed in
# The settings that won the handoff's validation seasons (its tune.py, 2026-10-27): two hidden layers of 256, dropout
# 0.1, an L2 penalty of 1e-4 on the weights, Adam at 5e-4 with cosine decay, batches of 4,096, early stopping.
BEST = dict(widths=(256, 256), lr=5e-4, wd=1e-4, batch=4096, drop=0.1, max_epochs=60, patience=5)


# ------------------------------------------------------------------------------------------ the games as arrays
@dataclass
class Games:
    """The game rows as arrays, sorted by season, player, then game, each player-season followed by its end row: a
    game after his last one, with no outcome, whose state is his whole season (`games_from_rows`)."""
    frame: pd.DataFrame        # the rows' keys and context, in this order; `end` marks an end row
    player_season: np.ndarray  # (n,) the row's player-season, numbered 0, 1, ...
    position: np.ndarray       # (n,) his games before this one this season; an end row's is his number of games
    season: np.ndarray         # (n,)
    counts: np.ndarray         # (n, 21) each stat's count, or its makes for a percentage
    exposure: np.ndarray       # (n, 21) each stat's exposure, before the opponent factor
    available: np.ndarray      # (n, 21) the game informs the stat: an exposure above 0 and the count recorded
    possessions: np.ndarray    # (n,) the average of his two ends: what discounts the sums
    by_position: list          # [rows at position 0, rows at position 1, ...]

    @property
    def n_player_seasons(self) -> int:
        return int(self.player_season.max()) + 1 if len(self.player_season) else 0


def _end_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """One end row per player-season: his last row's keys, a day after the season's last game, nothing played."""
    last = rows.groupby(["season", "player_id"], sort=False).tail(1).copy()
    season_end = rows.groupby("season")["game_date"].transform("max")
    last["game_date"] = season_end.loc[last.index] + pd.Timedelta(days=1)
    last["game_idx"] = np.iinfo(np.int32).max
    last["game_id"] = ""
    last["phase"] = "RS"
    numeric = [column for column in rows.columns if column not in
               ("season", "phase", "game_id", "game_date", "game_idx", "player_id", "team_id", "opponent_id")]
    last[numeric] = 0.0
    hustle = [column for stat in STATS[HUSTLE_STATS[0]:] for column in stat.counts]
    last[hustle] = np.nan
    return last


def games_from_rows(rows: pd.DataFrame) -> Games:
    """`Games` from the per-game table (data/player_games.py): its rows of any seasons, sorted, end rows added."""
    rows = rows.assign(end=False)
    rows = pd.concat([rows, _end_rows(rows).assign(end=True)], ignore_index=True)
    rows = rows.sort_values(["season", "player_id", "game_idx"], kind="stable").reset_index(drop=True)
    player_season = rows.groupby(["season", "player_id"], sort=False).ngroup().to_numpy()
    position = rows.groupby(["season", "player_id"], sort=False).cumcount().to_numpy()
    counts = np.column_stack([rows[list(stat.counts)].sum(axis=1, min_count=len(stat.counts)).to_numpy(dtype=float)
                              for stat in STATS])
    exposure = np.empty_like(counts)
    for j, stat in enumerate(STATS):
        if stat.exposure == "off":
            exposure[:, j] = rows["poss_off"].to_numpy(dtype=float)
        elif stat.exposure == "def":
            exposure[:, j] = rows["poss_def"].to_numpy(dtype=float)
        elif stat.exposure == "both":
            exposure[:, j] = 0.5 * (rows["poss_off"].to_numpy(dtype=float) + rows["poss_def"].to_numpy(dtype=float))
    for j, attempts in ATTEMPTS_OF.items():
        exposure[:, j] = counts[:, attempts]
    end = rows["end"].to_numpy(dtype=bool)
    available = (exposure > 0) & np.isfinite(counts) & ~end[:, None]
    counts = np.where(available, counts, 0.0)
    exposure = np.where(available, exposure, 0.0)
    order = np.argsort(position, kind="stable")
    breaks = np.flatnonzero(np.diff(position[order])) + 1
    return Games(frame=rows, player_season=player_season, position=position, season=rows["season"].to_numpy(),
                 counts=counts, exposure=exposure, available=available,
                 possessions=0.5 * (rows["poss_off"].to_numpy(dtype=float) + rows["poss_def"].to_numpy(dtype=float)),
                 by_position=np.split(order, breaks))


# ------------------------------------------------------------------------------------------ league and opponent
def league_levels(games: Games) -> pd.DataFrame:
    """Each season's league level of every stat (rows: seasons, columns: `NAMES`), over the games handed in: in a
    within-season build, its fit games."""
    out = {}
    for season in np.unique(games.season):
        rows = games.season == season
        made = games.counts[rows].sum(axis=0)
        exposure = games.exposure[rows].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[int(season)] = np.where(IS_PCT, made / exposure, 100.0 * made / exposure)
    return pd.DataFrame.from_dict(out, orient="index", columns=NAMES)


def league_by_row(games: Games, league: pd.DataFrame) -> np.ndarray:
    """(n, 21): each row's season's league levels."""
    return league.reindex(games.season).to_numpy(dtype=float)


def opponent_sums(games: Games, league_row: np.ndarray) -> tuple:
    """(allowed, faced), each (n, 21): what each row's opponent allowed in its games before this one, as exposure at
    the league level (100 x counts / league level), and the exposure its opponents had, over the games handed in.  An
    end row reads 0 for both, so its factor is 1."""
    frame = games.frame
    real = ~frame["end"].to_numpy(dtype=bool)
    keys = frame.loc[real, ["season", "game_idx", "opponent_id"]].reset_index(drop=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.where(IS_PCT | ~np.isfinite(league_row[real]), 0.0, 100.0 * games.counts[real] / league_row[real])
    n_stats = len(STATS)
    allowed = pd.concat([keys, pd.DataFrame(scaled, columns=[f"a{j}" for j in range(n_stats)]),
                         pd.DataFrame(games.exposure[real], columns=[f"e{j}" for j in range(n_stats)])], axis=1)
    team_games = allowed.groupby(["season", "opponent_id", "game_idx"], as_index=False).sum()
    values = [f"a{j}" for j in range(n_stats)] + [f"e{j}" for j in range(n_stats)]
    team_games[values] = team_games.groupby(["season", "opponent_id"], sort=False)[values].cumsum() - team_games[values]
    joined = keys.merge(team_games, on=["season", "game_idx", "opponent_id"], how="left")
    out_allowed, out_faced = np.zeros((len(frame), n_stats)), np.zeros((len(frame), n_stats))
    out_allowed[real] = joined[values[:n_stats]].to_numpy(dtype=float)
    out_faced[real] = joined[values[n_stats:]].to_numpy(dtype=float)
    return out_allowed, out_faced


def opponent_factors(sums: tuple, pads: np.ndarray) -> np.ndarray:
    """(n, 21): each row's opponent factor (module docstring) from `opponent_sums`; 1 for a percentage and wherever
    the pad is infinite.

    Example: a defence that allowed 1,100 offensive rebounds where the league level expected 1,000 on the 40,000
    possessions its opponents had, with a pad of 2,000 possessions: (44,000 + 2,000) / (40,000 + 2,000) = 1.095."""
    allowed, faced = sums
    pads = np.broadcast_to(np.asarray(pads, dtype=float), (len(STATS),))
    finite = np.isfinite(pads) & ~IS_PCT
    safe = np.where(finite, pads, 1.0)
    return np.where(finite, (allowed + safe) / (faced + safe), 1.0)


# ------------------------------------------------------------------------------------------ the padding
def _observed(games: Games, factors: np.ndarray) -> tuple:
    """(each row's observed value: 100 x count / exposure x factor, or makes / attempts; the exposure the padding
    weighs it by: exposure x factor, or the attempts)."""
    adjusted = np.where(IS_PCT, games.exposure, games.exposure * factors)
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = np.where(IS_PCT, games.counts / games.exposure, 100.0 * games.counts / adjusted)
    return np.where(games.available, observed, 0.0), adjusted


def _padding_pass(games: Games, league_row: np.ndarray, factors: np.ndarray, ks: np.ndarray, visit) -> None:
    """The per-stat recursion over every player-season at once, game number by game number; `ks` (candidates, 21);
    `visit(rows, forecasts)` sees each game's forecasts, (candidates, rows, 21), before the game updates them."""
    observed, adjusted = _observed(games, factors)
    state = np.zeros((ks.shape[0], games.n_player_seasons, len(STATS)))
    first = games.by_position[0]
    state[:, games.player_season[first]] = np.nan_to_num(league_row[first], nan=0.0)
    for rows in games.by_position:
        who = games.player_season[rows]
        current = state[:, who]
        visit(rows, current)
        weight = np.where(games.available[rows], adjusted[rows] / (adjusted[rows] + ks[:, None, :]), 0.0)
        state[:, who] = current + weight * (observed[rows] - current)


def padding_forecasts(games: Games, league_row: np.ndarray, factors: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """(n, 21): the per-stat recursion's forecast before every row (module docstring), each player-season started at
    its league level (0 for a stat its season does not count)."""
    out = np.empty((len(games.position), len(STATS)))

    def keep(rows, current):
        out[rows] = current[0]
    _padding_pass(games, league_row, factors, np.asarray(ks, dtype=float)[None, :], keep)
    return out


def padding_loss(games: Games, league_row: np.ndarray, factors: np.ndarray, ks: np.ndarray,
                 training: np.ndarray) -> np.ndarray:
    """(candidates, 21): the padding's mean NLL per stat over the `training` rows (a mask) for each row of `ks`."""
    total = np.zeros(ks.shape)

    def add(rows, current):
        inside = training[rows]
        if inside.any():
            total[:] += _nll_rows(current[:, inside], games, factors, rows[inside]).sum(axis=1)
    _padding_pass(games, league_row, factors, ks, add)
    return total / np.maximum(games.available[training].sum(axis=0), 1)


def _nll_rows(forecast: np.ndarray, games: Games, factors: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The NLL (module docstring) of `rows` under `forecast` ((..., len(rows), 21)), 0 where a game does not inform a
    stat."""
    eps = 1e-9
    counts, exposure, available = games.counts[rows], games.exposure[rows], games.available[rows]
    adjusted = exposure * factors[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_mean = np.maximum(forecast, eps) * adjusted / 100.0
        rate = rate_mean - counts * np.log(np.maximum(rate_mean, eps))
        share = np.clip(forecast, eps, 1.0 - eps)
        pct = -(counts * np.log(share) + (exposure - counts) * np.log1p(-share))
    return np.where(available, np.where(IS_PCT, pct, rate), 0.0)


def nll(forecast: np.ndarray, games: Games, factors: np.ndarray) -> np.ndarray:
    """(n, 21): every row's NLL under `forecast` (module docstring), 0 where the game does not inform the stat."""
    return _nll_rows(forecast, games, factors, np.arange(len(games.position)))


def _grid_fit(loss, low: float, high: float, coarse: int = 25, fine: int = 17) -> np.ndarray:
    """One value per stat minimising `loss` (a function of (candidates, 21) values, returning (candidates, 21) mean
    losses), searched on a log grid from `low` to `high`, then on a finer one around each stat's best."""
    grid = np.exp(np.linspace(np.log(low), np.log(high), coarse))
    losses = loss(np.repeat(grid[:, None], len(STATS), axis=1))
    best = grid[np.argmin(losses, axis=0)]
    step = (np.log(high) - np.log(low)) / (coarse - 1)
    offsets = np.linspace(-step, step, fine)
    candidates = np.exp(np.log(best)[None, :] + offsets[:, None])
    losses = loss(candidates)
    return candidates[np.argmin(losses, axis=0), np.arange(len(STATS))]


def fit_pads(games: Games, league_row: np.ndarray, training: np.ndarray, opponent: bool = True) -> tuple:
    """(ks, opponent pads): each stat's padding pad k, then its opponent pad, then k again, each minimising the mean
    NLL of the padding's forecasts on the `training` rows (a mask).  A percentage's opponent pad is infinity, and so
    is every stat's without `opponent` (the comparison arm: k alone).

    Each search is two log grids over every stat at once: k from 1 to 100,000 exposure units, the opponent pad from
    100 to 10,000,000 possessions, where the factor is 1 to within about a percent."""
    no_factor = np.ones_like(games.exposure)
    ks = _grid_fit(lambda k: padding_loss(games, league_row, no_factor, k, training), 1.0, 1e5)
    if not opponent:
        return ks, np.full(len(STATS), np.inf)
    sums = opponent_sums(games, league_row)

    def pad_loss(pads):
        out = np.empty(pads.shape)
        for c in range(pads.shape[0]):
            out[c] = padding_loss(games, league_row, opponent_factors(sums, pads[c]), ks[None, :], training)[0]
        return out
    pads = np.where(IS_PCT, np.inf, _grid_fit(pad_loss, 1e2, 1e7, coarse=13, fine=9))
    factors = opponent_factors(sums, pads)
    ks = _grid_fit(lambda k: padding_loss(games, league_row, factors, k, training), 1.0, 1e5)
    return ks, pads


# ------------------------------------------------------------------------------------------ the state
_SETS = [f"recent{int(tau)}" for tau in TAUS] + ["season"]
_PLAYING = ["games_played", "first_game", "days_since_last_game", "possessions_per_game", "team_possession_share",
            "share_of_team_games", "starts_share", "season_progress", "home", "playoffs", "hustle_counted"]


def input_names() -> list:
    """The state's columns, in order: per sum set every stat's level and the exposures, the playing time and context,
    the padding's forecasts."""
    names = []
    for name in _SETS:
        names += [f"{name}_{stat}" for stat in NAMES]
        names += [f"{name}_offensive_possessions", f"{name}_defensive_possessions"]
        names += [f"{name}_{NAMES[j]}_attempts" for j in ATTEMPTS_OF]
    return names + _PLAYING + [f"padding_{stat}" for stat in NAMES]


def _link(value: np.ndarray, league: np.ndarray) -> np.ndarray:
    """A forecast as an input: log over the league level for a count, the logit's difference for a percentage, so the
    league level reads 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.clip(value, 1e-6, 1 - 1e-6)
        level = np.clip(league, 1e-6, 1 - 1e-6)
        out = np.where(IS_PCT, np.log(share) - np.log1p(-share) - (np.log(level) - np.log1p(-level)),
                       np.log(np.maximum(value, 1e-9)) - np.log(np.maximum(league, 1e-9)))
    return out


def playing_time(games: Games) -> np.ndarray:
    """(n, len(_PLAYING)) the playing time and context before every row, from his earlier games of the season and the
    team's: games played (log), first game, days since his last game (log, at most 30), possessions per game (log),
    share of his team's possessions in its games so far, his games over his team's, share of games started (padded
    by two games toward a half), the share of the regular season played league-wide by that day, home (+1, -1, 0 at a
    neutral site), playoffs, whether hustle is counted.  An end row reads his whole season, at no site, as the regular
    season ends."""
    frame = games.frame
    end = frame["end"].to_numpy(dtype=bool)
    key = [frame["season"], frame["player_id"]]
    played = games.position.astype(float)
    both = (frame["poss_off"] + frame["poss_def"]).to_numpy(dtype=float)
    own_before = pd.Series(both).groupby(key).cumsum().to_numpy() - both
    starts_before = pd.Series(frame["started"].to_numpy(dtype=float)).groupby(key).cumsum().to_numpy() - \
        frame["started"].to_numpy(dtype=float)
    previous = frame.groupby(["season", "player_id"], sort=False)["game_date"].shift(1)
    days = (frame["game_date"] - previous).dt.days.to_numpy(dtype=float)
    team_before, team_games_before = _team_before(frame)
    with_team = [frame["season"], frame["player_id"], frame["team_id"]]
    team_since = team_games_before - pd.Series(team_games_before).groupby(with_team).transform("first").to_numpy()
    played_with_team = pd.Series(np.ones(len(frame))).groupby(with_team).cumcount().to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.clip(np.where(team_before > 0, own_before / team_before, 0.0), 0.0, 0.9)
        games_share = np.where(team_since > 0, played_with_team / np.maximum(team_since, 1.0), 1.0)
    progress = _season_progress(frame)
    out = np.column_stack([
        np.log1p(played), (games.position == 0).astype(float), np.log1p(np.minimum(np.nan_to_num(days, nan=30.0), 30)),
        np.log1p(np.where(played > 0, 0.5 * own_before / np.maximum(played, 1.0), 0.0)), share,
        np.clip(games_share, 0.0, 1.0), (starts_before + 1.0) / (played + 2.0), progress,
        np.where(end, 0.0, frame["home"].to_numpy(dtype=float)),
        np.where(end, 0.0, (frame["phase"] == "PO").to_numpy(dtype=float)),
        (frame["season"].to_numpy() >= 2017).astype(float)])
    return out


def _team_before(frame: pd.DataFrame) -> tuple:
    """(his team's possessions at both ends in its games before this one, its games before this one), per row; an end
    row reads the team's whole season."""
    real = ~frame["end"].to_numpy(dtype=bool)
    team_games = (frame.loc[real].assign(poss=frame["team_poss_off"] + frame["team_poss_def"])
                  .drop_duplicates(["season", "team_id", "game_idx"])
                  [["season", "team_id", "game_idx", "poss"]].sort_values(["season", "team_id", "game_idx"]))
    grouped = team_games.groupby(["season", "team_id"], sort=False)
    team_games["poss_before"] = grouped["poss"].cumsum() - team_games["poss"]
    team_games["games_before"] = grouped.cumcount().astype(float)
    totals = grouped.agg(total=("poss", "sum"), n=("poss", "size")).reset_index()
    joined = frame[["season", "team_id", "game_idx"]].merge(
        team_games[["season", "team_id", "game_idx", "poss_before", "games_before"]],
        on=["season", "team_id", "game_idx"], how="left").merge(totals, on=["season", "team_id"], how="left")
    poss = np.where(real, joined["poss_before"], joined["total"])
    count = np.where(real, joined["games_before"], joined["n"])
    return poss.astype(float), count.astype(float)


def _season_progress(frame: pd.DataFrame) -> np.ndarray:
    """The share of the season's regular-season games played on days before the row's game: 1 in the playoffs and on
    an end row."""
    real = ~frame["end"].to_numpy(dtype=bool)
    regular = frame[real & (frame["phase"] == "RS").to_numpy()].drop_duplicates("game_id")[["season", "game_date"]]
    out = np.ones(len(frame))
    for season, dates in regular.groupby("season")["game_date"]:
        ordered = np.sort(dates.to_numpy())
        rows = np.flatnonzero(real & (frame["season"] == season).to_numpy() & (frame["phase"] == "RS").to_numpy())
        out[rows] = np.searchsorted(ordered, frame["game_date"].to_numpy()[rows], side="left") / len(ordered)
    return out


def network_inputs(games: Games, league_row: np.ndarray, factors: np.ndarray, padding: np.ndarray) -> np.ndarray:
    """(n, len(input_names())) float32: the state before every row (module docstring).  A game's own counts never
    reach its own row: each row is read before the game is added to the sums."""
    n_ps, n_stats, n_sets = games.n_player_seasons, len(STATS), len(_SETS)
    made = np.zeros((n_ps, n_stats, n_sets))
    seen = np.zeros((n_ps, n_stats, n_sets))
    poss = np.zeros((n_ps, 2, n_sets))
    _, adjusted = _observed(games, factors)
    off = games.frame["poss_off"].to_numpy(dtype=float)
    dfn = games.frame["poss_def"].to_numpy(dtype=float)
    attempts_columns = list(ATTEMPTS_OF)
    per_set = n_stats + 2 + len(attempts_columns)
    sums = np.empty((len(games.position), n_sets * per_set), dtype=np.float32)
    taus = np.array([*TAUS, np.inf])
    for rows in games.by_position:
        who = games.player_season[rows]
        league = league_row[rows][:, :, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.log((100.0 * made[who] + K0 * league) / (seen[who] + K0)) - np.log(league)
            share = (made[who] + A0 * league) / (seen[who] + A0)
            share = np.clip(share, 1e-6, 1 - 1e-6)
            level = np.clip(league, 1e-6, 1 - 1e-6)
            pct = np.log(share) - np.log1p(-share) - (np.log(level) - np.log1p(-level))
        levels = np.where(IS_PCT[None, :, None], pct, rate)
        block = np.concatenate([levels, np.log1p(poss[who]), np.log1p(seen[who][:, attempts_columns])], axis=1)
        sums[rows] = np.nan_to_num(block.transpose(0, 2, 1).reshape(len(rows), -1), nan=0.0)
        decay = np.exp(-games.possessions[rows][:, None] / taus[None, :])
        available = games.available[rows][:, :, None]
        made[who] = made[who] * decay[:, None, :] + np.where(available, games.counts[rows][:, :, None], 0.0)
        seen[who] = seen[who] * decay[:, None, :] + np.where(available, adjusted[rows][:, :, None], 0.0)
        poss[who] = poss[who] * decay[:, None, :] + np.stack([off[rows], dfn[rows]], axis=1)[:, :, None]
    return np.concatenate([sums, playing_time(games).astype(np.float32),
                           np.nan_to_num(_link(padding, league_row), nan=0.0).astype(np.float32)], axis=1)


@dataclass
class Standardised:
    """The inputs' training means and standard deviations, and where home and playoffs sit among them."""
    mean: np.ndarray
    sd: np.ndarray

    def apply(self, inputs: np.ndarray) -> np.ndarray:
        return ((inputs - self.mean) / self.sd).astype(np.float32)


def standardise(inputs: np.ndarray, training: np.ndarray) -> Standardised:
    """Each input's mean and standard deviation over the `training` rows (a mask); a constant input divides by 1."""
    mean = inputs[training].mean(axis=0, dtype=np.float64)
    sd = inputs[training].std(axis=0, dtype=np.float64)
    return Standardised(mean=mean.astype(np.float32), sd=np.where(sd > 1e-9, sd, 1.0).astype(np.float32))


def neutral_inputs(inputs: np.ndarray) -> np.ndarray:
    """The state with the game itself made neutral: no home court, the regular season.  What a forecast against an
    average opponent reads (the opponent is already out: it enters as the exposure's factor, never as an input)."""
    out = inputs.copy()
    names = input_names()
    out[:, names.index("home")] = 0.0
    out[:, names.index("playoffs")] = 0.0
    return out


# ------------------------------------------------------------------------------------------ the network
def _offsets(padding: np.ndarray) -> np.ndarray:
    """The padding's forecasts as links: log of a count per 100, logit of a percentage."""
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.clip(padding, 1e-6, 1 - 1e-6)
        links = np.where(IS_PCT, np.log(share) - np.log1p(-share), np.log(np.maximum(padding, 1e-9)))
    return np.nan_to_num(links, nan=0.0).astype(np.float32)


def _scales(games: Games, factors: np.ndarray) -> np.ndarray:
    """(n, 21): what the loss multiplies the link's value by: log(exposure x factor / 100) for a count, the attempts
    for a percentage."""
    _, adjusted = _observed(games, factors)
    with np.errstate(divide="ignore"):
        return np.where(IS_PCT, games.exposure, np.log(np.maximum(adjusted, 1e-9) / 100.0)).astype(np.float32)


class _Torch:
    """The network as PyTorch sees it while training: its layers, its forward pass and its loss, on one device."""

    def __init__(self, torch, n_inputs: int, widths, seed: int, device: str):
        self.torch, self.device = torch, device
        self.is_pct = torch.as_tensor(IS_PCT, device=device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        sizes = [n_inputs, *widths]
        self.params = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            weight = torch.randn(fan_in, fan_out, generator=generator) * np.sqrt(2.0 / fan_in)
            self.params += [weight.to(device).requires_grad_(),
                            torch.zeros(fan_out, device=device, requires_grad=True)]
        # the last layer at zero: before training the network IS the padding
        self.params += [torch.zeros(sizes[-1], len(STATS), device=device, requires_grad=True),
                        torch.zeros(len(STATS), device=device, requires_grad=True)]
        self.weights = self.params[0:-2:2] + [self.params[-2]]

    def forward(self, x, offset, drop: float):
        functional = self.torch.nn.functional
        h = x
        for layer in range(0, len(self.params) - 2, 2):
            h = functional.silu(h @ self.params[layer] + self.params[layer + 1])
            if drop > 0:
                h = functional.dropout(h, p=drop, training=True)
        return offset + h @ self.params[-2] + self.params[-1]

    def summed_nll(self, link, y, scale, mask):
        """Each row's NLL summed over the 21 stats; a stat the game does not inform adds exactly 0."""
        torch = self.torch
        rate = torch.exp(link + scale) - y * (link + scale)
        pct = scale * torch.nn.functional.softplus(link) - y * link
        return (torch.where(self.is_pct, pct, torch.where(mask > 0, rate, torch.zeros_like(rate))) * mask).sum(dim=1)

    def mean_nll(self, data, rows: np.ndarray, chunk: int = 65536) -> float:
        """The mean summed NLL of `rows`, no dropout, no gradient."""
        total = 0.0
        with self.torch.no_grad():
            for start in range(0, len(rows), chunk):
                index = self.torch.as_tensor(rows[start:start + chunk], device=self.device)
                x, offset, y, scale, mask = (array[index] for array in data)
                total += float(self.summed_nll(self.forward(x, offset, 0.0), y, scale, mask).sum())
        return total / max(len(rows), 1)

    def snapshot(self) -> list:
        return [param.detach().clone() for param in self.params]


def _one_pass(net: _Torch, optimiser, data, order: np.ndarray, config: dict, step: int, total_steps: int) -> tuple:
    """One pass over the training rows in batches: (the mean training loss, the steps taken so far)."""
    running, batches = 0.0, 0
    for start in range(0, len(order), config["batch"]):
        index = net.torch.as_tensor(order[start:start + config["batch"]], device=net.device)
        x, offset, y, scale, mask = (array[index] for array in data)
        step += 1
        for group in optimiser.param_groups:
            group["lr"] = config["lr"] * 0.5 * (1.0 + np.cos(np.pi * step / total_steps))
        loss = net.summed_nll(net.forward(x, offset, config["drop"]), y, scale, mask).mean()
        loss = loss + config["wd"] * sum((weight ** 2).sum() for weight in net.weights)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        running, batches = running + float(loss.detach()), batches + 1
    return running / max(batches, 1), step


def train_network(inputs: np.ndarray, games: Games, factors: np.ndarray, padding: np.ndarray, training: np.ndarray,
                  validation: np.ndarray, seed: int = 0, device: str = "cuda", verbose: bool = False,
                  **settings) -> tuple:
    """(the best weights as numpy [(W, b), ...], the history [(pass, training loss, validation NLL)]): the network
    fitted on the `training` rows (a mask) by the summed NLL, stopped on the `validation` rows' NLL.

    PyTorch is imported here, so nothing else needs it; on the GPU with deterministic algorithms, so the same draws
    give the same network.  The handoff's loop (scratch/2026-10-27_multistat/model.py), ported: weights He-initialised,
    the last layer at zero, SiLU between layers, dropout on each hidden layer, an L2 penalty on the weights added to
    the mean summed NLL, Adam with the learning rate decayed on a cosine over every step."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    config = {**BEST, **settings}
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    data = [torch.as_tensor(array, device=device) for array in
            (inputs, _offsets(padding), games.counts.astype(np.float32), _scales(games, factors),
             games.available.astype(np.float32))]
    net = _Torch(torch, inputs.shape[1], config["widths"], seed, device)
    optimiser = torch.optim.Adam(net.params, lr=config["lr"], betas=(0.9, 0.999), eps=1e-8)
    training_rows, validation_rows = np.flatnonzero(training), np.flatnonzero(validation)
    rng = np.random.default_rng(seed)
    total_steps = int(np.ceil(len(training_rows) / config["batch"])) * config["max_epochs"]
    best, best_loss, bad, step, history = net.snapshot(), np.inf, 0, 0, []
    for epoch in range(config["max_epochs"]):
        started = time.perf_counter()
        loss, step = _one_pass(net, optimiser, data, rng.permutation(training_rows), config, step, total_steps)
        held = net.mean_nll(data, validation_rows)
        history.append((epoch, loss, held))
        if verbose:
            print(f"    pass {epoch:2d}  training loss {loss:9.5f}  validation NLL {held:9.5f}  "
                  f"{time.perf_counter() - started:5.1f}s", flush=True)
        if held < best_loss - 1e-6:
            best, best_loss, bad = net.snapshot(), held, 0
        else:
            bad += 1
            if bad >= config["patience"]:
                break
    arrays = [param.cpu().numpy().astype(np.float64) for param in best]
    return [(arrays[i], arrays[i + 1]) for i in range(0, len(arrays), 2)], history


def forward_numpy(weights: list, inputs: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """The trained network in plain numpy, no PyTorch: (n, 21) links."""
    h = np.asarray(inputs, dtype=np.float64)
    for weight, bias in weights[:-1]:
        a = h @ weight + bias
        h = a / (1.0 + np.exp(-a))
    weight, bias = weights[-1]
    return np.asarray(offsets, dtype=np.float64) + h @ weight + bias


def forecast(weights: list, scaling: Standardised, inputs: np.ndarray, padding: np.ndarray,
             neutral: bool = True, chunk: int = 200_000) -> np.ndarray:
    """(n, 21) the network's forecasts, as rates per 100 and make shares: at a neutral game (`neutral_inputs`) by
    default, against an average opponent always."""
    rows = neutral_inputs(inputs) if neutral else inputs
    offsets = _offsets(padding)
    out = np.empty((len(rows), len(STATS)))
    for start in range(0, len(rows), chunk):
        link = forward_numpy(weights, scaling.apply(rows[start:start + chunk]), offsets[start:start + chunk])
        out[start:start + chunk] = np.where(IS_PCT, 1.0 / (1.0 + np.exp(-link)), np.exp(link))
    return out
