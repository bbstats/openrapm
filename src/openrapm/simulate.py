"""Simulated seasons with a known truth: stints and per-game box counts from players whose impact is set.

The tests check the model on these: a fit that cannot recover a truth it was given is wrong, whatever it does on
real seasons.

Words used below
    rate      a player's true count per 100 possessions of each box-score count (design.FEATURES)
    impact    a player's true effect on points per 100, offence and defence: rate . beta + u, with u a part no
              box count explains
    leak      made shots enter a stint's points mechanically (3 / 2 / 1 per make, the counts that happened),
              while every other count, and the "net" value of a make, enters through the player's true rate

With the leak on, the true coefficient on threes made is beta_3 (< 3) but the game's own count of threes made is
correlated with the stint's residual with slope 3 -- exactly what full-season rates leak and leave-one-game-out
rates do not.

Example: with the default league means, a player who makes 2.4 threes and 5.6 twos per 100 (and the league mean of
every other count) has offensive impact 1.6 x 2.4 + 0.9 x 5.6 + ... + u.

The random numbers are drawn in one fixed order (season by season: roster moves, rates and impacts, playing time,
schedule, then game by game and stint by stint), so a seed gives the same league on every machine.

The season panel the box-score prior reads (step 2) needs more than box counts: a player's role, career and body.
`player_profiles` counts the role and career columns from the simulated games and draws a height and weight that say
nothing about impact; `simulated_panel` adds the padded box-score rates of each season's design, as data/panel.py does
for real seasons.  Neither touches `simulate`'s random numbers.  examples/tutorial.py runs steps 1-3 on the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from .box_rates import box_rates, rate_table
from .design import AWAY_SLOTS, FEATURES, HOME_SLOTS
from .features import BOX_RATES, SEASON_PANEL_COLUMNS

LEAGUE_MEAN = np.array([2.4, 4.2, 5.6, 5.2, 3.6, 1.0, 2.0, 6.6, 4.8, 2.8, 1.6, 1.0, 4.0])
BETA_O = np.array([1.6, -0.7, 0.9, -0.6, 0.5, -0.4, 0.5, 0.1, 0.4, -1.0, 0.1, 0.0, -0.1])
BETA_D = np.array([-0.1, 0.1, -0.1, 0.2, 0.0, 0.1, 0.0, -0.3, -0.1, 0.1, -0.8, -0.6, 0.3])
# Same-game mechanical slopes: realised counts enter the stint's points with these values
# (a make scores; a miss forfeits the possession's ~1 point unless an ORB follows; an ORB
# restores it; a turnover forfeits it).  The player's TRUE rate enters with beta, so the
# true coefficient is beta while the game's box count leaks with slope m.
MECH = {"fg3m": 3.0, "fg2m": 2.0, "ftm": 1.0, "fg3_miss": -0.75, "fg2_miss": -0.75, "orb": 0.75, "tov": -1.0}
GAMMA = dict(home=1.5, intercept=60.0, is_gt=-2.0, margin=-0.15, margin_frem=-0.10, po_int=-1.0, po_home=0.5)


@dataclass(frozen=True)
class _League:
    """What stays the same over every simulated season."""
    features: list
    mech: np.ndarray             # each count's same-game slope (MECH), 0 for a count without one
    is_mech: np.ndarray          # mech != 0
    pid: np.ndarray              # player ids: 1000 x (first team + 1) + place in the team (1, 2, ...)
    stints_per_game: tuple       # (fewest, most) stints in a game
    eps_var: float               # the noise variance of a stint's points, per possession
    leak: bool


class _Season(NamedTuple):
    """One simulated season's players."""
    season: int
    team_of: np.ndarray          # each player's team this season
    mu: np.ndarray               # (players x counts) true rates
    impact: dict                 # "RS" / "PO" -> (offensive impacts, defensive impacts)
    prop: np.ndarray             # each player's weight in the draw of who is on the court


@dataclass
class _Game:
    """One game being played: who can play, and what has happened so far."""
    game_id: str
    game_number: int
    phase: str
    series: str
    home_players: np.ndarray
    away_players: np.ndarray
    box: np.ndarray              # (players x counts) the game's box counts
    possessions: np.ndarray      # (players,) the game's offensive possessions
    score_h: float = 0.0
    score_a: float = 0.0


def _sample_lineup(rng, weights):
    """Five distinct players, probability ~ weights (Gumbel top-k)."""
    keys = np.log(weights) + rng.gumbel(size=len(weights))
    return np.sort(np.argpartition(-keys, 5)[:5])


def simulate(n_seasons=2, n_teams=10, players_per_team=12, games_per_season=150, stints_per_game=(30, 50),
             beta_O=BETA_O, beta_D=BETA_D, tau_u=(1.5, 1.5), eps_var=0.5, rate_cv=0.5, seed=0, leak=True,
             po_series_per_season=0, delta_O=None, delta_D=None, first_season=2001, rho=None, turnover=0.0):
    """Return dict(stints=DataFrame, box=DataFrame, truth=dict).

    `rho` and `turnover` exist for the year-over-year test (openrapm.scoring), which needs a player's
    talent to PERSIST across seasons and rosters to CHANGE:
      rho       None (default) redraws every player's rates and impacts independently each season, the
                original behaviour, which every test fixture without `rho` was built with.  A
                float in [0, 1] makes the impacts AR(1) across seasons with that correlation (and the
                rates AR(1) on the log scale), so a rating fit on one season carries information about
                the next.
      turnover  the fraction of players moved to a different team each season (a permutation among the
                movers, so every team keeps its size).  Player ids persist either way.
    `po_series_per_season` playoff series of 4 to 7 games follow each regular season; in them every player's
    impact moves by rate . delta_O (offence) and rate . delta_D (defence).
    """
    rng = np.random.default_rng(seed)
    beta_O, beta_D = np.asarray(beta_O, dtype=float), np.asarray(beta_D, dtype=float)
    delta_O = np.zeros(len(FEATURES)) if delta_O is None else np.asarray(delta_O, dtype=float)
    delta_D = np.zeros(len(FEATURES)) if delta_D is None else np.asarray(delta_D, dtype=float)
    team_of = np.repeat(np.arange(n_teams), players_per_team)
    place = np.tile(np.arange(players_per_team), n_teams)
    mech = np.array([MECH.get(count, 0.0) for count in FEATURES])
    league = _League(FEATURES, mech, mech != 0, 1000 * (team_of + 1) + place + 1, stints_per_game, eps_var, leak)
    stint_rows, box_rows, truth_rows = [], [], []
    game_number = 0
    talent = (None, None, None)
    for number in range(n_seasons):
        if number > 0 and turnover > 0:
            team_of = _move_players(rng, team_of, turnover)
        talent = _season_talent(rng, talent, rho, tau_u, rate_cv, len(team_of))
        state = _season_state(rng, league, first_season + number, team_of, talent, (beta_O, beta_D, delta_O, delta_D),
                              place)
        truth_rows.append(_truth_frame(league, state, talent))
        for home, away, phase, series in _schedule(rng, n_teams, games_per_season, po_series_per_season,
                                                   state.season):
            game_number += 1
            stints, box = _play_game(rng, league, state, home, away, phase, series, game_number)
            stint_rows += stints
            box_rows += box
    truth = dict(ps=pd.concat(truth_rows, ignore_index=True), beta_O=beta_O, beta_D=beta_D,
                 delta_O=delta_O, delta_D=delta_D, gamma=GAMMA, features=FEATURES)
    return dict(stints=pd.DataFrame(stint_rows), box=pd.DataFrame(box_rows), truth=truth)


def _move_players(rng, team_of: np.ndarray, turnover: float) -> np.ndarray:
    """A share `turnover` of the players, drawn at random, swap teams among themselves (teams keep their size)."""
    n_players = len(team_of)
    who = rng.choice(n_players, int(round(turnover * n_players)), replace=False)
    team_of = team_of.copy()
    team_of[who] = team_of[rng.permutation(who)]
    return team_of


def _season_talent(rng, previous: tuple, rho, tau_u, rate_cv: float, n_players: int) -> tuple:
    """(rates, u offence, u defence) of one season.

    rho None: rates gamma with mean LEAGUE_MEAN and coefficient of variation rate_cv, u normal with standard
    deviations tau_u, redrawn every season.  rho a number: the first season from the stationary law (log-normal
    rates with the same mean and coefficient of variation), so a player's spread is the same in every season and
    his neighbours predict him with slope 2 rho / (1 + rho^2); every later season AR(1) with correlation rho, the
    rates on the log scale."""
    mu, u_off, u_def = previous
    n_counts = len(LEAGUE_MEAN)
    shape = 1.0 / rate_cv ** 2
    sd_log = np.sqrt(np.log1p(rate_cv ** 2))
    mean_log = np.log(LEAGUE_MEAN) - 0.5 * sd_log ** 2
    if rho is not None and mu is not None:
        z = rng.normal(size=(n_players, n_counts))
        mu = np.exp(mean_log + rho * (np.log(mu) - mean_log) + np.sqrt(1 - rho ** 2) * sd_log * z)
        u_off = rho * u_off + np.sqrt(1 - rho ** 2) * rng.normal(0, tau_u[0], n_players)
        u_def = rho * u_def + np.sqrt(1 - rho ** 2) * rng.normal(0, tau_u[1], n_players)
        return mu, u_off, u_def
    if rho is None:
        mu = rng.gamma(shape, LEAGUE_MEAN / shape, size=(n_players, n_counts))
    else:
        mu = np.exp(mean_log + sd_log * rng.normal(size=(n_players, n_counts)))
    u_off = rng.normal(0, tau_u[0], n_players)
    u_def = rng.normal(0, tau_u[1], n_players)
    return mu, u_off, u_def


def _season_state(rng, league: _League, season: int, team_of, talent: tuple, slopes: tuple, place) -> _Season:
    """The season's impacts (regular season and playoffs) and playing-time weights: a team's first five weigh
    three times as much as the rest, times a gamma(2, 1) draw per player."""
    mu, u_off, u_def = talent
    beta_O, beta_D, delta_O, delta_D = slopes
    impact_O = mu @ beta_O + u_off
    impact_D = mu @ beta_D + u_def
    prop = rng.gamma(2.0, 1.0, len(team_of)) * np.where(place < 5, 3.0, 1.0)
    impact = {"RS": (impact_O, impact_D), "PO": (impact_O + mu @ delta_O, impact_D + mu @ delta_D)}
    return _Season(season, team_of, mu, impact, prop)


def _truth_frame(league: _League, state: _Season, talent: tuple) -> pd.DataFrame:
    """One row per player of the season: id, team, u, impacts and true rates."""
    mu, u_off, u_def = talent
    impact_O, impact_D = state.impact["RS"]
    return pd.DataFrame({"player_id": league.pid, "season": state.season, "team": state.team_of, "u_O": u_off,
                         "u_D": u_def, "impact_O": impact_O, "impact_D": impact_D,
                         **{f"rate_{count}": mu[:, index] for index, count in enumerate(league.features)}})


def _schedule(rng, n_teams: int, games_per_season: int, po_series_per_season: int, season: int) -> list:
    """(home, away, phase, series id) of every game: regular-season games between two random teams, then playoff
    series of 4 to 7 games with home court 2-2-1-1-1."""
    games = []
    for _ in range(games_per_season):
        home, away = rng.choice(n_teams, 2, replace=False)
        games.append((home, away, "RS", ""))
    for _ in range(po_series_per_season):
        first, second = rng.choice(n_teams, 2, replace=False)
        n_games = int(rng.integers(4, 8))
        for game in range(n_games):
            home, away = (first, second) if game in (0, 1, 4, 6) else (second, first)
            games.append((home, away, "PO", f"{season}-{min(first, second)}-{max(first, second)}"))
    return games


def _play_game(rng, league: _League, state: _Season, home, away, phase: str, series: str,
               game_number: int) -> tuple[list, list]:
    """(stint rows, box rows) of one game."""
    n_players, n_counts = state.mu.shape
    game = _Game(f"{state.season}{game_number:05d}", game_number, phase, series,
                 np.flatnonzero(state.team_of == home), np.flatnonzero(state.team_of == away),
                 np.zeros((n_players, n_counts)), np.zeros(n_players))
    n_stints = int(rng.integers(league.stints_per_game[0], league.stints_per_game[1] + 1))
    stint_rows = [_play_stint(rng, league, state, game, 1.0 - t / n_stints) for t in range(n_stints)]
    return stint_rows, _box_rows(rng, league, state, game)


def _play_stint(rng, league: _League, state: _Season, game: _Game, frac_rem: float) -> dict:
    """One stint: both fives drawn, possessions drawn, each side's points; the score moves on."""
    margin_h = game.score_h - game.score_a
    is_gt = (frac_rem < 0.125 and abs(margin_h) >= 15) or (frac_rem < 0.25 and abs(margin_h) >= 20)
    home_five = game.home_players[_sample_lineup(rng, state.prop[game.home_players])]
    away_five = game.away_players[_sample_lineup(rng, state.prop[game.away_players])]
    poss_h = 1 + rng.poisson(6)
    poss_a = max(1, poss_h + int(rng.integers(-1, 2)))
    pts = {}
    for off, de, poss, home, margin in ((home_five, away_five, poss_h, 1.0, margin_h),
                                        (away_five, home_five, poss_a, -1.0, -margin_h)):
        pts[home] = _side_points(rng, league, state, game, (off, de, poss, home, margin, frac_rem, is_gt))
    row = dict(game_id=game.game_id, season=state.season, phase=game.phase, series_id=game.series,
               game_date=game.game_number, period=1 + min(int(4 * (1 - frac_rem)), 3),
               **{slot: int(league.pid[player]) for slot, player in zip(HOME_SLOTS, home_five)},
               **{slot: int(league.pid[player]) for slot, player in zip(AWAY_SLOTS, away_five)},
               poss_h=poss_h, poss_a=poss_a, pts_h=pts[1.0], pts_a=pts[-1.0],
               margin_h=margin_h, frac_rem=frac_rem, is_gt=bool(is_gt), neutral=False)
    game.score_h += pts[1.0]
    game.score_a += pts[-1.0]
    return row


def _side_points(rng, league: _League, state: _Season, game: _Game, side: tuple) -> float:
    """The points of one side of a stint: the expected points per 100 from the ten impacts and the context, times
    possessions / 100, plus normal noise, plus (with the leak) the game's own mechanical counts minus their
    expectation."""
    off, de, poss, home, margin, frac_rem, is_gt = side
    impact_O, impact_D = state.impact["PO" if game.phase == "PO" else "RS"]
    e100 = impact_O[off].sum() + impact_D[de].sum() + GAMMA["intercept"] + GAMMA["home"] * home \
        + GAMMA["is_gt"] * is_gt + GAMMA["margin"] * np.clip(margin, -25, 25) \
        + GAMMA["margin_frem"] * np.clip(margin, -25, 25) * frac_rem
    if game.phase == "PO":
        e100 += GAMMA["po_int"] + GAMMA["po_home"] * home
    e_pts = e100 * poss / 100.0
    points = e_pts + rng.normal(0, np.sqrt(league.eps_var * poss))
    if league.leak:
        points += _leaked_points(rng, league, state, game, off, poss)
    game.possessions[off] += poss
    return points


def _leaked_points(rng, league: _League, state: _Season, game: _Game, off, poss) -> float:
    """The five's mechanical counts drawn for this stint (into the game's box score), and what they add to the
    points beyond their expectation."""
    mech = league.mech[league.is_mech]
    expected = state.mu[off][:, league.is_mech]
    counts = rng.poisson(expected * poss / 100.0)
    game.box[off[:, None], np.flatnonzero(league.is_mech)[None, :]] += counts
    return (counts * mech).sum() - (expected * mech).sum() * poss / 100.0


def _box_rows(rng, league: _League, state: _Season, game: _Game) -> list:
    """The game's box rows, one per player who played: the counts with no mechanical slope (all of them without
    the leak) drawn from his rate and possessions."""
    played = np.flatnonzero(game.possessions > 0)
    other = ~league.is_mech if league.leak else np.ones(len(league.features), bool)
    game.box[played[:, None], np.flatnonzero(other)[None, :]] = rng.poisson(
        state.mu[played][:, other] * game.possessions[played, None] / 100.0)
    return [dict(game_id=game.game_id, player_id=int(league.pid[player]), season=state.season, phase=game.phase,
                 **{count: float(game.box[player, index]) for index, count in enumerate(league.features)})
            for player in played]


# ------------------------------------------------------------------------------ the season panel of a simulated league
# The body of a simulated player: height normal with this mean and standard deviation (inches), weight a line in
# height plus normal noise (pounds).  Nothing in the league reads either, so a prior that leans on them fits noise.
HEIGHT_MEAN, HEIGHT_SD = 79.0, 3.3
WEIGHT_AT_MEAN_HEIGHT, POUNDS_PER_INCH, WEIGHT_SD = 215.0, 6.5, 12.0
PROFILE_COLUMNS = ["team_possession_share", "games_started_share", "seasons_before", "career_possessions_before_k",
                   "seasons_with_team", "height_in", "weight_lb"]


def player_profiles(sim: dict, seed: int = 0) -> pd.DataFrame:
    """The season panel's role, career and body columns for every player-season of a simulated league.

    The box score gives the panel its padded rates (box_rates.py); the shipped prior also reads the columns below,
    which `simulate` does not make.  Counted from the simulated games (regular season and playoffs):

        team_possession_share        his offensive possessions / his team's offensive possessions that season
        games_started_share          the share of his games in which he was on the court for the first stint
        seasons_before               the simulated seasons he played before this one (the league begins with the
                                     first simulated season, so everyone is a rookie in it)
        career_possessions_before_k  his offensive possessions in those seasons, in thousands
        seasons_with_team            the seasons in a row he has been with this season's team, this one included

    Drawn once per player from a generator of their own, numpy's default_rng([seed, 1]), so `simulate`'s draws are
    untouched, and unrelated to his impact:

        height_in   normal, mean 79, standard deviation 3.3
        weight_lb   215 + 6.5 x (height - 79) + normal noise with standard deviation 12

    Example: a player on the court for 4,200 of his team's 8,400 offensive possessions, who started 20 of his 30 games,
    has team_possession_share 0.5 and games_started_share 0.67.
    """
    seasons = _season_appearances(sim["stints"]).merge(sim["truth"]["ps"][["player_id", "season", "team"]],
                                                       on=["player_id", "season"], how="left")
    team_possessions = seasons.groupby(["season", "team"]).possessions.transform("sum") / 5.0
    seasons["team_possession_share"] = seasons.possessions / team_possessions
    seasons["games_started_share"] = seasons.starts / seasons.games
    seasons = seasons.sort_values(["player_id", "season"]).reset_index(drop=True)
    per_player = seasons.groupby("player_id")
    seasons["seasons_before"] = per_player.cumcount().astype(float)
    seasons["career_possessions_before_k"] = (per_player.possessions.cumsum() - seasons.possessions) / 1000.0
    seasons["seasons_with_team"] = _seasons_in_a_row(seasons)
    return seasons.merge(_bodies(seasons.player_id.unique(), seed), on="player_id")[
        ["player_id", "season", *PROFILE_COLUMNS]]


def _season_appearances(stints: pd.DataFrame) -> pd.DataFrame:
    """Per player-season: his offensive possessions, his games, and the games he started (on the court for the
    game's first stint)."""
    first = (~stints["game_id"].duplicated()).to_numpy()
    parts = [pd.DataFrame({"player_id": stints[slot].to_numpy(), "season": stints["season"].to_numpy(),
                           "game_id": stints["game_id"].to_numpy(), "possessions": stints[possessions].to_numpy(),
                           "started": first})
             for slots, possessions in ((HOME_SLOTS, "poss_h"), (AWAY_SLOTS, "poss_a")) for slot in slots]
    appearances = pd.concat(parts, ignore_index=True)
    return (appearances.groupby(["player_id", "season"], as_index=False)
            .agg(possessions=("possessions", "sum"), games=("game_id", "nunique"), starts=("started", "sum")))


def _seasons_in_a_row(seasons: pd.DataFrame) -> np.ndarray:
    """Each row's run of consecutive seasons with the same team, counting this one; `seasons` sorted by player, then
    season.  A season he missed breaks the run."""
    same = ((seasons.player_id == seasons.player_id.shift()) & (seasons.team == seasons.team.shift())
            & (seasons.season == seasons.season.shift() + 1))
    run_start = (~same).cumsum()
    return (seasons.groupby(run_start).cumcount() + 1).to_numpy(dtype=float)


def _bodies(player_ids, seed: int) -> pd.DataFrame:
    """Each player's height and weight, from numpy's default_rng([seed, 1])."""
    rng = np.random.default_rng([seed, 1])
    n_players = len(player_ids)
    height = rng.normal(HEIGHT_MEAN, HEIGHT_SD, n_players)
    weight = WEIGHT_AT_MEAN_HEIGHT + POUNDS_PER_INCH * (height - HEIGHT_MEAN) + rng.normal(0, WEIGHT_SD, n_players)
    return pd.DataFrame({"player_id": player_ids, "height_in": height, "weight_lb": weight})


def simulated_panel(designs: dict, profiles: pd.DataFrame, pad_target: str = "poss_conditional") -> pd.DataFrame:
    """The season panel of a simulated league: one row per player, season and side, as data/panel.py builds it
    for real seasons, with the columns a simulated league can give.

    `designs` maps each season to its own design (one season each, `design.assemble`); `profiles` is
    `player_profiles`.  Per season and side: his possessions on that side and the thirteen padded rates with their
    `_uncentred` twins (`box_rates.box_rates`, the same call the real panel makes), then his profile.  Columns in
    `features.SEASON_PANEL_COLUMNS` order; the score state, shot totals, on-court and playoff columns are absent.
    """
    rows = []
    for season in sorted(designs):
        rates = box_rates(designs[season], pad_target=pad_target)
        players = designs[season].spec.players["player_id"].to_numpy()
        for side, chosen in (("O", rates.offence), ("D", rates.defence)):
            keys = pd.DataFrame({"side": side, "player_id": players, "season": season,
                                 "possessions": chosen.possessions})
            rows.append(pd.concat([keys, rate_table(rates, side, BOX_RATES)], axis=1))
    panel = pd.concat(rows, ignore_index=True).merge(profiles, on=["player_id", "season"], how="left")
    return panel[[column for column in SEASON_PANEL_COLUMNS if column in panel.columns]]


def truth_aligned(truth: dict, spec) -> pd.DataFrame:
    """Truth table aligned to spec.players (one row per player column, the order of Z_O).

    With one column per player (player_unit "player") a player has one column but several seasons of truth, so
    the truth is averaged over his seasons before the join.
    """
    ps = truth["ps"]
    if spec.player_unit == "player":
        ps = ps.groupby("player_id", as_index=False).mean(numeric_only=True).drop(columns="season", errors="ignore")
        return spec.players.merge(ps, on="player_id", how="left")
    return spec.players.merge(ps, on=["player_id", "season"], how="left")
