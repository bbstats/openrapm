"""The dashboard inputs of the season panel: tracking and hustle season totals, padded per 100 possessions, and the
matchup assignments.

Why (the owner, 2026-10-14: "let's look for hustle, tracking, synergy play types, and matchups").  The box score cannot
see a contested shot, a defended shot at the rim, a rebound chance or a screen assist.  `openrapm data dashboards`
fetches the dashboards (data/dashboards.py); this turns each season's cached totals into the panel's columns
(features.TRACKING_RATES, HUSTLE_RATES, MATCHUP_RATES).

Words used below
    count        a player's season total of one dashboard column, the regular season and playoffs added (the panel's
                 rows pool both phases)
    rate         100 x count / n, n the possessions of the row's side: his team's on an offence row, the opponent's on a
                 defence row (every column is built on both rows; the feature lists read the ones that mean something on
                 that side)
    pad k        how many possessions' worth of the typical value is blended in, by the method of moments on the
                 season's players: noise variance of one possession's worth of the rate (100 x the league rate, an
                 event count's Poisson noise) over the variance of the true rates between players, which is the observed
                 variance of the rates less the noise averaged over the players.  Distance and time with the ball are
                 measured, not counted, so they take the light fixed pad `FIXED_PAD`
    target       the typical value a rate is padded toward: the season's rate among the players of his possession bin
                 (box_rates.possession_bins), or the season's league rate when the bin is too small
    vs league    (padded rate - league rate) / league rate, the league rate being the season's counts over its
                 possessions, so 0.2 reads 20% above the league average.  Definitions moved between seasons (a player's
                 box-outs fell by four fifths from 2018 to 2026) and a ratio to the league follows a move in the level
                 that a difference would not
    over expected  the makes against a defender minus the makes the league's shooters had against the same kind of
                 shot (the dashboard's own expected percentage), per shot faced, padded toward 0 by `pad` shots

    assignment   who a player was matched with, from the matchups dashboard (`LeagueSeasonMatchups`: the possessions and
                 points of every attacker-defender pair).  The results of a matchup are outcome data and are not read;
                 only who guarded whom is.  A defender's assignment is the scoring of the attackers he guarded: each
                 attacker's points per matchup possession computed WITHOUT the rows against that defender (his own
                 results are not in it), padded `MATCHUP_PAD` possessions toward the league's, averaged by the
                 possessions the defender spent on each.  An attacker's assignment is the mirror, how stingy the
                 defenders were who guarded him, sign-flipped so that above 0 reads better defenders.  Both are
                 read against the league's rate, like the counts

Missing data: a season before the dashboard starts, or a player the dashboard does not list, gives NaN; the prior
treats it as missing (box_prior.py), not as an average player.

Folds: the tracking dashboards are season totals, not per game, so a within-season fold (commands/within_season.py)
cannot rebuild them from its fit games alone.  It scales each count by the fit games' share of the season's possessions
(`fit_share`), which keeps the rate right on average; the test games' share of every player's count still enters the
input (a known leak, notes/WITHIN_SEASON_LEDGER.md).  The hustle counts come game by game since 2026-10-15
(`dashboards.hustle_game_totals`, 2017 on; 2016, a third of the players, is left missing): summed over the games
handed in, the fit games in a fold, so they do not leak, and are not scaled.  A season-total input in the folds was
measured to leak badly: whole-season box-score inputs read z -31 on the with-or-without-you test (notes/DECISIONS.md,
"The leak control for season-total inputs").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..box_rates import K_MAX, MIN_BIN_PLAYERS, POSSESSION_BINS, bin_of, possession_bins, shrink
from ..features import DASHBOARD_RATES, HUSTLE_PLACEBO, HUSTLE_RATES, MATCHUP_RATES
from .dashboards import TRACKING_GAME_COLUMNS, game_box_totals, hustle_game_totals, read_dashboard

__all__ = ["Count", "OverExpected", "COUNTS", "OVER_EXPECTED", "FIXED_PAD", "MATCHUP_PAD", "moment_k",
           "season_counts", "assignment_rates", "matchup_rates", "dashboard_rates"]

MIN_POSSESSIONS = 200.0          # a player enters the pad k's estimate with at least this many possessions
FIXED_PAD = 50.0                 # the pad k, in possessions, of a measured (not counted) quantity
MATCHUP_PAD = 50.0               # matchup possessions of the league's scoring rate blended into an attacker's rate, and
                                 # again into a player's assignment


@dataclass(frozen=True)
class Count:
    """One panel column built from one dashboard column."""
    name: str                    # the panel column (features.TRACKING_RATES, HUSTLE_RATES)
    source: str                  # "tracking" or "hustle"
    kind: str                    # the dashboard folder (data/dashboards.py)
    id_column: str               # the column naming the player the row is about
    column: str                  # the endpoint's count
    measured: bool = False       # a distance or a time rather than an event count: padded by FIXED_PAD


@dataclass(frozen=True)
class OverExpected:
    """One panel column: the makes against a defender above the expected makes, per shot faced."""
    name: str
    kind: str
    made: str
    attempts: str
    expected_pct: str            # the dashboard's expected make probability for those shots
    pad: float                   # shots


COUNTS = (
    Count("drives_vs_league_avg", "tracking", "pt_drives", "PLAYER_ID", "DRIVES"),
    Count("drive_points_vs_league_avg", "tracking", "pt_drives", "PLAYER_ID", "DRIVE_PTS"),
    Count("potential_assists_vs_league_avg", "tracking", "pt_passing", "PLAYER_ID", "POTENTIAL_AST"),
    Count("passes_made_vs_league_avg", "tracking", "pt_passing", "PLAYER_ID", "PASSES_MADE"),
    Count("touches_vs_league_avg", "tracking", "pt_possessions", "PLAYER_ID", "TOUCHES"),
    Count("paint_touches_vs_league_avg", "tracking", "pt_possessions", "PLAYER_ID", "PAINT_TOUCHES"),
    Count("post_touches_vs_league_avg", "tracking", "pt_possessions", "PLAYER_ID", "POST_TOUCHES"),
    Count("elbow_touches_vs_league_avg", "tracking", "pt_possessions", "PLAYER_ID", "ELBOW_TOUCHES"),
    Count("minutes_with_ball_vs_league_avg", "tracking", "pt_possessions", "PLAYER_ID", "TIME_OF_POSS", True),
    Count("miles_on_offense_vs_league_avg", "tracking", "pt_speeddistance", "PLAYER_ID", "DIST_MILES_OFF", True),
    Count("off_reb_chances_vs_league_avg", "tracking", "pt_rebounding", "PLAYER_ID", "OREB_CHANCES"),
    Count("off_reb_contested_vs_league_avg", "tracking", "pt_rebounding", "PLAYER_ID", "OREB_CONTEST"),
    Count("def_reb_chances_vs_league_avg", "tracking", "pt_rebounding", "PLAYER_ID", "DREB_CHANCES"),
    Count("def_reb_contested_vs_league_avg", "tracking", "pt_rebounding", "PLAYER_ID", "DREB_CONTEST"),
    Count("rim_shots_defended_vs_league_avg", "tracking", "pt_defense", "PLAYER_ID", "DEF_RIM_FGA"),
    Count("twos_defended_vs_league_avg", "tracking", "defend_2_pointers", "CLOSE_DEF_PERSON_ID", "FG2A"),
    Count("threes_defended_vs_league_avg", "tracking", "defend_3_pointers", "CLOSE_DEF_PERSON_ID", "FG3A"),
    Count("miles_on_defense_vs_league_avg", "tracking", "pt_speeddistance", "PLAYER_ID", "DIST_MILES_DEF", True),
    Count("deflections_vs_league_avg", "hustle", "players", "PLAYER_ID", "DEFLECTIONS"),
    Count("contested_twos_vs_league_avg", "hustle", "players", "PLAYER_ID", "CONTESTED_SHOTS_2PT"),
    Count("contested_threes_vs_league_avg", "hustle", "players", "PLAYER_ID", "CONTESTED_SHOTS_3PT"),
    Count("charges_drawn_vs_league_avg", "hustle", "players", "PLAYER_ID", "CHARGES_DRAWN"),
    Count("loose_balls_recovered_vs_league_avg", "hustle", "players", "PLAYER_ID", "LOOSE_BALLS_RECOVERED"),
    Count("def_box_outs_vs_league_avg", "hustle", "players", "PLAYER_ID", "DEF_BOXOUTS"),
    Count("off_box_outs_vs_league_avg", "hustle", "players", "PLAYER_ID", "OFF_BOXOUTS"),
    Count("screen_assists_vs_league_avg", "hustle", "players", "PLAYER_ID", "SCREEN_ASSISTS"),
    Count("screen_assist_points_vs_league_avg", "hustle", "players", "PLAYER_ID", "SCREEN_AST_PTS"),
)
OVER_EXPECTED = (
    OverExpected("rim_makes_over_expected_per_shot", "defend_less_than_6ft", "FGM_LT_06", "FGA_LT_06", "NS_LT_06_PCT",
                 150.0),
    OverExpected("twos_makes_over_expected_per_shot", "defend_2_pointers", "FG2M", "FG2A", "NS_FG2_PCT", 250.0),
    OverExpected("threes_makes_over_expected_per_shot", "defend_3_pointers", "FG3M", "FG3A", "NS_FG3_PCT", 450.0),
)


# ------------------------------------------------------------------------------------------------ the totals
def _by_player(frame: pd.DataFrame, id_column: str, values: dict) -> pd.DataFrame:
    """`values` ({name: a column's numbers}) summed by player over the phases; a player is NaN where no row has a
    number.

    Example: a player with 40 deflections in the regular season and 6 in the playoffs reads 46."""
    ids = pd.to_numeric(frame[id_column], errors="coerce")
    table = pd.DataFrame({name: pd.to_numeric(numbers, errors="coerce") for name, numbers in values.items()})
    return table.groupby(ids.to_numpy()).sum(min_count=1)


def season_counts(cfg: dict, season: int, game_ids=None) -> pd.DataFrame:
    """One row per player listed on any dashboard in the season, a column per `COUNTS` and `OVER_EXPECTED` name (the
    attempts of an over-expected column in `<name>__shots`): his totals, regular season and playoffs added.  Empty
    when no dashboard has data yet.  With `game_ids`, the hustle counts are summed over those games from the per-game
    files when the season has them (`counts`)."""
    return counts(cfg, season, game_ids)[0]


def _per_game(spec: Count, hustle, tracking, games_named: bool) -> tuple:
    """(whether the per-game box scores decide this count, its column summed over the games or None).

    Hustle, when games are named, is game by game only: a season without per-game files (2016, a third of the
    players) is left missing rather than read from season totals, which a fold could not cut to its fit games.  The
    five tracking columns the per-game box score holds are summed over the games from 2014 on; the other tracking
    columns stay season totals.
    """
    if spec.source == "hustle" and games_named:
        return True, None if hustle is None else hustle[[spec.column]].rename(columns={spec.column: spec.name})
    if spec.source == "tracking" and tracking is not None and spec.column in TRACKING_GAME_COLUMNS:
        return True, tracking[[spec.column]].rename(columns={spec.column: spec.name})
    return False, None


def counts(cfg: dict, season: int, game_ids=None) -> tuple:
    """(`season_counts`' frame, the columns summed game by game over `game_ids`, the hustle games read)."""
    hustle, games_read = (None, set()) if game_ids is None else hustle_game_totals(cfg, season, game_ids)
    tracking, tracking_read = (None, set()) if game_ids is None else game_box_totals(cfg, "tracking", season, game_ids)
    games_read = games_read | tracking_read
    parts, cache, exact = [], {}, set()

    def table(source: str, kind: str):
        if (source, kind) not in cache:
            cache[(source, kind)] = read_dashboard(cfg, source, kind, season)
        return cache[(source, kind)]
    for spec in COUNTS:
        handled, part = _per_game(spec, hustle, tracking, game_ids is not None)
        if part is not None:
            parts.append(part)
            exact.add(spec.name)
        if handled:
            continue
        frame = table(spec.source, spec.kind)
        if frame is not None and spec.column in frame.columns:
            parts.append(_by_player(frame, spec.id_column, {spec.name: frame[spec.column]}))
    for spec in OVER_EXPECTED:
        frame = table("tracking", spec.kind)
        if frame is not None and spec.made in frame.columns:
            made, shots = (pd.to_numeric(frame[spec.made], errors="coerce"),
                           pd.to_numeric(frame[spec.attempts], errors="coerce"))
            expected = pd.to_numeric(frame[spec.expected_pct], errors="coerce") * shots
            parts.append(_by_player(frame, "CLOSE_DEF_PERSON_ID", {spec.name: made - expected,
                                                                     f"{spec.name}__shots": shots}))
    if not parts:
        return pd.DataFrame(), exact, games_read
    return pd.concat(parts, axis=1).astype(float), exact, games_read


# ------------------------------------------------------------------------------------------------ the padding
def moment_k(counts: np.ndarray, possessions: np.ndarray, min_possessions: float = MIN_POSSESSIONS) -> float:
    """The pad k, in possessions, of an event count's rate per 100, by the method of moments:

        league rate  mu = 100 x sum of counts / sum of possessions
        noise        one possession's worth of the rate varies by 100 mu (Poisson: variance = mean)
        tau^2        = Var_w(rate) - mean_w(100 mu / n),  the weights w the possessions n, and at least 1e-6
        k            = 100 mu / tau^2, at most box_rates.K_MAX

    Players under `min_possessions` are left out of the variances.  With fewer than 20 players left, or a league rate
    of 0, nothing can be told apart and k is K_MAX (a rate is padded all the way to its target).

    Example: rates with true spread 0.6 (variance 0.36) around a league rate of 1.6 per 100 give k = 160 / 0.36 = 444.
    """
    counts, possessions = np.asarray(counts, dtype=float), np.asarray(possessions, dtype=float)
    valid = np.isfinite(counts) & (possessions > 0)
    league = 100.0 * counts[valid].sum() / possessions[valid].sum() if valid.any() else 0.0
    keep = valid & (possessions >= min_possessions)
    if keep.sum() < 20 or league <= 0:
        return K_MAX
    n, rate = possessions[keep], 100.0 * counts[keep] / possessions[keep]
    centre = np.average(rate, weights=n)
    observed = np.average((rate - centre) ** 2, weights=n)
    noise = np.average(100.0 * league / n, weights=n)
    return float(min(100.0 * league / max(observed - noise, 1e-6), K_MAX))


def _bin_targets(counts: np.ndarray, possessions: np.ndarray, present: np.ndarray, league: float) -> np.ndarray:
    """Each player's target: his possession bin's rate among the players with data, the league rate for a bin of fewer
    than box_rates.MIN_BIN_PLAYERS players (and for every player when there are too few players for bins)."""
    target = np.full(len(counts), league)
    edges = possession_bins(possessions[present], possessions[present])
    if edges is None:
        return target
    bins = bin_of(edges, possessions)
    for bin_ in range(POSSESSION_BINS):
        inside = present & (bins == bin_)
        if inside.sum() >= MIN_BIN_PLAYERS and possessions[inside].sum() > 0:
            target[bins == bin_] = 100.0 * counts[inside].sum() / possessions[inside].sum()
    return target


def padded_relative(counts: np.ndarray, possessions: np.ndarray, measured: bool = False) -> np.ndarray:
    """(padded rate - league rate) / league rate for every player, NaN for a player without a count or possessions.

    The rate is 100 x count / possessions, padded by `moment_k` (`FIXED_PAD` for a measured quantity) toward his
    possession bin's rate (`_bin_targets`).  A league rate of 0 (a column no one has) gives NaN for everyone.

    Example: 8.4 deflections per 100, a bin rate of 7.0 and k = 0 against a league rate of 7.0 read 0.2.
    """
    counts, possessions = np.asarray(counts, dtype=float), np.asarray(possessions, dtype=float)
    present = np.isfinite(counts) & (possessions > 0)
    out = np.full(len(counts), np.nan)
    if not present.any():
        return out
    league = 100.0 * counts[present].sum() / possessions[present].sum()
    if league <= 0:
        return out
    k = FIXED_PAD if measured else moment_k(counts, possessions)
    safe = np.where(present, counts, 0.0)
    rate = np.where(present, 100.0 * safe / np.where(present, possessions, 1.0), 0.0)
    padded = shrink(rate, possessions, k, _bin_targets(safe, possessions, present, league))
    out[present] = (padded[present] - league) / league
    return out


# ------------------------------------------------------------------------------------------------ the assignments
def assignment_rates(pairs: pd.DataFrame, own: str, other: str) -> tuple:
    """(each `own` player's assignment, the league's points per matchup possession).

    `pairs` has a row per (defender, attacker) with `PARTIAL_POSS` and `PLAYER_PTS`; `own` and `other` name the id
    columns (the defender's and the attacker's, or the reverse).  For each pair the `other` player's rate is his totals
    without the rows against `own`, padded `MATCHUP_PAD` possessions toward the league's; the assignment is the mean of
    those rates over the pairs, weighted by possessions, padded the same way.  Not yet read against the league.

    Example: a league at 1.0 points per possession, an attacker with 100 points on 100 possessions in all, 20 of them on
    the 20 possessions he had against this defender: without those rows he is at 80 / 80 = 1.0, padded 1.0.
    """
    league = float(pairs.PLAYER_PTS.sum() / pairs.PARTIAL_POSS.sum())
    totals = pairs.groupby(other)[["PLAYER_PTS", "PARTIAL_POSS"]].sum()
    joined = pairs.join(totals, on=other, rsuffix="_all")
    without_points = joined.PLAYER_PTS_all - joined.PLAYER_PTS
    without_poss = joined.PARTIAL_POSS_all - joined.PARTIAL_POSS
    weight = joined.PARTIAL_POSS
    attacker_rate = (without_points + MATCHUP_PAD * league) / (without_poss + MATCHUP_PAD)
    weighted = (weight * attacker_rate).groupby(joined[own]).sum()
    time = weight.groupby(joined[own]).sum()
    return (weighted + MATCHUP_PAD * league) / (time + MATCHUP_PAD), league


def matchup_rates(cfg: dict, season: int, player_ids, fit_share: float | None = None) -> pd.DataFrame:
    """A frame of the `MATCHUP_RATES` columns, one row per player in `player_ids`: the scoring of the attackers he
    guarded and the stinginess of the defenders who guarded him, each against the league's rate (0.1 is 10% above).
    NaN for a player the dashboard does not list, and for every player before 2017 (the first season with it).

    `fit_share` scales the possessions and points, as in `dashboard_rates` (a fold fits on part of the season).

    Example: defenders who guarded attackers scoring 1.05 points per possession in a league at 1.00 read 0.05.
    """
    ids = np.asarray(player_ids)
    out = pd.DataFrame(np.nan, index=range(len(ids)), columns=MATCHUP_RATES)
    table = read_dashboard(cfg, "matchups", "season", season)
    if table is None:
        return out
    scale = 1.0 if fit_share is None else float(fit_share)
    pairs = (table.assign(PARTIAL_POSS=pd.to_numeric(table.PARTIAL_POSS, errors="coerce") * scale,
                          PLAYER_PTS=pd.to_numeric(table.PLAYER_PTS, errors="coerce") * scale)
             .groupby(["DEF_PLAYER_ID", "OFF_PLAYER_ID"], as_index=False)[["PARTIAL_POSS", "PLAYER_PTS"]].sum())
    if pairs.PARTIAL_POSS.sum() <= 0:
        return out
    guarded, league = assignment_rates(pairs, "DEF_PLAYER_ID", "OFF_PLAYER_ID")
    faced, _ = assignment_rates(pairs, "OFF_PLAYER_ID", "DEF_PLAYER_ID")
    out[MATCHUP_RATES[0]] = ((guarded.reindex(ids).to_numpy(float) - league) / league)
    out[MATCHUP_RATES[1]] = -((faced.reindex(ids).to_numpy(float) - league) / league)
    return out


def shuffled_rows(values: np.ndarray, season: int) -> np.ndarray:
    """The placebo (features.HUSTLE_PLACEBO): the rows that have any value moved among themselves by one permutation
    drawn with the seed (season, 6601), every column together; a row with no value stays empty.

    Example: three players with values and one without; the three swap profiles, the fourth stays missing.
    """
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    present = np.flatnonzero(np.isfinite(values).any(axis=1))
    order = np.random.default_rng([int(season), 6601]).permutation(len(present))
    out[present] = values[present[order]]
    return out


def dashboard_rates(cfg: dict, season: int, player_ids, possessions: dict, fit_share: float | None = None,
                    game_ids=None) -> dict:
    """{"O", "D"}: a frame per side with every `DASHBOARD_RATES` column, one row per player in `player_ids`; and
    "hustle_games", the games the per-game hustle counts were summed over (empty without them).

    `possessions` is {"O": offensive possessions, "D": defensive possessions}, each in the order of `player_ids`.
    `fit_share` is the share of the season a within-season fold fits on (None for the whole season): each season-total
    count is multiplied by it (the module docstring).  `game_ids` are the games the rows are built from: the hustle
    counts are summed over them where the season has per-game files, and not scaled.  A column no dashboard has for
    the season is NaN throughout.

    Example: a 2026 player with 12.3 contested twos per 100 possessions in a league at 10.0 reads 0.23 in
    `contested_twos_vs_league_avg` (before padding); a 2012 player reads NaN there.
    """
    totals, exact, games_read = counts(cfg, season, game_ids)
    scale = 1.0 if fit_share is None else float(fit_share)
    ids = np.asarray(player_ids)
    assignments = matchup_rates(cfg, season, ids, fit_share)
    out = {}
    for side in ("O", "D"):
        n = np.asarray(possessions[side], dtype=float)
        frame = pd.DataFrame(index=range(len(ids)), columns=DASHBOARD_RATES, dtype=float)
        for spec in COUNTS:
            if spec.name in totals.columns:
                factor = 1.0 if spec.name in exact else scale
                frame[spec.name] = padded_relative(factor * totals[spec.name].reindex(ids).to_numpy(float), n,
                                                   spec.measured)
        for spec in OVER_EXPECTED:
            if spec.name in totals.columns:
                made = totals[spec.name].reindex(ids).to_numpy(float)
                shots = totals[f"{spec.name}__shots"].reindex(ids).to_numpy(float)
                frame[spec.name] = scale * made / (scale * shots + spec.pad)
        frame[MATCHUP_RATES] = assignments.to_numpy()
        frame[HUSTLE_PLACEBO] = shuffled_rows(frame[HUSTLE_RATES].to_numpy(float), season)
        out[side] = frame[DASHBOARD_RATES]
    out["hustle_games"] = games_read
    return out
