"""The stint files: every game of a season parsed into stints, with its diagnostics, names and shots.

The parsing of one game lives in stint_parser.py (what a possession and a stint are, and how the ten players
on the court are found).  This module runs it over a season and phase (RS regular season, PO playoffs) and
writes four files into paths.stints:

    {season}_{phase}.parquet         the stints: design.STINT_COLUMNS, start_clock, game_date, every
                                     design.POSS_COUNTERS column per side (_h, _a), the shooter slot counters
                                     (design.SLOT_COUNTERS per side) and the game's half
    {season}_{phase}_diag.parquet    one row per game: the parser's diagnostics (unresolved substitutions,
                                     forced switches, invalid possessions, ...), possession and point totals
                                     with the box score's points beside them, or `error` for a game that failed
    {season}_{phase}_shots.parquet   one row per game and shooter (stint_parser.SHOT_TABLE): two- and
                                     three-point attempts, makes, and the league's make probability summed
                                     over his attempts, with the game's half
    {season}_{phase}_names.parquet   player id -> his most common name in that season's play-by-play

Words used below
    half      the cross-fitting half of a game, A or B: regular-season games alternate A, B, A, ... in date
              order, playoff games within each series, so both halves get the same mix of games
    schema    STINT_SCHEMA, the version of the stint columns, written into the diag file and checked on load

Example: the 2003 playoffs are 88 games.  They give 2,653 stints (about 30 a game), 1,643 game-and-shooter
rows and the play-by-play names of 196 players.
"""
from __future__ import annotations

import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from ..design import AWAY_SLOTS, HOME_SLOTS, SEASON_PHASES
from . import box_scores, shot_curves
from .config import resolve
from .ingest import OfflineFetch, fetch_boxscore, fetch_pbp, fetch_period_box, game_table, load_gamelog, pbpstats_games
from .stint_parser import SHOT_TABLE, GameParser

# Bumped whenever the stint schema changes.  It is written into the per-season diag frame and checked on
# load, because the cache is keyed on path existence alone: without this, adding a column leaves every stale
# file loading happily with the new columns silently absent.
#   1  the original schema: poss/pts/margin/lineups
#   2  per-possession counters (design.POSS_COUNTERS), for the luck-adjusted target
#   3  xftm / xftm_tech: expected free-throw makes, the first luck-adjusted term
#   4  per-shooter attempt counters by lineup slot (design.SLOT_COUNTERS), the first attempt's realised
#      points and rebound chance (pts1, chance1), the cross-fitting half per game, and the per-game
#      shooter table -- the inputs of the shooter-level expected points (data/shots.py)
STINT_SCHEMA = 4


class StaleStintCache(RuntimeError):
    """A cached stint file predates the current schema and must be rebuilt."""


def build_game(game_id: str, home_id: int, away_id: int, cfg, gt_rule=None, ft_rates=None, curve=None):
    """One game: its stints, its diagnostics (a dict), its play-by-play names and its shooter table."""
    pbp = fetch_pbp(game_id, cfg)
    box = fetch_boxscore(game_id, cfg)
    parser = GameParser(pbp, box, home_id, away_id, game_id, period_box_fetcher=partial(fetch_period_box, cfg=cfg),
                        gt_rule=gt_rule, ft_rates=ft_rates, curve=curve)
    poss = parser.possessions()
    stints = parser.stints(poss)
    shots = pd.DataFrame([dict(player_id=pid, **totals) for pid, totals in parser.shots.items()],
                         columns=["player_id", *SHOT_TABLE])
    shots["game_id"] = game_id
    names = pd.DataFrame({"player_id": list(parser.day_names), "pbp_name": list(parser.day_names.values())})
    return stints, game_diagnostics(parser, poss, stints), names, shots


def game_diagnostics(parser: GameParser, poss: pd.DataFrame, stints: pd.DataFrame) -> dict:
    """The parser's counts, then possession and point totals with the box score's points beside them."""
    diag = dict(parser.diag)
    diag.update(game_id=parser.game_id, n_poss=len(poss), **_possession_totals(poss, parser.home, parser.away),
                box_pts_home=parser.box_pts.get(parser.home, np.nan),
                box_pts_away=parser.box_pts.get(parser.away, np.nan), n_stints=len(stints))
    return diag


def _possession_totals(poss: pd.DataFrame, home: int, away: int) -> dict:
    if not len(poss):
        return dict(n_poss_home=0, n_poss_away=0, n_valid_poss=0, pts_home=0.0, pts_away=0.0)
    home_rows, away_rows = poss["offense"] == home, poss["offense"] == away
    return dict(n_poss_home=int(home_rows.sum()), n_poss_away=int(away_rows.sum()),
                n_valid_poss=int(poss["valid"].sum()),
                pts_home=float(poss.loc[home_rows, "points"].sum()),
                pts_away=float(poss.loc[away_rows, "points"].sum()))


def assign_halves(games: pd.DataFrame) -> pd.Series:
    """The cross-fitting half of every game, by game_id: regular-season games alternate A/B in
    chronological order within a season, playoff games within a series.  The SAME rule as
    design._order_games, applied at stint-build time so the shooter tables and the design agree."""
    ordered = games.copy()
    ordered["_ph"] = (ordered["phase"] == "PO").astype(int)
    ordered = ordered.sort_values(["season", "_ph", "game_date", "game_id"]).reset_index(drop=True)
    regular = (ordered["phase"] == "RS").to_numpy()
    half = np.empty(len(ordered), dtype=object)
    if regular.any():
        position = ordered[regular].groupby("season").cumcount().to_numpy()
        half[regular] = np.where(position % 2 == 0, "A", "B")
    if (~regular).any():
        position = ordered[~regular].groupby(["season", "series_id"]).cumcount().to_numpy()
        half[~regular] = np.where(position % 2 == 0, "A", "B")
    return pd.Series(half, index=ordered["game_id"].to_numpy())


def build_season(season: int, phase: str, cfg, force=False, verbose=True):
    """Stints for one season and phase -> paths.stints/{season}_{phase}.parquet, plus the _diag, _shots and
    _names files.  A cached pair (stints and diag) is returned as it is unless `force`."""
    out_dir = resolve(cfg, "stints")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{season}_{phase}.parquet"
    diag_path = out_dir / f"{season}_{phase}_diag.parquet"
    if path.exists() and diag_path.exists() and not force:
        return _load_cached(path, diag_path, season, phase)
    gamelog = load_gamelog(season, phase, cfg)
    games = game_table(gamelog)
    gt_rule = cfg.get("garbage_time", {})
    # the shooters' season free-throw totals, for the leave-one-game-out expected makes
    season_box = box_scores.box_from_gamelog(gamelog)
    ft_rates = (box_scores.ft_totals(season_box), *box_scores.ft_padding(season_box))
    # the league's make probability by distance this season (the regular season's curve, for the playoffs
    # too).  `force` reaches the curve as well: a curve cached from an incomplete scrape used to survive
    # every later `02_stints.py --force`, because only the stints were rebuilt.
    curve = shot_curves.curve_for(season, cfg, force=force, verbose=verbose)
    neutral_seasons = set(cfg.get("neutral_site_seasons_po", []))
    parts, diags, name_parts, shot_parts = [], [], [], []
    t0 = time.time()
    for k, game in enumerate(games.itertuples(index=False)):
        try:
            game_stints, diag, names, shots = build_game(game.game_id, game.home_team_id, game.away_team_id, cfg,
                                                         gt_rule, ft_rates, curve)
        except OfflineFetch:
            raise                    # offline, a missing raw file is a copying mistake, not a game to drop
        except Exception as error:
            diags.append(dict(game_id=game.game_id, error=repr(error)[:300]))
            continue
        name_parts.append(names)
        shot_parts.append(shots)
        if len(game_stints):
            parts.append(_label_stints(game_stints, game, season, phase, neutral_seasons))
        diag["home_team_id"] = game.home_team_id
        diag["away_team_id"] = game.away_team_id
        diags.append(diag)
        if verbose and (k + 1) % 100 == 0:
            print(f"  {season} {phase}: {k + 1}/{len(games)} games  {time.time() - t0:.0f}s")
    stints = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    # the cross-fitting half of every game that produced stints, stored so the design and the shooter
    # tables use the same one
    halves = None
    if len(stints):
        halves = assign_halves(stints[["game_id", "season", "phase", "game_date", "series_id"]]
                               .drop_duplicates("game_id"))
        stints["half"] = stints["game_id"].map(halves)
    diag = pd.DataFrame(diags)
    diag["seconds_total"] = time.time() - t0
    diag["stint_schema"] = STINT_SCHEMA
    stints.to_parquet(path, index=False)
    diag.to_parquet(diag_path, index=False)
    _write_shots(shot_parts, halves, season, phase, out_dir)
    _write_names(name_parts, season, phase, out_dir)
    return stints, diag


def _load_cached(path, diag_path, season: int, phase: str):
    """A cached season, refused when it was built at an older schema.  A cache written before the counters
    existed would still load, and its missing columns would become NaN the moment it met a fresh one, flowing
    silently into the target and the weights."""
    stints, diag = pd.read_parquet(path), pd.read_parquet(diag_path)
    have = int(diag["stint_schema"].iloc[0]) if "stint_schema" in diag.columns and len(diag) else 0
    if have >= STINT_SCHEMA:
        return stints, diag
    raise StaleStintCache(
        f"{path.name} was built at schema {have}, this code needs {STINT_SCHEMA}. "
        f"Rebuild with: python scripts/02_stints.py {season} {season} {phase} --force")


def _label_stints(game_stints: pd.DataFrame, game, season: int, phase: str, neutral_seasons: set) -> pd.DataFrame:
    """The game's columns on its stints.  A playoff series is "season-lower team id-higher team id"; a game is
    neutral in a neutral-site postseason (the 2020 bubble) or when the game log lists no home team."""
    low, high = min(game.home_team_id, game.away_team_id), max(game.home_team_id, game.away_team_id)
    neutral = (phase == "PO" and season in neutral_seasons) or bool(getattr(game, "neutral", False))
    game_stints["game_id"] = game.game_id
    game_stints["season"] = season
    game_stints["phase"] = phase
    game_stints["game_date"] = game.game_date
    game_stints["series_id"] = f"{season}-{low}-{high}" if phase == "PO" else ""
    game_stints["neutral"] = bool(neutral)
    return game_stints


def _write_shots(shot_parts: list, halves, season: int, phase: str, out_dir) -> None:
    if shot_parts:
        shots = pd.concat(shot_parts, ignore_index=True)
    else:
        shots = pd.DataFrame(columns=["player_id", *SHOT_TABLE, "game_id"])
    shots["season"] = season
    shots["phase"] = phase
    shots["half"] = shots["game_id"].map(halves) if halves is not None else None
    shots.to_parquet(out_dir / f"{season}_{phase}_shots.parquet", index=False)


def _write_names(name_parts: list, season: int, phase: str, out_dir) -> None:
    """Each player's most common play-by-play name over the season's games."""
    if not name_parts:
        return
    names = pd.concat(name_parts, ignore_index=True)
    names = names.groupby("player_id")["pbp_name"].agg(lambda s: s.value_counts().index[0]).reset_index()
    names["season"] = season
    names.to_parquet(out_dir / f"{season}_{phase}_names.parquet", index=False)


def season_names(season: int, phase: str, cfg) -> pd.DataFrame:
    """player_id, season, pbp_name: the player's name as written in that season's play-by-play."""
    p = resolve(cfg, "stints") / f"{season}_{phase}_names.parquet"
    if not p.exists():
        build_season(season, phase, cfg)
    return pd.read_parquet(p)


def season_diagnostics(season: int, phase: str, cfg) -> dict:
    """Per-season summary: lineup validity, possessions per game vs pbpstats, player-poss identity, points check."""
    stints, diag = build_season(season, phase, cfg)
    out = dict(season=season, phase=phase, n_games=int(diag["game_id"].nunique()), n_stints=len(stints))
    if "error" in diag.columns:
        out["games_failed"] = int(diag["error"].notna().sum())
    parsed = diag[diag["error"].isna()] if "error" in diag.columns else diag
    out["stints_per_game"] = len(stints) / max(out["n_games"], 1)
    out["poss_per_team_game"] = float((parsed["n_poss_home"] + parsed["n_poss_away"]).mean() / 2)
    out["valid_poss_frac"] = float(parsed["n_valid_poss"].sum() / parsed["n_poss"].sum())
    out["unresolved_subs_per_game"] = float(parsed["unresolved_subs"].mean())
    out["sub_out_not_on_floor_per_game"] = float(parsed["sub_out_not_on_floor"].mean())
    out["forced_switches_per_game"] = float(parsed["forced_switches"].mean())
    out["periods_from_box_frac"] = float(parsed["periods_from_box"].sum() / parsed["n_periods"].sum())
    out["periods_starters_failed"] = int(parsed["periods_starters_failed"].sum())
    out["points_match_frac"] = float(((parsed["pts_home"] == parsed["box_pts_home"])
                                      & (parsed["pts_away"] == parsed["box_pts_away"])).mean())
    out.update(_versus_pbpstats(parsed, season, phase, cfg))
    # sum of player possessions = 5 x team possessions holds by construction; check five distinct players
    if len(stints):
        home, away = stints[HOME_SLOTS].to_numpy(), stints[AWAY_SLOTS].to_numpy()
        out["five_distinct_frac"] = float(np.mean([(len(set(h)) == 5) and (len(set(a)) == 5)
                                                   for h, a in zip(home, away)]))
    return out


def _versus_pbpstats(parsed: pd.DataFrame, season: int, phase: str, cfg) -> dict:
    """Possessions per game against api.pbpstats.com's count (an empty file before 2000-01).  A failure here
    is reported in the summary, not raised, except offline."""
    out = {}
    try:
        pbpstats = pbpstats_games(season, phase, cfg)
        if len(pbpstats):
            matched = parsed.merge(pbpstats[["GameId", "HomePossessions", "AwayPossessions"]],
                                   left_on="game_id", right_on="GameId")
            diff = ((matched["n_poss_home"] - matched["HomePossessions"]).abs()
                    + (matched["n_poss_away"] - matched["AwayPossessions"]).abs())
            out["pbpstats_games_matched"] = len(matched)
            out["poss_abs_diff_per_game_vs_pbpstats"] = float(diff.mean() / 2)
            out["poss_within_2_frac"] = float((diff <= 4).mean())
    except OfflineFetch:
        raise
    except Exception as error:
        out["pbpstats_error"] = repr(error)[:100]
    return out


def load_season_stints(season: int, cfg, columns, phases=SEASON_PHASES, with_teams: bool = True) -> pd.DataFrame:
    """A season's stints, regular season then playoffs, with `columns` (and, `with_teams`, each game's teams).

    The stints carry no team ids; `with_teams` joins home_team_id and away_team_id from the per-game diagnostics
    written beside them.  A phase with no file (no playoffs yet) is skipped; no stints at all is an error.
    """
    folder = resolve(cfg, "stints")
    frames = []
    for phase in phases:
        path = folder / f"{season}_{phase}.parquet"
        if not path.exists():
            continue
        stints = pd.read_parquet(path, columns=list(columns))
        if with_teams:
            stints = _with_teams(stints, folder / f"{season}_{phase}_diag.parquet", path.name)
        frames.append(stints)
    if not frames:
        raise FileNotFoundError(f"no stints on disk for {season} in {folder}")
    return pd.concat(frames, ignore_index=True)


def _with_teams(stints: pd.DataFrame, diagnostics: Path, name: str) -> pd.DataFrame:
    """The stints with home_team_id and away_team_id joined on game_id from the diagnostics."""
    teams = pd.read_parquet(diagnostics, columns=["game_id", "home_team_id", "away_team_id"]).drop_duplicates("game_id")
    stints = stints.merge(teams, on="game_id", how="left", sort=False)
    if stints.home_team_id.isna().any():
        raise ValueError(f"{name}: a stint's game has no row in the diagnostics; rebuild the stints")
    return stints
