"""League dashboards from stats.nba.com that the ratings do not read yet: hustle, tracking, play types and matchups.

Why (the owner, 2026-10-14: "let's look for hustle, tracking, synergy play types, and matchups").  The box score cannot
see a deflection, a defended shot at the rim, the share of a player's possessions that end in an isolation, or whom
a defender guarded, and the season panel's defensive inputs are the weakest part of the box-score prior (steals are
overrated, big men underrated; notes/DECISIONS.md, "The checkpoint").  Each of these is a season total for each player
and so a this-season fact about him, which the single-season rule allows.

Words used below
    pull       one endpoint with fixed arguments, and the file it writes for each season and phase
    source     hustle (`LeagueHustleStatsPlayer`: deflections, contested shots, charges drawn, loose balls recovered,
               screen assists, box outs), tracking (`LeagueDashPtStats` by measure and `LeagueDashPtDefend` by shot
               zone: defended shots at the rim, rebound chances, drives, passes, touches, speed and distance),
               synergy (`SynergyPlayTypes`: possessions, points and shooting by play type, offence and defence),
               matchups (`LeagueSeasonMatchups`: each offensive player a defender guarded, with possessions and points)
    probe      a few seasons of every pull, to learn which fields exist, from which season, and how many of the rated
               players each covers, before anything is fetched in full
    coverage   of the players the season panel rates in a season, the share a file holds, and the share of their
               possessions

Cached: one file per pull, season and phase at paths.raw/dashboards/<source>/<kind>/<season>_<phase>.parquet, the
endpoint's first table as it answers.  A response with no rows is cached too (the season has no data for the pull); a
failed request is not, and is reported.  With OPENRAPM_OFFLINE=1 a missing file stops the pull.

Example: `pulls()` holds 34 pulls; the hustle pull for 2026 regular season is written to
data/raw/dashboards/hustle/players/2026_RS.parquet.

Per game (2026-10-15).  A season total includes the games a within-season fold holds back, and the with-or-without-you
test scores exactly those games: a fold whose prior read whole-season inputs scored z -31 better than an honest one
(notes/DECISIONS.md, "The leak control for season-total inputs").  So the hustle numbers are also fetched game by game
(`BoxScoreHustleV2`, one request a game, 2016-17 on), which a fold can sum over its fit games alone:
data/raw/dashboards/hustle/games/<season>/<game_id>.parquet, the endpoint's player table as it answers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import (
    SEASON_TYPES,
    TIMEOUT,
    OfflineFetch,
    _retry,
    game_table,
    load_gamelog,
    raw_dir,
    refuse_if_offline,
    season_of_game,
    season_str,
)

SOURCES = ("hustle", "tracking", "synergy", "matchups")
MEASURES = ("Defense", "Rebounding", "Drives", "Passing", "Possessions", "SpeedDistance")        # LeagueDashPtStats
DEFENDED_ZONES = ("Overall", "3 Pointers", "2 Pointers", "Less Than 6Ft")                       # LeagueDashPtDefend
PLAY_TYPES = ("Isolation", "Transition", "PRBallHandler", "PRRollman", "Postup", "Spotup", "Handoff", "Cut",
              "OffScreen", "OffRebound", "Misc")
GROUPINGS = ("offensive", "defensive")
BREAK_SEASON = 2018        # tracking's provider changed in 2017-18: its fields before and after are not one scale


@dataclass(frozen=True)
class Pull:
    """One endpoint with its fixed arguments; `first_guess` is the first season it is believed to have data."""
    source: str
    kind: str                      # the folder under the source
    endpoint: str                  # the nba_api class
    arguments: dict = field(default_factory=dict)
    id_column: str = "PLAYER_ID"   # the column holding the player the row is about
    first_guess: int = 2014
    season_type_argument: str = "season_type_all_star"


def pulls(sources=SOURCES) -> list[Pull]:
    """Every pull of the sources named, in a fixed order."""
    every = [Pull("hustle", "players", "LeagueHustleStatsPlayer", dict(per_mode_time="Totals"), first_guess=2016)]
    every += [Pull("tracking", f"pt_{measure.lower()}", "LeagueDashPtStats",
                   dict(player_or_team="Player", pt_measure_type=measure, per_mode_simple="Totals"), first_guess=2014)
              for measure in MEASURES]
    every += [Pull("tracking", f"defend_{zone.lower().replace(' ', '_')}", "LeagueDashPtDefend",
                   dict(defense_category=zone, per_mode_simple="Totals"), "CLOSE_DEF_PERSON_ID", 2014)
              for zone in DEFENDED_ZONES]
    every += [Pull("synergy", f"{grouping}_{play.lower()}", "SynergyPlayTypes",
                   dict(play_type_nullable=play, type_grouping_nullable=grouping, player_or_team_abbreviation="P",
                        per_mode_simple="Totals"), first_guess=2016)
              for grouping in GROUPINGS for play in PLAY_TYPES]
    every.append(Pull("matchups", "season", "LeagueSeasonMatchups", dict(per_mode_simple="Totals"), "DEF_PLAYER_ID",
                      2018, "season_type_playoffs"))
    return [pull for pull in every if pull.source in sources]


def dashboard_path(cfg: dict, pull: Pull, season: int, phase: str) -> Path:
    return raw_dir(cfg) / "dashboards" / pull.source / pull.kind / f"{season}_{phase}.parquet"


def read_dashboard(cfg: dict, source: str, kind: str, season: int, phases=("RS", "PO")) -> pd.DataFrame | None:
    """One cached dashboard's tables for a season, the phases stacked (a `phase` column), read and never fetched.

    None when the season is before the pull's first season with data (`Pull.first_guess`: the table does not exist
    yet); a missing file from that season on raises, naming the command that fetches it, so a panel is never built
    quietly without a season the dashboards cover.  A cached table with no rows counts as no table for its phase.
    """
    pull = next(p for p in pulls((source,)) if p.kind == kind)
    if int(season) < pull.first_guess:
        return None
    parts = []
    for phase in phases:
        path = dashboard_path(cfg, pull, int(season), phase)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist; `openrapm data dashboards` fetches the {source} {kind} "
                                    f"dashboards")
        frame = pd.read_parquet(path)
        if len(frame):
            parts.append(frame.assign(phase=phase))
    return pd.concat(parts, ignore_index=True) if parts else None


def _endpoint_class(name: str):
    """The nba_api class called `name` (a function of its own so a test can answer for it)."""
    from nba_api.stats import endpoints
    return getattr(endpoints, name)


def fetch_dashboard(pull: Pull, season: int, phase: str, cfg: dict, force: bool = False, tries: int = 4) -> tuple:
    """(the pull's first table for the season and phase, whether it was read from the cache), cached at
    `dashboard_path`.  A table with no rows is cached as it is."""
    path = dashboard_path(cfg, pull, season, phase)
    if path.exists() and not force:
        return pd.read_parquet(path), True
    refuse_if_offline(path, f"the {pull.source} {pull.kind} dashboard {season} {phase}")

    def ask():
        endpoint = _endpoint_class(pull.endpoint)(**pull.arguments, season=season_str(season),
                                                  **{pull.season_type_argument: SEASON_TYPES[phase]}, timeout=TIMEOUT)
        return endpoint.get_data_frames()[0]
    frame = _retry(ask, tries=tries, base_sleep=3.0, what=f"{pull.source} {pull.kind} {season} {phase}")
    frame = frame.astype({column: str for column in frame.columns if frame[column].dtype == object})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame, False


def pull_all(cfg: dict, wanted: list, phases=("RS", "PO"), delay: float = 0.6, force: bool = False,
             tries: int = 4) -> list[dict]:
    """Every (pull, season) of `wanted` ([(Pull, [seasons])]) for every phase, one request at a time, `delay` seconds
    apart (none after a cached file): one record each, {pull, season, phase, status, rows, error, frame}.

    `status` is "cached", "fetched", "empty" (the answer had no rows) or "failed" (after `tries` tries; not cached).
    OfflineFetch is not forgiven: offline, a missing file is a copying mistake.
    """
    records, started = [], time.time()
    total = sum(len(seasons) for _, seasons in wanted) * len(phases)
    for pull, seasons in wanted:
        for season in seasons:
            for phase in phases:
                record = dict(pull=pull, season=int(season), phase=phase, rows=0, error="", frame=None)
                try:
                    frame, cached = fetch_dashboard(pull, season, phase, cfg, force=force, tries=tries)
                    record.update(frame=frame, rows=len(frame),
                                  status="cached" if cached else ("fetched" if len(frame) else "empty"))
                    if not cached:
                        time.sleep(delay)
                except OfflineFetch:
                    raise
                except RuntimeError as error:
                    record.update(status="failed", error=str(error)[:300])
                records.append(record)
                if len(records) % 25 == 0 or len(records) == total:
                    print(f"  dashboards: {len(records)} of {total} ({time.time() - started:.0f}s)", flush=True)
    return records


def probe_seasons(pull: Pull, last_season: int) -> list[int]:
    """The seasons a probe asks of a pull: the one before it should start (expected empty), the first, the tracking
    break's season and the latest."""
    return sorted({pull.first_guess - 1, pull.first_guess, BREAK_SEASON, int(last_season)})


def coverage(records: list[dict], panel: pd.DataFrame) -> pd.DataFrame:
    """One row per record: source, kind, season, phase, status, rows, columns, and how much of the season's rated
    players the file holds (the module docstring).  `panel` is the season panel (player_id, season, side,
    possessions); a rated player is one with offensive possessions.

    Example: a hustle file naming 520 of 560 rated players who played 98% of the possessions reads share_players
    0.93, share_possessions 0.98.
    """
    rated = panel[(panel.side == "O") & (panel.possessions > 0)]
    rows = []
    for record in records:
        pull, frame = record["pull"], record["frame"]
        row = dict(source=pull.source, kind=pull.kind, season=record["season"], phase=record["phase"],
                   status=record["status"], rows=record["rows"], n_columns=0 if frame is None else frame.shape[1],
                   rated_players=np.nan, in_file=np.nan, share_players=np.nan, share_possessions=np.nan,
                   error=record["error"])
        if frame is not None and len(frame) and pull.id_column in frame.columns:
            season_rows = rated[rated.season == record["season"]]
            ids = pd.to_numeric(frame[pull.id_column], errors="coerce").dropna().astype(np.int64).unique()
            held = season_rows.player_id.isin(ids)
            row.update(rated_players=len(season_rows), in_file=int(held.sum()),
                       share_players=float(held.mean()) if len(season_rows) else np.nan,
                       share_possessions=float(season_rows.possessions[held].sum() / season_rows.possessions.sum())
                       if len(season_rows) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def fields(records: list[dict]) -> pd.DataFrame:
    """One row per pull: the names of its fields, from the first of its records that has rows."""
    seen: dict = {}
    for record in records:
        frame, pull = record["frame"], record["pull"]
        if frame is not None and len(frame) and (pull.source, pull.kind) not in seen:
            seen[(pull.source, pull.kind)] = dict(source=pull.source, kind=pull.kind, season=record["season"],
                                                  fields=", ".join(map(str, frame.columns)))
    return pd.DataFrame(list(seen.values()), columns=["source", "kind", "season", "fields"])


def panel_rows(cfg: dict) -> pd.DataFrame:
    """The season panel's keys and possessions (what `coverage` reads), through the panel module's own reader."""
    from .panel import panel_keys
    return panel_keys(cfg)


# ------------------------------------------------------------------------------------------ hustle, game by game
HUSTLE_GAMES_FIRST = 2017       # BoxScoreHustleV2 holds every player from 2016-17; a 2015-16 game answers two rows


def hustle_game_path(cfg: dict, game_id: str) -> Path:
    return raw_dir(cfg) / "dashboards" / "hustle" / "games" / str(season_of_game(game_id)) / f"{game_id}.parquet"


def fetch_hustle_game(game_id: str, cfg: dict, force: bool = False, tries: int = 4) -> tuple:
    """(one game's hustle numbers, one row per player, whether it was read from the cache), cached at
    `hustle_game_path`.  A game answered with no rows is cached as it is."""
    path = hustle_game_path(cfg, game_id)
    if path.exists() and not force:
        return pd.read_parquet(path), True
    refuse_if_offline(path, f"the hustle box score of {game_id}")

    def ask():
        return _endpoint_class("BoxScoreHustleV2")(game_id=game_id, timeout=TIMEOUT).get_data_frames()[0]
    frame = _retry(ask, tries=tries, base_sleep=3.0, what=f"hustle {game_id}")
    frame = frame.astype({column: str for column in frame.columns if frame[column].dtype == object})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame, False


def season_game_ids(cfg: dict, season: int, phases=("RS", "PO")) -> list[str]:
    """Every game of the season's phases, from the cached game logs, in game-id order."""
    return sorted(str(game) for phase in phases for game in game_table(load_gamelog(season, phase, cfg))["game_id"])


def fetch_hustle_games(cfg: dict, seasons, phases=("RS", "PO"), delay: float = 0.6, log_every: int = 250) -> dict:
    """Every game's hustle numbers for the seasons and phases, one request at a time, `delay` seconds apart (none
    after a cached file).  Returns {"games", "fetched", "empty", "failed": [(game_id, error)]}; one bad game does not
    stop the run, and a rerun fetches only what is missing.  OfflineFetch is not forgiven."""
    games = [game for season in seasons for game in season_game_ids(cfg, int(season), phases)]
    out, started = dict(games=len(games), fetched=0, empty=0, failed=[]), time.time()
    for done, game_id in enumerate(games, start=1):
        try:
            frame, cached = fetch_hustle_game(game_id, cfg)
            if not cached:
                out["fetched"] += 1
                out["empty"] += int(len(frame) == 0)
                time.sleep(delay)
        except OfflineFetch:
            raise
        except RuntimeError as error:
            out["failed"].append((game_id, str(error)[:300]))
        if log_every and (done % log_every == 0 or done == len(games)):
            print(f"  hustle games: {done} of {len(games)}, {out['fetched']} fetched, {len(out['failed'])} failed "
                  f"({time.time() - started:.0f}s)", flush=True)
    return out
