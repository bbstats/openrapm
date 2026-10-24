"""Raw data pulls, cached per game / per season as parquet. Never re-download, but for the season in progress.

Sources (stats.nba.com via nba_api):
  - LeagueGameLog (player rows), one call per season and phase -> game list + per-game box counts
  - PlayByPlayV3 per game (V2 is dead: the API returns empty JSON)
  - BoxScoreTraditionalV3 per game (roster, starters, minutes) and, on demand, per period
    (range_type=2) to resolve period starters the play-by-play cannot identify
  - api.pbpstats.com get-games per season (possessions per team-game) as a cross-check, 2000-01+

Layout under cfg paths.raw:
  gamelog/{season}_{phase}.parquet
  pbp/{season}/{game_id}.parquet
  box/{season}/{game_id}.parquet
  periodbox/{season}/{game_id}_{period}.parquet
  pbpstats_games/{season}_{phase}.parquet

The season in progress (2026-10-23; the owner: "let's pull data for the current season so far. we also need to pull
data regularly").  Its game log grows every night, so `openrapm data ingest --refresh` fetches the log again instead of
reading the cache, and keeps only the games dated before today in US Eastern time (`eastern_today`): a game of today may
still be in play, and its play-by-play, once cached, would never be fetched again; until 4 AM Eastern, today is still
the night before.  Example: refreshed at 9:45 PM on 2026-10-23, or at 12:45 AM on 2026-10-24, the 2027 log holds the
games of 2026-10-21 and 2026-10-22; the 6 AM refresh of 2026-10-24 adds 2026-10-23's.

Offline: with the environment variable OPENRAPM_OFFLINE=1, every fetch raises `OfflineFetch` (a
RuntimeError naming the file it would have written) instead of reaching the network, so a rebuild from
a copied raw folder fails loudly on the first missing file rather than quietly downloading it.  The loops
that forgive one failed game (here, in stints.py and in shot_curves.py) let `OfflineFetch` through: offline,
a missing file is a copying mistake, not a game to skip.
"""
from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

from .config import resolve

warnings.filterwarnings("ignore", category=DeprecationWarning)

SEASON_TYPES = {"RS": "Regular Season", "PO": "Playoffs"}
GAME_PREFIX = {"RS": "002", "PO": "004"}
TIMEOUT = 60
HEADERS_PBPSTATS = {"User-Agent": "Mozilla/5.0"}
OFFLINE_ENV = "OPENRAPM_OFFLINE"


class OfflineFetch(RuntimeError):
    """A fetch was needed while OPENRAPM_OFFLINE=1."""


def offline() -> bool:
    """True when OPENRAPM_OFFLINE asks that nothing be fetched from the network."""
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in ("1", "true", "yes")


def refuse_if_offline(path, what: str) -> None:
    """Raise instead of fetching `what` (to be cached at `path`) when the offline switch is on."""
    if offline():
        raise OfflineFetch(
            f"{OFFLINE_ENV}=1 and {what} would have to be fetched from the network: {path} is missing "
            f"(or a refetch was forced).  Copy the file in, or unset {OFFLINE_ENV} to download it.")


def season_str(season: int) -> str:
    """2014 -> '2013-14'."""
    return f"{season - 1}-{str(season)[-2:]}"


def season_of_game(game_id: str) -> int:
    yy = int(game_id[3:5])
    return (1900 + yy + 1) if yy >= 90 else (2000 + yy + 1)


def raw_dir(cfg) -> Path:
    return resolve(cfg, "raw")


def _retry(fn, tries=4, base_sleep=5.0, what=""):
    last = None
    for t in range(tries):
        try:
            return fn()
        except Exception as e:            # any failure of a pull (timeout, HTTP error, bad JSON) is retried
            last = e
            if t == tries - 1:            # do not sleep after the final attempt and then give up
                break
            time.sleep(base_sleep * (3 ** t))
    raise RuntimeError(f"failed after {tries} tries: {what}: {last!r}")


# ----------------------------------------------------------------------------- game logs
LATE_GAME_HOURS = 4   # a night's last games end after midnight in US Eastern time: until 4 AM it is still that night


def eastern_today(now=None) -> pd.Timestamp:
    """Today's date in US Eastern time, the calendar the NBA dates its games by, without a time zone; until 4 AM it
    is still the night before, whose last games may be in play (at 12:45 AM on 2026-10-24, today is 2026-10-23)."""
    now = pd.Timestamp.now(tz="America/New_York") if now is None else pd.Timestamp(now).tz_convert("America/New_York")
    return (now - pd.Timedelta(hours=LATE_GAME_HOURS)).normalize().tz_localize(None)


def load_gamelog(season: int, phase: str, cfg, force=False, before=None) -> pd.DataFrame:
    """Player-level game log for a season and phase (cached).  `force` fetches it again, as the season in progress
    needs (the module docstring); `before`, a date, keeps only the games dated before it."""
    path = raw_dir(cfg) / "gamelog" / f"{season}_{phase}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    refuse_if_offline(path, f"the game log {season} {phase}")
    from nba_api.stats.endpoints import leaguegamelog
    def pull():
        return leaguegamelog.LeagueGameLog(season=season_str(season), season_type_all_star=SEASON_TYPES[phase],
                                           player_or_team_abbreviation="P", timeout=TIMEOUT).get_data_frames()[0]
    gl = _retry(pull, what=f"gamelog {season} {phase}")
    gl = gl[gl["GAME_ID"].str.startswith(GAME_PREFIX[phase])].copy()   # drop play-in etc.
    if before is not None:
        gl = gl[pd.to_datetime(gl["GAME_DATE"]) < pd.Timestamp(before)].copy()
    gl["season"] = season
    gl["phase"] = phase
    path.parent.mkdir(parents=True, exist_ok=True)
    gl.to_parquet(path, index=False)
    return gl


def played_phases(season: int, cfg: dict, phases=("RS", "PO")) -> list:
    """The phases of `phases` the season has games in: all but one whose game log is on disk with no rows, the playoffs
    of the season in progress.  A phase with no game log at all is kept, so building it fails loudly as before."""
    out = []
    for phase in phases:
        path = raw_dir(cfg) / "gamelog" / f"{int(season)}_{phase}.parquet"
        if not path.exists() or pq.ParquetFile(path).metadata.num_rows > 0:
            out.append(phase)
    return out


def game_table(gl: pd.DataFrame) -> pd.DataFrame:
    """One row per game: game_id, game_date, home_team_id, away_team_id, season, phase.  A phase not yet played (the
    playoffs of the season in progress) gives no rows."""
    if gl.empty:
        return pd.DataFrame({"game_id": pd.Series(dtype=str), "game_date": pd.Series(dtype="datetime64[ns]"),
                             "season": pd.Series(dtype=int), "phase": pd.Series(dtype=str),
                             "home_team_id": pd.Series(dtype=int), "away_team_id": pd.Series(dtype=int),
                             "neutral": pd.Series(dtype=bool)})
    g = gl[["GAME_ID", "GAME_DATE", "TEAM_ID", "MATCHUP", "season", "phase"]].drop_duplicates(["GAME_ID", "TEAM_ID"])
    g = g.assign(is_home=g["MATCHUP"].str.contains(" vs. "))
    # neutral-site games (NBA Cup knockout rounds, international games) list both teams as "@" (or both "vs."):
    # take the lower team id as the nominal home side and flag the game neutral (home effect = 0 in the design)
    n_home = g.groupby("GAME_ID")["is_home"].transform("sum")
    amb = n_home != 1
    g.loc[amb, "is_home"] = g.loc[amb, "TEAM_ID"] == g.loc[amb].groupby("GAME_ID")["TEAM_ID"].transform("min")
    home = g[g.is_home].set_index("GAME_ID")["TEAM_ID"]
    away = g[~g.is_home].set_index("GAME_ID")["TEAM_ID"]
    meta = g.drop_duplicates("GAME_ID").set_index("GAME_ID")[["GAME_DATE", "season", "phase"]]
    neutral = g.groupby("GAME_ID").apply(lambda d: bool((d["MATCHUP"].str.contains(" vs. ").sum()) != 1))
    neutral = neutral.rename("neutral")
    out = meta.join(home.rename("home_team_id")).join(away.rename("away_team_id")).join(neutral).reset_index()
    out = out.rename(columns={"GAME_ID": "game_id", "GAME_DATE": "game_date"})
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out.sort_values(["game_date", "game_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- per-game pulls
def _pbp_path(game_id, cfg):
    return raw_dir(cfg) / "pbp" / str(season_of_game(game_id)) / f"{game_id}.parquet"


def _box_path(game_id, cfg):
    return raw_dir(cfg) / "box" / str(season_of_game(game_id)) / f"{game_id}.parquet"


def _pbox_path(game_id, period, cfg):
    return raw_dir(cfg) / "periodbox" / str(season_of_game(game_id)) / f"{game_id}_{period}.parquet"


def fetch_pbp(game_id: str, cfg, force=False) -> pd.DataFrame:
    path = _pbp_path(game_id, cfg)
    if path.exists() and not force:
        return pd.read_parquet(path)
    refuse_if_offline(path, f"the play-by-play of {game_id}")
    from nba_api.stats.endpoints import playbyplayv3
    def pull():
        df = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=TIMEOUT).get_data_frames()[0]
        if len(df) == 0:
            raise ValueError("empty pbp")
        return df
    df = _retry(pull, what=f"pbp {game_id}")
    df = df.astype({c: str for c in df.columns if df[c].dtype == object})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def fetch_boxscore(game_id: str, cfg, force=False) -> pd.DataFrame:
    path = _box_path(game_id, cfg)
    if path.exists() and not force:
        return pd.read_parquet(path)
    refuse_if_offline(path, f"the box score of {game_id}")
    from nba_api.stats.endpoints import boxscoretraditionalv3
    def pull():
        df = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=TIMEOUT).get_data_frames()[0]
        if len(df) == 0:
            raise ValueError("empty boxscore")
        return df
    df = _retry(pull, what=f"box {game_id}")
    df = df.astype({c: str for c in df.columns if df[c].dtype == object})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def period_range(period: int):
    if period <= 4:
        return (period - 1) * 7200, period * 7200
    start = 28800 + (period - 5) * 3000
    return start, start + 3000


def fetch_period_box(game_id: str, period: int, cfg, force=False) -> pd.DataFrame:
    """Players with minutes in one period (BoxScoreTraditionalV3 with range_type=2)."""
    path = _pbox_path(game_id, period, cfg)
    if path.exists() and not force:
        return pd.read_parquet(path)
    refuse_if_offline(path, f"the period {period} box score of {game_id}")
    from nba_api.stats.endpoints import boxscoretraditionalv3
    s, e = period_range(period)
    def pull():
        return boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, start_period=period, end_period=period,
                                                           range_type=2, start_range=s, end_range=e,
                                                           timeout=TIMEOUT).get_data_frames()[0]
    df = _retry(pull, what=f"periodbox {game_id} p{period}")
    df = df.astype({c: str for c in df.columns if df[c].dtype == object})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def pbpstats_games(season: int, phase: str, cfg, force=False) -> pd.DataFrame:
    """Per-game possessions from api.pbpstats.com (cross-check only; empty before 2000-01)."""
    path = raw_dir(cfg) / "pbpstats_games" / f"{season}_{phase}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    refuse_if_offline(path, f"the pbpstats games {season} {phase}")
    def pull():
        r = requests.get("https://api.pbpstats.com/get-games/nba",
                         params={"Season": season_str(season), "SeasonType": SEASON_TYPES[phase]},
                         headers=HEADERS_PBPSTATS, timeout=TIMEOUT)
        r.raise_for_status()
        return pd.DataFrame(r.json().get("results", []))
    df = _retry(pull, what=f"pbpstats games {season} {phase}")
    if len(df) == 0:
        df = pd.DataFrame(columns=["GameId", "Date", "HomeTeamId", "AwayTeamId", "HomePoints", "AwayPoints",
                                   "HomePossessions", "AwayPossessions"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


# ----------------------------------------------------------------------------- season driver
def ingest_season(season: int, phases=("RS", "PO"), cfg=None, workers=2, delay=0.6, log_every=100,
                  verbose=True, refresh=False) -> dict:
    """Pull game logs, then PBP + boxscore for every game not yet cached. Idempotent.  With `refresh`, the season in
    progress: the game log and the pbpstats cross-check are fetched again, the log cut to the games before today in
    US Eastern time (the module docstring)."""
    summary = {}
    before = eastern_today() if refresh else None
    for phase in phases:
        log = load_gamelog(season, phase, cfg, force=refresh, before=before)
        if log.empty:
            summary[phase] = dict(n_games=0, fetched=0, failed=[], seconds=0.0)
            if verbose:
                print(f"{season} {phase}: no games yet")
            continue
        games = game_table(log)
        _pbpstats_cross_check(season, phase, cfg, verbose, force=refresh)
        todo = [game_id for game_id in games["game_id"]
                if not (_pbp_path(game_id, cfg).exists() and _box_path(game_id, cfg).exists())]
        if verbose:
            print(f"{season} {phase}: {len(games)} games, {len(todo)} to fetch")
        started = time.time()
        failed = _fetch_games(todo, cfg, workers, delay, log_every if verbose else 0)
        summary[phase] = dict(n_games=len(games), fetched=len(todo), failed=failed, seconds=time.time() - started)
        if verbose:
            print(f"  done {phase}: {len(todo)} fetched in {time.time() - started:.0f}s, {len(failed)} failed")
    return summary


def _pbpstats_cross_check(season: int, phase: str, cfg, verbose: bool, force: bool = False) -> None:
    """Cache the pbpstats possessions of the season (a cross-check only, so a failed pull is printed, not raised)."""
    try:
        pbpstats_games(season, phase, cfg, force=force)
    except OfflineFetch:
        raise                                  # offline, a missing file stops the run; it is not a failed pull
    except Exception as error:
        if verbose:
            print(f"  pbpstats games {season} {phase}: {error!r}")


def _fetch_games(todo: list, cfg, workers: int, delay: float, log_every: int) -> list:
    """Fetch the play-by-play and the box score of every game in `todo` on `workers` threads, each thread waiting
    `delay` seconds after every request; progress is printed every `log_every` games (never when 0).  Returns
    (game_id, error) for every game that failed: one bad game does not stop a season."""
    def work(game_id):
        fetch_pbp(game_id, cfg)
        time.sleep(delay)
        fetch_boxscore(game_id, cfg)
        time.sleep(delay)
        return game_id

    started = time.time()
    failed, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, game_id): game_id for game_id in todo}
        for future in as_completed(futures):
            done += 1
            error = _fetch_error(future)
            if error is not None:
                failed.append((futures[future], error))
            if log_every and done % log_every == 0:
                elapsed = time.time() - started
                print(f"  {done}/{len(todo)}  {elapsed:.0f}s  ({elapsed / done:.2f}s/game)")
    return failed


def _fetch_error(future) -> str | None:
    """The error of a finished fetch, shortened, or None; offline, a missing file is raised, not recorded."""
    try:
        future.result()
    except OfflineFetch:
        raise
    except Exception as error:
        return repr(error)[:200]
    return None
