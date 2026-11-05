"""The outside rating systems the Bias page compares OpenRAPM with: each one's public table pulled, checked and stored.

The Bias page (bias.py, commands/site.py) asks where OpenRAPM rates players higher or lower than a consensus of five
outside systems does.  This module gets the five tables.  Each system publishes one rating per player and side for
the season in progress on a public page, keyed by the NBA's player id, so no name is ever matched.

Words used below
    system     one outside rating (`SYSTEMS`): xRAPM, xrapm.com's 3-year RAPM, LOKI, DARKO, Pulse
    address    the one page or file a system's table is read from; one request a pull, never a retry
    table      player_id, offense, defense: points per 100 possessions, positive good on both sides.  xrapm.com prints
               defense as points allowed, "less is better", so its sign is flipped on the way in
    as of      the date a system says its ratings run through; None when its page prints none
    pull       one morning's request of every system, each stored in paths.public_ratings/<season>/<key>/<day>.parquet

The five are the owner's choice of 2026-11-05 (notes/DECISIONS.md, "The Bias page's consensus").  What is not pulled,
and must not be: EPM, since the terms of use of dunksandthrees.com forbid copying by script; nbarapm.com's time-decay
RAPM, since its robots.txt disallows the address its table comes from; xPM, which vorp.app serves to an account key
only.  Read a site's terms and its robots.txt before adding a system here.

The stored tables stay out of git (.gitignore, /data/*) and off the site: the repository is public and the ratings
are their makers'.  The page publishes only its own slopes, with each system named, credited and linked.

Example: `pull(cfg, 2027, "2026-11-05")` makes five requests and writes five files.  If lumahoops.org is down, LOKI's
record reads "failed" with the reason, its files of earlier days stay, and `latest` goes on returning the newest.
"""
from __future__ import annotations

import gzip
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import resolve
from .ingest import refuse_if_offline

USER_AGENT = "OpenRAPM (https://openrapm.com, github.com/bbstats/openrapm): one request a day for its Bias page"
TIMEOUT = 30            # seconds; a system that has not answered by then is asked again the next morning
PAUSE = 1.0             # seconds between two requests: two of the five go to xrapm.com
MIN_ROWS = 300          # every system lists 500 or more players, so fewer means its page changed
MAX_RATING = 25.0       # points per 100: no system rates anyone near it, so a larger number is a misread column
SUM_TOLERANCE = 0.11    # a system's own total may miss its offense + defense by rounding: xrapm.com prints one decimal
TABLE_COLUMNS = ["player_id", "offense", "defense"]
GZIP_MAGIC = b"\x1f\x8b"


class FormatChanged(ValueError):
    """A system's page no longer reads as it did when its parser was written."""


@dataclass(frozen=True)
class PublicRatings:
    """One system's table (`TABLE_COLUMNS`), the date the system says it runs through (None when it prints none) and
    the day it was pulled (filled in by `latest`)."""
    table: pd.DataFrame
    as_of: str | None
    pulled: str = ""


@dataclass(frozen=True)
class System:
    """One outside system: its key in file names and in bias.CONSENSUS_WEIGHTS, the name and the maker the page
    credits, the page a reader is sent to, the address its table is read from ({season} filled in) and its parser."""
    key: str
    name: str
    by: str
    link: str
    address: str
    parse: Callable


# ------------------------------------------------------------------------------------ the checks every table passes
def checked(name: str, frame: pd.DataFrame, tolerance: float = SUM_TOLERANCE) -> pd.DataFrame:
    """A parsed table as every reader gets it (`TABLE_COLUMNS`, by player id), or `FormatChanged` saying what is wrong.

    `frame` holds player_id, offense, defense and, when the system prints one, its own `total`.  A player a system
    lists without a rating on either side is left out.  Refused: fewer than `MIN_ROWS` rated players, a player listed
    twice, a rating that is not a number or is beyond `MAX_RATING`, and a total that is not offense + defense within
    `tolerance`, which is how a flipped defense column shows.

    Example: xrapm.com's row 3.5, -8.0, 11.5 is read as offense 3.5, defense +8.0, and 3.5 + 8.0 = 11.5 passes.
    """
    rated = frame.dropna(subset=["player_id", "offense", "defense"])
    if len(rated) < MIN_ROWS:
        raise FormatChanged(f"{name}: {len(rated)} rated players read, under the {MIN_ROWS} every system lists")
    table = rated.astype({"player_id": "int64", "offense": float, "defense": float})
    if table.player_id.duplicated().any():
        twice = int(table.player_id[table.player_id.duplicated()].iloc[0])
        raise FormatChanged(f"{name}: player {twice} is listed twice")
    values = table[["offense", "defense"]].to_numpy()
    if not np.isfinite(values).all() or np.abs(values).max() > MAX_RATING:
        raise FormatChanged(f"{name}: a rating is not a number or is beyond {MAX_RATING:g} points per 100")
    if "total" in table:
        gaps = (pd.to_numeric(table.total) - table.offense - table.defense).abs().dropna()
        if len(gaps) and float(gaps.max()) > tolerance:
            raise FormatChanged(f"{name}: its total misses offense + defense by {gaps.max():.2f}; a column changed "
                                f"its sign")
    return table[TABLE_COLUMNS].sort_values("player_id").reset_index(drop=True)


def _a_page_read(parse: Callable) -> Callable:
    """`parse`, with every way a changed page can break it (a missing key, a short row, text that is not JSON)
    raised as `FormatChanged`, so `pull` has one failure to record."""
    @wraps(parse)
    def reading(text: str, season: int) -> PublicRatings:
        try:
            return parse(text, season)
        except FormatChanged:
            raise
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise FormatChanged(f"{parse.__name__}: {type(error).__name__}: {error}") from error
    return reading


# ------------------------------------------------------------------------------------ xrapm.com: xRAPM, 3-year RAPM
# One row of either table: the link to the player's page, which carries his NBA id, his team, then offense, defense and
# total, each followed by its percentile.  The 3-year page's cells carry a style attribute, the front page's do not.
XRAPM_ROW = re.compile(
    r"""player_pages/(\d+)\.html["']?>[^<]*(?:</a>)?\s*</td>\s*<td[^>]*>[^<]*</td>\s*"""
    r"<td[^>]*>\s*(-?\d+(?:\.\d+)?)\s*\(\d+\)\s*</td>\s*"
    r"<td[^>]*>\s*(-?\d+(?:\.\d+)?)\s*\(\d+\)\s*</td>\s*"
    r"<td[^>]*>\s*(-?\d+(?:\.\d+)?)\s*\(\d+\)\s*</td>")
XRAPM_DATE = re.compile(r"Metrics updated on:\s*(\d{4}-\d{2}-\d{2})")


@_a_page_read
def parse_xrapm(text: str, season: int) -> PublicRatings:
    """xrapm.com's table: the front page's xRAPM, or table_pages/RAPM_3y.html's 3-year RAPM, which prints no date.

    The page's Defense column is points allowed, "less is better", and its Total is Offense minus that column: the
    defense read here is its negative.  The page names no season, so `season` is not read.

    Example: the cells "3.5 (98)", "-8.0 (99)", "11.5 (99)" give offense 3.5 and defense 8.0.
    """
    rows = [(int(player), float(offense), -float(allowed), float(total))
            for player, offense, allowed, total in XRAPM_ROW.findall(text)]
    frame = pd.DataFrame(rows, columns=["player_id", "offense", "defense", "total"])
    date = XRAPM_DATE.search(text)
    return PublicRatings(checked("xrapm.com", frame), date.group(1) if date else None)


# ------------------------------------------------------------------------------------ darko.app: DARKO's DPM
def _darko_values(document: dict) -> list:
    """The one flat list of darko.app's __data.json that holds the leaderboard: the node whose first item names
    `players`.  SvelteKit writes every value once into that list and refers to it by its position."""
    for node in document["nodes"]:
        values = node.get("data") if isinstance(node, dict) else None
        if isinstance(values, list) and values and isinstance(values[0], dict) and "players" in values[0]:
            return values
    raise FormatChanged("darko.app: no leaderboard in __data.json")


@_a_page_read
def parse_darko(text: str, season: int) -> PublicRatings:
    """darko.app's leaderboard from its __data.json: `dpm`, `o_dpm` and `d_dpm` per `nba_id`, defense positive good.

    In the flat list, `values[0]` points at the table and at the date; the table points at its column names and at
    its columns, and a column is a list of positions, a negative one standing for a missing value.

    Example: values = [{"players": 1, "ratingsThrough": 5}, {"keys": 2, "values": 4}, [3], "nba_id", [6],
    "2026-11-04", [7], 203999] reads as one column, nba_id = [203999], as of 2026-11-04.
    """
    values = _darko_values(json.loads(text))
    table = values[values[0]["players"]]
    names = [values[position] for position in values[table["keys"]]]
    columns = {name: [None if position < 0 else values[position] for position in values[column]]
               for name, column in zip(names, values[table["values"]])}
    frame = pd.DataFrame(columns)
    if set(frame.season.astype(int)) != {int(season)}:
        raise FormatChanged(f"darko.app: seasons {sorted(set(frame.season))} where {season} was asked")
    frame = frame.rename(columns={"nba_id": "player_id", "o_dpm": "offense", "d_dpm": "defense", "dpm": "total"})
    through = str(values[values[0]["ratingsThrough"]])[:10]
    return PublicRatings(checked("darko.app", frame[["player_id", "offense", "defense", "total"]]),
                         through if re.fullmatch(r"\d{4}-\d{2}-\d{2}", through) else None)


# ------------------------------------------------------------------------------------ lumahoops.org: LOKI
# The page is written by Next.js, which ships its data as JavaScript strings pushed one after another; joined, they
# hold each data file under its name.  A number JSON cannot hold is written as a string: "$-0" is negative zero
# (a defense of -0.00, met on 2026-11-05) and "$NaN" no number at all.
NEXT_STRING = re.compile(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)')
NEXT_NUMBERS = {"$-0": 0.0, "$NaN": None, "$undefined": None}


@_a_page_read
def parse_loki(text: str, season: int) -> PublicRatings:
    """lumahoops.org/loki's current board, refreshed on Mondays: the object the page holds under
    "loki/<season>/current.json", with `cols` and `rows`; `pid` is the NBA id, `O` and `D` the two sides (defense
    positive good), `net` their sum and `asof` the date.

    Example: cols ["pid", "name", "team", "net", "O", "D", ...] and the row [1628983, "...", "OKC", 12.89, 10.2, 2.69]
    give offense 10.2 and defense 2.69.
    """
    payload = "".join(json.loads(f'"{piece}"') for piece in NEXT_STRING.findall(text))
    name = f'"loki/{int(season)}/current.json":'
    if name not in payload:
        raise FormatChanged(f"lumahoops.org: the page holds no {name[:-1]}")
    board, _ = json.JSONDecoder().raw_decode(payload, payload.index("{", payload.index(name) + len(name)))
    if int(board["season"]) != int(season):
        raise FormatChanged(f"lumahoops.org: season {board['season']} where {season} was asked")
    rows = [[NEXT_NUMBERS.get(value, value) if isinstance(value, str) else value for value in row]
            for row in board["rows"]]
    frame = pd.DataFrame(rows, columns=board["cols"]).rename(
        columns={"pid": "player_id", "O": "offense", "D": "defense", "net": "total"})
    return PublicRatings(checked("lumahoops.org", frame[["player_id", "offense", "defense", "total"]]),
                         str(board["asof"])[:10])


# ------------------------------------------------------------------------------------ courtsignal.net: Pulse
@_a_page_read
def parse_pulse(text: str, season: int) -> PublicRatings:
    """courtsignal.net's daily board of the season, data/model-boards/current/season-<season>.json: `rows` holds one
    list of players per date, each [NBA id, total, offense, defense, ...] with defense positive good; the latest
    date is read.

    Example: {"season": 2027, "dates": ["2026-11-04"], "rows": {"2026-11-04": [[203999, 9.0, 7.95, 1.05, ...]]}}
    gives offense 7.95 and defense 1.05, as of 2026-11-04.
    """
    document = json.loads(text)
    if int(document["season"]) != int(season):
        raise FormatChanged(f"courtsignal.net: season {document['season']} where {season} was asked")
    day = max(document["dates"])
    frame = pd.DataFrame([row[:4] for row in document["rows"][day]],
                         columns=["player_id", "total", "offense", "defense"])
    return PublicRatings(checked("courtsignal.net", frame), str(day)[:10])


# ------------------------------------------------------------------------------------ the five, and their pull
SYSTEMS = {system.key: system for system in (
    System("xrapm", "xRAPM", "Jeremias Engelmann", "https://xrapm.com/", "https://xrapm.com/", parse_xrapm),
    System("rapm_3y", "3-year RAPM", "Jeremias Engelmann", "https://xrapm.com/table_pages/RAPM_3y.html",
           "https://xrapm.com/table_pages/RAPM_3y.html", parse_xrapm),
    # LUMA's outputs are open with credit to both names (lumahoops.org/luma/data/_notice.json, 2026-11-05)
    System("loki", "LOKI", "LUMA Basketball Research and CourtShare", "https://lumahoops.org/loki",
           "https://lumahoops.org/loki", parse_loki),
    System("darko", "DARKO", "Kostya Medvedovsky", "https://www.darko.app/", "https://www.darko.app/__data.json",
           parse_darko),
    System("pulse", "Pulse", "CourtSignal", "https://courtsignal.net/ratings",
           "https://courtsignal.net/data/model-boards/current/season-{season}.json", parse_pulse),
)}


def store_folder(cfg: dict, season: int, key: str) -> Path:
    """Where one system's pulls of one season are stored: paths.public_ratings/<season>/<key>."""
    return resolve(cfg, "public_ratings") / str(int(season)) / key


def fetch(system: System, season: int, folder: Path) -> str:
    """The text at the system's address: one request that names OpenRAPM, and no second try, since a system that
    does not answer is asked again the next morning.  xrapm.com sends gzip whatever it is asked for."""
    refuse_if_offline(folder, f"{system.name}'s table")
    response = requests.get(system.address.format(season=int(season)), headers={"User-Agent": USER_AGENT},
                            timeout=TIMEOUT)
    response.raise_for_status()
    body = response.content
    return (gzip.decompress(body) if body[:2] == GZIP_MAGIC else body).decode("utf-8")


def pull_one(cfg: dict, system: System, season: int, day: str) -> dict:
    """One system asked and stored under `day`: {"key", "status": "pulled", "rows", "as_of"}, or {"key", "status":
    "failed", "error"} with nothing written but, when the page was read and not understood, the page itself as
    <day>.unparsed beside the earlier pulls."""
    folder = store_folder(cfg, season, system.key)
    try:
        text = fetch(system, season, folder)
    except requests.RequestException as error:
        return dict(key=system.key, status="failed", error=f"{type(error).__name__}: {error}")
    folder.mkdir(parents=True, exist_ok=True)
    try:
        ratings = system.parse(text, season)
    except FormatChanged as error:
        (folder / f"{day}.unparsed").write_text(text, encoding="utf-8")
        return dict(key=system.key, status="failed", error=str(error))
    ratings.table.assign(as_of=ratings.as_of).to_parquet(folder / f"{day}.parquet", index=False)
    return dict(key=system.key, status="pulled", rows=len(ratings.table), as_of=ratings.as_of)


def pull(cfg: dict, season: int, day: str, keys=None, pause: float = PAUSE) -> list[dict]:
    """Every system of `keys` (default all five) asked once, `pause` seconds apart: one record each (`pull_one`).
    With OPENRAPM_OFFLINE=1 the first request raises `OfflineFetch`."""
    records = []
    for position, key in enumerate(keys or SYSTEMS):
        if position:
            time.sleep(pause)
        records.append(pull_one(cfg, SYSTEMS[key], season, day))
    return records


def latest(cfg: dict, season: int) -> dict:
    """{key: PublicRatings} from each system's newest stored pull of `season`; a system never pulled is left out."""
    out = {}
    for key in SYSTEMS:
        files = sorted(store_folder(cfg, season, key).glob("*.parquet"))
        if files:
            stored = pd.read_parquet(files[-1])
            as_of = stored.as_of.iloc[0]
            out[key] = PublicRatings(stored[TABLE_COLUMNS], None if pd.isna(as_of) else str(as_of), files[-1].stem)
    return out
