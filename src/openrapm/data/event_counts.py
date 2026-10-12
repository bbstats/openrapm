"""Counts read straight from the play-by-play: the parts of four box-score counts.

The 13 box counts (config.yaml's box_counts) come from the box score: one number per player and game.  The
play-by-play says more about four of them, event by event:

    a defensive rebound      the miss it ended: a free throw, a three, a two at the rim, a two from farther out
    a steal                  the turnover it took: a pass (an interception) or a lost ball (the ball taken from a
                             dribbler)
    a made field goal        whether an assist is written beside it
    a three-point attempt    where it was taken: the corner or above the break
    a block                  where the blocked shot was taken, and whether the blocker's team got the ball back
    a turnover               whether the defence stole it
    an assist                where the shot it set up was taken (credited to the passer)
and the offensive fouls a player commits.

`season_event_counts` counts them per game, player and phase (RS regular season, PO playoffs), and `openrapm data
events` caches each season in paths.event_counts.  The season panel pads each count per 100 possessions as it pads
the 13 (data/panel.py, box_rates.py), so the box-score prior can read a box count and its parts side by side (the
owner's stat-splits plan, 2026-10-12, experiments 52-55; features.EVENT_RATES).

How each part is read.  Rows are taken in the file's order, which is the order of play (`actionNumber` is not: it
runs backwards on 1-2% of rows, where an event was entered late).

    defensive rebound   a player's Rebound row whose running "(Off:a Def:b)" count raised Def (the feed has no
                        offensive / defensive tag).  The miss it ended is the last row before it that has an action
                        type: the rows with none are the STEAL and BLOCK rows the feed writes right after the event
                        they belong to.  A missed free throw is a Free Throw row whose text starts with "MISS"; a
                        missed shot is a two or a three by its shotValue
    rim two             a two within 6 ft of the basket by the shot's coordinates (xLegacy, yLegacy: tenths of a
                        foot, the basket at (0, 0)), the project's rim zone.  A two logged without a location, at
                        (0, 0), is a rim two too: up to 38% of missed twos before 2011 have none, and nearly all of
                        those are layups, dunks and tip-ins.  `shotDistance` is not read: it is the feet written in
                        the text, 0 for a sixth of all threes
    steal               the STEAL row (no action type, the stealer named) right after the Turnover row it took.
                        The turnover's subType says which kind: a pass ("Bad Pass", "Stolen Pass Turnover") or,
                        for every other, a lost ball ("Lost Ball" or "Poss Lost Ball Turnover" all but a handful
                        of times).  The owner allowed the turnover's type for this on 2026-10-12; the shot-type
                        words (dunk, layup, pull-up, ...) stay out of every input
    assisted            a made shot whose text carries "(Name n AST)"
    corner three        |x| >= 22 ft and y <= 9.25 ft: the straight part of the three-point line, out from the
                        baseline.  A three logged without a location (1% in 1997, none since 2005) counts as above
                        the break
    block               the BLOCK row (no action type, the blocker named) right after the missed shot it blocked:
                        a rim two, a two from beyond 6 ft or a three by that shot.  A Russell block (Bill Russell
                        kept his blocks in play) is one whose rebound, within the next three rows, belongs to the
                        blocker's team, a team rebound included
    stolen turnover     a Turnover row with a STEAL row right after it
    offensive foul      a Foul row whose subType starts "Offensive" (charges get a subType of their own only from
                        about 2006, so they are counted with the rest)
    assist by zone      the passer named in "(Name n AST)", matched by name to a player of the shooting team in
                        the game (the feed has no passer id), and the zone of the make he set up: a rim two, a two
                        from beyond 6 ft, a corner three, an above-the-break three.  A name that matches no player,
                        or two, is left out

Example: the rows "MISS Sengun 25' 3PT Jump Shot" then "Wallace REBOUND (Off:0 Def:1)" give Wallace one defensive
rebound of a missed three, and "Holmgren 1' Cutting Layup Shot (2 PTS) (Dort 1 AST)" gives Holmgren 2 assisted
points.
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..features import EVENT_RATES
from .config import resolve
from .ingest import GAME_PREFIX, raw_dir

__all__ = ["EVENT_COUNTS", "read_game", "plain_name", "game_event_counts", "season_event_counts",
           "event_counts_path", "build_event_counts", "load_event_counts", "event_design"]

# the per-game counts, in features.EVENT_RATES order: "bad_pass_steals" is padded into "bad_pass_steals_per100"
EVENT_COUNTS = [rate.removesuffix("_per100") for rate in EVENT_RATES]
(DEF_REBOUNDS_OF_FREE_THROWS, DEF_REBOUNDS_OF_THREES, DEF_REBOUNDS_OF_RIM_TWOS, DEF_REBOUNDS_OF_OUTSIDE_TWOS,
 BAD_PASS_STEALS, LOST_BALL_STEALS, ASSISTED_POINTS, UNASSISTED_POINTS, CORNER_THREES, ABOVE_BREAK_THREES,
 ASSISTED_THREES, UNASSISTED_THREES, RUSSELL_BLOCKS, RIM_BLOCKS, MID_RANGE_BLOCKS, THREE_POINT_BLOCKS,
 OFFENSIVE_FOULS, STOLEN_TURNOVERS, ASSISTS_TO_RIM, ASSISTS_TO_MID_RANGE, ASSISTS_TO_CORNER_THREES,
 ASSISTS_TO_ABOVE_BREAK_THREES) = EVENT_COUNTS
# NBA team ids run 1610612737-1610612766; a row whose personId is a team (a team rebound) is no player's
TEAM_IDS = (1_610_612_700, 1_610_612_800)
RIM_FEET = 6.0
CORNER_X, CORNER_Y = 220.0, 92.5          # tenths of a foot
RUSSELL_REACH = 3                         # rows after a block in which its rebound is looked for
PBP_COLUMNS = ["actionType", "subType", "description", "personId", "teamId", "playerName", "playerNameI",
               "shotValue", "xLegacy", "yLegacy"]
ASSIST = re.compile(r"\(([^()]*?) \d+ AST\)")
NAME_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?$")
ASSIST_ZONES = (ASSISTS_TO_RIM, ASSISTS_TO_MID_RANGE, ASSISTS_TO_CORNER_THREES, ASSISTS_TO_ABOVE_BREAK_THREES)
POINTS = (ASSISTED_POINTS, UNASSISTED_POINTS)       # counted in points; every other count in events
KEYS = ["game_id", "season", "phase", "player_id"]


# ------------------------------------------------------------------------------------------------ one game
@dataclass
class _Game:
    """One game's play-by-play as arrays, rows in the file's order."""
    action: np.ndarray            # actionType, "" for the STEAL and BLOCK rows
    kind: np.ndarray              # subType
    text: pd.Series               # description
    person: np.ndarray            # personId: a player, or a team on a team row
    team: np.ndarray              # the team the row belongs to (teamId, or personId on a team rebound)
    value: np.ndarray             # shotValue: 2 or 3 on a shot
    rim: np.ndarray               # a shot within RIM_FEET, or logged without a location
    corner: np.ndarray            # a three from the corner
    player: np.ndarray            # the row names a player, not a team
    names: np.ndarray             # playerName
    initials: np.ndarray          # playerNameI ("S. Gilgeous-Alexander")


def _numbers(pbp: pd.DataFrame, column: str, dtype=float) -> np.ndarray:
    return pd.to_numeric(pbp[column], errors="coerce").fillna(0).to_numpy(dtype=dtype)


def _is_team(ids: np.ndarray) -> np.ndarray:
    return (ids >= TEAM_IDS[0]) & (ids < TEAM_IDS[1])


def _text(pbp: pd.DataFrame, column: str) -> np.ndarray:
    return pbp[column].fillna("").astype(str).str.strip().to_numpy()


def read_game(pbp: pd.DataFrame) -> _Game:
    person, team_id = _numbers(pbp, "personId", np.int64), _numbers(pbp, "teamId", np.int64)
    value, x, y = _numbers(pbp, "shotValue"), _numbers(pbp, "xLegacy"), _numbers(pbp, "yLegacy")
    unlocated = (x == 0) & (y == 0)
    return _Game(action=_text(pbp, "actionType"), kind=_text(pbp, "subType"),
                 text=pbp["description"].fillna("").astype(str).reset_index(drop=True), person=person,
                 team=np.where(_is_team(team_id), team_id, np.where(_is_team(person), person, 0)), value=value,
                 rim=unlocated | (np.hypot(x, y) / 10.0 <= RIM_FEET),
                 corner=(value == 3) & ~unlocated & (np.abs(x) >= CORNER_X) & (y <= CORNER_Y),
                 player=(person > 0) & ~_is_team(person), names=_text(pbp, "playerName"),
                 initials=_text(pbp, "playerNameI"))


def _flagged(game: _Game, word: str) -> np.ndarray:
    """The feed's rows with no action type that carry `word` ("STEAL", "BLOCK") and name a player."""
    return (game.action == "") & game.player & game.text.str.contains(word, regex=False).to_numpy()


def _defensive_rebounds(game: _Game) -> np.ndarray:
    """The rows that are a player's defensive rebound: his running Def count is higher than on his rebound before
    (0 before his first)."""
    rebound = np.flatnonzero((game.action == "Rebound") & game.player)
    counted = (game.text.iloc[rebound].str.extract(r"Def:\s*(\d+)", expand=False).astype(float)
               .reset_index(drop=True))
    who = pd.Series(game.person[rebound])
    before = counted.groupby(who).ffill().groupby(who).shift(1).fillna(0.0)
    defensive = np.zeros(len(game.action), dtype=bool)
    defensive[rebound] = (counted > before).to_numpy()
    return defensive


def _labelled_before(action: np.ndarray) -> np.ndarray:
    """For every row, the last row before it that has an action type (-1 when there is none)."""
    labelled = np.where(action != "", np.arange(len(action)), -1)
    return np.r_[-1, np.maximum.accumulate(labelled)[:-1]] if len(action) else labelled


def _rebound_parts(game: _Game) -> dict:
    """{count: the rebound rows of that part}: each defensive rebound filed under the miss it ended, or under none
    when the row before is not a miss (a few in ten thousand)."""
    rebounds = np.flatnonzero(_defensive_rebounds(game))
    miss = _labelled_before(game.action)[rebounds]
    found = miss >= 0
    at = np.where(found, miss, 0)
    free_throw = found & (game.action[at] == "Free Throw") & game.text.iloc[at].str.startswith("MISS").to_numpy()
    shot = found & (game.action[at] == "Missed Shot")
    two = shot & (game.value[at] == 2)
    return {DEF_REBOUNDS_OF_FREE_THROWS: rebounds[free_throw],
            DEF_REBOUNDS_OF_THREES: rebounds[shot & (game.value[at] == 3)],
            DEF_REBOUNDS_OF_RIM_TWOS: rebounds[two & game.rim[at]],
            DEF_REBOUNDS_OF_OUTSIDE_TWOS: rebounds[two & ~game.rim[at]]}


def _steal_parts(game: _Game) -> dict:
    """{count: the steal rows of that part}: a pass when the Turnover row it took (the row before, else the row
    after) names a pass, a lost ball otherwise; and the Turnover rows a STEAL row follows."""
    flagged = _flagged(game, "STEAL")
    steals = np.flatnonzero(flagged)
    before, after = np.maximum(steals - 1, 0), np.minimum(steals + 1, len(game.action) - 1)
    turnover = np.where((steals > 0) & (game.action[before] == "Turnover"), before, after)
    a_pass = ((game.action[turnover] == "Turnover")
              & pd.Series(game.kind[turnover]).str.contains("Pass", regex=False).to_numpy())
    stolen = (game.action == "Turnover") & game.player & np.r_[flagged[1:], False]
    return {BAD_PASS_STEALS: steals[a_pass], LOST_BALL_STEALS: steals[~a_pass],
            STOLEN_TURNOVERS: np.flatnonzero(stolen)}


def _recovered(game: _Game, blocks: np.ndarray) -> np.ndarray:
    """Each block whose rebound, the first Rebound row within RUSSELL_REACH rows, belongs to the blocker's team."""
    out = np.zeros(len(blocks), dtype=bool)
    for place, row in enumerate(blocks):
        later = [at for at in range(row + 1, min(row + 1 + RUSSELL_REACH, len(game.action)))
                 if game.action[at] == "Rebound"]
        out[place] = bool(later) and game.team[row] != 0 and game.team[later[0]] == game.team[row]
    return out


def _block_parts(game: _Game) -> dict:
    """{count: the block rows of that part}: by the blocked shot, the row above, and the Russell blocks."""
    blocks = np.flatnonzero(_flagged(game, "BLOCK"))
    shot = np.maximum(blocks - 1, 0)
    two = game.value[shot] == 2
    return {RUSSELL_BLOCKS: blocks[_recovered(game, blocks)], RIM_BLOCKS: blocks[two & game.rim[shot]],
            MID_RANGE_BLOCKS: blocks[two & ~game.rim[shot]], THREE_POINT_BLOCKS: blocks[game.value[shot] == 3]}


def plain_name(name: str) -> str:
    """A name the description and the roster columns agree on: no accents, no suffix, no case.

    Example: "Dončić" and "Doncic" both give "doncic"; "Jaren Jackson Jr." gives "jaren jackson"."""
    letters = unicodedata.normalize("NFKD", str(name))
    letters = "".join(c for c in letters if not unicodedata.combining(c)).lower().strip()
    letters = NAME_SUFFIX.sub("", letters).strip().rstrip(".").strip()
    return re.sub(r"\s+", " ", letters)


def _roster(game: _Game) -> dict:
    """{(team, plain name): {player ids}} over every player row of the game, under both name columns."""
    roster: dict = {}
    for row in np.flatnonzero(game.player & (game.team != 0)):
        for name in (game.names[row], game.initials[row]):
            if name and name != "nan":
                roster.setdefault((int(game.team[row]), plain_name(name)), set()).add(int(game.person[row]))
    return roster


def _passer(roster: dict, team: int, written: str) -> int | None:
    """The one player of `team` the assist names, by his full name, else by surname alone; None if not one."""
    name = plain_name(written)
    found = roster.get((team, name), set())
    if len(found) != 1:
        surname = name.split(" ")[-1]
        found = {pid for (side, known), ids in roster.items() if side == team and known.split(" ")[-1] == surname
                 for pid in ids}
    return next(iter(found)) if len(found) == 1 else None


def _assist_parts(game: _Game, assisted: np.ndarray) -> dict:
    """{count: the passers' ids}: each assisted make's passer, filed under the zone of the make."""
    roster, out = _roster(game), {name: [] for name in ASSIST_ZONES}
    for row in np.flatnonzero(assisted):
        match = ASSIST.search(game.text.iat[row])
        passer = None if match is None else _passer(roster, int(game.team[row]), match.group(1))
        if passer is not None:
            zone = ((ASSISTS_TO_CORNER_THREES if game.corner[row] else ASSISTS_TO_ABOVE_BREAK_THREES)
                    if game.value[row] == 3 else (ASSISTS_TO_RIM if game.rim[row] else ASSISTS_TO_MID_RANGE))
            out[zone].append(passer)
    return {name: np.asarray(ids, dtype=np.int64) for name, ids in out.items()}


def game_event_counts(pbp: pd.DataFrame) -> pd.DataFrame:
    """One game's play-by-play, rows in the file's order -> one row per player with any of the counts: player_id,
    then `EVENT_COUNTS` (points for the two points counts, events for the others)."""
    game = read_game(pbp)
    made = (game.action == "Made Shot") & game.player
    three = np.isin(game.action, ("Made Shot", "Missed Shot")) & game.player & (game.value == 3)
    assisted = made & game.text.str.contains("AST)", regex=False).to_numpy()
    rows = {**_rebound_parts(game), **_steal_parts(game), **_block_parts(game),
            ASSISTED_POINTS: np.flatnonzero(assisted), UNASSISTED_POINTS: np.flatnonzero(made & ~assisted),
            CORNER_THREES: np.flatnonzero(three & game.corner),
            ABOVE_BREAK_THREES: np.flatnonzero(three & ~game.corner),
            ASSISTED_THREES: np.flatnonzero(assisted & (game.value == 3)),
            UNASSISTED_THREES: np.flatnonzero(made & ~assisted & (game.value == 3)),
            OFFENSIVE_FOULS: np.flatnonzero((game.action == "Foul") & game.player
                                            & pd.Series(game.kind).str.startswith("Offensive").to_numpy())}
    hits = {name: pd.Series(game.value[at] if name in POINTS else np.ones(len(at)), index=game.person[at])
            for name, at in rows.items()}
    hits.update({name: pd.Series(np.ones(len(ids)), index=ids) for name, ids in _assist_parts(game, assisted).items()})
    out = pd.DataFrame({name: values.groupby(level=0).sum() for name, values in hits.items()})
    out = out.reindex(columns=EVENT_COUNTS).fillna(0.0)
    return out.rename_axis("player_id").reset_index()


# ---------------------------------------------------------------------------------------------- one season
def season_event_counts(season: int, cfg: dict) -> pd.DataFrame:
    """Every game of the season, regular season then playoffs, from data/raw/pbp/{season}: `KEYS` then
    `EVENT_COUNTS`, one row per game, phase and player with any count."""
    folder = raw_dir(cfg) / "pbp" / str(season)
    parts = []
    for phase, prefix in GAME_PREFIX.items():
        for path in sorted(folder.glob(f"{prefix}*.parquet")):
            counts = game_event_counts(pd.read_parquet(path, columns=PBP_COLUMNS))
            parts.append(counts.assign(game_id=path.stem, season=int(season), phase=phase))
    if not parts:
        raise FileNotFoundError(f"no play-by-play for {season} in {folder}; run `openrapm data ingest {season}`")
    return pd.concat(parts, ignore_index=True)[[*KEYS, *EVENT_COUNTS]]


def event_counts_path(season: int, cfg: dict):
    return resolve(cfg, "event_counts") / f"{int(season)}.parquet"


def build_event_counts(cfg: dict, seasons, force: bool = False) -> pd.DataFrame:
    """Count and cache every season asked for (a cached season is read, unless `force`); one line per season.
    Returns each season's totals per count, for holding against the box score."""
    totals = []
    for season in seasons:
        started, path = time.time(), event_counts_path(season, cfg)
        if force or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            season_event_counts(season, cfg).to_parquet(path, index=False)
        counts = pd.read_parquet(path)
        totals.append(counts[EVENT_COUNTS].sum().rename(int(season)))
        print(f"  {season}: {counts.game_id.nunique()} games, {len(counts)} rows ({time.time() - started:.0f}s)",
              flush=True)
    return pd.DataFrame(totals)


def load_event_counts(season: int, cfg: dict) -> pd.DataFrame:
    """The season's cached counts (`KEYS` + `EVENT_COUNTS`); refused with the command to run when not built."""
    path = event_counts_path(season, cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run `openrapm data events {int(season)}` (the play-by-play "
                                f"event counts the season panel reads)")
    counts = pd.read_parquet(path)
    return counts.assign(game_id=counts.game_id.astype(str))


# ------------------------------------------------------------------------------------- a design to pad them on
def event_design(design, counts: pd.DataFrame) -> tuple:
    """(`design` with its box table replaced by the event counts of the same game, player-season and phase rows and
    its box_counts by `EVENT_COUNTS`; the game ids whose counts were read), for box_rates.box_rates to pad.

    The rows are the box table's own, so the parts are counted over exactly the games and possessions the 13 rates
    are: a design cut to part of a season keeps part of its box table (data/panel.cut_design), and so part of the
    counts.  A row of the box table with no such events counts zeros; a player with events and no box line is not
    counted, as the 13 rates do not count him.
    """
    games = design.games[["game_id", "game_idx"]].assign(game_id=lambda frame: frame.game_id.astype(str))
    player_seasons = design.spec.player_seasons[["player_season", "player_id", "season"]]
    events = counts.merge(games, on="game_id").merge(player_seasons, on=["player_id", "season"])
    keys = design.game_box[["game_idx", "player_season", "phase"]]
    table = keys.merge(events[["game_idx", "player_season", "phase", *EVENT_COUNTS]],
                       on=["game_idx", "player_season", "phase"], how="left")
    table[EVENT_COUNTS] = table[EVENT_COUNTS].fillna(0.0)
    read = set(games.game_id[games.game_idx.isin(keys.game_idx.unique())])
    return replace(design, game_box=table, spec=replace(design.spec, box_counts=list(EVENT_COUNTS))), read
