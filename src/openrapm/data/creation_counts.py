"""Self-creation's counts from the play-by-play (experiment 111, the owner, 2026-10-25: "On FGM we use actual.  On
FGX and FTA, we estimate potentially assisted.  For a single number", then "come up with a more robust estimate").

A made shot says whether it was assisted.  A missed shot and a free-throw trip do not, so each gets the chance that
it came off a pass that would have been an assist had the shot gone in:

    a shot's cell      its spot and how the possession got there (`SPOTS` x `CONTEXTS`):
                       spot      rim (4 ft or closer, or logged without a location), paint (4-10 ft), mid (10-16 ft),
                                 long two (16 ft+), corner three (|x| >= 22 ft and y <= 9.25 ft), above-the-break three
                       context   putback: within 3 s of his team's offensive rebound; second chance: later in a
                                 possession that had one; transition: the first shot within 7 s of a defensive rebound
                                 or a steal; early: within 7 s of any other start; then 7-14, 14-20 and 20+ seconds
                                 into the possession
    p(cell)            the season's assisted share of made shots in the cell, padded `CELL_PAD` makes toward
                       `CELL_PRIOR`, the cell's share over 1997-2026
    a miss's chance    p(cell) moved by the cell's miss factor (`MISS_FACTOR`, a logit shift), learned from 2016
                       tracking: a make model that also knows the dribbles and the touch time before each shot scores
                       the misses, and the shift moves p(cell) to its mean over the cell's misses.  Misses at the rim
                       come off a pass less often than makes there, so most rim shifts are about -0.4
    a trip's chance    the season's mean chance of a missed rim shot x `FT_FACTOR`, fitted so that the trips off a
                       pass with a free throw made add up to the NBA's free-throw assists over 2014-2017
    a trip             the first free throw of a trip, technicals out; an and-one's free throw is left to its made
                       shot, which already counts as an attempt

The player's own lean is not here: a per-game count cannot hold it.  The panel pads every count of `CREATION_COUNTS`
per 100 possessions over the games it is built from (data/panel.py), a fold's fit games alone, and
features.CREATION_ESTIMATED reads his lean from the padded rates: in each spot group, inside (rim and paint), other
twos and threes, (assisted makes + pad) / (expected assisted makes + pad).

Example: a corner three 4 s after a steal is in the cell corner_three|transition; 2016 holds 632 such makes, 97.9%
assisted, padded to 0.979, and the cell's miss factor is -0.56, so a miss there gets expit(logit(0.979) - 0.56) =
0.964.

`openrapm data events` builds these beside the event counts and caches them in paths.event_counts:
<season>_creation_shots.parquet (one row per shot and per trip) and <season>_creation.parquet (`KEYS` +
`CREATION_COUNTS`).
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from ..features import CREATION_GROUPS as GROUPS
from ..features import CREATION_RATES
from .config import resolve
from .ingest import GAME_PREFIX, raw_dir

__all__ = ["SPOTS", "CONTEXTS", "SPOT_GROUPS", "CREATION_COUNTS", "game_shots", "season_shots", "cell_shares",
           "season_creation_counts", "build_creation_counts", "load_creation_counts"]

SPOTS = ["rim", "paint", "mid", "long_two", "corner_three", "above_break_three"]
CONTEXTS = ["putback", "second_chance", "transition", "early", "s7_14", "s14_20", "s20_plus"]
SPOT_GROUPS = {"rim": "inside", "paint": "inside", "mid": "two", "long_two": "two", "corner_three": "three",
               "above_break_three": "three"}
CELL_PAD = 50.0
CORNER_X, CORNER_Y = 220.0, 92.5          # tenths of a foot
PUTBACK_SECONDS, EARLY_SECONDS = 3.0, 7.0
PBP_COLUMNS = ["actionNumber", "clock", "period", "teamId", "personId", "actionType", "subType", "description",
               "shotValue", "xLegacy", "yLegacy"]
ASSISTED = re.compile(r"\d+ AST\)")
FREE_THROW = re.compile(r"(\d) of (\d)")
CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")
KEYS = ["game_id", "season", "phase", "player_id"]
SHOT_COLUMNS = ["game_id", "phase", "action_number", "period", "clock", "team", "shooter", "made", "value", "x", "y",
                "assisted", "and_one", "start_kind", "seconds_in", "attempt", "since_offensive_rebound"]
TRIP_COLUMNS = ["game_id", "phase", "player_id", "team", "trips", "and_one_trips", "technical"]
# the per-game counts, in features.CREATION_RATES order: "misses_inside" is padded into "misses_inside_per100"
CREATION_COUNTS = [rate.removesuffix("_per100") for rate in CREATION_RATES]

# Fitted once (scratch/2026-10-25_creation111/fit_constants.py on these shot tables): each cell's assisted share of
# made shots over 1997-2026, the miss factors from 2016 tracking, and the trips' factor.
CELL_PRIOR: dict[str, float] = {
    "rim|putback": 0.0759, "rim|second_chance": 0.5847, "rim|transition": 0.6634, "rim|early": 0.5912,
    "rim|s7_14": 0.641, "rim|s14_20": 0.6363, "rim|s20_plus": 0.592, "paint|putback": 0.1442,
    "paint|second_chance": 0.442, "paint|transition": 0.467, "paint|early": 0.392, "paint|s7_14": 0.468,
    "paint|s14_20": 0.4431, "paint|s20_plus": 0.4023, "mid|putback": 0.4272, "mid|second_chance": 0.4946,
    "mid|transition": 0.4846, "mid|early": 0.4083, "mid|s7_14": 0.4841, "mid|s14_20": 0.4861, "mid|s20_plus": 0.4378,
    "long_two|putback": 0.6834, "long_two|second_chance": 0.6223, "long_two|transition": 0.567,
    "long_two|early": 0.513, "long_two|s7_14": 0.6083, "long_two|s14_20": 0.6319, "long_two|s20_plus": 0.5625,
    "corner_three|putback": 0.8635, "corner_three|second_chance": 0.9282, "corner_three|transition": 0.9735,
    "corner_three|early": 0.9603, "corner_three|s7_14": 0.9628, "corner_three|s14_20": 0.9671,
    "corner_three|s20_plus": 0.946, "above_break_three|putback": 0.9003, "above_break_three|second_chance": 0.843,
    "above_break_three|transition": 0.7635, "above_break_three|early": 0.6107, "above_break_three|s7_14": 0.7843,
    "above_break_three|s14_20": 0.8199, "above_break_three|s20_plus": 0.7847
}
MISS_FACTOR: dict[str, float] = {
    "rim|putback": -0.0713, "rim|second_chance": -0.3553, "rim|transition": -0.6238, "rim|early": -0.2905,
    "rim|s7_14": -0.5033, "rim|s14_20": -0.434, "rim|s20_plus": -0.4023, "paint|putback": 0.0119,
    "paint|second_chance": -0.0657, "paint|transition": 0.0261, "paint|early": 0.0845, "paint|s7_14": -0.0237,
    "paint|s14_20": -0.0618, "paint|s20_plus": -0.0371, "mid|putback": -0.0282, "mid|second_chance": -0.1517,
    "mid|transition": -0.1139, "mid|early": 0.052, "mid|s7_14": -0.0998, "mid|s14_20": -0.0542,
    "mid|s20_plus": -0.1164, "long_two|putback": -0.1067, "long_two|second_chance": -0.1108,
    "long_two|transition": -0.1244, "long_two|early": -0.0264, "long_two|s7_14": -0.0609, "long_two|s14_20": -0.1175,
    "long_two|s20_plus": -0.0901, "corner_three|putback": -0.0987, "corner_three|second_chance": -0.132,
    "corner_three|transition": -0.5603, "corner_three|early": 0.1017, "corner_three|s7_14": -0.2222,
    "corner_three|s14_20": -0.1627, "corner_three|s20_plus": -0.6145, "above_break_three|putback": -0.2201,
    "above_break_three|second_chance": -0.3036, "above_break_three|transition": -0.0056,
    "above_break_three|early": 0.2895, "above_break_three|s7_14": -0.1569, "above_break_three|s14_20": -0.1629,
    "above_break_three|s20_plus": -0.3171
}
FT_FACTOR = 0.6509


# ------------------------------------------------------------------------------------------------ one game
def _seconds(clock: str) -> float:
    found = CLOCK.match(str(clock))
    return float(found.group(1)) * 60.0 + float(found.group(2)) if found else 0.0


def _team(team_id: int, person_id: int) -> int:
    """The team a row belongs to: teamId, or personId on a team row (a team rebound, a team turnover)."""
    return int(team_id) if int(team_id) else int(person_id)


class _Possession:
    """Who has the ball, since when and how they got it; the attempts and the last offensive rebound since."""

    def __init__(self):
        self.team, self.start, self.kind, self.attempts, self.offensive_rebound = 0, 720.0, "period_start", 0, None

    def change(self, team: int, clock: float, kind: str) -> None:
        self.team, self.start, self.kind, self.attempts, self.offensive_rebound = team, clock, kind, 0, None


class _Reader:
    """One game's play-by-play read row by row (`game_shots`): one handler per action type (`HANDLERS`)."""

    def __init__(self, pbp: pd.DataFrame):
        self.action = pbp.actionType.astype(str).to_numpy()
        self.text = pbp.description.astype(str).to_numpy()
        self.sub = pbp.subType.astype(str).to_numpy()
        self.teams = [_team(t, p) for t, p in zip(pbp.teamId.to_numpy(), pbp.personId.to_numpy())]
        self.clocks = [_seconds(c) for c in pbp.clock.to_numpy()]
        self.periods, self.people = pbp.period.to_numpy(int), pbp.personId.to_numpy(int)
        self.numbers, self.values = pbp.actionNumber.to_numpy(int), pbp.shotValue.to_numpy(int)
        self.xs, self.ys = pbp.xLegacy.to_numpy(float), pbp.yLegacy.to_numpy(float)
        playing = sorted({t for t, a in zip(self.teams, self.action)
                          if a in ("Made Shot", "Missed Shot", "Free Throw") and t})
        self.other = {playing[0]: playing[-1], playing[-1]: playing[0]} if len(playing) == 2 else {}
        self.possession, self.shots, self.trips = _Possession(), [], {}
        self.miss_team, self.miss_dead, self.last_clock, self.last_make = 0, False, 720.0, None

    def read(self) -> None:
        for i, kind in enumerate(self.action):
            handler = HANDLERS.get(kind)
            if handler is not None:
                handler(self, i)
            if kind not in ("", "Substitution", "Timeout", "Instant Replay"):
                self.last_clock = self.clocks[i]

    def period(self, i: int) -> None:
        if self.sub[i] == "start":
            self.possession.change(0, self.clocks[i], "period_start")
            self.miss_team, self.last_make = 0, None

    def shot(self, i: int) -> None:
        team, clock, person, possession = self.teams[i], self.clocks[i], int(self.people[i]), self.possession
        if team != possession.team:
            # the first shot of a period keeps the period start; any other shot by the team without the ball
            # starts a possession at the last event's clock
            start, kind = (possession.start, possession.kind) if possession.team == 0 else (self.last_clock, "other")
            possession.change(team, start, kind)
        possession.attempts += 1
        made = self.action[i] == "Made Shot"
        since = possession.offensive_rebound - clock if possession.offensive_rebound is not None else np.nan
        self.shots.append({"action_number": int(self.numbers[i]), "period": int(self.periods[i]), "clock": clock,
                           "team": team, "shooter": person, "made": int(made), "value": int(self.values[i]),
                           "x": float(self.xs[i]), "y": float(self.ys[i]),
                           "assisted": int(made and bool(ASSISTED.search(self.text[i]))), "and_one": 0,
                           "start_kind": possession.kind, "seconds_in": possession.start - clock,
                           "attempt": possession.attempts, "since_offensive_rebound": since})
        if made:
            self.last_make = (len(self.shots) - 1, person, int(self.periods[i]), clock)
            possession.change(self.other.get(team, 0), clock, "made_fg")
        else:
            self.miss_team, self.miss_dead = team, False

    def rebound(self, i: int) -> None:
        if self.miss_dead or not self.miss_team:
            return
        team, clock = self.teams[i], self.clocks[i]
        if team == self.miss_team:
            if self.possession.team != team:      # a missed and-one free throw, after the make handed the ball over
                self.possession.change(team, clock, "other")
            self.possession.offensive_rebound = clock
        else:
            self.possession.change(team, clock, "dreb")
        self.miss_team = 0

    def turnover(self, i: int) -> None:
        stolen = i + 1 < len(self.action) and self.action[i + 1] == "" and "STEAL" in self.text[i + 1]
        self.possession.change(self.other.get(self.teams[i], 0), self.clocks[i], "steal" if stolen else "dead_tov")

    def free_throw(self, i: int) -> None:
        team, clock, shooter = self.teams[i], self.clocks[i], int(self.people[i])
        entry = self.trips.setdefault(shooter, {"team": team, "trips": 0, "and_one_trips": 0, "technical": 0})
        if "Technical" in self.sub[i]:
            entry["technical"] += 1
            return
        found = FREE_THROW.search(self.sub[i])
        number, of = (int(found.group(1)), int(found.group(2))) if found else (1, 1)
        if number == 1:
            entry["trips"] += 1
            if of == 1 and self.last_make is not None and self.last_make[1:] == (shooter, int(self.periods[i]), clock):
                self.shots[self.last_make[0]]["and_one"] = 1
                entry["and_one_trips"] += 1
        missed = self.text[i].startswith("MISS")
        if number == of and not missed:
            self.possession.change(self.other.get(team, 0), clock, "made_ft")
        elif missed:
            # a miss before the trip's last free throw is followed by the feed's dead-ball team rebound
            self.miss_team, self.miss_dead = team, number != of


HANDLERS = {"period": _Reader.period, "Made Shot": _Reader.shot, "Missed Shot": _Reader.shot,
            "Rebound": _Reader.rebound, "Turnover": _Reader.turnover, "Free Throw": _Reader.free_throw}


def game_shots(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One game's play-by-play, rows in the file's order -> (one row per field-goal attempt, `SHOT_COLUMNS` but the
    game and phase; one row per player with a free throw, his trips, and-one trips and technical free throws).

    The possession follows the ball: a defensive rebound, a turnover (a steal when a STEAL row comes right after it),
    a make (a made last free throw) and a period start each hand it over; an offensive rebound keeps it.  A rebound
    after a free throw that is not the trip's last (the feed's dead-ball team rebound) is passed over.  A shot by the
    team without the ball starts a possession at the last event's clock, kind "other".  An and-one is a trip of one
    free throw by the maker of a shot, in the same period at the same clock.
    """
    reader = _Reader(pbp)
    reader.read()
    shot_table = pd.DataFrame(reader.shots, columns=SHOT_COLUMNS[2:])
    trip_table = pd.DataFrame([{"player_id": k, **v} for k, v in reader.trips.items()],
                              columns=["player_id", "team", "trips", "and_one_trips", "technical"])
    return shot_table, trip_table


def add_cells(shots: pd.DataFrame) -> pd.DataFrame:
    """Each shot's spot, context and cell (`spot|context`), and its spot group."""
    x, y = shots.x.to_numpy(float), shots.y.to_numpy(float)
    three = shots.value.to_numpy() == 3
    corner = three & (np.abs(x) >= CORNER_X) & (y <= CORNER_Y)
    feet = np.hypot(x, y) / 10.0
    spot = np.select([corner, three, feet <= 4, feet <= 10, feet <= 16], [*SPOTS[4:], *SPOTS[:3]], SPOTS[3])
    since = shots.since_offensive_rebound.to_numpy(float)
    after = ~np.isnan(since)
    seconds = shots.seconds_in.to_numpy(float)
    live = np.isin(shots.start_kind.to_numpy(), ["dreb", "steal"])
    first = shots.attempt.to_numpy() == 1
    context = np.select([after & (since <= PUTBACK_SECONDS), after | ~first, live & (seconds <= EARLY_SECONDS),
                         seconds <= EARLY_SECONDS, seconds <= 14, seconds <= 20], CONTEXTS[:6], CONTEXTS[6])
    spot = pd.Series(spot, index=shots.index)
    return shots.assign(spot=spot, context=context, cell=spot + "|" + context, group=spot.map(SPOT_GROUPS))


# ---------------------------------------------------------------------------------------------- one season
def season_shots(season: int, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(every field-goal attempt of the season with its cell, every player-game's free-throw trips), both phases."""
    folder = raw_dir(cfg) / "pbp" / str(season)
    shots, trips = [], []
    for phase, prefix in GAME_PREFIX.items():
        for path in sorted(folder.glob(f"{prefix}*.parquet")):
            game_shot, game_trip = game_shots(pd.read_parquet(path, columns=PBP_COLUMNS))
            shots.append(game_shot.assign(game_id=path.stem, phase=phase))
            trips.append(game_trip.assign(game_id=path.stem, phase=phase))
    if not shots:
        raise FileNotFoundError(f"no play-by-play for {season} in {folder}; run `openrapm data ingest {season}`")
    return (add_cells(pd.concat(shots, ignore_index=True)[SHOT_COLUMNS]),
            pd.concat(trips, ignore_index=True)[TRIP_COLUMNS])


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def cell_shares(shots: pd.DataFrame, prior: dict | None = None) -> pd.Series:
    """The assisted share of made shots by cell, padded `CELL_PAD` makes toward `prior` (`CELL_PRIOR` by default)."""
    prior = CELL_PRIOR if prior is None else prior
    made = shots[shots.made == 1].groupby("cell").assisted.agg(["sum", "size"])
    cells = pd.Index([f"{spot}|{context}" for spot in SPOTS for context in CONTEXTS])
    made = made.reindex(cells, fill_value=0)
    level = made["sum"].sum() / max(made["size"].sum(), 1)
    toward = pd.Series(prior, dtype=float).reindex(cells).fillna(level)
    return (made["sum"] + CELL_PAD * toward) / (made["size"] + CELL_PAD)


def season_creation_counts(shots: pd.DataFrame, trips: pd.DataFrame, season: int) -> pd.DataFrame:
    """`KEYS` + `CREATION_COUNTS`, one row per game, phase and player with a shot or a free throw."""
    share = cell_shares(shots)
    shots = shots.assign(p_cell=shots.cell.map(share).to_numpy(float))
    shots["p_miss"] = expit(logit(shots.p_cell) + shots.cell.map(MISS_FACTOR).fillna(0.0).to_numpy(float))
    rim = (shots.made == 0) & (shots.spot == "rim")
    trip_chance = float(shots.loc[rim, "p_miss"].mean()) * FT_FACTOR if rim.any() else 0.0
    keys = ["game_id", "phase", "shooter"]
    made, missed = shots[shots.made == 1], shots[shots.made == 0]
    parts = []
    for group in GROUPS:
        m, x = made[made.group == group], missed[missed.group == group]
        parts += [m.groupby(keys).assisted.sum().rename(f"assisted_makes_{group}"),
                  m.groupby(keys).p_cell.sum().rename(f"expected_assisted_makes_{group}"),
                  x.groupby(keys).size().rename(f"misses_{group}").astype(float),
                  x.groupby(keys).p_miss.sum().rename(f"expected_passed_misses_{group}")]
    t = trips.rename(columns={"player_id": "shooter"}).set_index(keys)
    without = (t.trips - t.and_one_trips).clip(lower=0).groupby(level=keys).sum().astype(float)
    parts += [without.rename("trips_without_make"), (without * trip_chance).rename("expected_passed_trips")]
    out = pd.concat(parts, axis=1).fillna(0.0).reset_index().rename(columns={"shooter": "player_id"})
    out["season"] = int(season)
    return out[[*KEYS, *CREATION_COUNTS]]


def _paths(season: int, cfg: dict):
    folder = resolve(cfg, "event_counts")
    return folder / f"{int(season)}_creation_shots.parquet", folder / f"{int(season)}_creation_trips.parquet", \
        folder / f"{int(season)}_creation.parquet"


def build_creation_counts(cfg: dict, seasons, force: bool = False) -> None:
    """Read and cache every season asked for (a cached season is read, unless `force`); one line per season."""
    for season in seasons:
        started = time.time()
        shots_path, trips_path, counts_path = _paths(season, cfg)
        if force or not shots_path.exists():
            shots_path.parent.mkdir(parents=True, exist_ok=True)
            shots, trips = season_shots(season, cfg)
            shots.to_parquet(shots_path, index=False)
            trips.to_parquet(trips_path, index=False)
        if MISS_FACTOR and (force or not counts_path.exists()):
            season_creation_counts(pd.read_parquet(shots_path), pd.read_parquet(trips_path), season).to_parquet(
                counts_path, index=False)
        print(f"  {season}: self-creation counts ({time.time() - started:.0f}s)", flush=True)


def load_creation_counts(season: int, cfg: dict) -> pd.DataFrame:
    """The season's cached counts (`KEYS` + `CREATION_COUNTS`); refused with the command to run when not built."""
    path = _paths(season, cfg)[2]
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run `openrapm data events {int(season)}` (self-creation's "
                                f"counts, data/creation_counts.py)")
    counts = pd.read_parquet(path)
    return counts.assign(game_id=counts.game_id.astype(str))
