"""One game's play-by-play -> possessions, each with the ten players on the court -> stints.

RAPM is fitted on stints, and this module is where they come from.  It reads one game's play-by-play
(stats.nba.com PlayByPlayV3: one row per event, with its period, clock, team, player, action type and a
written description) and its box score, and walks the rows in order, one period at a time, keeping track
of who has the ball, who is on the court, and what each possession has done so far.

Words used below
    possession  one team's turn with the ball: from when it gains the ball until the other team gains it
                or the period ends.  It is credited to the ten players on the court when it ENDS.
    offence     the team with the ball (the column is spelt `offense`, like every other table)
    lineup      one team's five players on the court, stored sorted by player id
    valid       both lineups are five known players.  Possessions with an unknown lineup are counted
                (diagnostic `invalid_possessions`) but never reach a stint.
    stint       consecutive valid possessions of one game, in one period, with the same ten players on
                the court.  A substitution or the end of a period starts a new one.
    attempt     an attempt event: a field-goal attempt, or a free-throw trip.  Three kinds of trip are not
                attempts: an and-one; a shooting-foul trip after a missed field goal the feed also recorded;
                a flagrant, clear-path or away-from-play trip (GameParser._start_trip says why).
                design.POSS_COUNTERS says why a trip counts.
    trip        the free throws awarded for one foul ("Free Throw 1 of 2", "Free Throw 2 of 2")
    and-one     a made field goal plus the single free throw for the foul on it

How a possession ends
    made field goal       unless the shooting team's free throw follows at the same clock (an and-one):
                          then it ends on that free throw if it is made; a missed one is a live rebound
    defensive rebound     the rebounding team has the ball
    turnover
    made last free throw  of a trip ("2 of 2" made).  A missed last free throw is a live rebound.
    end of period
    forced switch         the other team acts on offence (a shot, a turnover, a free throw) with no
                          recorded change of possession: the feed missed a row, so the possession in
                          progress is closed there (diagnostic `forced_switches`)
    jump ball             won by the team without the ball, after the team with it has acted
                          (diagnostic `jump_switches`)
An offensive rebound continues the possession.  Technical free throws neither end nor start one: their
points go to the shooting team's possession in progress when it has the ball and has acted; otherwise
they are held and added to its next recorded possession (or, if the period ends first, its last one).
After a flagrant or clear-path trip the shooting team keeps the ball, so those trips do not end the
possession either.

How the parser decides who is on the court
    Period starters: the players who act in the period before being substituted in.  When that does not
    give exactly five for a team, in this order:
      1. period 1 only: when every player found is among the box score's first five rows (V3 lists the
         starters first), those five;
      2. any period: the period box score (BoxScoreTraditionalV3, range_type=2) says who played in the
         period.  When more than five were found and exactly five of them played, those five; else those
         who played, minus those who came on by substitution, plus those found, when that makes five.
    If that still does not give five, the team's lineup is unknown for the whole period.
    Substitutions: a V3 substitution row carries only the outgoing player's id.  The incoming player is
    found by name ("SUB: Jones FOR Smith") among the team's players not on the court: first by the name as
    written in this game's play-by-play, then by the box score's family name, then by the closest
    spelling.  When two players fit, the first of them to appear in a later row of the period is taken.
    A substitution that cannot be resolved makes the team's lineup unknown until a later substitution
    brings it back to five known players.

Example (the synthetic game in tests/test_stints.py)
    12:00  jump ball, tipped to home          home has the ball
    11:40  home 1 makes a two                 possession 1 ends: home, 2 points, reason "made_fg"
    11:20  away 11 misses a two
    11:18  away 12 offensive rebound          the possession continues: reb_cont = 1
    11:10  away 12 makes a three              possession 2 ends: away, 3 points, att = 2
    10:50  home 2 makes a layup, is fouled and makes the free throw
                                              possession 3 ends at the free throw: home, 3 points, att = 1
                                              (the and-one trip is not a new attempt), reason "made_ft"
    10:50  away substitution, 16 for 11       the next possession starts a new stint
The first three possessions share the same ten players, so they are one stint: home 2 possessions and
5 points, away 1 possession and 3 points.

What a possession carries
    its points, both lineups, its start clock and score, the reason it ended, and two counter families:
    design.POSS_COUNTERS     per-possession counts: shooting by distance bucket, free throws, attempts,
                             the rebound chain, the first attempt's bucket and points, expected free-throw
                             makes.  The two identities they obey are written beside them in design.py.
    design.SHOOTER_COUNTERS  per-shooter counts for the possession: attempts and the league's make
                             probability summed over them.  The stint collapse files them under the
                             shooter's place in the sorted lineup (design.SLOT_COUNTERS).
The parser also keeps one table for the whole game, `shots` (columns SHOT_TABLE): per shooter his two-
and three-point attempts, makes and summed league make probability, every attempt counted.  It becomes
the per-game shots file that data/shots.load_shots reads.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from ..design import AWAY_SLOTS, HOME_SLOTS, POSS_COUNTERS, SHOOTER_COUNTERS, SLOT_COUNTERS
from .shot_curves import FALLBACK

CLOCK_RE = re.compile(r"PT(\d+)M(\d+(?:\.\d+)?)S")
SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)$")
TIP_RE = re.compile(r"Tip to\s+(.+)$")
FT_RE = re.compile(r"Free Throw(?: (\w[\w ]*?))? (\d) of (\d)")
# action types that say nothing about who started a period on the court
IGNORED = frozenset({"", "Timeout", "Instant Replay", "Violation", "Ejection"})
# the action types whose description starts with the acting player's name as written that day
NAMED_ACTIONS = ("Made Shot", "Missed Shot", "Rebound", "Turnover", "Foul", "Free Throw", "")
# rows that end the search for a made shot's and-one free throw
AND_ONE_STOPS = ("Made Shot", "Missed Shot", "Turnover", "period")
RIM_FT = 3.0          # feet; a two at or inside this distance is in the "rim" bucket, beyond it "mid"
SHOT_TABLE = ("fg2a", "fg2m", "xl2", "fg3a", "fg3m", "xl3")     # the per-game shooter table
SLOT_COLUMNS = tuple(f"{counter}_{side}" for counter in SLOT_COUNTERS for side in ("h", "a"))

# Expected free throws replace the ones that actually went in, which is the first and cleanest piece of
# the luck adjustment: whether a free throw drops is close to pure noise, and it is a fifth of all scoring.
# `xftm` is the sum over the possession's attempts of the SHOOTER's own leave-one-game-out padded
# percentage -- the shooter's own rate, not the lineup's, because free-throw percentage belongs to one
# identified player and the defence barely touches it.  This is the rate when no season totals are given.
FT_LEAGUE_DEFAULT = 0.75

# A description token that ends the player's name: a number, a parenthesis, or one of these whole words.
STOP_TOKEN = re.compile(r"^(\d|\(|3PT|[A-Z](\.[A-Z])*\.FOUL$|T\.Foul$|OFF\.|FLAGRANT|"
                        r"(REBOUND|Free|Turnover|STEAL|BLOCK|Bad|Lost|Traveling|Offensive|Out|Step|Shot|Jump|"
                        r"Layup|Dunk|Hook|Tip|Running|Driving|Turnaround|Fadeaway|Floating|Pullup|Putback|Alley|"
                        r"Finger|Reverse|Cutting|Slam|Bank|Violation|Personal|Shooting|Loose|Away|Technical|"
                        r"Flagrant|No|Kicked|Def\.|Foul|Double|Inbound|Lane|Backcourt|Palming|Illegal|Discontinue|"
                        r"Punched|Poss|Hanging|Taunting|Excess|Delay|Jumper|Regular|Shooting|Clear|Transition|"
                        r"Elbow|Charge|Blocking|Defensive|Inbound)$)", re.IGNORECASE)


def parse_clock(text: str) -> float:
    """Seconds left in the period.  Example: "PT06M17.00S" -> 377.0."""
    match = CLOCK_RE.match(str(text))
    if not match:
        return np.nan
    return int(match.group(1)) * 60 + float(match.group(2))


def regulation_remaining(period: int, clock: float) -> float:
    """Seconds of regulation left: (4 - period) x 720 + clock, and 0 in overtime."""
    return (4 - period) * 720.0 + clock if period <= 4 else 0.0


def clean_events(pbp: pd.DataFrame) -> pd.DataFrame:
    """The play-by-play with stripped text, numeric ids and scores, and two new columns: `clock_s` (seconds
    left in the period) and `shot_dist` (feet; -1 when the season does not carry shotDistance, which puts
    every two in the "mid" bucket)."""
    events = pbp.copy()
    for column in ("actionType", "subType", "description", "clock", "playerName"):
        events[column] = events[column].astype(str).str.strip()      # some seasons pad these with spaces
    events["clock_s"] = events["clock"].map(parse_clock)
    for column in ("teamId", "personId", "period", "shotValue"):
        events[column] = pd.to_numeric(events[column], errors="coerce").fillna(0).astype(int)
    distance = events["shotDistance"] if "shotDistance" in events.columns else -1.0
    events["shot_dist"] = pd.to_numeric(pd.Series(distance, index=events.index),
                                        errors="coerce").fillna(-1.0).astype(float)
    for column in ("scoreHome", "scoreAway"):
        events[column] = pd.to_numeric(events[column], errors="coerce").fillna(0).astype(int)
    return events.reset_index(drop=True)


def free_throw_made(description: str, score_total: int) -> bool:
    """Made free throws carry "(N PTS)"; some misses have a blank description with no "MISS" and no score."""
    return ("PTS" in description) or (not description.startswith("MISS") and score_total > 0)


def _name_key(name: str) -> str:
    """A name reduced for matching: accents dropped, lower case, letters only.  "Schröder" -> "schroder"."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", text.lower())


def _leading_name(descriptions) -> str:
    """The player's name as written in this game's play-by-play: the words before the action keyword, the
    most common such prefix over his rows.  Example: "Nene 2' Layup (2 PTS)" -> "Nene"."""
    names = []
    for description in descriptions:
        words = []
        for token in str(description).split():
            if STOP_TOKEN.match(token):
                break
            words.append(token)
        if words:
            names.append(" ".join(words))
    if not names:
        return ""
    return Counter(names).most_common(1)[0][0]


@dataclass
class HeldFreeThrows:
    """Technical free throws a team shot while it did not have the ball, held until it does.

    The points go to the team's next possession (or, when the period ends first, its last one); the
    attempts, makes and expected makes travel with them so the counters stay complete."""
    points: int = 0
    attempts: int = 0
    makes: int = 0
    expected_makes: float = 0.0


@dataclass
class GameTally:
    """What carries from one period to the next: the score, the held free throws, the possession records."""
    score: dict                                   # team -> points so far
    held: dict                                    # team -> HeldFreeThrows
    records: list = field(default_factory=list)   # one dict per recorded possession, in order


@dataclass
class Possession:
    """The possession being built: who has the ball, its points and counters, and the rebound chain's state."""
    offense: int | None            # the team with the ball; None until an action in the period says who
    start_clock: float
    start_score: dict              # team -> points when the possession began
    active: bool = False           # the offence has acted; a possession that never became active is not recorded
    points: int = 0
    and1: bool = False             # a made field goal is waiting for its and-one free throw
    counters: dict = field(default_factory=lambda: dict.fromkeys(POSS_COUNTERS, 0))
    shooters: dict = field(default_factory=dict)   # player id -> SHOOTER_COUNTERS
    # the rebound chain (reb_chance, reb_cont, cont_dead, att_retained: design.POSS_COUNTERS)
    pending_chance: int = 0        # a miss is waiting to be resolved into a rebound
    and1_used: bool = False        # the possession's and-one free throw has been seen
    cont_team: bool = False        # the last continuation was a TEAM offensive rebound (teamId 0)
    att_since_cont: int = 0        # attempts since that continuation
    retain_pending: int = 0        # a flagrant, clear-path or away-from-play trip: the next attempt is att_retained


@dataclass
class PeriodState:
    """One period being parsed: its rows, who is on the court, the possession in progress, and the tally."""
    period: int
    rows: list
    starters: dict                 # team -> the five period starters, or None when they could not be found
    on: dict                       # team -> set of the player ids on the court now
    valid: dict                    # team -> True while the players on the court are five known players
    possession: Possession
    tally: GameTally
    last_foul_shooting: bool = False   # the period's last foul was a shooting foul


class GameParser:
    """Turn one game's V3 play-by-play into possessions with lineups, and those into stints."""

    def __init__(self, pbp: pd.DataFrame, box: pd.DataFrame, home_id: int, away_id: int, game_id: str,
                 period_box_fetcher=None, gt_rule=None, ft_rates=None, curve=None):
        """`period_box_fetcher(game_id, period)` returns a period box score; `gt_rule` is config `garbage_time`;
        `ft_rates` = (season free-throw totals by player, league rate, padding) for the expected makes;
        `curve` is the season's shot_curves.ShotCurve (None: the flat FALLBACK rates)."""
        self.game_id = game_id
        self.curve = curve
        self.shots = {}                  # player id -> per-game shooter totals (SHOT_TABLE), every attempt
        self.home, self.away = int(home_id), int(away_id)
        self.teams = (self.home, self.away)
        self.fetch_period_box = period_box_fetcher
        self.gt_rule = gt_rule or {}
        self.diag = dict(unresolved_subs=0, sub_out_not_on_floor=0, forced_switches=0, jump_switches=0,
                         periods_from_box=0, periods_starters_failed=0, invalid_possessions=0, n_periods=0)
        self._read_box(box)
        self.ev = clean_events(pbp)
        self._ft = self._game_free_throws(ft_rates)
        self._read_day_names()

    # ---------------------------------------------------------------- reading the game
    def _read_box(self, box: pd.DataFrame) -> None:
        """Roster, names, starters and points from the box score."""
        self.roster = {}                 # player id -> team
        self.names = {team: {} for team in self.teams}   # team -> family-name key -> [player id]
        self.name_i = {}                 # player id -> key of "F. Family"
        for row in box.itertuples(index=False):
            player, team = int(row.personId), int(row.teamId)
            self.roster[player] = team
            self.names.setdefault(team, {}).setdefault(_name_key(row.familyName), []).append(player)
            self.name_i[player] = _name_key(row.nameI)
        # V3 lists each team's starters as its first five rows (the position field is not a starter flag)
        team_of_row = box.teamId.astype(int)
        self.box_starters = {team: set(box[team_of_row == team]["personId"].astype(int).head(5).tolist())
                             for team in self.teams}
        has_points = "points" in box.columns
        self.box_pts = {team: float(box.loc[team_of_row == team, "points"].astype(float).sum()) if has_points
                        else np.nan for team in self.teams}

    def _game_free_throws(self, ft_rates):
        """Season free-throw totals plus this game's own makes and attempts per shooter, so ft_prob can take
        the game out: a player's own makes never price the possessions he is being scored on."""
        if ft_rates is None:
            return None
        totals, league, padding = ft_rates
        made_in_game, attempts_in_game = {}, {}
        for row in self.ev.itertuples(index=False):
            if row.actionType != "Free Throw":
                continue
            player = int(row.personId)
            made = free_throw_made(str(row.description), row.scoreHome + row.scoreAway)
            attempts_in_game[player] = attempts_in_game.get(player, 0) + 1
            made_in_game[player] = made_in_game.get(player, 0) + int(made)
        return totals, float(league), float(padding), made_in_game, attempts_in_game

    def _read_day_names(self) -> None:
        """Names as written in this game's play-by-play.  The box score carries players' current names, the
        play-by-play the names of the day (Nene vs Hilario, Kanter vs Freedom), and substitutions use the latter."""
        self.pbp_names = {team: {} for team in self.teams}   # team -> name key -> {player id}
        self.day_names = {}                                  # player id -> name as written
        for player, descriptions in self._descriptions_by_player().items():
            name = _leading_name(descriptions)
            if name:
                self._add_day_name(player, name)

    def _descriptions_by_player(self) -> dict:
        """player id -> the descriptions of his named rows, with a leading "MISS " removed."""
        descriptions = {}
        for player, action, text in zip(self.ev["personId"], self.ev["actionType"], self.ev["description"]):
            player = int(player)
            if self.roster.get(player) in self.teams and action in NAMED_ACTIONS:
                descriptions.setdefault(player, []).append(str(text).removeprefix("MISS "))
        return descriptions

    def _add_day_name(self, player: int, name: str) -> None:
        """File the name under its full key and, when different, under its last word ("Jones")."""
        team = self.roster[player]
        self.day_names[player] = name
        self.pbp_names[team].setdefault(_name_key(name), set()).add(player)
        last = _name_key(name.split()[-1])
        if last != _name_key(name):
            self.pbp_names[team].setdefault(last, set()).add(player)

    # ---------------------------------------------------------------- small lookups
    def _row_team(self, row) -> int:
        """The team a row belongs to: its teamId, or, on team rows (team rebounds), its personId."""
        if row.teamId in self.teams:
            return row.teamId
        if row.personId in self.teams:
            return row.personId
        return 0

    def _other(self, team):
        return self.away if team == self.home else self.home

    def _player_team(self, player):
        return self.roster.get(player, 0)

    def shot_p(self, three: bool, dist: float, desc: str) -> float:
        """The LEAGUE make probability of this attempt at this distance this season (shot_curves)."""
        if self.curve is None:
            return FALLBACK[bool(three)]
        return self.curve.prob(dist, three, desc)

    def ft_prob(self, pid: int) -> float:
        """The shooter's padded free-throw percentage with this game's own attempts removed:
        (makes + padding x league) / (attempts + padding), season totals minus this game."""
        if self._ft is None:
            return FT_LEAGUE_DEFAULT
        totals, league, padding, made_in_game, attempts_in_game = self._ft
        made, attempts = totals.get(int(pid), (0.0, 0.0))
        made -= made_in_game.get(int(pid), 0)
        attempts -= attempts_in_game.get(int(pid), 0)
        return (max(made, 0.0) + padding * league) / (max(attempts, 0.0) + padding)

    # ---------------------------------------------------------------- who is on the court
    def _resolve_in(self, team, in_name, on_court, index, period_rows):
        """The incoming player of 'SUB: <in_name> FOR <out>' on `team`, or None.  The rows after `index` in
        `period_rows` break a tie between two players of the same name."""
        key = _name_key(in_name)
        candidates = [player for player in self._name_candidates(team, key) if player not in on_court]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1 and "." in in_name:
            narrowed = [player for player in candidates if self.name_i.get(player, "") == key]
            if len(narrowed) == 1:
                return narrowed[0]
        if not candidates:
            candidates = self._closest_name(team, key, on_court)
        if len(candidates) > 1:
            return next((row.personId for row in period_rows[index + 1:] if row.personId in candidates), None)
        return candidates[0] if candidates else None

    def _name_candidates(self, team, key) -> list:
        """Players whose name matches: the day's name, else the family name, else a suffix match either way."""
        candidates = list(self.pbp_names.get(team, {}).get(key, []))
        if not candidates:
            candidates = list(self.names.get(team, {}).get(key, []))
        if not candidates:
            for family, players in self.names.get(team, {}).items():
                if key.endswith(family) or family.endswith(key):
                    candidates.extend(players)
        return candidates

    def _closest_name(self, team, key, on_court) -> list:
        """No name match (spelling drift): the closest family name not on the court, if close enough (0.75)."""
        pool = {family: players for family, players in self.names.get(team, {}).items()
                if any(player not in on_court for player in players)}
        best = difflib.get_close_matches(key, list(pool), n=1, cutoff=0.75)
        return [player for player in pool[best[0]] if player not in on_court] if best else []

    def _starters(self, period, rows) -> dict:
        """team -> the period's five starters, or None (the team's lineup is unknown all period)."""
        found = self._starters_from_pbp(rows)
        return {team: self._team_starters(team, period, rows, found[team]) for team in self.teams}

    def _starters_from_pbp(self, rows) -> dict:
        """team -> the players who act in the period before anyone substitutes them in, in order of appearance."""
        starters = {team: [] for team in self.teams}
        entered = {team: set() for team in self.teams}
        for row in rows:
            if row.actionType == "Substitution":
                self._starter_substitution(row, starters, entered)
            elif self._acts_on_court(row):
                self._add_starter(row.personId, starters, entered)
        return starters

    def _starter_substitution(self, row, starters, entered) -> None:
        """The outgoing player was on the court, so he started unless he came on earlier; note who came on."""
        team = self._add_starter(row.personId, starters, entered)
        match = SUB_RE.match(str(row.description))
        if match and team in self.teams:
            incoming = self._resolve_in(team, match.group(1), set(starters[team]) | entered[team], 0, [])
            if incoming is not None:
                entered[team].add(incoming)

    def _add_starter(self, player, starters, entered) -> int:
        """Count `player` as a starter if no substitution has brought him on yet; return his team."""
        team = self._player_team(player)
        if team in self.teams and player not in entered[team] and player not in starters[team]:
            starters[team].append(player)
        return team

    @staticmethod
    def _acts_on_court(row) -> bool:
        """Rows that prove a player is on the court: not the ignored kinds, not technical fouls or free throws
        (a technical can be called on, and shot by, a player on the bench)."""
        if row.actionType in IGNORED or row.actionType == "period":
            return False
        return not (row.actionType in ("Foul", "Free Throw") and "Technical" in str(row.subType))

    def _team_starters(self, team, period, rows, found):
        """Five from the play-by-play; else, in period 1, the box score's five; else the period box score."""
        if len(found) == 5:
            return set(found)
        box_five = self.box_starters[team]
        if period == 1 and len(box_five) == 5 and set(found) <= box_five:
            return set(box_five)
        if self.fetch_period_box is not None:
            from_box = self._starters_from_period_box(team, period, rows, found)
            if from_box is not None:
                return from_box
        self.diag["periods_starters_failed"] += 1
        return None

    def _starters_from_period_box(self, team, period, rows, found):
        """From the players with minutes in the period: when more than five were found and exactly five of
        them played, those five; else those who played, minus those who came on by substitution, plus
        `found`, when that makes five."""
        self.diag["periods_from_box"] += 1
        box = self.fetch_period_box(self.game_id, period)
        played = {int(player) for player, player_team, minutes in zip(box.personId, box.teamId, box.minutes)
                  if int(player_team) == team and str(minutes) not in ("", "0:00", "None", "nan")}
        if len(found) > 5 and len(set(found) & played) == 5:
            return set(found) & played
        candidates = (played - self._came_on(team, rows)) | set(found)
        return candidates if len(candidates) == 5 else None

    def _came_on(self, team, rows) -> set:
        """Players of `team` brought on by a substitution before they left the court in the period."""
        came_on, went_off = set(), set()
        for row in rows:
            if row.actionType != "Substitution":
                continue
            own_team = self._player_team(row.personId) == team
            if own_team and row.personId not in came_on:
                went_off.add(row.personId)
            match = SUB_RE.match(str(row.description))
            incoming = self._resolve_in(team, match.group(1), set(), 0, []) if match and own_team else None
            if incoming is not None and incoming not in went_off:
                came_on.add(incoming)
        return came_on

    # ---------------------------------------------------------------- possessions
    def possessions(self) -> pd.DataFrame:
        """Every recorded possession of the game, in order: one row each, with its lineups and counters.

        Each row goes to the handler for its action type (HANDLERS below); an action type without one has no
        effect on the possession."""
        tally = GameTally(score={self.home: 0, self.away: 0},
                          held={self.home: HeldFreeThrows(), self.away: HeldFreeThrows()})
        for period, group in self.ev.groupby("period", sort=True):
            state = self._start_period(period, list(group.itertuples(index=False)), tally)
            for index, row in enumerate(state.rows):
                handler = self.HANDLERS.get(row.actionType)
                if handler is not None:
                    handler(self, state, index, row)
        return pd.DataFrame(tally.records)

    def _start_period(self, period, rows, tally: GameTally) -> PeriodState:
        """The period's starters on the court, and an empty possession with nobody yet on offence."""
        self.diag["n_periods"] += 1
        starters = self._starters(period, rows)
        start_clock = 720.0 if period <= 4 else 300.0
        return PeriodState(
            period=period, rows=rows, starters=starters,
            on={team: (set(starters[team]) if starters[team] is not None else set()) for team in self.teams},
            valid={team: starters[team] is not None for team in self.teams},
            possession=Possession(offense=None, start_clock=start_clock, start_score=dict(tally.score)),
            tally=tally)

    def _close(self, state: PeriodState, reason: str, end_clock: float, next_offense) -> None:
        """End the possession in progress: record it if its offence acted, then start the next one."""
        possession = state.possession
        self._settle_last_continuation(possession, reason)
        if possession.offense is not None and possession.active:
            state.tally.records.append(self._record(state, reason, end_clock))
        state.possession = Possession(offense=next_offense, start_clock=end_clock,
                                      start_score=dict(state.tally.score))

    @staticmethod
    def _settle_last_continuation(possession: Possession, reason: str) -> None:
        """A continuation (offensive rebound) followed by no attempt is `cont_dead`, except a TEAM rebound
        with the period expiring: that continued nothing and was no contested chance, so it is taken back
        out (`reb_drop`).  About 5% of continuations; dropping them takes the possession reconciliation
        from 0.7% to 0.004% on 2024 (FINDINGS.md section 14)."""
        counters = possession.counters
        if counters["reb_cont"] <= 0 or possession.att_since_cont != 0:
            return
        if reason != "period_end" or not possession.cont_team:
            counters["cont_dead"] += 1
            return
        counters["reb_cont"] -= 1
        counters["reb_chance"] -= 1
        counters["oreb_t"] -= 1
        counters["reb_drop"] += 1
        if counters["att"] == 1:
            counters["chance1"] -= 1          # it was the first attempt's chance

    def _record(self, state: PeriodState, reason: str, end_clock: float) -> dict:
        """The possession's row, with the offence's held technical free throws added to it."""
        possession = state.possession
        counters = possession.counters
        held = state.tally.held[possession.offense]
        points = possession.points + held.points
        counters["pts_tech"] += held.points
        counters["fta_tech"] += held.attempts
        counters["ftm_tech"] += held.makes
        counters["xftm_tech"] += held.expected_makes
        state.tally.held[possession.offense] = HeldFreeThrows()
        valid = self._lineups_known(state)
        if not valid:
            self.diag["invalid_possessions"] += 1
        return dict(period=state.period, offense=possession.offense, points=points,
                    start_clock=possession.start_clock, end_clock=end_clock,
                    score_home=possession.start_score[self.home], score_away=possession.start_score[self.away],
                    home_lineup=tuple(sorted(state.on[self.home])), away_lineup=tuple(sorted(state.on[self.away])),
                    valid=valid, reason=reason, shooters=dict(possession.shooters), **counters)

    def _lineups_known(self, state: PeriodState) -> bool:
        """Both teams have five known players on the court."""
        return (state.valid[self.home] and state.valid[self.away]
                and len(state.on[self.home]) == 5 and len(state.on[self.away]) == 5)

    def _ensure_offense(self, state: PeriodState, team: int, row) -> None:
        """An offensive action by `team`: if the other team had the ball, close its possession (a forced
        switch).  Read `state.possession` again after calling this: it may be a new possession."""
        possession = state.possession
        if possession.offense is None:
            possession.offense = team
        elif possession.offense != team:
            self.diag["forced_switches"] += 1
            self._close(state, "forced", row.clock_s, team)
        state.possession.active = True

    @staticmethod
    def _note_attempt(possession: Possession, bucket: str) -> None:
        """Register an attempt event and, if it is the possession's first, its bucket (rim, mid, thr, ft).

        Only the FIRST attempt is recorded by type: every later one exists only because a rebound went the
        offence's way, and conditioning on that is exactly the luck the expected-points target removes.
        A new attempt while a miss is still unresolved means the offence kept the ball with no rebound row
        in the feed (a block or a tip): a real continuation, `oreb_unattr`, about one possession in 100,000."""
        counters = possession.counters
        if possession.retain_pending and counters["att"] > 0:
            counters["att_retained"] += 1       # a retention foul, not a rebound, bought this one
            possession.retain_pending = 0
        elif possession.pending_chance:
            counters["reb_chance"] += 1
            counters["reb_cont"] += 1
            counters["oreb_unattr"] += 1
            if counters["att"] == 1:
                counters["chance1"] += 1
            possession.pending_chance = 0
            possession.cont_team = False
            possession.att_since_cont = 0
        if counters["att"] == 0:
            counters[f"att1_{bucket}"] += 1
        counters["att"] += 1
        possession.att_since_cont += 1

    # ---------------------------------------------------------------- one handler per action type
    # Every handler takes (state, index, row): the period being parsed, the row's position in state.rows,
    # and the row itself.
    def _on_period(self, state: PeriodState, index: int, row) -> None:
        """End of period: close the possession in progress, then place the held technical free throws."""
        if row.subType != "end":
            return
        self._close(state, "period_end", 0.0, None)
        self._credit_held_free_throws(state.tally)

    def _credit_held_free_throws(self, tally: GameTally) -> None:
        """Technical free throws a team shot after its last possession of the period go to that possession."""
        for team in self.teams:
            held = tally.held[team]
            if held.points <= 0 and held.attempts <= 0:
                continue
            record = next((earlier for earlier in reversed(tally.records) if earlier["offense"] == team), None)
            if record is None:
                continue
            record["points"] += held.points
            record["pts_tech"] += held.points       # keeps points - pts_tech equal to the shooting counters
            record["fta_tech"] += held.attempts
            record["ftm_tech"] += held.makes
            record["xftm_tech"] += held.expected_makes
            tally.held[team] = HeldFreeThrows()

    def _on_substitution(self, state: PeriodState, index: int, row) -> None:
        """'SUB: <incoming> FOR <outgoing>': the row's player leaves the court and the named one comes on."""
        team = self._player_team(row.personId)
        if team not in self.teams:
            return
        match = SUB_RE.match(str(row.description))
        incoming = self._resolve_in(team, match.group(1), state.on[team], index, state.rows) if match else None
        self._take_off(state, team, row.personId)
        self._put_on(state, team, incoming)

    def _take_off(self, state: PeriodState, team: int, player: int) -> None:
        if player in state.on[team]:
            state.on[team].discard(player)
        else:
            self.diag["sub_out_not_on_floor"] += 1
            state.valid[team] = False

    def _put_on(self, state: PeriodState, team: int, incoming) -> None:
        if incoming is None:
            self.diag["unresolved_subs"] += 1
            state.valid[team] = False
            return
        state.on[team].add(incoming)
        if len(state.on[team]) != 5:
            state.valid[team] = False
        elif not state.valid[team] and state.starters[team] is not None:
            state.valid[team] = True          # back to five known players

    def _on_jump_ball(self, state: PeriodState, index: int, row) -> None:
        """'Tip to <name>': the tip's team gets the ball.  When the other team had it and had acted, its
        possession ends here; when nobody had it yet, the tip's team is on offence (not yet active)."""
        tip_team = self._tip_team(row.description)
        possession = state.possession
        if tip_team is None:
            return
        if possession.offense is not None and tip_team != possession.offense and possession.active:
            self.diag["jump_switches"] += 1
            self._close(state, "jump_ball", row.clock_s, tip_team)
        elif possession.offense is None:
            possession.offense = tip_team

    def _tip_team(self, description):
        """The team of the player the ball was tipped to, when his name points to one team only."""
        match = TIP_RE.search(str(description))
        if not match:
            return None
        key = _name_key(match.group(1))
        hits = [player for team in self.teams for player in self.names.get(team, {}).get(key, [])]
        teams_hit = {self._player_team(player) for player in hits}
        return teams_hit.pop() if len(teams_hit) == 1 else None

    def _on_shot(self, state: PeriodState, index: int, row) -> None:
        """A field-goal attempt ("Made Shot", "Missed Shot", "Heave").  A miss arms a rebound chance; a make
        scores and ends the possession, unless an and-one free throw follows."""
        team = self._row_team(row)
        if team not in self.teams:
            return
        self._ensure_offense(state, team, row)    # may close the possession; every count below follows it
        possession = state.possession
        counters = possession.counters
        made = row.actionType == "Made Shot"
        three = int(row.shotValue) == 3
        bucket = "thr" if three else ("rim" if 0 <= row.shot_dist <= RIM_FT else "mid")
        counters["fga"] += 1
        counters[f"fga_{bucket}"] += 1
        counters["fg3a"] += three
        if made:
            counters["fgm"] += 1
            counters[f"fgm_{bucket}"] += 1
            counters["fg3m"] += three
        self._note_attempt(possession, bucket)    # resolves an earlier miss before this one is armed
        first = counters["att"] == 1
        league_p = self.shot_p(three, row.shot_dist, str(row.description))
        self._count_shooter_attempt(possession, int(row.personId), three, made, league_p, first)
        if not made:
            possession.pending_chance = 1
            return
        self._score_made_shot(state, index, row, team, first)

    def _count_shooter_attempt(self, possession: Possession, player: int, three: bool, made: bool,
                               league_p: float, first: bool) -> None:
        """The shooter's possession counters (SHOOTER_COUNTERS) and his per-game totals (SHOT_TABLE)."""
        kind = "3" if three else "2"
        shooter = possession.shooters.setdefault(player, dict.fromkeys(SHOOTER_COUNTERS, 0.0))
        shooter[f"fg{kind}a"] += 1
        shooter[f"xl{kind}"] += league_p
        shooter[f"xv{kind}"] += league_p * (1.0 - league_p)
        if first:
            shooter[f"fg{kind}a1"] += 1
            shooter[f"xl{kind}_1"] += league_p
        game = self.shots.setdefault(player, dict.fromkeys(SHOT_TABLE, 0.0))
        game[f"fg{kind}a"] += 1
        game[f"xl{kind}"] += league_p
        game[f"fg{kind}m"] += made

    def _score_made_shot(self, state: PeriodState, index: int, row, team: int, first: bool) -> None:
        """The shot's points (its shotValue, 2 when that is neither 2 nor 3), then the possession ends, unless
        an and-one free throw follows: then it waits for the free throw."""
        points = int(row.shotValue) if int(row.shotValue) in (2, 3) else 2
        state.tally.score[team] += points
        possession = state.possession
        possession.points += points
        if first:
            possession.counters["pts1"] += points
        if self._and_one_follows(state.rows, index, team):
            possession.and1 = True
        else:
            self._close(state, "made_fg", row.clock_s, self._other(team))

    def _and_one_follows(self, rows, index: int, team: int) -> bool:
        """Within the next 11 rows at the same clock, a non-technical free throw by the shooting team, before
        any other shot, turnover, defensive rebound or the period's end."""
        for later in rows[index + 1:index + 12]:
            if later.clock_s != rows[index].clock_s or later.actionType in AND_ONE_STOPS:
                return False
            if later.actionType == "Rebound" and self._row_team(later) == self._other(team):
                return False
            if (later.actionType == "Free Throw" and self._row_team(later) == team
                    and "Technical" not in str(later.subType)):
                return True
        return False

    def _on_turnover(self, state: PeriodState, index: int, row) -> None:
        """A turnover ends the possession; the other team has the ball."""
        team = self._row_team(row)
        if team not in self.teams:
            return
        self._ensure_offense(state, team, row)
        state.possession.counters["tov"] += 1
        self._close(state, "turnover", row.clock_s, self._other(team))

    def _on_rebound(self, state: PeriodState, index: int, row) -> None:
        """The offence's own rebound continues the possession; the defence's ends it.  A rebound with nobody
        yet on offence gives the rebounder's team the ball."""
        team = self._row_team(row)
        if team not in self.teams:
            return
        possession = state.possession
        if possession.offense is None:
            possession.offense = team
            possession.active = True
        elif team == possession.offense:
            self._offensive_rebound(possession, row)
        else:
            self._defensive_rebound(state, row, team)

    @staticmethod
    def _offensive_rebound(possession: Possession, row) -> None:
        """Resolves the pending chance as a continuation.  Team rebounds (teamId 0) count here too; the
        period-end ones are taken back out in _settle_last_continuation."""
        if not possession.pending_chance:
            return
        counters = possession.counters
        counters["reb_chance"] += 1
        counters["reb_cont"] += 1
        if counters["att"] == 1:
            counters["chance1"] += 1
        team_rebound = int(row.teamId) == 0
        counters["oreb_t" if team_rebound else "oreb_p"] += 1
        possession.pending_chance = 0
        possession.cont_team = team_rebound
        possession.att_since_cont = 0

    def _defensive_rebound(self, state: PeriodState, row, team: int) -> None:
        """Resolves the chance BEFORE the close, which records the possession; the rebounders then have the ball."""
        possession = state.possession
        if possession.pending_chance:
            possession.counters["reb_chance"] += 1
            if possession.counters["att"] == 1:
                possession.counters["chance1"] += 1
            possession.pending_chance = 0
        self._close(state, "dreb", row.clock_s, team)
        state.possession.active = True

    def _on_free_throw(self, state: PeriodState, index: int, row) -> None:
        """A free throw, "Free Throw <nth> of <trip_size>".  The first one starts a trip; a made last one ends
        the possession and a missed last one arms a rebound chance, except after a flagrant or clear-path
        foul (the shooting team keeps the ball) or when the sub-type has no "<nth> of <trip_size>".
        Technical free throws are handled apart."""
        team = self._row_team(row)
        if team not in self.teams:
            return
        sub_type = str(row.subType)
        made = free_throw_made(str(row.description), row.scoreHome + row.scoreAway)
        if "Technical" in sub_type:
            self._technical_free_throw(state, row, team, made)
            return
        self._ensure_offense(state, team, row)    # may close the possession; every count below follows it
        possession = state.possession
        match = FT_RE.search(sub_type)
        nth, trip_size = (int(match.group(2)), int(match.group(3))) if match else (1, 1)
        flagrant_or_clear_path = ("Flagrant" in sub_type) or ("Clear Path" in sub_type)
        if nth == 1:
            self._start_trip(state, possession, flagrant_or_clear_path or "Away From Play" in sub_type)
        self._count_free_throw(possession, row, made, nth == trip_size)
        if made:
            state.tally.score[team] += 1
            possession.points += 1
        if flagrant_or_clear_path or not match or nth != trip_size:
            return
        if made:
            self._close(state, "made_ft", row.clock_s, self._other(team))
        else:
            possession.pending_chance = 1        # a missed last free throw is a live rebound

    def _technical_free_throw(self, state: PeriodState, row, team: int, made: bool) -> None:
        """Counted in the shooting team's possession when it has the ball and has acted, else held for it."""
        possession = state.possession
        if made:
            state.tally.score[team] += 1
        expected = self.ft_prob(row.personId)
        if possession.offense == team and possession.active:
            possession.points += int(made)
            possession.counters["pts_tech"] += int(made)
            possession.counters["fta_tech"] += 1
            possession.counters["ftm_tech"] += made
            possession.counters["xftm_tech"] += expected
            return
        held = state.tally.held[team]
        held.points += int(made)
        held.attempts += 1
        held.makes += made
        held.expected_makes += expected

    def _start_trip(self, state: PeriodState, possession: Possession, ball_returned: bool) -> None:
        """The first free throw of a trip: is the trip a new attempt event?  Three kinds are not:
            and-one      the field goal it follows already counted
            fouled_fga   a shooting foul on a miss that the feed ALSO recorded as a missed field goal.  Usually
                         a shooting foul on a miss records no field-goal attempt at all, which is why trips count
                         as attempts; when both appear they are one attempt, and the whistle means that miss was
                         never a live rebound either
            retained     a flagrant, clear-path or away-from-play trip (`ball_returned`): the possession's next
                         attempt, if it has one, is counted as `att_retained`, not as a continuation.  Flagrant
                         and clear-path trips never end the possession; an away-from-play trip ends it like any
                         other trip, on a made last free throw
        """
        counters = possession.counters
        double = state.last_foul_shooting and possession.pending_chance == 1
        if possession.and1 and not possession.and1_used:
            possession.and1_used = True
            counters["and1"] += 1
        elif double:
            possession.pending_chance = 0     # the whistle killed the rebound chance
            counters["fga_fouled"] += 1
        elif not ball_returned:
            counters["trip_shoot" if state.last_foul_shooting else "trip_ns"] += 1
            self._note_attempt(possession, "ft")
        if ball_returned:
            possession.retain_pending = 1

    def _count_free_throw(self, possession: Possession, row, made: bool, last_of_trip: bool) -> None:
        """The free throw in the possession's counters and the shooter's; `fta1` and `ftlast1` when it belongs to
        the possession's first attempt (the last one's miss is the first attempt's rebound chance)."""
        counters = possession.counters
        counters["fta"] += 1
        counters["ftm"] += made
        counters["xftm"] += self.ft_prob(row.personId)
        shooter = possession.shooters.setdefault(int(row.personId), dict.fromkeys(SHOOTER_COUNTERS, 0.0))
        shooter["fta"] += 1
        if counters["att"] > 1:
            return
        counters["fta1"] += 1
        shooter["fta1"] += 1
        if made:
            counters["pts1"] += 1
        if last_of_trip:
            shooter["ftlast1"] += 1

    def _on_foul(self, state: PeriodState, index: int, row) -> None:
        """A foul ends nothing, but the trip after it needs to know whether it was a shooting foul: a shooting
        foul on a MISS records no field-goal attempt at all."""
        state.last_foul_shooting = "Shooting" in str(row.subType)

    # The dispatch table.  Every other action type ("Timeout", "Violation", "Instant Replay", "Ejection", the
    # blank steal and block rows) has no effect on the possession.
    HANDLERS = MappingProxyType({
        "period": _on_period, "Substitution": _on_substitution, "Jump Ball": _on_jump_ball,
        "Made Shot": _on_shot, "Missed Shot": _on_shot, "Heave": _on_shot, "Turnover": _on_turnover,
        "Rebound": _on_rebound, "Free Throw": _on_free_throw, "Foul": _on_foul})

    # ---------------------------------------------------------------- stints
    def stints(self, poss: pd.DataFrame | None = None) -> pd.DataFrame:
        """The possessions collapsed into stints: a new stint at every change of period or lineup and around
        every invalid possession; invalid possessions are dropped.  One row per stint, home and away sums side
        by side (`_h`, `_a`), the context of its first possession, the lineup slots, the shooter slot counters."""
        if poss is None:
            poss = self.possessions()
        if len(poss) == 0:
            return pd.DataFrame()
        key = poss["period"].astype(str) + "|" + poss["home_lineup"].astype(str) + "|" + poss["away_lineup"].astype(str)
        new = (key != key.shift()) | (~poss["valid"]) | (~poss["valid"].shift(fill_value=True))
        stint_id = new.cumsum()
        valid = poss[poss["valid"]]
        if len(valid) == 0:
            return pd.DataFrame()
        ids = stint_id[poss["valid"]].to_numpy()
        sums, counts = side_sums(valid, ids, self.home)
        first = valid.groupby(ids, sort=True).head(1)
        out = stint_frame(first, sums, counts, self.gt_rule)
        block = shooter_slot_block(valid, ids, sums.index, self.home)
        return pd.concat([out, pd.DataFrame(block, columns=list(SLOT_COLUMNS), index=out.index)], axis=1)


def side_sums(valid: pd.DataFrame, ids, home: int):
    """Per stint, points and every POSS_COUNTERS column summed for each side, and the possessions per side.

    One groupby over a masked matrix rather than a slice-sum per counter per stint: with 35 counters on both
    sides that was about 2.4 million pandas slice-sums a season and dominated the whole parse."""
    columns = ["points"] + list(POSS_COUNTERS)
    values = valid[columns].to_numpy(dtype=float)
    is_home = (valid["offense"].to_numpy() == home)
    both = np.concatenate([values * is_home[:, None], values * (~is_home)[:, None]], axis=1)
    names = [f"{column}_h" for column in columns] + [f"{column}_a" for column in columns]
    sums = pd.DataFrame(both, columns=names).groupby(ids, sort=True).sum()
    sums = sums.rename(columns={"points_h": "pts_h", "points_a": "pts_a"})
    counts = pd.DataFrame({"poss_h": is_home.astype(int), "poss_a": (~is_home).astype(int)}
                          ).groupby(ids, sort=True).sum()
    return sums, counts


def stint_frame(first: pd.DataFrame, sums: pd.DataFrame, counts: pd.DataFrame, gt_rule: dict) -> pd.DataFrame:
    """The stint rows: context from each stint's FIRST possession (period, clock, home margin, share of
    regulation left, garbage time), possessions and points per side, every counter per side, the lineup slots.

    Every counter is defined for the team ON OFFENCE, so the opponent's defensive version of it is the other
    side's column; the design emits both."""
    period = first["period"].to_numpy().astype(int)
    clock = first["start_clock"].to_numpy().astype(float)
    margin_h = (first["score_home"] - first["score_away"]).to_numpy().astype(int)
    frac_rem = np.array([regulation_remaining(int(period_no), float(seconds))
                         for period_no, seconds in zip(period, clock)]) / 2880.0
    out = pd.DataFrame({"period": period, "poss_h": counts["poss_h"].to_numpy(),
                        "poss_a": counts["poss_a"].to_numpy(), "pts_h": sums["pts_h"].to_numpy(),
                        "pts_a": sums["pts_a"].to_numpy(), "margin_h": margin_h,
                        "frac_rem": frac_rem, "is_gt": garbage_time(period, clock, margin_h, gt_rule),
                        "start_clock": clock})
    for counter in POSS_COUNTERS:
        out[f"{counter}_h"] = sums[f"{counter}_h"].to_numpy()
        out[f"{counter}_a"] = sums[f"{counter}_a"].to_numpy()
    for slots, column in ((HOME_SLOTS, "home_lineup"), (AWAY_SLOTS, "away_lineup")):
        lineups = np.array([list(lineup) for lineup in first[column]], dtype=np.int64)
        for position, name in enumerate(slots):
            out[name] = lineups[:, position]
    return out


def garbage_time(period, clock, margin_h, gt_rule: dict):
    """Fourth quarter with |margin| >= q4_margin_any (20), or >= q4_margin_late (15) with under late_minutes (6)
    left (config `garbage_time`)."""
    margin = np.abs(margin_h)
    late = (margin >= gt_rule.get("q4_margin_late", 15)) & (clock < 60 * gt_rule.get("late_minutes", 6))
    return (period == 4) & ((margin >= gt_rule.get("q4_margin_any", 20)) | late)


def shooter_slot_block(valid: pd.DataFrame, ids, stint_index, home: int) -> np.ndarray:
    """The per-shooter counters summed per stint, by the shooter's slot in the stint's sorted offensive lineup
    (columns SLOT_COLUMNS).  Every possession of a stint shares the lineup, so a player's slot is fixed within
    it; a shooter no longer on the court when the possession closed goes to slot `x`."""
    row_of = {stint: position for position, stint in enumerate(stint_index)}
    column_of = {column: position for position, column in enumerate(SLOT_COLUMNS)}
    block = np.zeros((len(stint_index), len(SLOT_COLUMNS)))
    rows = zip(ids, valid["offense"].to_numpy(), valid["home_lineup"], valid["away_lineup"], valid["shooters"])
    for stint, offense, home_lineup, away_lineup, shooters in rows:
        side, lineup = ("h", home_lineup) if offense == home else ("a", away_lineup)
        slot_of = {player: str(position + 1) for position, player in enumerate(lineup)}
        for player, counts in shooters.items():
            slot = slot_of.get(player, "x")
            for name, value in counts.items():
                if value:
                    block[row_of[stint], column_of[f"{name}_s{slot}_{side}"]] += value
    return block
