"""Who was on the floor at every play-by-play row of a season.

    .venv/Scripts/python scratch/2026-11-25_closest_defender/shared/floor_log.py 2015 RS [games]

writes scratch/2026-11-25_closest_defender/shared/floor_<season>_<phase>.parquet, one row per play-by-play row:

    game_id, action_number, action_id, period, clock_s (seconds left in the period), team_id, person_id,
    action_type, sub_type, description, shot_value, x, y (xLegacy, yLegacy: tenths of a foot from the rim),
    shot_dist (feet), home_id, away_id,
    h1..h5, a1..a5   the player ids on the floor BEFORE the row takes effect, sorted; 0 where fewer than five are known
    valid            both teams have five known players

The lineups are the stint parser's own (openrapm.data.stint_parser.GameParser), so they match the ratings' stints.
A block is a row with a blank action_type and "BLOCK" in its description, sharing action_number with its missed shot;
a foul row's person_id is the fouler and its sub_type says "Shooting" for a shooting foul.
Only files already on disk are read (OPENRAPM_OFFLINE=1): a game that fails is listed in
floor_<season>_<phase>_errors.csv and left out.  The optional third argument stops after that many games.
"""
import os
import sys
import time
from functools import partial
from pathlib import Path

os.environ["OPENRAPM_OFFLINE"] = "1"

import pandas as pd  # noqa: E402

from openrapm.data.config import load_config  # noqa: E402
from openrapm.data.ingest import fetch_boxscore, fetch_pbp, fetch_period_box, game_table, load_gamelog  # noqa: E402
from openrapm.data.stint_parser import GameParser, GameTally, HeldFreeThrows  # noqa: E402

HERE = Path(__file__).resolve().parent
HOME_SLOTS = ["h1", "h2", "h3", "h4", "h5"]
AWAY_SLOTS = ["a1", "a2", "a3", "a4", "a5"]


def five(players) -> list:
    """Five ids, sorted, padded with 0 when fewer are known and cut at five when the parser holds more."""
    players = sorted(int(player) for player in players)[:5]
    return players + [0] * (5 - len(players))


class FloorLogger(GameParser):
    """The stint parser, also writing down the ten players on the floor at every row."""

    def possessions(self) -> pd.DataFrame:
        tally = GameTally(score={self.home: 0, self.away: 0},
                          held={self.home: HeldFreeThrows(), self.away: HeldFreeThrows()})
        self.floor = []
        for period, group in self.ev.groupby("period", sort=True):
            state = self._start_period(period, list(group.itertuples(index=False)), tally)
            for index, row in enumerate(state.rows):
                self.floor.append(self._floor_row(state, row))
                handler = self.HANDLERS.get(row.actionType)
                if handler is not None:
                    handler(self, state, index, row)
        return pd.DataFrame(tally.records)

    def _floor_row(self, state, row) -> tuple:
        return (self.game_id, int(row.actionNumber), int(row.actionId), int(row.period), float(row.clock_s),
                int(row.teamId), int(row.personId), row.actionType, row.subType, row.description,
                int(row.shotValue), float(row.xLegacy), float(row.yLegacy), float(row.shot_dist),
                self.home, self.away, *five(state.on[self.home]), *five(state.on[self.away]),
                bool(self._lineups_known(state)))


COLUMNS = ["game_id", "action_number", "action_id", "period", "clock_s", "team_id", "person_id", "action_type",
           "sub_type", "description", "shot_value", "x", "y", "shot_dist", "home_id", "away_id",
           *HOME_SLOTS, *AWAY_SLOTS, "valid"]


def main() -> None:
    season, phase = int(sys.argv[1]), sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    cfg = load_config()
    games = game_table(load_gamelog(season, phase, cfg))
    if limit:
        games = games.head(limit)
    rows, errors, started = [], [], time.time()
    for done, game in enumerate(games.itertuples(index=False), start=1):
        try:
            parser = FloorLogger(fetch_pbp(game.game_id, cfg), fetch_boxscore(game.game_id, cfg), game.home_team_id,
                                 game.away_team_id, game.game_id,
                                 period_box_fetcher=partial(fetch_period_box, cfg=cfg))
            parser.possessions()
        except Exception as error:      # a game the parser cannot read is listed, never guessed
            errors.append(dict(game_id=game.game_id, error=repr(error)[:300]))
            continue
        rows.extend(parser.floor)
        if done % 200 == 0:
            print(f"  {done} of {len(games)} games, {time.time() - started:.0f} s", flush=True)
    table = pd.DataFrame(rows, columns=COLUMNS)
    name = f"floor_{season}_{phase}" + (f"_first{limit}" if limit else "")
    table.to_parquet(HERE / f"{name}.parquet", index=False)
    pd.DataFrame(errors, columns=["game_id", "error"]).to_csv(HERE / f"{name}_errors.csv", index=False)
    shots = table[table.action_type.isin(["Made Shot", "Missed Shot"])]
    print(f"{season} {phase}: {table.game_id.nunique()} games, {len(table)} rows, {len(shots)} field-goal attempts, "
          f"{shots.valid.mean():.4f} of them with ten known players, {len(errors)} games failed, "
          f"{time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
