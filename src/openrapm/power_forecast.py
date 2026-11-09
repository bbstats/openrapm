"""The Power page's one rating: Power at its fitted multipliers plus a team's last-10-games stats.

The owner, 2026-11-09: "I want a single table", then "One rating, stats fade".  notes/DECISIONS.md, "Check 159" to
"Check 165", holds every number.

A team's offense is its players' offense ratings times their shares of its possessions, times a multiplier, plus its
own last-10 stats on offense times their weights.  Its defense is the same with its defense ratings and the stats of its
opponents.  Example, on a day with multipliers 1.20 and 1.25: a team with Power of +2.00 on offense and -1.00 on
defense, stats parts of +0.40 and +0.10, reads offense 1.20 x 2.00 + 0.40 = +2.80 and defense
1.25 x -1.00 + 0.10 = -1.15.

The stats, from a team's last 10 games, each game counting once
    attempts per 100 possessions from the rim, short mid, long mid, three and long three
    makes over expected in the same five zones: makes less what the league's percentage in the zone expects from the
        team's attempts, per 100.  A team with 30 rim attempts per 100 and 21 makes where the league makes 65% is +1.5
    offensive rebounds over expected, from the team's rebound chances, per 100
    turnovers per 100
    free throw attempts per 100 and free throws made over expected, technicals left out
Every stat is centered over the 30 teams, so the average team reads 0.

The multipliers and weights change through the season, because the stats help a forecast of the next 10 games early on
and not later (checks 163 and 164).  `params/power_forecast.json` holds them at four points of the regular season, its
knots, from `scripts/checks/165_power_forecast_weights.py`.  Between two knots each number is the straight line between
them; before the first it holds the first's, after the last the last's, whose stats weights are 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data.config import resolve

WEIGHTS_FILE = "params/power_forecast.json"
FLAT_SHARES_BEFORE = 0.25  # the share of the regular season before which each of the last 10 games counts once
LAST_GAMES = 10
ZONES = ("rim", "smid", "lmid", "thr", "lthr")
COUNTS = (
    "poss",
    "fta",
    "ftm",
    "fta_tech",
    "ftm_tech",
    "tov",
    "reb_chance",
    "reb_cont",
    *(f"{kind}_{zone}" for kind in ("fga", "fgm") for zone in ZONES),
)


def team_game_counts(season: int, cfg: dict) -> pd.DataFrame:
    """One row per team per regular-season game: game_id, game_date, tm, the team's counts `*_o` and its opponents'
    `*_d`, three-pointers in `thr` without the long ones in `lthr`."""
    columns = ["game_id", "game_date", *(f"{count}_{end}" for count in COUNTS for end in "ha")]
    stints = pd.read_parquet(resolve(cfg, "stints") / f"{season}_RS.parquet", columns=columns)
    stints["game_id"] = stints.game_id.astype(str).str.zfill(10)
    games = stints.groupby("game_id", as_index=False).agg(
        game_date=("game_date", "first"), **{c: (c, "sum") for c in columns[2:]}
    )
    for end in "ha":  # the stints' three-pointers hold the long ones
        games[f"fga_thr_{end}"] -= games[f"fga_lthr_{end}"]
        games[f"fgm_thr_{end}"] -= games[f"fgm_lthr_{end}"]
    log = pd.read_parquet(
        resolve(cfg, "raw") / "gamelog" / f"{season}_RS.parquet",
        columns=["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP"],
    ).drop_duplicates(["GAME_ID", "TEAM_ID"])
    log["GAME_ID"] = log.GAME_ID.astype(str).str.zfill(10)
    # a neutral-site game lists both teams as "@": the stints' home side is then the lower team id (data/ingest.py)
    log["is_home"] = log.MATCHUP.str.contains(" vs. ", regex=False)
    log = log.sort_values(["GAME_ID", "is_home", "TEAM_ID"], ascending=[True, False, True], kind="stable")
    games["home_tm"] = games.game_id.map(log.groupby("GAME_ID").TEAM_ABBREVIATION.first())
    games["away_tm"] = games.game_id.map(log.groupby("GAME_ID").TEAM_ABBREVIATION.last())
    return team_rows(games)


def team_rows(games: pd.DataFrame) -> pd.DataFrame:
    """`games`, one row per game with `*_h` and `*_a` counts and home_tm, away_tm, as one row per team per game."""
    rows = []
    for mine, theirs, tm in (("h", "a", "home_tm"), ("a", "h", "away_tm")):
        part = games[["game_id", "game_date"]].assign(tm=games[tm])
        for count in COUNTS:
            part[f"{count}_o"], part[f"{count}_d"] = games[f"{count}_{mine}"], games[f"{count}_{theirs}"]
        rows.append(part)
    return pd.concat(rows, ignore_index=True).sort_values(["tm", "game_date", "game_id"], kind="stable")


def last_ten_stats(team_games: pd.DataFrame) -> pd.DataFrame:
    """The stats of each team over its last 10 games, one row per team indexed by tm, each centered over the teams: for
    each of end "o", the team's offense, and "d", its opponents' on offense against it, the columns
    attempts_<zone>_<end>, makes_over_<zone>_<end>, orb_over_<end>, tov_<end>, ft_attempts_<end> and
    ft_makes_over_<end>."""
    last = team_games.sort_values(["tm", "game_date", "game_id"], kind="stable").groupby("tm").tail(LAST_GAMES)
    sums = last.groupby("tm")[[f"{count}_{end}" for count in COUNTS for end in "od"]].sum()
    league = sums.sum()
    out = pd.DataFrame(index=sums.index)
    for end in "od":
        per100 = 100 / sums[f"poss_{end}"]
        for zone in ZONES:
            attempts, makes = sums[f"fga_{zone}_{end}"], sums[f"fgm_{zone}_{end}"]
            rate = (league[f"fgm_{zone}_o"] + league[f"fgm_{zone}_d"]) / (
                league[f"fga_{zone}_o"] + league[f"fga_{zone}_d"]
            )
            out[f"attempts_{zone}_{end}"] = attempts * per100
            out[f"makes_over_{zone}_{end}"] = (makes - rate * attempts) * per100
        rate = (league.reb_cont_o + league.reb_cont_d) / (league.reb_chance_o + league.reb_chance_d)
        out[f"orb_over_{end}"] = (sums[f"reb_cont_{end}"] - rate * sums[f"reb_chance_{end}"]) * per100
        out[f"tov_{end}"] = sums[f"tov_{end}"] * per100
        attempts = sums[f"fta_{end}"] - sums[f"fta_tech_{end}"]
        makes = sums[f"ftm_{end}"] - sums[f"ftm_tech_{end}"]
        rate = (league.ftm_o + league.ftm_d - league.ftm_tech_o - league.ftm_tech_d) / (
            league.fta_o + league.fta_d - league.fta_tech_o - league.fta_tech_d
        )
        out[f"ft_attempts_{end}"] = attempts * per100
        out[f"ft_makes_over_{end}"] = (makes - rate * attempts) * per100
    return out - out.mean()


def load_knots(path: Path) -> list:
    """The knots of `params/power_forecast.json`, each {"share", "offense_multiplier", "defense_multiplier", "stats"},
    earliest first."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return sorted(document["knots"], key=lambda knot: knot["share"])


def weights_at(share: float, knots: list) -> dict:
    """The multipliers and stats weights at `share` of the regular season, by the straight line between the knots: held
    at the first knot before it and at the last after it.  Returns {"offense_multiplier", "defense_multiplier",
    "stats"}.

    Example: halfway between knots of 0.10 and 0.25, a multiplier of 1.20 and 1.10 reads 1.15."""
    shares = [knot["share"] for knot in knots]

    def line(values: list) -> float:
        return float(np.interp(share, shares, values))

    names = sorted(set().union(*(knot["stats"] for knot in knots)))
    return {
        "offense_multiplier": line([knot["offense_multiplier"] for knot in knots]),
        "defense_multiplier": line([knot["defense_multiplier"] for knot in knots]),
        "stats": {name: line([knot["stats"].get(name, 0.0) for knot in knots]) for name in names},
    }


def flat_shares(share: float) -> bool:
    """Whether each of a team's last 10 games counts once in its players' shares at `share` of the regular season."""
    return share < FLAT_SHARES_BEFORE


def stats_parts(stats: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Each team's stats part on offense and on defense from `last_ten_stats` and `weights_at`'s stats: columns o, d,
    indexed by tm.  A column ending _o is the team's offense, one ending _d its opponents' on offense against it."""
    named = pd.Series(weights["stats"], dtype=float)
    parts = {}
    for side in "od":
        columns = [name for name in named.index if name.endswith(f"_{side}")]
        parts[side] = (stats[columns] * named[columns]).sum(axis=1) if columns else pd.Series(0.0, index=stats.index)
    return pd.DataFrame(parts).reindex(stats.index)
