"""The year-over-year test: one season's ratings predict the games of the seasons either side of it.

This is the one test in the project that is both external to the model (it scores actual points in seasons the
ratings never saw) and legal to select on (it uses no outside data).  scripts/checks/63_yoy.py runs it; how a
prediction is made and scored is in `openrapm.scoring`.

Words used below
    scored season    a season whose games are predicted: 1998-2025 by default, every season with a season on
                     both sides of it
    direction        "prev": the season BEFORE the scored one is the rated season, so its ratings look forward;
                     "next": the season after, so they look back.  Each scored season is predicted from both,
                     two observations a season
    offset           how far the rated season sits from the scored one: 1 by default.  The trade set
                     (scripts/checks/70_tradeset.py) is fitted on the games one season either side of the rated one,
                     so it is scored at offset 2
    neighbour table  one ratings table asked in one direction (`NeighbourTable`): nothing is fitted, so what is
                     measured is exactly the table handed in
    fill             what a player the rated season's table has no row for scores (`fill_rule`): 0 (the average
                     player), a replacement level read off the table itself, or two numbers set by hand
    split            the same score again inside groups of rows: `movers` (how many of the ten on the court
                     changed team) or `game_bench_share` (the share of the team-game's possessions played by
                     players with fewer than 500 possessions in the rated season)

Every result row is one scored season x neighbour table x split x group, with the columns of `RESULT_COLUMNS`.
Three of them are constant and kept so the files keep the schema every reader of them (scripts 66, 101, 135)
expects: k = 1 (one rated season), lam = 0 (nothing is fitted) and cut = NaN.

Example: with `--rankings=new=a.parquet,incumbent=b.parquet` the 2004 games are predicted four times, by the 2003
and the 2005 rows of each table, and the report pairs `new:prev` with `incumbent:prev` over the 28 scored seasons.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .. import scoring
from ..data.box_scores import main_team_by_minutes
from ..data.designs import DesignCache

__all__ = ["RESULT_COLUMNS", "SPLITS", "scored_seasons", "fill_rule", "NeighbourTable", "neighbour_tables",
           "split_labels", "score_season", "run", "by_direction", "report"]

RESULT_COLUMNS = ("held_out", "k", "train", "system", "lam", "cut", "split", "group", "n", "mse", "base", "calib",
                  "calib_side", "scale", "scale_off", "scale_def", "covered", "tg", "tg_base", "tg_n", "seconds")
SPLITS = ("movers", "game_bench_share")
DIRECTIONS = (("prev", -1), ("next", +1))

REPORT_COLUMNS = ("system", "seasons", "armse", "vs_no_ratings", "calib_side", "scale_off", "scale_def", "covered",
                  "game_armse", "game_vs_no_ratings")
REPORT_FRAMES = (("prev", "the PREVIOUS season's rankings predict this season"),
                 ("next", "the NEXT season's rankings predict this season"),
                 ("both", "both directions together (two observations per scored season)"))
REPORT_PAIRED = (("tg", "team-game level"), ("mse", "stint level"),
                 ("calib_side", "stint level with each side rescaled to the scored season"))
REPORT_HEADER = (
    "=== year-over-year, pooled over scored seasons, LOWER IS BETTER.",
    "    game_armse: what a typical team-game misses by, points per 100.  armse: the same per stint.",
    "    scale_*: what the scored season wants each side multiplied by (1 = calibrated, below 1 = too wide).",
    "    calib_side: stint MSE after each side is multiplied by that scale -- the error that is left once",
    "    amplitude is taken out, i.e. the part that is ranking; a diagnostic, never a score.")


def scored_seasons(cfg: dict, first: int | None = None, last: int | None = None) -> list[int]:
    """The scored seasons: `first` to `last`, by default every season with a season on both sides of it.

    Example: config.yaml's 1997-2026 gives 1998-2025.
    """
    first_season, last_season = int(cfg["first_season"]), int(cfg["last_season"])
    first = first_season + 1 if first is None else int(first)
    last = last_season - 1 if last is None else int(last)
    return [season for season in range(first, last + 1) if first_season <= season <= last_season]


# ------------------------------------------------------------------------------------- the neighbour tables
def no_fill(table: pd.DataFrame) -> tuple[float, float]:
    """The test's default for a player with no row: 0, the average player."""
    return 0.0, 0.0


def fixed_fill(offence: float, defence: float, table: pd.DataFrame) -> tuple[float, float]:
    """A fill set by hand, already in raw sign."""
    return offence, defence


def fill_rule(spec: str | None) -> Callable[[pd.DataFrame], tuple[float, float]]:
    """The fill a `--fill` entry names, as a function of the rated season's rows.

        ""            no fill: 0, the average player
        "500"         the replacement level of the table's own players under 500 possessions (scoring.replacement_level)
        "500x0.25"    a quarter of that level, the depth the test settled on (DECISIONS.md)
        "-1.6/-0.5"   set by hand, positive-good, offence first: a defensive rating is stored points-allowed,
                      so its sign flips going in

    Example: fill_rule("-1.6/-0.5") gives a player with no row o = -1.6 and d = +0.5.
    """
    if not spec:
        return no_fill
    if "/" in spec:
        offence, defence = (float(x) for x in spec.split("/"))
        return partial(fixed_fill, offence, -defence)
    max_poss, _, shrink = spec.partition("x")
    return partial(scoring.replacement_level, max_poss=float(max_poss), shrink=float(shrink or 1.0))


@dataclass(frozen=True)
class NeighbourTable:
    """One ratings table asked in one direction: the season `offset` away from the scored one, fitted on nothing.

    `table` has player_id, season, o, d (raw sign) and poss; `seasons` are the seasons it has rows for.
    """
    name: str
    table: pd.DataFrame
    offset: int
    seasons: frozenset
    fill: Callable[[pd.DataFrame], tuple[float, float]] = no_fill

    def rated_season(self, scored: int) -> int | None:
        """The season whose rows predict `scored`, or None when the table has no rows for it."""
        season = int(scored) + self.offset
        return season if season in self.seasons else None

    def ratings(self, rated: int) -> scoring.Ratings:
        """The rated season's rows, with the fill they imply for a player they do not cover."""
        rows = self.table[self.table.season == rated][["player_id", "o", "d", "poss"]].reset_index(drop=True)
        fill_o, fill_d = self.fill(rows)
        return scoring.Ratings(rows, fill_o=fill_o, fill_d=fill_d)


def neighbour_tables(tables: dict, offset: int = 1, fills: dict | None = None) -> list[NeighbourTable]:
    """Two neighbour tables per named table, "<name>:prev" then "<name>:next", in the order the tables are named.

    `tables` is {name: {"prev": frame, "next": frame}} (usually the same frame twice: scripts/checks/63_yoy.py reads
    `name=a.parquet|b.parquet` as one file per direction); `fills` is {name: a `fill_rule` spec}.
    """
    fills = fills or {}
    systems = []
    for name, by_direction in tables.items():
        for direction, sign in DIRECTIONS:
            table = by_direction[direction]
            systems.append(NeighbourTable(f"{name}:{direction}", table, sign * int(offset),
                                          frozenset(table.season.unique()), fill_rule(fills.get(name))))
    return systems


# ------------------------------------------------------------------------------------------------- the run
def split_labels(split: str, prediction, design, cfg: dict, scored: int, rated: int) -> np.ndarray:
    """The group of every row of the scored season under `split` (one of SPLITS)."""
    if split == "movers":
        return scoring.movers_split(design, main_team_by_minutes(scored, cfg), main_team_by_minutes(rated, cfg))
    if split == "game_bench_share":
        return scoring.game_bench_share_split(prediction, design)
    raise ValueError(f"unknown split {split!r}; the year-over-year test has {', '.join(SPLITS)}")


def score_season(scored: int, systems: list, design, splits, cfg: dict) -> list[dict]:
    """The result rows of one scored season: each neighbour table on all rows, then inside every split's groups."""
    rows = []
    for system in systems:
        rated = system.rated_season(scored)
        if rated is None:
            continue
        started = time.time()
        prediction = scoring.predict_season(system.ratings(rated), design)
        base = dict(held_out=scored, k=1, train=str(rated), system=system.name, lam=0.0, cut=np.nan)
        rows.append(dict(**base, split="all", group="all", **scoring.score(prediction), seconds=time.time() - started))
        for split in splits:
            labels = split_labels(split, prediction, design, cfg, scored, rated)
            for group in pd.unique(labels):
                rows.append(dict(**base, split=split, group=str(group), **scoring.score(prediction, labels == group),
                                 seconds=0.0))
    return rows


def run(systems: list, designs: DesignCache, seasons: list, splits=(), out: Path | None = None,
        verbose: bool = True) -> pd.DataFrame:
    """Every scored season predicted by every neighbour table: one row per season x table x split x group.

    Each season is scored on its points design, playoffs included, because a season is one entity.  With `out`
    the rows so far are written after every season (a run can be read before it ends) and once at the end.
    """
    unknown = [split for split in splits if split not in SPLITS]
    if unknown:
        raise ValueError(f"unknown split(s) {unknown}; the year-over-year test has {', '.join(SPLITS)}")
    rows = []
    started = time.time()
    if verbose:
        print(f"held-out {seasons[0]}-{seasons[-1]} ({len(seasons)}), systems {[s.name for s in systems]}",
              flush=True)
    for scored in seasons:
        rows += score_season(scored, systems, designs.season(scored, "pts"), splits, designs.cfg)
        if verbose:
            print(f"  {scored} done ({time.time() - started:.0f}s)", flush=True)
        if out is not None:
            pd.DataFrame(rows)[list(RESULT_COLUMNS)].to_parquet(out, index=False)
    result = pd.DataFrame(rows)[list(RESULT_COLUMNS)]
    if out is not None:
        result.to_parquet(out, index=False)
    return result


# ---------------------------------------------------------------------------------------------- the report
def by_direction(res: pd.DataFrame) -> dict:
    """{"prev", "next", "both"}: the results of each direction, and of both as two observations per season.

    The table name and its direction are split out of `system` ("new:prev" is system "new", direction "prev");
    in "both", a next-direction season is keyed half a season later so the two observations pair separately.
    """
    res = res.copy()
    res["direction"] = res.system.str.rsplit(":", n=1).str[1]
    res["system"] = res.system.str.rsplit(":", n=1).str[0]
    both = res.copy()
    both["held_out"] = both.held_out + both.direction.map({"prev": 0.0, "next": 0.5})
    return {"prev": res[res.direction == "prev"], "next": res[res.direction == "next"], "both": both}


def _pooled_lines(frame: pd.DataFrame) -> list[str]:
    lines = []
    for (split, group), d in scoring.pooled(frame).groupby(["split", "group"], sort=True):
        if split != "all":
            lines.append(f"    [{split} = {group}, share of possessions {d.share.iloc[0]:.2f}]")
        lines.append(d[list(REPORT_COLUMNS)].to_string(index=False))
    return lines


def _paired_lines(frame: pd.DataFrame, ref: str) -> list[str]:
    lines = [f"\n    paired by scored season against {ref!r}; mean_diff below zero = better than {ref!r};",
             "    se = standard error of the mean difference; z = mean_diff / se; wins = seasons better"]
    for value, label in REPORT_PAIRED:
        table = scoring.paired(frame, ref, value)
        if len(table):
            lines.append(f"    -- {label}")
            lines.append(table.drop(columns=["ref", "value", "k", "lam"]).to_string(index=False))
    return lines


def report(res: pd.DataFrame, ref: str, compare: bool = True) -> str:
    """The printed summary: pooled scores per direction and split, then (`compare`) each table paired with `ref`."""
    lines = list(REPORT_HEADER)
    frames = by_direction(res)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 4):
        for key, title in REPORT_FRAMES:
            frame = frames[key]
            lines.append(f"\n--- {title}")
            lines += _pooled_lines(frame)
            if compare:
                lines += _paired_lines(frame, ref)
    return "\n".join(lines)
