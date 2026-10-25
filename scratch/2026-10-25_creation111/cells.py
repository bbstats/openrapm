"""Experiment 111's estimate, the shared pieces: every shot's cell (where it was taken and how the possession got
there), each season's assisted share of made shots by cell, and the miss factor learned from 2016 tracking.

    spot      rim (4 ft or closer, or logged without a location), paint (4-10 ft), mid (10-16 ft), long two (16 ft+),
              corner three (|x| >= 22 ft and y <= 9.25 ft), above-the-break three
    context   putback: within 3 s of his team's offensive rebound; second chance: later after one; transition: the
              first shot within 7 s of a defensive rebound or a steal; early: within 7 s of any other start (a make,
              a dead-ball turnover, a period start); then 7-14 s, 14-20 s, 20 s or more into the possession
    p(cell)   the season's made shots in the cell that were assisted, over its made shots there, padded 50 makes
              toward the cell's share over every season
    miss factor  a logit shift per cell: on 2016's tracked shots a make model that also knows dribbles and touch time
              scores every miss; the shift moves p(cell) to that model's mean over the cell's misses
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPOTS = ["rim", "paint", "mid", "long_two", "corner_three", "above_break_three"]
CONTEXTS = ["putback", "second_chance", "transition", "early", "s7_14", "s14_20", "s20_plus"]
CELL_PAD = 50.0
SHOT_COLUMNS = ["game_id", "phase", "season", "team", "shooter", "made", "value", "at_heave", "x", "y", "dist",
                "clock", "poss_start", "poss_start_clock", "anchor_kind", "anchor_clock", "att_no", "post_assisted",
                "post_and1", "action_number"]


def load_shots(season: int, phases=("RS", "PO")) -> pd.DataFrame:
    """The season's field-goal attempts from data/shotframe, heaves out, each with its spot, context and cell."""
    frames = []
    for phase in phases:
        try:
            frames.append(pd.read_parquet(f"data/shotframe/{season}_{phase}.parquet", columns=SHOT_COLUMNS))
        except FileNotFoundError:
            continue
    shots = pd.concat(frames, ignore_index=True)
    shots = shots[shots.at_heave == 0].reset_index(drop=True)
    return add_cells(shots)


def add_cells(shots: pd.DataFrame) -> pd.DataFrame:
    x, y = shots.x.to_numpy(float) / 10.0, shots.y.to_numpy(float) / 10.0
    three = shots.value.to_numpy() == 3
    corner = three & (np.abs(x) >= 22.0) & (y <= 9.25)
    dist = shots.dist.to_numpy(float)
    spot = np.select([corner, three, dist <= 4, dist <= 10, dist <= 16], SPOTS[4:] + SPOTS[:3], SPOTS[3])
    after_oreb = shots.anchor_kind.to_numpy() == "oreb"
    since_anchor = shots.anchor_clock.to_numpy(float) - shots.clock.to_numpy(float)
    into = shots.poss_start_clock.to_numpy(float) - shots.clock.to_numpy(float)
    live = np.isin(shots.poss_start.to_numpy(), ["dreb", "dreb_team", "steal"])
    first = shots.att_no.to_numpy() == 1
    context = np.select(
        [after_oreb & (since_anchor <= 3), after_oreb | ~first, live & (into <= 7), into <= 7, into <= 14,
         into <= 20],
        CONTEXTS[:6], CONTEXTS[6])
    return shots.assign(spot=spot, context=context, cell=pd.Series(spot) + "|" + pd.Series(context))


def cell_rates(makes: pd.DataFrame, prior: pd.Series | None = None) -> pd.Series:
    """Assisted share of made shots by cell, padded CELL_PAD makes toward `prior` (the all-seasons share)."""
    grouped = makes.groupby("cell").post_assisted.agg(["sum", "size"])
    if prior is None:
        return grouped["sum"] / grouped["size"]
    prior = prior.reindex(grouped.index).fillna(grouped["sum"].sum() / grouped["size"].sum())
    return (grouped["sum"] + CELL_PAD * prior) / (grouped["size"] + CELL_PAD)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def shift_to_mean(base: np.ndarray, target_mean: float) -> float:
    """The logit shift d with mean(expit(logit(base) + d)) = target_mean, by bisection."""
    lo, hi, z = -6.0, 6.0, logit(base)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if expit(z + mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
