"""Adversarial checks on one assumption: "a free level per team-season absorbs coach, system, schedule, team luck and
anything else the whole roster shares" (the owner, 2026-11-24: "I want you to adversarially validate this
assumption").  Read-only; no rating moves.

The fit is team_vs_player.py's within-team one: the Bias page's "vs game results, last 10 seasons" (2018-2027, each
season against its own published ratings, its own width, level and home court) with a free level per team and side.
A team-season level is a constant, so it takes away only what is constant over a team's season.  Each variant below
takes away one thing it leaves:

    season levels               the fit as reported, to reproduce team_vs_player.py
    no garbage time             garbage-time stints out of the team-games
    close score                 only stints that start with the margin within 5: minutes that follow a lopsided score
                                (stars rested when far ahead, kept in when behind) cannot drive the share
    quarter levels              a level per team and side for each quarter of its regular season in date order, the
                                playoffs one more: coaching changes, trades, other players' injuries, form
    eighth levels               the same in eighths
    no back-to-backs            team-games where either team plays the second night of a back-to-back out: rest
    no final quarter            each team's last quarter of its regular season and the playoffs out: shutdowns,
                                tanking, playoff rotations

The fit is solved by partialling out (Frisch-Waugh): every season's own columns (its predictions, fixed columns and
levels) are taken out of the miss and the stat columns season by season, which is exact because no two seasons share
one, then the two stat coefficients are fitted on what is left over all seasons.  OpenRAPM minus the games = b_D - b_O,
points per 100 per standard deviation; its error leaves one team out of every season at once.

    .venv/Scripts/python scratch/2026-11-24_bias_split/adversarial.py [--table=season_ratings_product_step3.parquet]
                                                                       [--variants="season levels,close score"]

--table reads another ratings table in outputs/: the rating before the shrink, step 3's, asks whether a lean is
OpenRAPM's error or the shrink holding a season's rating back from its own games on purpose.
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from functools import partial

import numpy as np
import pandas as pd

from openrapm import game_results, trade_set
from openrapm.bias import STATS
from openrapm.commands.site import panel_of_season, season_designs, shown_z
from openrapm.data.config import load_config, resolve
from openrapm.data.tables import team_of_rows

SEASONS = list(range(2018, 2028))
BOX = STATS[:13]            # on-court and off-court plus-minus are counted from the same games: circular here
FREE_WIDTH = True          # --fixed-width: each season's prediction at OpenRAPM's own width, no multiplier
CLOSE = 5.0                 # points: a close-score stint starts with the margin within this
OUT = "outputs/bias_split_adversarial{}.csv"


def filtered(design, keep):
    """The design's rows where `keep` is true."""
    Z, F = trade_set._player_block(design), trade_set._fixed_block(design)
    return dataclasses.replace(design, rows=design.rows[keep].reset_index(drop=True), y=design.y[keep],
                               w=design.w[keep], parts={"Z": Z[keep], "F": F[keep]})


def stint_mask(design, rows: str):
    """Which stints a variant keeps: all, those outside garbage time, or those that start with the score close."""
    if rows == "no garbage time":
        return ~design.rows.is_gt.to_numpy().astype(bool)
    if rows == "close score":
        names = list(design.spec.context_names)
        margin = trade_set._fixed_block(design)[:, names.index("margin")]
        return np.abs(margin) <= CLOSE
    return np.ones(len(design.rows), dtype=bool)


@dataclasses.dataclass
class Season:
    """One season's team-games for one way of keeping stints: the games, each row's date and phase, its teams, and
    the stat columns of every statistic."""
    games: game_results.SeasonGames
    dates: np.ndarray
    phase: np.ndarray
    stats: dict


def load(cfg, table, season: int, rows: str) -> Season:
    designs = season_designs(cfg, season)
    if rows != "all":
        mask = stint_mask(designs["offense"], rows)
        designs = {side: filtered(design, mask) for side, design in designs.items()}
    dates = designs["offense"].games.set_index("game_idx").game_date
    games = game_results.season_games(designs, table, season, partial(team_of_rows, cfg=cfg))
    keys = games.games["offense"].keys
    z = pd.DataFrame({"player_id": games.player_ids}).merge(
        shown_z(table, season, panel_of_season(cfg, season)), on="player_id", how="left").fillna(0.0)
    ts = games.games["offense"]
    n = ts.n_players
    stats = {stat: np.column_stack([ts.Z[:, :n] @ z[stat].to_numpy(float), ts.Z[:, n:] @ z[stat].to_numpy(float)])
             for stat in BOX}
    return Season(games, pd.to_datetime(dates.reindex(keys.game_idx).to_numpy()), keys.phase.to_numpy(), stats)


def positions(season: Season) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per row, each of its two teams' share of their regular season played before this game (0 to 1; 2 in the
    playoffs) and whether that team plays the second night of a back-to-back."""
    teams = season.games.teams
    out = {}
    for column in ("team_off", "team_def"):
        frame = pd.DataFrame({"team": teams[column].to_numpy(), "date": season.dates, "phase": season.phase})
        share = np.full(len(frame), 2.0)
        second = np.zeros(len(frame), dtype=bool)
        for team, part in frame.groupby("team"):
            regular = part[part.phase == "RS"]
            days = np.sort(regular.date.unique())
            rank = {day: i / len(days) for i, day in enumerate(days)}
            share[regular.index] = regular.date.map(rank).to_numpy()
            every = np.sort(part.date.unique())
            back = {day for day, before in zip(every[1:], every[:-1]) if (day - before) == np.timedelta64(1, "D")}
            second[part.index] = part.date.isin(back).to_numpy()
        out[column] = (share, second)
    return out["team_off"][0], out["team_def"][0], out["team_off"][1], out["team_def"][1]


def levels(season: Season, blocks: int) -> np.ndarray:
    """The free levels: one per team and side, per block of its regular season in date order (the playoffs one
    more), one per side left as the season's own level."""
    share_off, share_def, _, _ = positions(season)
    columns = []
    for team, share in ((season.games.teams.team_off.to_numpy(), share_off),
                        (season.games.teams.team_def.to_numpy(), share_def)):
        block = np.where(share >= 2.0, blocks, np.minimum((share * blocks).astype(int), blocks - 1))
        label = pd.Series([f"{t}:{b}" for t, b in zip(team, block)])
        names = sorted(label.unique())[1:]
        columns.append((label.to_numpy()[:, None] == np.array(names)[None, :]).astype(float))
    return np.column_stack(columns)


def keep_rows(season: Season, rows: str) -> np.ndarray:
    share_off, share_def, second_off, second_def = positions(season)
    if rows == "no back-to-backs":
        return ~(second_off | second_def)
    if rows == "no final quarter":
        return (share_off < 0.75) & (share_def < 0.75)
    return np.ones(len(share_off), dtype=bool)


def residuals(season: Season, blocks: int, keep: np.ndarray) -> dict:
    """Every season's own columns taken out, weighted: the two misses and every stat's two columns, on `keep`."""
    ts_off, ts_def = season.games.games["offense"], season.games.games["defense"]
    width = [trade_set.rating_columns(ts_off, season.games.offset)] if FREE_WIDTH else []
    own = np.column_stack([*width, ts_off.F[:, np.any(ts_off.F != 0.0, axis=0)], levels(season, blocks)])
    root = np.sqrt(ts_off.w)
    misses = np.column_stack([ts_off.y - ts_off.Z @ season.games.offset, ts_def.y - ts_def.Z @ season.games.offset])
    right = np.column_stack([misses, *season.stats.values()]) * root[:, None]
    left = (own * root[:, None])[keep]
    left = left[:, np.any(left != 0.0, axis=0)]
    solved = np.linalg.lstsq(left, right[keep], rcond=None)[0]
    return right[keep] - left @ solved


def leans(parts: list) -> np.ndarray:
    """b_D - b_O per statistic over the seasons' residuals."""
    stacked = np.vstack(parts)
    off_miss, def_miss = stacked[:, 0], stacked[:, 1]
    out = []
    for position in range(len(BOX)):
        columns = stacked[:, 2 + 2 * position: 4 + 2 * position]
        b_off = np.linalg.lstsq(columns, off_miss, rcond=None)[0][0]
        b_def = np.linalg.lstsq(columns, def_miss, rcond=None)[0][1]
        out.append(b_def - b_off)
    return np.array(out)


def variant(seasons: dict, rows: str, blocks: int) -> pd.DataFrame:
    keeps = {s: keep_rows(season, rows) for s, season in seasons.items()}
    value = leans([residuals(season, blocks, keeps[s]) for s, season in seasons.items()])
    teams = np.unique(np.concatenate([np.concatenate([season.games.teams.team_off, season.games.teams.team_def])
                                      for season in seasons.values()]))
    left_out = []
    for team in teams:
        parts = []
        for s, season in seasons.items():
            others = (season.games.teams.team_off.to_numpy() != team) & (season.games.teams.team_def.to_numpy() != team)
            parts.append(residuals(season, blocks, keeps[s] & others))
        left_out.append(leans(parts))
    left_out = np.array(left_out)
    k = len(left_out)
    error = np.sqrt((k - 1) / k * np.sum((left_out - left_out.mean(axis=0)) ** 2, axis=0))
    rows_kept = sum(int(keep.sum()) for keep in keeps.values())
    return pd.DataFrame({"stat": BOX, "lean": value, "error": error, "team_games": rows_kept})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="season_ratings_product.parquet")
    parser.add_argument("--variants", default="")
    parser.add_argument("--fixed-width", action="store_true")
    args = parser.parse_args()
    global FREE_WIDTH
    FREE_WIDTH = not args.fixed_width
    cfg = load_config()
    table = pd.read_parquet(resolve(cfg, "outputs") / args.table)
    started = time.time()
    plan = [("season levels", "all", 1), ("no garbage time", "no garbage time", 1), ("close score", "close score", 1),
            ("quarter levels", "all", 4), ("eighth levels", "all", 8), ("no back-to-backs", "no back-to-backs", 1),
            ("no final quarter", "no final quarter", 1)]
    if args.variants:
        plan = [step for step in plan if step[0] in args.variants.split(",")]
    loaded = {}
    frames = []
    for name, rows, blocks in plan:
        stints = rows if rows in ("no garbage time", "close score") else "all"
        if stints not in loaded:
            loaded[stints] = {s: load(cfg, table, s, stints) for s in SEASONS}
            print(f"loaded {stints} ({time.time() - started:.0f}s)", flush=True)
        frame = variant(loaded[stints], rows, blocks).assign(variant=name)
        frames.append(frame)
        print(f"{name} ({time.time() - started:.0f}s)", flush=True)
    out = pd.concat(frames, ignore_index=True)
    tag = "" if args.table == "season_ratings_product.parquet" else "_" + args.table.split(".")[0]
    tag += "_fixed_width" if args.fixed_width else ""
    out.to_csv(OUT.format(tag), index=False)
    wide = out.pivot_table(index="stat", columns="variant", values=["lean", "error"], sort=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(wide["lean"][[name for name, _, _ in plan]].round(2).to_string())
        print(wide["error"][[name for name, _, _ in plan]].round(2).to_string())
        print(out.groupby("variant", sort=False).team_games.first().to_string())


if __name__ == "__main__":
    main()
