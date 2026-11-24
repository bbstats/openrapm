"""Lineup synergies, the career null: check 154's yardstick for every statistic's synergy term.

Every player is handed another player's statistic, season by season along that player's career (the donor's last
season when his career is shorter), and every reading is refitted.  A lineup term built from ANY number a player
carries picks up the misfit of the players who carry it, and a player's misfit, and a pair's, carries over from
season to season, so the spread of the per-season estimates understates the uncertainty; the null keeps that
carry-over (scripts/checks/154_self_creation.py, "career null").

The donor's number is his standing in his own season: his possession-weighted average over the season's folds,
minus the season's possession-weighted average, over its standard deviation; in a fold it is that standing times the
fold's player standard deviation, centered on the fold's players.  So a 2020 donor carries a 2020 standing, never a
2020 three-point rate into 1998.  A stand-in keeps the fold's own stand-in value.  The free terms and the miss are the
real ones.

    .venv/Scripts/python scratch/2026-11-23_synergy/null.py --data=outputs/synergy/rd143 --draws=200

Writes DATA/null_<season>.parquet (a season on disk is skipped) and DATA/null_summary.csv: each statistic and end, the
real across-season mean of the synergy per lineup sd against the null draws' means over the same seasons.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm.commands.console import one_thread, use_utf8

HERE = Path(__file__).resolve().parent
CHECKS = HERE.parents[1] / "scripts" / "checks"
SEED = 20261123


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scan = _load("synergy_scan", HERE / "scan.py")
TRAITS = _load("synergy_extract", HERE / "extract.py").TRAITS
SIDES = (("off", "O", "o", "poss_off"), ("def", "D", "d", "poss_def"))


class Careers:
    """Every player's standing in each statistic, season by season, both sides, from the saved fold tables."""

    def __init__(self, data: Path, first: int, last: int):
        frames = []
        for season in range(first, last + 1):
            players = pd.read_parquet(data / f"players_{season}.parquet")
            for _, side, _, poss_column in SIDES:
                rows = players[players.side == side]
                rows = rows[rows[poss_column] > 0]
                w = rows[poss_column].to_numpy(float)
                values = rows[list(TRAITS)].to_numpy(float) * w[:, None]
                sums = pd.DataFrame(values, columns=TRAITS).groupby(rows.player_id.to_numpy()).sum()
                weight = pd.Series(w).groupby(rows.player_id.to_numpy()).sum()
                mean = sums.div(weight, axis=0)
                centre = np.average(mean.to_numpy(), axis=0, weights=weight.to_numpy())
                spread = np.sqrt(np.average((mean.to_numpy() - centre) ** 2, axis=0, weights=weight.to_numpy()))
                standing = (mean - centre) / np.where(spread > 0, spread, 1.0)
                frames.append(standing.assign(season=season, side=side).rename_axis("player_id").reset_index())
        table = pd.concat(frames, ignore_index=True)
        self.players = np.sort(table.player_id.unique())
        self.place = pd.Index(self.players)
        seasons = table.drop_duplicates(["player_id", "season"])[["player_id", "season"]]
        seasons = seasons.sort_values(["player_id", "season"], kind="stable")
        seasons["step"] = seasons.groupby("player_id").cumcount()
        self.step = {(p, s): k for p, s, k in seasons[["player_id", "season", "step"]].itertuples(index=False)}
        length = seasons.groupby("player_id").step.max() + 1
        self.length = length.reindex(self.players).to_numpy(int)
        steps = int(self.length.max())
        self.values = {}
        for _, side, _, _ in SIDES:
            part = table[table.side == side].merge(seasons, on=["player_id", "season"])
            grid = np.full((len(self.players), steps, len(TRAITS)), np.nan)
            grid[self.place.get_indexer(part.player_id), part.step.to_numpy(int)] = part[list(TRAITS)].to_numpy(float)
            self.values[side] = grid

    def donated(self, side: str, permutation: np.ndarray, players: np.ndarray, season: int) -> np.ndarray:
        """players x traits: each player's donor's standing at the same step of the donor's career."""
        place = self.place.get_indexer(players)
        step = np.array([self.step.get((p, season), 0) for p in players])
        donor = permutation[place]
        at = np.minimum(step, self.length[donor] - 1)
        out = self.values[side][donor, at]
        return np.nan_to_num(out, nan=0.0)


def season_null(m154, careers: Careers, data: Path, season: int, permutations: np.ndarray) -> pd.DataFrame:
    stints = pd.read_parquet(data / f"stints_{season}.parquet")
    players = pd.read_parquet(data / f"players_{season}.parquet")
    folds, targets, frees, weights = [], [], [], []
    season_players = {side: np.sort(players[players.side == side].player_id.unique()) for _, side, _, _ in SIDES}
    for (repeat, fold), group in stints.groupby(["repeat", "fold"], sort=True):
        group = group.reset_index(drop=True)
        mine = players[(players.repeat == repeat) & (players.fold == fold)]
        projected, _, _, _ = scan.fold_columns(m154, group, mine, ())
        w = group.w.to_numpy(float)
        targets.append(projected["miss"])
        frees.append(np.column_stack([projected[name] for name in scan.FREE]))
        weights.append(w)
        sides = {}
        for end, side, prefix, poss_column in SIDES:
            values = scan.FoldValues(mine, side, TRAITS)
            rows = mine[(mine.side == side) & (mine[poss_column] > 0)].reset_index(drop=True)
            ids = group[[f"{prefix}{k}" for k in range(1, 6)]].to_numpy(np.int64)
            where = values.index.get_indexer(ids.ravel()).reshape(ids.shape)
            sides[end] = dict(
                where=where, order=np.searchsorted(season_players[side], rows.player_id.to_numpy()),
                poss=rows[poss_column].to_numpy(float),
                sd=np.array([values.sd[t] for t in TRAITS]), fill=np.array([values.fill[t] for t in TRAITS]))
        folds.append(dict(project=m154.projector(m154.team_levels(group), w), sides=sides))
    target, free, w = np.concatenate(targets), np.vstack(frees), np.concatenate(weights)
    root = np.sqrt(w)
    ff = (free * w[:, None]).T @ free
    fy = (free * w[:, None]).T @ target
    rows = []
    for draw, permutation in enumerate(permutations):
        donated = {side: careers.donated(side, permutation, season_players[side], season) for _, side, _, _ in SIDES}
        terms, raws = [], []
        for fold in folds:
            columns, raw = [], []
            for end, side, _, _ in SIDES:
                info = fold["sides"][end]
                standing = donated[side][info["order"]]
                value = standing * info["sd"]
                value = value - np.average(value, axis=0, weights=info["poss"])
                where = info["where"]
                lineup = np.where(where[..., None] >= 0, value[np.maximum(where, 0)], info["fill"])
                total, own = lineup.sum(axis=1), (lineup ** 2).sum(axis=1)
                cross = (total ** 2 - own) / 2.0
                columns += [total, own, cross]
                raw.append(cross)
            block = np.hstack(columns)
            terms.append(fold["project"](block))
            raws.append(np.hstack(raw))
        x_all, cross_raw = np.vstack(terms), np.vstack(raws)
        n_traits = len(TRAITS)
        for k, trait in enumerate(TRAITS):
            picks = [e * 3 * n_traits + t * n_traits + k for e in range(2) for t in range(3)]
            x = x_all[:, picks]
            xw = x * w[:, None]
            gram = np.block([[xw.T @ x, xw.T @ free], [(xw.T @ free).T, ff]])
            rhs = np.concatenate([xw.T @ target, fy])
            coef = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            for e, (end, _, _, _) in enumerate(SIDES):
                scale = scan.weighted_sd(cross_raw[:, e * n_traits + k], w)
                sign = 1.0 if end == "off" else -1.0
                rows.append(dict(season=season, draw=draw, end=end, trait=trait,
                                 value=float(sign * coef[3 * e + 2] * scale)))
    del root
    return pd.DataFrame(rows)


def summary(data: Path, first: int, last: int) -> pd.DataFrame:
    real = pd.concat([pd.read_parquet(data / f"scan_{s}.parquet") for s in range(first, last + 1)])
    real = real[(real.analysis == "single") & (real.term == "synergy per lineup sd")]
    null = pd.concat([pd.read_parquet(data / f"null_{s}.parquet") for s in range(first, last + 1)
                      if (data / f"null_{s}.parquet").exists()])
    rows = []
    for (end, trait), group in null.groupby(["end", "trait"]):
        mine = real[(real.end == end) & (real.trait == trait)].set_index("season").value
        draws = group.pivot(index="draw", columns="season", values="value")
        for label, lo, hi in scan.ERAS:
            seasons = [s for s in draws.columns if lo <= s <= hi]
            value = float(mine[seasons].mean())
            means = draws[seasons].mean(axis=1).to_numpy()
            centre, spread = float(means.mean()), float(means.std(ddof=1))
            rows.append(dict(end=end, trait=trait, era=label, mean=value, null_mean=centre, null_sd=spread,
                             z_against_null=(value - centre) / spread,
                             share_of_draws_as_far=float(np.mean(np.abs(means - centre) >= abs(value - centre))),
                             draws=len(means), seasons=len(seasons)))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data", default="outputs/synergy/rd143")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--first", type=int, default=1997)
    parser.add_argument("--last", type=int, default=2026)
    parser.add_argument("--only", default="", help="first:last, the seasons this process scores")
    args = parser.parse_args(argv)
    m154 = _load("selfcreation_check", CHECKS / "154_self_creation.py")
    data = Path(args.data)
    lo, hi = (int(x) for x in args.only.split(":")) if args.only else (args.first, args.last)
    started = time.time()
    with one_thread():
        careers = Careers(data, args.first, args.last)
        rng = np.random.default_rng(SEED)
        permutations = np.array([rng.permutation(len(careers.players)) for _ in range(args.draws)])
        print(f"careers of {len(careers.players):,} players ({time.time() - started:.0f}s)", flush=True)
        for season in range(lo, hi + 1):
            out = data / f"null_{season}.parquet"
            if out.exists():
                continue
            season_null(m154, careers, data, season, permutations).to_parquet(out, index=False)
            print(f"  {season}: {args.draws} draws ({time.time() - started:.0f}s)", flush=True)
    if not args.only:
        summary(data, args.first, args.last).to_csv(data / "null_summary.csv", index=False)
        print(f"done in {time.time() - started:.0f}s: {data / 'null_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
