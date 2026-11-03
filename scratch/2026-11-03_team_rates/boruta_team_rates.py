"""Read-only: Boruta on the second stage with the team's rates in (notes/DECISIONS.md, experiment 128; the owner,
2026-10-21: "you need to make sure you are boruta-ing both steps").

The candidates, per side: today's 29 inputs of the line, as the shipped line ctx110 read them on each fold
(outputs/second_prior/ctx110/inputs_<season>.parquet), and the 16 team rates from the fold's fit games
(second_prior.team_rate_numbers).  The swaps: every lineup swap of the held-back games of repeat 0 of creation110's
within-season folds, all thirty seasons, the target and weights the line is fitted on (second_prior.swap_targets, the
swap's information h), the fold's step-4 ratings at the test setting.

Each round adds a shadow of every candidate, its values shuffled among the fold's players, fits the line on all 90 by
weighted least squares on the swaps, and scores each input by its mean absolute Shapley value on the swaps, |weight| x
the weighted mean |difference between the two swapped players|; a hit is a candidate above the best shadow.  Twenty
rounds; accepted when the hits are significantly more than half (two-sided binomial, 5%), rejected when significantly
fewer, tentative between.  Per side, over all seasons and in each era, as scratch/2026-10-21_boruta2/boruta_line.py.

The registered rule: a stat leaves the list only if both its numbers, him on and him off, are rejected on both ends in
every era.  Writes outputs/csv/boruta_team128.csv.

    .venv/Scripts/python scratch/2026-11-03_team_rates/boruta_team_rates.py
"""
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from openrapm import prior_shrink, second_prior, swap_adjust, swap_test
from openrapm.commands.build import season_lineup_rows, step4_folds, step4_multipliers
from openrapm.data.config import load_config, resolve
from openrapm.data.within_folds import fold_games
from openrapm.settings import TEST

ROUNDS = 20
ERAS = [("1997-2026", 1997, 2026), ("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026)]
TODAY = second_prior.INPUT_SETS["context_and_types_centred"]
NAMES = [*TODAY, *second_prior.TEAM_RATES]


def collect(cfg: dict) -> dict:
    """{side: (player matrix, fold of each player row, a, b, d, h, season of each swap)}."""
    started = time.time()
    shrink = replace(TEST.shrink, folds_tag="creation110_early,creation110_within")
    folds, where = step4_folds(cfg, shrink)
    multipliers = step4_multipliers(sorted({fold.season for fold in folds}), folds, shrink)
    chosen = [fold for fold in folds if fold.repeat == 0]
    store = {side: dict(x=[], fold=[], a=[], b=[], d=[], h=[], season=[], offset=0) for side in ("O", "D")}
    directory = resolve(cfg, "outputs") / "second_prior" / "ctx110"
    current, rows, stored = None, None, None
    for number, fold in enumerate(chosen):
        season = fold.season
        if season != current:
            rows = season_lineup_rows(cfg, season)
            stored = pd.read_parquet(directory / f"inputs_{season}.parquet")
            current = season
        game = rows.game_id.astype(str)
        fit_ids, held_ids = fold_games(where[fold.stem], fold)
        team_rates = second_prior.team_rate_numbers(rows[game.isin(fit_ids).to_numpy()].reset_index(drop=True))
        held = swap_test.prepare(rows[game.isin(held_ids).to_numpy()].reset_index(drop=True), season)
        ratings = swap_test.ratings_in_raw_sign(
            prior_shrink.shrink_fold(fold.rows, season, prior_shrink.season_factors(multipliers, season)))
        pairs = second_prior.swap_targets(swap_adjust.residual_pairs(held, ratings, swap_adjust.RESIDUAL_CONTEXT),
                                          ratings)
        here = stored[(stored.repeat == fold.repeat) & (stored.fold == fold.fold)]
        for long_side, side in second_prior.SIDE_KEYS.items():
            inputs = here[here.side == side].set_index("player_id")[TODAY].astype(float)
            frame = second_prior.with_team_rates(inputs, team_rates[side])[NAMES]
            s = store[side]
            position = pd.Series(np.arange(len(frame)) + s["offset"], index=frame.index)
            swaps = pairs[long_side]
            swaps = swaps[swaps.fifth_a.isin(frame.index) & swaps.fifth_b.isin(frame.index)]
            s["x"].append(frame.to_numpy(float))
            s["fold"].append(np.full(len(frame), number))
            s["a"].append(position.reindex(swaps.fifth_a).to_numpy())
            s["b"].append(position.reindex(swaps.fifth_b).to_numpy())
            s["d"].append(swaps.r.to_numpy(float))
            s["h"].append(swaps.h.to_numpy(float))
            s["season"].append(np.full(len(swaps), season))
            s["offset"] += len(frame)
        if number % 12 == 0:
            print(f"  {number + 1} of {len(chosen)} folds ({time.time() - started:.0f}s)", flush=True)
    return {side: dict(x=np.vstack(s["x"]), fold=np.concatenate(s["fold"]), a=np.concatenate(s["a"]),
                       b=np.concatenate(s["b"]), d=np.concatenate(s["d"]), h=np.concatenate(s["h"]),
                       season=np.concatenate(s["season"])) for side, s in store.items()}


def fit(delta: np.ndarray, d: np.ndarray, h: np.ndarray) -> np.ndarray:
    gram = (delta * h[:, None]).T @ delta
    gram += 1e-4 * np.mean(np.diag(gram)) * np.eye(len(gram))
    return np.linalg.solve(gram, (delta * h[:, None]).T @ d)


def shuffled(x: np.ndarray, fold: np.ndarray, rng) -> np.ndarray:
    """Each column's values shuffled among the players of each fold."""
    out = x.copy()
    for f in np.unique(fold):
        rows = np.flatnonzero(fold == f)
        for j in range(x.shape[1]):
            out[rows, j] = x[rng.permutation(rows), j]
    return out


def boruta(data: dict, mask: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x, a, b = data["x"], data["a"][mask], data["b"][mask]
    d, h = data["d"][mask], data["h"][mask]
    real = x[a] - x[b]
    n = x.shape[1]
    hits, importance, weights = np.zeros(n), np.zeros(n), np.zeros(n)
    for _ in range(ROUNDS):
        shadow = shuffled(x, data["fold"], rng)
        delta = np.hstack([real, shadow[a] - shadow[b]])
        beta = fit(delta, d, h)
        score = np.abs(beta) * (np.abs(delta) * h[:, None]).sum(axis=0) / h.sum()
        hits += score[:n] > score[n:].max()
        importance += score[:n]
        weights += beta[:n]
    rows = []
    for j, name in enumerate(NAMES):
        k = int(hits[j])
        p = binomtest(k, ROUNDS, 0.5).pvalue
        verdict = "tentative" if p >= 0.05 else ("accepted" if k > ROUNDS / 2 else "rejected")
        rows.append(dict(input=name, hits=k, verdict=verdict, importance=importance[j] / ROUNDS,
                         weight=weights[j] / ROUNDS))
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    data = collect(cfg)
    tables = []
    for side, label in (("D", "defense"), ("O", "offense")):
        started = time.time()
        for era, first, last in ERAS:
            mask = (data[side]["season"] >= first) & (data[side]["season"] <= last)
            table = boruta(data[side], mask, seed=first).assign(side=label, era=era)
            tables.append(table)
            print(f"{label} {era}: {int(mask.sum()):,} swaps ({time.time() - started:.0f}s)", flush=True)
    long = pd.concat(tables, ignore_index=True)
    wide = long.pivot(index=["side", "input"], columns="era", values="verdict").reset_index()
    every = long[long.era == "1997-2026"][["side", "input", "importance", "weight", "hits"]]
    out = every.merge(wide, on=["side", "input"]).sort_values(["side", "importance"], ascending=[True, False])
    path = resolve(cfg, "outputs") / "csv" / "boruta_team128.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
