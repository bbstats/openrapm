"""Read-only: which team numbers are left once the direct +/- is out (the owner, 2026-11-03: "can we just run the test
again but eliminate the +/- direct type stuff? ... just figure it out"; notes/DECISIONS.md, experiment 128).

Experiment 128's team rates with him on the court are results of his own lineups, and team makes are his on-court points
split by shot type.  The candidates, from least to most removed from those results:

    all eight        the 8 team rates with him on and off, the list that failed in the full build
    no makes         team threes, twos and free throws attempted, ORB and TOV, him on and off
    no makes, on     the same, him on the court only
    no makes, off    the same, him on the bench only: no possession of his own in them
    eight, off       the 8 team rates, him on the bench only
    teammates on     the 13 box rates of the teammates he shares the court with, each teammate's own padded season rate
                     weighted by shared possessions: who he plays with and not how those lineups came out, built as the
                     context numbers are built from priors (second_prior._Entries)
    teammates both   the same and his team's five when he sits

Each on the swaps of creation110's folds with inputs from the fit games, two targets: `d`, the line's own, the swap with
the whole ratings added back; `d_prior`, with only the prior parts added back, so the games' part stays out of what the
line may claim (measurement trap: "The line's target adds the games' part back").  Each season predicted by the line
fitted without it and its neighbors; the change from adding the list to today's 29 inputs, below 0 is better.  Shuffled
copies, among a fold's players, are the noise yardstick.

    .venv/Scripts/python scratch/2026-11-03_team_rates/read_no_plus_minus.py
"""
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from openrapm import prior_shrink, second_prior, swap_adjust, swap_test
from openrapm.commands.build import season_lineup_rows, step4_folds, step4_multipliers
from openrapm.data.config import load_config, resolve
from openrapm.data.within_folds import fold_games
from openrapm.features import BOX_RATES
from openrapm.second_prior import _Entries, pad_toward_level
from openrapm.settings import TEST
from openrapm.swap_test import DEF, OFF, TEAM_COUNTS

started = time.time()
SEASONS = list(range(1997, 2027))
ERAS = {"1997-2006": range(1997, 2007), "2007-2016": range(2007, 2017), "2017-2026": range(2017, 2027)}
TARGETS = ("d", "d_prior")
COUNTS = [*TEAM_COUNTS, "threes_taken", "twos_taken", "free_throws_taken"]
PROFILE = ["threes_taken", "twos_taken", "free_throws_taken", "off_rebounds", "turnovers"]
cfg = load_config()
directory = resolve(cfg, "outputs") / "second_prior" / "ctx110"
TODAY = list(json.loads((directory / "run.json").read_text())["inputs"])
shrink = replace(TEST.shrink, folds_tag="creation110_early,creation110_within")
folds, where = step4_folds(cfg, shrink)
multipliers = step4_multipliers(sorted({fold.season for fold in folds}), folds, shrink)
print("folds loaded", len(folds), round(time.time() - started), flush=True)


def with_attempts(rows):
    return rows.assign(threes_taken=rows.threes_made + rows.threes_missed, twos_taken=rows.twos_made + rows.twos_missed,
                       free_throws_taken=rows.free_throws_made + rows.free_throws_missed)


def team_numbers(entries, rows, side):
    """{name: per player}: each count's team rate with him on and off on one end, padded as the module pads."""
    n_rows = len(rows)
    counted = np.zeros((2 * n_rows, len(COUNTS)))
    weight = np.zeros(2 * n_rows)
    at = slice(0, n_rows) if side == "O" else slice(n_rows, 2 * n_rows)
    counted[at] = 100.0 * rows[COUNTS].to_numpy(dtype=float)
    weight[at] = rows.poss.to_numpy(dtype=float)
    team_weight = np.asarray(entries.teams.T @ weight).ravel()
    team_rate = np.asarray(entries.teams.T @ counted) / np.where(team_weight > 0, team_weight, 1.0)[:, None]
    level = np.asarray(entries.mix @ team_rate) / np.where(entries.on_weight > 0, entries.on_weight, 1.0)[:, None]
    out = {}
    for name, total in (("on", entries.on), ("off", entries.bench)):
        behind = np.maximum(total(weight), 0.0)
        rate = total(counted) / np.where(behind > 0, behind, 1.0)[:, None]
        spread = (total(counted * counted) - 2.0 * rate * total(counted * weight[:, None])
                  + rate * rate * total(weight * weight)[:, None])
        noise = np.maximum(spread, 0.0).sum(axis=0) / max(float(behind.sum()), 1e-12)
        for j, count in enumerate(COUNTS):
            out[f"team_{count}_{name}"] = pad_toward_level(rate[:, j], behind, level[:, j], float(noise[j]))
    return out


def teammate_numbers(entries, values, keep):
    """{name: per player}: each stat of `values` (players x stats) over his teammates on the court and over his team's
    five when he sits, padded toward the team's level, as second_prior.context_numbers does with priors."""
    raw, levels = entries.numbers(values)
    out = {}
    on, bench = entries.on_weight, entries.bench_weight
    for j, stat in enumerate(BOX_RATES):
        noise = entries.noise(values[:, j], keep)
        short = stat.replace("_per100", "")
        out[f"mates_{short}_on"] = pad_toward_level(raw["teammates_{}_on"][:, j], on, levels["teammates"][:, j],
                                                    noise["teammates_{}_on"])
        out[f"mates_{short}_bench"] = pad_toward_level(raw["teammates_{}_bench"][:, j], bench, levels["teammates"][:, j],
                                                       noise["teammates_{}_bench"])
    return out


store, spread = {}, {}
rng = np.random.default_rng(0)
names = None
for season in SEASONS:
    rows = with_attempts(season_lineup_rows(cfg, season))
    game = rows.game_id.astype(str)
    stored = pd.read_parquet(directory / f"inputs_{season}.parquet")
    factors = prior_shrink.season_factors(multipliers, season)
    for fold in (fold for fold in folds if fold.season == season):
        fit_ids, held_ids = fold_games(where[fold.stem], fold)
        fit_rows = rows[game.isin(fit_ids).to_numpy()].reset_index(drop=True)
        ids = np.unique(np.concatenate([fit_rows[OFF].to_numpy(dtype=np.int64).ravel(),
                                        fit_rows[DEF].to_numpy(dtype=np.int64).ravel()]))
        entries = _Entries(fit_rows, ids)
        here = stored[(stored.repeat == fold.repeat) & (stored.fold == fold.fold)]
        own = here[here.side == "O"].set_index("player_id")[BOX_RATES].astype(float).reindex(ids).fillna(0.0).to_numpy()
        keep = np.flatnonzero(entries.on_weight > 0)
        mates = pd.DataFrame(teammate_numbers(entries, own, keep), index=ids)
        held = swap_test.prepare(rows[game.isin(held_ids).to_numpy()].reset_index(drop=True), season)
        table = prior_shrink.shrink_fold(fold.rows, season, factors)
        ratings, priors = swap_test.ratings_in_raw_sign(table), swap_test.ratings_in_raw_sign(table, "prior")
        pairs = swap_adjust.residual_pairs(held, ratings, swap_adjust.RESIDUAL_CONTEXT)
        whole, prior_only = second_prior.swap_targets(pairs, ratings), second_prior.swap_targets(pairs, priors)
        for long_side, side in second_prior.SIDE_KEYS.items():
            inputs = here[here.side == side].set_index("player_id")[TODAY].astype(float)
            team = pd.DataFrame(team_numbers(entries, fit_rows, side), index=ids).reindex(inputs.index)
            candidates = team.join(mates.reindex(inputs.index))
            candidates = candidates.fillna(candidates.mean())
            shadow = candidates.copy()
            shadow.iloc[:, :] = np.column_stack([rng.permutation(shadow.iloc[:, j].to_numpy())
                                                 for j in range(shadow.shape[1])])
            shadow.columns = ["shadow_" + c for c in shadow.columns]
            frame = pd.concat([inputs, candidates, shadow], axis=1)
            names = list(frame.columns)
            s = spread.setdefault(side, dict(n=0, s1=np.zeros(len(names)), s2=np.zeros(len(names))))
            s["n"] += len(frame)
            s["s1"] += frame.to_numpy().sum(axis=0)
            s["s2"] += (frame.to_numpy() ** 2).sum(axis=0)
            used = (pairs[long_side].fifth_a.isin(frame.index) & pairs[long_side].fifth_b.isin(frame.index)).to_numpy()
            swaps = pairs[long_side][used]
            dx = (frame.reindex(swaps.fifth_a.to_numpy()).to_numpy(float)
                  - frame.reindex(swaps.fifth_b.to_numpy()).to_numpy(float))
            h = swaps.h.to_numpy(float)
            aim = {"d": whole[long_side].r.to_numpy(float)[used], "d_prior": prior_only[long_side].r.to_numpy(float)[used]}
            sums = store.setdefault((season, side), {"G": 0.0, "sh": 0.0, **{t: [np.zeros(len(names)), 0.0] for t in TARGETS}})
            sums["G"] = sums["G"] + (dx * h[:, None]).T @ dx
            sums["sh"] += float(h.sum())
            for target in TARGETS:
                sums[target][0] = sums[target][0] + dx.T @ (h * aim[target])
                sums[target][1] += float((h * aim[target] ** 2).sum())
    print(season, round(time.time() - started), flush=True)

position = {name: at for at, name in enumerate(names)}


def scales(side):
    s = spread[side]
    sd = np.sqrt(np.maximum(s["s2"] / s["n"] - (s["s1"] / s["n"]) ** 2, 1e-12))
    return np.array([1.0 if name in TODAY else 1.0 / sd[position[name]] for name in names])


def fit(side, seasons, columns, target):
    f = scales(side)
    at = [position[c] for c in columns]
    gram = sum(store[(y, side)]["G"] for y in seasons)[np.ix_(at, at)] * np.outer(f[at], f[at])
    right = sum(store[(y, side)][target][0] for y in seasons)[at] * f[at]
    return np.linalg.solve(gram + 1e-4 * np.mean(np.diag(gram)) * np.eye(len(at)), right)


def loss(side, season, columns, target, weights):
    f = scales(side)
    at = [position[c] for c in columns]
    sums = store[(season, side)]
    gram = sums["G"][np.ix_(at, at)] * np.outer(f[at], f[at])
    return (sums[target][1] - 2.0 * weights @ (sums[target][0][at] * f[at]) + weights @ gram @ weights) / sums["sh"]


def held_out(side, columns, target):
    return np.array([loss(side, y, columns, target, fit(side, [o for o in SEASONS if abs(o - y) > 1], columns, target))
                     for y in SEASONS])


def line(change):
    z = change.mean() / (change.std(ddof=1) / np.sqrt(len(change)))
    eras = " | ".join("%+.3f %+.2f" % (change[[SEASONS.index(y) for y in ys]].mean(),
                                       change[[SEASONS.index(y) for y in ys]].mean()
                                       / (change[[SEASONS.index(y) for y in ys]].std(ddof=1) / np.sqrt(len(ys))))
                      for ys in ERAS.values())
    return f"{change.mean():+.3f} | {z:+.2f} | {int((change < 0).sum())} | {eras}"


def on(stats):
    return [f"team_{s}_on" for s in stats]


def off(stats):
    return [f"team_{s}_off" for s in stats]


def shuffled(columns):
    return ["shadow_" + c for c in columns]


STATS = [rate.replace("_per100", "") for rate in BOX_RATES]
MATES_ON = [f"mates_{s}_on" for s in STATS]
MATES_BOTH = [*MATES_ON, *[f"mates_{s}_bench" for s in STATS]]
LISTS = {"all eight, on and off": on(TEAM_COUNTS) + off(TEAM_COUNTS),
         "no makes, on and off": on(PROFILE) + off(PROFILE),
         "SHUFFLED no makes, on and off": shuffled(on(PROFILE) + off(PROFILE)),
         "no makes, him on": on(PROFILE), "no makes, him off": off(PROFILE),
         "all eight, him off": off(TEAM_COUNTS),
         "teammates on": MATES_ON, "teammates on and his team when he sits": MATES_BOTH,
         "SHUFFLED teammates on and his team when he sits": shuffled(MATES_BOTH)}
for side, label in (("O", "OFFENSE"), ("D", "DEFENSE")):
    for target in TARGETS:
        today = held_out(side, TODAY, target)
        print(f"\n===== {label}, target {target}: list | change | z | seasons better of 30 | 1997-2006 change z | "
              "2007-2016 change z | 2017-2026 change z")
        for name, columns in LISTS.items():
            print(f"{name} | {line(held_out(side, [*TODAY, *columns], target) - today)}")
    print(f"\n{label}: weights per sd on target d_prior, teammates on and his team when he sits: stat | on | when he sits")
    both = dict(zip([*TODAY, *MATES_BOTH], fit(side, SEASONS, [*TODAY, *MATES_BOTH], "d_prior")))
    for s in STATS:
        print(f"{s} | {both[f'mates_{s}_on']:+.3f} | {both[f'mates_{s}_bench']:+.3f}")
print("done", round(time.time() - started), flush=True)
