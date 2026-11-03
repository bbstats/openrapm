"""Read-only: why the team rates win on the line's own target and lose after the ridge (notes/DECISIONS.md, experiment 128).

The line is fitted on d, a swap's points per 100 in the held-back games less only the opponents and the context, so it
may use anything that forecasts those games, the ridge's own evidence included.  Step 3 then reads the same fit games a
second time.  The question is how much of what the team rates forecast the fold's whole rating, games' part included,
already holds.  Three targets on the same swaps, all from creation110's folds, inputs from the fit games:

    d        the line's target: the swap's difference with the fold's two players' whole ratings added back
    d_prior  the same with only their prior parts added back, so the games' part stays out of what the line may claim
    r        what the fold's whole rating leaves unexplained

and three sets of inputs: today's 29, today's 29 and the 16 team rates, the 16 team rates alone.  Each season is predicted
by the line fitted without it and its neighbors; below 0 is better, against no line at all on r and against today's 29 on
d and d_prior.

    .venv/Scripts/python scratch/2026-11-03_team_rates/why_it_lost.py
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
from openrapm.settings import TEST

started = time.time()
SEASONS = list(range(1997, 2027))
ERAS = {"1997-2006": range(1997, 2007), "2007-2016": range(2007, 2017), "2017-2026": range(2017, 2027)}
TARGETS = ("d", "d_prior", "r")
RATES = second_prior.TEAM_RATES
cfg = load_config()
directory = resolve(cfg, "outputs") / "second_prior" / "ctx110"
TODAY = list(json.loads((directory / "run.json").read_text())["inputs"])
NAMES = [*TODAY, *RATES]
position = {name: at for at, name in enumerate(NAMES)}
shrink = replace(TEST.shrink, folds_tag="creation110_early,creation110_within")
folds, where = step4_folds(cfg, shrink)
multipliers = step4_multipliers(sorted({fold.season for fold in folds}), folds, shrink)
store = {}
for season in SEASONS:
    rows = season_lineup_rows(cfg, season)
    game = rows.game_id.astype(str)
    stored = pd.read_parquet(directory / f"inputs_{season}.parquet")
    factors = prior_shrink.season_factors(multipliers, season)
    for fold in (fold for fold in folds if fold.season == season):
        fit_ids, held_ids = fold_games(where[fold.stem], fold)
        team_rates = second_prior.team_rate_numbers(rows[game.isin(fit_ids).to_numpy()].reset_index(drop=True))
        held = swap_test.prepare(rows[game.isin(held_ids).to_numpy()].reset_index(drop=True), season)
        table = prior_shrink.shrink_fold(fold.rows, season, factors)
        ratings = swap_test.ratings_in_raw_sign(table)
        priors = swap_test.ratings_in_raw_sign(table, "prior")
        pairs = swap_adjust.residual_pairs(held, ratings, swap_adjust.RESIDUAL_CONTEXT)
        whole, prior_only = second_prior.swap_targets(pairs, ratings), second_prior.swap_targets(pairs, priors)
        here = stored[(stored.repeat == fold.repeat) & (stored.fold == fold.fold)]
        for long_side, side in second_prior.SIDE_KEYS.items():
            inputs = here[here.side == side].set_index("player_id")[TODAY].astype(float)
            frame = second_prior.with_team_rates(inputs, team_rates[side])[NAMES]
            keep = (pairs[long_side].fifth_a.isin(frame.index) & pairs[long_side].fifth_b.isin(frame.index)).to_numpy()
            swaps = pairs[long_side][keep]
            dx = (frame.reindex(swaps.fifth_a.to_numpy()).to_numpy(float)
                  - frame.reindex(swaps.fifth_b.to_numpy()).to_numpy(float))
            h = swaps.h.to_numpy(float)
            aim = {"d": whole[long_side].r.to_numpy(float)[keep], "d_prior": prior_only[long_side].r.to_numpy(float)[keep],
                   "r": swaps.r.to_numpy(float)}
            sums = store.setdefault((season, side), {"G": 0.0, "sh": 0.0, **{t: [np.zeros(len(NAMES)), 0.0] for t in TARGETS}})
            sums["G"] = sums["G"] + (dx * h[:, None]).T @ dx
            sums["sh"] += float(h.sum())
            for target in TARGETS:
                sums[target][0] = sums[target][0] + dx.T @ (h * aim[target])
                sums[target][1] += float((h * aim[target] ** 2).sum())
    print(season, round(time.time() - started), flush=True)


def fit(side, seasons, columns, target):
    at = [position[c] for c in columns]
    gram = sum(store[(y, side)]["G"] for y in seasons)[np.ix_(at, at)]
    right = sum(store[(y, side)][target][0] for y in seasons)[at]
    return np.linalg.solve(gram + 1e-4 * np.mean(np.diag(gram)) * np.eye(len(at)), right)


def loss(side, season, columns, target, weights):
    at = [position[c] for c in columns]
    sums = store[(season, side)]
    return (sums[target][1] - 2.0 * weights @ sums[target][0][at]
            + weights @ sums["G"][np.ix_(at, at)] @ weights) / sums["sh"]


def held_out(side, columns, target):
    return np.array([loss(side, y, columns, target, fit(side, [o for o in SEASONS if abs(o - y) > 1], columns, target))
                     for y in SEASONS])


def nothing(side, target):
    return np.array([store[(y, side)][target][1] / store[(y, side)]["sh"] for y in SEASONS])


def line(change):
    mean, z = change.mean(), change.mean() / (change.std(ddof=1) / np.sqrt(len(change)))
    eras = " | ".join("%+.3f %+.2f" % (change[[SEASONS.index(y) for y in ys]].mean(),
                                       change[[SEASONS.index(y) for y in ys]].mean()
                                       / (change[[SEASONS.index(y) for y in ys]].std(ddof=1) / np.sqrt(len(ys))))
                      for ys in ERAS.values())
    return f"{mean:+.3f} | {z:+.2f} | {int((change < 0).sum())} | {eras}"


for side, label in (("O", "OFFENSE"), ("D", "DEFENSE")):
    print(f"\n===== {label}: change | z | seasons better of 30 | 1997-2006 change z | 2007-2016 change z | 2017-2026 change z")
    for target in TARGETS:
        today = held_out(side, TODAY, target)
        print(f"{target}: 16 team rates added to today's 29, against today's 29 | {line(held_out(side, NAMES, target) - today)}")
        print(f"{target}: today's 29, against no line | {line(today - nothing(side, target))}")
        print(f"{target}: 16 team rates alone, against no line | {line(held_out(side, RATES, target) - nothing(side, target))}")
    print("\nteam rate weights, per 100: input | on d | on d_prior | on r | on r 1997-2006 | 2007-2016 | 2017-2026")
    on = {target: dict(zip(NAMES, fit(side, SEASONS, NAMES, target))) for target in TARGETS}
    per = {era: dict(zip(NAMES, fit(side, list(ys), NAMES, "r"))) for era, ys in ERAS.items()}
    for name in RATES:
        print(f"{name} | {on['d'][name]:+.3f} | {on['d_prior'][name]:+.3f} | {on['r'][name]:+.3f} | "
              + " | ".join(f"{per[era][name]:+.3f}" for era in ERAS))
print("done", round(time.time() - started), flush=True)

# Read after the tables above, so a proposal and not a result: the two stats whose weights on what the rating leaves kept
# their sign in every era on defense, on the same swaps, one list on both ends.
NARROW = {"team ORB, on and off": ["team_off_rebounds_on", "team_off_rebounds_off"],
          "team made threes, on and off": ["team_threes_made_on", "team_threes_made_off"],
          "team ORB and made threes, on and off": ["team_off_rebounds_on", "team_off_rebounds_off",
                                                   "team_threes_made_on", "team_threes_made_off"]}
for side, label in (("O", "OFFENSE"), ("D", "DEFENSE")):
    print(f"\n===== {label}, narrower lists added to today's 29: change | z | seasons better of 30 | eras")
    for target in ("d", "d_prior"):
        today = held_out(side, TODAY, target)
        for name, columns in NARROW.items():
            print(f"{target}: {name} | {line(held_out(side, [*TODAY, *columns], target) - today)}")
print("narrow done", round(time.time() - started), flush=True)
