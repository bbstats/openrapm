"""Read-only: the second stage with its inputs residualized on stage one (Frisch-Waugh-Lovell), against the joint fit and against
stage two on the raw inputs.  Registered in notes/DECISIONS.md, 2026-11-05."""
import sys

import numpy as np
import pandas as pd

from openrapm import rating_line, second_prior as sp, swap_adjust
from openrapm.commands.build import season_lineup_rows
from openrapm.commands.within_season import prior_rows, saved_booster
from openrapm.data.config import load_config
from openrapm.data.seasons import trainable
from openrapm.data.second_prior_files import line_setup, load_grams
from openrapm.settings import PRODUCT

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
F, K, EXCLUDE_NEAR = rating_line.FITTED_PARTS, 45, 1
SIX = [2000, 2005, 2010, 2015, 2022, 2026]
cfg, settings = load_config(), PRODUCT.prior
names = sp.INPUT_SETS[settings.second_prior_inputs]
type_scale = line_setup(cfg, "rating134", names)["type_scale"]
sums = load_grams(cfg, "rating134")
seasons, finished = sorted(sums), set(trainable(cfg).seasons)
ADJ = [n for n in names if n.startswith(("teammates_", "opponents_")) or (n.startswith("team_") and n != "team_possession_share")]
OWN = [n for n in names if n not in ADJ]
own_at = [F + names.index(n) for n in OWN] + [F + K + names.index(n) for n in OWN]
adj_at = [F + names.index(n) for n in ADJ] + [F + K + names.index(n) for n in ADJ]
S1 = [0, 1] + own_at
ALL = list(range(F + 2 * K))

print("A made-up example: x2 = 0.8 x1 + noise, y = x1 + x2 + noise, 100,000 rows")
rng = np.random.default_rng(0)
x1 = rng.standard_normal(100_000)
x2 = 0.8 * x1 + 0.6 * rng.standard_normal(100_000)
y = x1 + x2 + rng.standard_normal(100_000)


def ols(columns, target):
    return np.linalg.lstsq(np.column_stack(columns), target, rcond=None)[0]


joint = ols([x1, x2], y)
b1 = ols([x1], y)[0]
residual = y - b1 * x1
raw = ols([x2], residual)[0]
a = ols([x1], x2)[0]
x2r = x2 - a * x1
clean = ols([x2r], residual)[0]
rows = [("one regression with both", joint[0], joint[1], np.mean((y - joint[0] * x1 - joint[1] * x2) ** 2)),
        ("residuals on x2 as it stands", b1, raw, np.mean((residual - raw * x2) ** 2)),
        ("residuals on x2 residualized on x1", b1, clean, np.mean((residual - clean * x2r) ** 2))]
print(pd.DataFrame(rows, columns=["way", "weight on x1", "weight on x2", "mean squared error"]).round(4).to_string(index=False))


def solve(total, at, right, penalty):
    """rating_line.coefficients' scaling and penalty; `right` may hold several columns."""
    Q = total.Q[np.ix_(at, at)]
    mean_square = np.diag(Q) / total.sw
    scale = 1.0 / np.sqrt(mean_square)
    gram = Q * np.outer(scale, scale)
    lam = float(penalty) * float(np.mean(np.diag(gram)))
    solved = np.linalg.solve(gram + lam * np.eye(len(at)), (right.T * scale).T)
    return (solved.T * scale).T


def fits(total, penalty):
    joint = np.zeros(F + 2 * K)
    joint[ALL] = solve(total, ALL, total.c, penalty)
    one = np.zeros(F + 2 * K)
    one[S1] = solve(total, S1, total.c[S1], penalty)
    # stage two on the inputs as they stand: the first read
    frozen = one.copy()
    frozen[:2] = 0.0
    two_at = [0, 1] + adj_at
    raw = frozen.copy()
    raw[two_at] = solve(total, two_at, total.c[two_at] - total.Q[np.ix_(two_at, own_at)] @ one[own_at], penalty)
    # stage two on the inputs residualized on stage one's columns; recomposed in the original inputs
    A = solve(total, S1, total.Q[np.ix_(S1, adj_at)], penalty)
    Q_r = total.Q[np.ix_(adj_at, adj_at)] - total.Q[np.ix_(adj_at, S1)] @ A
    c_r = total.c[adj_at] - total.Q[np.ix_(adj_at, S1)] @ one[S1]
    scale = 1.0 / np.sqrt(np.diag(total.Q[np.ix_(adj_at, adj_at)]) / total.sw)
    gram = Q_r * np.outer(scale, scale)
    lam = float(penalty) * float(np.mean(np.diag(total.Q[np.ix_(adj_at, adj_at)] * np.outer(scale, scale))))
    gamma = np.linalg.solve(gram + lam * np.eye(len(adj_at)), c_r * scale) * scale
    clean = np.zeros(F + 2 * K)
    clean[adj_at] = gamma
    clean[S1] = one[S1] - A @ gamma
    return joint, one, raw, clean


records, lines = [], {}
for season in seasons:
    train = swap_adjust.training_seasons(season, seasons, finished, EXCLUDE_NEAR)
    parts = [sums[other] for other in train]
    total = sum(parts[1:], parts[0])
    joint, one, raw, clean = fits(total, settings.second_prior_penalty)
    joint0, _, raw0, clean0 = fits(total, 0.0)
    lines[season] = (joint, one)
    held = sums[season]
    records.append(dict(season=season, gap_weights_no_penalty=np.abs(clean0 - joint0).max(), gap_weights_shipped=np.abs(clean - joint).max(),
                        train_raw_minus_joint=total.error(raw) - total.error(joint), train_clean_minus_joint=total.error(clean) - total.error(joint),
                        train0_raw_minus_joint=total.error(raw0) - total.error(joint0), train0_clean_minus_joint=total.error(clean0) - total.error(joint0),
                        held_raw_minus_joint=held.error(raw) - held.error(joint), held_clean_minus_joint=held.error(clean) - held.error(joint)))
t = pd.DataFrame(records)
t.to_csv("outputs/csv/two_stage_residualized.csv", index=False)
print("\nlargest gap, recomposed weights against the joint weights: no penalty", f"{t.gap_weights_no_penalty.max():.1e}", "shipped penalty", f"{t.gap_weights_shipped.max():.1e}")
print("\non the seasons each line trains on, minus the joint fit, 30 lines")
for column, words in (("train0_raw_minus_joint", "raw stage two, no penalty"), ("train0_clean_minus_joint", "residualized, no penalty"), ("train_raw_minus_joint", "raw stage two, shipped penalty"), ("train_clean_minus_joint", "residualized, shipped penalty")):
    print(f"  {words}: mean {t[column].mean():+.6f}, smallest {t[column].min():+.6f}, largest {t[column].max():+.6f}, lines worse than joint {int((t[column] > 1e-9).sum())}")
print("\non the rated season, minus the joint fit")
for column, words in (("held_raw_minus_joint", "raw stage two"), ("held_clean_minus_joint", "residualized")):
    d = t[column].to_numpy()
    print(" ", words, {era: round(float(d[lo:hi].mean()), 6) for era, lo, hi in (("1997-2006", 0, 10), ("2007-2016", 10, 20), ("2017-2026", 20, 30), ("all 30", 0, 30))}, "largest", round(float(np.abs(d).max()), 6))


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


def wcorr(x, y, w):
    return float(np.average((x - np.average(x, weights=w)) * (y - np.average(y, weights=w)), weights=w) / (wsd(x, w) * wsd(y, w)))


table = pd.read_parquet("outputs/season_ratings_mid136.parquet")
multipliers = pd.read_csv("outputs/csv/season_ratings_mid136_step4_multipliers.csv").set_index("season")
panel = pd.read_parquet("outputs/season_panel.parquet")
models = pd.read_pickle("outputs/prior_models_rating134.pkl")
out = []
for season in SIX:
    saved = models[season]
    rows = panel[panel.season == season]
    answers = {side: sp.booster_answers(saved_booster(saved[side]), prior_rows(rows[rows.side == side], side, saved[side])) for side in "OD"}
    box = {side: saved[side]["prior"] for side in "OD"}
    types = sp.standardised_types(rows[rows.side == "O"], type_scale)
    stints = season_lineup_rows(cfg, season)
    joint, one = lines[season]
    line = {"O": np.tile(joint[F:F + K], (settings.second_prior_folds + 1, 1)), "D": np.tile(joint[F + K:], (settings.second_prior_folds + 1, 1))}
    _, inputs = sp.season_second_priors(stints, answers, box, line, settings.second_prior_folds, settings.second_prior_team_totals, types, settings.second_prior_inputs)
    rated = table[table.season == season].set_index("player_id")
    for side, poss, scale_column, within, sign, start in (("O", "poss_off", "prior_scale_off", "within_multiplier_off", 1.0, F), ("D", "poss_def", "prior_scale_def", "within_multiplier_def", -1.0, F + K)):
        page = float(rated[scale_column].iloc[0] * multipliers.loc[season, within])
        frame = inputs[side]
        team_poss = sp.team_possessions(stints, side)
        keep = rated.index[rated[poss] >= 1000].intersection(frame.index)
        w = rated.loc[keep, poss].to_numpy(float)
        weights_joint, weights_one = dict(zip(names, joint[start:start + K])), dict(zip(names, one[start:start + K]))

        def held_out(weights, members):
            values = sum(sign * page * weights[n] * frame[n] for n in members)
            return sp.hold_team_totals(values, pd.Series(0.0, index=values.index), team_poss).loc[keep].to_numpy(float)

        whole = held_out(weights_joint, names)
        first = held_out(weights_one, OWN)
        second = whole - first
        rest_on_own = held_out({n: weights_joint[n] - weights_one[n] for n in OWN}, OWN)
        others = held_out(weights_joint, ADJ)
        out.append(dict(season=season, side=side, stage_one_column=wsd(first, w), adjustment_column=wsd(second, w), correlation=wcorr(first, second, w), whole_prior=wsd(whole, w),
                        adjustment_others_rows=wsd(others, w), adjustment_own_rows=wsd(rest_on_own, w), own_rows_with_others_rows=wcorr(rest_on_own, others, w)))
c = pd.DataFrame(out)
c.to_csv("outputs/csv/two_stage_residualized_columns.csv", index=False)
print("\namong teammates, today's prior scales, no rating moves")
print(c.round(3).to_string(index=False))
print(c.groupby("side", sort=False).mean(numeric_only=True).drop(columns="season").round(3).to_string())
