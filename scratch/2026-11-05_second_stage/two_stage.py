"""Read-only: the box-score prior's second stage from the saved sums (notes/DECISIONS.md, registered 2026-11-05).

Stage one: the fitted parts and his own stats.  Stage two: those weights frozen, the fitted parts again, and the others' priors
and team rates.  Writes outputs/csv/two_stage_seasons.csv, two_stage_weights.csv and two_stage_lines.npz.
"""
import sys

import numpy as np
import pandas as pd

from openrapm import rating_line, second_prior as sp, swap_adjust
from openrapm.data.config import load_config
from openrapm.data.seasons import trainable
from openrapm.data.second_prior_files import line_setup, load_grams
from openrapm.settings import PRODUCT

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
TAG, EXCLUDE_NEAR, F = "rating134", 1, rating_line.FITTED_PARTS
cfg, settings = load_config(), PRODUCT.prior
names = sp.INPUT_SETS[settings.second_prior_inputs]
K, penalty = len(names), settings.second_prior_penalty
line_setup(cfg, TAG, names)
sums = load_grams(cfg, TAG)
finished = set(trainable(cfg).seasons)
seasons = sorted(sums)
ADJ = [n for n in names if n.startswith(("teammates_", "opponents_")) or (n.startswith("team_") and n != "team_possession_share")]
OWN = [n for n in names if n not in ADJ]
assert len(OWN) == 21 and len(ADJ) == 24 and OWN[:2] == ["box_prior", "box_prior_other_end"]
own = [names.index(n) for n in OWN]
adj = [names.index(n) for n in ADJ]
FIT = [0, 1]
OWN_AT = [F + j for j in own] + [F + K + j for j in own]
ADJ_AT = [F + j for j in adj] + [F + K + j for j in adj]
assert len(FIT + OWN_AT + ADJ_AT) == F + 2 * K


def solve(total, at, right):
    """rating_line.coefficients' scaling and penalty on the columns `at`, against `right`."""
    Q = total.Q[np.ix_(at, at)]
    mean_square = np.diag(Q) / total.sw
    has = mean_square > rating_line.NO_SPREAD
    scale = np.where(has, 1.0 / np.sqrt(np.where(has, mean_square, 1.0)), 0.0)
    gram = Q * np.outer(scale, scale)
    ridge = float(penalty) * float(np.mean(np.diag(gram)))
    return np.linalg.solve(gram + ridge * np.eye(len(at)), right * scale) * scale


def lines(season):
    train = swap_adjust.training_seasons(season, seasons, finished, EXCLUDE_NEAR)
    parts = [sums[other] for other in train]
    total = sum(parts[1:], parts[0])
    joint = solve(total, list(range(F + 2 * K)), total.c)
    library = rating_line.coefficients(parts, settings.second_prior_folds, penalty)
    assert np.allclose(joint[F:F + K], library["O"][-1], rtol=1e-12, atol=1e-13) and np.allclose(joint[F + K:], library["D"][-1], rtol=1e-12, atol=1e-13)
    # stage one
    one_at = FIT + OWN_AT
    one = np.zeros(F + 2 * K)
    one[one_at] = solve(total, one_at, total.c[one_at])
    # stage two: stage one's inputs frozen, the fitted parts free again
    frozen = np.zeros(F + 2 * K)
    frozen[OWN_AT] = one[OWN_AT]
    two_at = FIT + ADJ_AT
    right = total.c[two_at] - total.Q[np.ix_(two_at, OWN_AT)] @ frozen[OWN_AT]
    two = frozen.copy()
    two[two_at] = solve(total, two_at, right)
    return joint, one, two


def own_part(weights):
    out = np.zeros_like(weights)
    out[OWN_AT] = weights[OWN_AT]
    return out


rows, stored = [], {}
for season in seasons:
    joint, one, two = lines(season)
    stored[str(season)] = np.vstack([joint, one, two])
    e = sums[season]
    same_fitted = two.copy()
    same_fitted[FIT] = joint[FIT]
    rows.append(dict(season=season, today=e.error(joint), two_stage=e.error(two), two_stage_todays_fitted=e.error(same_fitted), stage_one_only=e.error(one),
                     fit_off_today=joint[0], fit_def_today=joint[1], fit_off_one=one[0], fit_def_one=one[1], fit_off_two=two[0], fit_def_two=two[1]))
t = pd.DataFrame(rows)
t.to_csv("outputs/csv/two_stage_seasons.csv", index=False)
np.savez("outputs/csv/two_stage_lines.npz", names=np.array(names), **stored)

print("the joint solve equals rating_line.coefficients in all 30 seasons (assert passed)")
models = pd.read_pickle(f"outputs/prior_models_{TAG}.pkl")
gap = max(np.abs(np.asarray(models[s][side]["line"])[-1] - stored[str(s)][2 * 0][F + (0 if side == "O" else K):F + (K if side == "O" else 2 * K)]).max() for s in seasons for side in "OD")
print(f"the joint solve against the line saved in the boosters' file, 30 seasons, largest gap {gap:.2e}")

check = pd.read_csv("outputs/csv/wowy_rating134_seasons.csv")
check = check[(check.system == "rating134") & (check.stage == "step4") & (check.slice == "all")].set_index("season").error
for label, file in (("rating134's", "season_ratings_rating134_step4_multipliers"), ("padbins127's", "season_ratings_padbins127_step4_multipliers")):
    mult = pd.read_csv(f"outputs/csv/{file}.csv").set_index("season")
    out = []
    for s in seasons:
        w = stored[str(s)][0].copy()
        w[FIT] = mult.loc[s, ["within_games_multiplier_off", "within_games_multiplier_def"]].to_numpy(float)
        out.append(sums[s].error(w) - check[s])
    out = np.array(out)
    print(f"today's line, the fitted parts at {label} within layer, formula minus check 151: mean {out.mean():+.3f}, standard deviation {out.std(ddof=1):.3f}, largest {np.abs(out).max():.3f}")
out = np.array([sums[s].error(stored[str(s)][0]) - check[s] for s in seasons])
print(f"today's line, the fitted parts as solved, formula minus check 151: mean {out.mean():+.3f}, standard deviation {out.std(ddof=1):.3f}, largest {np.abs(out).max():.3f}")


def paired(column, reference="today"):
    res = []
    for era, lo, hi in (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("all 30", 1997, 2026)):
        d = (t[column] - t[reference])[(t.season >= lo) & (t.season <= hi)].to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        res.append(dict(era=era, change=d.mean(), standard_error=se, z=d.mean() / se, seasons_better=int((d < 0).sum()), seasons=len(d)))
    return pd.DataFrame(res).round(3)


for column, label in (("two_stage", "two stages minus today's, each arm's fitted parts"), ("two_stage_todays_fitted", "two stages minus today's, both at today's fitted parts"), ("stage_one_only", "stage one alone minus today's")):
    print("\n" + label)
    print(paired(column).to_string(index=False))
print("\nfitted parts, mean over 30 seasons:", t[[c for c in t if c.startswith("fit_")]].mean().round(3).to_dict())

mean = {k: np.mean([stored[str(s)][i] for s in seasons], axis=0) for i, k in enumerate(("today", "one", "two"))}
y26 = stored["2026"]
tab = []
for j, n in enumerate(names):
    tab.append(dict(input=n, stage="own" if n in OWN else "adjustment", off_today=y26[0][F + j], off_two=y26[2][F + j], def_today=y26[0][F + K + j], def_two=y26[2][F + K + j],
                    off_today_30=mean["today"][F + j], off_two_30=mean["two"][F + j], def_today_30=mean["today"][F + K + j], def_two_30=mean["two"][F + K + j]))
w = pd.DataFrame(tab).sort_values("stage", ascending=False, kind="stable")
w.to_csv("outputs/csv/two_stage_weights.csv", index=False)
print("\nweights, 2026's line trained on 1997-2024")
print(w.round(3).to_string(index=False))
