"""Read-only: experiment 139's screen from outputs/second_prior/zones139/sums_<season>.npz (notes/DECISIONS.md).

Four line sets on the same held-back stints, each rated season's line solved on the seasons the test setting lets it
train on, the formula's error on the rated season: today's 45 inputs an end, plus the 20 zone rates, plus the 20 shuffled
ones, and the zones in place of the threes and twos rows."""
import sys

import numpy as np
import pandas as pd

from openrapm import rating_line, second_prior as sp, swap_adjust
from openrapm.data.config import load_config
from openrapm.data.seasons import trainable
from openrapm.data.second_prior_files import load_grams
from openrapm.settings import PRODUCT

sys.stdout.reconfigure(encoding="utf-8")
F = rating_line.FITTED_PARTS
cfg, prior = load_config(), PRODUCT.prior
names = sp.INPUT_SETS["context_types_team_rates_zones_centred"]
K = len(names)
sums, today_sums = load_grams(cfg, "zones139"), load_grams(cfg, "rating134")
seasons, finished = sorted(sums), set(trainable(cfg).seasons)
shooting = [f"team_{count}_{where}" for where in ("on", "off") for count in ("threes_made", "threes_missed", "twos_made", "twos_missed")]
SETS = {"today": names[:45],
        "plus zones": names[:65],
        "plus shuffled zones": names[:45] + sp.SHUFFLED_ZONE_RATES,
        "zones in place of threes and twos": [n for n in names[:45] if n not in shooting] + sp.ZONE_RATES}


def columns(chosen):
    at = [names.index(n) for n in chosen]
    return np.array([0, 1, *[F + j for j in at], *[F + K + j for j in at]])


def solve(total, at):
    Q = total.Q[np.ix_(at, at)]
    mean_square = np.diag(Q) / total.sw
    has = mean_square > rating_line.NO_SPREAD
    scale = np.where(has, 1.0 / np.sqrt(np.where(has, mean_square, 1.0)), 0.0)
    gram = Q * np.outer(scale, scale)
    lam = float(prior.second_prior_penalty) * float(np.mean(np.diag(gram)))
    return np.linalg.solve(gram + lam * np.eye(len(at)), total.c[at] * scale) * scale


def error(sums_one, at, weights):
    return float((sums_one.yy - 2.0 * weights @ sums_one.c[at] + weights @ sums_one.Q[np.ix_(at, at)] @ weights) / sums_one.sw)


rows = []
check = 0.0
for season in seasons:
    train = swap_adjust.training_seasons(season, seasons, finished, 1)
    total = sum([sums[s] for s in train[1:]], sums[train[0]])
    record = {"season": season}
    for label, chosen in SETS.items():
        at = columns(chosen)
        record[label] = error(sums[season], at, solve(total, at))
    # today's columns must read rating134's sums exactly
    old_total = sum([today_sums[s] for s in train[1:]], today_sums[train[0]])
    old_at = np.arange(F + 90)
    check = max(check, abs(record["today"] - error(today_sums[season], old_at, solve(old_total, old_at))))
    rows.append(record)
t = pd.DataFrame(rows)
t.to_csv("outputs/csv/zones139_screen.csv", index=False)
print(f"today's columns against rating134's sums, largest gap in the error: {check:.2e}")


def paired(column):
    out = []
    for era, lo, hi in (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("all 30", 1997, 2026)):
        d = (t[column] - t["today"])[(t.season >= lo) & (t.season <= hi)].to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        out.append(dict(reading=column, era=era, change=round(d.mean(), 3), standard_error=round(se, 3), z=round(d.mean() / se, 2), seasons_better=f"{int((d < 0).sum())} of {len(d)}"))
    return out


print(pd.DataFrame([row for column in list(SETS)[1:] for row in paired(column)]).to_string(index=False))
