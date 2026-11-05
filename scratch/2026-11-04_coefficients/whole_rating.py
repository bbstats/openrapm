"""Read-only: the whole published rating in one table.  A rating is the sum of the Detail view's five columns, so each column's
share of the rating's variance, its covariance with the rating over the rating's variance, adds to 100%.  Regulars of 1,000+
possessions, weighted by possessions, each season on its own and then averaged by era.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
PIECES = ["box", "on", "off", "sos", "adj", "rating"]
doc = json.load(open("docs/data/decompose_openrapm.json", encoding="utf-8"))
rows = []
for season, table in doc["rows"].items():
    for r in table:
        for side, values in (("O", r[3]), ("D", r[4])):
            rows.append(dict(season=int(season), name=r[0], p=r[2], team=r[6], side=side, **dict(zip(PIECES, values))))
t = pd.DataFrame(rows)
t = t[t.p >= 1000]


def weighted(frame):
    w = frame.p.to_numpy(float)
    rating = frame.rating.to_numpy(float)
    mean = np.average(rating, weights=w)
    variance = np.average((rating - mean) ** 2, weights=w)
    out = dict(rating_sd=variance ** 0.5)
    for piece in PIECES[:-1]:
        x = frame[piece].to_numpy(float)
        mx = np.average(x, weights=w)
        out[f"{piece}_sd"] = float(np.average((x - mx) ** 2, weights=w) ** 0.5)
        out[f"{piece}_share"] = float(np.average((x - mx) * (rating - mean), weights=w) / variance)
    out["gap"] = float(np.abs(frame[PIECES[:-1]].sum(axis=1) - frame.rating).max())
    return pd.Series(out)


by = t[t.season <= 2026].groupby(["season", "side"]).apply(weighted, include_groups=False).reset_index()
for era, low, high in (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("2026", 2026, 2026)):
    print(era)
    print(by[(by.season >= low) & (by.season <= high)].groupby("side").mean(numeric_only=True).drop(columns="season").round(3).T.to_string())
