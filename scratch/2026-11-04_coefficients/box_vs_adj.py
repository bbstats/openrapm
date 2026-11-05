"""Read-only: the Detail view's Box score and Box Adj columns side by side, published against the version before it."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PIECES = ["box", "on", "off", "sos", "adj", "rating"]


def load(commit=None):
    if commit:
        raw = subprocess.run(["git", "show", f"{commit}:docs/data/decompose_openrapm.json"], capture_output=True, check=True).stdout.decode("utf-8")
        doc = json.loads(raw)
    else:
        doc = json.load(open("docs/data/decompose_openrapm.json", encoding="utf-8"))
    rows = []
    for season, table in doc["rows"].items():
        for r in table:
            for side, values in (("O", r[3]), ("D", r[4])):
                rows.append(dict(season=int(season), name=r[0], p=r[2], tm=r[6], side=side, **dict(zip(PIECES, values))))
    return pd.DataFrame(rows)


def wsd(x, w):
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / (wsd(x, w) * wsd(y, w)))


def read(t, a, b):
    out = []
    for side in ("O", "D"):
        s = t[(t.side == side) & (t.season >= a) & (t.season <= b) & (t.p >= 1000)].copy()
        s["box_in"] = s.box - s.groupby(["season", "tm"]).box.transform("mean")
        sums = {k: [] for k in ("box", "adj", "box_in", "prior", "rating", "corr", "corr_in", "on")}
        for _, g in s.groupby("season"):
            w = g.p.to_numpy(float)
            sums["box"].append(wsd(g.box, w) ** 2)
            sums["adj"].append(wsd(g.adj, w) ** 2)
            sums["box_in"].append(wsd(g.box_in, w) ** 2)
            sums["prior"].append(wsd(g.box + g.adj, w) ** 2)
            sums["rating"].append(wsd(g.rating, w) ** 2)
            sums["on"].append(wsd(g.on, w) ** 2)
            sums["corr"].append(wcorr(g.box, g.adj, w))
            sums["corr_in"].append(wcorr(g.box_in, g.adj, w))
        out.append(dict(side=side, box=np.sqrt(np.mean(sums["box"])), box_among_teammates=np.sqrt(np.mean(sums["box_in"])), adj=np.sqrt(np.mean(sums["adj"])),
                        adj_over_box=np.sqrt(np.mean(sums["adj"])) / np.sqrt(np.mean(sums["box"])), both=np.sqrt(np.mean(sums["prior"])), on=np.sqrt(np.mean(sums["on"])),
                        rating=np.sqrt(np.mean(sums["rating"])), corr=np.mean(sums["corr"]), corr_among_teammates=np.mean(sums["corr_in"])))
    return pd.DataFrame(out).set_index("side").round(3)


pd.set_option("display.width", 200)
new, old = load(), load("a39f1b3")
for label, a, b in (("2026", 2026, 2026), ("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026)):
    print(f"\n{label}, published"); print(read(new, a, b).to_string())
    print(f"{label}, before"); print(read(old, a, b).to_string())
s = new[(new.season == 2026) & (new.p >= 1000)]
for side in ("O", "D"):
    g = s[s.side == side]
    print(f"\n2026 {side}: largest Box Adj up"); print(g.nlargest(6, "adj")[["name", "box", "adj", "rating"]].to_string(index=False))
    print(f"2026 {side}: largest Box Adj down"); print(g.nsmallest(6, "adj")[["name", "box", "adj", "rating"]].to_string(index=False))
