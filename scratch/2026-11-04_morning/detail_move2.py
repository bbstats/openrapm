"""Which piece of each 2027 rating moved between two commits of the site, largest first."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SITE = "A:/code/spmm_site"
PIECES = ["box_score", "on_court", "off_court", "team_sos", "box_adj", "rating"]


def load(commit):
    raw = subprocess.run(["git", "-C", SITE, "show", f"{commit}:docs/data/decompose_openrapm.json"], capture_output=True, check=True).stdout
    rows = json.loads(raw.decode("utf-8"))["rows"]["2027"]
    out = []
    for r in rows:
        row = {"n": r[0], "id": r[1], "p": r[2], "tm": r[6]}
        for side, values in (("O", r[3]), ("D", r[4])):
            for piece, value in zip(PIECES, values):
                row[f"{side}_{piece}"] = value
        out.append(row)
    return pd.DataFrame(out).set_index("id")


a, b = load(sys.argv[1]), load(sys.argv[2])
assert len(a) == len(b)
cols = [c for c in a.columns if c[:2] in ("O_", "D_")]
d = (b[cols] - a.loc[b.index, cols]).round(2)
print("possessions differ:", int((a.loc[b.index, "p"] != b.p).sum()), " team differ:", int((a.loc[b.index, "tm"] != b.tm).sum()))
print("\nplayers touched, largest move, sum of absolute moves, by piece")
print(pd.DataFrame({"players": (d != 0).sum(), "largest": d.abs().max(), "sum_abs": d.abs().sum()}).to_string())
d["total"] = d.O_rating + d.D_rating
d.insert(0, "n", b.n)
d.insert(1, "tm", b.tm)
d.insert(2, "p", b.p)
show = ["n", "tm", "p", "O_box_score", "O_box_adj", "O_on_court", "O_off_court", "O_rating", "D_box_score", "D_box_adj", "D_on_court", "D_off_court", "D_rating", "total"]
print("\nlargest movers")
print(d.reindex(d.total.abs().sort_values(ascending=False).index)[show].head(14).to_string(index=False))
print("\nby team, sum of absolute total moves")
print(d.assign(a=d.total.abs()).groupby("tm").a.sum().sort_values(ascending=False).head(10).round(2).to_string())
