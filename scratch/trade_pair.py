"""Paired test of every mapped row in a tracked dump against the shipped board's mapped row.

    .venv/Scripts/python scratch/trade_pair.py <base> <cand> [<cand2> ...]
Like scratch/pairsys.py but a candidate dump may carry several maps; each is reported on its own line.
"""
import sys

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from eracoef.holdout import pooled

names = sys.argv[1:]


def load(system):
    r = pd.read_parquet(f"outputs/holdout_track_{system}.parquet")
    return r[r.system != system]


base = load(names[0])
if base.system.nunique() != 1:
    base = base[base.system == sorted(base.system.unique())[0]]
bg = float(pooled(base).game.iloc[0])
bs = base.set_index("held_out")[["tg"]]
print(f"\n  base {names[0]}: {bg:.4f} over {len(bs)} seasons ({base.system.iloc[0]})\n")
print(f"  {'system / map':70s} {'game':>9s} {'vs base':>9s} {'z':>7s} {'wins':>8s}")
for n in names[1:]:
    c = load(n)
    for m in sorted(c.system.unique()):
        cm = c[c.system == m]
        j = bs.join(cm.set_index("held_out")[["tg"]], rsuffix="_c", how="inner")
        d = (j.tg_c - j.tg).to_numpy()
        z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std() > 0 else 0.0
        g = float(pooled(cm).game.iloc[0])
        print(f"  {m[:70]:70s} {g:9.4f} {g - bg:+9.4f} {z:7.2f} {int((d < 0).sum()):4d}/{len(d)}")
