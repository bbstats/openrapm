"""Read-only: settings halfway between today's and the search winner's, on the within-season test, all thirty seasons."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
path = Path("scripts/checks/156_shrink_refit.py")
spec = importlib.util.spec_from_file_location("refit156", path)
refit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = refit
spec.loader.exec_module(refit)
MID = tuple(float(x) for x in sys.argv[1].split(","))
wowy = refit._wowy()
audit = wowy._audit()
mid = wowy.system_errors(audit, refit.within_dir(), "rating134", refit.ALL_SEASONS, label="mid",
                         defence_factors=refit.defence_pairs(MID), within_factors=tuple(MID[:2]))
mid = mid[(mid.stage == "step4") & (mid.slice == "all")].set_index("season").error
base = pd.read_csv("outputs/csv/refit156_rating134_seasons.csv")
base = base[(base.stage == "step4") & (base.slice == "all")].pivot(index="season", columns="system", values="error")["rating134@shipped"]
d = (mid - base).dropna()
for era, a, b in (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("all 30", 1997, 2026)):
    p = d[(d.index >= a) & (d.index <= b)]
    print(f"{era}: change {p.mean():+.3f}, z {p.mean() / (p.std(ddof=1) / np.sqrt(len(p))):+.2f}, better {int((p < 0).sum())} of {len(p)}")
