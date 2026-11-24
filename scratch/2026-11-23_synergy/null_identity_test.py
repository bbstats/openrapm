"""Sanity check of null.py: with every player his own donor, the null's reading should sit near the real one (not
equal: the real reading uses each fold's own values, the null each player's whole-season standing)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0]]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import null  # noqa: E402

data = Path("outputs/synergy/rd143")
m154 = null._load("selfcreation_check", null.CHECKS / "154_self_creation.py")
careers = null.Careers(data, 1997, 2026)
identity = np.arange(len(careers.players))[None, :]
rng = np.random.default_rng(1)
shuffled = rng.permutation(len(careers.players))[None, :]
got = null.season_null(m154, careers, data, 2016, np.vstack([identity, shuffled]))
real = pd.read_parquet(data / "scan_2016.parquet")
real = real[(real.analysis == "single") & (real.term == "synergy per lineup sd")][["end", "trait", "value"]]
table = got.pivot_table(index=["end", "trait"], columns="draw", values="value").join(
    real.set_index(["end", "trait"]).value.rename("real"))
print(table.round(3).to_string())
print("correlation, own donor against real:", round(float(np.corrcoef(table[0], table.real)[0, 1]), 3))
