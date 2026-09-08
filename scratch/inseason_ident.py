"""The in-season identities against real data: the flat kernel is the shipped fit, a cut of 1 is no cut,
and a cut of 0 at anchor a is the same fit as the shorter kernel on the seasons before a (the leak test)."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config
from eracoef.holdout import Context
from eracoef.systems import registry
from dataclasses import replace

cfg = load_config()
ctx = Context.load(cfg)
S = registry(cfg)
TRAIN = [2022, 2023, 2024]


def diff(a, b, what):
    m = a.df.merge(b.df, on="player_id", suffixes=("_a", "_b"))
    do = np.abs(m.o_a - m.o_b).max(); dd = np.abs(m.d_a - m.d_b).max()
    dp = np.abs(m.poss_a - m.poss_b).max()
    print(f"{what}: rows {len(a.df)} vs {len(b.df)}, common {len(m)}, max|do| {do:.3e} max|dd| {dd:.3e} max|dposs| {dp:.3e}", flush=True)
    return max(do, dd)


t0 = time.time()
base = S["tune501_b7_pasto_pOD"].fit(TRAIN, ctx)
print(f"  board fit {time.time()-t0:.0f}s", flush=True)
flat = S["ks11"].fit(TRAIN, ctx)
diff(base, flat, "kernel {1,1,1} vs the board")

cut1 = replace(S["ks11"], cut=1.0).fit(TRAIN, ctx)
diff(flat, cut1, "cut 1.0 vs no cut")

# the leak test: nothing of 2024 at all, against the two-season kernel anchored at 2023
cut0 = replace(S["ks55"], cut=0.0).fit(TRAIN, ctx)
short = replace(S["ks55"], name="short", kernel={0: 0.5, -1: 0.5}).fit([2022, 2023], ctx)
diff(cut0, short, "cut 0 at 2024 vs the same kernel on 2022-2023")

# is the residual difference the prior's EXCLUSION set?  cut 0 trains on [2022,2023,2024], so the GBDT
# prior stays off 2024-2026 as well as 2021-2023; the two-season fit only excludes 2021-2023.
ctx.current_h = 2024
short_x = replace(S["ks55"], name="short_x", kernel={0: 0.5, -1: 0.5}).fit([2022, 2023], ctx)
ctx.current_h = None
diff(cut0, short_x, "cut 0 vs the two-season fit with 2024-2026 also excluded")
m = cut0.df.merge(short_x.df, on="player_id", suffixes=("_a", "_b"))
print("  prior only: max|dpo|", np.abs(m.prior_o_a - m.prior_o_b).max(),
      "max|dpd|", np.abs(m.prior_d_a - m.prior_d_b).max())
print("  residual  : max|duo|", np.abs((m.o_a - m.prior_o_a) - (m.o_b - m.prior_o_b)).max(),
      "max|dud|", np.abs((m.d_a - m.prior_d_a) - (m.d_b - m.prior_d_b)).max())
