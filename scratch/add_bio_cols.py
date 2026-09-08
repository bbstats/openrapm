"""Add the player-level columns (bio.PLAYER_INPUTS: height, weight, draft_pick, tenure, n_teams) to
outputs/role_panel.parquet, touching nothing else.  A join from data/raw/bio and data/cache/roles.parquet --
no APM, no SPM, no ridge.  Backup at .parquet.bak5.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from eracoef.bio import PLAYER_INPUTS, player_inputs
from eracoef.config import load_config
from eracoef.roles import build_roles
from eracoef.windows import window_label, window_seasons

cfg = load_config()
PANEL = Path(cfg["_root"]) / cfg.get("paths", {}).get("role_panel", "outputs/role_panel.parquet")
P = pd.read_parquet(PANEL)
before = P.copy()
roles = build_roles(cfg)
for c in PLAYER_INPUTS:
    P[c] = np.nan

for w in window_seasons(cfg):
    seasons = list(range(w[0], w[1] + 1))
    lab = window_label(seasons)
    sel = P.window == lab
    ids = P.loc[sel, "player_id"].to_numpy()
    x = player_inputs(cfg, roles, seasons, ids)
    P.loc[sel, PLAYER_INPUTS] = x[PLAYER_INPUTS].to_numpy(dtype=float)
    o = sel & (P.side == "O")
    print(f"  {lab}: {int(sel.sum())} rows; height {P.loc[o, 'height'].mean():.1f} in, weight {P.loc[o, 'weight'].mean():.0f}, "
          f"undrafted {(P.loc[o, 'draft_pick'] >= 61).mean():.1%}, tenure {P.loc[o, 'tenure'].mean():.2f}, "
          f"n_teams {P.loc[o, 'n_teams'].mean():.2f}")

assert P[PLAYER_INPUTS].notna().all().all(), "some rows got no player inputs"
for c in before.columns:
    a, b = before[c].to_numpy(), P[c].to_numpy()
    assert np.array_equal(a, b) if a.dtype.kind not in "fc" else np.array_equal(a, b, equal_nan=True), c
shutil.copy(PANEL, PANEL.with_suffix(".parquet.bak5"))
P.to_parquet(PANEL, index=False)
print(f"wrote {PANEL} with {PLAYER_INPUTS}; every old column identical (backup .bak5)")
print(P.loc[P.side == "O", PLAYER_INPUTS].describe().round(2).to_string())
