"""Add the luck-adjusted on-court ratings to an existing role panel, in place, with a backup.

    python scratch/add_onc_cols.py [--panel=outputs/role_panel.parquet] [--check]

The owner, 2026-09-09: *"what about luck-adj on-court ORTG/DRTG in the prior?"*  `investigate.oncourt_rates`
computes them; `scripts/49_role_panel.py` now writes them, so a rebuild is not destructive.  This script
puts them into a panel that already exists so the whole thing does not have to be rebuilt to test the idea.

Four columns per row, both sides on every row (the `PAST_CROSS` shape):
    onc_o        his on-court offensive rating on the free-throw-adjusted target, centred on the window
    onc_d        his on-court defensive rating on the opponent-three-adjusted target, RAW SIGN
                 (lower is a better defender), centred on the window
    onc_poss_o / onc_poss_d   the possessions behind each

Writes a backup beside the panel before touching it.  `--check` recomputes one window and compares against
what is already stored, which is the identical-paths check (and read HANDOFF's trap: agreement between two
paths proves they agree, never that either is right).
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from eracoef.config import load_config                      # noqa: E402
from eracoef.investigate import oncourt_rates               # noqa: E402
from eracoef.windows import build_window, window_label      # noqa: E402
from eracoef.xshoot import DEFENSE_TARGETS                  # noqa: E402

COLS = ("onc_o", "onc_d", "onc_poss_o", "onc_poss_d")


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[-1].split("=", 1)[1] if hit else default


def window_onc(seasons, cfg) -> pd.DataFrame:
    """The four columns for one window, keyed on ps_idx (the panel's own player order)."""
    wd_pts = build_window(seasons, cfg)
    wd_o = build_window(seasons, cfg, target="xpts_ft")
    wd_d, _ = DEFENSE_TARGETS["x3def"](seasons, cfg, wd_pts)
    del wd_pts
    return oncourt_rates(wd_o, wd_d)


def main():
    cfg = load_config()
    path = Path(cfg["_root"]) / flag("panel", "outputs/role_panel.parquet")
    P = pd.read_parquet(path)
    wins = sorted(P.window.unique())
    print(f"{path.name}: {len(P)} rows, {len(wins)} windows")

    check = "--check" in sys.argv
    if not check:
        bak = path.with_suffix(".parquet.bak_onc")
        shutil.copy2(path, bak)
        print(f"backup -> {bak.name}")

    t0 = time.time()
    parts = []
    for lab in wins:
        seasons = [int(x) for x in lab.split("-")]
        seasons = list(range(seasons[0], seasons[-1] + 1))
        if window_label(seasons) != lab:
            raise ValueError(f"cannot parse window {lab!r} back into its seasons")
        onc = window_onc(seasons, cfg).assign(window=lab)
        parts.append(onc)
        print(f"  {lab}  {len(onc)} players  ({time.time() - t0:.0f}s)", flush=True)
    A = pd.concat(parts, ignore_index=True)

    if check:
        miss = [c for c in COLS if c not in P.columns]
        if miss:
            print(f"the panel does not carry {miss}; run without --check first")
            return 1
        j = P.merge(A, on=["window", "ps_idx"], suffixes=("", "_new"))
        print(f"\n=== stored against recomputed, {len(j)} rows")
        for c in COLS:
            print(f"  {c:12s} max|diff| {np.abs(j[c] - j[f'{c}_new']).max():.2e}")
        return 0

    P = P.drop(columns=[c for c in COLS if c in P.columns])
    P = P.merge(A, on=["window", "ps_idx"], how="left")
    if P[list(COLS)].isna().any().any():
        n = int(P[list(COLS)].isna().any(axis=1).sum())
        raise ValueError(f"{n} panel rows got no on-court rating: the ps_idx layouts disagree")
    P.to_parquet(path, index=False)
    hi = P[(P.side == "O") & (P.onc_poss_o >= 3000)]
    print(f"\nwrote {path.name}: {len(P)} rows, +{len(COLS)} columns ({time.time() - t0:.0f}s)")
    print(f"  over {len(hi)} rows with 3000+ offensive possessions: "
          f"sd(onc_o) {hi.onc_o.std():.2f}, sd(onc_d) {hi.onc_d.std():.2f}, "
          f"corr(onc_o, apm) {hi.onc_o.corr(hi.apm):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
