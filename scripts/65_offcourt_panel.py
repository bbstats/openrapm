"""Write each player-season's OFF-court record into the season panel, beside the on-court one.

    python scripts/65_offcourt_panel.py [--panel=outputs/role_panel_season.parquet]

For every season: the two designs the panel's `onc_*` were built from (`xpts_ft` on offence, `x3def` on
defence), `investigate.offcourt_rates` on them, and the on/off net (`net_o = onc_o - offc_o`, same for
defence).  Also recomputes `onc_*` the same way and reports the largest gap against what the panel
carries, which must be ~0 -- if it is not, the two families were built from different games and the net
is meaningless.  The panel is backed up to `<panel>_pre_offc.parquet` before it is overwritten.
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context  # noqa: E402
from eracoef.investigate import offcourt_rates, oncourt_rates  # noqa: E402
from eracoef.xshoot import DEFENSE_TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def main():
    path = ROOT / _flag("panel", "outputs/role_panel_season.parquet")
    cfg = load_config(ROOT / "config.yaml")
    ctx = Context.load(cfg)
    panel = pd.read_parquet(path)
    seasons = sorted(panel.season.unique())
    t0 = time.time()
    parts = []
    for season in seasons:
        wd_o = ctx.design([season], "xpts_ft")
        wd_d = ctx.design([season], DEFENSE_TARGETS["x3def"])
        ids = wd_o.spec.ps_table["player_id"].to_numpy()
        onc = oncourt_rates(wd_o, wd_d)
        offc = offcourt_rates(wd_o, wd_d)
        frame = pd.DataFrame({"player_id": ids, "season": season,
                              **{c: onc[c].to_numpy(dtype=float) for c in sy.ONC},
                              **{c: offc[c].to_numpy(dtype=float) for c in sy.OFFC}})
        frame["net_o"] = frame.onc_o - frame.offc_o
        frame["net_d"] = frame.onc_d - frame.offc_d
        parts.append(frame)
        print(f"  {season}: {len(ids)} players, pad k off {offc.attrs['pad_k']['o']:.0f} / "
              f"{offc.attrs['pad_k']['d']:.0f} possessions  ({time.time() - t0:.0f}s)", flush=True)
    new = pd.concat(parts, ignore_index=True)

    # the on-court columns recomputed here must be the panel's own, or the net mixes two windows
    check = panel.merge(new[["player_id", "season"] + sy.ONC], on=["player_id", "season"],
                        how="inner", suffixes=("", "_re"))
    gap = max(float((check[c] - check[f"{c}_re"]).abs().max()) for c in sy.ONC)
    print(f"largest gap between the panel's onc_* and the recomputation: {gap:.6f}")
    assert gap < 1e-6, "the panel's on-court columns were not built from these designs"

    backup = path.with_name(path.stem + "_pre_offc.parquet")
    if not backup.exists():
        shutil.copy(path, backup)
    panel = panel.drop(columns=[c for c in sy.OFFC + sy.NET if c in panel.columns])
    panel = panel.merge(new[["player_id", "season"] + sy.OFFC + sy.NET], on=["player_id", "season"], how="left")
    for c in sy.OFFC + sy.NET:
        panel[c] = panel[c].fillna(0.0)
    panel.to_parquet(path, index=False)
    wide = panel[(panel.side == "O") & (panel.poss >= 1000)]
    print(f"wrote {path}: {len(panel)} rows; among 1,000+ possession player-seasons "
          f"sd onc_o {wide.onc_o.std():.2f} offc_o {wide.offc_o.std():.2f} net_o {wide.net_o.std():.2f}, "
          f"onc_d {wide.onc_d.std():.2f} offc_d {wide.offc_d.std():.2f} net_d {wide.net_d.std():.2f}; "
          f"corr(onc_o, offc_o) {np.corrcoef(wide.onc_o, wide.offc_o)[0, 1]:.2f}")


if __name__ == "__main__":
    main()
