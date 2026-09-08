"""Trade calibration, step 3: turnover terms in the calibration map, on the shipped board's existing dump.

    .venv/Scripts/python scratch/trade_maps.py [system] [--set=main|control|all]

Same loop as scratch/maps.py (no refits, ~2 s a map plus the turnover covariates), the shipped map first as
the base.  Each variant adds terms on top of it, offense:defense.
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from eracoef.calmap import evaluate, load_frames, parse_maps, unmapped_rows
from eracoef.config import load_config
from eracoef.holdout import Context, Holdout, pooled

cfg = load_config()
args = [a for a in sys.argv[1:] if not a.startswith("--")]
system = args[0] if args else "tune501_b7"
which = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--set=")), "main")
K = 3
BO, BD = "linear+log2&xlog&prior&tshare", "linear+log2&xlog"
MAIN = {
    "ship": (BO, BD),
    "moved level (21.7 again)": (BO + "&moved", BD + "&moved"),
    "moved x rating, moved x prior": (BO + "&movedx&mprior", BD + "&movedx&mprior"),
    "turn level": (BO + "&turn", BD + "&turn"),
    "turn x rating, turn x prior": (BO + "&turnx&turnprior", BD + "&turnx&turnprior"),
    "turn level + both slopes": (BO + "&turn&turnx&turnprior", BD + "&turn&turnx&turnprior"),
    "turn level, offense only": (BO + "&turn", BD),
    "turn level, defense only": (BO, BD + "&turn"),
    "turnp level (past seasons of the block)": (BO + "&turnp", BD + "&turnp"),
    "turnp level + both slopes": (BO + "&turnp&turnpx&turnpprior", BD + "&turnp&turnpx&turnpprior"),
}
CONTROL = {
    "ship": (BO, BD),
    "turn level": (BO + "&turn", BD + "&turn"),
    "CONTROL turna level (half of H)": (BO + "&turna", BD + "&turna"),
    "turn level + both slopes": (BO + "&turn&turnx&turnprior", BD + "&turn&turnx&turnprior"),
    "CONTROL turna level + both slopes": (BO + "&turna&turnax&turnaprior", BD + "&turna&turnax&turnaprior"),
}
SETS = {"main": MAIN, "control": CONTROL, "all": {**MAIN, **CONTROL}}[which]

dump = pd.read_parquet(f"outputs/ratings_track_{system}.parquet")
ho = Holdout.from_config(cfg, ks=[K])
ctx = Context.load(cfg)
frames = load_frames(ctx, ho.seasons(), level=ho.level, verbose=False)
rows, per, params = [unmapped_rows(dump, frames, system, K)], {}, {}
t0 = time.time()
for label, (mo, md) in SETS.items():
    fam = f"{mo}:{md}"
    map_o, map_d, bend = parse_maps(fam)
    r, p = evaluate(dump, frames, system, K, map_o, map_d, label, bend=bend)
    rows.append(r)
    per[label] = r.set_index("held_out")[["tg", "tg_n"]]
    params[label] = p[p.held_out == -1].iloc[0]
    print(f"  {label} ({time.time() - t0:.0f}s)", flush=True)
P = pooled(pd.concat(rows, ignore_index=True)).set_index("system")
base = per["ship"]
print(f"\n{system}, K = {K}, {len(frames)} seasons.  unmapped {P.loc[system].game:.4f}\n")
print(f"  {'map':44s} {'game':>9s} {'vs ship':>9s} {'z':>7s} {'wins':>7s}")
for label in SETS:
    d = (per[label].tg - base.tg).to_numpy()
    z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std() > 0 else 0.0
    print(f"  {label:44s} {P.loc[label].game:9.4f} {P.loc[label].game - P.loc['ship'].game:+9.4f} {z:7.2f} {int((d < 0).sum()):4d}/{len(d)}")
print("\nall-seasons parameters of the added terms (offense o*, defense d*; the base map's own come first):")
for label, (mo, md) in SETS.items():
    if label == "ship":
        continue
    p = params[label]
    # the base maps own 5 offensive parameters (a, log2, xlog, prior, tshare) and 3 defensive ones (a, log2, xlog);
    # the added terms follow in the order written
    added_o = mo[len(BO):].strip("&").split("&") if len(mo) > len(BO) else []
    added_d = md[len(BD):].strip("&").split("&") if len(md) > len(BD) else []
    po = [f"{n}={p[f'o{5 + j}']:+.3f}" for j, n in enumerate(added_o)]
    pdd = [f"{n}={p[f'd{3 + j}']:+.3f}" for j, n in enumerate(added_d)]
    print(f"  {label:44s} O: {' '.join(po):44s} D: {' '.join(pdd)}")
