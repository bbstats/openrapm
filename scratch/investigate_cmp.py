"""The investigator's score for several tracked systems on the same held-out rows: the share of the
out-of-season residual's variance that a player ridge can attribute to players, pooled over the 28 seasons,
and paired against the first system.

    python scratch/investigate_cmp.py <system1> <system2> ... [--lam=2000] [--maps=<family>]
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from eracoef.calmap import (build_design, cut_of, fit_theta, load_frames, mapped_ratings, parse_maps,
                            ratings_for, train_of)
from eracoef.config import load_config
from eracoef.holdout import Context, Holdout, predict_season
from eracoef.investigate import attributable

cfg = load_config()
names = [a for a in sys.argv[1:] if not a.startswith("--")]
lam = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--lam=")), 2000.0))
fam = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--maps=")), "linear+log2&xlog&prior&tshare:linear+log2&xlog")
K = 3
ho = Holdout.from_config(cfg, ks=[K])
ctx = Context.load(cfg)
map_o, map_d, _ = parse_maps(fam)
t0 = time.time()
res = {}
FR: dict = {}                     # one set of frames per cut: an in-season system is scored after its cut
for s in names:
    dump = pd.read_parquet(f"outputs/ratings_track_{s}.parquet")
    cut = cut_of(dump, s)
    if cut not in FR:
        FR[cut] = load_frames(ctx, ho.seasons(), level=ho.level, verbose=False, cut=cut)
    frames = FR[cut]
    D = build_design(dump, frames, s, K, map_o, map_d)
    rows = []
    for h, f in frames.items():
        th = fit_theta(D, exclude_h=h, map_o=map_o, map_d=map_d)
        rat = mapped_ratings(ratings_for(dump, s, K, h), th, map_o, map_d, D.scale_o, D.scale_d,
                             extra=f.covariates(K, train_of(dump, s, K, h)))
        p = predict_season(rat, f.wd, level=ho.level)
        a = attributable(f.Zo, f.Zd, p.y - p.pred, p.w, lam)
        rows.append(dict(held_out=h, w=float(p.w.sum()), **a))
    res[s] = pd.DataFrame(rows)
    print(f"  {s} ({time.time() - t0:.0f}s)", flush=True)
first = names[0]
print(f"\ninvestigator's score, K = {K}, ridge {lam:g}, {len(frames)} seasons: the out-of-season residual variance a "
      f"player ridge attributes to players (lower is better; 'total' is the residual variance itself)\n")
print(f"  {'system':40s} {'total':>9s} {'player':>9s} {'vs first':>9s} {'z':>6s} {'wins':>6s} {'ss_o':>7s} {'ss_d':>7s}")
for s, d in res.items():
    b = res[first]
    tot = float(np.average(d.total, weights=d.w)); pl = float(np.average(d.player, weights=d.w))
    diff = (d.player - b.player).to_numpy()
    z = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) if diff.std() > 0 else 0.0
    print(f"  {s:40s} {tot:9.3f} {pl:9.3f} {pl - float(np.average(b.player, weights=b.w)):+9.3f} {z:6.2f} "
          f"{int((diff < 0).sum()):3d}/{len(diff)} {float(np.average(d.ss_o, weights=d.w)):7.3f} {float(np.average(d.ss_d, weights=d.w)):7.3f}")
