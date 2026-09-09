"""Why a ridge on the residual is not zero: the same season, the games the fit SAW against the games it did not.

For held-out season H at cut 0.75 the fit trained on the first three quarters of H (and the two seasons
before it) and is scored on the last quarter.  Same season, same players, mostly the same lineups -- the only
difference is whether the fit had the games.  Run the identical residual ridge on both halves.

In-sample, a least-squares residual is orthogonal to the design and the second ridge finds nothing; that is
the whole reason within-season stint error cannot see attribution (FINDINGS memory trap 2).  Out of sample
there is no such guarantee, which is what the credit score reads.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from eracoef.calmap import (build_design, cut_of, fit_theta, load_frames, mapped_ratings, parse_maps,
                            ratings_for, train_of)
from eracoef.config import load_config
from eracoef.holdout import Context, Holdout, cut_season, predict_season
from eracoef.inseason import season_frac
from eracoef.investigate import attributable

cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
FAM = "linear+log2&xlog&prior&tshare:linear+log2&xlog"
SYS, K, LAM, CUT = "ks52_lam05_q75", 3, 2000.0, 0.75
SEASONS = [int(a) for a in sys.argv[1:]] or [2010, 2014, 2018, 2022, 2025]

dump = pd.read_parquet(OUT / "ratings_insea_ship_q75.parquet")
ho = Holdout.from_config(cfg, ks=[K])
ctx = Context.load(cfg)
map_o, map_d, _ = parse_maps(FAM)
frames = load_frames(ctx, ho.seasons(), level=ho.level, verbose=False, cut=cut_of(dump, SYS))
D = build_design(dump, frames, SYS, K, map_o, map_d)

rows, t0 = [], time.time()
for h in SEASONS:
    f = frames[h]
    th = fit_theta(D, exclude_h=h, map_o=map_o, map_d=map_d)
    rat = mapped_ratings(ratings_for(dump, SYS, K, h), th, map_o, map_d, D.scale_o, D.scale_d,
                         extra=f.covariates(K, train_of(dump, SYS, K, h)))
    full = ctx.design([h], "pts")
    frac = season_frac(full.games)[full.rows["game_idx"].to_numpy()]
    m_ps = full.spec.n_ps
    for label, mask in (("seen (first 75%)", frac < CUT),
                        # the same NUMBER of games, also seen by the fit: removes the sample-size advantage
                        ("seen (3rd quarter only)", (frac >= 0.5) & (frac < CUT)),
                        ("unseen (last 25%)", frac >= CUT)):
        wd = full.subset(mask)
        p = predict_season(rat, wd, level=ho.level)          # the level is refit on whichever rows are scored
        Zo, Zd = wd.X[:, :m_ps].tocsr(), wd.X[:, m_ps:2 * m_ps].tocsr()
        r = p.y - p.pred
        a = attributable(Zo, Zd, r, p.w, LAM)
        # the noise floor: the same residual with its link to the LINEUPS broken but its own row kept
        # comparable -- shuffled only among rows of similar length, because points per 100 on a two-possession
        # stint is enormous and a blind shuffle would hand that value to a forty-possession row
        rng = np.random.default_rng(0)
        strata = pd.qcut(p.poss, 20, labels=False, duplicates="drop")
        nulls = []
        for _ in range(3):
            rp = r.copy()
            for b in np.unique(strata):
                idx = np.flatnonzero(strata == b)
                rp[idx] = r[rng.permutation(idx)]
            nulls.append(attributable(Zo, Zd, rp, p.w, LAM)["player"])
        null = float(np.mean(nulls))
        rows.append(dict(held_out=h, rows=label, poss=float(p.w.sum()), total=a["total"], player=a["player"],
                         noise=null, real=a["player"] - null,
                         left_o=np.sqrt(a["ss_o"]), left_d=np.sqrt(a["ss_d"])))
    print(f"  {h} ({time.time() - t0:.0f}s)", flush=True)

R = pd.DataFrame(rows)
print("\n=== the same residual ridge on the games the fit SAW and the games it did not\n")
print("  player = residual variance the ridge puts on players; share = as a % of the residual variance")
print("  left o/d = the per-player miss it finds, points per 100\n")
g = R.groupby("rows", sort=False)[["poss", "total", "player", "noise", "real", "left_o", "left_d"]].mean()
print(g.round(3).to_string())
print("\nper season:\n")
print(R.pivot(index="held_out", columns="rows", values="real").round(2).to_string())
