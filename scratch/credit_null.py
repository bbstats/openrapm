"""Does the credit score survive its own noise floor?

`investigate.attributable` reports how much of the out-of-season residual a player ridge can put on players.
A ridge on 20 games of noise finds "players" too, so the level of that number is mostly noise.  The question
that matters is whether the DIFFERENCE between two boards is: a worse board leaves a bigger residual, and a
bigger residual raises the noise floor as well as the real signal.

For each held-out season and each board: the score, a stratified permutation null (the residual shuffled only
among rows of similar length, so the lineup link is broken and the row scale is kept), and their difference.
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
from eracoef.holdout import Context, Holdout, predict_season
from eracoef.investigate import attributable

cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
FAM = "linear+log2&xlog&prior&tshare:linear+log2&xlog"
K, LAM, NPERM = 3, 2000.0, 4
SEASONS = [int(a) for a in sys.argv[1:] if a.isdigit()] or [2002, 2006, 2010, 2014, 2018, 2021, 2023, 2025]
TAG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tag=")), "ship")
dump = pd.read_parquet(OUT / f"ratings_insea_{TAG}_q75.parquet")
SYSTEMS = (next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--systems=")), "") or
           ",".join(sorted(dump.system.unique()))).split(",")
ho = Holdout.from_config(cfg, ks=[K])
ctx = Context.load(cfg)
map_o, map_d, _ = parse_maps(FAM)
frames = load_frames(ctx, ho.seasons(), level=ho.level, verbose=False, cut=cut_of(dump, SYSTEMS[0]))

rows, t0 = [], time.time()
for s in SYSTEMS:
    have = set(dump.loc[(dump.system == s) & (dump.k == K), "held_out"].astype(int))
    fr = {h: f for h, f in frames.items() if h in have}
    D = build_design(dump, fr, s, K, map_o, map_d)
    for h in SEASONS:
        if h not in fr:
            continue
        f = fr[h]
        th = fit_theta(D, exclude_h=h, map_o=map_o, map_d=map_d)
        rat = mapped_ratings(ratings_for(dump, s, K, h), th, map_o, map_d, D.scale_o, D.scale_d,
                             extra=f.covariates(K, train_of(dump, s, K, h)))
        p = predict_season(rat, f.wd, level=ho.level)
        r = p.y - p.pred
        a = attributable(f.Zo, f.Zd, r, p.w, LAM)
        rng = np.random.default_rng(7)
        strata = pd.qcut(p.poss, 20, labels=False, duplicates="drop")
        idx_by_bin = [np.flatnonzero(strata == b) for b in np.unique(strata)]
        nulls = []
        for _ in range(NPERM):
            rp = r.copy()
            for idx in idx_by_bin:
                rp[idx] = r[rng.permutation(idx)]
            nulls.append(attributable(f.Zo, f.Zd, rp, p.w, LAM)["player"])
        rows.append(dict(system=s, held_out=h, total=a["total"], player=a["player"],
                         noise=float(np.mean(nulls)), real=a["player"] - float(np.mean(nulls))))
        if s == SYSTEMS[0]:          # the ceiling: the same two numbers with NO ratings at all
            r0 = p.y - p.base_pred
            a0 = attributable(f.Zo, f.Zd, r0, p.w, LAM)
            n0 = []
            for _ in range(NPERM):
                rp = r0.copy()
                for idx in idx_by_bin:
                    rp[idx] = r0[rng.permutation(idx)]
                n0.append(attributable(f.Zo, f.Zd, rp, p.w, LAM)["player"])
            rows.append(dict(system="no ratings", held_out=h, total=a0["total"], player=a0["player"],
                             noise=float(np.mean(n0)), real=a0["player"] - float(np.mean(n0))))
    print(f"  {s} ({time.time() - t0:.0f}s)", flush=True)

R = pd.DataFrame(rows)
R.to_csv(OUT / "csv" / "credit_null.csv", index=False)
g = R.groupby("system", sort=False)[["total", "player", "noise", "real"]].mean()
print(f"\n=== the credit score against its noise floor, {len(SEASONS)} seasons, ridge {LAM:g}\n")
print("  player = what the ridge attributes;  noise = the same on a lineup-shuffled residual")
print("  real   = player - noise: the part that is actually about who was on the floor (lower = better board)\n")
print(g.round(2).to_string())
z0 = R[R.system == "no ratings"]
print("")
print(f"  real credit captured = 1 - real(board) / real(no ratings) = "
      f"{100 * (1 - g.loc[SYSTEMS[0], 'real'] / z0.real.mean()):.1f}%")
print(f"  no ratings: player {z0.player.mean():.1f}, noise {z0.noise.mean():.1f}, real {z0.real.mean():.1f}")
base = R[R.system == SYSTEMS[0]].set_index("held_out")
print("\n  paired against", SYSTEMS[0], "(positive = worse board)\n")
print(f"  {'system':18s} {'d player':>9s} {'d noise':>9s} {'d real':>8s} {'z(real)':>8s} {'seasons':>7s}")
for s in SYSTEMS[1:]:
    d = R[R.system == s].set_index("held_out")
    j = d.join(base, rsuffix="_b", how="inner")
    dr = (j.real - j.real_b).dropna()
    z = float(dr.mean() / (dr.std(ddof=1) / np.sqrt(len(dr)))) if len(dr) > 1 else np.nan
    print(f"  {s:18s} {(j.player - j.player_b).mean():9.2f} {(j.noise - j.noise_b).mean():9.2f} "
          f"{dr.mean():8.2f} {z:8.2f} {len(dr):7d}")
