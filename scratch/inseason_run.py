"""The in-season instrument: every kernel at every cut, on BOTH instruments, in one pass.

    python scratch/inseason_run.py --systems=ks11,ks00,blk --cuts=0,0.25,0.5,0.75 [--k=3]
        [--held=all|search|confirm] [--workers=4] [--ref=ks11] [--tag=base] [--lam=2000]

For each cut it dumps every system once (`calmap.dump_ratings`, the runner's own parallel layout), builds the
season frames CUT THE SAME WAY -- so every number is scored on the games the fit has not seen -- fits the
shipping calibration map leave-one-season-out on that dump, and reads the same residual twice:

    game        the criterion: pooled team-game MSE of the mapped prediction (`calmap.evaluate`), lower better
    player      the investigator: the residual variance a player ridge can still put on named players
                (`investigate.attributable`), lower better

and pairs both against `--ref` season by season.  `--held=search` is the odd-indexed half of the held-out
seasons and `confirm` the other half (the 22.7 protocol: choose on one, read on the other).

Writes outputs/inseason_<tag>.csv (the table) and outputs/ratings_insea_<tag>_q<qq>.parquet (the dumps).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from eracoef.calmap import (build_design, dump_ratings, evaluate, fit_theta, load_frames, mapped_ratings,
                            parse_maps, ratings_for, train_of, unmapped_rows)
from eracoef.config import load_config
from eracoef.holdout import Context, Holdout, paired, pooled, predict_season
from eracoef.investigate import attributable

cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
FAM = "linear+log2&xlog&prior&tshare:linear+log2&xlog"      # the shipping map family


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def qtag(q):
    return f"q{int(round(float(q) * 100))}"


def main():
    bases = (flag("systems") or "ks11,ks00,blk").split(",")
    cuts = [float(x) for x in (flag("cuts") or "0,0.25,0.5,0.75").split(",")]
    k = int(flag("k", 3))
    workers = int(flag("workers", cfg.get("holdout", {}).get("workers", 4)))
    ref_base = flag("ref", bases[0])
    tag = flag("tag", "base")
    lam = float(flag("lam", 2000.0))
    which = flag("held", "all")
    ho = Holdout.from_config(cfg, ks=[k])
    seasons = ho.seasons()
    if which == "search":
        seasons = seasons[::2]
    elif which == "confirm":
        seasons = seasons[1::2]
    ctx = Context.load(cfg)
    map_o, map_d, bend = parse_maps(FAM)
    rows, t0 = [], time.time()
    for q in cuts:
        names = [f"{b}_{qtag(q)}" for b in bases]
        dump_path = OUT / f"ratings_insea_{tag}_{qtag(q)}.parquet"
        R = (pd.read_parquet(dump_path) if (dump_path.exists() and "--reuse" in sys.argv)
             else dump_ratings(ho, names, dump_path, workers=workers, verbose=True, held=seasons))
        frames = load_frames(ctx, seasons, level=ho.level, verbose=False, cut=q)
        res, par = [], []
        for s in names:
            if s not in set(R.system):
                continue
            # a system can miss a season (the block baseline has no finished window before 2000): score it
            # on the seasons it dumped, and say how many they were
            have = set(R.loc[(R.system == s) & (R.k == k), "held_out"].astype(int))
            fr = {h: f for h, f in frames.items() if int(h) in have}
            r, pm = evaluate(R, fr, s, k, map_o, map_d, f"{s}_map", bend=bend)
            res.append(r)
            par.append(pm)
            res.append(unmapped_rows(R, fr, s, k))
            # the second instrument, on the same mapped residual
            D = build_design(R, fr, s, k, map_o, map_d)
            for h, f in fr.items():
                th = fit_theta(D, exclude_h=h, map_o=map_o, map_d=map_d)
                rat = mapped_ratings(ratings_for(R, s, k, h), th, map_o, map_d, D.scale_o, D.scale_d,
                                     extra=f.covariates(k, train_of(R, s, k, h)))
                p = predict_season(rat, f.wd, level=ho.level)
                a = attributable(f.Zo, f.Zd, p.y - p.pred, p.w, lam)
                rows.append(dict(cut=q, system=s, held_out=int(h), w=float(p.w.sum()), **a))
        res = pd.concat(res, ignore_index=True)
        res["cut"] = q
        # the map parameters, so a board can be shipped under the map fitted on these fits (60_season_board.py)
        pd.concat(par, ignore_index=True).to_parquet(OUT / f"calmap_insea_{tag}_{qtag(q)}.parquet", index=False)
        P = pooled(res).set_index("system")
        ref = f"{ref_base}_{qtag(q)}"
        pg = paired(res[res.split == "all"], f"{ref}_map", value="tg")
        print(f"\n=== cut {q:g} ({len(seasons)} held-out seasons, {time.time() - t0:.0f}s) "
              f"-- lower is better; 'vs ref' is against {ref}", flush=True)
        print(f"  {'system':22s} {'game':>9s} {'unmapped':>9s} {'stint':>10s} {'covered':>8s} "
              f"{'vs ref':>8s} {'z':>6s} {'wins':>5s}")
        for b in bases:
            s = f"{b}_{qtag(q)}"
            if f"{s}_map" not in P.index:
                continue
            m, u = P.loc[f"{s}_map"], P.loc[s]
            d = pg[pg.system == f"{s}_map"]
            dz = (float(d.mean_diff.iloc[0]), float(d.z.iloc[0]), int(d.wins.iloc[0])) if len(d) else (0.0, 0.0, 0)
            n_s = int(res[(res.system == s)].held_out.nunique())
            print(f"  {s:22s} {m.game:9.3f} {u.game:9.3f} {m.mse:10.2f} {u.covered:8.3f} "
                  f"{dz[0]:8.3f} {dz[1]:6.2f} {dz[2]:3d}/{n_s}")
    A = pd.DataFrame(rows)
    A.to_csv(OUT / f"inseason_{tag}_attrib.csv", index=False)
    print("\n=== the attribution instrument (investigate.attributable, ridge "
          f"{lam:g}): residual variance a player ridge can put on players\n")
    print(f"  {'system':22s} {'cut':>5s} {'total':>9s} {'player':>9s} {'vs ref':>8s} {'z':>6s} {'wins':>5s}")
    for q in cuts:
        ref = f"{ref_base}_{qtag(q)}"
        base = A[(A.cut == q) & (A.system == ref)].set_index("held_out")["player"]
        for b in bases:
            s = f"{b}_{qtag(q)}"
            d = A[(A.cut == q) & (A.system == s)].set_index("held_out")
            if not len(d):
                continue
            diff = (d["player"] - base).dropna()
            z = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) if len(diff) > 1 and diff.std(ddof=1) > 0 else 0.0
            print(f"  {s:22s} {q:5.2f} {d.total.mean():9.3f} {d.player.mean():9.3f} "
                  f"{diff.mean():8.3f} {z:6.2f} {int((diff < 0).sum()):3d}/{len(diff)}")
    print(f"\ndone ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
