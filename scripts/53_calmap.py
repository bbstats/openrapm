"""A smooth calibration map for the finished ratings, fitted and scored on the out-of-season criterion.

    python scripts/53_calmap.py dump --systems=mspi,def3_p0 --k=2,3,4 --workers=4 --tag=chain
        fit every system once per held-out season and K and keep the ratings (outputs/ratings_<tag>.parquet)

    python scripts/53_calmap.py fit --tag=chain --systems=mspi --k=2,3,4 [--maps=linear,poly2+sat,...]
        for each map -- "<rating family>[+<exposure term>]", the same on both sides, or side-specific with o:d
        (e.g. poly2+sat:linear+sat); families calmap.FAMILIES, exposure terms calmap.EXPOSURES -- fit it
        leave-one-season-out on the TEAM-GAME residuals, score every held-out season with the criterion's scorer,
        and print the ladder against the unmapped system.  Writes outputs/holdout_calmap_<tag>.parquet (scores)
        and outputs/calmap_<tag>.parquet (the per-season parameters, held_out = -1 for the all-seasons fit).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.calmap import SideMap, dump_ratings, evaluate, load_frames, parse_maps, unmapped_rows  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context, Holdout, paired, pooled  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"


def _flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def _list(name, default, conv=str):
    v = _flag(name)
    return default if v is None else [conv(x) for x in v.split(",") if x]


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "fit"
    tag = _flag("tag", "chain")
    names = _list("systems", ["mspi"])
    ks = _list("k", [2, 4], int)
    ho = Holdout.from_config(cfg, ks=ks)
    if cmd == "dump":
        dump_ratings(ho, names, OUT / f"ratings_{tag}.parquet", workers=int(_flag("workers", 4)), rankmap=_flag("rankmap"))
        return
    dump = pd.read_parquet(OUT / f"ratings_{tag}.parquet")
    ctx = Context.load(cfg)
    frames = load_frames(ctx, ho.seasons(), level=ho.level)
    fams = _list("maps", ["linear", "poly2", "hinge", "linear+unseen", "linear+bins", "linear+log", "linear+log2",
                          "linear+sat", "linear+sat500", "linear+sat2000", "poly2+sat", "hinge+sat"])
    ridge = float(_flag("ridge", 0.0))
    t0 = time.time()
    res, params = [], []
    for system in names:
        for k in ks:
            res.append(unmapped_rows(dump, frames, system, k))
            for fam in fams:
                map_o, map_d, bend = parse_maps(fam)
                name = f"{system}_{fam.replace(':', '_').replace('|', '_')}"
                r, p = evaluate(dump, frames, system, k, map_o, map_d, name, ridge=ridge, bend=bend)
                res.append(r)
                params.append(p)
                print(f"  {name} K={k} done ({time.time() - t0:.0f}s)", flush=True)
    R = pd.concat(res, ignore_index=True)
    P = pd.concat(params, ignore_index=True)
    R.to_parquet(OUT / f"holdout_calmap_{tag}.parquet", index=False)
    P.to_parquet(OUT / f"calmap_{tag}.parquet", index=False)
    (OUT / "csv").mkdir(exist_ok=True)
    P.round(5).to_csv(OUT / "csv" / f"calmap_{tag}.csv", index=False)
    pool = pooled(R)
    print("\n=== pooled over held-out seasons (lower is better); scale_* = the stint-level scalar the season still wants")
    print(pool.pivot_table(index="system", columns="k", values=["game", "mse", "scale_off", "scale_def"]).round(3).to_string())
    for system in names:
        print(f"\n=== paired against {system}, team-game level; negative = better")
        t = paired(R, system, "tg")
        print(t[["k", "system", "mean_diff", "se", "z", "wins", "n_seasons"]].to_string(index=False))
    print("\n=== parameters, all-seasons fit (held_out = -1)")
    print(P[P.held_out == -1].drop(columns=["held_out"]).round(3).to_string(index=False))
    print(f"\nwrote outputs/holdout_calmap_{tag}.parquet, outputs/calmap_{tag}.parquet ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
