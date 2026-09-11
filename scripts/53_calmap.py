"""A smooth calibration map for the finished ratings, fitted and scored on the out-of-season criterion.

    python scripts/53_calmap.py dump --systems=mspi,def3_p0 --k=2,3,4 --workers=4 --tag=chain
        fit every system once per held-out season and K and keep the ratings (outputs/ratings_<tag>.parquet)

    python scripts/53_calmap.py fit --tag=chain --systems=mspi --k=2,3,4 [--maps=linear,poly2+sat,...] [--splits=exposure]
        for each map -- "<rating family>[+<exposure term>]", the same on both sides, or side-specific with o:d
        (e.g. poly2+sat:linear+sat); families calmap.FAMILIES, exposure terms calmap.EXPOSURES -- fit it
        leave-one-season-out on the TEAM-GAME residuals, score every held-out season with the criterion's scorer,
        and print the ladder against the unmapped system.  Writes outputs/holdout_calmap_<tag>.parquet (scores)
        and outputs/calmap_<tag>.parquet (the per-season parameters, held_out = -1 for the all-seasons fit).

    --players   also score every candidate at PLAYER level -- Kendall tau, a top-k concordance and the dollars
        misallocated on an average two-player trade, every player counted once, by the player's own
        possessions.  The team-game criterion weights a player by how much he played and cannot see the
        bottom of the board at all; this is the loss that can.  Writes
        outputs/holdout_calmap_<tag>_players.parquet.  --truth-lam=<x> re-scores against a
        differently shrunk truth; read any verdict at both lam_plugin and spm.apm_lam (100).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.calmap import (SideMap, dump_ratings, evaluate, frames_for_dump, parse_maps,  # noqa: E402
                            unmapped_rows)
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import SPLITS, Context, Holdout, paired, player_report, pooled  # noqa: E402

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
    # --truth-lam: the penalty of the PRIOR-FREE fit the players are scored against.  Default lam_plugin
    # (low variance, but it shrinks a low-possession player harder than a starter); spm.apm_lam (100) is
    # nearly unbiased and very noisy.  A player-level verdict that flips between them has decided nothing.
    tl = _flag("truth-lam")
    ho = Holdout.from_config(cfg, ks=ks, truth_lam=(float(tl) if tl else None))
    if cmd == "dump":
        dump_ratings(ho, names, OUT / f"ratings_{tag}.parquet", workers=int(_flag("workers", 4)), rankmap=_flag("rankmap"))
        return
    dump = pd.read_parquet(OUT / f"ratings_{tag}.parquet")
    ctx = Context.load(cfg)
    # `frames_for_dump`, not `load_frames`: an in-season system must be scored on the games after its cut,
    # the same rows the criterion uses.  54_track.py and 57_investigate.py always did this; this script
    # did not, and scored a q75 fit on the whole season including its own training games.
    frames = frames_for_dump(ctx, ho.seasons(), dump, names, level=ho.level)
    fams = _list("maps", ["linear", "poly2", "hinge", "linear+unseen", "linear+bins", "linear+log", "linear+log2",
                          "linear+sat", "linear+sat500", "linear+sat2000", "poly2+sat", "hinge+sat"])
    ridge = float(_flag("ridge", 0.0))
    # the criterion is a team-game number, so it cannot see a 200-possession player at all. `--splits` scores
    # each held-out season again inside groups of rows (holdout.SPLITS): "exposure" bins by the smallest
    # TRAINING exposure among the ten on the floor, which asks the bench-player question of the criterion
    # itself rather than of a side diagnostic. Default off; the pooled `split=all` row is always written.
    sp = [x for x in (_flag("splits", "") or "").split(",") if x]
    bad = [x for x in sp if x not in SPLITS]
    if bad:
        raise SystemExit(f"unknown split(s) {bad}; have {sorted(SPLITS)}")
    splits = {x: SPLITS[x] for x in sp} or None
    t0 = time.time()
    # the criterion is a team-game number and a list of PLAYERS is what ships, so --players scores the same
    # candidates a second way: Kendall tau, a top-k concordance and the dollars misallocated on an average
    # two-player trade, every player counting once (holdout.player_scores).  The truth is built once per
    # held-out frame and shared, so every candidate in a run is scored against the same one.
    pout = [] if "--players" in sys.argv else None
    truths = {}
    res, params = [], []
    for system in names:
        for k in ks:
            res.append(unmapped_rows(dump, frames, system, k, splits=splits, ctx=ctx, ho=ho,
                                     player_out=pout, truths=truths))
            for fam in fams:
                map_o, map_d, bend = parse_maps(fam)
                name = f"{system}_{fam.replace(':', '_').replace('|', '_')}"
                r, p = evaluate(dump, frames, system, k, map_o, map_d, name, ridge=ridge, bend=bend,
                                splits=splits, ctx=ctx, ho=ho, player_out=pout, truths=truths)
                res.append(r)
                params.append(p)
                print(f"  {name} K={k} done ({time.time() - t0:.0f}s)", flush=True)
    R = pd.concat(res, ignore_index=True)
    P = pd.concat(params, ignore_index=True)
    R.to_parquet(OUT / f"holdout_calmap_{tag}.parquet", index=False)
    P.to_parquet(OUT / f"calmap_{tag}.parquet", index=False)
    (OUT / "csv").mkdir(exist_ok=True)
    P.round(5).to_csv(OUT / "csv" / f"calmap_{tag}.csv", index=False)
    pool = pooled(R[R.split == "all"])
    print("\n=== pooled over held-out seasons (lower is better); scale_* = the stint-level scalar the season still wants")
    print(pool.pivot_table(index="system", columns="k", values=["game", "mse", "scale_off", "scale_def"]).round(3).to_string())
    for system in names:
        print(f"\n=== paired against {system}, team-game level; negative = better")
        t = paired(R[R.split == "all"], system, "tg")
        print(t[["k", "system", "mean_diff", "se", "z", "wins", "n_seasons"]].to_string(index=False))
    for sname in sp:
        for system in names:
            print(f"\n=== paired against {system} within --splits={sname}, team-game level; negative = better")
            t = paired(R[R.split == sname], system, "tg")
            print(t[["k", "group", "system", "mean_diff", "se", "z", "wins", "n_seasons"]].to_string(index=False))
    if pout:
        PL = pd.concat(pout, ignore_index=True)
        PL.to_parquet(OUT / f"holdout_calmap_{tag}_players.parquet", index=False)
        print()
        print(player_report(PL, ref=names[0]))
        print(f"wrote outputs/holdout_calmap_{tag}_players.parquet")
    print("\n=== parameters, all-seasons fit (held_out = -1)")
    print(P[P.held_out == -1].drop(columns=["held_out"]).round(3).to_string(index=False))
    print(f"\nwrote outputs/holdout_calmap_{tag}.parquet, outputs/calmap_{tag}.parquet ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
