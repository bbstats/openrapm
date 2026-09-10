"""How wrong the prior is at the BOTTOM of the board, which is the only number ruling 1's second sentence
cares about: *"low minute players should have REALLY GOOD REASONABLE PRIORS now."*

A single-season rating is 20% on-court evidence on offense and 44% on defense, so for a bench player the
prior essentially IS the rating -- and the out-of-season criterion cannot see him, because it is scored at
TEAM-GAME level, where a 200-possession player is a rounding error.  This script scores the prior directly,
leave-one-window-out, and reports the error by how many possessions the player himself played.

    python scripts/61_lowposs.py --systems=ks00_lam05_ow_w0.25,board_bioD

For each panel window w the prior is refit with w excluded and asked to predict w's own rows; "actual" is the
row's training target -- the player's value pooled over his OTHER windows (`training_rows`), or the exact
other-window value of the pair (`pair_rows`) when the feature list carries a PAST block.  Buckets are
SEASON-EQUIVALENT possessions, the window's possessions divided by its length, so "under 500" means the same
thing it means in `low_poss_threshold` now that the board rates one season.

The honest caveat: on the pooled path a row in window v has a target that pools v's other windows, w
included, so excluding w does not remove every trace of it from training.  That is exactly the exclusion the
shipped prior uses, so the number is the production one; the pair path (offense) has no such trace.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context  # noqa: E402
from eracoef.systems import registry  # noqa: E402
from eracoef.windows import label_step  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)

# season-equivalent possessions.  500 and 1500 are `low_poss_threshold` / `starter_poss_threshold`.
EDGES = [0.0, 250.0, 500.0, 1500.0, 4500.0, np.inf]
LABELS = ["<250", "250-500", "500-1500", "1500-4500", "4500+"]


def _flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def prior_for(system, ctx):
    """The GBDTPrior the system's chain would build (spm.chain_offset's arguments, per side)."""
    t = system.target
    col = t if (t == "apm" or t.startswith("blend")) else None
    out = {}
    for side in ("O", "D"):
        params = system.gbdt_params if side == "O" else (system.gbdt_params_d or system.gbdt_params)
        wd = system.win_decay if side == "O" else (system.win_decay if system.win_decay_d is None else system.win_decay_d)
        tcol = col if side == "O" else (system.target_d if side == "D" and system.target_d else col)
        if tcol is not None and tcol not in ("apm",) and not tcol.startswith("blend"):
            tcol = None if tcol == "rapm1" else tcol
        turn = (system.turn or False) if side in (system.turn_sides or ()) else False
        turn = ("pairs" if turn == "pairs" else bool(turn)) if turn else False
        out[side] = ctx.prior(system.mode, tcol, params, system.panel, system.gbdt_features,
                              float(wd), turn=turn, folds=int(system.gbdt_folds or 0))
    return out


def score(system, ctx, panel) -> pd.DataFrame:
    step = label_step(sorted(panel.window.unique()))
    poss = panel[["player_id", "window", "side", "poss"]].rename(columns={"poss": "poss_feat"})
    priors = prior_for(system, ctx)
    parts = []
    for side in ("O", "D"):
        prior = priors[side]
        rows = prior.rows(side).merge(poss[poss.side == side].drop(columns="side"), on=["player_id", "window"], how="left")
        for w in sorted(panel.window.unique()):
            sub = rows[rows.window == w]
            if not len(sub):
                continue
            m, _ = prior.model(side, exclude={w})
            pred = np.asarray(m.predict(sub[prior.features[side]].to_numpy(dtype=float)), dtype=float)
            parts.append(pd.DataFrame(dict(side=side, window=w, player_id=sub.player_id.to_numpy(),
                                           poss_feat=sub.poss_feat.to_numpy(),
                                           target=sub.target.to_numpy(), pred=pred)))
    d = pd.concat(parts, ignore_index=True)
    d["bucket"] = pd.cut(d.poss_feat / float(step), bins=EDGES, labels=LABELS, right=False)
    d["err2"] = (d.pred - d.target) ** 2
    # the null a prior has to beat at the bottom: predict the panel's own weighted mean for everyone
    ref = d.groupby("side").target.transform("mean")
    d["null2"] = (ref - d.target) ** 2
    g = d.groupby(["side", "bucket"], observed=True).agg(n=("err2", "size"), rmse=("err2", "mean"),
                                                         null=("null2", "mean"),
                                                         sd_pred=("pred", "std"), sd_act=("target", "std"))
    g["rmse"], g["null"] = np.sqrt(g.rmse), np.sqrt(g["null"])
    g["skill"] = 1.0 - (g.rmse / g["null"]) ** 2
    return g.reset_index().assign(system=system.name), d.assign(system=system.name)


def main():
    cfg = load_config()
    ctx = Context.load(cfg)
    reg = registry(cfg)
    names = (_flag("systems") or "ks00_lam05_ow_w0.25").split(",")
    t0 = time.time()
    out, raw = [], []
    for n in names:
        g, d = score(reg[n], ctx, ctx.panel_frame(reg[n].panel))
        out.append(g)
        raw.append(d)
        print(f"  {n} done ({time.time() - t0:.0f}s)", flush=True)
    R = pd.concat(out, ignore_index=True)
    cols = ["system", "side", "bucket", "n", "rmse", "null", "skill", "sd_pred", "sd_act"]
    print("\n=== prior error by SEASON-EQUIVALENT possessions, leave-one-window-out "
          "(rmse against the player's other windows; null = predict the panel mean; skill = 1 - mse/null_mse)")
    for side in ("O", "D"):
        print(f"\n-- {side}")
        print(R[R.side == side][cols].drop(columns="side").to_string(index=False))
    if len(names) > 1:
        # the same rows, the same targets and the same window folds, so the difference of squared errors is
        # paired row by row; the z is over the panel WINDOWS (10 of them), which is the unit that repeats
        base = raw[0].set_index(["side", "window", "player_id", "poss_feat"], drop=False)
        print()
        print("=== paired against " + names[0] + ", per bucket; negative = better (z over the panel windows)")
        rep = []
        for d in raw[1:]:
            m = base.reset_index(drop=True).merge(d, on=["side", "window", "player_id", "poss_feat", "target"],
                                                  suffixes=("_b", ""))
            m["diff"] = (m.pred - m.target) ** 2 - (m.pred_b - m.target) ** 2
            for (side, bucket), gg in m.groupby(["side", "bucket"], observed=True):
                per_w = gg.groupby("window")["diff"].mean()
                se = per_w.std(ddof=1) / np.sqrt(len(per_w)) if len(per_w) > 1 else np.nan
                rep.append(dict(system=d.system.iloc[0], side=side, bucket=bucket, n=len(gg),
                                d_mse=per_w.mean(), se=se, z=per_w.mean() / se if se else np.nan,
                                wins=int((per_w < 0).sum()), n_win=len(per_w)))
        print(pd.DataFrame(rep).round(4).to_string(index=False))
    Path(cfg["_root"], "outputs").mkdir(exist_ok=True)
    R.to_csv(Path(cfg["_root"]) / "outputs" / "csv" / "lowposs.csv", index=False)
    print(f"\nwrote outputs/csv/lowposs.csv ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
