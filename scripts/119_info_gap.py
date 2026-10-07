"""The information-gap pilot (stage 4): on the tracked 2014-15 attempts, does tracking say more than play-by-play?

    python scripts/119_info_gap.py [--folds=5] [--fractions=0.1,0.3,1.0] [--seasons=2015,2016] [--tag=] [--base=<dir>]

Uses the 2014-15 model table (scripts/120 --table=1) joined to the shot log (outputs/shotclock/joined_2015.parquet,
scripts/116).  Held out by GAME, five folds:

    pbp       the play-by-play model, every block (shotmodel.BLOCKS)
    teacher   the same plus the tracking block: closest-defender distance, the true shot clock, dribbles,
              touch time, catch-and-shoot
    spot+trk  the spot and the tracking block only (the KOBE-like model)

Every arm is fitted with shooter-season and arena offsets and scored with them zeroed (a league-average shooter),
so the arms differ only in what they can see.  A learning curve (10 / 30 / 100% of the training folds) shows
whether more tracked attempts would help.

Then the size of the owner's update: v, the variance of tracking quality among attempts play-by-play cannot tell
apart, by shot class -- estimated as the covariance of (teacher - pbp) between two teachers fitted on DIFFERENT
games, both scored on a third set, so neither teacher's own error inflates it.

The kill gate (the plan): the teacher beats play-by-play by >= 0.003 log loss at game-paired z <= -3, and the
residual sd sqrt(v) is >= 0.03 in at least one shot class.  Writes outputs/shottest/info_gap_*.csv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import shotmodel as sm  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.shottest import logloss  # noqa: E402

pd.set_option("display.width", 220, "display.precision", 4)
PBP = list(sm.BLOCKS)
ARMS = {"pbp": PBP, "teacher": PBP + ["tracking"], "spot+trk": ["spot", "tracking"]}


def tracked(cfg, seasons=(2015,)) -> pd.DataFrame:
    """The tracked attempts of these seasons (scripts/121's reader: the 2014-15 log, the 2015-16 movement labels)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("t121", Path(__file__).resolve().parent / "121_tracking_teacher.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    parts = [m.tracked_season(cfg, s) for s in seasons]
    return pd.concat([p for p in parts if p is not None], ignore_index=True)


def fold_of(games: pd.Series, k: int) -> np.ndarray:
    u = np.sort(games.unique())
    rng = np.random.default_rng(7)
    lab = dict(zip(u, rng.permutation(len(u)) % k))
    return games.map(lab).to_numpy()


def score_arm(f, blocks, train_mask, test_mask, frac=1.0, seed=0):
    tr = f[train_mask]
    if frac < 1.0:
        g = tr["game_id"].unique()
        keep = np.random.default_rng(seed).choice(g, max(int(len(g) * frac), 5), replace=False)
        tr = tr[tr["game_id"].isin(set(keep))]
    m = sm.fit(tr, blocks, offsets=True)
    m.seasons = ()                                         # one season held out by game, not by season
    return sm.probability(m.predict_raw(f[test_mask]))


def main():
    check_flags()
    cfg = load_config()
    K = int(flag("folds", "5"))
    fracs = [float(x) for x in flag("fractions", "0.1,0.3,1.0").split(",")]
    seasons = tuple(int(s) for s in flag("seasons", "2015").split(","))
    f = tracked(cfg, seasons)
    print(f"{len(f):,} tracked attempts in {f.game_id.nunique()} games")
    fold = fold_of(f["game_id"], K)
    preds = {(a, fr): np.full(len(f), np.nan) for a in ARMS for fr in fracs}
    for k in range(K):
        te = fold == k
        for a, blocks in ARMS.items():
            for fr in fracs:
                if a == "spot+trk" and fr < 1.0:
                    continue
                preds[(a, fr)][te] = score_arm(f, blocks, ~te, te, fr, seed=k)
        print(f"  fold {k + 1}/{K} done", flush=True)
    y = f["made"].to_numpy(float)
    sub = sm.submodel_of(f)
    rows = []
    for (a, fr), p in preds.items():
        if np.isnan(p).all():
            continue
        ll = logloss(y, p)
        rows.append(dict(arm=a, fraction=fr, logloss=float(ll.mean()),
                         **{f"ll_{s}": float(ll[sub == s].mean()) for s in sm.SUBMODELS}))
    T = pd.DataFrame(rows)
    print("\nheld-out log loss (by game, five folds):")
    print(T.to_string(index=False))
    g = f["game_id"].to_numpy()
    d = pd.Series(logloss(y, preds[("teacher", 1.0)]) - logloss(y, preds[("pbp", 1.0)])).groupby(g).mean()
    z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    print(f"\nteacher minus pbp: {d.mean():+.5f} per attempt, game-paired z {z:+.2f}, better in {int((d < 0).sum())}/{len(d)} games")

    # v: two teachers on different games, both scored on a third set.  --base=<dir>: the play-by-play quality is a
    # priced candidate's (data/shotq/<dir>, fitted on other seasons), so v is what tracking still knows beyond IT
    base = flag("base", "")
    qb = None
    if base:
        P_ = pd.concat([pd.read_parquet(Path(cfg["_root"]) / "data" / "shotq" / base / f"{s}.parquet",
                                        columns=["game_id", "action_number", "q"]) for s in seasons])
        qb = f[["game_id", "action_number"]].merge(P_, on=["game_id", "action_number"], how="left")["q"].to_numpy(float)
        if np.isnan(qb).any():
            raise SystemExit(f"{base}: {int(np.isnan(qb).sum())} tracked attempts unpriced")
    third = fold_of(f["game_id"], 3)
    gaps = []
    for k in range(3):
        a_, b_, c_ = (third == k), (third == (k + 1) % 3), (third == (k + 2) % 3)
        m_pbp = qb[c_] if qb is not None else score_arm(f, PBP, a_ | b_, c_)
        t1 = score_arm(f, ARMS["teacher"], a_, c_)
        t2 = score_arm(f, ARMS["teacher"], b_, c_)
        gaps.append(pd.DataFrame(dict(idx=np.flatnonzero(c_), m=m_pbp, g1=t1 - m_pbp, g2=t2 - m_pbp)))
    G = pd.concat(gaps).set_index("idx").sort_index()
    G["sub"] = sub[G.index]
    G["band"] = sm.update_band_of(f.loc[G.index])
    sc = f.loc[G.index, "sc_eff"].to_numpy(float)
    G["clock"] = np.where(sc < 4, "0-4", np.where(sc < 10, "4-10", "10+"))
    V = G.groupby(["sub", "band", "clock"]).apply(
        lambda d: pd.Series(dict(n=len(d), v=float(np.cov(d.g1, d.g2)[0, 1]), m=float(d.m.mean()))), include_groups=False)
    V["sd"] = np.sqrt(V["v"].clip(lower=0.0))
    V["weight"] = (V["v"].clip(lower=0.0) / (V["m"] * (1 - V["m"]))).clip(upper=0.15)
    print("\nthe update's size by shot class: sd of tracking quality among attempts play-by-play cannot tell apart,")
    print("and the weight a shot's own result gets (v / m(1-m), capped at 0.15)")
    print(V.round(4).to_string())
    out = Path(cfg["_root"]) / "outputs" / "shottest"
    out.mkdir(parents=True, exist_ok=True)
    tag = flag("tag", "")
    T.to_csv(out / f"info_gap_logloss{tag}.csv", index=False)
    V.to_csv(out / f"info_gap_v{tag}.csv")
    gate = (d.mean() <= -0.003) and (z <= -3.0) and bool((V["sd"] >= 0.03).any())
    print(f"\nkill gate {'PASSED' if gate else 'FAILED'}: teacher {d.mean():+.4f} (z {z:+.2f}); largest residual sd {V['sd'].max():.3f}")


if __name__ == "__main__":
    main()
