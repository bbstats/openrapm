"""The tracking teacher: out-of-fold tracking quality for every tracked attempt (stage 7 of the shot-quality build).

    python scripts/121_tracking_teacher.py [--folds=5] [--seasons=2015,2016]

Our own teacher (the owner: "build our own"), the STATS era only: the play-by-play model's blocks plus the
tracking block (shotmodel.block_columns "tracking": closest-defender distance, the true shot clock, dribbles,
touch time, catch-and-shoot), shooter-season and arena offsets fitted and zeroed, so it prices a league-average
shooter.  Each attempt's label comes from the fit that did NOT see its game (five folds by game, within its
season), so a student trained on these labels never learns an attempt's own result through them.

Tracked attempts:
    2015   the 2014-15 shot log joined by shot order (outputs/shotclock/joined_2015.parquet, scripts/116)
    2016   the 2015-16 movement labels (data/tracking/movement_shots_2016.parquet, scripts/124), when built

Writes data/shotq/teacher/labels.parquet: season, game_id, action_number, q_teacher (a league-average shooter:
the tracking QUALITY, for the update and the yardstick), q_teacher_full (this shooter, offsets kept: the soft
label a student with its own shooter offsets must be taught with), fold.
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

BLOCKS = list(sm.BLOCKS) + ["tracking"]


def tracked_season(cfg, season: int) -> pd.DataFrame | None:
    root = Path(cfg["_root"])
    t = pd.read_parquet(root / "data" / "shotq" / "_table" / f"{season}.parquet")
    t = t[(t.phase == "RS") & ~t.heave]
    if season == 2015:
        j = pd.read_parquet(root / "outputs" / "shotclock" / "joined_2015.parquet",
                            columns=["game_id", "action_number", "shot_clock", "dribbles", "touch_time", "def_dist",
                                     "log_dist"])
        j = j.rename(columns={"shot_clock": "true_sc", "log_dist": "true_dist"})
    elif season == 2016:
        p = root / "data" / "tracking" / "movement_shots_2016.parquet"
        if not p.exists():
            return None
        j = pd.read_parquet(p, columns=["game_id", "action_number", "mv_shot_clock", "dribbles", "touch_time", "def_dist",
                                        "mv_dist"])
        j = j.rename(columns={"mv_shot_clock": "true_sc", "mv_dist": "true_dist"})
    else:
        raise ValueError(f"no tracked attempts for {season} (the owner's split: STATS era only)")
    f = t.merge(j, on=["game_id", "action_number"], how="inner")
    f = f[f["def_dist"].notna() & f["touch_time"].notna() & f["dribbles"].notna()].reset_index(drop=True)
    # an attempt without a true distance keeps the coded logged one (shotmodel's tracking block reads true_dist)
    f["true_dist"] = f["true_dist"].fillna(f["dist_xy"])
    f["touch_time"] = f["touch_time"].clip(lower=0.0)
    return f


def main():
    check_flags()
    cfg = load_config()
    K = int(flag("folds", "5"))
    seasons = [int(s) for s in flag("seasons", "2015,2016").split(",")]
    out = Path(cfg["_root"]) / "data" / "shotq" / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for s in seasons:
        f = tracked_season(cfg, s)
        if f is None:
            print(f"{s}: no tracked attempts built yet; skipped")
            continue
        games = np.sort(f["game_id"].unique())
        lab = dict(zip(games, np.random.default_rng(11).permutation(len(games)) % K))
        fold = f["game_id"].map(lab).to_numpy()
        q = np.full(len(f), np.nan)
        qf = np.full(len(f), np.nan)
        for k in range(K):
            m = sm.fit(f[fold != k], BLOCKS, offsets=True)
            m.seasons = ()                    # held out by game within the season, by design
            q[fold == k] = sm.probability(m.predict_raw(f[fold == k]))
            qf[fold == k] = sm.probability(m.predict_raw(f[fold == k], with_offsets=True))
        ll = logloss(f["made"].to_numpy(float), q).mean()
        print(f"{s}: {len(f):,} tracked attempts, {len(games)} games; out-of-fold log loss {ll:.4f}")
        parts.append(pd.DataFrame(dict(season=s, game_id=f["game_id"], action_number=f["action_number"],
                                       q_teacher=q, q_teacher_full=qf, fold=fold)))
    L = pd.concat(parts, ignore_index=True)
    L.to_parquet(out / "labels.parquet", index=False)
    print(f"wrote {out / 'labels.parquet'}: {len(L):,} labels")


if __name__ == "__main__":
    main()
