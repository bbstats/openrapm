"""Does teaching the play-by-play model with tracking beat teaching it with makes?  (stage 8, the nested test)

    python scripts/122_tracking_taught.py [--lams=0,0.5,0.8,1] [--weights=1,4] [--max_rows=1500000]

The owner's question, asked as directly as the data allows.  Two tracked seasons exist (2014-15: the shot log;
2015-16: the movement labels), so each is held out in turn:

    train   every other season's attempts 1997-2025 (never 2026), with makes as the label -- and, on the OTHER
            tracked season's attempts, the label (1 - lam) x made + lam x the teacher's prediction for that shooter
            (data/shotq/teacher/labels.parquet, q_teacher_full; out-of-fold within its season).  lam = 0 is the
            plain make model, trained on exactly the same rows.
    score   the held-out tracked season, every arm priced for a league-average shooter (offsets zeroed), level
            from the other half (shotmodel.relevel):
              - gap to the teacher's quality (q_teacher: a league-average shooter, out-of-fold), squared points
              - log loss on the held-out makes
              - the contest left (shottest.contest_left against the official dashboards)

Unlike the ratings models, these fits may see the tracked season next to the one they score: the year-over-year
hygiene is about the ratings, and this test asks only whether tracking-taught training generalises to games
it never saw.  Every arm sees the same seasons, so the comparison is like for like.  Writes
outputs/shottest/tracking_taught.csv.
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
from eracoef import shottest as st  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.shottest import logloss  # noqa: E402
from eracoef.tracking import dashboard_counts  # noqa: E402

FULL = ["spot", "start", "putback", "clock", "context", "fatigue", "prev"]
TRACKED = (2015, 2016)


def table(cfg, s):
    return pd.read_parquet(Path(cfg["_root"]) / "data" / "shotq" / "_table" / f"{s}.parquet")


def main():
    check_flags()
    cfg = load_config()
    lams = [float(x) for x in flag("lams", "0,0.5,0.8,1").split(",")]
    weights = [float(x) for x in flag("weights", "1,4").split(",")]
    max_rows = int(flag("max_rows", "1500000"))
    root = Path(cfg["_root"])
    labels = pd.read_parquet(root / "data" / "shotq" / "teacher" / "labels.parquet")
    seasons = list(range(1997, 2026))
    rng = np.random.default_rng(0)
    rows, per_game = [], []
    for test_s in TRACKED:
        other = [s for s in TRACKED if s != test_s][0]
        train_s = [s for s in seasons if s != test_s]
        per = max_rows // len(train_s)
        parts = []
        lab_other = labels[labels.season == other][["game_id", "action_number", "q_teacher_full"]]
        for s in train_s:
            t = table(cfg, s)
            t = t[~t["heave"]]
            if len(t) > per:
                keep = np.zeros(len(t), dtype=bool)
                keep[rng.choice(len(t), per, replace=False)] = True
                if s == other:
                    keep |= t.set_index(["game_id", "action_number"]).index.isin(
                        lab_other.set_index(["game_id", "action_number"]).index)
                t = t[keep]
            parts.append(t)
        base = pd.concat(parts, ignore_index=True).merge(lab_other, on=["game_id", "action_number"], how="left")
        tracked = base["q_teacher_full"].notna().to_numpy()
        test = table(cfg, test_s)
        test = test[(test.phase == "RS") & ~test.heave].reset_index(drop=True)
        lab_test = labels[labels.season == test_s][["game_id", "action_number", "q_teacher"]]
        y_test = test["made"].to_numpy(float)
        dash = dashboard_counts(cfg, test_s, "def").reset_index()
        tight = dash.get("0-2 Feet - Very Tight", 0) + dash.get("2-4 Feet - Tight", 0)
        D = pd.DataFrame(dict(date=dash["date"], PLAYER_ID=dash["PLAYER_ID"], tight=tight,
                              fga=dash[[c for c in dash.columns if "Feet" in str(c)]].sum(axis=1)))
        meta = pd.read_parquet(root / "data" / "shotframe" / f"{test_s}_RS.parquet", columns=["game_id", "action_number", "game_date"])
        # lam 0 with a weight is the control: the same rows up-weighted, makes as the label (is it tracking, or
        # just leaning on the season next door?)
        arms = [(0.0, 1.0)] + [(lam, w) for lam in lams for w in weights if not (lam == 0 and w == 1.0)]
        for lam, w in arms:
            tr = base.copy()
            y = tr["made"].to_numpy(float)
            y[tracked] = (1.0 - lam) * y[tracked] + lam * tr.loc[tracked, "q_teacher_full"].to_numpy(float)
            tr["made"] = y
            wt = sm.closeness_weights(tr["season"].to_numpy(), [test_s]) * np.where(tracked, w, 1.0)
            model = sm.fit(tr, FULL, weights=wt, offsets=True)
            model.seasons = ()                                  # this test sees every season but the scored one
            eta = model.predict_raw(test)
            q = sm.probability(sm.relevel(eta, test)[0])
            name = f"lam{lam:g}_w{w:g}" if (lam > 0 or w != 1.0) else "makes"
            ev = test.assign(q=q).merge(lab_test, on=["game_id", "action_number"], how="left")
            m = ev["q_teacher"].notna()
            gap = 1e4 * (ev.loc[m, "q"] - ev.loc[m, "q_teacher"]) ** 2
            ll = logloss(y_test, q)
            C = st.contest_left(ev.merge(meta, on=["game_id", "action_number"]).assign(arm_q=q), ["arm_q"], D)
            rows.append(dict(season=test_s, arm=name, lam=lam, weight=w, gap_teacher=float(gap.mean()),
                             logloss=float(ll.mean()), contest_slope=float(C["slope"].iloc[0]),
                             contest_se=float(C["se"].iloc[0])))
            per_game.append(pd.DataFrame(dict(season=test_s, arm=name, game_id=ev["game_id"], ll=ll,
                                              gap=np.where(m, 1e4 * (ev["q"] - ev["q_teacher"]) ** 2, np.nan))))
            print(f"  {test_s} {name:12s} gap to teacher {gap.mean():7.2f}  log loss {ll.mean():.5f}  "
                  f"contest left {C['slope'].iloc[0]:+.4f}", flush=True)
    R = pd.DataFrame(rows)
    G = pd.concat(per_game, ignore_index=True).groupby(["season", "arm", "game_id"])[["ll", "gap"]].mean().reset_index()
    print("\neach arm minus the make model, paired by game within the held-out season:")
    for s in TRACKED:
        g = G[G.season == s].pivot_table(index="game_id", columns="arm", values=["ll", "gap"])
        for a in R[R.season == s].arm:
            if a == "makes":
                continue
            line = f"  {s} {a:12s}"
            for metric in ("gap", "ll"):
                d = (g[(metric, a)] - g[(metric, "makes")]).dropna()
                z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                line += f"  {metric} {d.mean():+.5f} (z {z:+.1f}, {int((d < 0).sum())}/{len(d)} games)"
            print(line)
    out = root / "outputs" / "shottest"
    R.to_csv(out / "tracking_taught.csv", index=False)
    G.to_csv(out / "tracking_taught_games.csv", index=False)


if __name__ == "__main__":
    main()
