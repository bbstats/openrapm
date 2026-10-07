"""How much of true (tracking) shot quality can play-by-play features learn, and can makes teach it as well?

    python scripts/126_quality_ceiling.py

The owner, 2026-10-06, on the nested test: "my idea that the features could actually learn what true shot quality
is for the seasons in which we have good tracking data is false? I feel like that is not highly likely."  Each
tracked season held out in turn: the gap to the tracking quality of the current model, of additive and boosted-tree
models taught by tracking labels or by makes, and the ceiling (boosted trees on the same season's tracking quality,
held out by game).  Writes outputs/shottest/ceiling.csv.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd
sys.path.insert(0, r"A:/code/spmm/src")
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from eracoef import shotmodel as sm
from eracoef.config import load_config

cfg = load_config()
root = r"A:/code/spmm/data/shotq"
L = pd.read_parquet(root + "/teacher/labels.parquet")
FULL = ["spot", "start", "putback", "clock", "context", "fatigue", "prev"]
START = {k: i for i, k in enumerate(sm.START_TYPES)}
RESET = {k: i for i, k in enumerate(list(sm.START_TYPES) + list(sm.RESET_KINDS))}

def feats(t):
    X = pd.DataFrame({
        "value": t.value, "dist": t.dist_xy.clip(0, 40), "angle": t.angle, "corner": t.corner3.astype(int), "noloc": t.noloc.astype(int),
        "start": t.poss_start.map(START).fillna(-1), "tposs": t.secs_into_poss.clip(0, 40),
        "oreb": t.secs_since_oreb.fillna(99).clip(0, 99), "sc": t.sc_eff.clip(0, 24), "clock_off": t.clock_off.astype(int),
        "reset": t.reset_kind.map(RESET).fillna(-1), "period": t.period.clip(1, 5), "left": t.clock.clip(0, 720),
        "margin": t.margin.clip(-30, 30), "on": t.shooter_secs_on.fillna(300).clip(0, 1500),
        "pv": t.start_prev_value, "pm": t.start_prev_made, "pb": t.start_prev_blocked, "pd": t.start_prev_dist.fillna(-1)})
    return X

data = {}
for s in (2015, 2016):
    t = pd.read_parquet(f"{root}/_table/{s}.parquet")
    t = t[(t.phase == "RS") & ~t.heave].reset_index(drop=True)
    t = t.merge(L[["game_id", "action_number", "q_teacher", "q_teacher_full"]], on=["game_id", "action_number"], how="left")
    data[s] = t

def gap(p, q):
    return float(1e4 * np.mean((p - q) ** 2))

rows = []
for test_s, train_s in ((2015, 2016), (2016, 2015)):
    te = data[test_s][data[test_s].q_teacher.notna()].reset_index(drop=True)
    tr_all = data[train_s]
    tr = tr_all[tr_all.q_teacher.notna()].reset_index(drop=True)
    q = te.q_teacher.to_numpy()
    var = gap(q, q.mean())
    # the explained share needs a level: within each 1-ft band the mean, as every model gets from relevel
    band = sm.band_of(te)
    var_band = gap(q, pd.Series(q).groupby(band).transform("mean").to_numpy())
    rows.append(dict(test=test_s, arm="spread of true quality (no model)", gap=var))
    rows.append(dict(test=test_s, arm="distance bands only (each band's mean)", gap=var_band))
    # the shipped ratings-model version (6M makes, additive): from the nested run, makes-trained, other seasons
    nested = pd.read_csv(r"A:/code/spmm/outputs/shottest/tracking_taught.csv")
    rows.append(dict(test=test_s, arm="additive model, 6M makes of every other season", gap=float(nested[(nested.season == test_s) & (nested.arm == "makes")].gap_teacher.iloc[0])))
    # small data: the other tracked season only, additive model, makes vs tracking labels
    m_makes = sm.fit(tr, FULL, offsets=True); m_makes.seasons = ()
    p = sm.probability(sm.relevel(m_makes.predict_raw(te), te)[0])
    rows.append(dict(test=test_s, arm=f"additive model, makes of {train_s} only", gap=gap(p, q)))
    tr2 = tr.assign(made=tr.q_teacher_full)
    m_trk = sm.fit(tr2, FULL, offsets=True); m_trk.seasons = ()
    p = sm.probability(sm.relevel(m_trk.predict_raw(te), te)[0])
    rows.append(dict(test=test_s, arm=f"additive model, tracking labels of {train_s} only", gap=gap(p, q)))
    # flexible model (boosted trees) on the same play-by-play inputs
    Xtr, Xte = feats(tr), feats(te)
    gb = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
                                       categorical_features=[Xtr.columns.get_loc(c) for c in ("start", "reset", "value")], random_state=0)
    gb.fit(Xtr, tr.q_teacher)
    rows.append(dict(test=test_s, arm=f"boosted trees, tracking quality of {train_s}", gap=gap(np.clip(gb.predict(Xte), 0.01, 0.99), q)))
    gc = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
                                        categorical_features=[Xtr.columns.get_loc(c) for c in ("start", "reset", "value")], random_state=0)
    gc.fit(Xtr, tr.made)
    rows.append(dict(test=test_s, arm=f"boosted trees, makes of {train_s}", gap=gap(gc.predict_proba(Xte)[:, 1], q)))
    # boosted trees on makes of MANY seasons (every season but the two tracked ones and their neighbours' test)
    big = pd.concat([pd.read_parquet(f"{root}/_table/{s}.parquet").query("phase == 'RS' and not heave").sample(60000, random_state=s)
                     for s in range(1997, 2026) if s not in (test_s,)], ignore_index=True)
    gc2 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=400, l2_regularization=1.0,
                                         categorical_features=[Xtr.columns.get_loc(c) for c in ("start", "reset", "value")], random_state=0)
    gc2.fit(feats(big), big.made)
    rows.append(dict(test=test_s, arm="boosted trees, 1.7M makes of every other season", gap=gap(gc2.predict_proba(Xte)[:, 1], q)))
    # the ceiling: boosted trees on tracking quality, held out by game WITHIN the test season (5 folds)
    games = te.game_id.unique(); fold = te.game_id.map(dict(zip(games, np.arange(len(games)) % 5))).to_numpy()
    p = np.zeros(len(te))
    for k in range(5):
        g = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
                                          categorical_features=[Xte.columns.get_loc(c) for c in ("start", "reset", "value")], random_state=0)
        g.fit(Xte[fold != k], te.q_teacher[fold != k]); p[fold == k] = g.predict(Xte[fold == k])
    rows.append(dict(test=test_s, arm="ceiling: boosted trees, tracking quality of the same season (other games)", gap=gap(np.clip(p, 0.01, 0.99), q)))
    print(f"done {test_s}", flush=True)
R = pd.DataFrame(rows)
for s in (2015, 2016):
    r = R[R.test == s].copy()
    tot = r.gap.iloc[0]
    r["share_of_true_quality_captured"] = (1 - r.gap / tot).round(3)
    print(f"\n=== held-out season {s}: mean squared gap to the tracking model's quality (squared points of make probability)")
    print(r[["arm", "gap", "share_of_true_quality_captured"]].round(2).to_string(index=False))
R.to_csv(r"A:/code/spmm/outputs/shottest/ceiling.csv", index=False)
