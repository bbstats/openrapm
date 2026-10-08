"""The LightGBM prior (`eracoef.lgbprior`) and the leave-one-player-out mode of scripts/62's `OutOfPlayerSPM`."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eracoef import lgbprior

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"n_estimators": 60, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 5}
FEATS = ["a", "b"]


def _rows(n_players=120, rows_each=2, seed=0):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n_players), rows_each)
    a, b = rng.normal(size=pid.size), rng.normal(size=pid.size)
    target = 0.8 * a + np.sin(2 * b) + rng.normal(scale=0.3, size=pid.size)
    return pd.DataFrame({"a": a, "b": b, "target": target, "weight": rng.uniform(100, 3000, pid.size)},
                        index=pd.Index(pid, name="player_id"))


def _arrays(train):
    return (train[FEATS].to_numpy(float), train.target.to_numpy(float), train.weight.to_numpy(float),
            train.index.to_numpy())


def test_the_fit_is_deterministic_and_blind_to_the_weight_scale():
    X, y, w, _ = _arrays(_rows())
    one = lgbprior.fit(PARAMS, X, y, w).predict(X)
    assert np.array_equal(one, lgbprior.fit(PARAMS, X, y, w).predict(X))
    assert np.array_equal(one, lgbprior.fit(PARAMS, X, y, 1000.0 * w).predict(X))
    # subsampling asked for is overridden: the settings that would make it random never reach LightGBM
    noisy = {**PARAMS, "subsample": 0.5, "subsample_freq": 1, "colsample_bytree": 0.5, "random_state": 7}
    assert np.array_equal(one, lgbprior.fit(noisy, X, y, w).predict(X))


def test_each_player_gets_a_model_fitted_without_his_rows():
    train = _rows()
    X, y, w, players = _arrays(train)
    needed = [3, 50, 999]                                   # 999 has no rows: no model
    models = lgbprior.fit_leave_one_out(PARAMS, X, y, w, players, needed)
    assert sorted(models) == [3, 50]
    for p in (3, 50):
        keep = players != p
        direct = lgbprior.fit(PARAMS, X[keep], y[keep], w[keep])
        assert np.array_equal(models[p].predict(X), direct.predict(X))
    parallel = lgbprior.fit_leave_one_out(PARAMS, X, y, w, players, needed, n_jobs=2)
    assert all(np.array_equal(parallel[p].predict(X), models[p].predict(X)) for p in models)


@pytest.fixture(scope="module")
def board():
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_board_lgb", ROOT / "scripts" / "62_single_year_board.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_query_rows_are_answered_by_the_model_without_their_player():
    train = _rows()
    X, y, w, players = _arrays(train)
    rng = np.random.default_rng(1)
    q_players = np.array([3, 3, 50, 999, 50, 3])          # 999 has no training rows
    Xq = rng.normal(size=(q_players.size, 2))
    got = lgbprior.predict_leave_one_out(PARAMS, X, y, w, players, Xq, q_players)
    assert np.isnan(got[3]) and not np.isnan(np.delete(got, 3)).any()
    for p in (3, 50):
        keep = players != p
        direct = lgbprior.fit(PARAMS, X[keep], y[keep], w[keep]).predict(Xq[q_players == p])
        assert np.array_equal(got[q_players == p], direct)
    parallel = lgbprior.predict_leave_one_out(PARAMS, X, y, w, players, Xq, q_players, n_jobs=2)
    assert np.array_equal(parallel, got, equal_nan=True)


def test_the_leave_one_out_prior_answers_its_query_rows(board):
    train = _rows()
    query = train.reset_index().drop_duplicates("player_id").iloc[:12].copy()
    query.loc[query.index[-2:], "player_id"] = [-1, -2]        # two players with no training rows
    model = board.OutOfPlayerSPM(PARAMS, 5, learner="lightgbm", lopo=True).fit(train, FEATS, query=query)
    X, y, w, players = _arrays(train)
    Xq = query[FEATS].to_numpy(float)
    for i, p in enumerate(query.player_id):
        if p < 0:
            want = model.full_.predict(Xq[i:i + 1])[0]
        else:
            keep = players != p
            want = lgbprior.fit(PARAMS, X[keep], y[keep], w[keep]).predict(Xq[i:i + 1])[0]
        assert model.query_pred_[i] == want
    assert model.n_own_ == 10 and not model.fold_models_
    with pytest.raises(RuntimeError):
        model.predict(query, FEATS)                          # it keeps no models to ask


def test_the_lightgbm_learner_with_folds_is_out_of_player(board):
    train = _rows()
    model = board.OutOfPlayerSPM(PARAMS, 5, learner="lightgbm").fit(train, FEATS)
    assert len(model.fold_models_) == 5
    frame = train.reset_index().drop_duplicates("player_id").iloc[:10]
    got = model.predict(frame, FEATS)
    X = frame[FEATS].to_numpy(float)
    for i, p in enumerate(frame.player_id):
        f = next(k for k, ex in enumerate(model.excluded_) if p in ex)
        assert got[i] == pytest.approx(model.fold_models_[f].predict(X[i:i + 1])[0])
    with pytest.raises(AssertionError):
        board.OutOfPlayerSPM(PARAMS, 5, lopo=True)          # chimeraboost leave-one-out is refused
