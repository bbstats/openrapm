"""The stacked prior (`stackprior.StackedSPM`): both halves used, the blend fitted out of player fold."""
import numpy as np
import pandas as pd
import pytest

from eracoef.stackprior import StackedSPM, weighted_standardise


def _rows(n_players=300, rows_each=3, seed=0):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n_players), rows_each)
    onc_o = rng.normal(size=pid.size)
    box = rng.normal(size=pid.size)
    target = 0.8 * onc_o + np.sin(2 * box) + rng.normal(scale=0.3, size=pid.size)
    return pd.DataFrame({"onc_o": onc_o, "box": box, "target": target,
                         "row_weight": rng.uniform(100, 3000, pid.size)}, index=pd.Index(pid, name="player_id"))


def test_weighted_standardise_matches_numpy():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    mean, sd = weighted_standardise(X, np.array([1.0, 1.0, 2.0]))
    assert mean == pytest.approx([3.5, 5.0])
    assert sd[0] == pytest.approx(np.sqrt((2.5 ** 2 + 0.5 ** 2 + 2 * 1.5 ** 2) / 4))
    assert sd[1] == 1.0                                   # a constant column is left unscaled


def test_stack_uses_both_halves_and_beats_each():
    train = _rows()
    m = StackedSPM({"depth": 3}, n_folds=5, linear=["onc_o"], quality=None).fit(train, ["onc_o", "box"])
    assert m.lin_names_ == ["onc_o"] and m.boost_names_ == ["box"]
    assert (m.blend_.coef_ >= 0).all() and (m.blend_.coef_ > 0.1).all()
    assert m.oof_rmse_["stack"] < min(m.oof_rmse_["linear"], m.oof_rmse_["booster"])


def test_each_player_gets_the_fit_that_never_saw_him():
    train = _rows()
    feats = ["onc_o", "box"]
    m = StackedSPM({"depth": 3}, n_folds=5, linear=["onc_o"], quality=None).fit(train, feats)
    frame = train.reset_index().iloc[:30]
    got = m.predict(frame, feats)
    X_lin, X_boost = frame[["onc_o"]].to_numpy(float), frame[["box"]].to_numpy(float)
    for i, p in enumerate(frame.player_id):
        f = next(k for k, ex in enumerate(m.excluded_) if p in ex)
        want = m._predict_pair(m.fold_models_[f], X_lin[i:i + 1], X_boost[i:i + 1])[0]
        assert got[i] == pytest.approx(want)
    unseen = frame.assign(player_id=-1)                   # nobody's rows: the full fit
    assert m.predict(unseen, feats) == pytest.approx(m._predict_pair(m.full_, X_lin, X_boost))


def test_needs_folds_and_both_halves():
    with pytest.raises(ValueError):
        StackedSPM({}, n_folds=1)
    with pytest.raises(ValueError):
        StackedSPM({}, n_folds=5, linear=["onc_o"], quality=None).fit(_rows(50), ["onc_o"])
