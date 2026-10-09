"""The out-of-player prior averaged over several groupings of the player folds (`62 --prior_groupings=K`)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))          # 62 imports `_cli` from beside itself
_spec = importlib.util.spec_from_file_location("_board_groupings", ROOT / "scripts" / "62_single_year_board.py")
_board = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_board)
OutOfPlayerSPM = _board.OutOfPlayerSPM
PARAMS = {"depth": 3}
FEATS = ["a", "b"]


def _rows(n_players=200, rows_each=2, seed=0):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n_players), rows_each)
    a, b = rng.normal(size=pid.size), rng.normal(size=pid.size)
    target = 0.8 * a + np.sin(2 * b) + rng.normal(scale=0.3, size=pid.size)
    return pd.DataFrame({"a": a, "b": b, "target": target, "row_weight": rng.uniform(100, 3000, pid.size)},
                        index=pd.Index(pid, name="player_id"))


@pytest.fixture(scope="module")
def fitted():
    train = _rows()
    one = OutOfPlayerSPM(PARAMS, 5).fit(train, FEATS)
    three = OutOfPlayerSPM(PARAMS, 5, groupings=3).fit(train, FEATS)
    frame = train.reset_index().drop_duplicates("player_id").iloc[:40]
    return train, one, three, frame


def _predict_before_the_option(model, frame):
    """`OutOfPlayerSPM.predict` as it was before `--prior_groupings`: the held-out fold's fit overwrites the full."""
    X = frame[FEATS].to_numpy(float)
    out = model.full_.predict(X)
    ids = frame.player_id.to_numpy()
    for fold, excluded in zip(model.fold_models_, model.excluded_):
        idx = np.fromiter((p in excluded for p in ids), dtype=bool, count=len(ids))
        if idx.any():
            out[idx] = fold.predict(X[idx])
    return out


def test_one_grouping_is_the_shipped_fit(fitted):
    """K = 1: each player's prior is exactly the one fold fit that held him out, as before the option."""
    train, one, _, frame = fitted
    assert len(one.fold_models_) == 5
    with_unseen = pd.concat([frame, frame.iloc[:5].assign(player_id=-1)], ignore_index=True)
    assert np.array_equal(one.predict(with_unseen, FEATS), _predict_before_the_option(one, with_unseen))
    X = frame[FEATS].to_numpy(float)
    got = one.predict(frame, FEATS)
    for i, p in enumerate(frame.player_id):
        f = next(k for k, ex in enumerate(one.excluded_) if p in ex)
        assert got[i] == pytest.approx(one.fold_models_[f].predict(X[i:i + 1])[0])


def test_the_prior_is_the_mean_of_one_out_of_player_fit_per_grouping(fitted):
    train, one, three, frame = fitted
    each = three.predict_groupings(frame, FEATS)
    assert each.shape == (len(frame), 3) and not np.isnan(each).any()
    assert len(three.fold_models_) == 15
    assert np.allclose(three.predict(frame, FEATS), each.mean(axis=1))
    # every column comes from a fit that never saw the player, and each grouping holds him out exactly once
    for p in frame.player_id:
        holders = [three.fold_grouping_[k] for k, ex in enumerate(three.excluded_) if p in ex]
        assert sorted(holders) == [0, 1, 2]
    # grouping 0 is the shipped deal: the same boosters as the single-grouping fit
    assert np.array_equal(each[:, 0], one.predict(frame, FEATS))
    # the other groupings are different, equally balanced deals
    assert not np.allclose(each[:, 1], each[:, 0])
    assert all(np.abs(s).max() < 0.2 for s in three.shifts_)


def test_a_player_with_no_training_rows_gets_the_full_fit(fitted):
    _, _, three, frame = fitted
    unseen = frame.assign(player_id=-1)
    assert np.array_equal(three.predict(unseen, FEATS), three.full_.predict(unseen[FEATS].to_numpy(float)))
    assert np.isnan(three.predict_groupings(unseen, FEATS)).all()


def test_grouping_first_deals_a_disjoint_set():
    train = _rows(120)
    later = OutOfPlayerSPM(PARAMS, 5, groupings=2, grouping_first=5).fit(train, FEATS)
    assert sorted(set(later.fold_grouping_)) == [0, 1]                       # columns, not seeds
    snake = OutOfPlayerSPM(PARAMS, 5).fit(train, FEATS)
    frame = train.reset_index().drop_duplicates("player_id").iloc[:20]
    assert not np.allclose(later.predict_groupings(frame, FEATS)[:, 0], snake.predict(frame, FEATS))
