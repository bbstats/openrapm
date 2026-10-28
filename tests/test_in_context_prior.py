"""The box-score prior answered in context by the Kumo network (src/openrapm/in_context_prior.py, experiment 118).

The network itself needs a GPU and its downloaded weights, so these tests put a stand-in where it answers
(`_network_answers`): the rows each member is shown, how answers are kept, and the fixed-size batches that keep a
row's answer the same whatever is asked beside it.
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from openrapm import box_prior, in_context_prior
from openrapm.data.config import booster_params, load_config
from openrapm.in_context_prior import InContextPrior, context_draws


@pytest.fixture
def stand_in(monkeypatch):
    """A stand-in network: each row's answer is the mean label of the rows shown, plus the row's first input, so an
    answer depends on what the members were shown; every batch's size is recorded."""
    batches = []

    def answers(size, context_x, context_y, rows, generator_seed):
        batches.append(len(rows))
        return np.full(len(rows), float(context_y.mean())) + rows[:, 0]
    monkeypatch.setattr(in_context_prior, "_network_answers", answers)
    return batches


def test_every_row_is_shown_when_there_are_no_more_than_the_rows_a_member_takes():
    shown = context_draws(np.ones(5), members=3, context_rows=10, draw=0, seed=0)
    assert shown.shape == (3, 5) and (shown == np.arange(5)).all()


def test_heavier_rows_are_shown_more_often_and_each_member_draws_its_own():
    weights = np.r_[np.ones(900), np.full(100, 50.0)]
    shown = context_draws(weights, members=4, context_rows=200, draw=0, seed=0)
    assert shown.shape == (4, 200)
    heavy = (shown >= 900).sum(axis=1)
    assert (heavy >= 90).all()                     # nearly every heavy row, in every member
    assert not (shown[0] == shown[1]).all()        # each member its own light rows
    assert (shown == context_draws(weights, 4, 200, 0, 0)).all()          # the same draw again
    assert not (shown == context_draws(weights, 4, 200, 1, 0)).all()      # another draw, other rows


def test_drawn_equally_every_row_has_the_same_chance_whatever_its_weight():
    weights = np.r_[np.ones(900), np.full(100, 50.0)]
    params = {"learner": in_context_prior.LEARNER, "draw_by": "equal"}
    prior = InContextPrior.from_rows(params, np.zeros((1000, 2)), np.zeros(1000), weights)
    assert (prior.weights == 1).all()
    with pytest.raises(ValueError):
        InContextPrior.from_rows({**params, "draw_by": "possessions"}, np.zeros((3, 2)), np.zeros(3), np.ones(3))


def test_a_row_without_weight_cannot_be_drawn():
    with pytest.raises(ValueError):
        context_draws(np.r_[1.0, 0.0, 2.0], members=1, context_rows=1, draw=0, seed=0)


def test_answers_are_kept_and_asked_in_batches_of_one_size(stand_in):
    rng = np.random.default_rng(0)
    prior = InContextPrior(inputs=rng.normal(size=(50, 3)).astype(np.float32), label=rng.normal(size=50),
                           weights=np.ones(50), members=2, context_rows=20, query_rows=8)
    rows = rng.normal(size=(11, 3))
    first = prior.predict(rows)
    assert stand_in == [8, 8]                      # 11 rows: a full batch and one filled up to 8
    assert (prior.predict(rows[::-1]) == first[::-1]).all() and stand_in == [8, 8]    # kept, not asked again
    prior.predict(np.vstack([rows[:2], rng.normal(size=(1, 3))]))
    assert stand_in == [8, 8, 8]                   # only the new row was asked


def test_a_sent_or_saved_prior_starts_without_kept_answers(stand_in):
    prior = InContextPrior(inputs=np.zeros((4, 2), np.float32), label=np.arange(4.0), weights=np.ones(4))
    prior.predict(np.ones((3, 2)))
    copy = pickle.loads(pickle.dumps(prior))
    assert prior.answers and not copy.answers
    assert copy.predict(np.ones((1, 2)))[0] == prior.predict(np.ones((1, 2)))[0]


def test_the_folds_keep_a_players_own_rows_from_the_rows_his_prior_is_shown(stand_in):
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)},
                         index=pd.Index(np.repeat(np.arange(20), 3), name="player_id"))
    train = train.assign(label=np.repeat(rng.normal(size=20), 3), row_weight=rng.uniform(1, 5, size=60))
    params = {"learner": in_context_prior.LEARNER, "members": 1, "context_rows": 1000, "query_rows": 16}
    prior = box_prior.fit_prior(train, ["a", "b"], params, n_folds=5, bags=0)
    assert isinstance(prior.full, InContextPrior)
    assert all(isinstance(model, InContextPrior) for model in prior.fold_models)
    for model, held_out in zip(prior.fold_models, prior.held_out_players):
        kept = train[~train.index.isin(held_out)]
        assert len(model.label) == len(kept) and np.allclose(np.sort(model.label), np.sort(kept.label))


def test_a_params_file_naming_the_kumo_learner_is_read_with_its_learner_and_lightgbms_as_it_is():
    cfg = load_config()
    for size in in_context_prior.SIZES:
        params = booster_params(cfg, f"params/booster_params_kumo_{size}.json")
        assert params["O"]["learner"] == in_context_prior.LEARNER and params["O"]["size"] == size
    shipped = booster_params(cfg, "params/booster_params_lgb1_tune2.json")
    assert "learner" not in shipped["O"] and "learner" not in shipped["D"]
