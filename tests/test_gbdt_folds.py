"""Player-grouped cross-fitting of the boosted prior (GBDTPrior folds).

With folds on, a player is scored by a model that never saw a row of his.  The properties that make that
true: folds off is the old prior exactly; a fold's model has none of that fold's players in its rows; each
player is routed to the model of his own fold; a player the panel does not know goes to the all-rows model;
and a pure identifier -- the raw player id as a feature -- is worth nothing once his rows are out (the
positive control of scratch/foldtest.py, here as an assertion).
"""
import numpy as np
import pandas as pd

from eracoef.design import FEATURES
from eracoef.gbdt_prior import GBDTPrior
from test_gbdt_prior import CFG, WINDOWS, _panel

FEATS = [*FEATURES, "season"]


def _X(p, side, window):
    d = p[(p.side == side) & (p.window == window)].sort_values("player_id")
    return d[FEATS].astype(float).reset_index(drop=True), d.player_id.to_numpy()


def test_folds_off_is_the_old_prior_exactly():
    p = _panel()
    a = GBDTPrior(p, CFG, mode="full", features=FEATS)
    b = GBDTPrior(p, CFG, mode="full", features=FEATS, folds=0)
    X, ids = _X(p, "O", "W1")
    assert np.array_equal(a.predict("O", X, {"W1"}), b.predict("O", X, {"W1"}, player_ids=ids))


def test_a_folds_model_has_none_of_its_players_and_routing_matches_the_fold():
    p = _panel()
    g = GBDTPrior(p, CFG, mode="full", features=FEATS, folds=5)
    X, ids = _X(p, "O", "W1")
    pred = g.predict("O", X, {"W1"}, player_ids=ids)
    fold = g._fold_of.reindex(ids).to_numpy()
    for k in range(5):
        m, rep = g.model("O", {"W1"}, fold=k)
        rows = g.rows("O", {"W1"})
        kept = rows[rows.player_id.map(g._fold_of).to_numpy() != k]
        assert rep["n_rows"] == len(kept)                                    # the model trained on exactly those
        assert not set(kept.player_id) & set(g._fold_of.index[g._fold_of == k])   # none of the fold's players
        sel = fold == k
        assert np.allclose(pred[sel], m.predict(X.to_numpy(float)[sel]))     # and the fold's players used it
    assert len(g._models) == 5                                               # one model per fold, no extra


def test_an_unknown_player_goes_to_the_all_rows_model():
    p = _panel()
    g = GBDTPrior(p, CFG, mode="full", features=FEATS, folds=4)
    X, ids = _X(p, "O", "W2")
    ids = ids.copy()
    ids[0] = 10_000_000                                                       # nobody the panel has heard of
    pred = g.predict("O", X, {"W2"}, player_ids=ids)
    m, _ = g.model("O", {"W2"})                                               # fold=None: every row
    assert np.isclose(pred[0], m.predict(X.to_numpy(float)[:1])[0])


def test_a_raw_player_id_is_worth_nothing_once_his_rows_are_out():
    """The positive control: with the player's rows in training, his id lets the model read his target off
    his other rows; with them out it is a random number.  Compares held-out MSE with and without `pid` under
    each regime."""
    p = _panel(seed=3, n_players=300)
    p["pid"] = p.player_id.astype(float)
    tcol = "rapm1"

    def mse(features, folds):
        g = GBDTPrior(p, CFG, mode="full", features=features, folds=folds)
        err, wt = [], []
        for lab in WINDOWS:
            held = g.rows("O", ())                                   # the shipped target for the scored rows
            held = held[held.window == lab]
            X = held[features].astype(float).reset_index(drop=True)
            pr = g.predict("O", X, {lab}, player_ids=held.player_id.to_numpy())
            err.append((held.target.to_numpy() - pr) ** 2)
            wt.append(held.weight.to_numpy())
        return float(np.average(np.concatenate(err), weights=np.concatenate(wt)))

    base_in, id_in = mse(FEATS, 0), mse([*FEATS, "pid"], 0)
    base_out, id_out = mse(FEATS, 5), mse([*FEATS, "pid"], 5)
    gain_in, gain_out = id_in - base_in, id_out - base_out
    assert gain_in < -0.02                                   # the id helps when his rows are there
    # ... and buys nothing once they are not: an unseen id is a random number, so it may even cost a little
    # (the booster spends splits on noise), but the benefit itself must be gone
    assert gain_out > 0.25 * gain_in
