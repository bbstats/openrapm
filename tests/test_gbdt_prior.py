"""The boosted box prior (src/eracoef/gbdt_prior.py): pooled-other-window targets, leave-window-out that also
keeps the excluded window out of every target, the drag counterbalance, the offset, and the Boruta shim."""
import numpy as np
import pandas as pd
import pytest

from eracoef.design import FEATURES
from eracoef.gbdt_prior import (DEFAULT_FEATURES, GBDTPrior, counterbalance, drag, gbdt_offset, reference_mean,
                                training_rows)

WINDOWS = ["W0", "W1", "W2", "W3"]
CFG = {"gbdt": {"drag_tol": 0.02, "seed": 0, "thread_count": 2}}


def _panel(seed=0, n_players=240):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_players):
        wins = rng.choice(4, size=rng.integers(2, 5), replace=False)
        skill = rng.normal(size=len(FEATURES))                          # a player's rate profile persists across windows
        base_share = float(rng.uniform(0.05, 0.9))                       # and so does his role
        for wi in wins:
            r = skill + 0.3 * rng.normal(size=len(FEATURES))
            season = 2000 + 3 * int(wi) + int(rng.integers(0, 3))
            base = 1.5 * r[0] - 1.0 * r[9] + 0.5 * r[8] + 0.02 * (season - 2004)
            poss_pct = float(np.clip(base_share + 0.05 * rng.normal(), 0.02, 0.9))
            poss = poss_pct * 15000.0
            gs, age = float(rng.uniform(0, 1)), float(rng.uniform(19, 38))
            role = 3.0 * poss_pct - 2.0                                     # the role prior's part of RAPM_1
            for side, sign in (("O", 1.0), ("D", -1.0)):
                u = sign * base + rng.normal(0, 0.5)
                rows.append(dict(window=WINDOWS[wi], side=side, player_id=pid, poss=poss, season=season,
                                 **dict(zip(FEATURES, r)), poss_pct=poss_pct, gs_pct=gs, age=age,
                                 spm=sign * role, u=u, rapm1=sign * role + u))
    return pd.DataFrame(rows)


def test_pooled_target_by_hand():
    p = pd.DataFrame([dict(window="W0", side="O", player_id=1, poss=100.0, season=2000, rapm1=1.0),
                      dict(window="W1", side="O", player_id=1, poss=300.0, season=2003, rapm1=3.0),
                      dict(window="W0", side="O", player_id=2, poss=100.0, season=2000, rapm1=5.0)])
    for f in FEATURES:
        p[f] = 0.0
    r = training_rows(p, "O", target_col="rapm1").set_index(["player_id", "window"])
    assert r.loc[(1, "W0"), "target"] == 3.0 and r.loc[(1, "W0"), "weight"] == 300.0
    assert r.loc[(1, "W1"), "target"] == 1.0 and r.loc[(1, "W1"), "weight"] == 100.0
    assert (2, "W0") not in r.index                                     # one window only: scored, not trained
    assert len(training_rows(p, "O", exclude={"W1"}, target_col="rapm1")) == 0   # player 1 has nothing left to pool


def test_pair_rows_pool_back_to_the_training_rows_and_carry_turnover():
    from eracoef.gbdt_prior import pair_rows
    p = _panel()
    wins = sorted(p.window.unique())
    pairs = [(a, b) for a in wins for b in wins if a != b]
    rng = np.random.default_rng(2)
    turn = pd.DataFrame([dict(player_id=pid, window=a, window_to=b, turnover=float(rng.uniform()))
                         for pid in p.player_id.unique() for a, b in pairs])
    # (the pooled rows measure window distance among the windows LEFT after the exclusion, so with a decay and
    # an excluded middle window the two disagree by design; the pair rows keep the calendar distance)
    for decay, exclude in ((1.0, {"W2"}), (0.5, set())):
        pr = pair_rows(p, "O", exclude=exclude, target_col="rapm1", win_decay=decay, turn=turn)
        tr = training_rows(p, "O", exclude=exclude, target_col="rapm1", win_decay=decay)
        assert not (set(pr.window) | set(pr.window_to)) & exclude
        assert "turn" in pr.columns and pr.turn.between(0, 1).all()
        # the weighted mean of a player-window's pair targets is the pooled target, and the weights add up
        pr["_wt"] = pr.weight * pr.target
        g = pr.groupby(["player_id", "window"]).agg(w=("weight", "sum"), wt=("_wt", "sum"))
        g["target"] = g.wt / g.w
        j = tr.set_index(["player_id", "window"]).join(g, rsuffix="_pair", how="inner")
        assert len(j) == len(tr)
        assert np.allclose(j.target, j.target_pair) and np.allclose(j.weight, j.w)
    # a pair the turnover table does not cover is dropped; without a table there is no turn column
    short = turn[~((turn.player_id == 0) & (turn.window == "W0"))]
    pr2 = pair_rows(p, "O", target_col="rapm1", turn=short)
    assert not ((pr2.player_id == 0) & (pr2.window == "W0")).any()
    assert "turn" not in pair_rows(p, "O", target_col="rapm1").columns


def test_turn_prior_trains_on_pairs_and_predicts_with_the_turn_column():
    p = _panel()
    wins = sorted(p.window.unique())
    rng = np.random.default_rng(1)
    turn = pd.DataFrame([dict(player_id=pid, window=a, window_to=b, turnover=float(rng.uniform()))
                         for pid in p.player_id.unique() for a in wins for b in wins if a != b])
    cfg = {"gbdt": {**CFG["gbdt"], "params": {"n_estimators": 30}}}
    prior = GBDTPrior(p, cfg, mode="full", target_col="rapm1", turn=turn)
    assert prior.features["O"][-1] == "turn" and prior.features["D"][-1] == "turn"
    X = p[(p.side == "O") & (p.window == "W3")][prior.features["O"][:-1]].copy()
    lo, hi = prior.predict("O", X.assign(turn=0.35), exclude={"W3"}), prior.predict("O", X.assign(turn=1.0), exclude={"W3"})
    assert lo.shape == hi.shape == (len(X),) and np.isfinite(lo).all() and np.isfinite(hi).all()
    assert prior.reports and prior.reports[0]["n_rows"] > len(X)      # pair rows, more than one per player-window
    plain = GBDTPrior(p, cfg, mode="full", target_col="rapm1", pairs=True)     # pair rows, no turnover feature
    assert "turn" not in plain.features["O"]
    assert np.isfinite(plain.predict("O", X, exclude={"W3"})).all()


def test_excluded_window_never_reaches_rows_or_targets():
    p = _panel()
    poisoned = p.copy()
    poisoned.loc[poisoned.window == "W3", ["u", "rapm1"]] = 1e6
    rows = training_rows(poisoned, "O", exclude={"W3"})
    assert (rows.window != "W3").all() and rows.target.abs().max() < 1e3
    cfg = {"gbdt": {**CFG["gbdt"], "drag_tol": 1e12}}                   # no counterbalance, so the models must agree exactly
    a, b = GBDTPrior(p, cfg), GBDTPrior(poisoned, cfg)
    X = p[(p.side == "O") & (p.window == "W3")][DEFAULT_FEATURES]
    assert np.allclose(a.predict("O", X, exclude={"W3"}), b.predict("O", X, exclude={"W3"}))
    assert a.reports[-1]["exclude"] == "W3" and a.reports[-1]["n_rows"] == len(rows)


def test_counterbalance_restores_the_mean():
    rng = np.random.default_rng(1)
    rows = pd.DataFrame({"target": rng.normal(1.0, 2.0, 500), "weight": rng.uniform(1, 10, 500)})
    full = float(np.average(rows.target, weights=rows.weight)) - 0.8    # the "full" mean sits below the training mean
    assert drag(rows, full) > 0.02
    out, rep = counterbalance(rows, full, tol=0.02)
    assert rep["applied"] and 0.0 <= rep["factor"] < 1.0 and abs(rep["drag_after"]) < 1e-9
    assert abs(drag(out, full)) < 1e-9 and (out.weight <= rows.weight + 1e-12).all()
    same, rep2 = counterbalance(rows, float(np.average(rows.target, weights=rows.weight)) + 0.01, tol=0.02)
    assert not rep2["applied"] and same is rows


def test_prior_learns_the_signal_and_offset_is_centred():
    p = _panel()
    prior = GBDTPrior(p, CFG)
    held = p[(p.side == "O") & (p.window == "W2")]
    g = prior.predict("O", held[DEFAULT_FEATURES], exclude={"W2"})
    truth = training_rows(p, "O")                                       # the pooled target the model is asked for
    t = held.merge(truth[["player_id", "window", "target"]], on=["player_id", "window"], how="left")
    ok = t.target.notna().to_numpy()
    assert np.corrcoef(g[ok], t.target.to_numpy()[ok])[0, 1] > 0.6
    m = len(held)
    ro = held[FEATURES].to_numpy()
    off = gbdt_offset(prior, ro, ro, held.season.to_numpy(), held.poss.to_numpy(), held.poss.to_numpy(),
                      exclude={"W2"}, sides=("O",))
    assert off.shape == (2 * m,) and np.all(off[m:] == 0.0)
    assert abs(np.average(off[:m], weights=held.poss)) < 1e-9
    assert abs(reference_mean(p, "O")) < 1.0


def test_boruta_shim_accepts_informative_features():
    try:
        from eracoef.gbdt_prior import run_boruta
        from eracoef.gbdt_prior import _import_borutashap
        _import_borutashap()
    except ImportError:
        pytest.skip("BorutaShap not installed")
    p = _panel(seed=2, n_players=300)
    rows = training_rows(p, "O")
    res = run_boruta(rows, DEFAULT_FEATURES, n_trials=20, seed=0, thread_count=4)   # the Bonferroni-corrected binomial test needs ~20 trials to decide
    assert set(res) >= {"accepted", "tentative", "rejected"}
    assert "fg3m" in res["accepted"] or "tov" in res["accepted"]
    assert len(res["rejected"]) > 0


def test_full_mode_uses_role_inputs():
    from eracoef.gbdt_prior import FULL_FEATURES
    p = _panel()
    prior = GBDTPrior(p, CFG, mode="full")
    assert prior.target_col == "rapm1" and prior.features["O"] == FULL_FEATURES
    held = p[(p.side == "O") & (p.window == "W1")]
    m = len(held)
    ro = held[FEATURES].to_numpy()
    off = gbdt_offset(prior, ro, ro, held.season.to_numpy(), held.poss.to_numpy(), held.poss.to_numpy(),
                      exclude={"W1"}, sides=("O", "D"), extra=held[["poss_pct", "gs_pct", "age"]].reset_index(drop=True))
    assert off.shape == (2 * m,) and abs(np.average(off[:m], weights=held.poss)) < 1e-9
    assert np.corrcoef(off[:m], held.poss_pct)[0, 1] > 0.3                    # the role level is in the prior now
