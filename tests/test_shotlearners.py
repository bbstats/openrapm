"""The search's learner plug-ins (eracoef.shotlearners): boosted trees on top of the regression's margin learn the
shot and not who takes it, each library's trees-only prediction is the trees alone, chimeraboost's packed label
reproduces its own log loss, and the registry, spaces and penalty scaling say what the driver expects."""
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from eracoef import shotfeatures
from eracoef import shotlearners as sl
from eracoef import shotmodel as sm

SEASONS = (2001, 2002, 2003, 2004, 2005)
PARAMS = {"xgb": dict(eta=0.1, max_depth=3), "lgbm": dict(learning_rate=0.1, num_leaves=7, min_data_in_leaf=200),
          "cat": dict(learning_rate=0.1, depth=3, task_type="CPU"), "chimera": dict(learning_rate=0.1, depth=3)}


def _frame(seed=0, seasons=SEASONS, n=10000, n_shooters=40):
    """A planted league.  Good shooters take the late-clock shots, but the late clock itself changes nothing; a
    steal start adds 0.5 to the logit whoever shoots.  The regression sees only the spot, so the trees must find
    the steal, and must NOT read the late clock as easy."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        value = np.where(rng.random(n) < 0.3, 3, 2)
        d = np.where(value == 3, rng.uniform(23.0, 27.0, n), rng.uniform(0.0, 20.0, n))
        shooter = rng.integers(0, n_shooters, n)
        skill = np.linspace(-0.8, 0.8, n_shooters)[shooter]
        late = rng.random(n) < np.where(skill > 0.3, 0.5, 0.1)
        clock = np.where(late, rng.uniform(0, 4, n), rng.uniform(4, 24, n))
        steal = rng.random(n) < 0.2
        true = 0.9 - 0.08 * d - 0.4 * (value == 3) + 0.5 * steal
        made = (rng.random(n) < 1 / (1 + np.exp(-(true + skill)))).astype(int)
        rows.append(pd.DataFrame(dict(season=s, phase="RS", half=np.where(np.arange(n) % 2 == 0, "A", "B"),
                                      shooter=shooter, arena=rng.integers(0, 30, n), value=value, made=made,
                                      dist_xy=d, angle=rng.uniform(0, 90, n), noloc=False, corner3=False,
                                      clock_left=clock, steal=steal.astype(float), true=true)))
    return pd.concat(rows, ignore_index=True)


def _feats(f):
    return f[["dist_xy", "value", "clock_left", "steal"]].astype(np.float32)


@pytest.fixture(scope="module")
def league():
    train = _frame()
    base = sl.fit_glm(train, None, dict(blocks=("spot",), lam_shooter=5.0, lam_arena=5.0))
    return train, _frame(seed=1, seasons=(2010,)), base, (train["season"] == 2005).to_numpy()


def _fit(kind, league, **kw):
    train, _, base, hold = league
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sl.fit_booster(kind, train, None, dict(PARAMS[kind]), base, hold, features=_feats, **kw)


@pytest.mark.parametrize("kind", sl.BOOSTERS)
def test_trees_on_the_offset_margin_learn_the_shot_and_not_who_takes_it(kind, league):
    _, test, _, _ = league
    late = (test["clock_left"] < 4).to_numpy()
    steal = test["steal"].to_numpy() == 1
    out = {}
    for off in (True, False):
        ft = _fit(kind, league, offsets=off)
        c = ft.correction(test)
        eta = ft.predict_raw(test)
        err = eta - test["true"].to_numpy()
        out[off] = dict(late=c[late].mean() - c[~late].mean(), steal=c[steal].mean() - c[~steal].mean(),
                        err=np.abs(err - err.mean()).mean())
        # the trees alone, read back by each library's own trees-only prediction, give the library's own
        # holdout loss once the holdout's margin is added: the margin went in, and only the trees come out
        assert abs(ft.info["holdout_loss"] - ft.info["holdout_loss_lib"]) < 1e-6
        assert ft.info["holdout_loss"] < ft.info["holdout_loss_base"]
        assert ft.info["n_fit"] == 40000 and ft.info["n_holdout"] == 10000 and ft.info["offsets"] is off
    assert abs(out[True]["late"]) < 0.1                    # no late-clock effect, as planted
    assert out[False]["late"] > 0.2                        # the control reads who shoots late as an easy shot
    assert out[True]["steal"] > 0.3 and out[False]["steal"] > 0.3     # both find the real shot effect (+0.5)
    assert out[True]["err"] < out[False]["err"]


def test_xgboost_without_a_base_margin_predicts_the_trees_alone(league):
    import xgboost as xgb
    train, _, base, _ = league
    ft = _fit("xgb", league)
    glm = base.model.margin_train(train, with_offsets=False)
    m = base.model.margin_train(train, with_offsets=True)
    A, cols = sl.tree_inputs(train, glm, _feats)
    full = ft.booster.predict(xgb.DMatrix(A, base_margin=m, feature_names=cols), output_margin=True)
    trees = ft.booster.predict(xgb.DMatrix(A, feature_names=cols), output_margin=True)
    assert np.max(np.abs((full - m) - trees)) < 1e-4
    assert np.max(np.abs(trees)) > 0.05                    # and the trees did learn something


def test_catboost_on_the_gpu_takes_the_baseline(league):
    if sl.cat_task_type() != "GPU":
        pytest.skip("no card that takes a baseline: the CPU path is the one the other tests run")
    train, _, base, hold = league
    ft = sl.fit_booster("cat", train, None, dict(learning_rate=0.1, depth=3), base, hold, features=_feats)
    assert ft.info["device"] == "GPU"
    assert abs(ft.info["holdout_loss"] - ft.info["holdout_loss_lib"]) < 1e-5


def test_the_packed_objective_reproduces_chimeraboosts_own_log_loss():
    """The gate: a margin that is the classifier's own starting constant gives its validation curve to the last
    bits; a margin of zero (start at 0, not the mean) reaches the same best loss within noise."""
    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
    rng = np.random.default_rng(0)
    n = 30000
    X = rng.normal(size=(n, 5))
    z = 0.3 + 0.8 * X[:, 0] - 0.5 * (X[:, 1] > 0.5) + 0.4 * X[:, 2] * X[:, 3]
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    tr, va = np.arange(n) < 22000, np.arange(n) >= 22000
    common = dict(n_estimators=400, learning_rate=0.1, depth=4, l2_leaf_reg=1.0, min_child_weight=1.0,
                  leaf_estimation_iterations=1, linear_leaves=False, cross_features=False, random_state=0,
                  early_stopping_rounds=30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls = ChimeraBoostClassifier(**common).fit(X[tr], y[tr], eval_set=(X[va], y[va]))
    hc = np.asarray(cls.validation_history_)
    for m, exact in ((np.full(n, cls.model_.init_), True), (np.zeros(n), False)):
        reg = ChimeraBoostRegressor(loss=sl.OffsetLogloss(), **common)
        reg.fit(X[tr], sl.pack(m[tr], y[tr]), eval_set=(X[va], sl.pack(m[va], y[va])))
        hr = np.asarray(reg.validation_history_)
        if exact:
            assert reg.best_iteration_ == cls.best_iteration_
            assert len(hr) == len(hc) and np.max(np.abs(hr - hc)) < 1e-9
        else:
            assert abs(hr.min() - hc.min()) < 5e-4


def test_the_offset_objective_packs_unpacks_and_pickles():
    margin = np.array([-3.2, 0.0, 1.7, 0.4])
    made = np.array([1.0, 0.0, 1.0, 0.0])
    y = sl.pack(margin, made)
    m2, off = sl.unpack(y)
    assert np.array_equal(m2, made) and np.allclose(off, margin, atol=1e-12)
    obj = pickle.loads(pickle.dumps(sl.OffsetLogloss()))       # bag members unpickle it in worker processes
    raw = np.array([0.3, -0.2, 0.0, 1.0])
    g, h = obj.grad_hess(y, raw)
    p = 1 / (1 + np.exp(-(raw + margin)))
    assert np.allclose(g, p - made) and np.allclose(h, p * (1 - p))
    assert np.isclose(obj.eval(y, raw), np.mean(-(made * np.log(p) + (1 - made) * np.log(1 - p))))
    assert obj.init(y) == 0.0 and np.array_equal(obj.transform(raw), raw)
    with pytest.raises(ValueError):
        sl.pack(np.array([600.0]), np.array([1.0]))


def test_fit_glm_passes_what_shotmodel_fit_takes_and_lists_the_rest():
    f = _frame(seasons=(2001, 2002, 2006), n=6000)
    kw = dict(ridge=2.0, lam_shooter=20.0, half_life_past=4.0, block=(2003, 2005))
    ft = sl.fit_glm(f, None, dict(blocks=("spot",), bogus=1, **kw))
    ref = sm.fit(f, ("spot",), **kw)
    for s in sm.SUBMODELS:
        assert np.allclose(ft.model.coef[s][1], ref.coef[s][1], atol=1e-12, rtol=0)
    assert ft.info["ignored"] == ["bogus"] and ft.kind == "glm"
    test = _frame(seed=3, seasons=(2010,), n=2000)
    assert np.allclose(ft.predict_raw(test), ref.predict_raw(test)) and not ft.correction(test).any()


def test_banned_inputs_never_reach_the_trees(league):
    train, _, base, hold = league
    for bad in ("shooter", "post_assisted", "arena_x", "lp", "season"):
        with pytest.raises(ValueError):
            sl.fit_booster("xgb", train, None, PARAMS["xgb"], base, hold,
                           features=lambda f, c=bad: _feats(f).assign(**{c: 0.0}))
    with pytest.raises(ValueError):                         # 'glm' is the regression's logit, never a feature
        sl.fit_booster("xgb", train, None, PARAMS["xgb"], base, hold, features=lambda f: _feats(f).assign(glm=0.0))


def test_a_booster_uses_tree_features_by_default_and_refuses_its_training_seasons(league, monkeypatch):
    train, test, base, hold = league
    monkeypatch.setattr(shotfeatures, "tree_features", _feats, raising=False)
    ft = sl.fit_booster("lgbm", train, None, PARAMS["lgbm"], base, hold)
    assert ft.features is None and ft.columns == ("dist_xy", "value", "clock_left", "steal", "sub", "glm")
    assert np.isfinite(ft.predict_raw(test)).all()
    with pytest.raises(ValueError):
        ft.predict_raw(train)


@pytest.mark.parametrize("kind", sl.BOOSTERS)
def test_refit_retrains_on_every_row_at_the_scaled_round_count(kind, league):
    ft = _fit(kind, league, refit=True)
    assert ft.info["refit"] and ft.info["rounds"] == max(1, round(ft.info["best_iter"] * 50000 / 40000))
    b = ft.booster
    trees = {"xgb": lambda: b.num_boosted_rounds(), "lgbm": lambda: b.num_trees(), "cat": lambda: b.tree_count_,
             "chimera": lambda: len(b.model_.trees_)}[kind]()
    assert trees == ft.info["rounds"]
    if kind == "chimera":
        assert b.model_.lr_ == 0.1                          # the rate that chose the count, pinned


def test_without_a_holdout_the_round_count_is_required(league):
    train, _, base, _ = league
    with pytest.raises(ValueError):
        sl.fit_booster("xgb", train, None, PARAMS["xgb"], base, None, features=_feats)
    ft = sl.fit_booster("xgb", train, None, dict(PARAMS["xgb"], num_boost_round=7), base, None, features=_feats)
    assert ft.booster.num_boosted_rounds() == 7 and ft.info["n_holdout"] == 0


def test_the_registry_holds_every_family_and_its_control(league):
    assert set(sl.LEARNERS) == {"glm"} | set(sl.BOOSTERS) | {f"{k}_nooff" for k in sl.BOOSTERS}
    train, _, base, hold = league
    ft = sl.fit("xgb_nooff", train, None, PARAMS["xgb"], base=base, holdout=hold, features=_feats)
    assert ft.kind == "xgb" and ft.info["offsets"] is False
    ft = sl.fit_booster("lgbm_nooff", train, None, PARAMS["lgbm"], base, hold, features=_feats)
    assert ft.kind == "lgbm" and ft.info["offsets"] is False
    assert sl.fit("glm", train, None, dict(blocks=("spot",))).kind == "glm"


def test_suggest_draws_inside_the_registered_spaces():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for kind in sl.SPACES:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        for _ in range(12):
            t = study.ask()
            p = sl.suggest(t, kind, toggles=("late",) if kind == "glm" else ())
            for name, v in t.params.items():
                d = (sl.SPACES[kind].get(name) or sl.P("cat", choices=(False, True)))
                if d.type == "cat":
                    assert v in d.choices
                else:
                    assert d.lo <= v <= d.hi
            if kind == "glm":
                # every key is shotmodel.fit's own keyword: a renamed option fails here, not silently in the search
                assert set(p) - {"blocks"} <= sl.glm_takes()
                assert p["standardise"] is True
                assert set(p["blocks"]) >= set(sm.BLOCKS) - {"clock"}
                assert ("clockfine" in p["blocks"]) == (t.params["clock"] == "fine")
                assert ("late" in p["blocks"]) == t.params["feat_late"]
                assert set(p["inter"]) == {b for b in sm.INTER_BLOCKS if t.params[f"inter_{b}"]}
                assert p["knots"] == dict(dist=t.params["knots_dist"], tposs=t.params["knots_tposs"],
                                          on=t.params["knots_on"])
                assert (p["half_life_past"] == np.inf) == (not t.params["decay_past"])
                assert ("era_lambda" in p) == t.params["era"]
            if kind == "xgb":
                assert ("max_leaves" in p) == (p["grow_policy"] == "lossguide")
            if kind == "lgbm":
                assert p["bagging_freq"] == 1
            study.tell(t, 0.0)
    # 'the XGBoost plateau translated, plus trials over a few knobs'
    study = optuna.create_study()
    t = study.ask()
    base = sl.translate_xgb(dict(eta=0.05, max_depth=6, min_child_weight=50.0, subsample=0.8), "lgbm")
    p = sl.suggest(t, "lgbm", fixed=base, only=("num_leaves", "min_data_in_leaf"))
    assert set(t.params) == {"num_leaves", "min_data_in_leaf"} and p["learning_rate"] == 0.05
    assert p["min_sum_hessian_in_leaf"] == 50.0 and p["bagging_fraction"] == 0.8


def test_penalties_are_entered_per_full_build_and_scaled_for_the_search_sample():
    p = sl.scale_penalties("xgb", dict(eta=0.1, min_child_weight=100.0, **{"lambda": 4.0}), 0.5)
    assert p == dict(eta=0.1, min_child_weight=50.0, **{"lambda": 2.0})
    q = sl.scale_penalties("lgbm_nooff", dict(min_data_in_leaf=1001, num_leaves=31), 0.5)
    assert q == dict(min_data_in_leaf=500, num_leaves=31)
    g = sl.scale_penalties("glm", dict(ridge=1.0, lam_shooter=50.0, rounds=4, half_life_past=3.0, era_lambda=None), 0.5)
    assert g == dict(ridge=0.5, lam_shooter=25.0, rounds=4, half_life_past=3.0, era_lambda=None)
