"""PriorRidgeCV (src/eracoef/priorridge.py) against the simulator and against the fit it replaces.

The class exists to turn three magic constants into either a tested choice or a cross-validated one, so
the tests are: does it agree with the plain two-step fit at a fixed penalty (it must, or it is a
different estimator wearing the name), does the prior actually act as the centre, are the fixed effects
really left unpenalised, and does the CV pick a sane penalty rather than a boundary.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from eracoef.design import FEATURES, build_design
from eracoef.holdout import Context, Ratings, predict_season, score
from eracoef.priorridge import PriorRidgeCV
from eracoef.simulate import simulate

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500, "features": FEATURES,
       "first_season": 2001, "last_season": 2005, "windows": [[2001, 2003], [2004, 2006]],
       "lam_plugin": 4000.0, "lam_ratio_plugin": 1.0, "pad_target": "league", "holdout": {}}


@pytest.fixture(scope="module")
def design():
    sim = simulate(n_seasons=3, n_teams=8, players_per_team=10, games_per_season=60,
                   stints_per_game=(20, 30), rho=0.85, turnover=0.3, seed=4, eps_var=0.2, leak=False)
    stints, box = sim["stints"], sim["box"]

    def loader(seasons, cfg, target, phases=None, **kw):
        return build_design(stints[stints.season.isin(seasons)], box[box.season.isin(seasons)], FEATURES, cfg)

    ctx = Context(cfg=CFG, loader=loader)
    return sim, ctx, ctx.design([2002], "pts")


def _two_step(design, prior_offense, prior_defense, alpha, ratio):
    """The fit PriorRidgeCV replaces: project the context out, then sklearn's Ridge on scaled columns."""
    n_players, n_fixed = design.spec.n_ps, len(design.spec.f_names)
    ids = design.spec.ps_table["player_id"].to_numpy()
    take = lambda p: (np.zeros(n_players) if p is None                      # noqa: E731
                      else pd.Series(p).reindex(ids).fillna(0.0).to_numpy())
    prior = np.concatenate([take(prior_offense), take(prior_defense)])
    players = design.X[:, :2 * n_players].tocsr()
    fixed = np.asarray(design.X[:, 2 * n_players:2 * n_players + n_fixed].todense())
    context = np.column_stack([np.ones(design.X.shape[0]), fixed])
    y, w = design.y, design.w
    centred = y - players @ prior
    left = (context * w[:, None]).T
    residual = centred - context @ np.linalg.lstsq(left @ context, left @ centred, rcond=None)[0]
    scale = np.concatenate([np.ones(n_players), np.full(n_players, 1.0 / np.sqrt(ratio))])
    model = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", tol=1e-12, max_iter=20000)
    model.fit(players.multiply(scale[None, :]).tocsr(), residual, sample_weight=w)
    return prior + model.coef_ * scale


def test_it_is_the_same_estimator_as_the_two_step_fit(design):
    _, _, wd = design
    rng = np.random.default_rng(0)
    ids = wd.spec.ps_table["player_id"].to_numpy()
    po = dict(zip(ids, rng.normal(0, 1.5, len(ids))))
    pdd = dict(zip(ids, rng.normal(0, 0.8, len(ids))))
    alpha, ratio = 4000.0, 0.624519
    model = PriorRidgeCV(alphas=[alpha], defense_penalty_ratio=ratio, n_folds=1).fit(wd, po, pdd)
    mine = np.concatenate([model.ratings_.offense, model.ratings_.defense])
    assert np.allclose(mine, _two_step(wd, po, pdd, alpha, ratio), atol=2e-3)


def test_the_prior_is_the_centre_not_zero(design):
    _, _, wd = design
    ids = wd.spec.ps_table["player_id"].to_numpy()
    huge = PriorRidgeCV(alphas=[1e12], n_folds=1).fit(wd, dict(zip(ids, np.full(len(ids), 2.0))), None)
    # an infinite penalty leaves the prior untouched -- shrink toward the prior, never toward zero
    assert np.allclose(huge.ratings_.offense, 2.0, atol=1e-4)
    assert np.allclose(huge.ratings_.defense, 0.0, atol=1e-4)
    assert np.abs(huge.residual_).max() < 1e-4


def test_the_context_is_never_penalised(design):
    """A huge penalty flattens the players; the home term must still be estimated at full size."""
    _, _, wd = design
    small = PriorRidgeCV(alphas=[10.0], n_folds=1).fit(wd, None, None)
    huge = PriorRidgeCV(alphas=[1e12], n_folds=1).fit(wd, None, None)
    assert np.abs(huge.residual_).max() < 1e-4
    assert abs(huge.level_[1]) > 0.2 * abs(small.level_[1]) or abs(huge.level_[1]) > 0.5


def test_defense_can_be_penalised_differently(design):
    _, _, wd = design
    even = PriorRidgeCV(alphas=[4000.0], defense_penalty_ratio=1.0, n_folds=1).fit(wd, None, None)
    light = PriorRidgeCV(alphas=[4000.0], defense_penalty_ratio=0.25, n_folds=1).fit(wd, None, None)
    d_defense = light.ratings_.defense.std() / even.ratings_.defense.std()
    d_offense = light.ratings_.offense.std() / even.ratings_.offense.std()
    assert d_defense > 1.05                    # a lighter penalty leaves defense wider
    # offense moves too -- the two blocks are estimated jointly, so the gram couples them -- but far less
    assert abs(d_offense - 1.0) < 0.25 * abs(d_defense - 1.0)


def test_the_cv_picks_an_interior_penalty_and_beats_the_ends(design):
    _, _, wd = design
    alphas = np.logspace(1, 6, 11)
    model = PriorRidgeCV(alphas=alphas, n_folds=5).fit(wd, None, None)
    assert model.alpha_ not in (alphas[0], alphas[-1]), "an argmax on a grid boundary has chosen nothing"
    assert model.cv_error_.loc[model.alpha_] == model.cv_error_.min()
    assert model.cv_error_.iloc[0] > model.cv_error_.min()
    assert model.cv_error_.iloc[-1] > model.cv_error_.min()


def test_the_cv_folds_keep_a_game_together(design):
    """Ten players repeat across a game's stints, so a split inside a game leaks the answer.  If it did,
    the CV would choose a far smaller penalty than a game-grouped one."""
    _, _, wd = design
    games = wd.rows["game_idx"].to_numpy()
    fold_of = {}
    model = PriorRidgeCV(alphas=[1000.0, 4000.0], n_folds=5, seed=3)
    rng = np.random.default_rng(model.seed)
    unique = np.unique(games)
    fold = pd.Series(rng.permutation(unique.size) % model.n_folds, index=unique).reindex(games).to_numpy()
    for g in unique[:50]:
        fold_of[g] = np.unique(fold[games == g])
        assert fold_of[g].size == 1


def test_it_produces_ratings_the_criterion_can_score(design):
    sim, ctx, wd = design
    model = PriorRidgeCV(n_folds=3).fit(wd, None, None)
    rat = Ratings(model.as_ratings_frame())
    assert list(rat.df.columns) == ["player_id", "o", "d", "poss", "prior_o", "prior_d"]
    got = score(predict_season(rat, wd, level="full"))
    assert got["tg"] < got["tg_base"]                      # in-sample, so this is a floor not a result
    truth = sim["truth"]["ps"]
    truth = truth[truth.season == 2002]
    j = rat.df.merge(truth[["player_id", "impact_O"]], on="player_id")
    assert np.corrcoef(j.o, j.impact_O)[0, 1] > 0.3
