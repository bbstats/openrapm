"""PriorRidgeCV (src/eracoef/priorridge.py) against the simulator and against the fit it replaces.

The class exists to turn four magic constants into a tested choice or a cross-validated one, so the tests
are: does it agree with the plain two-step fit when the context penalty is zero (it must, or it is a
different estimator wearing the name), does the prior act as the centre, are the three penalties really
independent, does the objective aggregate to team-games and lean on close ones, and does the CV pick an
interior triple rather than a boundary.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from eracoef.design import FEATURES, build_design
from eracoef.holdout import Context, Ratings, predict_season, score
from eracoef.priorridge import (DEFAULT_CONTEXT_LAMBDAS, MAE_SCALE, PriorRidgeCV, armse, penalty_grid,
                                team_game_mse, team_game_weights)
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


def _fixed(offense, defense=None, context=0.0, **kw):
    return PriorRidgeCV(offense_lambdas=[offense], defense_lambdas=[offense if defense is None else defense],
                        context_lambdas=[context], n_folds=1, **kw)


def _two_step(design, prior_offense, prior_defense, offense_lambda, defense_lambda):
    """The Frisch-Waugh fit this class does at context_lambda = 0: project the context out, then ridge."""
    n_players, n_fixed = design.spec.n_ps, len(design.spec.f_names)
    ids = design.spec.ps_table["player_id"].to_numpy()

    def take(p):
        return np.zeros(n_players) if p is None else pd.Series(p).reindex(ids).fillna(0.0).to_numpy()

    prior = np.concatenate([take(prior_offense), take(prior_defense)])
    players = design.X[:, :2 * n_players].tocsr()
    context = np.asarray(design.X[:, 2 * n_players:2 * n_players + n_fixed].todense())
    y, w = design.y, design.w
    centred = y - players @ prior
    left = (context * w[:, None]).T
    inner = left @ context
    # proper Frisch-Waugh residualises BOTH sides on the context.  The version this class used to ship
    # residualised only the target, which left the context's correlation inside the player columns.
    residual = centred - context @ np.linalg.lstsq(inner, left @ centred, rcond=None)[0]
    dense = np.asarray(players.todense())
    dense = dense - context @ np.linalg.lstsq(inner, left @ dense, rcond=None)[0]
    ratio = defense_lambda / offense_lambda
    scale = np.concatenate([np.ones(n_players), np.full(n_players, 1.0 / np.sqrt(ratio))])
    model = Ridge(alpha=offense_lambda, fit_intercept=False, solver="lsqr", tol=1e-12, max_iter=20000)
    model.fit(dense * scale[None, :], residual, sample_weight=w)
    return prior + model.coef_ * scale


def test_a_zero_context_penalty_is_the_frisch_waugh_fit(design):
    _, _, wd = design
    rng = np.random.default_rng(0)
    ids = wd.spec.ps_table["player_id"].to_numpy()
    po = dict(zip(ids, rng.normal(0, 1.5, len(ids))))
    pdd = dict(zip(ids, rng.normal(0, 0.8, len(ids))))
    model = _fixed(4000.0, 4000.0 * 0.624519).fit(wd, po, pdd)
    mine = np.concatenate([model.ratings_.offense, model.ratings_.defense])
    assert np.allclose(mine, _two_step(wd, po, pdd, 4000.0, 4000.0 * 0.624519), atol=3e-3)


def test_the_prior_is_the_centre_not_zero(design):
    _, _, wd = design
    ids = wd.spec.ps_table["player_id"].to_numpy()
    huge = _fixed(1e12).fit(wd, dict(zip(ids, np.full(len(ids), 2.0))), None)
    assert np.allclose(huge.ratings_.offense, 2.0, atol=1e-4)
    assert np.allclose(huge.ratings_.defense, 0.0, atol=1e-4)
    assert np.abs(huge.residual_).max() < 1e-4


# ------------------------------------------------------------------ three penalties, independently
def test_the_three_penalties_move_three_different_things(design):
    _, _, wd = design
    base = _fixed(4000.0, 4000.0, 0.0).fit(wd, None, None)
    light_defense = _fixed(4000.0, 800.0, 0.0).fit(wd, None, None)
    heavy_context = _fixed(4000.0, 4000.0, 1e14).fit(wd, None, None)

    assert light_defense.ratings_.defense.std() > 1.1 * base.ratings_.defense.std()
    # offense moves too -- the blocks are estimated jointly -- but far less
    assert abs(light_defense.ratings_.offense.std() / base.ratings_.offense.std() - 1.0) < 0.25
    assert np.abs(heavy_context.level_).max() < 1e-3          # the context really is penalised now
    assert np.abs(base.level_).max() > 0.1


def test_zero_stays_reachable_on_the_context_grid():
    assert 0.0 in set(DEFAULT_CONTEXT_LAMBDAS)               # the old Frisch-Waugh fit must stay reachable
    grid = penalty_grid([1.0, 2.0], None, [0.0, 5.0])
    assert len(grid) == 2 * 2 * 2                            # defense=None reuses the offense grid
    assert (1.0, 2.0, 5.0) in grid


def test_the_cv_picks_an_interior_triple(design):
    _, _, wd = design
    offense = np.logspace(1.5, 6, 6)
    model = PriorRidgeCV(offense_lambdas=offense, defense_lambdas=offense,
                         context_lambdas=[0.0, 1e4], n_folds=4).fit(wd, None, None)
    assert list(model.cv_armse_.columns) == ["offense_lambda", "defense_lambda", "context_lambda",
                                             "mse", "armse"]
    assert len(model.cv_armse_) == 6 * 6 * 2
    assert model.cv_armse_.armse.is_monotonic_increasing                    # sorted best first
    assert model.offense_lambda_ not in (offense[0], offense[-1]), "an argmax on a boundary chose nothing"
    assert model.defense_lambda_ not in (offense[0], offense[-1])


# ------------------------------------------------------------------ the objective
def test_armse_is_the_root_mean_square_on_the_mean_absolute_scale():
    assert MAE_SCALE == pytest.approx(np.sqrt(2.0 / np.pi))
    assert armse(100.0) == pytest.approx(10.0 * MAE_SCALE)
    sample = np.random.default_rng(0).normal(0.0, 3.0, 400_000)
    assert armse(float((sample ** 2).mean())) == pytest.approx(float(np.abs(sample).mean()), rel=0.01)


def test_close_games_get_more_weight_than_blowouts(design):
    _, _, wd = design
    key, possessions, weight, average_margin = team_game_weights(wd)
    assert average_margin.min() >= 0.0 and average_margin.max() <= 25.0     # the design clips at margin_clip
    assert np.unique(key).size == 2 * np.unique(wd.rows["game_idx"]).size   # two team-games per game

    flat = team_game_weights(wd, weight_by_closeness=False)[2]
    per_possession = weight / np.maximum(flat, 1e-9)
    margin_of_team_game = np.zeros(weight.size)
    margin_of_team_game[key] = average_margin[
        np.unique(wd.rows["game_idx"].to_numpy(), return_inverse=True)[1]]
    close = per_possession[margin_of_team_game <= np.median(margin_of_team_game)]
    wide = per_possession[margin_of_team_game > np.median(margin_of_team_game)]
    assert close.mean() > wide.mean(), "a close game must weigh more per possession than a blowout"


def test_a_team_game_error_is_smaller_than_a_stint_one(design):
    """A team-game pools ~100 possessions, so its error is far below a single stint's."""
    _, _, wd = design
    key, possessions, weight, _ = team_game_weights(wd)
    row_error = wd.y - np.average(wd.y, weights=wd.w)
    pooled, _ = team_game_mse(key, possessions, weight, row_error)
    stint = float(np.average(row_error ** 2, weights=wd.w))
    assert pooled < 0.5 * stint
    assert 0.0 < armse(pooled) < 25.0


def test_the_floor_keeps_a_perfectly_tied_game_finite(design):
    _, _, wd = design
    weight = team_game_weights(wd, closeness_floor=1.0)[2]
    assert np.isfinite(weight).all() and (weight > 0).all()


# ------------------------------------------------------------------ the folds
def test_the_cv_folds_keep_a_game_together(design):
    """Ten players repeat across a game's stints, so a split inside a game leaks the answer."""
    _, _, wd = design
    games = wd.rows["game_idx"].to_numpy()
    model = PriorRidgeCV(n_folds=5, seed=3)
    unique = np.unique(games)
    fold = pd.Series(np.random.default_rng(model.seed).permutation(unique.size) % model.n_folds,
                     index=unique).reindex(games).to_numpy()
    for g in unique[:50]:
        assert np.unique(fold[games == g]).size == 1


def test_it_produces_ratings_the_criterion_can_score(design):
    sim, ctx, wd = design
    model = PriorRidgeCV(offense_lambdas=np.logspace(2.5, 5, 4), context_lambdas=[0.0],
                         n_folds=3).fit(wd, None, None)
    rat = Ratings(model.as_ratings_frame())
    assert list(rat.df.columns) == ["player_id", "o", "d", "poss", "prior_o", "prior_d"]
    got = score(predict_season(rat, wd, level="full"))
    assert got["tg"] < got["tg_base"]                      # in-sample, so this is a floor not a result
    truth = sim["truth"]["ps"]
    truth = truth[truth.season == 2002]
    j = rat.df.merge(truth[["player_id", "impact_O"]], on="player_id")
    assert np.corrcoef(j.o, j.impact_O)[0, 1] > 0.3
