"""LeaveSeasonOutRAPM (src/eracoef/looseason.py).

The design rests on one claim: with player units the normal equations are additive over seasons, so the
fit that leaves season H out is the total gram minus H's.  If that is wrong everything downstream is wrong
and nothing else would notice, so it is tested against a direct fit on the same subset.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef.design import FEATURES, build_design
from eracoef.holdout import Context
from eracoef.looseason import LeaveSeasonOutRAPM
from eracoef.simulate import simulate

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500, "features": FEATURES,
       "first_season": 2001, "last_season": 2004, "windows": [[2001, 2003], [2004, 2006]],
       "lam_plugin": 4000.0, "lam_ratio_plugin": 1.0, "pad_target": "league", "holdout": {}}
SEASONS = [2001, 2002, 2003, 2004]


@pytest.fixture(scope="module")
def world():
    sim = simulate(n_seasons=4, n_teams=8, players_per_team=10, games_per_season=60,
                   stints_per_game=(20, 30), rho=0.9, turnover=0.25, seed=6, eps_var=0.2, leak=False)
    stints, box = sim["stints"], sim["box"]

    def loader(seasons, cfg, target, phases=None, **kw):
        return build_design(stints[stints.season.isin(seasons)], box[box.season.isin(seasons)], FEATURES, cfg)

    ctx = Context(cfg=CFG, loader=loader)
    designs = {s: ctx.design([s], "pts") for s in SEASONS}
    return sim, ctx, designs


def _fitted(designs, seasons, alpha=2000.0, min_possessions=0.0):
    model = LeaveSeasonOutRAPM(defense_penalty_ratio=1.0, min_possessions=min_possessions)
    for s in seasons:
        model.add_season(s, designs[s])
    return model


# ------------------------------------------------------------ the claim the whole design rests on
def test_leaving_a_season_out_equals_fitting_without_it(world):
    """gram_total - gram_H must be the gram of the other seasons, or the subtraction is a fiction."""
    _, _, designs = world
    everything = _fitted(designs, SEASONS)
    subset = _fitted(designs, [s for s in SEASONS if s != 2003])

    held = everything.ratings(held_out_season=2003, alpha=2000.0).set_index("player_id")
    direct = subset.ratings(alpha=2000.0).set_index("player_id")
    shared = held.index.intersection(direct.index)
    assert len(shared) > 50
    assert np.allclose(held.loc[shared, "offense"], direct.loc[shared, "offense"], atol=1e-8)
    assert np.allclose(held.loc[shared, "defense"], direct.loc[shared, "defense"], atol=1e-8)
    assert np.allclose(held.loc[shared, "possessions"], direct.loc[shared, "possessions"])


def test_the_order_seasons_are_added_in_does_not_matter(world):
    _, _, designs = world
    forward = _fitted(designs, SEASONS).ratings(alpha=2000.0).set_index("player_id").sort_index()
    backward = _fitted(designs, SEASONS[::-1]).ratings(alpha=2000.0).set_index("player_id").sort_index()
    assert np.allclose(forward.offense, backward.offense, atol=1e-9)


# ------------------------------------------------------------ what the ratings mean
def test_it_recovers_the_simulated_talent(world):
    sim, _, designs = world
    model = _fitted(designs, SEASONS)
    rating = model.ratings(alpha=1000.0)
    truth = sim["truth"]["ps"].groupby("player_id")[["impact_O", "impact_D"]].mean().reset_index()
    joined = rating.merge(truth, on="player_id")
    assert len(joined) > 50
    assert np.corrcoef(joined.offense, joined.impact_O)[0, 1] > 0.6
    assert np.corrcoef(joined.defense, joined.impact_D)[0, 1] > 0.6


def test_more_seasons_beat_fewer(world):
    """The point of pooling: four seasons of evidence must track talent better than one."""
    sim, _, designs = world
    truth = sim["truth"]["ps"].groupby("player_id")[["impact_O"]].mean().reset_index()

    def quality(seasons):
        r = _fitted(designs, seasons).ratings(alpha=1000.0).merge(truth, on="player_id")
        return np.corrcoef(r.offense, r.impact_O)[0, 1]

    assert quality(SEASONS) > quality([2002])


def test_a_bigger_penalty_shrinks_the_ratings(world):
    _, _, designs = world
    model = _fitted(designs, SEASONS)
    light = model.ratings(alpha=200.0)
    heavy = model.ratings(alpha=50000.0)
    assert heavy.offense.std() < 0.5 * light.offense.std()
    assert len(light) == len(heavy)


def test_defense_can_be_penalised_differently(world):
    _, _, designs = world
    even = LeaveSeasonOutRAPM(defense_penalty_ratio=1.0, min_possessions=0.0)
    light = LeaveSeasonOutRAPM(defense_penalty_ratio=0.2, min_possessions=0.0)
    for s in SEASONS:
        even.add_season(s, designs[s])
        light.add_season(s, designs[s])
    assert light.ratings(alpha=5000.0).defense.std() > even.ratings(alpha=5000.0).defense.std()


# ------------------------------------------------------------ the possession floor
def test_the_possession_floor_drops_the_barely_seen(world):
    _, _, designs = world
    model = _fitted(designs, SEASONS)
    everyone = model.ratings(alpha=2000.0)
    floored = LeaveSeasonOutRAPM(defense_penalty_ratio=1.0, min_possessions=1e9)
    for s in SEASONS:
        floored.add_season(s, designs[s])
    assert len(floored.ratings(alpha=2000.0)) == 0
    assert (everyone.possessions > 0).all()
    # the floor counts a possession ONCE, not once per side
    model_100 = _fitted(designs, SEASONS, min_possessions=100.0)
    kept = model_100.ratings(alpha=2000.0)
    assert (kept.possessions >= 100.0).all()
    assert len(kept) <= len(everyone)


# ------------------------------------------------------------ the per-season context
def test_each_season_keeps_its_own_level(world):
    """Add 40 points per 100 to one season's scoring.  The context is projected out per season, so the
    ratings must not move -- if the level were pooled the shifted season would drag everyone."""
    _, _, designs = world
    base = _fitted(designs, SEASONS).ratings(alpha=2000.0).set_index("player_id")

    from dataclasses import replace
    shifted = dict(designs)
    bumped = designs[2002]
    shifted[2002] = replace(bumped, y=bumped.y + 40.0)
    moved = _fitted(shifted, SEASONS).ratings(alpha=2000.0).set_index("player_id")

    shared = base.index.intersection(moved.index)
    assert np.allclose(base.loc[shared, "offense"], moved.loc[shared, "offense"], atol=1e-6)


# ------------------------------------------------------------ choosing the penalty
def test_the_sweep_prefers_an_interior_penalty_and_reports_armse(world):
    _, _, designs = world
    model = _fitted(designs, SEASONS, min_possessions=50.0)
    alphas = np.logspace(1, 6, 6)
    table = model.sweep(alphas, lambda s: designs[s], scoring_seasons=[2002, 2003], verbose=False)
    assert list(table.columns) == ["mse", "armse"]
    assert np.allclose(table.armse, np.sqrt(table.mse) * np.sqrt(2.0 / np.pi))
    best = float(table.armse.idxmin())
    assert best not in (alphas[0], alphas[-1]), "an argmax on a grid boundary has chosen nothing"


def test_the_sweep_never_scores_a_season_it_trained_on(world):
    """Scoring season S must be out of the fit that predicts it, or the sweep picks no penalty at all."""
    _, _, designs = world
    model = _fitted(designs, SEASONS, min_possessions=0.0)
    honest = model.sweep([2000.0], lambda s: designs[s], scoring_seasons=[2002], verbose=False)
    leaky_rating = model.ratings(alpha=2000.0)
    from eracoef.priorridge import team_game_mse, team_game_weights
    key, poss, weight, _ = team_game_weights(designs[2002])
    leaky, _ = team_game_mse(key, poss, weight, model.predict_error(designs[2002], leaky_rating))
    assert leaky < float(honest.mse.iloc[0]), "training on the scored season should look better; it is not honest"


def test_an_unrated_player_scores_as_average(world):
    _, _, designs = world
    model = _fitted(designs, SEASONS)
    rating = model.ratings(alpha=2000.0)
    error_full = model.predict_error(designs[2002], rating)
    error_empty = model.predict_error(designs[2002], rating.iloc[:0])
    assert np.isfinite(error_full).all() and np.isfinite(error_empty).all()
    assert np.abs(error_empty).mean() > np.abs(error_full).mean()
