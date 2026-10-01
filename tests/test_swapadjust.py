"""The swap adjustment (src/eracoef/swapadjust.py) on the synthetic league of tests/test_swaptest.py."""
import numpy as np
import pandas as pd

from eracoef import swapadjust as sa
from eracoef import swaptest as sw
from test_swaptest import _league


def _season(seed=0):
    stints, truth, roster = _league(seed, games=320)
    return sw.prepare(sw.season_rows(stints)), truth, roster


def _team_totals(c, team_poss):
    t = team_poss.assign(c=team_poss.player_id.map(c))
    return t.groupby("team").apply(lambda g: float(np.sum(g.poss * g.c)), include_groups=False)


def test_every_team_total_stays_put():
    season, truth, _ = _season()
    pairs = sa.residual_pairs(season, truth.assign(o=truth.o + 1.0), context="none")
    poss = sa.team_possessions(season)
    rng = np.random.default_rng(3)
    mean = pd.Series(rng.normal(0, 2, len(truth)), index=truth.player_id)
    for side in sa.SIDES:
        for tau in (100.0, np.inf):
            c = sa.corrections(pairs[side], poss[side], mean, tau)
            assert np.allclose(_team_totals(c, poss[side]).to_numpy(), 0.0, atol=1e-6), (side, tau)


def test_a_planted_split_inside_a_team_is_found_and_undone():
    season, truth, roster = _season()
    good, bad = int(roster[1][0]), int(roster[1][1])
    rated = truth.set_index("player_id").copy()
    rated.loc[good, "o"] -= 2.0                       # the ratings take two points of offense from one teammate
    rated.loc[bad, "o"] += 2.0                        # and hand them to another: the team's total is unchanged
    pairs = sa.residual_pairs(season, rated.reset_index(), context="none")
    c = sa.corrections(pairs["offense"], sa.team_possessions(season)["offense"], pd.Series(dtype=float), tau=50.0)
    assert c[good] - c[bad] > 2.5                     # the gap was off by 4; the swaps take most of it back
    fixed = rated.o + c.reindex(rated.index).fillna(0.0)
    true = truth.set_index("player_id").o
    assert abs((fixed[good] - fixed[bad]) - (true[good] - true[bad])) < 1.5


def test_the_type_model_finds_a_planted_feature():
    season, truth, roster = _season()
    rng = np.random.default_rng(5)
    feats = pd.DataFrame({"x": rng.normal(0, 1, len(truth)), "noise": rng.normal(0, 1, len(truth))},
                         index=truth.player_id)
    rated = truth.set_index("player_id").copy()
    rated["o"] = rated.o - 1.5 * feats.x              # the ratings under-credit x by 1.5 a standard deviation
    pairs = sa.residual_pairs(season, rated.reset_index(), context="none")["offense"]
    fold_of = pd.Series(truth.player_id.to_numpy() % 5, index=truth.player_id)
    g = sa.pair_grams(pairs, feats, fold_of, folds=5)
    beta = sa.type_coefficients([g])
    assert abs(beta[0] - 1.5) < 0.3 and abs(beta[1]) < 0.3
    for k in range(5):                                # a fold's model is exactly the model of the other folds' pairs
        fa, fb = pairs.fifth_a.map(fold_of), pairs.fifth_b.map(fold_of)
        without = sa.pair_grams(pairs[(fa != k) & (fb != k)], feats, fold_of, folds=5)
        assert np.allclose(sa.type_coefficients([g], fold=k), sa.type_coefficients([without]))


def test_an_infinite_penalty_moves_the_type_prediction_onto_zero_team_totals_as_little_as_possible():
    """The nearest point to the type prediction with every team's total at zero: each player gives back an
    amount in proportion to his share of the team's possessions, the same multiple for a whole team."""
    season, truth, _ = _season()
    poss = sa.team_possessions(season)["offense"]
    mean = pd.Series(np.arange(len(truth), dtype=float), index=truth.player_id)
    c = sa.corrections(sa.residual_pairs(season, truth, context="none")["offense"], poss, mean, np.inf)
    t = poss.assign(m=poss.player_id.map(mean), c=poss.player_id.map(c))
    t["share"] = t.poss / t.groupby("team").poss.transform("sum")
    multiple = (t.m - t.c) / t.share
    assert np.allclose(multiple, multiple.groupby(t.team).transform("mean"))
    assert np.allclose((t.share * t.c).groupby(t.team).sum(), 0.0)


def test_holding_the_spread_inside_teams_keeps_team_totals_and_the_order_inside_each_team():
    rng = np.random.default_rng(11)
    team = np.repeat([1, 2, 3], 6).astype(float)
    weight = rng.uniform(100, 3000, len(team))
    base = rng.normal(0, 1, len(team)) + np.repeat([1.0, -0.5, 0.2], 6)
    adjusted = base + rng.normal(0, 1.5, len(team))            # wider inside each team, as the adjustment is
    held, k = sa.hold_spread(adjusted, base, weight, team, "within")
    assert 0 < k < 1
    for t in (1, 2, 3):
        m = team == t
        assert np.isclose(np.average(held[m], weights=weight[m]), np.average(adjusted[m], weights=weight[m]))
        assert (np.argsort(held[m]) == np.argsort(adjusted[m])).all()

    def within(x):
        means = {t: np.average(x[team == t], weights=weight[team == t]) for t in (1, 2, 3)}
        return np.average((x - np.array([means[t] for t in team])) ** 2, weights=weight)
    assert np.isclose(within(held), within(base))


def test_holding_the_whole_spread_keeps_the_order_and_the_centre():
    rng = np.random.default_rng(12)
    weight = rng.uniform(100, 3000, 40)
    base = rng.normal(0, 1, 40)
    adjusted = 1.4 * base + rng.normal(0, 0.5, 40)
    held, k = sa.hold_spread(adjusted, base, weight, np.full(40, np.nan), "whole")
    assert (np.argsort(held) == np.argsort(adjusted)).all()
    assert np.isclose(np.average(held, weights=weight), np.average(adjusted, weights=weight))
    var = lambda x: np.average((x - np.average(x, weights=weight)) ** 2, weights=weight)  # noqa: E731
    assert np.isclose(var(held), var(base))
