"""Step 3 (src/openrapm/prior_rapm.py): the ridge around the prior, the season's table, centring, the schema.

The fit is a plain ridge with two more unpenalised columns (the lineup priors), so it must agree with the
Frisch-Waugh two-step fit and with the fixed-centre fit at the scale it chose; halving the prior must double the
scale and move nothing else.
"""
import numpy as np
import pandas as pd
import pytest

from openrapm import prior_rapm, rapm
from openrapm.settings import PRODUCT


@pytest.fixture(scope="module")
def design(synthetic_league, synthetic_design):
    """2002, the middle of three seasons of a league whose talent persists."""
    return synthetic_design(synthetic_league(n_seasons=3, seed=4), [2002])


def _shrunk_prior(wd, penalty=3000.0):
    """A prior with the shape the games like and less than the amplitude they want: a plain RAPM of the design."""
    sums = rapm.SeasonSums(0.0).add(2002, wd)
    base = sums.ratings((), (penalty, penalty, 0.0))
    return dict(zip(base.player_id, base.offense)), dict(zip(base.player_id, base.defense))


def test_the_deviation_is_the_frisch_waugh_ridge(design):
    """Project the context and the two lineup priors out of both sides, then ridge: the same deviation."""
    prior_o, prior_d = _shrunk_prior(design)
    got = prior_rapm.fit(design, prior_o, prior_d, penalty=3000.0)
    players, context = rapm.design_blocks(design)
    n = design.spec.n_players
    ids = design.spec.players["player_id"].to_numpy()
    prior = np.concatenate([prior_rapm.aligned_prior(prior_o, ids), prior_rapm.aligned_prior(prior_d, ids)])
    free = np.column_stack([context, *prior_rapm.lineup_prior_sums(players, prior, n)])
    w = design.w
    both = np.column_stack([players.toarray(), design.y])
    weighted = (free * w[:, None]).T
    both = both - free @ np.linalg.lstsq(weighted @ free, weighted @ both, rcond=None)[0]
    z_res, y_res = both[:, :-1], both[:, -1]
    expected = np.linalg.solve((z_res * w[:, None]).T @ z_res + 3000.0 * np.eye(2 * n), (z_res * w[:, None]).T @ y_res)
    assert np.allclose(got.deviation, expected, atol=1e-7)


def test_the_free_scale_is_the_fixed_centre_fit_at_the_scale_it_chose(design):
    """Hand the fit s x prior with s pinned: the target minus the lineup priors, ridged plainly, gives r."""
    prior_o, prior_d = _shrunk_prior(design)
    got = prior_rapm.fit(design, prior_o, prior_d, penalty=3000.0)
    s_o, s_d = got.prior_scale
    players, context = rapm.design_blocks(design)
    n = design.spec.n_players
    ids = design.spec.players["player_id"].to_numpy()
    prior = np.concatenate([prior_rapm.aligned_prior(prior_o, ids), prior_rapm.aligned_prior(prior_d, ids)])
    sum_o, sum_d = prior_rapm.lineup_prior_sums(players, prior, n)
    sums = rapm.normal_equations(players, context, design.y - s_o * sum_o - s_d * sum_d, design.w)
    gram = np.block([[sums.gram_players, sums.gram_players_context],
                     [sums.gram_players_context.T, sums.gram_context]])
    theta = rapm.solve_ridge(gram, np.concatenate([sums.rhs_players, sums.rhs_context]),
                             rapm.block_penalty((n, n, context.shape[1]), (3000.0, 3000.0, 0.0)))
    assert np.allclose(theta[:2 * n], got.deviation, atol=1e-8)
    assert np.allclose(got.ratings.offense, got.ratings.prior_offense + got.deviation[:n])


def test_a_shrunk_prior_is_stretched_back_and_halving_it_doubles_the_scale(design):
    prior_o, prior_d = _shrunk_prior(design)
    full = prior_rapm.fit(design, prior_o, prior_d, penalty=3000.0)
    assert full.prior_scale[0] > 1.2, full.prior_scale
    half = prior_rapm.fit(design, {k: 0.5 * v for k, v in prior_o.items()},
                          {k: 0.5 * v for k, v in prior_d.items()}, penalty=3000.0)
    assert np.allclose(half.prior_scale, 2.0 * full.prior_scale, rtol=1e-6)
    assert np.allclose(half.ratings.offense, full.ratings.offense, atol=1e-8)
    assert np.allclose(half.ratings.prior_defense, full.ratings.prior_defense, atol=1e-8)


def test_with_a_huge_penalty_the_rating_is_the_scaled_prior(design):
    prior_o, prior_d = _shrunk_prior(design)
    got = prior_rapm.fit(design, prior_o, prior_d, penalty=1e12)
    assert np.abs(got.deviation).max() < 1e-4
    assert np.allclose(got.ratings.offense, got.ratings.prior_offense, atol=1e-4)


def test_a_player_without_a_prior_is_centred_on_zero(design):
    prior_o, prior_d = _shrunk_prior(design)
    first = design.spec.players.player_id.iloc[0]
    prior_o.pop(first)
    got = prior_rapm.fit(design, prior_o, prior_d, penalty=3000.0)
    assert got.ratings.set_index("player_id").prior_offense.loc[first] == 0.0


# ------------------------------------------------------------------ the season's table
def _season(design, season=2002):
    prior_o, prior_d = _shrunk_prior(design)
    offence = prior_rapm.fit(design, prior_o, prior_d, PRODUCT.ridge.penalty)
    defence_possessions = prior_rapm.defence_possessions(design)
    return prior_rapm.season_table(offence, offence, defence_possessions, season), defence_possessions


def test_the_season_table_has_the_published_build_column_order(design):
    table, defence_possessions = _season(design)
    assert list(table.columns) == ["player_id", "offense", "prior_offense", "possessions", "defense",
                                   "prior_defense", "poss_def", "season", "offense_lambda", "defense_lambda",
                                   "context_lambda", "prior_scale_off", "prior_scale_def"]
    assert (table.offense_lambda == 13037.0).all() and (table.context_lambda == 0.0).all()
    assert table.poss_def.to_numpy().tolist() == [defence_possessions.get(p, 0.0) for p in table.player_id]


def test_centring_makes_each_sides_possession_weighted_mean_zero(design):
    one, _ = _season(design, 2002)
    two = one.assign(season=2003, offense=one.offense + 1.5, prior_offense=one.prior_offense + 1.5)
    centred = prior_rapm.centre(pd.concat([one, two], ignore_index=True))
    for _, g in centred.groupby("season"):
        for side in ("offense", "defense"):
            assert abs(np.average(g[side], weights=g.possessions)) < 1e-12
    raw_u = one.offense - one.prior_offense
    assert np.allclose((centred.offense - centred.prior_offense)[:len(one)], raw_u, atol=1e-12)


def test_the_published_table_is_positive_good_and_named(design):
    table, _ = _season(design)
    centred = prior_rapm.centre(table)
    names = pd.DataFrame({"player_id": centred.player_id.iloc[1:], "player_name": "someone"})
    out = prior_rapm.published_table(centred, names)
    assert list(out.columns) == prior_rapm.PUBLISHED_COLUMNS
    assert np.array_equal(out.rating_def, -out.defense) and np.array_equal(out.prior_def, -out.prior_defense)
    assert np.allclose(out.rating_total, out.rating_off + out.rating_def)
    assert np.allclose(out.u_total, out.rating_total - out.prior_total)
    assert out.player_name.isna().tolist() == [True] + [False] * (len(out) - 1)


def test_rate_season_is_both_fits_and_the_centred_table(design):
    prior_o, prior_d = _shrunk_prior(design)
    designs = {PRODUCT.targets.offence: design, PRODUCT.targets.defence: design}
    got = prior_rapm.rate_season(designs, {"O": prior_o, "D": prior_d}, PRODUCT, 2002)
    table, _ = _season(design)
    assert got.equals(prior_rapm.centre(table))


def test_a_scale_held_at_the_one_the_games_chose_gives_the_same_fit(design):
    """Experiment 109: `scale` holds (s_O, s_D); held where the free fit put them, nothing moves."""
    prior_o, prior_d = _shrunk_prior(design)
    free = prior_rapm.fit(design, prior_o, prior_d, penalty=3000.0)
    held = prior_rapm.fit(design, prior_o, prior_d, penalty=3000.0, scale=tuple(free.prior_scale))
    assert np.allclose(held.deviation, free.deviation, atol=1e-8)
    assert np.allclose(held.ratings[["offense", "defense"]], free.ratings[["offense", "defense"]], atol=1e-8)
    assert np.allclose(held.unpenalised, free.unpenalised, atol=1e-8)


def test_a_held_scale_sets_the_reported_prior(design):
    prior_o, prior_d = _shrunk_prior(design)
    held = prior_rapm.fit(design, prior_o, prior_d, penalty=3000.0, scale=(1.5, 0.5))
    ids = design.spec.players["player_id"].to_numpy()
    assert list(held.prior_scale) == [1.5, 0.5]
    assert np.allclose(held.ratings.prior_offense, 1.5 * prior_rapm.aligned_prior(prior_o, ids))
    assert np.allclose(held.ratings.prior_defense, 0.5 * prior_rapm.aligned_prior(prior_d, ids))


def test_a_blended_scale_is_a_straight_line_from_held_to_fitted(design):
    """2026-10-31, the owner: blend the two shrinkages.  Weight 0 is the held fit, 1 the plain one, and between them the
    scales sit that share of the way from held to fitted, the ratings solved again at them."""
    prior_o, prior_d = _shrunk_prior(design)
    designs = {PRODUCT.targets.offence: design, PRODUCT.targets.defence: design}
    priors = {"O": prior_o, "D": prior_d}
    free = prior_rapm.rate_season(designs, priors, PRODUCT, 2002)
    fitted = (float(free.prior_scale_off.iloc[0]), float(free.prior_scale_def.iloc[0]))
    held = (fitted[0] + 0.4, fitted[1] - 0.3)
    at = {w: prior_rapm.rate_season_blended(designs, priors, PRODUCT, 2002, held, w) for w in (0.0, 0.25, 1.0)}
    pd.testing.assert_frame_equal(at[1.0], free)
    pd.testing.assert_frame_equal(at[0.0], prior_rapm.rate_season(designs, priors, PRODUCT, 2002, held))
    assert at[0.25].prior_scale_off.iloc[0] == pytest.approx(0.75 * held[0] + 0.25 * fitted[0])
    assert at[0.25].prior_scale_def.iloc[0] == pytest.approx(0.75 * held[1] + 0.25 * fitted[1])
    pd.testing.assert_frame_equal(prior_rapm.rate_season_blended(designs, priors, PRODUCT, 2002, None, 0.0), free)
    assert [prior_rapm.blend_weight(share, 0.75) for share in (0.0, 0.25, 0.5, 0.75, 0.9)] == pytest.approx(
        [0.0, 1 / 3, 2 / 3, 1.0, 1.0])
    assert prior_rapm.blend_weight(0.2, 0.0) == 1.0
    assert not PRODUCT.ridge.held_scale_blend, "off until its test passes"


def test_the_held_scale_is_the_mean_of_the_eight_nearest_other_seasons():
    seasons = np.arange(1997, 2027)
    table = pd.DataFrame({"season": np.repeat(seasons, 2), "prior_scale_off": np.repeat(seasons - 2000.0, 2),
                          "prior_scale_def": np.repeat(np.ones(len(seasons)), 2)})
    assert prior_rapm.held_prior_scale(table, 2010) == (np.mean([6, 7, 8, 9, 11, 12, 13, 14]), 1.0)
    assert prior_rapm.held_prior_scale(table, 2027)[0] == np.mean(range(19, 27)), "2019-2026 for the season in progress"
    assert prior_rapm.held_prior_scale(table, 1997)[0] == np.mean(range(-2, 6)), "1998-2005 at the start"
