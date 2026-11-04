"""The line fitted on the rating (src/openrapm/rating_line.py): a rating is a straight line in its prior, and the sums
give the within-season test's error at any weights."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from openrapm import prior_rapm, rapm
from openrapm import rating_line as rl
from openrapm import second_prior as sp2


@pytest.fixture(scope="module")
def design(synthetic_league, synthetic_design):
    """2002, the middle of three seasons of a league whose talent persists."""
    return synthetic_design(synthetic_league(n_seasons=3, seed=4), [2002])


def _priors(design, seed=0):
    n_players = design.spec.n_players
    return np.random.default_rng(seed).normal(0.0, 1.0, 2 * n_players)


def test_a_rating_is_a_straight_line_in_its_prior(design):
    # step 3 with its scales held at 1 is the prior plus what the ridge fits to the target less the lineups' priors
    prior = _priors(design)
    n_players = design.spec.n_players
    ids = design.spec.players["player_id"].to_numpy()
    fit = prior_rapm.fit(design, dict(zip(ids, prior[:n_players])), dict(zip(ids, prior[n_players:])), 13037.0,
                         scale=(1.0, 1.0))
    players, context = rapm.design_blocks(design)
    y, w = np.asarray(design.y, dtype=float), np.asarray(design.w, dtype=float)
    offence, defence, explained = rl.ridge_parts(players, context, w, y, y, prior[:, None], 13037.0)
    rating = prior + np.concatenate([offence, defence]) - explained[:, 0]
    assert np.allclose(rating[:n_players], fit.ratings.offense, atol=1e-8)
    assert np.allclose(rating[n_players:], fit.ratings.defense, atol=1e-8)


def test_each_fitted_part_reads_its_own_target(design):
    players, context = rapm.design_blocks(design)
    y, w = np.asarray(design.y, dtype=float), np.asarray(design.w, dtype=float)
    n_players = design.spec.n_players
    other = y + np.random.default_rng(1).normal(0.0, 5.0, len(y))
    none = np.zeros((2 * n_players, 1))
    offence, defence, _ = rl.ridge_parts(players, context, w, y, other, none, 13037.0)
    alone_o, alone_d, _ = rl.ridge_parts(players, context, w, y, y, none, 13037.0)
    swapped_o, swapped_d, _ = rl.ridge_parts(players, context, w, other, other, none, 13037.0)
    assert np.allclose(offence, alone_o) and np.allclose(defence, swapped_d)
    assert not np.allclose(defence, alone_d) and not np.allclose(offence, swapped_o)


def _held_back(seed=2, n_players=12, n_rows=80):
    rng = np.random.default_rng(seed)
    rows, columns = [], []
    for row in range(n_rows):
        rows += [row] * 10
        columns += list(rng.choice(n_players, 5, replace=False))
        columns += list(n_players + rng.choice(n_players, 5, replace=False))
    players = sp.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(n_rows, 2 * n_players))
    levels = rl.team_levels(rng.integers(0, 3, n_rows), rng.integers(0, 3, n_rows), rng.integers(0, 2, n_rows),
                            np.zeros(n_rows))
    rated = np.ones(2 * n_players, dtype=bool)
    rated[[0, n_players + 1]] = False
    return rl.HeldBack(players=players, y=rng.normal(110.0, 40.0, n_rows), w=rng.integers(1, 12, n_rows).astype(float),
                       levels=levels, rated=rated, stand_in=(-2.0, 1.5))


def test_the_sums_give_the_tests_error_at_any_weights():
    held = _held_back()
    rng = np.random.default_rng(3)
    half = held.players.shape[1] // 2
    fitted = rng.normal(0.0, 1.0, half), rng.normal(0.0, 1.0, half)
    priors, explained = rng.normal(0.0, 1.0, (2 * half, 4)), rng.normal(0.0, 0.3, (2 * half, 4))
    within = (0.7, 1.1, 0.8, 1.6)
    sums = rl.held_back_sums(held, fitted, priors, explained, within)
    beta = np.array([1.1, 1.6, 0.5, -0.4, 0.3, 0.2])
    # the rating each player would carry, then the test's own steps: predict, take the levels out, square
    rating = np.concatenate([1.1 * fitted[0] + 0.7 * (priors[:half] @ beta[2:]) - 1.1 * (explained[:half] @ beta[2:]),
                             1.6 * fitted[1] + 0.8 * (priors[half:] @ beta[2:]) - 1.6 * (explained[half:] @ beta[2:])])
    rating[~held.rated] = 0.0
    stand_ins = np.concatenate([np.full(half, -2.0), np.full(half, 1.5)]) * (~held.rated)
    keep = held.w > 0
    miss = (held.y - held.players @ (rating + stand_ins))[keep]
    root = np.sqrt(held.w[keep])
    levels = held.levels[keep].multiply(root[:, None]).toarray()
    coef, *_ = np.linalg.lstsq(levels, miss * root, rcond=None)
    left = miss * root - levels @ coef
    assert sums.error(beta) == pytest.approx(float(left @ left) / float(held.w[keep].sum()), rel=1e-10)


def test_sums_add_up_to_the_error_over_their_stints():
    a, b = _held_back(4), _held_back(5)
    half = a.players.shape[1] // 2
    none = np.zeros((2 * half, 1))
    fitted = np.zeros(half), np.zeros(half)
    first, second = (rl.held_back_sums(held, fitted, none, none, (1.0, 1.0, 1.0, 1.0)) for held in (a, b))
    both = first + second
    beta = np.zeros(3)
    assert both.error(beta) == pytest.approx((first.yy + second.yy) / (first.sw + second.sw))


def test_team_free_is_hold_team_totals_on_many_columns():
    # a player of two teams moves by his mix of their amounts, as second_prior.hold_team_totals moves him
    team_poss = pd.DataFrame({"player_id": [1, 2, 3, 3, 4, 5], "team": [10, 10, 10, 20, 20, 20],
                              "poss": [600.0, 400.0, 100.0, 300.0, 500.0, 200.0]})
    ids = np.array([1, 2, 3, 4, 5])
    values = np.array([[1.0, -1.0], [0.0, 2.0], [0.5, 0.0], [-1.0, 1.0], [2.0, 0.5]])
    moved_ids, moved = rl.team_free(values, ids, team_poss)
    assert list(moved_ids) == list(ids)
    for column in range(2):
        line = pd.Series(values[:, column], index=ids)
        held = sp2.hold_team_totals(line, pd.Series(0.0, index=ids), team_poss)
        assert np.allclose(moved[:, column], held.reindex(ids).to_numpy())
    poss = team_poss.pivot(index="team", columns="player_id", values="poss").fillna(0.0).to_numpy()
    assert np.allclose(poss @ moved, 0.0, atol=1e-9)


def test_the_box_score_prior_keeps_each_teams_total_and_the_rest_only_split_it():
    team_poss = {side: pd.DataFrame({"player_id": [1, 2, 3, 4], "team": [10, 10, 20, 20],
                                     "poss": [600.0, 400.0, 500.0, 500.0]}) for side in ("O", "D")}
    frame = pd.DataFrame({"box": [2.0, 1.0, -1.0, -2.0], "other": [1.0, 0.0, 3.0, 1.0]},
                         index=pd.Index([1, 2, 3, 4], name="player_id"))
    columns = rl.prior_columns({"O": frame, "D": frame * 2.0}, np.array([1, 2, 3, 4, 9]), team_poss)
    assert columns.shape == (10, 4)
    poss = np.array([600.0, 400.0, 500.0, 500.0])
    box_o, other_o = columns[:4, 0], columns[:4, 1]
    assert np.average(box_o, weights=poss) == pytest.approx(0.0)
    assert box_o[0] - box_o[1] == pytest.approx(1.0) and box_o[0] - box_o[2] == pytest.approx(3.0)
    assert (other_o[:2] @ poss[:2]) == pytest.approx(0.0) and (other_o[2:] @ poss[2:]) == pytest.approx(0.0)
    assert other_o[0] - other_o[1] == pytest.approx(1.0)
    assert np.allclose(columns[5:, :2], 0.0) and np.allclose(columns[:5, 2:], 0.0)     # each end on its own half
    assert np.allclose(columns[4], 0.0) and np.allclose(columns[9], 0.0)               # no inputs: 0
    assert columns[5, 2] - columns[6, 2] == pytest.approx(2.0)


def _sums_from(beta, seed, n_rows=4000, scale=None):
    rng = np.random.default_rng(seed)
    columns = rng.normal(0.0, 1.0, (n_rows, len(beta)))
    if scale is not None:
        columns = columns * scale
    aim = columns @ beta + rng.normal(0.0, 1.0, n_rows)
    return rl.RatingSums(Q=columns.T @ columns, c=columns.T @ aim, yy=float(aim @ aim), sw=float(n_rows))


def test_the_line_solves_both_ends_together_whatever_an_inputs_unit():
    beta = np.array([1.0, 1.5, 0.5, -0.25, 0.75, 0.1])
    scale = np.array([1.0, 1.0, 1.0, 100.0, 0.01, 1.0])            # a padded team rate has a small spread, age a large
    weights = rl.coefficients([_sums_from(beta / scale, seed, scale=scale) for seed in range(3)], 5, 1e-4)
    assert weights["O"].shape == (6, 2) and weights["D"].shape == (6, 2)
    assert np.allclose(weights["O"], weights["O"][0]) and np.allclose(weights["D"], weights["D"][0])
    assert np.allclose(weights["O"][0] * scale[2:4], beta[2:4], atol=0.05)
    assert np.allclose(weights["D"][0] * scale[4:], beta[4:], atol=0.05)


def test_an_input_with_no_spread_gets_no_weight():
    beta = np.array([1.0, 1.0, 0.5, 0.0, 0.25, 0.0])
    scale = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    weights = rl.coefficients([_sums_from(beta, 7, scale=scale)], 5, 1e-4)
    assert weights["O"][0, 1] == 0.0 and abs(weights["O"][0, 0] - 0.5) < 0.05


def test_the_line_never_reads_the_rated_season_or_its_neighbors():
    beta = np.array([1.0, 1.5, 0.5, -0.25, 0.75, 0.1])
    grams = {season: _sums_from(beta, season) for season in range(2000, 2011)}
    seasons = set(grams)
    line = sp2.season_coefficients(grams, 2005, seasons, 1, 5, 1e-4)
    for near in (2004, 2005, 2006):
        changed = {**grams, near: _sums_from(beta * -3.0, near + 100)}
        assert np.array_equal(sp2.season_coefficients(changed, 2005, seasons, 1, 5, 1e-4)["O"], line["O"])
    changed = {**grams, 2007: _sums_from(beta * -3.0, 2107)}
    assert not np.allclose(sp2.season_coefficients(changed, 2005, seasons, 1, 5, 1e-4)["O"], line["O"])
    held_out = {**grams}
    del held_out[2005]
    assert sp2.season_coefficients(held_out, 2005, seasons, 0, 5, 1e-4) is not None    # the product setting
    assert sp2.season_coefficients({2005: grams[2005]}, 2005, seasons, 0, 5, 1e-4) is None


def test_the_sums_files_read_back_and_a_line_of_the_wrong_size_is_refused(tmp_path):
    from openrapm.data import second_prior_files as files
    cfg = {"_root": str(tmp_path), "paths": {"outputs": str(tmp_path)}}
    names = sp2.INPUT_SETS["context_types_team_rates_centred"]
    directory = files.grams_directory(cfg, "rating")
    directory.mkdir(parents=True)
    files.write_run(directory, {"inputs": names, "padded": True, "target": "rating"})

    def sums(size):
        return rl.RatingSums(Q=np.eye(size), c=np.arange(float(size)), yy=3.0, sw=2.0)
    files.write_rating_sums(files.rating_sums_path(directory, 2020), sums(2 + 2 * len(sp2.INPUTS)))
    with pytest.raises(SystemExit, match="inputs"):
        files.line_setup(cfg, "rating", names)
    size = 2 + 2 * len(names)
    files.write_rating_sums(files.rating_sums_path(directory, 2020), sums(size))
    assert files.line_setup(cfg, "rating", names)["target"] == "rating"
    loaded = files.load_grams(cfg, "rating")
    assert list(loaded) == [2020] and isinstance(loaded[2020], rl.RatingSums)
    assert np.array_equal(loaded[2020].c, np.arange(float(size))) and loaded[2020].sw == 2.0
