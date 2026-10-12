"""His team with him on the court and off it (src/openrapm/on_court.py): a planted on-court effect is found (and shared
with his teammates), the on-court ratings are centred and padded toward the level, and the off-court possessions are
his team's possessions in his games without him, counted by hand."""
import numpy as np
import pytest
import scipy.sparse as sp

from openrapm.box_rates import lineups
from openrapm.features import OFF_COURT, ON_COURT
from openrapm.on_court import offcourt_rates, on_court, oncourt_rates
from openrapm.simulate import simulate


def _planted(seed=0, players=40, rows=6000):
    """Random lineups of `players`; player 3 adds 4 points per 100 on offence, player 7 takes 3 off on defence."""
    rng = np.random.default_rng(seed)
    on_offence = np.array([rng.choice(players, 5, replace=False) for _ in range(rows)])
    on_defence = np.array([rng.choice(players, 5, replace=False) for _ in range(rows)])
    row = np.repeat(np.arange(rows), 5)
    Zo = sp.csr_matrix((np.ones(5 * rows), (row, on_offence.ravel())), shape=(rows, players))
    Zd = sp.csr_matrix((np.ones(5 * rows), (row, on_defence.ravel())), shape=(rows, players))
    effect_o, effect_d = np.zeros(players), np.zeros(players)
    effect_o[3], effect_d[7] = 4.0, 3.0
    w = rng.uniform(5, 40, rows)
    r = Zo @ effect_o - Zd @ effect_d + rng.normal(0, 12, rows) / np.sqrt(w / 20)
    return Zo, Zd, r, w


@pytest.fixture(scope="module")
def season(synthetic_design):
    sim = simulate(n_seasons=1, n_teams=6, players_per_team=10, games_per_season=40, stints_per_game=(20, 30), seed=3)
    return synthetic_design(sim)


def test_on_court_finds_the_planted_players_and_credits_their_teammates_too():
    Zo, Zd, r, w = _planted()
    mean_o, poss_o, mean_d, poss_d = on_court(Zo, Zd, r, w)
    assert np.argmax(mean_o) == 3 and mean_o[3] > 2.0
    assert abs(poss_o.sum() - 5 * w.sum()) < 1e-6 * w.sum() and abs(poss_d.sum() - 5 * w.sum()) < 1e-6 * w.sum()
    assert np.delete(mean_o, 3).mean() > 0.05                 # everyone who shared the court with 3 gets some credit
    assert np.argmin(mean_d) == 7 and mean_d[7] < -1.5         # in the offence's sign: fewer points with 7 defending


def test_the_on_court_ratings_are_centred_and_padded_toward_the_level(season):
    on = oncourt_rates(season, season)
    assert list(on.columns) == ON_COURT
    n_players = season.spec.n_ps
    raw, weight, raw_d, weight_d = on_court(season.X[:, :n_players], season.X[:, n_players:2 * n_players],
                                            season.y, season.w)
    np.testing.assert_allclose(on.on_court_off_possessions, weight)
    np.testing.assert_allclose(on.on_court_def_possessions, weight_d)
    for padded, rating, n in ((on.on_court_scored_per100, raw, weight), (on.on_court_allowed_per100, raw_d, weight_d)):
        centred = rating - np.average(rating, weights=n)
        played = n > 0
        assert np.all(np.abs(padded.to_numpy()[played]) <= np.abs(centred[played]) + 1e-12)
        assert np.all(padded.to_numpy()[played] * centred[played] >= 0)               # pulled toward 0, never past


def test_the_off_court_possessions_are_his_team_in_his_games_without_him(season):
    off = offcourt_rates(season, season)
    assert list(off.columns) == OFF_COURT
    on_offence, on_defence = lineups(season)
    game = season.rows["game_idx"].to_numpy()
    home_offence = season.rows["is_home_off"].to_numpy(dtype=bool)
    expected = np.zeros(season.spec.n_ps)
    for player in range(season.spec.n_ps):
        attacking, defending = (on_offence == player).any(axis=1), (on_defence == player).any(axis=1)
        home_games = np.union1d(game[attacking & home_offence], game[defending & ~home_offence])
        away_games = np.union1d(game[attacking & ~home_offence], game[defending & home_offence])
        team_attacking = ((np.isin(game, home_games) & home_offence) | (np.isin(game, away_games) & ~home_offence))
        expected[player] = season.w[team_attacking].sum() - season.w[attacking].sum()
    np.testing.assert_allclose(off.off_court_off_possessions, np.maximum(expected, 0.0), rtol=1e-12, atol=1e-9)
    assert (off.off_court_def_possessions > 0).any()


def test_two_designs_of_different_players_are_refused(season, synthetic_design):
    sim = simulate(n_seasons=1, n_teams=4, players_per_team=9, games_per_season=10, stints_per_game=(10, 12), seed=4)
    other = synthetic_design(sim)
    with pytest.raises(ValueError, match="player layout"):
        oncourt_rates(season, other)
    with pytest.raises(ValueError, match="player layout"):
        offcourt_rates(season, other)
