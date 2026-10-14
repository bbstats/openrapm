"""The decomposition pages' model code (src/openrapm/decomposition.py): which rows a ridge coefficient comes from.

The claim everything rests on: step 3's games' part, a ridge coefficient, is a sum over rows of h_r z_r, so the rows'
groups add up to it exactly.  It is checked on a simulated league two ways: the groups sum to the coefficient, and
a refit with one group's outcomes made average moves the coefficient by exactly that group's piece.
"""
import numpy as np
import pandas as pd
import pytest

from openrapm import decomposition, prior_rapm, rapm


def _teams(sim: dict, season: int) -> pd.Series:
    """(game id, player id) -> team id of one simulated season, as the box score files give it."""
    box, ps = sim["box"], sim["truth"]["ps"]
    box = box[box.season == season].merge(ps[ps.season == season][["player_id", "team"]], on="player_id")
    return box.set_index(["game_id", "player_id"]).team


def _prior(design, penalty=3000.0):
    """A prior with the shape the games like: a plain RAPM of the design, both sides."""
    base = rapm.SeasonSums(0.0).add(2002, design).ratings((), (penalty, penalty, 0.0))
    return dict(zip(base.player_id, base.offense)), dict(zip(base.player_id, base.defense))


@pytest.fixture(scope="module", params=["with_playoffs", "regular_season_only"])
def world(request, synthetic_league, synthetic_design):
    """2002 of a three-season simulated league: (design, fit, keys).  Without playoffs the playoff columns are all
    zeros, so the solve takes rapm.solved_ridge's jitter path, like the real 2020."""
    changes = dict(po_series_per_season=2) if request.param == "with_playoffs" else {}
    sim = synthetic_league(n_seasons=3, seed=4, **changes)
    design = synthetic_design(sim, [2002])
    prior_o, prior_d = _prior(design)
    fit = decomposition.refit(design, prior_o, prior_d, penalty=3000.0)
    lineup_o, lineup_d = decomposition.lineups(fit.players, fit.n_players)
    keys = decomposition.row_keys(design, lineup_o, lineup_d, _teams(sim, 2002))
    return design, fit, keys, request.param == "regular_season_only"


# ------------------------------------------------------------------------------------------ the rows' teams
def test_a_side_takes_the_team_of_its_first_player_the_box_score_lists():
    teams = pd.Series([10, 20, 20], index=pd.MultiIndex.from_tuples([("g1", 1), ("g1", 2), ("g2", 3)]))
    game_ids = np.array(["g1", "g1", "g2"])
    players = np.array([[9, 1], [2, 1], [7, 8]])         # row 1: first player unknown; row 3: nobody known
    assert decomposition.side_team(teams, game_ids, players).tolist() == [10, 20, -1]


def test_team_codes_give_an_unplaced_side_a_code_of_its_own():
    rows = {"team_att": np.array([10, 10, -1]), "team_def": np.array([20, 20, 20]),
            "season": np.array([2026, 2026, 2026]), "game_ids": np.array(["g1", "g2", "g2"])}
    codes = decomposition.team_codes(rows)
    assert codes["ts_att"][0] == codes["ts_att"][1] != codes["ts_att"][2]       # one team-season, one unknown
    assert codes["gt_att"][0] != codes["gt_att"][1]                             # two team-games
    assert codes["unplaced"] == pytest.approx(1 / 3)


def test_lineups_read_off_z_are_the_designs_own(world):
    design, fit, _, _ = world
    lineup_o, lineup_d = decomposition.lineups(fit.players, fit.n_players)
    assert np.array_equal(lineup_o, design.parts["lineup_o"]) and np.array_equal(lineup_d, design.parts["lineup_d"])
    assert (lineup_o[:, 1:] > lineup_o[:, :-1]).all() and lineup_o.shape == (len(fit.w), 5)


def test_every_row_of_the_simulated_season_has_both_teams(world):
    _, _, keys, _ = world
    assert keys["unplaced"] == 0.0


# --------------------------------------------------------------------------------------- the groups of rows
def test_the_groups_of_rows_for_a_player_who_was_traded_and_missed_games():
    """Teams A, B, C, D.  He plays for A in game g1 (rows 0-3), misses A's game g2, is traded to B and plays g3, misses
    B's game g4, and g5 is two other teams.  Each game has two rows: each team attacks once."""
    games = [("g1", 10, 20), ("g1", 20, 10), ("g1", 10, 20), ("g1", 20, 10),      # rows 0-3: 0, 1 he is on court
             ("g2", 10, 30), ("g2", 30, 10),                                         # rows 4-5: A's game he missed
             ("g3", 20, 30), ("g3", 30, 20),                                         # rows 6-7: he plays for B
             ("g4", 20, 40), ("g4", 40, 20),                                         # rows 8-9: B's game he missed
             ("g5", 30, 40), ("g5", 40, 30)]                                         # rows 10-11: nobody of his
    keys = decomposition.team_codes(dict(
        team_att=np.array([g[1] for g in games]), team_def=np.array([g[2] for g in games]),
        season=np.full(len(games), 2026), game_ids=np.array([g[0] for g in games])))
    rows_off, rows_def = np.array([0, 6]), np.array([1, 7])       # A attacks with him (row 0), B attacks with him (6)
    groups = decomposition.row_groups(rows_off, rows_def, keys, len(games))
    as_rows = {name: np.flatnonzero(mask).tolist() for name, mask in groups.items()}
    assert as_rows == {"on": [0, 1, 6, 7], "without": [2, 3], "missed": [4, 5, 8, 9], "off_att": [2], "off_def": [3]}


# ------------------------------------------------------------------------------------------ the split itself
def test_the_refit_is_step_3s_own_fit(world):
    """Same system, same solver: the deviations are bit for bit the ones prior_rapm.fit returns."""
    design, fit, _, _ = world
    prior_o, prior_d = _prior(design)
    step_3 = prior_rapm.fit(design, prior_o, prior_d, penalty=3000.0)
    assert np.array_equal(fit.theta[:2 * fit.n_players], step_3.deviation)
    assert np.array_equal(fit.theta[-2:], step_3.prior_scale)


def test_jitter_is_reported_only_where_the_solve_needed_it(world):
    _, fit, _, regular_season_only = world
    assert fit.retried == regular_season_only


@pytest.mark.parametrize("side", ["off", "def"])
def test_the_three_groups_add_up_to_the_games_part(world, side):
    _, fit, keys, regular_season_only = world
    n = fit.n_players
    frame, misses = decomposition.split_games_part(fit, keys, side)
    sign, first = (1.0, 0) if side == "off" else (-1.0, n)
    assert np.array_equal(frame.player_id, fit.player_ids)
    assert np.allclose(frame.u, sign * fit.theta[first:first + n], atol=1e-12)           # positive good on both sides
    assert np.allclose(frame.on_court + frame.off_court + frame.team_sos, frame.u, atol=1e-12)
    assert misses["sum_hy"] < 1e-9                                                       # the coefficient is sum h y
    assert misses["leftover"] < (1e-6 if regular_season_only else 1e-9)                  # B'h = 0, but for the jitter
    assert misses["retried"] == regular_season_only
    assert frame[["on_court", "off_court", "team_sos"]].abs().to_numpy().max() > 0.01    # the groups are not all zero


@pytest.mark.parametrize("side", ["off", "def"])
def test_a_refit_with_a_groups_outcomes_made_average_moves_the_coefficient_by_its_piece(world, side):
    """The pieces' meaning: set a group's outcomes to what context and priors predict (z = 0 there) and refit; the
    player's coefficient falls by exactly the piece.  The refit never uses h, so this checks h too."""
    _, fit, keys, _ = world
    n = fit.n_players
    frame, _ = decomposition.split_games_part(fit, keys, side)
    z = fit.y - fit.unpenalised @ fit.theta[2 * n:]
    first, sign = (0, 1.0) if side == "off" else (n, -1.0)
    by_column = fit.players.tocsc()
    rhs = fit.players.T @ (fit.w * fit.y)
    rhs = np.concatenate([rhs, fit.unpenalised.T @ (fit.w * fit.y)])
    for j in np.argsort(-frame.u.abs().to_numpy())[:6]:
        groups = decomposition.row_groups(by_column.indices[by_column.indptr[j]:by_column.indptr[j + 1]],
                                          by_column.indices[by_column.indptr[n + j]:by_column.indptr[n + j + 1]],
                                          keys, len(fit.w))
        for name, mask in (("on_court", groups["on"]), ("off_court", groups["without"] | groups["missed"])):
            moved = np.where(mask, z, 0.0)                                  # make those rows' outcomes average
            shift = np.concatenate([fit.players.T @ (fit.w * moved), fit.unpenalised.T @ (fit.w * moved)])
            theta = np.linalg.solve(fit.matrix, rhs - shift)
            assert sign * (fit.theta[first + j] - theta[first + j]) == pytest.approx(frame[name][j], abs=1e-9), \
                (name, j)


def test_the_weights_are_orthogonal_to_the_unpenalised_columns(world):
    _, fit, _, regular_season_only = world
    unit = np.zeros(fit.matrix.shape[0])
    unit[3] = 1.0
    h = decomposition.row_weights(fit, np.linalg.solve(fit.matrix, unit))
    assert h @ fit.y == pytest.approx(fit.theta[3], abs=1e-9)
    assert np.abs(fit.unpenalised.T @ h).max() < (1e-5 if regular_season_only else 1e-9)
