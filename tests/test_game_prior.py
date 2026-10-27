"""The box-score prior game by game (game_prior.py) and step 3 reading it (prior_rapm.py's `game_tables`).

A prior that changes from game to game enters the ridge through the same sparse product as a season's prior, only
gathered at each row's game; with the same value in every game it must be the shipped fit, and the season value it
reports must be the ridge's own possession-weighted average credit.  The swap part of the prior spread over the games
must average back to the second prior exactly.
"""
import numpy as np
import pandas as pd
import pytest

from openrapm import game_prior, prior_rapm, rapm


@pytest.fixture(scope="module")
def design(synthetic_league, synthetic_design):
    """2002, the middle of three seasons of a league whose talent persists."""
    return synthetic_design(synthetic_league(n_seasons=3, seed=4), [2002])


def _prior(wd, penalty=3000.0):
    sums = rapm.SeasonSums(0.0).add(2002, wd)
    base = sums.ratings((), (penalty, penalty, 0.0))
    return dict(zip(base.player_id, base.offense)), dict(zip(base.player_id, base.defense))


def _constant_tables(wd, prior_o, prior_d):
    """Each player's season prior in every game of the design."""
    ids = wd.spec.players.player_id.to_numpy()
    n_games = int(wd.games.game_idx.max()) + 1
    return (np.tile(prior_rapm.aligned_prior(prior_o, ids), (n_games, 1)),
            np.tile(prior_rapm.aligned_prior(prior_d, ids), (n_games, 1)))


def test_one_value_in_every_game_is_the_shipped_fit(design):
    prior_o, prior_d = _prior(design)
    shipped = prior_rapm.fit(design, prior_o, prior_d, penalty=3000.0)
    by_game = prior_rapm.fit(design, {}, {}, penalty=3000.0, game_tables=_constant_tables(design, prior_o, prior_d))
    np.testing.assert_allclose(by_game.deviation, shipped.deviation, atol=1e-9)
    np.testing.assert_allclose(by_game.prior_scale, shipped.prior_scale, atol=1e-9)
    for column in ("offense", "defense", "prior_offense", "prior_defense"):
        np.testing.assert_allclose(by_game.ratings[column], shipped.ratings[column], atol=1e-9)


def test_the_season_value_is_the_ridges_average_credit(design):
    """Each row credits a player s x his game's prior + u; averaged over his rows by their weights that is the rating
    the fit reports, and the reported prior is s times his average game prior."""
    rng = np.random.default_rng(3)
    prior_o, prior_d = _prior(design)
    tables = tuple(table + rng.normal(0.0, 0.8, table.shape) for table in _constant_tables(design, prior_o, prior_d))
    got = prior_rapm.fit(design, {}, {}, penalty=3000.0, game_tables=tables)
    players, _ = rapm.design_blocks(design)
    n = design.spec.n_players
    game = design.rows.game_idx.to_numpy()
    w = design.w
    for side, (block, table, scale) in enumerate(((players[:, :n], tables[0], got.prior_scale[0]),
                                                  (players[:, n:], tables[1], got.prior_scale[1]))):
        cells = block.tocoo()
        credit = scale * table[game[cells.row], cells.col] + got.deviation[side * n + cells.col]
        weight = w[cells.row]
        average = np.bincount(cells.col, weights=weight * credit, minlength=n) / np.maximum(
            np.bincount(cells.col, weights=weight, minlength=n), 1e-12)
        played = np.bincount(cells.col, minlength=n) > 0
        column = ("offense", "defense")[side]
        np.testing.assert_allclose(got.ratings[column].to_numpy()[played], average[played], atol=1e-9)


def test_the_lineup_sums_are_the_sparse_product_gathered_by_game(design):
    rng = np.random.default_rng(5)
    players, _ = rapm.design_blocks(design)
    n = design.spec.n_players
    n_games = int(design.games.game_idx.max()) + 1
    tables = (rng.normal(size=(n_games, n)), rng.normal(size=(n_games, n)))
    game = design.rows.game_idx.to_numpy()
    sums_o, sums_d = prior_rapm.game_lineup_sums(players, game, tables, n)
    dense = players.toarray()
    expected_o = (dense[:, :n] * tables[0][game]).sum(axis=1)
    expected_d = (dense[:, n:] * tables[1][game]).sum(axis=1)
    np.testing.assert_allclose(sums_o, expected_o, atol=1e-12)
    np.testing.assert_allclose(sums_d, expected_d, atol=1e-12)


def _game_rows(seed=0, players=6, games=12):
    rng = np.random.default_rng(seed)
    rows = []
    for side in ("O", "D"):
        for pid in range(players):
            for g in range(games):
                if rng.random() < 0.15:
                    continue
                rows.append(dict(player_id=pid, side=side, game_id=f"g{g}", poss_off_game=float(rng.integers(10, 80)),
                                 poss_def_game=float(rng.integers(10, 80)), box_game=float(rng.normal())))
    return pd.DataFrame(rows)


def test_the_swap_part_spread_over_the_games_averages_back_to_the_second_prior():
    rows = _game_rows()
    box = game_prior.season_average(rows, "box_game")
    rng = np.random.default_rng(9)
    second = {side: {pid: value + rng.normal() for pid, value in box[side].items()} for side in ("O", "D")}
    weights = {side: {pid: 0.6 + 0.1 * pid for pid in box[side]} for side in ("O", "D")}
    rows["prior_game"] = game_prior.per_game_second(rows, "box_game", second, box, weights)
    back = game_prior.season_average(rows, "prior_game")
    for side in ("O", "D"):
        for pid, value in second[side].items():
            assert back[side][pid] == pytest.approx(value, abs=1e-12)


def test_game_tables_fill_a_missing_game_with_the_players_fallback(design):
    ids = design.spec.players.player_id.to_numpy()
    games = design.games
    first, second = str(games.game_id.iloc[0]), str(games.game_id.iloc[1])
    rows = pd.DataFrame({"player_id": [ids[0], ids[0]], "side": ["O", "D"], "game_id": [first, second],
                         "value": [1.5, -2.0]})
    fallback = {"O": {ids[0]: 0.25}, "D": {ids[0]: -0.5}}
    table_o, table_d = game_prior.game_tables(design, rows, "value", fallback)
    g0 = int(games.game_idx.iloc[0])
    g1 = int(games.game_idx.iloc[1])
    assert table_o[g0, 0] == 1.5 and table_d[g1, 0] == -2.0
    assert table_o[g1, 0] == 0.25 and table_d[g0, 0] == -0.5
    assert (table_o[:, 1:] == 0.0).all()
