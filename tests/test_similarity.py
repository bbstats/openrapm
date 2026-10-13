"""The similarity page's model (src/openrapm/similarity.py): the standardised inputs, the distance, the two lists, the
own-season ranks and the inputs themselves, on numbers whose answers are known by construction."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from openrapm.box_prior import rated_rows
from openrapm.features import BIO, CAREER, PANEL_INPUTS, ROLE, SHIPPED
from openrapm.similarity import (
    TABLE_COLUMNS,
    TABLE_INPUTS,
    closest,
    distances,
    own_season_ranks,
    player_seasons,
    side_inputs,
    similarity_inputs,
    standardise,
)


def _all_rows(z):
    return distances(z, slice(0, len(z)))


# ------------------------------------------------------------------------------ the distance
def test_the_docstring_example():
    d = _all_rows(np.array([[1.0, 0.0], [0.4, 0.8], [-1.0, 0.5]]))
    assert d[0, 1] == pytest.approx(1.0) and round(d[0, 2], 2) == 2.06


def test_standardised_inputs_have_mean_0_and_spread_1_and_a_flat_input_stays_0():
    inputs = pd.DataFrame({"a": [1.0, 2.0, 3.0, 6.0], "b": [100.0, 300.0, 200.0, 400.0], "flat": 7.0})
    z = standardise(inputs)
    assert np.allclose(z[:, :2].mean(axis=0), 0.0) and np.allclose(z[:, :2].std(axis=0), 1.0)
    assert np.array_equal(z[:, 2], np.zeros(4))


def test_the_distance_is_euclidean_symmetric_to_the_last_bit_and_zero_on_itself():
    z = np.random.default_rng(1).normal(size=(30, 6))
    whole = _all_rows(z)
    assert np.allclose(whole, cdist(z, z))
    assert np.array_equal(whole, whole.T) and np.array_equal(np.diagonal(whole), np.zeros(30))
    assert np.array_equal(distances(z, slice(7, 19)), whole[7:19])          # a block is those rows of the whole


# ------------------------------------------------------------------------------ the two lists
HAND = np.array([[0.0, 1.0, 0.5, 2.0],
                 [1.0, 0.0, 3.0, 0.2],
                 [0.5, 3.0, 0.0, 1.0],
                 [2.0, 0.2, 1.0, 0.0]])
HAND_PLAYERS = np.array([1, 1, 2, 3])


def test_the_lists_on_a_hand_case():
    lists = closest(lambda rows: HAND[rows], HAND_PLAYERS, matches=2)
    assert lists["seasons"].tolist() == [[2, 3], [3, 2], [0, 3], [1, 2]]
    assert lists["season_distances"][0].tolist() == [0.5, 2.0]
    # row 2 (player 2): player 1 at his nearer row 0 (0.5, not row 1's 3.0), then player 3
    assert lists["players"].tolist() == [[2, 3], [3, 2], [0, 3], [1, 2]]
    assert lists["player_distances"][2].tolist() == [0.5, 1.0]


def test_ties_go_to_the_earlier_row_and_the_player_seen_first():
    tied = np.array([[0.0, 1.0, 1.0, 1.0],
                     [1.0, 0.0, 2.0, 2.0],
                     [1.0, 2.0, 0.0, 2.0],
                     [1.0, 2.0, 2.0, 0.0]])
    lists = closest(lambda rows: tied[rows], np.array([1, 2, 2, 3]), matches=3)
    assert lists["seasons"][0].tolist() == [1, 2, 3]
    assert lists["players"][0, :2].tolist() == [1, 3]      # player 2 at row 1 (not 2), then player 3


def test_no_list_holds_the_rows_own_player_and_each_player_sits_at_his_nearest_row():
    rng = np.random.default_rng(5)
    n = 37
    z = rng.normal(size=(n, 4))
    players = rng.integers(0, 9, n)
    whole = _all_rows(z)
    lists = closest(lambda rows: distances(z, rows), players, matches=6, block_rows=5)
    for row in range(n):
        assert not np.isin(players[lists["seasons"][row]], players[row]).any()
        assert np.all(np.diff(lists["season_distances"][row]) >= 0)
        chosen = players[lists["players"][row]]
        assert len(set(chosen.tolist())) == 6 and players[row] not in chosen
        for position, distance in zip(lists["players"][row], lists["player_distances"][row]):
            mine = np.flatnonzero(players == players[position])
            assert distance == whole[row, mine].min() and position == mine[np.argmin(whole[row, mine])]


def test_the_block_size_changes_nothing():
    rng = np.random.default_rng(6)
    z = rng.normal(size=(23, 3))
    players = rng.integers(0, 7, 23)
    small = closest(lambda rows: distances(z, rows), players, matches=4, block_rows=3)
    whole = closest(lambda rows: distances(z, rows), players, matches=4, block_rows=100)
    for key in small:
        assert np.array_equal(small[key], whole[key])


def test_own_season_ranks_on_the_hand_case():
    ranks = own_season_ranks(lambda rows: HAND[rows], HAND_PLAYERS)
    # row 0: his other season (row 1) is at 1.0, and only row 2 (0.5) is nearer; row 1: only row 3 (0.2)
    assert ranks[:2].tolist() == [2.0, 2.0]
    assert np.isnan(ranks[2:]).all()          # players 2 and 3 have one season each


def test_a_tie_with_his_own_season_counts_in_his_favour():
    tied = np.array([[0.0, 1.0, 1.0],
                     [1.0, 0.0, 3.0],
                     [1.0, 3.0, 0.0]])
    assert own_season_ranks(lambda rows: tied[rows], np.array([1, 1, 2]))[0] == 1.0


# ------------------------------------------------------------------------------ the pool and its inputs
SHARED = set(ROLE) | set(CAREER) | set(BIO)          # equal on a player's two rows, as in the real panel


def _panel(players=(11, 12, 13), seasons=(2001, 2002, 2003), seed=7):
    """A made-up season panel: one offence and one defence row per player-season, every panel input filled; the
    role, career and body columns equal on both rows, the rates different."""
    rng = np.random.default_rng(seed)
    rows = []
    for player in players:
        for season in seasons:
            shared = {name: rng.uniform(1.0, 3.0) for name in SHARED}
            for side in ("O", "D"):
                own = {name: rng.uniform(1.0, 3.0) for name in PANEL_INPUTS if name not in SHARED}
                rows.append({"side": side, "player_id": player, "season": season,
                             "possessions": rng.uniform(1000, 3000), **own, **shared})
    return pd.DataFrame(rows)


def _table(panel, seed=8):
    """A made-up ratings table for the panel's player-seasons."""
    rng = np.random.default_rng(seed)
    keys = panel.loc[panel.side == "O", ["player_id", "season", "possessions"]].reset_index(drop=True)
    table = keys.rename(columns={"possessions": "poss_off"}).assign(
        poss_def=panel.loc[panel.side == "D", "possessions"].to_numpy(),
        player_name=[f"Player {p}" for p in keys.player_id])
    for name in TABLE_INPUTS:
        table[name] = rng.normal(size=len(table))
    return table


def test_the_pool_is_filtered_and_in_season_then_player_order():
    table = pd.DataFrame({"player_id": [5, 3, 4, 3], "season": [2002, 2002, 2001, 2001],
                          "player_name": list("abcd"), "poss_off": [1500.0, 999.0, 2000.0, 1000.0],
                          "poss_def": 1000.0, **{name: 0.0 for name in TABLE_INPUTS}})
    rows = player_seasons(table, 1000, seasons=[2001, 2002])
    assert list(rows.columns) == TABLE_COLUMNS
    assert rows[["season", "player_id"]].values.tolist() == [[2001, 3], [2001, 4], [2002, 5]]
    assert len(player_seasons(table, 1000, seasons=[2002])) == 1
    with pytest.raises(ValueError, match="twice"):
        player_seasons(pd.concat([table, table]), 1000, seasons=[2001, 2002])
    with pytest.raises(KeyError, match="u_off"):
        player_seasons(table.drop(columns="u_off"), 1000, seasons=[2001])


def test_all_seasons_at_once_equal_one_rated_season_at_a_time():
    panel = _panel()
    together = side_inputs(panel, "D", SHIPPED["D"]).sort_values(["season", "player_id"]).reset_index(drop=True)
    one_by_one = pd.concat([rated_rows(panel[(panel.side == "D") & (panel.season == s)], SHIPPED["D"])
                            for s in (2001, 2002, 2003)])
    one_by_one = one_by_one[together.columns].sort_values(["season", "player_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(together, one_by_one)


def test_the_inputs_are_the_prior_lists_the_possessions_and_the_table():
    panel = _panel()
    rows = player_seasons(_table(panel), 0, seasons=[2001, 2002, 2003])
    inputs = similarity_inputs(rows, panel, SHIPPED)
    both = [name for name in SHIPPED["O"] if name in SHIPPED["D"]]
    differ = [name for name in both if name not in SHARED]          # the rates, derived per side
    assert sorted(differ) == ["points_per100", "shots_per100", "steals_per100", "true_shooting_pct"]
    expected = ([f"{n}_off" if n in differ else n for n in SHIPPED["O"]]
                + [f"{n}_def" if n in differ else n for n in SHIPPED["D"] if n in differ or n not in SHIPPED["O"]]
                + ["poss_off", "poss_def", *TABLE_INPUTS])
    assert list(inputs.columns) == expected and len(expected) == 34
    defence = side_inputs(panel, "D", SHIPPED["D"]).set_index(["player_id", "season"])
    first = (rows.player_id[0], rows.season[0])
    assert inputs.true_shooting_pct_def[0] == defence.loc[first, "true_shooting_pct"]
    assert inputs.poss_def[0] == defence.loc[first, "row_possessions"]
    assert inputs.u_off.tolist() == rows.u_off.tolist()


def test_a_pool_row_without_a_side_row_is_refused():
    panel = _panel()
    rows = player_seasons(_table(panel), 0, seasons=[2001, 2002, 2003])
    gone = panel[~((panel.side == "D") & (panel.player_id == 12) & (panel.season == 2002))]
    with pytest.raises(ValueError, match="no D panel row"):
        similarity_inputs(rows, gone, SHIPPED)
