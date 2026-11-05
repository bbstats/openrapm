"""bias.py on made-up players: the consensus, the width match, the slope per standard deviation and its standard
error, each on numbers small enough to work by hand (the module's own example), then the properties the page
relies on: a stat's total is its offense plus its defense, and no number moves when only a scale or a level does.
"""
import numpy as np
import pandas as pd
import pytest

from openrapm import bias
from openrapm.features import BOX_RATES

WEIGHTS = np.array([100.0, 300.0, 300.0, 100.0])
BLOCKS = np.array([0.5, 0.5, 1.5, 1.5])             # weighted mean 1.0, standard deviation 0.5: z = -1, -1, +1, +1
OURS = {"offense": np.array([-3.0, 1.0, 1.0, 5.0]), "defense": np.array([1.0, -1.0, 0.0, 2.0])}
THEIRS = {"offense": np.array([2.0, 0.0, -1.0, 1.0]), "defense": np.array([-2.0, 2.0, -2.0, 2.0])}


def test_the_consensus_is_the_owners_weighted_mean_over_the_players_all_five_list():
    """A player at 3.0, 1.0, 3.0, 2.0 and 0.0 reads 0.9 + 0.2 + 0.6 + 0.3 + 0.0 = 2.0; one a system lacks is out."""
    values = {"xrapm": 3.0, "rapm_3y": 1.0, "loki": 3.0, "darko": 2.0, "pulse": 0.0}
    tables = {key: pd.DataFrame({"player_id": [7, 8], "offense": [value, 1.0], "defense": [-value, 1.0]})
              for key, value in values.items()}
    tables["pulse"] = tables["pulse"][tables["pulse"].player_id == 7]
    blend = bias.consensus(tables)
    assert list(blend.columns) == ["player_id", "offense", "defense"] and list(blend.player_id) == [7]
    assert blend.offense[0] == pytest.approx(2.0) and blend.defense[0] == pytest.approx(-2.0)


def test_the_weights_are_the_owners_and_add_up_to_one():
    assert bias.CONSENSUS_WEIGHTS == {"xrapm": 0.30, "rapm_3y": 0.20, "loki": 0.20, "darko": 0.15, "pulse": 0.15}
    assert sum(bias.CONSENSUS_WEIGHTS.values()) == pytest.approx(1.0)
    from openrapm.data import public_ratings
    assert set(bias.CONSENSUS_WEIGHTS) == set(public_ratings.SYSTEMS)


def test_a_missing_system_or_weights_that_do_not_add_to_one_are_refused():
    table = pd.DataFrame({"player_id": [1], "offense": [0.0], "defense": [0.0]})
    with pytest.raises(ValueError, match="missing"):
        bias.consensus({"xrapm": table})
    with pytest.raises(ValueError, match="not 1"):
        bias.consensus({"a": table, "b": table}, {"a": 0.5, "b": 0.6})


def test_the_width_match_multiplies_openrapm_to_the_consensus_spread():
    """Offense: OpenRAPM's standard deviation is 2 and the consensus's 1, so the multiplier is 0.5."""
    gap, multiplier = bias.width_match(OURS["offense"], THEIRS["offense"], WEIGHTS)
    assert multiplier == pytest.approx(0.5) and gap == pytest.approx([-4.0, 0.0, 1.0, 1.0])
    gap, multiplier = bias.width_match(OURS["defense"], THEIRS["defense"], WEIGHTS)
    assert multiplier == pytest.approx(2.0) and gap == pytest.approx([4.0, -4.0, 2.0, 2.0])


def test_the_slope_and_its_robust_standard_error_on_the_modules_example():
    offense, _ = bias.width_match(OURS["offense"], THEIRS["offense"], WEIGHTS)
    defense, _ = bias.width_match(OURS["defense"], THEIRS["defense"], WEIGHTS)
    assert bias.slope_per_sd(BLOCKS, offense, WEIGHTS) == pytest.approx((1.0, 0.75))
    assert bias.slope_per_sd(BLOCKS, defense, WEIGHTS) == pytest.approx((2.0, 1.5))
    assert bias.slope_per_sd(BLOCKS, offense + defense, WEIGHTS) == pytest.approx((3.0, 0.75))
    # the classical formula, which reads the possessions as precisions, would say 0.866 on offense


def test_a_stat_with_no_spread_has_no_slope():
    slope, error = bias.slope_per_sd(np.ones(4), np.array([1.0, -1.0, 2.0, -2.0]), WEIGHTS)
    assert np.isnan(slope) and np.isnan(error)


def _players(n=60, seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"player_id": np.arange(n), "weight": rng.integers(60, 700, n).astype(float),
                          "rating_off": rng.normal(0, 2, n), "rating_def": rng.normal(0, 1, n),
                          "consensus_offense": rng.normal(0, 1.5, n), "consensus_defense": rng.normal(0, 1.2, n)})
    for stat in bias.STATS:
        frame[stat] = rng.normal(0, 1, n)
    return frame


def test_the_table_has_a_row_per_stat_and_every_total_is_offense_plus_defense():
    table, multipliers = bias.bias_table(_players())
    assert list(table.stat) == bias.STATS and len(bias.STATS) == 15
    assert bias.STATS[:13] == BOX_RATES and bias.STATS[13:] == ["on_court_plus_minus_per100",
                                                                 "off_court_plus_minus_per100"]
    assert np.allclose(table.total, table.offense + table.defense)
    assert set(multipliers) == {"offense", "defense"} and all(value > 0 for value in multipliers.values())
    assert (table[["offense_se", "defense_se", "total_se"]] > 0).all().all()


def test_only_shape_moves_a_slope_never_a_level_or_a_scale():
    """A system that rates everyone 1 higher, a stat counted in other units, and an OpenRAPM three times as wide all
    leave every number where it was: the gap is taken from the means, the stat is standardized, the width matched."""
    players = _players()
    base, _ = bias.bias_table(players)
    moved = players.assign(consensus_offense=players.consensus_offense + 1.0,
                           blocks_per100=players.blocks_per100 * 7.0 + 2.0,
                           rating_off=players.rating_off * 3.0, rating_def=players.rating_def * 3.0)
    again, multipliers = bias.bias_table(moved)
    slopes = ["offense", "defense", "total", "offense_se", "defense_se", "total_se"]
    assert np.allclose(base[slopes], again[slopes])
    assert multipliers["offense"] == pytest.approx(bias.bias_table(players)[1]["offense"] / 3.0)


def test_a_gap_built_from_one_stat_is_read_back_on_that_stat():
    """OpenRAPM = the consensus plus 0.5 per standard deviation of blocks, at the same spread on offense as the
    consensus: the blocks row reads 0.5 on offense, 0 on defense."""
    players = _players(n=400, seed=11)
    weights = players.weight.to_numpy()
    z = (players.blocks_per100 - bias.weighted_mean(players.blocks_per100, weights)) \
        / bias.weighted_sd(players.blocks_per100, weights)
    players["rating_def"] = players.consensus_defense
    lifted = players.consensus_offense + 0.5 * z
    # brought back to the consensus's spread, so the width match has nothing to do and the lean is all that is left
    players["rating_off"] = lifted * bias.weighted_sd(players.consensus_offense, weights) / bias.weighted_sd(lifted,
                                                                                                         weights)
    table, multipliers = bias.bias_table(players)
    row = table.set_index("stat").loc["blocks_per100"]
    assert multipliers["offense"] == pytest.approx(1.0) and multipliers["defense"] == pytest.approx(1.0)
    assert row.defense == pytest.approx(0.0, abs=1e-12) and 0.4 < row.offense < 0.5
    assert row.total == pytest.approx(row.offense)


def test_a_players_stat_is_the_mean_of_his_two_panel_rows_and_plus_minus_is_scored_minus_allowed():
    court = {"on_court_scored_per100": [4.0, 4.0], "on_court_allowed_per100": [-1.0, -1.0],
             "off_court_scored_per100": [-2.0, -2.0], "off_court_allowed_per100": [0.5, 0.5]}
    panel = pd.DataFrame({"player_id": [9, 9], "side": ["O", "D"], **{rate: [1.0, 2.0] for rate in BOX_RATES},
                          **court})
    stats = bias.player_stats(panel)
    assert list(stats.columns) == ["player_id", *bias.STATS]
    assert stats.blocks_per100[0] == pytest.approx(1.5)
    assert stats.on_court_plus_minus_per100[0] == pytest.approx(5.0)
    assert stats.off_court_plus_minus_per100[0] == pytest.approx(-2.5)


def test_the_compared_players_are_those_rated_listed_by_the_consensus_and_in_the_panel():
    ratings = pd.DataFrame({"player_id": [3, 1, 2], "poss_season": [300.0, 100.0, 200.0],
                            "rating_off": [0.3, 0.1, 0.2], "rating_def": [0.0, 0.0, 0.0], "player_name": list("cab")})
    blend = pd.DataFrame({"player_id": [1, 3], "offense": [1.0, 3.0], "defense": [0.0, 0.0]})
    stats = pd.DataFrame({"player_id": [1, 2, 3], **{stat: [0.0, 0.0, 0.0] for stat in bias.STATS}})
    players = bias.compared_players(ratings, blend, stats)
    assert list(players.player_id) == [1, 3] and list(players.weight) == [100.0, 300.0]
    assert list(players.consensus_offense) == [1.0, 3.0] and "player_name" not in players
