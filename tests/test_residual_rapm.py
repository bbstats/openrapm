"""residual_rapm.py on a simulated season with a known truth: a rating handed in wrong by a known amount gets a residual
RAPM that points back at the truth, on both sides, positive good; and the table's defense goes in as points allowed."""
import numpy as np
import pandas as pd
import pytest

from openrapm import residual_rapm


@pytest.fixture(scope="module")
def season_world(synthetic_league, synthetic_design, small_cfg):
    sim = synthetic_league(n_seasons=1, n_teams=10, players_per_team=12, games_per_season=400, seed=23)
    design = synthetic_design(sim, cfg=small_cfg(player_unit="player"))
    truth = sim["truth"]["ps"].groupby(["player_id", "season"])[["impact_O", "impact_D"]].mean().reset_index()
    return design, truth


def test_the_tables_defense_goes_in_as_points_allowed():
    table = pd.DataFrame({"season": [2027, 2026], "player_id": [5, 5], "rating_off": [2.0, 9.0],
                          "rating_def": [1.5, 9.0], "poss_off": [1800.0, 10.0]})
    raw = residual_rapm.raw_ratings(table, 2027)
    assert raw.to_dict("records") == [{"player_id": 5, "o": 2.0, "d": -1.5, "poss": 1800.0}]


def test_a_rating_wrong_by_a_known_amount_gets_a_residual_pointing_back_at_the_truth(season_world):
    design, truth = season_world
    rng = np.random.default_rng(6)
    error_off, error_def = rng.normal(0, 1.5, len(truth)), rng.normal(0, 1.0, len(truth))
    # stored positive good: the simulator's defensive impact is points allowed
    table = pd.DataFrame({"season": truth.season, "player_id": truth.player_id,
                          "rating_off": truth.impact_O + error_off, "rating_def": -truth.impact_D + error_def,
                          "poss_off": 1000.0})
    # the simulated league has one target, points, so both sides read the same design
    frame, multipliers = residual_rapm.season_residual({"offense": design, "defense": design}, table,
                                                       int(truth.season.iloc[0]), penalty=500.0)
    assert list(frame.columns) == residual_rapm.COLUMNS
    assert set(multipliers) == {"offense", "defense"} and all(0.5 < value < 1.5 for value in multipliers.values())
    joined = frame.merge(table.assign(error_off=error_off, error_def=error_def), on="player_id")
    # rated too high, the games say lower: the residual follows the error's negative on both sides.  One season of
    # team-games tells teammates apart only by who played how much in which game, so the link is loose (0.28 on
    # offense at this seed), but a defense read with the wrong sign would turn it negative
    for side, error in (("offense", joined.error_off), ("defense", joined.error_def)):
        assert np.corrcoef(joined[f"residual_{side}"], -error)[0, 1] > 0.2, side
