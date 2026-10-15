"""The season panel's builder (src/openrapm/data/panel.py): the data layer's names mapped onto the panel's, the score
state and the playoff share counted by hand on two tiny games, the panel-wide fill of a row with no possessions, and,
with the data on disk, one season rebuilt and held to the stored panel."""
import numpy as np
import pandas as pd
import pytest

from openrapm.data import panel as panel_module
from openrapm.data.legacy_names import column_name
from openrapm.data.shots import SHOT_LEAGUE_COLS, SHOT_TOTAL_COLS
from openrapm.design import AWAY_SLOTS, HOME_SLOTS
from openrapm.features import CLOSENESS, OFF_COURT, ON_MINUS_OFF, SEASON_PANEL_COLUMNS, SHOT_LEAGUE, SHOT_TOTALS


def test_the_shot_columns_are_the_data_layers_renamed():
    assert [column_name(name) for name in [*SHOT_TOTAL_COLS, *SHOT_LEAGUE_COLS]] == [*SHOT_TOTALS, *SHOT_LEAGUE]
    assert list(panel_module.SHOT_NAMES.values()) == [*SHOT_TOTALS, *SHOT_LEAGUE]
    assert panel_module.PANEL_TARGETS == ("points_with_expected_free_throws",
                                          "points_with_expected_opponent_threes_all")


def _stints() -> pd.DataFrame:
    """Two stints of one game, the home five 1-5 against the away five 11-15: 10 home and 20 away possessions at a
    tied score, then 30 home possessions, none away, 12 points down in garbage time."""
    frame = pd.DataFrame({slot: [number, number] for slot, number in zip(HOME_SLOTS + AWAY_SLOTS,
                                                                        [1, 2, 3, 4, 5, 11, 12, 13, 14, 15])})
    return frame.assign(game_id="g1", poss_h=[10, 30], poss_a=[20, 0], margin_h=[0, -12], is_gt=[False, True],
                        phase="RS")


def test_the_score_state_by_hand():
    state = panel_module.score_state(_stints(), 2026).set_index(["player_id", "side"])
    assert list(state.columns) == ["season", *CLOSENESS]
    home_offence = state.loc[(1, "O")]
    assert home_offence.garbage_time_share == 30 / 40                          # 30 of his 40 possessions
    assert home_offence.closeness == (10 * 1.0 + 30 / 12) / 40                 # 1 / max(|margin|, 1)
    assert home_offence.mean_abs_margin == 30 * 12 / 40
    home_defence = state.loc[(1, "D")]                                         # 20 away possessions, all tied
    assert (home_defence.garbage_time_share, home_defence.closeness, home_defence.mean_abs_margin) == (0.0, 1.0, 0.0)
    assert state.loc[(11, "D")].garbage_time_share == 0.75                     # the away five defend the 40 too


def test_the_playoff_share_by_hand():
    """All ten play 10 + 20 possessions (both ends) in a regular-season game and 6 + 4 in a playoff game: 10 / 40."""
    stints = pd.concat([_stints().iloc[[0]], _stints().iloc[[0]].assign(game_id="g2", poss_h=6, poss_a=4,
                                                                         phase="PO")], ignore_index=True)
    games = pd.DataFrame({"game_id": ["g1", "g2"], "home_team_id": [100, 100], "away_team_id": [200, 200]})
    share = panel_module.playoff_share(stints, games)
    assert share.name == "playoff_share"
    assert abs(share[1] - 10 / 40) < 1e-12 and abs(share[15] - 10 / 40) < 1e-12
    regular_only = panel_module.playoff_share(stints[stints.phase == "RS"], games)
    assert (regular_only == 0.0).all()


def test_a_row_without_possessions_takes_the_whole_panels_score_state():
    panel = pd.DataFrame({"garbage_time_share": [0.1, 0.3, np.nan], "closeness": [0.2, 0.4, np.nan],
                          "mean_abs_margin": [8.0, 10.0, np.nan]})
    filled = panel_module.fill_missing_score_state(panel)
    assert filled.iloc[2].tolist() == pytest.approx([0.2, 0.3, 9.0])
    assert filled.iloc[0].tolist() == [0.1, 0.2, 8.0]


@pytest.mark.slow
@pytest.mark.parametrize("season", [2026, 1998])
@pytest.mark.needs_data("panel", "stints/2026_RS.parquet", "stints/1998_RS.parquet")
def test_a_season_rebuilt_from_all_its_games_is_the_stored_panel(season):
    """A season from all of its games, every column against the stored panel (config paths.panel, by default
    outputs/season_panel.parquet); 1998 reads veterans' seasons before 1997 (data/careers.py).  Exact, except the
    off-court columns: they sum through a dense matrix product (on_court.py), whose last digits depend on how many
    threads the linear-algebra library uses; the stored panel was built with the machine's twelve, the commands pin
    one (a few rows differ by under 1e-13)."""
    from openrapm.data.careers import careers_before
    from openrapm.data.config import load_config, resolve
    from openrapm.data.designs import DesignCache
    from openrapm.data.roles import build_roles, player_season_inputs
    cfg = load_config()
    roles = build_roles(cfg)
    sources = panel_module.SeasonSources(panel_module.SeasonTables(season, cfg), roles,
                                         player_season_inputs(roles, cap=panel_module.share_cap(cfg)),
                                         careers_before(cfg, roles))
    designs = DesignCache(cfg)
    whole = {target: designs.season(season, target) for target in panel_module.PANEL_TARGETS}
    rows = panel_module.season_panel(season, cfg, whole, sources).rows
    stored = pd.read_parquet(resolve(cfg, "panel"))
    stored = stored[stored.season == season].reset_index(drop=True)
    assert list(rows.columns) == SEASON_PANEL_COLUMNS and rows.dtypes.equals(stored.dtypes)
    assert rows[["side", "player_id", "season"]].equals(stored[["side", "player_id", "season"]])
    for column in SEASON_PANEL_COLUMNS[3:]:
        if column in OFF_COURT + ON_MINUS_OFF:
            np.testing.assert_allclose(rows[column], stored[column], rtol=0, atol=1e-12, err_msg=column)
        else:
            # a dashboard column is missing (NaN) before its dashboard starts: the same place in both, not a difference
            assert np.array_equal(rows[column].to_numpy(), stored[column].to_numpy(), equal_nan=True), column
