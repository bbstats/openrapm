"""scripts/89_stitch_by_move.py: the trade-flag test hands each player the rating that matches what he did next."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("stitch_by_move", ROOT / "scripts" / "89_stitch_by_move.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_player_keeps_his_team_only_with_the_same_main_team_in_both_seasons():
    m = _script()
    then = {1: 10, 2: 10, 3: 20}
    now = {1: 10, 2: 30}
    # 1 stayed, 2 moved, 3 is not in the other season, 4 is in neither
    assert m.kept_team(np.array([1, 2, 3, 4]), then, now).tolist() == [True, False, False, False]


def test_main_teams_are_franchises_and_empty_outside_the_data():
    m = _script()

    class Teams:
        def main_team(self, season):
            return {7: 1610612766, 8: 1610612747}

    teams = m.main_teams(Teams(), 2002, {2002})
    assert teams == {7: 1610612740, 8: 1610612747}       # Charlotte's 1997-2002 team is the New Orleans franchise
    assert m.main_teams(Teams(), 2027, {2002}) == {}


def test_every_rating_column_comes_from_the_stayed_list_where_the_player_kept_his_team():
    m = _script()
    traded = pd.DataFrame({"player_id": [1, 2], "season": [2020, 2020], "player_name": ["a", "b"],
                           "rating_off": [1.0, 2.0], "rating_def": [0.5, 0.5], "rating_total": [1.5, 2.5],
                           "poss_off": [100.0, 200.0]})
    stayed = traded.assign(rating_off=[1.4, 2.4], rating_total=[1.9, 2.9])
    out = m.stitch(traded, stayed, np.array([True, False]))
    assert out.rating_off.tolist() == [1.4, 2.0]
    assert out.rating_total.tolist() == [1.9, 2.5]
    assert out.rating_def.tolist() == [0.5, 0.5]
    assert out.stayed.tolist() == [True, False]
    assert out.player_name.tolist() == ["a", "b"]
