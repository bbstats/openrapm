"""Check 166 (scripts/checks/166_power_forecast_refit.py): the registered rule as code, and the published weights read
back as the line between knots the Power page applies."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def check():
    path = ROOT / "scripts" / "checks" / "166_power_forecast_refit.py"
    spec = importlib.util.spec_from_file_location("check166", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(direct_z: float, power_change: float) -> pd.DataFrame:
    return pd.DataFrame({"era": ["all seasons", "all seasons", "1997-2006"],
                         "line_against": ["direct fit", "Power alone at its own scale", "direct fit"],
                         "z": [direct_z, -1.0, 9.0], "change": [0.1, power_change, 9.0]})


def test_the_line_is_accepted_when_it_is_no_worse_than_a_direct_fit_and_beats_power_alone(check):
    """Check 165's rule: not worse than a direct fit at the day by a paired z of +1, and lower than Power alone at its
    own scale on the mean.  An era's reading does not rule."""
    assert check.accepted("17.5%", _table(direct_z=0.9, power_change=-0.2))
    assert not check.accepted("17.5%", _table(direct_z=1.1, power_change=-0.2))
    assert not check.accepted("17.5%", _table(direct_z=0.9, power_change=+0.2))


def test_a_day_without_the_stats_is_ruled_on_the_direct_fit_alone(check):
    """At 62.5% of a season the fit is Power alone, so "Power alone at its own scale" is the direct fit itself."""
    assert check.accepted("62.5%", _table(direct_z=0.9, power_change=+0.2))
    assert not check.accepted("62.5%", _table(direct_z=1.1, power_change=-0.2))


def test_the_published_weights_are_read_back_knot_by_knot_and_on_the_line_between(check):
    document = json.loads((ROOT / check.WEIGHTS_FILE).read_text(encoding="utf-8"))
    frames, shares = check.published_knots()
    assert list(frames) == [knot["day"] for knot in document["knots"]]
    first, last = document["knots"][0], document["knots"][-1]
    assert frames[first["day"]].power_o == first["offense_multiplier"] and shares[first["day"]] == first["share"]
    assert frames[first["day"]][list(first["stats"])].tolist() == list(first["stats"].values())
    assert list(frames[first["day"]].index) == check.NAMES
    # past the last knot its weights hold, as power_forecast.weights_at holds them
    held = check.C165.line(frames, shares, 0.99)
    assert held.power_o == last["offense_multiplier"] and held.power_d == last["defense_multiplier"]
    halfway = (document["knots"][0]["share"] + document["knots"][1]["share"]) / 2
    between = check.C165.line(frames, shares, halfway)
    second = document["knots"][1]
    assert between.power_o == pytest.approx((first["offense_multiplier"] + second["offense_multiplier"]) / 2)


def test_a_player_the_game_log_leaves_out_gets_the_team_of_his_side_of_the_floor(check):
    """Two player-games of the finished seasons are on the floor in the stints and in no game log, one possession
    each; the site's Power step stops on such a row, and the check gives him his side's team."""
    slots = [f"{side}{i}" for side in "ha" for i in range(1, 6)]
    stints = pd.DataFrame([["0029600896", *range(1, 6), *range(11, 16)],
                           ["0029600896", 1, 2, 3, 4, 99, *range(11, 16)]], columns=["game_id", *slots])
    teams = pd.DataFrame({"game_id": "0029600896", "player_id": [*range(1, 6), *range(11, 16)],
                          "tm": ["CHI"] * 5 + ["UTA"] * 5, "name": ["Bulls"] * 5 + ["Jazz"] * 5})
    out = check.with_unlogged_players(teams, stints)
    assert len(out) == 11 and out.iloc[:10].equals(teams)
    assert out.iloc[10].to_dict() == {"game_id": "0029600896", "player_id": 99, "tm": "CHI", "name": "Bulls"}
    assert check.with_unlogged_players(out, stints) is out                     # nobody left out: the table as it came


def test_the_knots_from_a_quarter_of_the_season_on_read_the_blended_folders(check):
    """Through a quarter of its season the published season in progress holds the prior's weight in full; from there
    it is on the line to its fitted one, and so are the folders the knots from a quarter on are fitted on."""
    assert [folder.rsplit("_", 1)[1] for folder in check.KNOTS.values()] == ["held", "held", "line", "line", "line"]
    assert check.NO_STATS == ("50%", "75%") and list(check.KNOTS)[-2:] == ["50%", "75%"]
    assert check.TESTS == {"17.5%": ("fade141_cut175_held", True), "62.5%": ("fade141_cut625_line", False)}
