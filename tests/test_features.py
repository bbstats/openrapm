"""The feature catalogue (src/openrapm/features.py): the shipped lists, the formulas, and that every name is defined.

The shipped lists are pinned here in full because the booster reads its inputs by POSITION: a reordered list
would load the published models and silently feed each tree the wrong column.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openrapm import features as feat
from openrapm.commands.glossary import END, START, generated_block, run_glossary

ROOT = Path(__file__).resolve().parents[1]

# DESIGN.md E, the shipped input order: the old alphabetical order of the codes, kept under the plain names
SHIPPED_O = ["assists_per100", "assists_minus_turnovers_per100", "effective_fg_pct", "career_possessions_before_k",
             "seasons_before", "three_point_pct", "shots_per100", "free_throw_attempts_per100", "off_rebounds_per100",
             "off_rebound_share", "three_point_attempt_share", "team_possession_share", "points_per100",
             "steals_per100", "seasons_with_team", "true_shooting_pct", "weight_lb"]
SHIPPED_D = ["blocks_per100", "def_rebounds_per100", "career_possessions_before_k", "seasons_before", "shots_per100",
             "games_started_share", "height_in", "team_possession_share", "points_per100", "steals_per100",
             "steals_plus_blocks_per100", "true_shooting_pct", "used_possessions_per100", "weight_lb"]
OLD_SHIPPED_O = ["ast", "creation", "efg", "exp_poss", "exp_yrs", "fg3p", "fga", "fta", "orb", "orbsh", "p3r",
                 "poss_pct", "pts", "stl", "tenure", "ts", "weight"]
OLD_SHIPPED_D = ["blk", "drb", "exp_poss", "exp_yrs", "fga", "gs_pct", "height", "poss_pct", "pts", "stl", "stocks",
                 "ts", "usage", "weight"]


def test_the_shipped_lists_are_pinned_in_order():
    assert feat.SHIPPED == {"O": SHIPPED_O, "D": SHIPPED_D}
    assert feat.FEATURE_SETS["box_score_and_bio"] == feat.SHIPPED
    assert feat.ROW_FEATURES == ["row_possessions", "row_seasons"]


def test_the_stat_splits_lists_append_their_parts_where_the_box_count_is_read():
    """Experiments 52-55: every shipped input keeps its position; a part goes on each list that reads its count."""
    rebounds = ["def_rebounds_of_free_throws_per100", "def_rebounds_of_threes_per100",
                "def_rebounds_of_rim_twos_per100", "def_rebounds_of_outside_twos_per100"]
    steals = ["bad_pass_steals_per100", "lost_ball_steals_per100"]
    points = ["assisted_points_per100", "unassisted_points_per100", "free_throws_made_per100"]
    threes = ["corner_threes_attempted_per100", "above_break_threes_attempted_per100", "assisted_threes_made_per100",
              "unassisted_threes_made_per100", "free_throw_pct"]
    assert feat.FEATURE_SETS == {
        "box_score_and_bio": feat.SHIPPED,
        "rebound_parts": {"O": SHIPPED_O, "D": SHIPPED_D + rebounds},
        "steal_parts": {"O": SHIPPED_O + steals, "D": SHIPPED_D + steals},
        "points_parts": {"O": SHIPPED_O + points, "D": SHIPPED_D + points},
        "three_point_parts": {"O": SHIPPED_O + threes, "D": SHIPPED_D},
        "stepwise_early": {"O": SHIPPED_O + ["def_rebounds_of_threes_per100"],
                           "D": SHIPPED_D + ["assists_to_mid_range_per100"]},
        "stepwise_late": {"O": SHIPPED_O + ["corner_threes_attempted_per100"],
                          "D": SHIPPED_D + ["free_throws_made_per100"]},
        "ratios_vs_league": {"O": [f"{name}_vs_league" if name in feat.RATIOS else name for name in SHIPPED_O],
                             "D": [f"{name}_vs_league" if name in feat.RATIOS else name for name in SHIPPED_D]}}
    wider = ["russell_blocks_per100", "rim_blocks_per100", "mid_range_blocks_per100", "three_point_blocks_per100",
             "offensive_fouls_per100", "stolen_turnovers_per100", "assists_to_rim_per100",
             "assists_to_mid_range_per100", "assists_to_corner_threes_per100", "assists_to_above_break_threes_per100"]
    assert feat.EVENT_RATES == rebounds + steals + points[:2] + threes[:4] + wider
    assert "def_rebounds_per100" in SHIPPED_D and "def_rebounds_per100" not in SHIPPED_O
    assert all("steals_per100" in side and "points_per100" in side for side in (SHIPPED_O, SHIPPED_D))
    assert "three_point_attempt_share" in SHIPPED_O and "three_point_attempt_share" not in SHIPPED_D


def test_the_shipped_lists_are_the_old_ones_renamed_position_by_position():
    """The published boosters were trained on the old lists; each position must hold the same column."""
    from openrapm.data.legacy_names import column_name
    assert [column_name(code) for code in OLD_SHIPPED_O] == feat.SHIPPED["O"]
    assert [column_name(code) for code in OLD_SHIPPED_D] == feat.SHIPPED["D"]
    assert [column_name(code) for code in ("chunk_poss", "chunk_seasons")] == feat.ROW_FEATURES


def test_the_term_order_of_the_sums_is_kept():
    """A sum is computed in the listed order, and float addition is not associative."""
    assert list(feat.DERIVED) == ["points_per100", "shots_per100", "free_throw_attempts_per100",
                                  "three_point_attempts_per100", "used_possessions_per100", "rebounds_per100",
                                  "steals_plus_blocks_per100", "assists_minus_turnovers_per100",
                                  "threes_minus_twos_attempted_per100"]
    assert list(feat.DERIVED["used_possessions_per100"]) == [
        "twos_made_per100", "twos_missed_per100", "threes_made_per100", "threes_missed_per100",
        "free_throws_made_per100", "free_throws_missed_per100", "turnovers_per100"]
    assert list(feat.DERIVED["used_possessions_per100"].values()) == [1.0, 1.0, 1.0, 1.0, 0.44, 0.44, 1.0]
    assert list(feat.RATIOS) == ["effective_fg_pct", "true_shooting_pct", "three_point_attempt_share",
                                 "free_throws_per_shot", "three_point_pct", "two_point_pct", "free_throw_pct",
                                 "assists_per_used_possession", "turnovers_per_used_possession", "off_rebound_share"]
    assert feat.RATIOS["true_shooting_pct"][2:] == (6.0, 0.54)
    assert list(feat.SHOTQ_PADS.values()) == [50.0, 50.0, 250.0, 450.0, 100.0, 300.0]
    assert feat.SHOTQ == ["two_point_expected_pct", "three_point_expected_pct", "two_point_pct_over_expected",
                          "three_point_pct_over_expected", "expected_points_per_shot", "points_per_shot_over_expected"]


def test_the_box_rates_follow_the_box_counts():
    """The padding returns its rates in config.yaml's box_counts order (design.FEATURES); BOX_RATES names them."""
    from openrapm.data.legacy_names import column_name
    from openrapm.design import FEATURES
    assert [column_name(count) for count in FEATURES] == feat.BOX_RATES
    assert feat.UNCENTRED_RATES == [f"{rate}_uncentred" for rate in feat.BOX_RATES]


def _one_row():
    row = {c: 1.0 for c in feat.BOX_RATES}
    row.update(twos_made_per100=2.0, threes_made_per100=0.5, free_throws_made_per100=-1.0, steals_per100=0.25,
               blocks_per100=0.75, assists_per100=3.0, turnovers_per100=1.0)
    row.update({c: 4.0 for c in feat.UNCENTRED_RATES})
    row.update(twos_made_per100_uncentred=6.0, twos_missed_per100_uncentred=6.0, threes_made_per100_uncentred=2.0,
               threes_missed_per100_uncentred=4.0, off_rebounds_per100_uncentred=3.0,
               def_rebounds_per100_uncentred=9.0)
    row.update(shot_twos_attempted=400.0, shot_twos_made=210.0, shot_twos_expected_made=200.0,
               shot_threes_attempted=300.0, shot_threes_made=111.0, shot_threes_expected_made=105.0,
               league_two_point_pct=0.5, league_three_point_pct=0.35, league_points_per_shot=1.0)
    return pd.DataFrame([row])


def test_the_formulas_on_one_row():
    out = feat.add_features(_one_row(), ["points_per100", "steals_plus_blocks_per100",
                                         "assists_minus_turnovers_per100", "effective_fg_pct", "off_rebound_share",
                                         "two_point_expected_pct", "three_point_pct_over_expected",
                                         "points_per_shot_over_expected"])
    assert out.points_per100[0] == 2 * 2.0 + 3 * 0.5 - 1.0
    assert out.steals_plus_blocks_per100[0] == 1.0 and out.assists_minus_turnovers_per100[0] == 2.0
    assert out.effective_fg_pct[0] == pytest.approx((6.0 + 1.5 * 2.0 + 3.0 * 0.5) / (6.0 + 6.0 + 2.0 + 4.0 + 3.0))
    assert out.off_rebound_share[0] == pytest.approx((3.0 + 3.0 * 0.22) / (3.0 + 9.0 + 3.0))
    assert out.two_point_expected_pct[0] == pytest.approx((200.0 + 50 * 0.5) / (400.0 + 50))
    assert out.three_point_pct_over_expected[0] == pytest.approx((111.0 - 105.0) / (300.0 + 450))
    assert out.points_per_shot_over_expected[0] == pytest.approx((2 * 10.0 + 3 * 6.0) / (700.0 + 300))


def test_the_ratios_against_the_league_on_one_row():
    """Experiment 56: (numerator - L x denominator) / (denominator + pad), L from the league's rates (uncentred minus
    centred); the same as padding toward L and subtracting it."""
    row = _one_row()                       # league offensive rebounds 3.0 - 1.0 = 2.0, defensive 9.0 - 1.0 = 8.0
    out = feat.add_features(row, ["off_rebound_share_vs_league", "true_shooting_pct_vs_league"])
    level = 2.0 / (2.0 + 8.0)
    assert out.off_rebound_share_vs_league[0] == pytest.approx((3.0 - level * 12.0) / (12.0 + 3.0))
    assert out.off_rebound_share_vs_league[0] == pytest.approx((3.0 + 3.0 * level) / (12.0 + 3.0) - level)
    num, den, pad, _ = feat.RATIOS["true_shooting_pct"]
    own = {rate: float(row[f"{rate}_uncentred"][0]) for rate in set(num) | set(den)}
    league = {rate: own[rate] - float(row[rate][0]) for rate in own}
    top, bottom = (sum(k * own[r] for r, k in num.items()), sum(k * own[r] for r, k in den.items()))
    level = sum(k * league[r] for r, k in num.items()) / sum(k * league[r] for r, k in den.items())
    assert out.true_shooting_pct_vs_league[0] == pytest.approx((top - level * bottom) / (bottom + pad))


def test_a_league_average_player_and_a_player_without_attempts_read_zero():
    row = _one_row()
    for rate in feat.BOX_RATES:            # his rates are the league's: every centred rate 0
        row[rate] = 0.0
    average = feat.add_features(row.copy(), list(feat.VS_LEAGUE))
    assert np.allclose(average[list(feat.VS_LEAGUE)].to_numpy(), 0.0)
    silent = _one_row()
    for rate in feat.BOX_RATES:            # no counts at all, the league's rates kept
        silent[rate] = -silent[f"{rate}_uncentred"]
        silent[f"{rate}_uncentred"] = 0.0
    assert np.allclose(feat.add_features(silent, list(feat.VS_LEAGUE))[list(feat.VS_LEAGUE)].to_numpy(), 0.0)


def test_each_ratio_has_one_twin_against_the_league():
    assert list(feat.VS_LEAGUE) == [f"{name}_vs_league" for name in feat.RATIOS]
    by_name = {f.name: f for f in feat.CATALOGUE}
    assert by_name["true_shooting_pct_vs_league"].meaning == "true shooting % above the league's"
    assert {by_name[name].scale for name in feat.VS_LEAGUE} == {"centred"}


def test_a_column_already_present_is_left_alone():
    frame = _one_row().assign(points_per100=99.0)
    assert feat.add_features(frame, ["points_per100"]).points_per100[0] == 99.0


def test_a_name_that_cannot_be_built_raises():
    with pytest.raises(KeyError, match="cannot build"):
        feat.add_features(_one_row().drop(columns=["threes_made_per100_uncentred"]), ["effective_fg_pct"])


def test_every_name_the_prior_can_read_is_in_the_catalogue():
    names = [f.name for f in feat.CATALOGUE]
    assert len(names) == len(set(names))
    wanted = set(feat.PANEL_INPUTS) | set(feat.PRIOR_FEATURES) | set(feat.ROW_FEATURES)
    wanted |= set(feat.SHIPPED["O"]) | set(feat.SHIPPED["D"]) | set(feat.OFF_COURT) | set(feat.ON_MINUS_OFF)
    assert not wanted - set(names)
    assert {f.scale for f in feat.CATALOGUE} <= {"centred", "level"}
    assert all(f.meaning and f.formula and f.unit and f.group for f in feat.CATALOGUE)
    assert len(feat.PRIOR_FEATURES) == len(set(feat.PRIOR_FEATURES)) == 85


def test_every_season_panel_column_but_the_keys_is_in_the_catalogue():
    names = {f.name for f in feat.CATALOGUE}
    assert feat.SEASON_PANEL_COLUMNS[:3] == feat.PANEL_KEYS == ["side", "player_id", "season"]
    assert not set(feat.SEASON_PANEL_COLUMNS[3:]) - names
    assert len(feat.SEASON_PANEL_COLUMNS) == len(set(feat.SEASON_PANEL_COLUMNS)) == 86
    assert feat.SEASON_PANEL_COLUMNS[-22:] == feat.EVENT_RATES, "the play-by-play parts come after every older column"


def test_the_audit_groups_and_the_swap_features_are_catalogue_names():
    from openrapm.swap_adjust import PANEL_FEATURES
    names = {f.name for f in feat.CATALOGUE}
    groups = json.loads((ROOT / "params" / "audit_groups.json").read_text(encoding="utf-8"))
    members = {member for side in groups["sides"].values() for group in side["groups"] for member in group["members"]}
    assert not (set(groups["codes"]) | members) - names
    assert not set(PANEL_FEATURES) - names


def test_the_catalogue_formulas_follow_the_constants():
    by_name = {f.name: f for f in feat.CATALOGUE}
    assert by_name["points_per100"].formula == "2 twos_made_per100 + 3 threes_made_per100 + free_throws_made_per100"
    assert by_name["assists_minus_turnovers_per100"].formula == "assists_per100 - turnovers_per100"
    assert by_name["true_shooting_pct"].formula.startswith(
        "(2 twos_made_per100 + 3 threes_made_per100 + free_throws_made_per100 + 6 x 0.54) / (2 twos_made_per100 + "
        "2 twos_missed_per100")


def test_meaning_reads_the_catalogue_and_returns_an_unknown_name_as_it_is():
    assert feat.meaning("steals_per100") == "steals"
    assert feat.meaning("not_a_column") == "not_a_column"


def test_the_glossary_has_one_row_per_name_and_escapes_bars():
    text = feat.glossary_markdown()
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    assert len(rows) == len(feat.CATALOGUE)
    closeness = next(line for line in rows if line.startswith("| `closeness`"))
    assert "\\|margin\\|" in closeness and closeness.count(" | ") == 4


def test_the_glossary_file_holds_the_catalogue():
    """GLOSSARY.md's generated block is the catalogue's table: `openrapm glossary` rewrites it after a change here."""
    assert generated_block((ROOT / "GLOSSARY.md").read_text(encoding="utf-8")) == feat.glossary_markdown()


def test_the_glossary_command_rewrites_only_the_block(tmp_path):
    path = tmp_path / "GLOSSARY.md"
    path.write_text(f"# Prose\n\nkept\n\n{START}\n| stale |\n{END}\nafter\n", encoding="utf-8")
    assert run_glossary(check=True, path=path) == 1
    assert "| stale |" in path.read_text(encoding="utf-8"), "--check never writes"
    assert run_glossary(path=path) == 0
    text = path.read_text(encoding="utf-8")
    assert text.startswith(f"# Prose\n\nkept\n\n{START}\n") and text.endswith(f"{END}\nafter\n")
    assert generated_block(text) == feat.glossary_markdown()
    assert run_glossary(check=True, path=path) == 0


def test_a_glossary_without_its_markers_is_refused(tmp_path):
    path = tmp_path / "GLOSSARY.md"
    path.write_text(f"# Prose\n{END}\n{START}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generated: features"):
        run_glossary(path=path)


def test_the_panel_inputs_are_averaged_columns_only():
    """Nothing derived is averaged: a ratio of averages is wanted, not an average of ratios."""
    derived = set(feat.DERIVED) | set(feat.RATIOS) | set(feat.SHOTQ)
    assert not derived & set(feat.PANEL_INPUTS)
    assert np.all([c in feat.PANEL_INPUTS for c in feat.UNCENTRED_RATES])


def test_the_body_inputs_are_the_data_layers_renamed_and_the_role_inputs_are_built_plain():
    """The data layer builds the body inputs under its own short names (data.bio) and the season panel carries
    them under the catalogue's, the same columns in the same order; data.roles builds the role and career inputs
    under the catalogue's names directly."""
    from openrapm.data import roles
    from openrapm.data.bio import PLAYER_INPUTS
    from openrapm.data.legacy_names import column_name
    assert [column_name(name) for name in PLAYER_INPUTS] == feat.BIO
    counts = pd.DataFrame({"player_id": [1, 1, 2], "season": [2001, 2002, 2002], "games": [10.0, 20.0, 5.0],
                           "starts": [5.0, 20.0, 0.0], "minutes": [100.0, 600.0, 20.0],
                           "poss_on": [400.0, 1600.0, 50.0], "team_poss": [8000.0, 8000.0, 8000.0],
                           "age": [24.0, 25.0, 21.0]})
    inputs = roles.player_season_inputs(counts)
    assert list(inputs.columns[-4:-1]) == feat.ROLE                    # then age_imputed
    assert list(roles.career_inputs(inputs, 2003, [1, 2]).columns) == feat.CAREER


def test_the_prior_can_read_what_a_tree_cannot_build():
    """A tree cannot form a ratio or a sum of two columns; every derived family is among the prior's inputs."""
    assert "season" not in feat.PRIOR_FEATURES, "a row pooled over twelve seasons has no season"
    for family in (feat.DERIVED, feat.RATIOS, feat.VS_LEAGUE, feat.SHOTQ):
        assert set(family) <= set(feat.PRIOR_FEATURES)
