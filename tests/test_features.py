"""The feature catalogue (src/openrapm/features.py): the shipped lists, the formulas, and that every code is defined.

The shipped lists are pinned here in full because the booster reads its inputs by POSITION: a reordered list
would load the published models and silently feed each tree the wrong column.
"""
import numpy as np
import pandas as pd
import pytest

from openrapm import features as feat

SHIPPED_O = ["ast", "creation", "efg", "exp_poss", "exp_yrs", "fg3p", "fga", "fta", "orb", "orbsh", "p3r",
             "poss_pct", "pts", "stl", "tenure", "ts", "weight"]
SHIPPED_D = ["blk", "drb", "exp_poss", "exp_yrs", "fga", "gs_pct", "height", "poss_pct", "pts", "stl", "stocks",
             "ts", "usage", "weight"]


def test_the_shipped_lists_are_pinned_in_order():
    assert feat.SHIPPED == {"O": SHIPPED_O, "D": SHIPPED_D}
    assert feat.FEATURE_SETS == {"boruta_noonc": feat.SHIPPED}
    assert feat.ROW_FEATURES == ["chunk_poss", "chunk_seasons"]


def test_the_term_order_of_the_sums_is_kept():
    """A sum is computed in the listed order, and float addition is not associative."""
    assert list(feat.DERIVED) == ["pts", "fga", "fta", "fg3a", "usage", "bigness", "reb", "stocks", "creation",
                                  "shotmix"]
    assert list(feat.DERIVED["usage"]) == ["fg2m", "fg2_miss", "fg3m", "fg3_miss", "ftm", "ft_miss", "tov"]
    assert list(feat.RATIOS) == ["efg", "ts", "p3r", "ftr", "fg3p", "fg2p", "ftp", "astr", "tovr", "orbsh"]
    assert feat.RATIOS["ts"][2:] == (6.0, 0.54)
    assert feat.SHOTQ_PADS == {"q2": 50.0, "q3": 50.0, "m2": 250.0, "m3": 450.0, "xps": 100.0, "mpts": 300.0}


def _one_row():
    row = {c: 1.0 for c in feat.BOX_RATES}
    row.update(fg2m=2.0, fg3m=0.5, ftm=-1.0, stl=0.25, blk=0.75, ast=3.0, tov=1.0)
    row.update({f"raw_{c}": 4.0 for c in feat.BOX_RATES})
    row.update(raw_fg2m=6.0, raw_fg2_miss=6.0, raw_fg3m=2.0, raw_fg3_miss=4.0, raw_orb=3.0, raw_drb=9.0)
    row.update(shot_fg2a=400.0, shot_fg2m=210.0, shot_xl2=200.0, shot_fg3a=300.0, shot_fg3m=111.0, shot_xl3=105.0,
               shot_lg2=0.5, shot_lg3=0.35, shot_lgpps=1.0)
    return pd.DataFrame([row])


def test_the_formulas_on_one_row():
    out = feat.add_features(_one_row(), ["pts", "stocks", "creation", "efg", "orbsh", "q2", "m3", "mpts"])
    assert out.pts[0] == 2 * 2.0 + 3 * 0.5 - 1.0
    assert out.stocks[0] == 1.0 and out.creation[0] == 2.0
    assert out.efg[0] == pytest.approx((6.0 + 1.5 * 2.0 + 3.0 * 0.5) / (6.0 + 6.0 + 2.0 + 4.0 + 3.0))
    assert out.orbsh[0] == pytest.approx((3.0 + 3.0 * 0.22) / (3.0 + 9.0 + 3.0))
    assert out.q2[0] == pytest.approx((200.0 + 50 * 0.5) / (400.0 + 50))
    assert out.m3[0] == pytest.approx((111.0 - 105.0) / (300.0 + 450))
    assert out.mpts[0] == pytest.approx((2 * 10.0 + 3 * 6.0) / (700.0 + 300))


def test_a_column_already_present_is_left_alone():
    frame = _one_row().assign(pts=99.0)
    assert feat.add_features(frame, ["pts"]).pts[0] == 99.0


def test_a_name_that_cannot_be_built_raises():
    with pytest.raises(KeyError, match="cannot build"):
        feat.add_features(_one_row().drop(columns=["raw_fg3m"]), ["efg"])


def test_every_code_the_prior_can_read_is_in_the_catalogue():
    names = [f.name for f in feat.CATALOGUE]
    assert len(names) == len(set(names))
    wanted = set(feat.PANEL_INPUTS) | set(feat.PRIOR_FEATURES) | set(feat.ROW_FEATURES)
    wanted |= set(feat.SHIPPED["O"]) | set(feat.SHIPPED["D"]) | set(feat.OFFC) | set(feat.NET)
    assert not wanted - set(names)
    assert {f.scale for f in feat.CATALOGUE} <= {"centred", "level"}
    assert all(f.meaning and f.formula and f.unit and f.group for f in feat.CATALOGUE)
    assert len(feat.PRIOR_FEATURES) == len(set(feat.PRIOR_FEATURES)) == 54


def test_the_catalogue_formulas_follow_the_constants():
    by_name = {f.name: f for f in feat.CATALOGUE}
    assert by_name["pts"].formula == "2 fg2m + 3 fg3m + ftm"
    assert by_name["creation"].formula == "ast - tov"
    assert by_name["ts"].formula.startswith("(2 fg2m + 3 fg3m + ftm + 6 x 0.54) / (2 fg2m + 2 fg2_miss")


def test_the_glossary_has_one_row_per_code_and_escapes_bars():
    text = feat.glossary_markdown()
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    assert len(rows) == len(feat.CATALOGUE)
    closeness = next(line for line in rows if line.startswith("| `closeness`"))
    assert "\\|margin\\|" in closeness and closeness.count(" | ") == 4


def test_the_panel_inputs_are_averaged_columns_only():
    """Nothing derived is averaged: a ratio of averages is wanted, not an average of ratios."""
    derived = set(feat.DERIVED) | set(feat.RATIOS) | set(feat.SHOTQ)
    assert not derived & set(feat.PANEL_INPUTS)
    assert np.all([c in feat.PANEL_INPUTS for c in feat.RAW_RATES])


def test_the_body_inputs_are_the_ones_the_data_layer_builds():
    """features.py may not import the data layer, so it repeats data.bio.PLAYER_INPUTS and data.roles' career
    names; each pair must agree."""
    from openrapm.data.bio import PLAYER_INPUTS
    from openrapm.data.roles import CAREER_INPUTS
    assert feat.BIO == list(PLAYER_INPUTS)
    assert feat.CAREER == list(CAREER_INPUTS)


def test_the_prior_can_read_what_a_tree_cannot_build():
    """A tree cannot form a ratio or a sum of two columns; every derived family is among the prior's inputs."""
    assert "season" not in feat.PRIOR_FEATURES, "a row pooled over twelve seasons has no season"
    for family in (feat.DERIVED, feat.RATIOS, feat.SHOTQ):
        assert set(family) <= set(feat.PRIOR_FEATURES)
