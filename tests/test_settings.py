"""The shipped settings (src/openrapm/settings.py), pinned: changing one is an experiment, not a refactor."""
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from openrapm.data.config import booster_params, load_config
from openrapm.settings import PRODUCT, TEST, describe_factors, factor_for

ROOT = Path(__file__).resolve().parents[1]
BOOSTER_KEYS = {"learning_rate", "n_estimators", "num_leaves", "max_depth", "min_child_samples", "reg_lambda",
                "reg_alpha", "max_bin", "path_smooth", "linear_tree", "linear_lambda",
                "colsample_bytree"}            # each tree's share of the inputs, since experiment 63


def test_the_label_settings():
    labels = PRODUCT.labels
    assert (labels.offense_penalty, labels.defense_penalty, labels.context_penalty) == (40000.0, 40000.0, 0.0)
    assert labels.min_possessions == 100.0
    assert labels.tier_edges == (0.0, 500.0, 1500.0, 4444.0)
    assert labels.full_trust_possessions == 4444.0
    assert labels.unshrink_columns == ("defense",)
    assert (labels.team_offense_penalty, labels.team_defense_penalty) == (0.0, 0.0)   # no team-season flags
    assert not labels.team_flags
    assert replace(labels, team_defense_penalty=500.0).team_flags


def test_the_targets_prior_and_ridge():
    assert (PRODUCT.targets.offence, PRODUCT.targets.defence) == (   # experiment 95, adopted 2026-10-22
        "points_with_expected_free_throws", "points_with_expected_opponent_threes_late_as_scored")
    assert PRODUCT.prior.chunk_sizes == (1, 2, 3)
    assert PRODUCT.prior.n_folds == 5 and PRODUCT.prior.fold_assignment == 0
    assert PRODUCT.prior.feature_set == "hustle_no_three_pct"     # experiment 77, adopted 2026-10-17
    assert PRODUCT.prior.booster_file == "params/booster_params_lgb1_tune2.json"
    assert PRODUCT.prior.era_halflife == 0.0                 # every training season counts the same
    assert PRODUCT.prior.bags == 10                          # experiment 78, adopted 2026-10-17
    assert PRODUCT.prior.second_prior == "ctx82"             # experiment 82, adopted 2026-10-20
    assert PRODUCT.prior.second_prior_team_totals == "held"
    assert PRODUCT.prior.second_prior_inputs == "context_and_types_centred"   # experiment 93, adopted 2026-10-21
    assert PRODUCT.ridge.penalty == 13037.0
    assert PRODUCT.exclude_neighbours == 0


def test_each_distinct_target_is_fitted_once():
    assert PRODUCT.targets.distinct() == ["points_with_expected_free_throws",
                                          "points_with_expected_opponent_threes_late_as_scored"]
    assert replace(PRODUCT.targets, defence="points_with_expected_free_throws").distinct() == [
        "points_with_expected_free_throws"]


def test_the_shrink_and_swap_settings():
    assert PRODUCT.shrink.fold_seasons == (1997, 2026) and PRODUCT.shrink.nearest == 8
    assert PRODUCT.shrink.folds_tag == "inputs99_early,inputs99_within"     # experiment 99, adopted 2026-10-22
    assert PRODUCT.shrink.defence_factors == ((0, 0.89), (2017, 0.84))     # experiment 98, adopted 2026-10-22
    assert PRODUCT.shrink.extrapolate_folds_tag == "inputs99_k10"         # experiment 101, adopted 2026-10-22
    assert [PRODUCT.shrink.defence_factor_for(s) for s in (1997, 2016, 2017, 2026, 2027)] == [0.89, 0.89, 0.84, 0.84,
                                                                                              0.84]
    swap = PRODUCT.swap
    assert (swap.applied_share, swap.type_model_penalty, swap.n_folds) == (0.0, 1e-4, 5)   # step 5 off, experiment 82


def test_a_factor_by_era_holds_from_its_first_season_until_the_next():
    factors = ((0, 0.9), (2007, 0.88), (2017, 0.84))
    assert [factor_for(factors, s) for s in (1990, 2006, 2007, 2016, 2017, 2030)] == [0.9, 0.9, 0.88, 0.88, 0.84,
                                                                                      0.84]
    assert factor_for(((0, 0.866),), 2026) == 0.866
    assert factor_for(((1997, 0.9),), 1990) == 0.9          # before the first pair: the first pair's
    assert describe_factors(((0, 0.866),)) == "0.866 every season"
    assert describe_factors(((0, 0.89), (2017, 0.84))) == "0.89 before 2017, 0.84 from 2017"


def test_the_test_setting_differs_in_three_places_only():
    assert TEST.exclude_neighbours == 1
    assert TEST.shrink.rule == "test" and PRODUCT.shrink.rule == "product"
    assert TEST.swap.exclude_near == 1 and PRODUCT.swap.exclude_near == 0
    same = [name for name in ("targets", "labels", "prior", "ridge")]
    assert all(getattr(TEST, name) == getattr(PRODUCT, name) for name in same)


def test_the_settings_cannot_be_changed_in_place():
    with pytest.raises(FrozenInstanceError):
        PRODUCT.ridge.penalty = 1.0


def test_the_booster_file_names_lightgbm_with_the_shipped_keys():
    spec = json.loads((ROOT / PRODUCT.prior.booster_file).read_text(encoding="utf-8"))
    for side in ("O", "D"):
        assert spec[side]["learner"] == "lightgbm"
        assert set(spec[side]["params"]) == BOOSTER_KEYS
    loaded = booster_params(load_config(), PRODUCT.prior.booster_file)
    assert loaded == {side: spec[side]["params"] for side in ("O", "D")}
    assert loaded["O"]["n_estimators"] == 1091 and loaded["D"]["n_estimators"] == 130      # experiment 76
