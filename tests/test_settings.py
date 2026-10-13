"""The shipped settings (src/openrapm/settings.py), pinned: changing one is an experiment, not a refactor."""
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from openrapm.data.config import booster_params, load_config
from openrapm.settings import PRODUCT, TEST

ROOT = Path(__file__).resolve().parents[1]
BOOSTER_KEYS = {"learning_rate", "n_estimators", "num_leaves", "max_depth", "min_child_samples", "reg_lambda",
                "reg_alpha", "max_bin", "path_smooth", "linear_tree", "linear_lambda"}


def test_the_label_settings():
    labels = PRODUCT.labels
    assert (labels.offense_penalty, labels.defense_penalty, labels.context_penalty) == (40000.0, 40000.0, 0.0)
    assert labels.min_possessions == 100.0
    assert labels.tier_edges == (0.0, 500.0, 1500.0, 4444.0)
    assert labels.full_trust_possessions == 4444.0
    assert labels.unshrink_columns == ("defense",)


def test_the_targets_prior_and_ridge():
    assert (PRODUCT.targets.offence, PRODUCT.targets.defence) == ("points_with_expected_free_throws",
                                                                  "points_with_expected_opponent_threes")
    assert PRODUCT.prior.chunk_sizes == (1, 2, 3)
    assert PRODUCT.prior.n_folds == 5 and PRODUCT.prior.fold_assignment == 0
    assert PRODUCT.prior.feature_set == "box_score_and_bio"
    assert PRODUCT.prior.booster_file == "params/booster_params_lgb1.json"
    assert PRODUCT.prior.era_halflife == 0.0                 # every training season counts the same
    assert PRODUCT.ridge.penalty == 13037.0
    assert PRODUCT.exclude_neighbours == 0


def test_each_distinct_target_is_fitted_once():
    assert PRODUCT.targets.distinct() == ["points_with_expected_free_throws", "points_with_expected_opponent_threes"]
    assert replace(PRODUCT.targets, defence="points_with_expected_free_throws").distinct() == [
        "points_with_expected_free_throws"]


def test_the_shrink_and_swap_settings():
    assert PRODUCT.shrink.fold_seasons == (1997, 2026) and PRODUCT.shrink.nearest == 8
    assert PRODUCT.shrink.folds_tag == "weightfix_early,weightfix_within"
    swap = PRODUCT.swap
    assert (swap.applied_share, swap.type_model_penalty, swap.n_folds) == (0.5, 1e-4, 5)


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
    assert loaded["O"]["n_estimators"] == 672 and loaded["D"]["n_estimators"] == 309
