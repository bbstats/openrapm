"""The within-season folds (src/openrapm/commands/within_season.py): the fold assignment, the saved models it
accepts, and the prior it rebuilds from them.  The folds themselves need the real seasons; these parts do not."""
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from openrapm import box_prior
from openrapm.commands import within_season as ws
from openrapm.settings import TEST

LEGACY_META = dict(argv=["scripts/62_single_year_board.py"], exclude_neighbours=1, features="boruta_noonc",
                   rows="chunks", chunk_label="career", unshrink=["defense"], player_folds=5, booster_params="lgb1",
                   crossfit="scale", pad_once=False, out="season_ratings_weightfix_2017.parquet")


def test_the_folds_are_seeded_by_season_and_repeat_and_the_same_size():
    games = [f"00{2600000 + i}" for i in range(1315)]
    first = ws.assign_folds(games, 4, 2026, 0)
    assert first.equals(ws.assign_folds(list(reversed(games)), 4, 2026, 0))       # the order handed in is moot
    assert not first.equals(ws.assign_folds(games, 4, 2026, 1))
    assert not first.equals(ws.assign_folds(games, 4, 2025, 0))
    counts = first.value_counts()
    assert counts.max() - counts.min() <= 1 and set(counts.index) == {0, 1, 2, 3}
    assert list(first.index) == sorted(games) and first.name == "r0"


def test_the_earlier_scripts_saved_models_are_accepted():
    ws.check_models(LEGACY_META)
    ws.check_models({**LEGACY_META, "schema": 2, "settings": {}})


@pytest.mark.parametrize("change", [dict(exclude_neighbours=0), dict(unshrink=[]), dict(player_folds=0),
                                    dict(crossfit="0"), dict(crossfit=None)])
def test_models_from_other_settings_are_refused(change):
    with pytest.raises(SystemExit, match="shipped test settings"):
        ws.check_models({**LEGACY_META, **change})


@pytest.mark.parametrize("settings", [{"targets": {"offence": "points_with_expected_free_throws",
                                                   "defence": "points_with_expected_opponent_threes_all"},
                                       "ridge": {"penalty": 13037.0}},
                                      {"targets": {"offence": "points_with_expected_free_throws",
                                                   "defence": "points_with_expected_opponent_threes"},
                                       "ridge": {"penalty": 10000.0}}])
def test_models_saved_with_another_target_or_ridge_penalty_are_refused(settings):
    """The folds rate with the shipped targets and penalty; a file that records others is refused, not rerated."""
    shipped = {"targets": asdict(TEST.targets), "ridge": asdict(TEST.ridge), "labels": asdict(TEST.labels)}
    ws.check_models({**LEGACY_META, "schema": 2, "settings": shipped})
    with pytest.raises(SystemExit, match="shipped test settings"):
        ws.check_models({**LEGACY_META, "schema": 2, "settings": {**shipped, **settings}})


def test_the_honest_cross_fitted_scale_is_refused_by_name():
    with pytest.raises(SystemExit, match="honest"):
        ws.check_models({**LEGACY_META, "crossfit": "honest"})


def _saved(unseen=(2025, 2026, 2027), groupings=1, folds=5):
    side = dict(folds=[object()] * folds, n_folds=5, groupings=groupings, unseen=list(unseen))
    return {2026: {"O": side, "D": side}}


def test_a_season_needs_its_models_and_models_that_never_saw_it_or_a_neighbour():
    assert ws.season_models(_saved(), 2026)["O"]["unseen"] == [2025, 2026, 2027]
    with pytest.raises(SystemExit, match="no saved models"):
        ws.season_models(_saved(), 2025)
    with pytest.raises(SystemExit, match="trained on"):
        ws.season_models(_saved(unseen=(2026,)), 2026)
    with pytest.raises(SystemExit, match="groupings"):
        ws.season_models(_saved(groupings=2, folds=10), 2026)


def test_the_saved_prior_is_the_out_of_player_prediction():
    """The saved boosters, rebuilt, answer exactly as the build that saved them did."""
    rng = np.random.default_rng(2)
    ids = np.repeat(np.arange(60), 2)
    train = pd.DataFrame({"a": rng.normal(size=ids.size), "b": rng.normal(size=ids.size)},
                         index=pd.Index(ids, name="player_id"))
    train = train.assign(label=train.a - train.b + rng.normal(scale=0.2, size=ids.size),
                         row_weight=rng.uniform(1, 3, ids.size))
    params = {"n_estimators": 30, "num_leaves": 5, "learning_rate": 0.1, "min_child_samples": 3}
    model = box_prior.fit_out_of_player(train, ["a", "b"], params, n_folds=5)
    panel = pd.DataFrame({"player_id": [3, 17, 999], "side": "O", "possessions": [800.0, 1200.0, 50.0],
                          "a": [0.1, -1.0, 2.0], "b": [0.5, 0.0, -0.3]})
    saved = dict(full=model.full, folds=model.fold_models, excluded=model.held_out_players, feats=[],
                 model_feats=["a", "b"])
    got = ws.saved_prior(panel, "O", saved)
    assert list(got) == [3, 17, 999]
    assert np.array_equal(np.array(list(got.values())), box_prior.predict(model, panel))


def test_boosters_saved_before_the_rename_are_read_under_the_plain_names(tmp_path, capsys):
    """A pickle of 2026-10-09 (old names, earlier meta) loads with its inputs renamed in order, passes the check,
    and says so in one line; its targets, saved in schema 2, compare equal to the shipped ones."""
    side = dict(full=None, folds=[], excluded=[], feats=["ast", "weight"],
                model_feats=["ast", "weight", "chunk_poss", "chunk_seasons"], prior={})
    settings = {"targets": {"offence": "xpts_ft", "defence": "x3def_w0.25"}, "ridge": asdict(TEST.ridge),
                "prior": {"feature_set": "boruta_noonc"}}
    pd.to_pickle({2026: {"O": side, "D": side}, "meta": {**LEGACY_META, "schema": 2, "settings": settings}},
                 tmp_path / "prior_models_old.pkl")
    saved = ws.load_models(tmp_path, "old")
    assert saved[2026]["O"]["model_feats"] == ["assists_per100", "weight_lb", "row_possessions", "row_seasons"]
    assert saved["meta"]["features"] == "box_score_and_bio"
    ws.check_models(saved["meta"])
    assert "before 2026-10-11" in capsys.readouterr().out
