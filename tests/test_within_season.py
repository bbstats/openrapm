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
                                                   "defence": "points_with_expected_opponent_threes"},
                                       "ridge": {"penalty": 10000.0}}])
def test_models_saved_with_another_ridge_penalty_are_refused(settings):
    """The folds rate with the shipped penalty; a file that records another is refused, not rerated.  Another target
    is read as the file's own since experiment 84 (`fold_settings`)."""
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


def test_the_saved_prior_builds_the_ratios_against_the_league_and_era_boosters_pass_the_check():
    """Experiment 56: a booster reading a `_vs_league` ratio is answered from the panel's own rates, as the build
    answered it; boosters fitted with an era half-life pass the check (it changes the training weights only)."""
    rng = np.random.default_rng(4)
    names = ["off_rebound_share_vs_league", "row_possessions", "row_seasons"]
    ids = np.repeat(np.arange(50), 2)
    train = pd.DataFrame({"off_rebound_share_vs_league": rng.normal(0, 0.05, ids.size),
                          "row_possessions": rng.uniform(200, 3000, ids.size), "row_seasons": 1.0},
                         index=pd.Index(ids, name="player_id"))
    train = train.assign(label=10 * train.off_rebound_share_vs_league, row_weight=1.0)
    params = {"n_estimators": 20, "num_leaves": 4, "learning_rate": 0.2, "min_child_samples": 3}
    model = box_prior.fit_out_of_player(train, names, params, n_folds=5)
    panel = pd.DataFrame({"player_id": [3, 8], "side": "O", "possessions": [900.0, 2400.0],
                          "off_rebounds_per100": [0.5, -1.0], "def_rebounds_per100": [-1.0, 2.0],
                          "off_rebounds_per100_uncentred": [3.0, 1.5], "def_rebounds_per100_uncentred": [6.0, 9.0]})
    saved = dict(full=model.full, folds=model.fold_models, excluded=model.held_out_players,
                 feats=["off_rebound_share_vs_league"], model_feats=names)
    got = ws.saved_prior(panel, "O", saved)
    rated = box_prior.rated_rows(panel, ["off_rebound_share_vs_league"])
    assert rated.off_rebound_share_vs_league.notna().all()
    assert np.array_equal(np.array(list(got.values())), box_prior.predict(model, rated))
    era = {"targets": asdict(TEST.targets), "ridge": asdict(TEST.ridge),
           "prior": {**asdict(TEST.prior), "era_halflife": 8.0}}
    ws.check_models({**LEGACY_META, "schema": 3, "settings": era})


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


def test_the_leak_control_feeds_the_boosters_the_whole_season_and_the_ridge_the_fit_games(monkeypatch):
    """--whole-season-inputs: the boosters read the panel handed in, the ridge still the world's own designs."""
    read, rated = [], {}
    monkeypatch.setattr(ws, "saved_prior", lambda panel, side, saved: read.append(panel) or {1: 0.0})

    def rate_season(designs, prior, settings, season):
        rated.update(designs)
        return pd.DataFrame({"possessions": [5.0]})

    monkeypatch.setattr(ws.prior_rapm, "rate_season", rate_season)
    fit_rows, whole_rows = pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]})

    class Design:
        rows = pd.DataFrame({"poss": [1.0]})

    designs = {target: Design() for target in ws.WORLD_TARGETS}
    world = ws.SeasonWorld(2026, np.array([1]), designs, type("Inputs", (), {"rows": fit_rows})(), {})
    ws.rate_world(world, {"O": {}, "D": {}})
    assert all(panel is fit_rows for panel in read)
    read.clear()
    ws.rate_world(world, {"O": {}, "D": {}}, prior_panel=whole_rows)
    assert len(read) == 2 and all(panel is whole_rows for panel in read)
    assert all(rated[target] is designs[target] for target in TEST.targets.distinct())


def test_a_fold_says_where_its_boosters_inputs_came_from():
    run = ws.SeasonRun(2026, {}, {}, None, {}, None)
    assert run.prior_inputs == "fit games"
    run.prior_panel = pd.DataFrame()
    assert run.prior_inputs == "whole season"


def test_the_folds_rate_on_the_saved_builds_own_targets():
    """Experiment 84 fitted the defence on the offence's target; its folds must rate the same way, and a file that
    names no targets is read on the shipped ones."""
    same = {"offence": "points_with_expected_free_throws", "defence": "points_with_expected_free_throws"}
    settings = ws.fold_settings({"settings": {"targets": same}})
    assert settings.targets.distinct() == ["points_with_expected_free_throws"]
    assert settings.exclude_neighbours == 1 and settings.ridge == ws.SETTINGS.ridge
    assert ws.fold_settings({}) is ws.TEST
    assert "points_with_expected_free_throws" in ws.world_targets(settings)
    assert not ws._step3_mismatches({"settings": {"targets": same, "ridge": {"penalty": 13037.0}}})
