"""`openrapm build` and `openrapm shrink` check step 4's within-season folds before anything else.

On fresh data `openrapm build` used to run steps 1-3, about fifteen minutes, before step 4 found no folds.  Now
both commands stop at once with a message naming the folds folder and the two commands that make it
(src/openrapm/commands/build.py, `require_folds`).
"""
import json
from dataclasses import replace

import pytest

from openrapm.commands import build
from openrapm.settings import PRODUCT

TAG = build.fold_tags(PRODUCT.shrink)[0]          # the first folder checked
SPAN = "{}-{}".format(*PRODUCT.shrink.fold_seasons)
MAKE_RATE = f"openrapm rate --setting test --seasons {SPAN} --save-models NAME"
MAKE_FOLDS = f"openrapm folds --models NAME --base season_ratings_test_step3 --seasons {SPAN} --tag {TAG}"
ONE_FOLDER = replace(PRODUCT.shrink, folds_tag=TAG, fold_seasons=(2026, 2026), extrapolate_folds_tag="")


def _cfg(tmp_path) -> dict:
    return {"_root": str(tmp_path), "first_season": 2017, "last_season": 2026,
            "paths": {"within": str(tmp_path / "within"), "outputs": str(tmp_path / "outputs")}}


def _never(*args, **kwargs):
    raise AssertionError("a step ran before the folds were checked")


def _write_folds(directory, season=2026, folds=range(4)):
    """One repeat of `season`: each fold's json (three-quarter-season, leak checks passed) and its two data files."""
    directory.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        stem = f"{season}_r0_f{fold}"
        record = {"summary": {"season": season, "repeat": 0, "fold": fold, "rate_from": "rest"},
                  "checks": {"a leak check": True}}
        (directory / f"fold_{stem}.json").write_text(json.dumps(record))
        (directory / f"players_{stem}.parquet").write_bytes(b"")
        (directory / f"test_{stem}.npz").write_bytes(b"")


def _message(stop) -> str:
    return str(stop.value)


def test_build_on_fresh_data_stops_before_step_1_naming_the_folder_and_the_two_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "rate_table", _never)
    with pytest.raises(SystemExit) as stop:
        build.run_build(_cfg(tmp_path), PRODUCT, [2026], "product")
    message = _message(stop)
    assert str(tmp_path / "within" / TAG) in message
    assert MAKE_RATE in message and MAKE_FOLDS in message
    assert message.index(MAKE_RATE) < message.index(MAKE_FOLDS)


def test_shrink_stops_before_reading_its_table(tmp_path, monkeypatch):
    monkeypatch.setattr(build.pd, "read_parquet", _never)
    with pytest.raises(SystemExit) as stop:
        build.run_shrink(_cfg(tmp_path), PRODUCT, "season_ratings_product_step3", "product")
    assert str(tmp_path / "within" / TAG) in _message(stop) and MAKE_FOLDS in _message(stop)


def test_an_incomplete_folder_is_refused_with_its_problem(tmp_path):
    """Three folds of a repeat's four, or a fold whose data files are gone, stop the run too."""
    settings = replace(PRODUCT, shrink=ONE_FOLDER)
    directory = tmp_path / "within" / TAG
    _write_folds(directory, folds=range(3))
    with pytest.raises(SystemExit) as stop:
        build.require_folds(_cfg(tmp_path), settings.shrink)
    assert "without exactly its folds" in _message(stop)
    _write_folds(directory)
    (directory / "test_2026_r0_f3.npz").unlink()
    with pytest.raises(SystemExit) as stop:
        build.require_folds(_cfg(tmp_path), settings.shrink)
    assert "test_2026_r0_f3.npz" in _message(stop)


def test_a_complete_folder_passes(tmp_path):
    settings = replace(PRODUCT, shrink=ONE_FOLDER)
    _write_folds(tmp_path / "within" / TAG)
    build.require_folds(_cfg(tmp_path), settings.shrink)


def test_the_folder_read_to_a_full_season_is_checked_too(tmp_path):
    """Experiment 101's second fold set, `extrapolate_folds_tag`, must hold complete folds as well."""
    _write_folds(tmp_path / "within" / TAG)
    with pytest.raises(SystemExit) as stop:
        build.require_folds(_cfg(tmp_path), replace(ONE_FOLDER, extrapolate_folds_tag="tenth"))
    assert str(tmp_path / "within" / "tenth") in _message(stop) and "--tag tenth" in _message(stop)


def test_every_folder_named_is_checked(tmp_path):
    """Several folders, comma-separated: each must hold complete folds, and the message names the one that does
    not."""
    shrink = replace(PRODUCT.shrink, folds_tag="early_within,late_within", fold_seasons=(2016, 2026),
                     extrapolate_folds_tag="")
    _write_folds(tmp_path / "within" / "early_within", season=2016)
    with pytest.raises(SystemExit) as stop:
        build.require_folds(_cfg(tmp_path), shrink)
    assert str(tmp_path / "within" / "late_within") in _message(stop) and "--tag late_within" in _message(stop)
    _write_folds(tmp_path / "within" / "late_within", season=2026)
    build.require_folds(_cfg(tmp_path), shrink)


def test_step_5_off_keeps_step_4_and_reads_nothing(tmp_path):
    """At an applied share of 0, shipped since experiment 82, step 5 hands step 4's table back with its two columns
    at 0 and opens no file: a build of one season, with no other season's multipliers, gets through it."""
    import pandas as pd
    step4 = pd.DataFrame({"player_id": [1, 2], "season": [2026, 2026], "rating_off": [1.5, -0.5],
                          "rating_def": [0.25, -1.0]})
    off = replace(PRODUCT, swap=replace(PRODUCT.swap, applied_share=0.0))
    final, factors, types = build.swap_seasons(_cfg(tmp_path), step4, off)
    pd.testing.assert_frame_equal(final.drop(columns=["c_off", "c_def"]), step4)
    assert (final.c_off == 0).all() and (final.c_def == 0).all()
    assert factors.empty and types.empty
