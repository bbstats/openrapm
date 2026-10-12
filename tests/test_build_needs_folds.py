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

TAG = PRODUCT.shrink.folds_tag
MAKE_RATE = "openrapm rate --setting test --seasons 2017-2026 --save-models NAME"
MAKE_FOLDS = f"openrapm folds --models NAME --base season_ratings_test_step3 --seasons 2017-2026 --tag {TAG}"


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
    settings = replace(PRODUCT, shrink=replace(PRODUCT.shrink, fold_seasons=(2026, 2026)))
    directory = tmp_path / "within" / TAG
    _write_folds(directory, folds=range(3))
    with pytest.raises(SystemExit) as stop:
        build.require_folds(_cfg(tmp_path), settings.shrink)
    assert "four folds" in _message(stop)
    _write_folds(directory)
    (directory / "test_2026_r0_f3.npz").unlink()
    with pytest.raises(SystemExit) as stop:
        build.require_folds(_cfg(tmp_path), settings.shrink)
    assert "test_2026_r0_f3.npz" in _message(stop)


def test_a_complete_folder_passes(tmp_path):
    settings = replace(PRODUCT, shrink=replace(PRODUCT.shrink, fold_seasons=(2026, 2026)))
    _write_folds(tmp_path / "within" / TAG)
    build.require_folds(_cfg(tmp_path), settings.shrink)
