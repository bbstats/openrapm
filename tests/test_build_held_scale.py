"""The season in progress (2026-10-30, experiment 109 adopted): its prior scales are held at the mean of the 8 finished
seasons nearest it until settings.RidgeSettings.held_scale_through of its regular season is played, and its rows tell
the box-score prior no less of a season than its training rows cover."""
from dataclasses import replace

import pandas as pd
import pytest

from openrapm import box_prior
from openrapm.commands import build
from openrapm.settings import PRODUCT


def test_the_held_scales_are_the_mean_of_the_nearest_finished_seasons(tmp_path, monkeypatch):
    scales = pd.DataFrame({"season": list(range(2015, 2028)), "prior_scale_off": [1.0 + 0.01 * k for k in range(13)],
                           "prior_scale_def": [0.9] * 12 + [5.0]})
    scales.to_parquet(tmp_path / "season_ratings_product_step3.parquet")
    cfg = {"_root": str(tmp_path), "paths": {"outputs": str(tmp_path)}}
    monkeypatch.setattr(build, "trainable_through", lambda cfg: 2026)
    held = build.held_scales(cfg, "season_ratings_product_step3", 2027)
    assert held == pytest.approx((sum(1.0 + 0.01 * k for k in range(4, 12)) / 8, 0.9))   # 2019-2026, never 2027
    assert build.held_scales(cfg, "season_ratings_missing", 2027) is None


def test_the_season_in_progress_holds_its_scales_only_before_the_share_set():
    assert 0 < PRODUCT.ridge.held_scale_through <= 1
    assert PRODUCT.ridge.held_scale_table == "season_ratings_product_step3"
    fitted = replace(PRODUCT, ridge=replace(PRODUCT.ridge, held_scale_through=0.0))
    assert fitted.ridge.held_scale_through == 0.0


def test_a_rated_row_reads_no_less_of_a_season_than_the_training_rows_cover():
    rows = pd.DataFrame({"player_id": [1, 2], "season": [2027, 2027], "side": ["O", "O"], "possessions": [250.0, 90.0],
                         "points_per100": [20.0, 10.0]})
    assert box_prior.least_share(PRODUCT) == min(PRODUCT.prior.part_season_shares) == 0.25
    held = box_prior.rated_rows(rows, ["points_per100"], 0.058, box_prior.least_share(PRODUCT))
    assert held.row_seasons.tolist() == [0.25, 0.25] and held.row_possessions.tolist() == [250.0, 90.0]
    assert box_prior.rated_rows(rows, ["points_per100"], 1.0, 0.25).row_seasons.tolist() == [1.0, 1.0]
    whole_only = replace(PRODUCT, prior=replace(PRODUCT.prior, part_season_shares=()))
    assert box_prior.least_share(whole_only) == 1.0
