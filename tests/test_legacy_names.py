"""The old names (src/openrapm/data/legacy_names.py): files written before 2026-10-11 read under the plain names.

Three kinds of file keep the old names and are translated as they are read: the season panel (migrated once, by
`openrapm data panel --migrate-from`), the saved boosters and the within-season player tables.  What must hold: no
value or row moves, a booster's inputs keep their positions, and an old panel is never read by mistake.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openrapm.data import legacy_names as legacy
from openrapm.data.panel import migrate_panel, panel_from_old_names, trainable_panel
from openrapm.features import CATALOGUE, SEASON_PANEL_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
OLD_NAME = {new: old for old, new in legacy.COLUMNS.items()}
DEAD = ["window", "ps_idx", "apm", "spm", "u", "a", "rapm1", "pc_on_rtg_o", "pc_team_sos_d", "mix_lift", "mix_n"]


def _old_panel(rows: int = 6, seed: int = 0) -> pd.DataFrame:
    """A panel as scripts/49 (+ 65, 69, 86) wrote it: every season-panel column under its old name, the dead
    columns among them, in an order of its own."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({OLD_NAME.get(name, name): rng.normal(size=rows) for name in SEASON_PANEL_COLUMNS})
    frame["side"], frame["player_id"], frame["season"] = "O", np.arange(rows), 2020 + np.arange(rows) % 3
    for column in DEAD:
        frame[column] = 1.0
    return frame[list(reversed(frame.columns))]


def test_every_plain_name_is_a_catalogue_name():
    known = {feature.name for feature in CATALOGUE}
    assert not set(legacy.COLUMNS.values()) - known - {"team_net_per100"}
    assert len(set(legacy.COLUMNS.values())) == len(legacy.COLUMNS)           # no two old names share a new one


def test_the_panel_is_renamed_the_dead_columns_dropped_and_the_rows_kept():
    old = _old_panel()
    panel = panel_from_old_names(old)
    assert list(panel.columns) == SEASON_PANEL_COLUMNS
    assert panel.possessions.tolist() == old.poss.tolist() and panel.weight_lb.tolist() == old.weight.tolist()
    assert panel.threes_made_per100_uncentred.tolist() == old.raw_fg3m.tolist()
    assert panel.player_id.tolist() == list(range(6))


def test_a_panel_missing_a_column_or_carrying_an_unknown_one_is_refused():
    with pytest.raises(ValueError, match="missing"):
        panel_from_old_names(_old_panel().drop(columns=["exp_poss"]))
    with pytest.raises(ValueError, match="unknown"):
        panel_from_old_names(_old_panel().assign(mystery=0.0))


def test_a_frame_under_the_plain_names_is_left_as_it_is():
    panel = panel_from_old_names(_old_panel())
    assert legacy.translate_frame(panel).equals(panel)


def _cfg(tmp_path: Path) -> dict:
    return {"_root": str(tmp_path), "first_season": 2020, "last_season": 2022,
            "paths": {"panel": str(tmp_path / "outputs" / "season_panel.parquet")}}


def test_the_migration_writes_once_and_never_overwrites(tmp_path):
    cfg = _cfg(tmp_path)
    source = tmp_path / "outputs" / "role_panel_season.parquet"
    source.parent.mkdir()
    _old_panel().to_parquet(source, index=False)
    target, panel, dropped = migrate_panel(source, cfg)
    assert pd.read_parquet(target).equals(panel) and sorted(dropped) == sorted(DEAD)
    with pytest.raises(FileExistsError):
        migrate_panel(source, cfg)


def test_an_unmigrated_panel_is_refused_with_the_command_to_run(tmp_path):
    cfg = _cfg(tmp_path)
    old = tmp_path / "outputs" / "role_panel_season.parquet"
    old.parent.mkdir()
    _old_panel().to_parquet(old, index=False)
    with pytest.raises(FileNotFoundError, match="--migrate-from"):
        trainable_panel(cfg)
    cfg["paths"]["panel"] = str(old)                       # pointed straight at the old-named file
    with pytest.raises(ValueError, match="before 2026-10-11"):
        trainable_panel(cfg)


def test_saved_boosters_keep_their_objects_and_their_input_order():
    booster = object()
    side = dict(full=booster, folds=[booster], excluded=[{1}], feats=["ast", "creation", "weight"],
                model_feats=["ast", "creation", "weight", "chunk_poss", "chunk_seasons"], prior={1: 0.5})
    saved = {2026: {"O": side, "D": side}, "meta": {"features": "boruta_noonc", "schema": 2,
                                                   "settings": {"targets": {"offence": "xpts_ft",
                                                                            "defence": "x3def_w0.25"},
                                                                "prior": {"feature_set": "boruta_noonc"}}}}
    plain, renamed = legacy.saved_models(saved)
    assert renamed
    assert plain[2026]["D"]["model_feats"] == ["assists_per100", "assists_minus_turnovers_per100", "weight_lb",
                                               "row_possessions", "row_seasons"]
    assert plain[2026]["O"]["full"] is booster and plain[2026]["O"]["prior"] == {1: 0.5}
    assert plain["meta"]["settings"]["targets"] == {"offence": "points_with_expected_free_throws",
                                                    "defence": "points_with_expected_opponent_threes"}
    assert plain["meta"]["features"] == plain["meta"]["settings"]["prior"]["feature_set"] == "box_score_and_bio"
    assert saved[2026]["O"]["feats"] == ["ast", "creation", "weight"]                 # the original untouched
    assert legacy.saved_models(plain) == (plain, False)


def test_the_target_and_input_list_names():
    assert legacy.target_name("x3def_w0.25") == "points_with_expected_opponent_threes"
    assert legacy.target_name("x3def") == "points_with_expected_opponent_threes_all"
    assert legacy.target_name("xpts_ft") == "points_with_expected_free_throws"
    assert legacy.target_name("pts") == "pts"
    assert legacy.feature_set_name("boruta_noonc") == "box_score_and_bio"


OLD_PANEL = ROOT / "outputs" / "role_panel_season.parquet"


@pytest.mark.skipif(not OLD_PANEL.exists(), reason="no outputs/role_panel_season.parquet")
def test_the_real_old_panel_maps_onto_the_season_panel():
    old = pd.read_parquet(OLD_PANEL)
    panel = panel_from_old_names(old)
    assert len(panel) == len(old) and list(panel.columns) == SEASON_PANEL_COLUMNS
    for name in SEASON_PANEL_COLUMNS:
        assert panel[name].equals(old[OLD_NAME.get(name, name)].rename(name)), name
