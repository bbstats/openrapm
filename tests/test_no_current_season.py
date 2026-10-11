"""Nothing fitted may see a season still being played, checked end to end rather than at the guard.

`tests/test_seasons.py` pins the guard itself: a title is sixteen playoff wins, a season whose Finals
are unfinished is not complete, `Trainable` refuses a season past its cutoff.  That is necessary and
it is not the claim.  The claim is about the artifacts -- that no fit in this project has a season in
progress anywhere in it -- and for a long time `src/openrapm/data/seasons.py` named this file as the thing
that established it.  The file did not exist.

What it establishes now, in four parts:

  1. the guard drops the right rows, including a multi-season block that only REACHES INTO a season
     in progress, which is the case a per-season filter would get wrong;
  2. the rows a fit receives are byte-identical when the in-progress season is replaced by noise.
     That is the "not one byte moves" property the docstring promised, stated where it can be checked
     exactly: a deterministic fit on identical rows is identical, so the rows are the thing to pin;
  3. every script that FITS from the season panel gates its read.  This is the part that had actually
     gone wrong: `holdout.Context.load` gated the block panel, and five scripts read the season panel
     straight off the disk with no gate at all.  Latent rather than live -- no season is in progress
     today, so the guard drops nothing -- but it is the shape of thing that is only ever noticed in
     the year it starts lying;
  4. `openrapm rate` and `openrapm folds` read the panel through one function, `data.panel.trainable_panel`,
     which gates it, and the labels of `rate` sum only seasons whose Finals are over; step 5 (`openrapm swap`)
     reads it through `data.panel.panel_and_trainable_seasons`, so its type model trains only on finished
     seasons while still describing the players of a season in progress.
"""
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openrapm.data import seasons as S  # noqa: E402
from openrapm.data.config import load_config  # noqa: E402

CFG = load_config()


def _tree(tmp_path, finished_through: int, first: int = 2020, last: int = 2024):
    """A config whose tree has finished playoffs up to `finished_through` and unfinished ones after."""
    d = tmp_path / "data" / "raw" / "gamelog"
    d.mkdir(parents=True, exist_ok=True)
    for season in range(first, last + 1):
        wins = S.title_wins(season) if season <= finished_through else S.title_wins(season) - 1
        rows = []
        for i in range(wins):
            rows.append({"TEAM_ID": 0, "GAME_ID": f"{season}-{i}", "WL": "W"})
            rows.append({"TEAM_ID": 1, "GAME_ID": f"{season}-{i}", "WL": "L"})
        pd.DataFrame(rows).to_parquet(d / f"{season}_PO.parquet", index=False)
    return {**CFG, "_root": str(tmp_path), "first_season": first, "last_season": last,
            "paths": {**CFG.get("paths", {}), "raw": "data/raw"}}


def _panel(windows, rows_each: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "window": np.repeat(list(windows), rows_each),
        "player_id": np.tile(np.arange(rows_each), len(windows)),
        "x": rng.normal(size=rows_each * len(windows)),
    })


# --------------------------------------------------------------- 1. the guard drops the right rows
def test_a_block_that_only_reaches_into_the_current_season_is_dropped_whole(tmp_path):
    """The case a per-season filter gets wrong.  A 2022-2024 block carries 2024's evidence in every
    one of its rows, so there is no honest way to keep part of it."""
    cfg = _tree(tmp_path, finished_through=2023)
    assert S.trainable_through(cfg) == 2023
    panel = _panel(["2021-2021", "2021-2023", "2022-2024", "2024-2024"])
    kept, dropped = S.drop_untrainable(panel, cfg)
    assert set(dropped) == {"2022-2024", "2024-2024"}
    assert set(kept.window) == {"2021-2021", "2021-2023"}


def test_nothing_is_dropped_when_every_season_has_finished(tmp_path):
    cfg = _tree(tmp_path, finished_through=2024)
    panel = _panel(["2022-2024", "2024-2024"])
    kept, dropped = S.drop_untrainable(panel, cfg)
    assert not dropped and len(kept) == len(panel)


# ------------------------------------------------- 2. not one byte of the training rows moves
def test_replacing_the_current_season_with_noise_does_not_move_the_training_rows(tmp_path):
    """The promise, checked where it can be checked exactly.

    Every fit downstream draws its rows from here, and a deterministic fit on identical rows is
    identical -- so if the rows do not move, nothing built from them can.
    """
    cfg = _tree(tmp_path, finished_through=2023)
    clean = _panel(["2021-2021", "2022-2022", "2022-2024", "2024-2024"], seed=1)

    corrupted = clean.copy()
    in_progress = S.unit_last_season(corrupted.window) > S.trainable_through(cfg)
    assert in_progress.any(), "the fixture must actually contain a season in progress"
    rng = np.random.default_rng(99)
    corrupted.loc[in_progress, "x"] = rng.normal(loc=1e6, scale=1e6, size=int(in_progress.sum()))

    from_clean, _ = S.drop_untrainable(clean, cfg)
    from_noise, _ = S.drop_untrainable(corrupted, cfg)
    pd.testing.assert_frame_equal(from_clean, from_noise)
    assert not from_clean.x.isin(corrupted.loc[in_progress, "x"]).any()


# ------------------------------------------------------- 3. every fit from the panel is gated
# Scripts that FIT something from the season or block panel.  Each must pass its read through
# `drop_untrainable`; a new one belongs here the day it is written.
FITS_FROM_PANEL = ["88_yoy_by_player.py", "95_miss_by_group.py", "137_stat_calibration.py", "138_heldout_audit.py"]

# Scripts that read the panel to WRITE columns back into it.  These must NOT gate: they rewrite the
# file in place, so dropping the current season would delete it from the only copy.
WRITES_THE_PANEL = {"49_role_panel.py", "65_offcourt_panel.py", "69_closeness_panel.py", "86_context_panel.py"}


def _calls(path: Path, name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(isinstance(n, ast.Call)
               and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name)
               for n in ast.walk(tree))


@pytest.mark.parametrize("script", FITS_FROM_PANEL)
def test_a_script_that_fits_from_the_panel_gates_its_read(script):
    path = ROOT / "scripts" / script
    assert path.exists(), f"{script} is on FITS_FROM_PANEL but does not exist -- update the list"
    assert _calls(path, "drop_untrainable"), (
        f"scripts/{script} reads the panel and fits from it without calling seasons.drop_untrainable.  "
        f"Every fit downstream of the panel -- the box prior, the blend coefficients, the feature "
        f"selection -- has to be gated at the read, or the year a season is in progress it will train "
        f"on it and say nothing.")


@pytest.mark.parametrize("script", sorted(WRITES_THE_PANEL))
def test_a_script_that_writes_the_panel_does_not_gate(script):
    """The opposite mistake, and the worse one: these rewrite the panel in place."""
    path = ROOT / "scripts" / script
    assert not _calls(path, "drop_untrainable"), (
        f"scripts/{script} writes the panel back in place; gating its read would drop the season in "
        f"progress out of the only copy of the file.  Gate the fits, not the writers.")


def test_no_script_reads_a_panel_unclassified():
    """A new reader must be put on one list or the other, so nothing lands here unnoticed."""
    known = set(FITS_FROM_PANEL) | WRITES_THE_PANEL
    readers = {p.name for p in sorted((ROOT / "scripts").glob("[0-9]*.py"))
               if "role_panel" in p.read_text(encoding="utf-8")
               and "read_parquet" in p.read_text(encoding="utf-8")}
    assert not readers - known, (
        f"scripts {sorted(readers - known)} read a role panel and are on neither list.  Add each to "
        f"FITS_FROM_PANEL (and gate its read) or to WRITES_THE_PANEL (and do not).")


# ------------------------------------------- 4. the commands read the panel through one gate
SRC = ROOT / "src" / "openrapm"
READS_PANEL_PATH = re.compile(r"""resolve\(\s*cfg\s*,\s*["']panel["']\s*\)""")


def test_the_panel_file_is_opened_in_one_place_and_gated_there():
    """Only data/panel.py resolves paths.panel, and it passes what it reads through drop_untrainable."""
    openers = sorted(p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
                     if READS_PANEL_PATH.search(p.read_text(encoding="utf-8")))
    assert openers == ["data/panel.py"]
    assert _calls(SRC / "data" / "panel.py", "drop_untrainable")


@pytest.mark.parametrize("command", ["build.py", "within_season.py"])
def test_a_command_that_fits_reads_the_gated_panel(command):
    path = SRC / "commands" / command
    assert _calls(path, "trainable_panel"), f"commands/{command} must read the panel through trainable_panel"
    assert "role_panel" not in path.read_text(encoding="utf-8"), f"commands/{command} names the panel file itself"


def test_the_gated_panel_drops_the_season_in_progress(tmp_path):
    from openrapm.data.panel import trainable_panel
    cfg = _tree(tmp_path, finished_through=2023)
    stored = tmp_path / "panel.parquet"
    _panel(["2022", "2023", "2024"]).to_parquet(stored, index=False)
    cfg["paths"] = {**cfg["paths"], "panel": str(stored)}
    assert sorted(set(trainable_panel(cfg).window)) == ["2022", "2023"]


def test_rate_sums_only_finished_seasons_for_its_labels(tmp_path, monkeypatch):
    """A season in progress may be rated, but its games never reach the labels the prior learns."""
    from openrapm.commands import build
    from openrapm.settings import PRODUCT
    cfg = _tree(tmp_path, finished_through=2023)            # seasons 2020-2024, 2024 still being played

    class Stop(Exception):
        pass

    seen = {}

    def record(designs, seasons, settings):
        seen["seasons"] = list(seasons)
        raise Stop

    monkeypatch.setattr(build, "accumulate_rapm", record)
    with pytest.raises(Stop):
        build.run_rate(cfg, PRODUCT, [2024], "never_written")
    assert seen["seasons"] == [2020, 2021, 2022, 2023]


def test_the_swap_step_reads_the_panel_through_its_gate():
    path = SRC / "commands" / "build.py"
    assert _calls(path, "panel_and_trainable_seasons"), "step 5 must read the panel through its gate"
    assert _calls(SRC / "data" / "panel.py", "drop_untrainable")


def test_the_swap_step_describes_a_season_in_progress_but_never_trains_on_it(tmp_path):
    from openrapm.data.panel import panel_and_trainable_seasons
    cfg = _tree(tmp_path, finished_through=2023)
    stored = tmp_path / "panel.parquet"
    panel = _panel(["2022", "2023", "2024"])
    panel.assign(season=panel.window.astype(int)).to_parquet(stored, index=False)
    cfg["paths"] = {**cfg["paths"], "panel": str(stored)}
    whole, trainable = panel_and_trainable_seasons(cfg)
    assert sorted(set(whole.window)) == ["2022", "2023", "2024"]
    assert trainable == {2022, 2023}


def test_the_swap_type_model_never_trains_on_a_season_outside_the_trainable_set():
    from openrapm.swap_adjust import training_seasons
    assert training_seasons(2023, [2020, 2021, 2022, 2023, 2024], {2020, 2021, 2022, 2023}, 0) == [2020, 2021, 2022]
    assert training_seasons(2022, [2020, 2021, 2022, 2023, 2024], {2020, 2021, 2022, 2023}, 1) == [2020]
