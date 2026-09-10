"""The trust boundary: a season may be fit on only once its Finals are over.

These tests do not need the scraped data.  `playoffs_complete` is exercised against a synthetic
game log, so the behaviour that matters -- a season in progress is refused -- is checked even in the
usual case where every season on disk is already finished.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef import seasons as S  # noqa: E402
from eracoef.config import load_config  # noqa: E402

CFG = load_config()


def _gamelog(champion_wins: int, teams: int = 16) -> pd.DataFrame:
    """A playoff game log in which the leading team has won `champion_wins` games."""
    rows = []
    gid = 0
    for t in range(teams):
        wins = champion_wins if t == 0 else max(0, champion_wins - 1 - t)
        for _ in range(wins):
            gid += 1
            rows.append({"TEAM_ID": t, "GAME_ID": f"g{gid}", "WL": "W"})
            rows.append({"TEAM_ID": teams + t, "GAME_ID": f"g{gid}", "WL": "L"})
    return pd.DataFrame(rows)


def _write(tmp_path, season, df):
    d = tmp_path / "data" / "raw" / "gamelog"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f"{season}_PO.parquet", index=False)
    return {**CFG, "_root": str(tmp_path), "paths": {**CFG.get("paths", {}), "raw": "data/raw"}}


# ------------------------------------------------------------------ the title-wins constant
def test_title_wins_is_16_from_2003_and_15_before():
    assert S.title_wins(2002) == 15 and S.title_wins(2003) == 16
    assert S.title_wins(1997) == 15 and S.title_wins(2026) == 16


def test_a_finished_playoffs_is_complete(tmp_path):
    cfg = _write(tmp_path, 2026, _gamelog(16))
    assert S.playoffs_complete(2026, cfg)


def test_a_season_in_progress_is_not_complete(tmp_path):
    """Fifteen wins in a modern season is a team one game from the title -- still not the title."""
    cfg = _write(tmp_path, 2026, _gamelog(15))
    assert not S.playoffs_complete(2026, cfg)


def test_the_1997_first_round_was_best_of_five(tmp_path):
    """Fifteen wins DID win the title before 2003, so the same log is complete in 1999 and not 2026."""
    cfg = _write(tmp_path, 1999, _gamelog(15))
    assert S.playoffs_complete(1999, cfg)


def test_a_season_with_no_playoff_file_is_not_complete(tmp_path):
    cfg = _write(tmp_path, 2026, _gamelog(16))
    assert not S.playoffs_complete(2027, cfg)


# ------------------------------------------------------------------ the types
def test_trainable_refuses_a_season_past_its_cutoff():
    with pytest.raises(S.LeakageError, match="2027"):
        S.Trainable((2025, 2026, 2027), 2026)


def test_trainable_accepts_its_cutoff_season():
    assert len(S.Trainable((2025, 2026), 2026)) == 2


def test_without_drops_a_held_out_season_and_keeps_the_cutoff():
    tr = S.Trainable((2023, 2024, 2025), 2025).without(2024)
    assert tr.seasons == (2023, 2025) and tr.cutoff == 2025


def test_check_trainable_passes_a_trainable_through_unchanged():
    tr = S.Trainable((2024, 2025), 2025)
    assert S.check_trainable(tr, CFG) is tr


def test_check_trainable_names_the_caller_in_its_error():
    with pytest.raises(S.LeakageError, match="the box prior"):
        S.check_trainable([S.trainable_through(CFG) + 1], CFG, "the box prior")


# ------------------------------------------------------------------ against the real data
def test_the_real_cutoff_is_a_season_we_have():
    cut = S.trainable_through(CFG)
    assert int(CFG["first_season"]) <= cut <= int(CFG["last_season"])


def test_trainable_and_in_progress_partition_every_season():
    tr, ip = S.trainable(CFG), S.in_progress(CFG)
    assert tuple(sorted([*tr.seasons, *ip])) == S.all_seasons(CFG)
    assert not set(tr.seasons) & set(ip)


def test_nothing_in_progress_is_ever_trainable():
    assert all(s > S.trainable_through(CFG) for s in S.in_progress(CFG))


def test_the_cutoff_follows_the_tree_it_is_pointed_at(tmp_path):
    """The regression behind the cache key: a cfg rooted at a temporary tree must read THAT tree.

    2026 is complete in the real data.  Here only 2024 is, so the cutoff must be 2024 -- if the
    cache or the lookup falls back to the repo's own config, this returns 2026 and fails.
    """
    cfg = _write(tmp_path, 2024, _gamelog(16))
    _write(tmp_path, 2025, _gamelog(9))          # a season abandoned mid-playoffs
    _write(tmp_path, 2026, _gamelog(15))         # and one a game from the title
    cfg = {**cfg, "first_season": 2020, "last_season": 2026}
    assert S.trainable_through(cfg) == 2024
    assert S.in_progress(cfg) == (2025, 2026)
    assert S.trainable(cfg).seasons == (2020, 2021, 2022, 2023, 2024)
