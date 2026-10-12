"""scripts/checks/73_tradeloss.py starts, answers to the flags the experiment chain types, and pairs two arms by
season on the player-season-sides both have: checked end to end on two small alpha tables whose answer is known."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "73_tradeloss.py"
MAE_SCALE = 0.7978845608028654          # sqrt(2 / pi)
SEASONS = (2001, 2002, 2003, 2004)


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, encoding="utf-8",
                          cwd=ROOT, timeout=300)


def alpha_table(size_of_season) -> pd.DataFrame:
    """Ten eligible players a season on both sides, corrections of +size and -size in turn."""
    rows = []
    for season in SEASONS:
        for side in ("offense", "defense"):
            for player in range(1, 11):
                sign = 1.0 if player % 2 else -1.0
                rows.append(dict(player_id=player, season=season, side=side, alpha_good=sign * size_of_season(season),
                                 eligible=True, without_poss=200.0, poss_on=1000.0, penalty=10000.0))
    return pd.DataFrame(rows)


@pytest.fixture
def arms(tmp_path) -> tuple:
    """The reference corrects every player by 2.0; the candidate by 1.0, 1.1, 1.2, 1.3 in the four seasons, and has
    no row for one player-season-side."""
    ref = alpha_table(lambda season: 2.0)
    cand = alpha_table(lambda season: 1.0 + 0.1 * (season - SEASONS[0]))
    cand = cand[~((cand.player_id == 10) & (cand.season == 2004) & (cand.side == "offense"))]
    paths = (tmp_path / "ref.parquet", tmp_path / "cand.parquet")
    ref.to_parquet(paths[0], index=False)
    cand.to_parquet(paths[1], index=False)
    return paths


def test_help_lists_every_flag_the_chain_and_the_docstring_use():
    result = run("--help")
    assert result.returncode == 0, result.stderr
    for flag in ("--alphas", "--ref", "--quality", "--tier", "--min_without", "--movers"):
        assert flag in result.stdout


def test_no_alphas_prints_the_docstring():
    result = run()
    assert result.returncode == 1 and "Check 73, the trade loss" in result.stderr


def test_the_paired_difference_on_shared_players(arms):
    result = run(f"--alphas=ref={arms[0]},cand={arms[1]}", "--ref", "ref")
    assert result.returncode == 0, result.stderr
    assert "79 player-season-sides in both; dropped 0 from cand and 1 from ref" in result.stdout
    # each season's loss: MAE_SCALE x the size, so the candidate minus the reference averages
    # MAE_SCALE x (mean of 1.0, 1.1, 1.2, 1.3 - 2.0) = -0.6782, better in all four seasons
    expected = f"{MAE_SCALE * (np.mean([1.0, 1.1, 1.2, 1.3]) - 2.0):.4f}"
    lines = {line.split()[0]: line.split() for line in result.stdout.splitlines()
             if line.split()[:1] in (["offense"], ["defense"])}
    # columns: side seasons players cand ref mean_diff se z wins
    assert lines["offense"][2] == "39" and lines["defense"][2] == "40"
    for side in ("offense", "defense"):
        assert lines[side][1] == "4" and lines[side][5] == expected and lines[side][-1] == "4", lines[side]
