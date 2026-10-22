"""scripts/checks/151_wowy.py (the with-or-without-you test): team levels are fitted out exactly, a regular's absence is
counted only between two games he played for the team, and the script starts."""
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "151_wowy.py"


@pytest.fixture(scope="module")
def wowy():
    spec = importlib.util.spec_from_file_location("wowy_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_help_lists_its_flags():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "--systems" in result.stdout and "--unit" in result.stdout


def test_a_miss_made_only_of_team_levels_and_home_leaves_nothing(wowy):
    """Four teams, every pairing home and away: a miss of attacking level + defending level + home edge is fitted
    out to zero, and what is left of any miss sums to zero within each attacking team (weights equal)."""
    pairs = [(a, d) for a in range(4) for d in range(4) if a != d]
    teams = {"team_off": np.array([a for a, _ in pairs] * 2), "team_def": np.array([d for _, d in pairs] * 2)}
    home = np.r_[np.ones(len(pairs)), np.zeros(len(pairs))]
    fold = SimpleNamespace(y=np.zeros(2 * len(pairs)), F=np.column_stack([home, np.ones_like(home)]))
    levels = wowy.team_levels(fold, teams)
    attack, defend = np.array([1.0, -2.0, 0.5, 3.0]), np.array([0.0, 1.5, -1.0, 2.0])
    miss = attack[teams["team_off"]] + defend[teams["team_def"]] + 2.5 * home
    w = np.ones(len(miss))
    assert np.allclose(wowy.wowy_miss(miss, levels, w), 0.0, atol=1e-9)
    noise = np.random.default_rng(0).normal(size=len(miss))
    left = wowy.wowy_miss(noise, levels, w)
    assert np.allclose(pd.Series(left).groupby(teams["team_off"]).sum(), 0.0, atol=1e-9)


def test_a_regular_is_absent_only_between_two_games_he_played_for_that_team(wowy):
    """Team 7's regular (player 1) plays held-back games 1, 3 and 5 and sits 2 and 4; game 0 comes before his first
    game (he had not joined) and game 6 after his last (he had left): only 2 and 4 are absences.  Player 2 plays every
    game, so he is never absent."""
    n = 7
    played = [1, 3, 5]
    offence = np.zeros((n, 2))
    offence[played, 0] = 1.0
    offence[:, 1] = 1.0
    rows = pd.DataFrame({"player_id": [1, 2], "team_id": [7, 7], "poss_off": [1000.0, 200.0],
                         "poss_def": [1000.0, 200.0]})
    fold = SimpleNamespace(y=np.zeros(n), rows={"O": rows, "D": rows.iloc[:0]},
                           Z={"O": sp.csr_matrix(offence), "D": sp.csr_matrix((n, 0))}, game_idx=np.arange(n))
    teams = {"team_off": np.full(n, 7), "team_def": np.full(n, 8),
             "game_id": np.array([f"002{i:07d}" for i in range(n)])}
    absent = wowy.regulars_absent(fold, teams)
    assert np.flatnonzero(absent).tolist() == [2, 4]


def test_a_traded_regular_is_not_absent_after_the_trade_when_a_game_id_is_out_of_date_order(wowy):
    """Team 7's regular plays its first three games by date and is then traded.  The second of those games carries the
    season's last game id, as an NBA Cup game of 2023-24 does (0022301229, played 2023-12-07).  In id order his last
    game for team 7 would come after its three games without him and flag all three; in date order none is."""
    n = 6
    offence = np.zeros((n, 1))
    offence[[0, 1, 2], 0] = 1.0
    rows = pd.DataFrame({"player_id": [1], "team_id": [7], "poss_off": [1000.0], "poss_def": [1000.0]})
    fold = SimpleNamespace(y=np.zeros(n), rows={"O": rows, "D": rows.iloc[:0]},
                           Z={"O": sp.csr_matrix(offence), "D": sp.csr_matrix((n, 0))}, game_idx=np.arange(n))
    ids = ["0020000001", "0020000999", "0020000003", "0020000004", "0020000005", "0020000006"]
    teams = {"team_off": np.full(n, 7), "team_def": np.full(n, 8), "game_id": np.array(ids)}
    assert not wowy.regulars_absent(fold, teams).any()


def test_the_newey_west_standard_error_is_the_plain_one_at_lag_0_and_larger_when_neighbours_move_together(wowy):
    rng = np.random.default_rng(1)
    independent = pd.Series(rng.normal(size=30), index=range(1997, 2027))
    plain = independent.std(ddof=1) / np.sqrt(30)
    assert wowy.newey_west_se(independent, lags=0) == pytest.approx(plain)
    # a slow drift shared by neighbouring seasons: the plain standard error counts it 30 times
    drifting = pd.Series(np.repeat(rng.normal(size=10), 3) + 0.1 * rng.normal(size=30), index=range(1997, 2027))
    assert wowy.newey_west_se(drifting) > drifting.std(ddof=1) / np.sqrt(30)
    assert int(4 * (30 / 100) ** (2 / 9)) == 3                          # the default lag for thirty seasons
