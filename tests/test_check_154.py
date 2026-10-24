"""scripts/checks/154_self_creation.py (self-creation in lineups): the owner's formula and its clip, the concentration,
an effect planted in held-back misses comes back through the per-fold team levels, a trade is dated from its first
game for the new team, and the script starts."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "154_self_creation.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("self_creation_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_help_lists_its_flags():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "--folds" in result.stdout and "--trade" in result.stdout


def test_the_share_assisted_is_the_owners_formula_clipped_to_zero_through_one(check):
    """10.2 assists and 11.3 makes per 100 give 0.73 - 0.306 - 0.113 = 0.311; a heavy passer and scorer clips at 0."""
    shares = check.share_assisted([10.2, 30.0, 0.0], [11.3, 20.0, 0.0])
    assert np.allclose(shares, [0.311, 0.0, 0.73])


def test_self_creation_reproduces_the_example_of_the_docstring(check):
    """LeBron James 2026, one fold's padded rates: 25.76 true shooting attempts x 0.689 = 17.75."""
    rates = {"twos_made": 9.246003, "twos_missed": 7.040046, "threes_made": 2.070452, "threes_missed": 4.011177,
             "free_throws_made": 5.583307, "free_throws_missed": 2.125459, "assists": 10.199733}
    rows = pd.DataFrame({f"{name}_per100_uncentred": [value] for name, value in rates.items()})
    assert check.self_creation(rows)[0] == pytest.approx(17.752, abs=1e-3)


def test_concentration_is_point_eight_for_equal_shares_and_zero_for_one_holder(check):
    values = np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [5.0, 0.0, 0.0, 0.0, 0.0], [2.0, 1.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0]])
    assert np.allclose(check.concentration(values), [0.8, 0.0, 0.625, 0.0])


def test_an_effect_planted_beside_team_levels_comes_back_through_the_fold_projection(check):
    """Three folds, six teams: each fold's misses are its own team levels, a home edge, 0.3 x margin, 1.5 x amount
    and noise.  Projecting the levels out fold by fold and pooling gives back 1.5 and 0.3."""
    rng = np.random.default_rng(7)
    parts = []
    for fold in range(3):
        n = 900
        frame = pd.DataFrame({"team_off": rng.integers(0, 6, n), "team_def": rng.integers(0, 6, n),
                              "is_home_off": rng.integers(0, 2, n).astype(float),
                              "playoffs": np.zeros(n), "amount": rng.normal(45, 6, n), "margin": rng.normal(0, 8, n)})
        attack, defend = rng.normal(0, 3, 6), rng.normal(0, 3, 6)
        miss = (attack[frame.team_off] + defend[frame.team_def] + 2.0 * frame.is_home_off + 0.3 * frame.margin
                + 1.5 * frame.amount + rng.normal(0, 4, n))
        w = rng.uniform(1, 10, n)
        levels = check.team_levels(frame)
        columns = check.project(levels, w, frame[["amount", "margin"]].to_numpy(float))
        target = check.project(levels, w, miss.to_numpy(float)[:, None]).ravel()
        parts.append((target, columns, w))
    target, columns, w = (np.concatenate([part[k] for part in parts]) for k in range(3))
    assert np.allclose(check.fit(target, columns, w), [1.5, 0.3], atol=0.05)


def test_the_null_projection_equals_the_one_of_the_readings(check):
    rng = np.random.default_rng(3)
    n = 400
    frame = pd.DataFrame({"team_off": rng.integers(0, 5, n), "team_def": rng.integers(0, 5, n),
                          "is_home_off": rng.integers(0, 2, n).astype(float), "playoffs": np.zeros(n)})
    levels, w, columns = check.team_levels(frame), rng.uniform(1, 9, n), rng.normal(size=(n, 2))
    assert np.allclose(check.projector(levels, w)(columns), check.project(levels, w, columns), atol=1e-8)


def test_a_donor_hands_over_the_same_step_of_his_career_or_his_last(check):
    """Player 1 plays 2000-2002, player 2 2010-2011; swapped, player 1's third season takes player 2's last one and
    player 2's first takes player 1's first.  Self-creation here is 0.27 x the attempts, no assists, no makes."""
    rates = ("twos_made", "twos_missed", "threes_made", "threes_missed", "free_throws_made", "free_throws_missed",
             "assists")
    attempts = [10.0, 20.0, 30.0, 40.0, 50.0]
    panel = pd.DataFrame({"side": "O", "player_id": [1, 1, 1, 2, 2], "season": [2000, 2001, 2002, 2010, 2011],
                          **{f"{name}_per100_uncentred": 0.0 for name in rates}})
    panel["twos_missed_per100_uncentred"] = attempts
    career = check.Career(panel, 2026)
    swap = np.array([1, 0])
    assert career.donated(swap, np.array([1, 2]), 2002)[0] == pytest.approx(0.27 * 50.0)
    assert career.donated(swap, np.array([2]), 2010)[0] == pytest.approx(0.27 * 10.0)
    assert np.isnan(career.donated(swap, np.array([1, 3]), 2011)).all()


def test_after_a_trade_runs_from_the_first_move_on_or_after_the_first_held_back_game(check):
    """Held-back games start at 5.  Team 1 moved at 3, before them, and again at 8; team 2 at 6; team 3 never."""
    frame = pd.DataFrame({"team_off": [1, 1, 1, 2, 2, 3], "game_idx": [5, 8, 9, 5, 7, 9],
                          "amount": np.ones(6), "concentration": np.ones(6)})
    moves = pd.DataFrame({"team_id": [1, 1, 2], "game_idx": [3, 8, 6]})
    check.add_trade(frame, moves)
    assert frame.trade.tolist() == [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]


def test_a_move_dates_both_teams_by_his_first_game_for_the_new_one_and_skips_small_players(check):
    """Player 7 plays games 0-4 for team 10 and 5-9 for team 20; player 8 moves too but had 100 possessions."""
    box = pd.DataFrame({"game_id": [f"g{k}" for k in range(10)] * 2, "player_id": [7] * 10 + [8] * 10,
                        "team_id": [10] * 5 + [20] * 5 + [10] * 3 + [30] * 7, "minutes": 30.0})
    games = pd.DataFrame({"game_id": [f"g{k}" for k in range(10)], "game_idx": range(10)})
    moves = check.trade_moves(box, games, pd.Series({7: 900.0, 8: 100.0}))
    assert moves.values.tolist() == [[10, 5], [20, 5]]
