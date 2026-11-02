"""scripts/checks/157_wins_model.py on a simulated season: the calibration's bins, the team sums, the penalty rule, the
long way round, and the script starts."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openrapm import swap_test, wins
from openrapm.simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "157_wins_model.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("wins_model_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def season():
    stints = simulate(n_seasons=1, n_teams=6, players_per_team=9, games_per_season=120, stints_per_game=(20, 30),
                      seed=3)["stints"].copy()
    stints["home_team_id"], stints["away_team_id"] = stints.h1 // 1000, stints.a1 // 1000
    stints["half"] = np.where(pd.factorize(stints.game_id)[0] % 2 == 0, "A", "B")
    seconds = swap_test.stint_durations(stints)
    player_ids = wins.player_index(stints)[0]
    rng = np.random.default_rng(0)
    offense, defense = rng.normal(0.0, 2.0, len(player_ids)), rng.normal(0.0, 1.5, len(player_ids))
    ratings = pd.DataFrame({"player_id": player_ids, "rating_off": offense, "rating_def": defense,
                            "rating_total": offense + defense})
    return stints, seconds, ratings, wins.fit_win_model(stints, seconds, 2001)


def test_the_calibration_covers_the_possessions_and_reads_from_the_team_ahead(check, season):
    stints, seconds, _, model = season
    cells = check.calibration(stints, seconds, model)
    assert cells.possessions.sum() <= (stints.poss_h + stints.poss_a).sum()
    assert cells.possessions.sum() >= 0.95 * (stints.poss_h + stints.poss_a).sum()
    assert cells.real_strength.between(0.0, 1.0).all() and cells.won.between(0.0, 1.0).all()
    assert (cells[cells.margin == "13+"].average_strength >= 0.5).all()      # ahead by 13 is never under even


def test_the_team_sums_keep_each_game_on_both_teams(check, season):
    stints, seconds, ratings, model = season
    teams = check.team_sums(stints, seconds, model, ratings, 27.0)
    assert len(teams) == 6
    assert teams.wins_above.sum() == pytest.approx(0.0) and teams.leverage_sum.sum() == pytest.approx(0.0)
    assert teams.rating_wins.sum() == pytest.approx(0.0)


def test_the_chosen_penalty_is_the_least_and_says_whether_it_is_interior(check):
    interior = pd.DataFrame({"penalty": [1.0, 10.0, 100.0], "error": [5.0, 4.0, 6.0]})
    edge = pd.DataFrame({"penalty": [1.0, 10.0, 100.0], "error": [3.0, 4.0, 6.0]})
    assert check.chosen_penalty(interior) == (10.0, True)
    assert check.chosen_penalty(edge) == (1.0, False)


def test_the_search_scores_both_directions_of_every_penalty(check, season):
    stints, seconds, _, _ = season
    lines = pd.DataFrame(check.search_season(stints, seconds, (1e3, 1e5), 15.0, 2001))
    assert len(lines) == 4 and set(lines.fit_half) == {"A", "B"} and (lines.error > 0.0).all()


def test_the_wins_equal_the_long_way_round(check, season):
    stints, seconds, ratings, model = season
    assert check.long_way_gap(stints, seconds, ratings, model) < 1e-9


def test_the_script_starts():
    done = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert done.returncode == 0 and "--part" in done.stdout
