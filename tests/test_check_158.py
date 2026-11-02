"""scripts/checks/158_high_leverage.py: a planted high-leverage skill is found and repeats, none is not, and the
controls keep what they say they keep."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from openrapm.simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "158_high_leverage.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("high_leverage_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _league(skill_sd=0.0, n_players=60, n_stints=6000, seed=0):
    """Stints of ten random players, a miss of per-100 noise over the possessions plus the five home and five away
    planted skills.  Returns the signed matrix, the misses, the weights, the skills."""
    rng = np.random.default_rng(seed)
    skills = rng.normal(0.0, skill_sd, n_players) if skill_sd else np.zeros(n_players)
    rows, columns, values = [], [], []
    weight = rng.integers(2, 12, n_stints).astype(float)
    miss = np.zeros(n_stints)
    for stint in range(n_stints):
        ten = rng.choice(n_players, 10, replace=False)
        rows += [stint] * 10
        columns += list(ten)
        values += [1.0] * 5 + [-1.0] * 5
        miss[stint] = skills[ten[:5]].sum() - skills[ten[5:]].sum() + rng.normal(0.0, 150.0 / np.sqrt(weight[stint]))
    signed = sp.csr_matrix((values, (rows, columns)), shape=(n_stints, n_players))
    return signed, miss, weight, skills


def test_high_leverage_is_the_wins_own_leverage_so_a_late_blowout_is_out_and_a_close_game_is_in(check):
    """The leverage decides, with no clock rule beside it: a game within two points counts in the third quarter too."""
    league = simulate(n_seasons=1, n_teams=6, players_per_team=9, games_per_season=200, stints_per_game=(20, 30),
                      seed=3)["stints"].copy()
    league["home_team_id"], league["away_team_id"] = league.h1 // 1000, league.a1 // 1000
    relative = check.stint_weight(league, 2001)
    period, margin = league.period.to_numpy(), np.abs(league.margin_h.to_numpy(dtype=float))
    assert np.average(relative, weights=league.poss_h + league.poss_a) == pytest.approx(1.0)
    late = (period == 4) & (league.start_clock.to_numpy() <= 300.0)
    assert relative[late & (margin >= 20)].max() < check.HIGH_LEVERAGE
    assert (relative[(period == 3) & (margin <= 2)] >= 1.0).mean() > 0.5


def test_a_trip_at_one_end_is_read_against_that_ends_possessions(check):
    """A shooter subbed in for one trip must not read as a star: the ball at one end is not a margin per possession."""
    five = dict(h1=1, h2=2, h3=3, h4=4, h5=5, a1=6, a2=7, a3=8, a4=9, a5=10)
    stints = pd.DataFrame([dict(poss_h=1, poss_a=0, pts_h=1.1, pts_a=0.0, **five),
                           dict(poss_h=0, poss_a=1, pts_h=0.0, pts_a=1.1, **five),
                           dict(poss_h=0, poss_a=0, pts_h=0.0, pts_a=0.0, **five)])
    ratings = pd.DataFrame({"player_id": range(1, 11), "rating_off": 0.0, "rating_def": 0.0})
    miss, weight = check.stint_misses(stints, ratings)
    assert miss.tolist() == pytest.approx([0.0, 0.0, 0.0]) and weight.tolist() == [0.5, 0.5, 0.0]
    rated = ratings.assign(rating_off=lambda frame: np.where(frame.player_id == 1, 10.0, 0.0))
    assert check.stint_misses(stints, rated)[0][0] == pytest.approx(-100.0 * 0.1 / 0.5)      # 1.2 expected, 1.1 scored
    assert check.stint_misses(stints, rated)[0][1] == pytest.approx(0.0)                     # he was not on offence


def test_a_planted_skill_comes_back_and_no_skill_stays_inside_its_standard_errors(check):
    signed, miss, weight, skills = _league(skill_sd=6.0)
    everything = np.ones(len(miss), dtype=bool)
    at_home = np.ones(len(miss), dtype=bool)
    table = check.deviations(signed, at_home, miss, weight, everything, check.PENALTY)
    assert np.corrcoef(table.deviation, skills)[0, 1] > 0.9
    signed, miss, weight, _ = _league()
    table = check.deviations(signed, at_home, miss, weight, everything, check.PENALTY)
    z = table.deviation / table.standard_error
    assert (z ** 2).mean() < 1.3 and (z.abs() > 3.0).sum() <= 1


def test_the_counts_keep_players_with_the_minimum_possessions(check):
    table = pd.DataFrame({"deviation": [5.0, 5.0, 1.0], "standard_error": [1.0, 2.0, 1.0],
                          "possessions": [100.0, 100.0, 10.0]})
    got = check.beyond(table)
    assert got["players"] == 2 and got["beyond_2"] == 2 and got["beyond_3"] == 1


def test_the_shuffle_keeps_each_games_count_of_stints(check):
    games = np.repeat(["a", "b", "c"], [10, 20, 30])
    flag = np.zeros(60, dtype=bool)
    flag[[1, 2, 15, 40, 41, 42]] = True
    drawn = check.shuffled_flag(games, flag, np.random.default_rng(1))
    counts = pd.Series(drawn).groupby(games).sum()
    assert counts.tolist() == [2, 1, 3] and not np.array_equal(drawn, flag)


def test_a_real_skill_repeats_between_separate_games_and_no_skill_does_not(check):
    signed, miss, weight, _ = _league(skill_sd=6.0)
    at_home = np.ones(len(miss), dtype=bool)
    first, second = np.arange(len(miss)) % 2 == 0, np.arange(len(miss)) % 2 == 1

    def pairs(miss_used):
        return check.repeat_table(check.deviations(signed, at_home, miss_used, weight, first, check.PENALTY),
                                  check.deviations(signed, at_home, miss_used, weight, second, check.PENALTY))

    real = check.repeat_reading(pairs(miss))
    assert real["correlation"] > 0.5 and real["z"] > 2.0 and real["real_part_sd"] > 2.0
    _, noise, _, _ = _league(seed=2)
    noise_reading = check.repeat_reading(pairs(noise))
    assert abs(noise_reading["correlation"]) < 0.4


def _verdict(check, real, shuffled, flipped, half_correlation, next_correlation):
    rng = np.random.default_rng(0)
    lines = pd.DataFrame({"beyond_2": [real], "beyond_3": [0], "players": [1000]})
    controls = pd.DataFrame([dict(control=name, draw=draw, beyond_2=level + (draw % 2))
                             for name, level in (("shuffled", shuffled), ("flipped", flipped)) for draw in range(10)])

    def correlated(correlation):
        one = rng.normal(0.0, 1.0, 2000)
        two = correlation * one + np.sqrt(1.0 - correlation ** 2) * rng.normal(0.0, 1.0, 2000)
        return pd.DataFrame({"one": one, "two": two, "weight": 1.0})

    return check.verdict(lines, controls, correlated(half_correlation), correlated(next_correlation), correlated(0.0))


def _parts(check, rating_error_sd=0.0, skill_sd=0.0, n_stints=6000, seed=0):
    """A season's parts: every player's rating wrong by a planted amount on all his stints, a planted skill on a fifth
    of the stints, the high-leverage ones, alone."""
    rng = np.random.default_rng(seed)
    signed, noise, weight, _ = _league(n_stints=n_stints, seed=seed)
    error = rng.normal(0.0, rating_error_sd, signed.shape[1]) if rating_error_sd else np.zeros(signed.shape[1])
    skill = rng.normal(0.0, skill_sd, signed.shape[1]) if skill_sd else np.zeros(signed.shape[1])
    flag = rng.random(n_stints) < 0.2
    miss = noise + np.asarray(signed @ error).ravel() + flag * np.asarray(signed @ skill).ravel()
    return dict(signed=signed, miss=miss, weight=weight, flag=flag, at_home=np.ones(n_stints, dtype=bool),
                player_ids=np.arange(signed.shape[1]), half=np.where(np.arange(n_stints) % 2 == 0, "A", "B"),
                games=np.arange(n_stints) // 20)


def test_a_rating_wrong_everywhere_repeats_only_until_the_rest_is_taken_off(check):
    parts = _parts(check, rating_error_sd=8.0)
    plain = check.repeat_reading(check.plain_half_pairs(parts, parts["flag"]))
    assert plain["correlation"] > 0.5 and plain["z"] > 2.0
    assert abs(check.repeat_reading(check.half_pairs(parts, parts["flag"]))["correlation"]) < 0.35


def test_a_real_high_leverage_skill_still_repeats_with_the_rest_taken_off(check):
    parts = _parts(check, rating_error_sd=8.0, skill_sd=12.0)
    reading = check.repeat_reading(check.half_pairs(parts, parts["flag"]))
    assert reading["correlation"] > 0.5 and reading["z"] > 2.0


def test_the_rule_needs_every_reading(check):
    assert _verdict(check, 100, 50, 50, 0.3, 0.3)["passes"]
    assert not _verdict(check, 50, 50, 50, 0.3, 0.3)["passes"]         # no more players out than the controls
    assert not _verdict(check, 100, 50, 50, 0.0, 0.3)["passes"]        # half A does not repeat in half B
    assert not _verdict(check, 100, 50, 50, 0.3, 0.0)["passes"]        # a season does not repeat in the next


def test_the_script_starts():
    done = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert done.returncode == 0 and "--tag" in done.stdout
