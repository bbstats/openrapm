"""The standard error of a rating (rating_error.py): the delete-a-group jackknife over a season's folds."""
import numpy as np
import pandas as pd
import pytest

from openrapm import rating_error


def _ratings(values, player_id=7, repeat=0, column="rating_off"):
    return pd.DataFrame({"player_id": player_id, "repeat": repeat, "fold": range(len(values)), column: values})


def test_the_docstrings_player_reads_0_73():
    values = [5.0, 5.4, 4.6, 5.2, 4.8, 5.0, 5.3, 4.7, 5.1, 4.9]
    got = rating_error.jackknife(_ratings(values), ["rating_off"])
    assert got.se_off.iloc[0] == pytest.approx(np.sqrt(0.9 * 0.60))
    assert round(got.se_off.iloc[0], 2) == 0.73 and got.repeats.iloc[0] == 1


def test_a_mean_of_games_gets_the_textbook_standard_error():
    """A rating that is the plain mean of its games: the jackknife over g equal folds is the standard error of the
    mean of the g fold means, sd / sqrt(g), exactly."""
    rng = np.random.default_rng(0)
    games = rng.normal(2.0, 12.0, size=(10, 120))           # ten folds of 120 games
    without = [(games.sum() - games[k].sum()) / (games.size - games.shape[1]) for k in range(10)]
    got = rating_error.jackknife(_ratings(without), ["rating_off"]).se_off.iloc[0]
    fold_means = games.mean(axis=1)
    assert got == pytest.approx(fold_means.std(ddof=1) / np.sqrt(10))
    assert got == pytest.approx(12.0 / np.sqrt(games.size), rel=0.5)      # and near the true 0.35


def test_the_repeats_variances_are_averaged():
    first = _ratings([1.0, 2.0, 3.0, 4.0], repeat=0)        # sum of squares 5.0, se^2 = 3.75
    second = _ratings([2.0, 2.0, 3.0, 3.0], repeat=1)       # sum of squares 1.0, se^2 = 0.75
    got = rating_error.jackknife(pd.concat([first, second]), ["rating_off"])
    assert got.se_off.iloc[0] == pytest.approx(np.sqrt((3.75 + 0.75) / 2)) and got.repeats.iloc[0] == 2


def test_a_repeat_with_a_fold_that_never_rated_him_is_left_out():
    whole = pd.concat([_ratings([1.0, 2.0, 3.0, 4.0], repeat=0), _ratings([2.0, 2.0, 3.0, 3.0], repeat=1)])
    short = pd.concat([_ratings([1.0, 2.0, 3.0, 4.0], player_id=8, repeat=0),
                       _ratings([2.0, 2.0, 3.0], player_id=8, repeat=1)])
    never = pd.concat([_ratings([1.0, 2.0, 3.0], player_id=9, repeat=0),
                       _ratings([2.0, 3.0], player_id=9, repeat=1)])
    got = rating_error.jackknife(pd.concat([whole, short, never]), ["rating_off"]).set_index("player_id")
    assert got.se_off[8] == pytest.approx(np.sqrt(3.75)) and got.repeats[8] == 1
    assert np.isnan(got.se_off[9]) and got.repeats[9] == 0
    assert list(got.index) == [7, 8, 9]


def test_every_rating_column_gets_its_own_error():
    table = _ratings([1.0, 3.0]).assign(rating_def=[0.0, 0.0], rating_total=[1.0, 3.0])
    got = rating_error.jackknife(table).iloc[0]
    assert list(rating_error.jackknife(table).columns) == ["player_id", *rating_error.ERRORS, "repeats"]
    assert got.se_off == pytest.approx(1.0) and got.se_def == 0.0 and got.se_total == pytest.approx(1.0)


def test_repeats_of_different_sizes_are_refused():
    with pytest.raises(ValueError, match="same number of folds"):
        rating_error.jackknife(pd.concat([_ratings([1.0, 2.0], repeat=0), _ratings([1.0, 2.0, 3.0], repeat=1)]),
                               ["rating_off"])
    with pytest.raises(ValueError, match="at least two"):
        rating_error.jackknife(_ratings([1.0]), ["rating_off"])
