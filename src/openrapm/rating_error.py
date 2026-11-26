"""The standard error of a rating: how far it would land from the one shown had the season's games come out as
another draw of the same schedule.

A rating reads the season's games three times: the box score behind the box-score prior, the lineups and shots behind
Box Adj, and the points behind the ridge.  Play the same schedule again and all three move.  The standard error is the
typical size of that movement, in the rating's own unit, points per 100 possessions.

It is measured, not derived: the season is rated again with a tenth of its games held back, every input rebuilt from
the games kept (`openrapm folds --folds-per-repeat 10`), ten times over so that every game is held back once, and the
ten ratings' spread is scaled up to what losing a whole season's games would do.  This is the delete-a-group
jackknife:

    se^2 = (g - 1) / g  x  sum over the g folds of (rating_k - mean of the g ratings)^2

with g the folds of one repeat.  The games are assigned to folds several times over (three repeats as shipped); the
repeats' se^2 are averaged, which steadies the estimate without changing what it estimates.

What it holds: the noise of the games in everything a rating reads from them.
What it leaves out: how far the box-score prior can miss a player whatever games are played.  A rating is pulled
toward that prior, so the distance from a player's true value is larger than his standard error.

Words used below
    fold            one rating of the season made without a tenth of its games
    repeat          one assignment of every game of the season to the g folds
    complete repeat a repeat in which the player has a rating in every fold.  A player whose every game fell in one
                    fold has no rating there, and that repeat is left out for him

Example: a player whose ten folds rate him 5.0, 5.4, 4.6, 5.2, 4.8, 5.0, 5.3, 4.7, 5.1 and 4.9 has a mean of 5.0 and
a sum of squares of 0.60, so se = sqrt(0.9 x 0.60) = 0.73.

Sizes, 1997-2026 (notes/DECISIONS.md, "A standard error for each player's rating"): a player with 1,500 or more
offensive possessions reads about 0.55 on offense, 0.55 on defense and 0.80 on the total; under 250 possessions the
total reads 1.5.  The same jackknife on a quarter of the games held back, g = 4, reads 6 to 8% smaller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["RATINGS", "ERRORS", "jackknife"]

RATINGS = ("rating_off", "rating_def", "rating_total")
ERRORS = ("se_off", "se_def", "se_total")


def jackknife(ratings: pd.DataFrame, columns=RATINGS) -> pd.DataFrame:
    """One row per player: player_id, the standard error of each of `columns` (rating_off -> se_off), and repeats,
    the complete repeats it was read from.  A player with no complete repeat has no standard error, NaN.

    `ratings` holds one row per player, repeat and fold: player_id, repeat, fold and `columns`.  Every repeat must
    hold the same number of folds, at least two.

    Example: the module docstring's player, in one repeat of ten folds, reads 0.73; had a second repeat read
    se^2 = 0.40, he would read sqrt((0.54 + 0.40) / 2) = 0.69.
    """
    columns = list(columns)
    folds = ratings.groupby("repeat").fold.nunique()
    if folds.nunique() != 1 or int(folds.iloc[0]) < 2:
        raise ValueError(f"every repeat must hold the same number of folds, at least two; got {folds.to_dict()}")
    g = int(folds.iloc[0])
    by_repeat = ratings.groupby(["player_id", "repeat"])
    complete = by_repeat.fold.nunique() == g
    centred = ratings[columns] - by_repeat[columns].transform("mean")
    squares = (centred ** 2).groupby([ratings.player_id, ratings.repeat]).sum()[complete]
    variance = (g - 1) / g * squares.groupby("player_id").mean()
    out = np.sqrt(variance).rename(columns={column: column.replace("rating_", "se_") for column in columns})
    out["repeats"] = complete[complete].groupby("player_id").size()
    players = pd.Index(np.sort(ratings.player_id.unique()), name="player_id")
    out = out.reindex(players)
    out["repeats"] = out.repeats.fillna(0).astype(int)
    return out.reset_index()
