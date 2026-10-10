"""OpenRAPM: open NBA player-impact ratings, one rating per player per season.

A ridge regression of every stint's points on the ten players on the court (RAPM), pulled toward
a box-score prior instead of toward zero. Read `rapm.py`, then `box_prior.py`, then `prior_rapm.py`.
"""

__version__ = "0.1.0"
