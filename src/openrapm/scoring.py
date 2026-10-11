"""How a miss is reported: a mean squared error read on the mean-absolute scale.

Every error is stored as a mean square, because mean squares pool and pair.  It is reported as ARMSE:

    ARMSE = sqrt(mean squared error) x sqrt(2 / pi)

A root mean square reads bigger than the typical miss, because squaring pays extra attention to the tail.  For a
normal error the mean absolute deviation is sqrt(2 / pi) = 0.7979 of the standard deviation, so the scaled root
says what a typical team-game misses by, in points per 100.

Example: a mean square of 100 is a root mean square of 10 and an ARMSE of 7.98.
"""
from __future__ import annotations

import numpy as np

__all__ = ["MAE_SCALE", "armse"]

MAE_SCALE = 0.7978845608028654          # sqrt(2 / pi)


def armse(mean_squared_error):
    """A mean squared error on the mean-absolute scale: sqrt(mse) x sqrt(2 / pi)."""
    return np.sqrt(np.asarray(mean_squared_error, dtype=float)) * MAE_SCALE
