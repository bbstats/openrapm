"""How a miss is reported (src/openrapm/scoring.py): a mean square read on the mean-absolute scale."""
import numpy as np
import pytest

from openrapm.scoring import MAE_SCALE, armse


def test_armse_is_the_root_mean_square_on_the_mean_absolute_scale():
    assert MAE_SCALE == pytest.approx(np.sqrt(2.0 / np.pi))
    assert armse(100.0) == pytest.approx(10.0 * MAE_SCALE)
    sample = np.random.default_rng(0).normal(0.0, 3.0, 400_000)
    assert armse(float((sample ** 2).mean())) == pytest.approx(float(np.abs(sample).mean()), rel=0.01)
