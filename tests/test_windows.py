"""The plug-in ratings table (src/openrapm/windows.py) on simulated seasons."""
import numpy as np
import pytest

from openrapm.design import FEATURES, build_design
from openrapm.simulate import simulate
from openrapm.windows import player_ratings_table

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500, "box_counts": FEATURES,
       "lam_plugin": 4000.0, "lam_ratio_plugin": 1.0, "pad_target": "league"}


@pytest.fixture(scope="module")
def design():
    sim = simulate(n_seasons=5, n_teams=8, players_per_team=10, games_per_season=60, stints_per_game=(20, 30),
                   rho=0.85, turnover=0.3, seed=11, eps_var=0.2, leak=False)
    st, box = sim["stints"], sim["box"]
    seasons = [2002, 2004]
    return build_design(st[st.season.isin(seasons)], box[box.season.isin(seasons)], FEATURES, CFG), seasons


def test_ratings_table_prior_parts(design):
    """With the chain offset, prior_* carry the offset (defence flipped), and rating = prior + u holds with no
    linear box term."""
    wd, seasons = design
    m = wd.spec.n_ps
    v = np.random.default_rng(7).normal(0, 1.0, 2 * m)
    zeros = np.zeros(2 * len(FEATURES))
    t = player_ratings_table(wd, zeros, CFG, seasons, prior_offset=v, prior_parts=(v[:m], v[m:]))
    assert np.allclose(t.prior_off, v[:m]) and np.allclose(t.prior_def, -v[m:])
    plain = player_ratings_table(wd, zeros, CFG, seasons)
    assert np.allclose(plain.prior_off, 0.0) and not np.allclose(plain.u_off, t.u_off)
    assert np.allclose(t.prior_total, t.prior_off + t.prior_def)
